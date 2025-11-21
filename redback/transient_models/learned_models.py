"""This file holds the functions to call the LearnedSurrogateModel models.

LearnedSurrogateModel are models from the redback_surrogates package that
have been trained to emulate more complex transient models and saved in ONNX format.

The workflow for these models is to load the model from an ONNX file using
the LearnedSurrogateModel.from_onnx_file() method, then use the
make_learned_model_callable() function to create a callable function that can be used
to evaluate the model given time and parameters.
"""
import astropy.units as uu
import numpy as np
import re

from astropy.cosmology import Planck18 as cosmo  # noqa
from collections import namedtuple
from scipy.interpolate import RegularGridInterpolator

import redback.sed as sed
from redback.utils import calc_kcorrected_properties, lambda_to_nu


def make_learned_model_callable(model):
    """
    This function takes in a LearnedSurrogateModel instance and returns a callable function
    that can be used to evaluate the model given time and parameters.

    The function's signature will match the expected format for redback with time as the
    first argument, followed by each of the model parameters and then any additional keyword
    arguments.

    :param model: LearnedSurrogateModel instance
    :return: Callable function to evaluate the model
    """
    # Make sure all the model's parameter names are safe to use as function arguments.
    identifier_re = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    for name in model.param_names:
        if not isinstance(name, str) or not identifier_re.match(name):
            raise ValueError(
                f"Parameter name '{name}' is invalid. Parameter names can "
                "only contain alphanumeric characters and underscores."
            )

    # Build the complete function string. We have already checked that the parameter names are safe.
    param_str = ", ".join(model.param_names)
    param_dict_str = (
        "{" + ", ".join([f"'{name}': {name}" for name in model.param_names]) + "}"
    )
    function_code = (
        f"def _dynamic_predict_grid(time, {param_str}, *, model=model, **kwargs):\n"
        f"    param_dict = {param_dict_str}\n"
        f"    return _eval_learned_surrogate(model, time, param_dict, **kwargs)\n"
    )

    # Execute the function definition and bind it to this instance. Note that we can only do exec
    # safely here only because we checked the parameter names earlier to ensure they are safe.
    local_namespace = {"model": model}
    exec(function_code, globals(), local_namespace)

    # Use partial to bind the model to the function so the user doesn't have to pass it in.
    return local_namespace["_dynamic_predict_grid"]


def _eval_learned_surrogate(model, time, params, **kwargs):
    """
    This is a common evaluation function for LearnedSurrogateModel models that can be called
    from each model's wrapper function.

    :param model: LearnedSurrogateModel instance
    :param time: Time in days in observer frame
    :param params: Dictionary of model parameters. Must include 'redshift' key.
    :param kwargs: Additional parameters for the model, such as:
    :param frequency: Required if output_format is 'flux_density'.
        frequency to calculate - Must be same length as time array or a single number).
    :param bands: Required if output_format is 'magnitude' or 'flux'.
    :param output_format: 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    :param lambda_array: Optional argument to set your desired wavelength array (in Angstroms) to evaluate the SED on.
    :param cosmology: Cosmology to use for luminosity distance calculation. Defaults to Planck18. Must be a astropy.cosmology object.
    :return: set by output format - 'flux_density', 'magnitude', 'spectra', 'flux', 'sncosmo_source'
    """
    cosmology = kwargs.get('cosmology', cosmo)
    redshift = params.get('redshift', 0.0)
    dl = cosmology.luminosity_distance(redshift).cgs

    # Get the rest-frame spectrum from the model.
    # These will always be f_lambda in erg/s/Angstrom
    luminosity_density = model.predict_spectra_grid(**params)
    lambda_rest = model.wavelengths  # Angstrom in rest frame
    time_rest = model.times  # days in rest frame

    # Apply cosmological dimming: L_nu / (4*pi*d_L^2) gives flux that
    # still needs (1+z) correction. Units are now erg/s/Hz/cm^2
    flux_density = luminosity_density / (4 * np.pi * dl ** 2)

    # Handle different output formats
    if kwargs.get('output_format') == 'flux_density':
        # Use redback's K-correction utilities
        frequency = kwargs['frequency']
        frequency, time = calc_kcorrected_properties(frequency=frequency, time=time, redshift=redshift)

        # Convert rest-frame wavelengths to rest-frame frequencies for interpolation
        nu_rest = lambda_to_nu(lambda_rest)

        # Convert flux density to mJy
        conversion_factor = (1.0 * uu.erg / uu.s / uu.Hz / (uu.cm ** 2)).to(uu.mJy).value
        fmjy = conversion_factor * flux_density

        # Create interpolator on rest-frame grid
        flux_interpolator = RegularGridInterpolator(
            (time_rest, nu_rest),
            fmjy,
            bounds_error=False,
            fill_value=0.0
        )

        # Prepare points for interpolation
        if isinstance(frequency, (int, float)):
            frequency = np.ones_like(time) * frequency

        # Create points for evaluation
        points = np.column_stack((time, frequency))

        # Return interpolated flux density with (1+z) correction for observer frame
        return flux_interpolator(points) / (1 + redshift)

    else:
        # Create denser grid for output (in rest frame)
        time_rest_dense = np.geomspace(np.min(time_rest), np.max(time_rest), 200)
        lambda_rest_dense = np.geomspace(np.min(lambda_rest), np.max(lambda_rest), 200)

        # Create interpolator for the flux density in rest frame
        flux_interpolator = RegularGridInterpolator(
            (time_rest, lambda_rest),
            flux_density.value,
            bounds_error=False,
            fill_value=0.0
        )

        # Create meshgrid for new grid points
        tt_mesh, ll_mesh = np.meshgrid(time_rest_dense, lambda_rest_dense, indexing='ij')
        points_to_evaluate = np.column_stack((tt_mesh.ravel(), ll_mesh.ravel()))

        # Interpolate flux density onto denser grid
        interpolated_values = flux_interpolator(points_to_evaluate)
        interpolated_flux = interpolated_values.reshape(tt_mesh.shape) * flux_density.unit

        # Convert to observer frame: times and wavelengths
        time_observer_frame = time_rest_dense * (1 + redshift)
        lambda_observer_frame = lambda_rest_dense * (1 + redshift)

        # Apply (1+z) correction to flux and convert units
        # After dividing by (1+z), flux is in observer frame at observer-frame wavelengths
        flux_observer = interpolated_flux / (1 + redshift)
        spectra = flux_observer.to(
            uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom,
            equivalencies=uu.spectral_density(wav=lambda_observer_frame * uu.Angstrom)
        )

        # Create output structure
        if kwargs.get('output_format') == 'spectra':
            return namedtuple('output', ['time', 'lambdas', 'spectra'])(
                time=time_observer_frame,
                lambdas=lambda_observer_frame,
                spectra=spectra
            )
        else:
            # Get correct output format using redback utility
            return sed.get_correct_output_format_from_spectra(
                time=time,  # Original observer frame time for evaluation
                time_eval=time_observer_frame,
                spectra=spectra,
                lambda_array=lambda_observer_frame,
                time_spline_degree=1,
                **kwargs
            )
