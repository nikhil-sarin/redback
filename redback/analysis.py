import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
from typing import Union, Optional
from pathlib import Path

import redback.model_library
from redback.utils import logger, find_nearest, bands_to_frequency
from redback.result import RedbackResult
from redback.constants import day_to_s
import matplotlib


def _setup_plotting_result(model, model_kwargs, parameters, transient):
    """
    Helper function to setup the plotting result

    :param model: model string or model function
    :param model_kwargs: keyword arguments passed to the model
    :param parameters: parameters to plot
    :param transient: transient object
    :return: a tuple of model, parameters, and result
    """
    if isinstance(parameters, dict):
        parameters = pd.DataFrame.from_dict(parameters)
    parameters["log_likelihood"] = np.arange(len(parameters))
    if isinstance(model, str):
        model = redback.model_library.all_models_dict[model]
    meta_data = dict(model=model.__name__, transient_type=transient.__class__.__name__.lower())
    transient_kwargs = {k.lstrip("_"): v for k, v in transient.__dict__.items()}
    meta_data.update(transient_kwargs)
    meta_data['model_kwargs'] = model_kwargs or dict()
    res = RedbackResult(label="None", outdir="None",
                        search_parameter_keys=None,
                        fixed_parameter_keys=None,
                        constraint_parameter_keys=None, priors=None,
                        sampler_kwargs=dict(), injection_parameters=None,
                        meta_data=meta_data, posterior=parameters, samples=None,
                        nested_samples=None, log_evidence=0,
                        log_evidence_err=0, information_gain=0,
                        log_noise_evidence=0, log_bayes_factor=0,
                        log_likelihood_evaluations=0,
                        log_prior_evaluations=0, sampling_time=0, nburn=0,
                        num_likelihood_evaluations=0, walkers=0,
                        max_autocorrelation_time=0, use_ratio=False,
                        version=None)
    return model, parameters, res


def plot_lightcurve(transient, parameters, model, model_kwargs=None,
                    show=True, save=False, **kwargs: None):
    """
    Plot a lightcurve for a given model and parameters

    :param transient: transient object
    :param parameters: parameters to plot
    :param model: model string or model function
    :param model_kwargs: keyword arguments passed to the model
    :return: plot_lightcurve
    """
    model, parameters, res = _setup_plotting_result(model, model_kwargs, parameters, transient)
    return res.plot_lightcurve(model=model, random_models=len(parameters), plot_max_likelihood=False,
                               save=save, show=show, **kwargs)


def plot_multiband_lightcurve(transient, parameters, model, model_kwargs=None,
                              show=True, save=False, **kwargs: None):
    """
    Plot a multiband lightcurve for a given model and parameters

    :param transient: transient object
    :param parameters: parameters to plot
    :param model: model string or model function
    :param model_kwargs: keyword arguments passed to the model
    :return: plot_multiband_lightcurve
    """
    model, parameters, res = _setup_plotting_result(model, model_kwargs, parameters, transient)
    return res.plot_multiband_lightcurve(model=model, random_models=len(parameters), plot_max_likelihood=False,
                                         save=save, show=show, **kwargs)


def plot_evolution_parameters(result, random_models=100):
    """
    Plot evolution parameters for a given evolving_magnetar result

    :param result: redback result
    :param random_models: number of random models to plot
    :return: fig and axes
    """
    logger.warning("This type of plot is only valid for evolving magnetar models")
    tmin = np.log10(np.min(result.metadata['time']))
    tmax = np.log10(np.max(result.metadata['time']))
    time = np.logspace(tmin, tmax, 100)
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(5, 10))
    for j in range(random_models):
        s = dict(result.posterior.iloc[np.random.randint(len(result.posterior))])
        s["output"] = "namedtuple"
        model = redback.model_library.all_models_dict["evolving_magnetar_only"]
        output = model(time, **s)
        nn = output.nn
        mu = output.mu
        alpha = output.alpha
        ax[0].plot(time, nn, "--", lw=1, color='red', zorder=-1)
        ax[1].plot(time, np.rad2deg(alpha), "--", lw=1, color='red', zorder=-1)
        ax[2].plot(time, mu, "--", lw=1, color='red', zorder=-1)
        ax[0].set_ylabel('braking index')
        ax[1].set_ylabel('inclination angle')
        ax[2].set_ylabel('magnetic moment')
    for x in range(3):
        ax[x].set_yscale('log')
        ax[x].set_xscale('log')
    fig.supxlabel(r"Time since burst [s]")
    return fig, ax

def plot_spectrum(model, parameters, time_to_plot, axes=None, **kwargs):
    """
    Plot a spectrum for a given model and parameters

    :param model: Model string for a redback model
    :param parameters: dictionary of parameters/alongside model specific keyword arguments.
        Must be one set of parameters. If you want to plot a posterior prediction of the spectrum,
        call this function in a loop.
    :param time_to_plot: Times to plot (in days) the spectrum at.
        The spectrum plotted will be at the nearest neighbour to this value
    :param axes: None or matplotlib axes object if you want to plot on an existing set of axes
    :param kwargs: Additional keyword arguments used by this function.
    :param colors_list: List of colors to use for each time to plot. Set randomly unless specified.
    :return: matplotlib axes
    """
    function = redback.model_library.all_models_dict[model]
    model_kwargs = {}
    model_kwargs.update(parameters)
    model_kwargs['output_format'] = 'spectra'
    model_kwargs['bands'] = 'lsstg'
    output = function(time_to_plot, **model_kwargs)
    lambdas = output.lambdas
    time_of_output = output.time/day_to_s

    #extract spectrum at the times of interest.
    spec = {}
    for tt in time_to_plot:
        _, idx = find_nearest(time_of_output, tt)
        spec[tt] = output.spectra[idx]

    if 'colors_list' in kwargs.keys():
        colors_list = kwargs.pop('colors_list')
    else:
        colors_list = matplotlib.cm.tab20(range(len(time_to_plot)))

    ax = axes or plt.gca()
    for i, tt in enumerate(time_to_plot):
        ax.semilogx(lambdas, spec[tt], color=colors_list[i], label=f"{tt:.1f} days")
    ax.set_xlabel(r'Wavelength ($\mathrm{\AA}$)')
    ax.set_ylabel(r'Flux ($10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\mathrm{\AA}$)')
    ax.legend(loc='upper left')
    return ax

def plot_gp_lightcurves(transient, gp_output, axes=None, band_colors=None, band_scaling=None):
    """
    Plot the Gaussian Process lightcurves

    :param transient: A transient object
    :param gp_output: The output of the fit_gp function
    :param axes: axes, ideally you should be passing the axes from the plot_data methods
    :param band_colors: a dictionary of band colors; again ideally you should be passing the band_colors from the plot_data methods
    :return: axes object with the GP lightcurves plotted
    """
    ax = axes or plt.gca()

    if transient.use_phase_model:
        ref_date = transient.x[0]
    else:
        ref_date = 0

    t_new = np.linspace(transient.x.min() - 10, transient.x.max() + 20, 100)

    if transient.data_mode in ['flux_density', 'flux', 'magnitude']:
        if band_colors is None:
            band_colors = dict(zip(transient.unique_bands, plt.cm.tab20(range(len(transient.unique_bands)))))
        else:
            band_colors = band_colors
        if gp_output.use_frequency:
            for band in transient.unique_bands:
                if band_scaling:
                    scaling = band_scaling[band]
                else:
                    scaling = 0
                f_new = np.ones_like(t_new) * bands_to_frequency([band])
                X_new = np.column_stack((f_new, t_new))
                gp = gp_output.gp
                y_pred, y_cov = gp.predict(gp_output.scaled_y, X_new, return_cov=True)
                y_std = np.sqrt(np.diag(y_cov))
                y_lower = y_pred - 0.5 * y_std
                y_upper = y_pred + 0.5 * y_std
                ax.plot(t_new - ref_date, (y_pred * gp_output.y_scaler) + scaling, color=band_colors[band])
                ax.fill_between(t_new - ref_date, (y_lower * gp_output.y_scaler) + scaling,
                                (y_upper * gp_output.y_scaler) + scaling, alpha=0.5,
                                color=band_colors[band])
        else:
            for band in transient.unique_bands:
                if band_scaling:
                    scaling = band_scaling[band]
                else:
                    scaling = 0
                gp = gp_output.gp[band]
                y_pred, y_cov = gp.predict(gp_output.scaled_y[band], t_new, return_cov=True)
                y_std = np.sqrt(np.diag(y_cov))
                y_lower = y_pred - 0.5 * y_std
                y_upper = y_pred + 0.5 * y_std
                ax.plot(t_new - ref_date, (y_pred * gp_output.y_scaler) + scaling, color=band_colors[band])
                ax.fill_between(t_new - ref_date, (y_lower * gp_output.y_scaler) + scaling,
                                (y_upper * gp_output.y_scaler) + scaling, alpha=0.5,
                                color=band_colors[band])
    else:
        y_pred, y_cov = gp_output.gp.predict(gp_output.scaled_y, t_new, return_cov=True)
        y_std = np.sqrt(np.diag(y_cov))
        y_lower = y_pred - 0.5 * y_std
        y_upper = y_pred + 0.5 * y_std

        ax.plot(t_new, y_pred * gp_output.y_scaler, color='red')
        ax.fill_between(t_new, y_lower * gp_output.y_scaler, y_upper * gp_output.y_scaler, alpha=0.5, color='red')
    return ax

def fit_temperature_and_radius_gp(data, kernelT, kernelR, plot=False, **kwargs):
    """
    Fit a Gaussian Process to the temperature and radius data

    :param data: DataFrame containing the temperature and radius data output of the transient.estimate_bb_params method.
    :param kernelT: george kernel for the temperature
    :param kernelR: george kernel for the radius
    :param plot: Whether to make a two-panel plot of the temperature and radius GP evolution and the data
    :param kwargs: Additional keyword arguments
    :param inflate_errors: If True, inflate the errors by 20%, default is False
    :return: Temperature and radius GP objects and plot fig and axes if requested
    """
    import george
    from scipy.optimize import minimize

    temperature = data['temperature']
    radius = data['radius']
    t_data = data['epoch_times']
    T_err = data['temp_err']
    R_err = data['radius_err']
    inflate_errors = kwargs.get('inflate_errors', True)
    if inflate_errors:
        error = kwargs.get('error', 1.5)
    else:
        error = 1
    gp_T_err_raw = T_err * error
    gp_R_err = R_err * error

    fit_in_log = kwargs.get("fit_in_log", False)
    if fit_in_log:
        # In log space, use: log10(T); propagate errors via: δ(log10T)=δT/(T*ln(10))
        temperature_fit = np.log10(temperature)
        gp_T_err = gp_T_err_raw / (temperature * np.log(10))
    else:
        temperature_fit = temperature
        gp_T_err = gp_T_err_raw

    gp_T = george.GP(kernelT)
    gp_T.compute(t_data, gp_T_err + 1e-8)

    def neg_ln_like_T(p):
        gp_T.set_parameter_vector(p)
        return -gp_T.log_likelihood(temperature_fit)

    def grad_neg_ln_like_T(p):
        gp_T.set_parameter_vector(p)
        return -gp_T.grad_log_likelihood(temperature_fit)

    p0_T = gp_T.get_parameter_vector()
    result_T = minimize(neg_ln_like_T, p0_T, jac=grad_neg_ln_like_T)
    gp_T.set_parameter_vector(result_T.x)

    logger.info("Finished GP fit for temperature")
    logger.info(f"GP final parameters: {gp_T.get_parameter_dict()}")

    gp_R = george.GP(kernelR)
    gp_R.compute(t_data, gp_R_err + 1e-8)

    def neg_ln_like_R(p):
        gp_R.set_parameter_vector(p)
        return -gp_R.log_likelihood(radius)

    def grad_neg_ln_like_R(p):
        gp_R.set_parameter_vector(p)
        return -gp_R.grad_log_likelihood(radius)

    p0_R = gp_R.get_parameter_vector()
    result_R = minimize(neg_ln_like_R, p0_R, jac=grad_neg_ln_like_R)
    gp_R.set_parameter_vector(result_R.x)

    logger.info("Finished GP fit for radius")
    logger.info(f"GP final parameters: {gp_R.get_parameter_dict()}")

    if plot:
        sigma_to_plot = kwargs.get('sigma_to_plot', 1)
        label = r"${}\sigma$ GP uncertainty".format(str(int(sigma_to_plot)))
        t_pred = np.linspace(t_data.min(), t_data.max(), 100)
        # Temperature prediction
        T_pred, T_pred_var = gp_T.predict(temperature_fit, t_pred, return_var=True)
        T_pred_std = np.sqrt(T_pred_var)

        # If fitting in log space, convert the prediction back to linear units.
        if fit_in_log:
            T_pred_lin = 10**T_pred
            # Propagate the uncertainty approximately: dT ≈ 10^x * ln(10) * sigma_x.
            T_pred_std_lin = 10**T_pred * np.log(10) * T_pred_std
        else:
            T_pred_lin = T_pred
            T_pred_std_lin = T_pred_std

        # Radius prediction
        R_pred, R_pred_var = gp_R.predict(radius, t_pred, return_var=True)
        R_pred_std = np.sqrt(R_pred_var)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
        ax1.errorbar(t_data, temperature, yerr=T_err, fmt='o', label='Data', color='blue')
        ax1.plot(t_pred, T_pred_lin, label='GP Prediction', color='red')
        ax1.fill_between(t_pred, T_pred_lin - sigma_to_plot*T_pred_std_lin, T_pred_lin + sigma_to_plot*T_pred_std_lin,
                         alpha=0.2, color='red', label=label)
        ax2.errorbar(t_data, radius, yerr=R_err, fmt='o', label='Data', color='blue')
        ax2.plot(t_pred, R_pred, label='GP Prediction', color='red')
        ax2.fill_between(t_pred, R_pred - sigma_to_plot*R_pred_std, R_pred + sigma_to_plot*R_pred_std, alpha=0.2, color='red',
                         label=label)

        ax1.set_xlabel("Time", fontsize=15)
        ax1.set_ylabel("Temperature [K]", fontsize=15)
        ax1.set_title("Temperature Evolution", fontsize=15)
        ax2.set_xlabel("Time", fontsize=15)
        ax2.set_ylabel("Radius [cm]", fontsize=15)
        ax2.set_title("Radius Evolution from GP", fontsize=15)

        ax1.set_yscale('log')
        ax2.set_yscale('log')

        ax1.legend()
        ax2.legend()
        plt.subplots_adjust(wspace=0.3)
        return gp_T, gp_R, fig, (ax1, ax2)
    else:
        return gp_T, gp_R

def generate_new_transient_data_from_gp(gp_out, t_new, transient, **kwargs):
    """
    Generates new transient data based on Gaussian Process (GP) predictions for the given time array
    and transient object. Depending on the data mode of the transient object
    (e.g., 'flux_density', 'flux', 'magnitude', or 'luminosity'), this function updates the data
    accordingly, adjusting errors and scaling by frequency if necessary.

    :param gp_out: The GP output object containing the Gaussian Process model, scaled data,
                   and other related attributes.
    :type gp_out: object
    :param t_new: Array of new time values for which GP predictions are to be generated.
    :type t_new: array-like
    :param transient: The transient object containing the original observation data and related
                      properties such as data mode and unique frequencies or bands.
    :type transient: object
    :param kwargs: Additional parameters to modify behavior, such as:

                   - **inflate_y_err** (bool): Flag to indicate whether to inflate GP errors.
                   - **error** (float): Multiplier for adjusting GP error inflation.

    :return: A new transient object with data updated using GP predictions.
    :rtype: object
    """
    data_mode = transient.data_mode
    logger.info(f"Data mode: {data_mode}")
    logger.info("Creating new {} data".format(data_mode))

    if data_mode not in ['flux_density', 'flux', 'magnitude', 'luminosity']:
        raise ValueError("Data mode {} not understood".format(data_mode))

    if kwargs.get('inflate_y_err', True):
        error = kwargs.get('error', 10)
    else:
        logger.info("Using GP predicted errors, this is likely being too conservative")
        error = 1.

    if gp_out.use_frequency:
        logger.info("GP is a 2D kernel with effective frequency")
        freqs = transient.unique_frequencies
        T, F = np.meshgrid(t_new, freqs)
        try:
            bands = redback.utils.frequency_to_bandname(F.flatten())
        except Exception:
            bands = F.flatten().astype(str)
        X_new = np.column_stack((F.flatten(), T.flatten()))
        y_pred, y_var = gp_out.gp.predict(gp_out.scaled_y, X_new, return_var=True)
        y_std = np.sqrt(y_var)
        y_err = y_std * error
        y_pred = y_pred * gp_out.y_scaler
        tts = T.flatten()
        freqs = F.flatten()
    else:
        logger.info("GP is a 1D kernel")
        if data_mode == 'flux_density':
            logger.warning("Bandnames/frequency attributes for the transient object may be weird, "
                           "Please check for yourself")
            tts = []
            ys = []
            yerrs = []
            bbs = []
            for key in gp_out.gp.keys():
                gp = gp_out.gp[key]
                y_pred, y_cov = gp.predict(gp_out.scaled_y[key], t_new, return_cov=True)
                y_std = np.sqrt(np.diag(y_cov))
                y_err = y_std * error
                y_pred = y_pred * gp_out.y_scaler
                _bands = np.repeat(key, len(t_new))
                bbs.append(key)
                tts.append(t_new)
                ys.append(y_pred)
                yerrs.append(y_err)
            temp_frame = pd.DataFrame({'time': tts, 'ys': ys, 'yerr': yerrs, 'band': bbs})
            temp_frame.sort_values('time', inplace=True)
            y_pred = temp_frame['ys']
            y_err = temp_frame['yerr']
            bands = temp_frame['band']
            freqs = temp_frame['band']
            tts = temp_frame['time']
        elif data_mode in ['flux', 'magnitude']:
            tts = []
            ys = []
            yerrs = []
            bbs = []
            for band in transient.unique_bands:
                gp = gp_out.gp[band]
                y_pred, y_cov = gp.predict(gp_out.scaled_y[band], t_new, return_cov=True)
                y_std = np.sqrt(np.diag(y_cov))
                y_err = y_std * error
                y_pred = y_pred * gp_out.y_scaler
                _bands = np.repeat(band, len(t_new))
                bbs.append(_bands)
                tts.append(t_new)
                ys.append(y_pred)
                yerrs.append(y_err)
            temp_frame = pd.DataFrame({'time':tts, 'ys':ys, 'yerr':yerrs, 'band':bbs})
            temp_frame.sort_values('time', inplace=True)
            y_pred = temp_frame['ys']
            y_err = temp_frame['yerr']
            bands = temp_frame['band']
            tts = temp_frame['time']
        elif data_mode == 'luminosity':
            y_pred, y_cov = gp_out.gp.predict(gp_out.scaled_y, t_new, return_cov=True)
            y_std = np.sqrt(np.diag(y_cov))
            y_err = y_std * error
            y_pred = y_pred * gp_out.y_scaler
            tts = t_new

    logger.info(f"Data mode: {data_mode}")
    logger.info("Creating new transient object with GP data")
    if data_mode == 'flux_density':
        new_transient = redback.transient.OpticalTransient(name=transient.name + '_gp',
                                                           flux_density=y_pred, flux_density_err=y_err,
                                                           time=tts, bands=bands, frequency=freqs,
                                                           data_mode=data_mode, redshift=transient.redshift)
    elif data_mode == 'flux':
        new_transient = redback.transient.OpticalTransient(name=transient.name + '_gp',
                                                           flux=y_pred, flux_err=y_err,
                                                           time=tts, bands=bands,
                                                           data_mode=data_mode, redshift=transient.redshift)
    elif data_mode == 'magnitude':
        new_transient = redback.transient.OpticalTransient(name=transient.name + '_gp',
                                                           magnitude=y_pred, magnitude_err=y_err,
                                                           time=tts, bands=bands,
                                                           data_mode=data_mode, redshift=transient.redshift)
    elif data_mode == 'luminosity':
        new_transient = redback.transient.OpticalTransient(name=transient.name + '_gp',
                                                           Lum50=y_pred, Lum50_err=y_err,
                                                           time_rest_frame=tts, data_mode=data_mode)
    return new_transient


class SpectralVelocityFitter:
    """
    Measure expansion velocities from spectral line profiles

    Used for:
    - Photospheric velocity evolution
    - High-velocity features (HVF)
    - Velocity gradients (dv/dt)

    Parameters
    ----------
    wavelength : array
        Wavelength array in Angstroms
    flux : array
        Flux density array
    flux_err : array, optional
        Flux density uncertainties

    Examples
    --------
    >>> fitter = SpectralVelocityFitter(wavelength, flux)
    >>> v_Si, v_err = fitter.measure_line_velocity(6355)
    >>> print(f"Si II velocity: {v_Si:.0f} +/- {v_err:.0f} km/s")
    """

    def __init__(self, wavelength, flux, flux_err=None):
        """
        Initialize SpectralVelocityFitter

        Parameters
        ----------
        wavelength : array
            Wavelength array in Angstroms
        flux : array
            Flux density array
        flux_err : array, optional
            Flux density uncertainties
        """
        self.wavelength = np.asarray(wavelength)
        self.flux = np.asarray(flux)
        if flux_err is not None:
            self.flux_err = np.asarray(flux_err)
        else:
            self.flux_err = None

    @classmethod
    def from_spectrum_object(cls, spectrum):
        """
        Create fitter from a redback Spectrum object

        Parameters
        ----------
        spectrum : object
            Object with .angstroms and .flux_density attributes

        Returns
        -------
        fitter : SpectralVelocityFitter
            Initialized fitter object
        """
        wavelength = spectrum.angstroms
        flux = spectrum.flux_density
        flux_err = getattr(spectrum, 'flux_density_err', None)
        return cls(wavelength, flux, flux_err)

    def measure_line_velocity(self, line_rest_wavelength, method='min', **kwargs):
        """
        Measure velocity from single absorption line

        Parameters
        ----------
        line_rest_wavelength : float
            Rest wavelength in Angstroms (e.g., 6355 for Si II)
        method : str
            'min' - use minimum flux (standard)
            'centroid' - use flux-weighted centroid
            'fit' - fit P-Cygni profile
            'gaussian' - fit Gaussian to absorption trough
        kwargs : dict
            Additional parameters:
            - v_window : float
                Velocity window for search (km/s, default 5000)
            - continuum_percentile : float
                Percentile for continuum estimation (default 90)

        Returns
        -------
        velocity : float
            Measured velocity in km/s (negative = blueshift)
        velocity_err : float
            Uncertainty in km/s

        Examples
        --------
        >>> fitter = SpectralVelocityFitter(wavelength, flux)
        >>> v_Si, verr = fitter.measure_line_velocity(6355, method='min')
        >>> print(f"Si II velocity: {v_Si:.0f} +/- {verr:.0f} km/s")
        """
        c_kms = 299792.458
        v_window = kwargs.get('v_window', 5000)  # km/s

        # Extract region around line
        lambda_window = line_rest_wavelength * v_window / c_kms

        mask = ((self.wavelength > line_rest_wavelength - lambda_window) &
                (self.wavelength < line_rest_wavelength + lambda_window))

        if np.sum(mask) < 5:
            logger.warning(f"Insufficient data points around {line_rest_wavelength} A")
            return np.nan, np.nan

        wave_line = self.wavelength[mask]
        flux_line = self.flux[mask]

        if method == 'min':
            # Find minimum flux (absorption trough)
            imin = np.argmin(flux_line)
            lambda_min = wave_line[imin]

            # Convert to velocity
            velocity = c_kms * (lambda_min - line_rest_wavelength) / line_rest_wavelength

            # Error estimate from nearby points
            n_err = min(3, len(wave_line) // 4)
            if n_err > 0:
                # Estimate error from wavelength resolution
                dlambda = np.median(np.diff(wave_line))
                velocity_err = c_kms * dlambda / line_rest_wavelength
            else:
                velocity_err = 100  # default estimate

        elif method == 'centroid':
            # Flux-weighted centroid (inverse for absorption)
            # Use inverse flux for absorption features
            continuum_pct = kwargs.get('continuum_percentile', 90)
            continuum = np.percentile(flux_line, continuum_pct)

            # Absorption depth
            absorption = continuum - flux_line
            absorption[absorption < 0] = 0

            if np.sum(absorption) > 0:
                lambda_centroid = np.sum(wave_line * absorption) / np.sum(absorption)
                velocity = c_kms * (lambda_centroid - line_rest_wavelength) / line_rest_wavelength

                # Error from scatter in absorption
                variance = np.sum(absorption * (wave_line - lambda_centroid)**2) / np.sum(absorption)
                lambda_err = np.sqrt(variance / np.sum(absorption > 0))
                velocity_err = c_kms * lambda_err / line_rest_wavelength
            else:
                velocity = 0.0
                velocity_err = 500.0

        elif method == 'gaussian':
            # Fit Gaussian to absorption trough
            from scipy.optimize import curve_fit

            # Estimate continuum
            continuum_pct = kwargs.get('continuum_percentile', 90)
            continuum = np.percentile(flux_line, continuum_pct)

            def gaussian_absorption(wave, center, depth, sigma):
                return continuum * (1 - depth * np.exp(-0.5 * ((wave - center) / sigma)**2))

            # Initial guess
            imin = np.argmin(flux_line)
            center_guess = wave_line[imin]
            depth_guess = (continuum - flux_line[imin]) / continuum
            sigma_guess = 10.0  # Angstroms

            try:
                popt, pcov = curve_fit(
                    gaussian_absorption, wave_line, flux_line,
                    p0=[center_guess, depth_guess, sigma_guess],
                    bounds=([wave_line.min(), 0.01, 1.0],
                            [wave_line.max(), 1.0, 200.0])
                )

                lambda_center = popt[0]
                velocity = c_kms * (lambda_center - line_rest_wavelength) / line_rest_wavelength
                velocity_err = c_kms * np.sqrt(pcov[0, 0]) / line_rest_wavelength

            except Exception as e:
                logger.warning(f"Gaussian fit failed: {e}")
                return self.measure_line_velocity(line_rest_wavelength, method='min')

        elif method == 'fit':
            # Fit P-Cygni profile
            from scipy.optimize import curve_fit
            from redback.transient_models.spectral_models import p_cygni_profile

            # Continuum level
            continuum_pct = kwargs.get('continuum_percentile', 90)
            continuum = np.percentile(flux_line, continuum_pct)

            def pcygni_model(wave, tau, v_phot):
                return p_cygni_profile(
                    wave, line_rest_wavelength, tau, v_phot, continuum, **kwargs
                )

            # Initial guess from minimum
            imin = np.argmin(flux_line)
            lambda_min = wave_line[imin]
            v_guess = np.abs(c_kms * (lambda_min - line_rest_wavelength) / line_rest_wavelength)

            try:
                popt, pcov = curve_fit(
                    pcygni_model, wave_line, flux_line,
                    p0=[3.0, v_guess],
                    bounds=([0.1, 1000], [100, 50000])
                )

                velocity = -popt[1]  # blueshifted, so negative
                velocity_err = np.sqrt(pcov[1, 1])

            except Exception as e:
                logger.warning(f"P-Cygni fit failed: {e}")
                return self.measure_line_velocity(line_rest_wavelength, method='min')

        else:
            raise ValueError(f"Unknown method: {method}")

        return velocity, velocity_err

    def measure_multiple_lines(self, line_dict, method='min', **kwargs):
        """
        Measure velocities for multiple lines

        Parameters
        ----------
        line_dict : dict
            {'Si II 6355': 6355, 'Fe II 5169': 5169, ...}
        method : str
            Method for velocity measurement (default 'min')
        kwargs : dict
            Additional parameters passed to measure_line_velocity

        Returns
        -------
        velocities : dict
            {'Si II 6355': (v, v_err), ...}

        Examples
        --------
        >>> lines = {
        ...     'Si II 6355': 6355,
        ...     'Ca II H&K': 3934,
        ...     'Fe II 5169': 5169
        ... }
        >>> velocities = fitter.measure_multiple_lines(lines)
        >>> for ion, (v, verr) in velocities.items():
        ...     print(f"{ion}: {v:.0f} +/- {verr:.0f} km/s")
        """
        velocities = {}
        for ion_name, rest_wave in line_dict.items():
            try:
                v, verr = self.measure_line_velocity(rest_wave, method=method, **kwargs)
                velocities[ion_name] = (v, verr)
            except Exception as e:
                logger.warning(f"Could not measure {ion_name}: {e}")
                velocities[ion_name] = (np.nan, np.nan)

        return velocities

    @staticmethod
    def photospheric_velocity_evolution(wavelength_list, flux_list, times,
                                         line_wavelength=6355, method='min', **kwargs):
        """
        Track photospheric velocity evolution over time

        Parameters
        ----------
        wavelength_list : list of arrays
            Wavelength arrays for each spectrum
        flux_list : list of arrays
            Flux arrays for each spectrum
        times : array
            Observation times (days)
        line_wavelength : float
            Which line to use (default Si II 6355)
        method : str
            Velocity measurement method

        Returns
        -------
        times : array
            Observation times
        velocities : array
            Measured velocities (km/s)
        errors : array
            Velocity uncertainties (km/s)

        Examples
        --------
        >>> times, vels, errs = SpectralVelocityFitter.photospheric_velocity_evolution(
        ...     wavelength_list, flux_list, obs_times, line_wavelength=6355
        ... )
        >>> plt.errorbar(times, -vels/1000, yerr=errs/1000)
        >>> plt.xlabel('Days since explosion')
        >>> plt.ylabel('Photospheric velocity (1000 km/s)')
        """
        velocities = []
        errors = []

        for wave, flux in zip(wavelength_list, flux_list):
            fitter = SpectralVelocityFitter(wave, flux)
            v, verr = fitter.measure_line_velocity(line_wavelength, method=method, **kwargs)

            velocities.append(v)
            errors.append(verr)

        return np.array(times), np.array(velocities), np.array(errors)

    def identify_high_velocity_features(self, line_rest_wavelength, v_phot_expected,
                                          threshold_factor=1.3):
        """
        Identify high-velocity features (HVF) in the spectrum

        HVFs are absorption features at higher velocities than the photosphere,
        often associated with circumstellar material or density enhancements.

        Parameters
        ----------
        line_rest_wavelength : float
            Rest wavelength of the line in Angstroms
        v_phot_expected : float
            Expected photospheric velocity in km/s
        threshold_factor : float
            Factor above v_phot to classify as HVF (default 1.3)

        Returns
        -------
        has_hvf : bool
            Whether HVF is detected
        v_hvf : float or None
            Velocity of HVF if detected (km/s)
        v_hvf_err : float or None
            Uncertainty in HVF velocity

        Examples
        --------
        >>> has_hvf, v_hvf, v_err = fitter.identify_high_velocity_features(
        ...     6355, v_phot_expected=11000
        ... )
        >>> if has_hvf:
        ...     print(f"HVF detected at {-v_hvf:.0f} km/s")
        """
        c_kms = 299792.458

        # Search for features at higher velocities
        v_search_max = v_phot_expected * 2.0  # Search up to 2x photospheric velocity
        v_search_min = v_phot_expected * threshold_factor

        lambda_min = line_rest_wavelength * (1 - v_search_max / c_kms)
        lambda_max = line_rest_wavelength * (1 - v_search_min / c_kms)

        mask = (self.wavelength > lambda_min) & (self.wavelength < lambda_max)

        if np.sum(mask) < 3:
            return False, None, None

        wave_hvf = self.wavelength[mask]
        flux_hvf = self.flux[mask]

        # Look for local minimum
        if len(flux_hvf) > 2:
            imin = np.argmin(flux_hvf)
            lambda_min = wave_hvf[imin]

            # Check if it's a significant absorption
            continuum = np.percentile(self.flux, 90)
            absorption_depth = (continuum - flux_hvf[imin]) / continuum

            if absorption_depth > 0.05:  # At least 5% absorption
                v_hvf = c_kms * (lambda_min - line_rest_wavelength) / line_rest_wavelength
                dlambda = np.median(np.diff(wave_hvf)) if len(wave_hvf) > 1 else 5.0
                v_hvf_err = c_kms * dlambda / line_rest_wavelength

                return True, v_hvf, v_hvf_err

        return False, None, None

    def measure_velocity_gradient(self, wavelength_list, flux_list, times,
                                   line_wavelength=6355, **kwargs):
        """
        Measure velocity gradient dv/dt from time series of spectra

        Parameters
        ----------
        wavelength_list : list of arrays
            Wavelength arrays for each spectrum
        flux_list : list of arrays
            Flux arrays for each spectrum
        times : array
            Observation times (days)
        line_wavelength : float
            Which line to use
        kwargs : dict
            Additional parameters passed to measure_line_velocity
            (e.g., v_window, method)

        Returns
        -------
        gradient : float
            Velocity gradient in km/s/day
        gradient_err : float
            Uncertainty in gradient

        Notes
        -----
        The velocity gradient is typically negative (decelerating) for
        normal SNe Ia (around -50 to -100 km/s/day), but can be different
        for peculiar objects.
        """
        times, velocities, errors = self.photospheric_velocity_evolution(
            wavelength_list, flux_list, times, line_wavelength, **kwargs
        )

        # Remove NaN values
        valid = ~np.isnan(velocities)
        if np.sum(valid) < 2:
            return np.nan, np.nan

        times_valid = times[valid]
        vel_valid = velocities[valid]
        err_valid = errors[valid]

        # Linear fit
        from numpy.polynomial import polynomial as P

        # Weighted fit if errors available
        if np.all(err_valid > 0) and np.all(~np.isnan(err_valid)):
            weights = 1 / err_valid**2
            coeffs = np.polyfit(times_valid, vel_valid, deg=1, w=weights)
        else:
            coeffs = np.polyfit(times_valid, vel_valid, deg=1)

        gradient = coeffs[0]  # km/s/day

        # Error estimate
        residuals = vel_valid - np.polyval(coeffs, times_valid)
        if len(times_valid) > 2:
            gradient_err = np.std(residuals) / np.sqrt(np.sum((times_valid - np.mean(times_valid))**2))
        else:
            gradient_err = np.nan

        return gradient, gradient_err



class ClassificationResult(dict):
    """
    Result of spectral or photometric transient classification.

    Behaves as a plain dict (for backward compatibility) while also providing
    convenience attributes and methods. The dict contains the keys:
    ``best_type``, ``best_phase``, ``best_redshift``, ``correlation``
    (= rlap), ``type_probabilities``, ``top_matches``, plus ``confidence``,
    ``best_template_name``, ``best_template_source``, ``method``, ``warnings``.

    Quality interpretation for rlap (spectral matching):
    - rlap > 8: high confidence match
    - rlap 5–8: medium confidence
    - rlap < 5: low confidence, treat with caution
    """

    def __init__(self, best_type: str, best_phase: float, best_redshift: float,
                 rlap: float, confidence: str, type_probabilities: dict,
                 top_matches: list, best_template_name: str,
                 best_template_source: Optional[str] = None,
                 method: str = 'rlap',
                 warnings: Optional[list] = None):
        super().__init__(
            best_type=best_type,
            best_phase=best_phase,
            best_redshift=best_redshift,
            correlation=rlap,    # alias for backward compat
            rlap=rlap,
            confidence=confidence,
            type_probabilities=type_probabilities,
            top_matches=top_matches,
            best_template_name=best_template_name,
            best_template_source=best_template_source,
            method=method,
            warnings=warnings if warnings is not None else [],
        )

    # Convenience attribute access via dict keys
    @property
    def best_type(self) -> str:
        return self['best_type']

    @property
    def best_phase(self) -> float:
        return self['best_phase']

    @property
    def best_redshift(self) -> float:
        return self['best_redshift']

    @property
    def rlap(self) -> float:
        return self['rlap']

    @property
    def confidence(self) -> str:
        return self['confidence']

    @property
    def type_probabilities(self) -> dict:
        return self['type_probabilities']

    @property
    def top_matches(self) -> list:
        return self['top_matches']

    @property
    def best_template_name(self) -> str:
        return self['best_template_name']

    @property
    def best_template_source(self) -> Optional[str]:
        return self['best_template_source']

    @property
    def method(self) -> str:
        return self['method']

    @property
    def warnings(self) -> list:
        return self['warnings']

    def summary(self) -> str:
        """Return a human-readable classification summary."""
        lines = [
            f"Classification: Type {self.best_type}",
            f"Phase:          {self.best_phase:+.1f} days from maximum",
            f"Redshift:       {self.best_redshift:.4f}",
            f"Quality (rlap): {self.rlap:.1f}  [{self.confidence} confidence]",
            "",
            "Type probabilities:",
        ]
        for t, p in sorted(self.type_probabilities.items(), key=lambda x: -x[1]):
            lines.append(f"  {t:10s}: {p * 100:5.1f}%")
        if self.warnings:
            lines.append("\nWarnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Return a plain dict copy (for explicit serialisation)."""
        return dict(self)


class SpectralTemplateMatcher(object):
    """
    Match spectra to template library (similar to SNID).

    Supports Pearson correlation, chi-squared, and the recommended SNID-style
    rlap cross-correlation metric. Templates can be loaded from SNID .lnw files,
    CSV/DAT libraries, downloaded from OSC or GitHub, or generated from
    sncosmo spectral models.

    The default template library uses sncosmo models (SALT2 for Type Ia,
    v19-1998bw for Ic-BL, nugent templates for Ib/c / IIP / IIn, and
    s11-2004hx for generic Type II), providing realistic spectral shapes at
    multiple phases for each type.

    The default matching method is 'rlap', which cross-correlates in log-wavelength
    space (= velocity space) and is shift-invariant — a small redshift error does
    not degrade the match quality. A good match has rlap > 5; an excellent match
    has rlap > 10.
    """

    # sncosmo source names and corresponding SN types for the default template library.
    # Each entry: (sncosmo_source_name, sn_type_label, phases_to_sample)
    _SNCOSMO_TEMPLATE_SOURCES = [
        ('salt2',        'Ia',    [-10, -5, 0, 5, 10, 15, 20]),
        ('v19-1998bw',   'Ic-BL', [-5, 0, 5, 10, 15, 20]),
        ('nugent-sn1bc', 'Ib/c',  [0, 5, 10, 15, 20, 30]),
        ('nugent-sn2p',  'IIP',   [0, 10, 20, 30, 50, 80]),
        ('nugent-sn2n',  'IIn',   [0, 10, 30, 60]),
        ('s11-2004hx',   'II',    [0, 10, 20, 30, 50]),
    ]

    def __init__(self, template_library_path: Optional[Union[str, Path]] = None,
                 templates: Optional[list] = None) -> None:
        """
        Initialize the SpectralTemplateMatcher with a template library.

        :param template_library_path: Path to a directory containing template files
            (CSV/DAT format). If None and templates is None, uses built-in sncosmo
            templates (SALT2, 1998bw, Nugent templates, etc.).
        :param templates: List of template dictionaries to use directly. Each template
            should have keys: 'wavelength', 'flux', 'type', 'phase', and optionally 'name'.
        """
        if templates is not None:
            self.templates = templates
        elif template_library_path is not None:
            self.templates = self._load_templates(template_library_path)
        else:
            self.templates = self._load_default_templates()

        logger.info(f"Loaded {len(self.templates)} templates into the matcher")

    def _load_default_templates(self) -> list:
        """
        Load built-in templates from sncosmo spectral models.

        Uses SALT2 (Type Ia), v19-1998bw (Ic-BL / 1998bw-like), nugent-sn1bc
        (Ib/c), nugent-sn2p (IIP), nugent-sn2n (IIn), and s11-2004hx (II).

        :return: List of template dictionaries
        """
        logger.info("Loading default spectral templates from sncosmo")
        return self.generate_sncosmo_templates()

    @classmethod
    def generate_sncosmo_templates(cls,
                                   sources: Optional[list] = None,
                                   wavelength_range: tuple = (3500, 9000),
                                   n_wavelength: int = 1000) -> list:
        """
        Generate spectral templates from sncosmo source models.

        By default uses SALT2, v19-1998bw (SN 1998bw / Ic-BL), nugent-sn1bc,
        nugent-sn2p, nugent-sn2n, and s11-2004hx. Each source is sampled at a
        set of representative phases.

        :param sources: Optional list of ``(source_name, type_label, phases)``
            tuples to override the default set (``_SNCOSMO_TEMPLATE_SOURCES``).
        :param wavelength_range: (min, max) wavelength in Angstroms
        :param n_wavelength: Number of wavelength points
        :return: List of template dicts with keys 'wavelength', 'flux', 'type',
            'phase', 'name', 'source'
        """
        import sncosmo

        wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], n_wavelength)
        source_list = sources if sources is not None else cls._SNCOSMO_TEMPLATE_SOURCES

        templates = []
        for source_name, sn_type, phases in source_list:
            try:
                src = sncosmo.get_source(source_name)
            except Exception as e:
                logger.warning(f"Could not load sncosmo source '{source_name}': {e}")
                continue

            for phase in phases:
                # Skip phases outside the model's valid range
                if phase < src.minphase() or phase > src.maxphase():
                    continue
                try:
                    # Clip wavelength range to model validity
                    wave_lo = max(wavelength_range[0], src.minwave())
                    wave_hi = min(wavelength_range[1], src.maxwave())
                    wave = np.linspace(wave_lo, wave_hi, n_wavelength)
                    flux = src.flux(phase, wave)
                    flux = np.asarray(flux, dtype=float)
                    if not np.all(np.isfinite(flux)):
                        logger.warning(
                            f"Template {source_name} phase {phase} contains non-finite "
                            "values; skipping."
                        )
                        continue
                    max_flux = np.max(np.abs(flux))
                    if max_flux <= 0:
                        continue
                    flux = flux / max_flux
                    safe_type = sn_type.replace('/', '-')
                    templates.append({
                        'wavelength': wave,
                        'flux': flux,
                        'type': sn_type,
                        'phase': float(phase),
                        'name': f'{source_name}_{safe_type}_phase{phase:+d}',
                        'source': source_name,
                    })
                except Exception as e:
                    logger.warning(
                        f"Failed to generate template {source_name} phase {phase}: {e}"
                    )

        logger.info(f"Generated {len(templates)} sncosmo templates")
        return templates

    @classmethod
    def generate_synthetic_templates(cls, sn_types: Optional[list] = None,
                                     wavelength_range: tuple = (3500, 9000),
                                     n_wavelength: int = 1000,
                                     r_photosphere: float = 1e15) -> list:
        """
        Generate spectral templates using sncosmo models (legacy alias for
        :meth:`generate_sncosmo_templates`).

        This method is retained for backward compatibility. New code should
        call :meth:`generate_sncosmo_templates` directly.

        :param sn_types: Ignored (kept for API compatibility).
        :param wavelength_range: (min, max) wavelength in Angstroms
        :param n_wavelength: Number of wavelength points
        :param r_photosphere: Ignored (kept for API compatibility).
        :return: List of template dicts
        """
        return cls.generate_sncosmo_templates(
            wavelength_range=wavelength_range,
            n_wavelength=n_wavelength,
        )

    @staticmethod
    def _blackbody_flux(wavelengths: np.ndarray, temperature: float) -> np.ndarray:
        """
        Compute a simple Planck blackbody flux (arbitrary units).

        :param wavelengths: Wavelength array in Angstroms
        :param temperature: Temperature in Kelvin
        :return: Flux array proportional to B_lambda(T), same shape as wavelengths
        """
        # h*c/k_B in Angstrom*K
        hc_over_k = 1.43878e8  # Angstrom * K
        wave = np.asarray(wavelengths, dtype=float)
        exponent = hc_over_k / (wave * temperature)
        # Clip exponent to avoid overflow
        exponent = np.clip(exponent, 0, 700)
        flux = wave ** (-5) / (np.exp(exponent) - 1.0)
        return flux

    @staticmethod
    def _flatten_spectrum(flux: np.ndarray, smooth_sigma: int = 30) -> np.ndarray:
        """
        Remove continuum by dividing by a Gaussian-smoothed version, returning
        zero-mean fractional deviations. This isolates spectral features from the
        broad continuum shape — equivalent to the 'flattening' step in SNID.

        :param flux: Flux array (on a uniform log-wavelength grid)
        :param smooth_sigma: Gaussian smoothing width in pixels. 30 pixels at
            dlog_lambda=0.001 corresponds to ~7000 km/s — removes broad continuum
            but preserves spectral lines.
        :return: Flattened flux array (zero mean, dimensionless)
        """
        from scipy.ndimage import gaussian_filter1d
        continuum = gaussian_filter1d(flux.astype(float), sigma=smooth_sigma)
        continuum = np.where(np.abs(continuum) > 0, continuum, 1e-30)
        return flux / continuum - 1.0

    @staticmethod
    def _compute_rlap(obs_wave: np.ndarray, obs_flux: np.ndarray,
                      tmpl_wave: np.ndarray, tmpl_flux: np.ndarray,
                      dlog_lambda: float = 0.001,
                      smooth_sigma: int = 30,
                      tmpl_pre_flattened: bool = False) -> tuple:
        """
        Compute the SNID-style rlap quality metric and best-fit redshift via
        cross-correlation in log-wavelength (= velocity) space.

        The algorithm:
        1. Build a common log-lambda grid over the wavelength overlap.
        2. Interpolate both spectra onto the grid.
        3. Flatten both (remove continuum) using ``_flatten_spectrum``.
        4. Apply a Hanning window to suppress edge ringing.
        5. Cross-correlate via FFT; normalise by geometric mean of auto-correlations.
        6. The CCF peak position gives the best-fit redshift offset.
        7. rlap = |peak_CCF| * n_overlap_pixels  (Blondin & Tonry 2007 definition).

        :param obs_wave: Observed wavelength array (Angstroms, ascending)
        :param obs_flux: Observed flux array
        :param tmpl_wave: Template wavelength array (Angstroms, rest frame)
        :param tmpl_flux: Template flux array
        :param dlog_lambda: Log-wavelength grid spacing (default 0.001 ≈ 230 km/s/pixel)
        :param smooth_sigma: Smoothing sigma for continuum removal (pixels)
        :return: (rlap, z_best, ccf_array, z_lag_array). Returns (0.0, 0.0, None, None)
            on failure.
        """
        from scipy.interpolate import interp1d

        log_obs = np.log10(obs_wave)
        log_tmpl = np.log10(tmpl_wave)
        log_min = max(log_obs.min(), log_tmpl.min())
        log_max = min(log_obs.max(), log_tmpl.max())

        if log_max <= log_min:
            return 0.0, 0.0, None, None

        n_grid = int((log_max - log_min) / dlog_lambda)
        if n_grid < 20:
            return 0.0, 0.0, None, None

        log_grid = np.linspace(log_min, log_max, n_grid)
        wave_grid = 10.0 ** log_grid

        f_obs = interp1d(obs_wave, obs_flux, bounds_error=False, fill_value=0.0)
        f_tmpl = interp1d(tmpl_wave, tmpl_flux, bounds_error=False, fill_value=0.0)
        obs_resampled = f_obs(wave_grid)
        tmpl_resampled = f_tmpl(wave_grid)

        obs_flat = SpectralTemplateMatcher._flatten_spectrum(obs_resampled, smooth_sigma)
        tmpl_flat = (tmpl_resampled if tmpl_pre_flattened
                     else SpectralTemplateMatcher._flatten_spectrum(tmpl_resampled, smooth_sigma))

        # Hanning taper to suppress edge ringing
        taper = np.hanning(n_grid)
        obs_flat = obs_flat * taper
        tmpl_flat = tmpl_flat * taper

        # Cross-correlation via FFT
        fft_obs = np.fft.rfft(obs_flat)
        fft_tmpl = np.fft.rfft(tmpl_flat)
        ccf = np.fft.irfft(fft_obs * np.conj(fft_tmpl), n=n_grid)

        # Normalise by geometric mean of auto-correlations
        ac_obs = float(np.sum(obs_flat ** 2))
        ac_tmpl = float(np.sum(tmpl_flat ** 2))
        norm = np.sqrt(ac_obs * ac_tmpl) if (ac_obs > 0 and ac_tmpl > 0) else 1.0
        ccf_norm = ccf / norm

        # Map lags to redshift offsets: lag in pixels → delta(log_lambda) → delta_z
        lags = np.fft.fftfreq(n_grid, d=1.0 / n_grid)   # pixel lags, unshifted
        log_lags = lags * dlog_lambda
        z_offsets = 10.0 ** log_lags - 1.0

        # Shift to centre (lag=0 in the middle)
        ccf_shifted = np.fft.fftshift(ccf_norm)
        z_shifted = np.fft.fftshift(z_offsets)

        i_peak = int(np.argmax(np.abs(ccf_shifted)))
        r_peak = float(ccf_shifted[i_peak])
        z_best = float(z_shifted[i_peak])

        # rlap follows Blondin & Tonry (2007):
        #   rlap = r * lap
        # where r is the normalised CCF peak (in [-1, 1]) and lap is the
        # fractional overlap of the log-wavelength range, scaled to [0, 10].
        # A good match has rlap > 5; an excellent match has rlap > 8.
        log_union_min = min(log_obs.min(), log_tmpl.min())
        log_union_max = max(log_obs.max(), log_tmpl.max())
        n_union = max(int((log_union_max - log_union_min) / dlog_lambda), 1)
        lap = (n_grid / n_union) * 10.0   # fractional overlap scaled to [0, 10]

        rlap = abs(r_peak) * lap
        return rlap, z_best, ccf_shifted, z_shifted

    def _load_templates(self, library_path: Union[str, Path]) -> list:
        """
        Load templates from a directory of files.

        Expected file format: CSV or whitespace-separated with columns:
        wavelength (Angstroms), flux

        File naming convention: {type}_{phase}.csv or {type}_{phase}.dat
        e.g., Ia_+5.csv, II_10.dat

        :param library_path: Path to template library directory
        :return: List of template dictionaries
        """
        library_path = Path(library_path)
        if not library_path.exists():
            raise FileNotFoundError(f"Template library path not found: {library_path}")

        templates = []

        # Look for CSV and DAT files
        template_files = list(library_path.glob("*.csv")) + list(library_path.glob("*.dat"))

        if len(template_files) == 0:
            raise ValueError(f"No template files found in {library_path}")

        for file_path in template_files:
            try:
                # Try to parse filename for type and phase
                stem = file_path.stem
                parts = stem.split('_')
                if len(parts) >= 2:
                    sn_type = parts[0]
                    phase_str = parts[1].replace('+', '')
                    try:
                        phase = float(phase_str)
                    except ValueError:
                        phase = 0.0
                else:
                    sn_type = stem
                    phase = 0.0

                # Load data - first check for metadata in comments
                with open(file_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith('#'):
                            # Check for metadata in comments
                            if 'Type:' in line or 'type:' in line:
                                try:
                                    sn_type = line.split(':')[1].strip()
                                except IndexError:
                                    pass
                            if 'Phase:' in line or 'phase:' in line:
                                try:
                                    phase = float(line.split(':')[1].strip())
                                except (IndexError, ValueError):
                                    pass

                if file_path.suffix == '.csv':
                    # Count comment lines and header row to skip
                    skip_count = 0
                    with open(file_path, 'r') as f:
                        for line in f:
                            if line.strip().startswith('#') or line.strip().startswith('wavelength'):
                                skip_count += 1
                            else:
                                break
                    data = np.loadtxt(file_path, delimiter=',', skiprows=skip_count)
                else:
                    data = np.loadtxt(file_path, comments='#')

                wavelength = data[:, 0]
                flux = data[:, 1]

                # Normalize flux
                flux = flux / np.max(flux)

                templates.append({
                    'wavelength': wavelength,
                    'flux': flux,
                    'type': sn_type,
                    'phase': phase,
                    'name': stem
                })

                logger.info(f"Loaded template: {stem}")

            except Exception as e:
                logger.warning(f"Failed to load template {file_path}: {e}")
                continue

        return templates

    def add_template(self, wavelength: np.ndarray, flux: np.ndarray,
                     sn_type: str, phase: float, name: Optional[str] = None) -> None:
        """
        Add a single template to the library.

        :param wavelength: Wavelength array in Angstroms
        :param flux: Flux array (will be normalized)
        :param sn_type: Type classification (e.g., 'Ia', 'II', 'Ib/c')
        :param phase: Phase in days from maximum light
        :param name: Optional name for the template
        """
        if name is None:
            name = f"{sn_type}_phase_{phase}"

        flux_normalized = flux / np.max(flux)

        self.templates.append({
            'wavelength': wavelength,
            'flux': flux_normalized,
            'type': sn_type,
            'phase': phase,
            'name': name
        })

        logger.info(f"Added template: {name}")

    def match_spectrum(self, spectrum, redshift_range: tuple = (0, 0.5),
                       n_redshift_points: int = 50,
                       method: str = 'rlap',
                       return_all_matches: bool = False,
                       rlap_threshold: float = 0.0) -> Union[dict, list, None]:
        """
        Find the best-matching template for an observed spectrum.

        :param spectrum: Spectrum object with angstroms and flux_density attributes
        :param redshift_range: (z_min, z_max) to restrict the redshift search.
            For method='rlap', the best-fit redshift comes directly from the CCF
            peak and is clipped to this range. For 'correlation' and 'chi2', a
            grid of n_redshift_points values is searched.
        :param n_redshift_points: Grid points for 'correlation'/'chi2' methods.
            Ignored for method='rlap'.
        :param method: Matching method:
            - 'rlap' (default): SNID-style cross-correlation in log-wavelength
              space. Shift-invariant. Returns rlap quality metric (>5 good, >8
              excellent). Recommended for all real use.
            - 'correlation': Pearson correlation on a redshift grid. Legacy method.
            - 'chi2': Chi-squared on a redshift grid (requires flux errors for
              meaningful values).
            - 'both': Pearson + chi2 with combined normalised score.
        :param return_all_matches: If True, return the full sorted list of match dicts.
        :param rlap_threshold: Minimum rlap to include in results (default 0 = no filter).
        :return: Best match dict (or sorted list if return_all_matches=True), or None.
            Match dict keys: 'type', 'phase', 'redshift', 'rlap', 'correlation',
            'template_name', and (if applicable) 'chi2', 'reduced_chi2', 'scale_factor'.
        """
        from scipy.interpolate import interp1d
        from scipy.stats import pearsonr

        if len(self.templates) == 0:
            raise ValueError("No templates loaded. Add templates before matching.")

        all_matches = []
        obs_wavelength = spectrum.angstroms
        obs_flux = spectrum.flux_density
        norm_factor = np.max(np.abs(obs_flux))
        obs_flux_norm = obs_flux / norm_factor

        has_errors = hasattr(spectrum, 'flux_density_err') and spectrum.flux_density_err is not None
        if has_errors:
            obs_flux_err_norm = spectrum.flux_density_err / norm_factor

        # --- rlap path: one CCF per template, no redshift grid needed ---
        if method == 'rlap':
            # If a non-trivial redshift range is specified, first check that at
            # least one template overlaps the spectrum at the requested redshift.
            # If none do, return None immediately (e.g. z_min=2 pushes all
            # templates far out of the observed wavelength range).
            z_lo, z_hi = redshift_range
            if z_lo > 0:
                has_overlap_at_z = False
                for template in self.templates:
                    tmpl_wave_shifted = template['wavelength'] * (1.0 + z_lo)
                    if (np.min(tmpl_wave_shifted) < np.max(obs_wavelength) and
                            np.max(tmpl_wave_shifted) > np.min(obs_wavelength)):
                        has_overlap_at_z = True
                        break
                if not has_overlap_at_z:
                    logger.warning(
                        f"No template overlaps the observed wavelength range at "
                        f"redshift_range={redshift_range}. Returning None."
                    )
                    return None

            from scipy.stats import pearsonr
            from scipy.interpolate import interp1d as _interp1d

            for template in self.templates:
                rlap, z_best, ccf, z_arr = self._compute_rlap(
                    obs_wavelength, obs_flux_norm,
                    template['wavelength'], template['flux'],
                    tmpl_pre_flattened=template.get('pre_flattened', False),
                )
                # ccf is None when there is no wavelength overlap — skip entirely
                if ccf is None:
                    continue
                # Clip z_best to the requested range
                z_best = float(np.clip(z_best, z_lo, z_hi))
                if rlap < rlap_threshold:
                    continue

                # Compute Pearson correlation at the best-fit redshift for the
                # 'correlation' key (used by downstream tests / backward compat)
                pearson_corr = rlap  # default fallback
                try:
                    tmpl_wave_z = template['wavelength'] * (1.0 + z_best)
                    f_tmpl = _interp1d(tmpl_wave_z, template['flux'],
                                       bounds_error=False, fill_value=np.nan)
                    tmpl_interp = f_tmpl(obs_wavelength)
                    valid = (~np.isnan(tmpl_interp) & np.isfinite(obs_flux_norm))
                    if np.sum(valid) >= 5:
                        corr_val, _ = pearsonr(obs_flux_norm[valid], tmpl_interp[valid])
                        pearson_corr = float(corr_val)
                except Exception:
                    pass

                all_matches.append({
                    'type': template['type'],
                    'phase': template['phase'],
                    'redshift': z_best,
                    'rlap': rlap,
                    'correlation': pearson_corr,
                    'template_name': template.get('name', f"{template['type']}_p{template['phase']}"),
                    'template_source': template.get('source', 'unknown'),
                    'n_valid_points': len(ccf),
                })
            # Sort by correlation (Pearson) for consistent ordering with other methods
            all_matches.sort(key=lambda x: -x['correlation'])

        # --- Pearson / chi2 / both: grid search over redshift ---
        else:
            for template in self.templates:
                for z in np.linspace(redshift_range[0], redshift_range[1], n_redshift_points):
                    template_wave_obs = template['wavelength'] * (1.0 + z)

                    min_overlap = max(np.min(template_wave_obs), np.min(obs_wavelength))
                    max_overlap = min(np.max(template_wave_obs), np.max(obs_wavelength))
                    if max_overlap <= min_overlap:
                        continue

                    interp_func = interp1d(template_wave_obs, template['flux'],
                                           bounds_error=False, fill_value=np.nan)
                    template_flux_interp = interp_func(obs_wavelength)

                    valid_mask = (~np.isnan(template_flux_interp) &
                                  ~np.isnan(obs_flux_norm) &
                                  (template_flux_interp != 0))
                    if np.sum(valid_mask) < 10:
                        continue

                    obs_valid = obs_flux_norm[valid_mask]
                    template_valid = template_flux_interp[valid_mask]

                    match_result = {
                        'type': template['type'],
                        'phase': template['phase'],
                        'redshift': z,
                        'rlap': 0.0,
                        'template_name': template.get('name', f"{template['type']}_p{template['phase']}"),
                        'n_valid_points': int(np.sum(valid_mask)),
                    }

                    if method in ('correlation', 'both'):
                        try:
                            corr, p_value = pearsonr(obs_valid, template_valid)
                            match_result['correlation'] = float(corr)
                            match_result['p_value'] = float(p_value)
                        except Exception:
                            match_result['correlation'] = -1.0
                            match_result['p_value'] = 1.0

                    if method in ('chi2', 'both'):
                        if has_errors:
                            err_valid = obs_flux_err_norm[valid_mask]
                            scale = (np.sum(obs_valid * template_valid / err_valid ** 2) /
                                     np.sum(template_valid ** 2 / err_valid ** 2))
                            residuals = obs_valid - scale * template_valid
                            chi2 = float(np.sum((residuals / err_valid) ** 2))
                            match_result['chi2'] = chi2
                            match_result['reduced_chi2'] = chi2 / max(len(obs_valid) - 1, 1)
                            match_result['scale_factor'] = float(scale)
                        else:
                            scale = (np.sum(obs_valid * template_valid) /
                                     np.sum(template_valid ** 2))
                            residuals = obs_valid - scale * template_valid
                            chi2 = float(np.sum(residuals ** 2) / max(np.var(obs_valid), 1e-30))
                            match_result['chi2'] = chi2
                            match_result['scale_factor'] = float(scale)

                    all_matches.append(match_result)

            if len(all_matches) == 0:
                logger.warning("No valid matches found. Check wavelength coverage and templates.")
                return None

            if method == 'chi2':
                all_matches.sort(key=lambda x: x.get('chi2', np.inf))
            elif method == 'correlation':
                all_matches.sort(key=lambda x: -x.get('correlation', -1.0))
            else:  # both — combined normalised score
                corr_vals = np.array([m.get('correlation', 0.0) for m in all_matches])
                chi2_vals = np.array([m.get('reduced_chi2', np.inf) for m in all_matches])
                c_range = corr_vals.ptp()
                q_range = chi2_vals.ptp()
                corr_norm = (corr_vals - corr_vals.min()) / (c_range if c_range > 0 else 1.0)
                chi2_norm = (chi2_vals - chi2_vals.min()) / (q_range if q_range > 0 else 1.0)
                for i, m in enumerate(all_matches):
                    m['combined_score'] = float(corr_norm[i] - 0.3 * chi2_norm[i])
                all_matches.sort(key=lambda x: -x.get('combined_score', 0.0))

        if len(all_matches) == 0:
            logger.warning("No valid matches found. Check wavelength coverage and templates.")
            return None

        return all_matches if return_all_matches else all_matches[0]

    def classify_spectrum(self, spectrum, redshift_range: tuple = (0, 0.5),
                          n_redshift_points: int = 50,
                          top_n: int = 10,
                          rlap_threshold: float = 3.0) -> ClassificationResult:
        """
        Classify a spectrum and return a :class:`ClassificationResult`.

        Type probabilities are computed via softmax over the mean rlap per type
        across the top_n matches. Using the mean (rather than sum) ensures that
        types with more templates in the library do not dominate.

        :param spectrum: Spectrum object with angstroms and flux_density attributes
        :param redshift_range: (z_min, z_max) redshift search range
        :param n_redshift_points: Grid points (only used if method falls back to
            Pearson when rlap fails)
        :param top_n: Number of top matches to use for probability estimation
        :param rlap_threshold: Matches below this rlap are excluded from the
            probability estimate. Set to 0 to include all matches.
        :return: :class:`ClassificationResult` instance
        """
        all_matches = self.match_spectrum(
            spectrum,
            redshift_range=redshift_range,
            method='rlap',
            return_all_matches=True,
        )

        warnings_list = []

        if all_matches is None or len(all_matches) == 0:
            return ClassificationResult(
                best_type=None,
                best_phase=0.0,
                best_redshift=0.0,
                rlap=0.0,
                confidence='low',
                type_probabilities={},
                top_matches=[],
                best_template_name='',
                method='rlap',
                warnings=['No valid matches found'],
            )

        # For classification, rank by rlap (regardless of match_spectrum sort order)
        all_matches_by_rlap = sorted(all_matches, key=lambda x: -x.get('rlap', 0.0))

        # Apply rlap threshold; fall back to all matches if nothing passes
        good_matches = [m for m in all_matches_by_rlap if m.get('rlap', 0) >= rlap_threshold]
        if len(good_matches) == 0:
            good_matches = all_matches_by_rlap
            warnings_list.append(
                f"No matches exceeded rlap_threshold={rlap_threshold:.1f}. "
                f"Best rlap was {all_matches_by_rlap[0].get('rlap', 0):.2f}. "
                "Classification may be unreliable."
            )

        top_matches = good_matches[:min(top_n, len(good_matches))]

        # Aggregate mean rlap per type
        from collections import defaultdict
        type_rlap = defaultdict(list)
        for m in top_matches:
            type_rlap[m['type']].append(m.get('rlap', 0.0))
        type_mean_rlap = {t: float(np.mean(v)) for t, v in type_rlap.items()}

        # Softmax normalisation (numerically stable)
        max_rlap = max(type_mean_rlap.values())
        exp_scores = {t: np.exp(s - max_rlap) for t, s in type_mean_rlap.items()}
        total = sum(exp_scores.values())
        type_probabilities = {t: float(v / total) for t, v in exp_scores.items()}

        best_match = top_matches[0]
        best_rlap = float(best_match.get('rlap', 0.0))
        confidence = 'high' if best_rlap > 8 else 'medium' if best_rlap > 5 else 'low'

        if best_rlap < 3.0:
            warnings_list.append(
                f"Best rlap={best_rlap:.2f} is below 3.0. "
                "Consider loading a larger or more appropriate template library."
            )

        return ClassificationResult(
            best_type=best_match['type'],
            best_phase=float(best_match['phase']),
            best_redshift=float(best_match['redshift']),
            rlap=best_rlap,
            confidence=confidence,
            type_probabilities=type_probabilities,
            top_matches=top_matches,
            best_template_name=best_match.get('template_name', ''),
            best_template_source=best_match.get('template_source'),
            method='rlap',
            warnings=warnings_list,
        )

    def plot_match(self, spectrum, match_result: dict, axes=None, **kwargs) -> matplotlib.axes.Axes:
        """
        Plot observed spectrum against best-matching template.

        :param spectrum: Observed Spectrum object
        :param match_result: Result dictionary from match_spectrum
        :param axes: Optional matplotlib axes to plot on
        :param kwargs: Additional plotting arguments
        :return: Matplotlib axes object
        """
        from scipy.interpolate import interp1d

        ax = axes or plt.gca()

        # Get observed spectrum
        obs_wavelength = spectrum.angstroms
        obs_flux_raw = spectrum.flux_density

        # Find matching template
        template = None
        for t in self.templates:
            if t.get('name') == match_result.get('template_name'):
                template = t
                break

        if template is None:
            # Try to find by type and phase
            for t in self.templates:
                if t['type'] == match_result['type'] and t['phase'] == match_result['phase']:
                    template = t
                    break

        if template is None:
            raise ValueError("Could not find matching template")

        pre_flattened = template.get('pre_flattened', False)

        # If template is already continuum-subtracted, the observed spectrum must
        # be put on the same scale.  We flatten it (remove continuum via Gaussian
        # division) and then normalise by the RMS so the amplitudes are comparable
        # regardless of whether the input is raw or already continuum-subtracted.
        if pre_flattened:
            norm = np.max(np.abs(obs_flux_raw))
            obs_flux_norm = obs_flux_raw / norm if norm > 0 else obs_flux_raw
            obs_flux_flat = SpectralTemplateMatcher._flatten_spectrum(obs_flux_norm)
            rms = np.sqrt(np.nanmean(obs_flux_flat ** 2))
            obs_flux_plot = obs_flux_flat / rms if rms > 0 else obs_flux_flat
            ylabel = 'Continuum-subtracted Flux'
        else:
            obs_flux_plot = obs_flux_raw / np.max(obs_flux_raw)
            ylabel = 'Normalized Flux'

        # Redshift template
        z = match_result['redshift']
        template_wave_obs = template['wavelength'] * (1 + z)

        # Interpolate template to observed wavelengths for comparison
        interp_func = interp1d(template_wave_obs, template['flux'],
                               bounds_error=False, fill_value=np.nan)
        template_flux_interp = interp_func(obs_wavelength)

        # Scale template to match observed flux
        if 'scale_factor' in match_result:
            scale = match_result['scale_factor']
        else:
            denom = np.nansum(template_flux_interp ** 2)
            scale = (np.nansum(obs_flux_plot * template_flux_interp) / denom
                     if denom > 0 else 1.0)

        # Plot
        ax.plot(obs_wavelength, obs_flux_plot, 'k-', label='Observed', alpha=0.8, lw=1.5)
        ax.plot(obs_wavelength, scale * template_flux_interp, 'r--',
                label=f"Template: {match_result['type']} (phase={match_result['phase']:.0f}d, z={z:.3f})",
                alpha=0.8, lw=1.5)

        ax.set_xlabel(r'Wavelength ($\mathrm{\AA}$)')
        ax.set_ylabel(ylabel)
        rlap_val = match_result.get('rlap', match_result.get('correlation', 0))
        title = f"Best Match: {match_result['type']}, rlap={rlap_val:.2f}"
        ax.set_title(title)
        ax.legend(loc='best')

        return ax

    @staticmethod
    def get_available_template_sources() -> dict:
        """
        Get information about available template sources.

        :return: Dictionary with source names and their descriptions/URLs
        """
        return {
            'snid_templates_2.0': {
                'description': 'Official SNID templates v2.0 from Blondin & Tonry',
                'url': 'https://people.lam.fr/blondin.stephane/software/snid/',
                'download_url': 'https://people.lam.fr/blondin.stephane/software/snid/templates-2.0.tgz',
                'citation': 'Blondin & Tonry 2007, ApJ, 666, 1024'
            },
            'super_snid': {
                'description': 'Super-SNID expanded templates (841 spectra, 161 objects)',
                'url': 'https://github.com/dkjmagill/QUB-SNID-Templates',
                'zenodo_doi': '10.5281/zenodo.15167198',
                'citation': 'Magill et al. 2025'
            },
            'sesn_templates': {
                'description': 'Stripped-envelope SN templates from METAL collaboration',
                'url': 'https://github.com/metal-sn/SESNtemple',
                'citation': 'Williamson et al. 2023, Yesmin et al. 2024'
            },
        }


    @staticmethod
    def parse_snid_template_file(file_path: Union[str, Path]):
        """
        Parse a SNID template file (.lnw format) or two-column ASCII template.

        For proper SNID .lnw files (Blondin & Tonry 2007), returns a list of
        dicts, one per epoch. For simple two-column ASCII files, returns a
        single dict.

        The SNID .lnw format (Blondin & Tonry 2007, Appendix B):

        **Line 1 — object header (8 tokens):**
          ``nwave  nspec  type_code  type_string  redshift  age_of_max  dm15  name``

        **Next nwave tokens — log10(wavelength) array** (may span multiple lines).
          wavelength = 10^token (Angstroms). The grid is log-spaced.

        **Then nspec epoch blocks, each with:**
          - One header line: ``phase_days  <ignored>``
          - nwave flux tokens (may span multiple lines)

        For two-column ASCII files, metadata can be provided via header comments::

            # Type: IIn
            # Phase: -3.5

        Comments are parsed case-insensitively. If a comment key is present but
        has no valid value, the filename is used as fallback.

        :param file_path: Path to a SNID .lnw template file or two-column ASCII
        :return: For .lnw files: list of template dicts, one per epoch.
            For ASCII files: a single template dict.
            Each dict has keys: 'wavelength', 'flux', 'type', 'phase', 'name', 'source'
        """
        file_path = Path(file_path)
        name = file_path.stem

        # --- Parse comment metadata (case-insensitive) from header lines ---
        comment_type = None
        comment_phase = None
        with open(file_path, 'r') as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                if not stripped.startswith('#'):
                    break  # stop at first data line
                lower = stripped.lower()
                if 'type:' in lower:
                    try:
                        comment_type = stripped.split(':', 1)[1].strip()
                    except IndexError:
                        comment_type = ''
                if 'phase:' in lower:
                    try:
                        comment_phase = float(stripped.split(':', 1)[1].strip())
                    except (IndexError, ValueError):
                        comment_phase = None  # mark as failed, fall back to filename

        # --- Tokenise the file (skip comment lines starting with '#') ---
        tokens = []
        with open(file_path, 'r') as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped or stripped.startswith('#'):
                    continue
                tokens.extend(stripped.split())

        # --- Attempt .lnw header parse (requires ≥8 tokens and valid header) ---
        if len(tokens) >= 8:
            try:
                nwave = int(tokens[0])
                nspec = int(tokens[1])
                # tokens[2] is integer type code — skip
                sn_type = tokens[3]
                source_redshift = float(tokens[4])
                age_of_max = float(tokens[5])
                # tokens[6] is dm15 — skip
                obj_name = tokens[7]

                if nwave < 10 or nspec < 1:
                    raise ValueError("Implausible nwave/nspec")

                required = 8 + nwave + nspec * (1 + nwave)
                if len(tokens) < required:
                    raise ValueError(
                        f"Not enough tokens: need {required}, have {len(tokens)}"
                    )

                # Read log-wavelength array
                pos = 8
                log_wave = np.array([float(tokens[pos + i]) for i in range(nwave)])
                wavelengths = 10.0 ** log_wave   # Angstroms
                pos += nwave

                templates = []
                for epoch_idx in range(nspec):
                    epoch_phase = float(tokens[pos]) - age_of_max
                    pos += 2   # phase + one ignored token
                    flux = np.array([float(tokens[pos + i]) for i in range(nwave)])
                    pos += nwave

                    max_flux = np.max(np.abs(flux))
                    if max_flux > 0:
                        flux = flux / max_flux

                    templates.append({
                        'wavelength': wavelengths,
                        'flux': flux,
                        'type': sn_type,
                        'phase': float(epoch_phase),
                        'name': f"{obj_name}_epoch{epoch_idx}",
                        'source': 'snid',
                    })

                return templates

            except (ValueError, IndexError):
                pass   # Fall through to Super-SNID format attempt

        # --- Attempt Super-SNID matrix format ---
        # Header: nspec  nwave  wmin  wmax  nfeatures  name  redshift  type  ...
        # Followed by nfeatures rows of feature data, then one epoch-header line
        # starting with 0 containing all nspec phases, then nwave rows each with
        # wavelength followed by nspec flux values (already continuum-subtracted).
        lines_raw = []
        with open(file_path, 'r') as fh:
            for line in fh:
                stripped = line.strip()
                if stripped and not stripped.startswith('#'):
                    lines_raw.append(stripped)

        if len(lines_raw) >= 3:
            try:
                header = lines_raw[0].split()
                nspec = int(header[0])
                nwave = int(header[1])
                sn_type = header[7]
                obj_name = header[5]

                if nspec < 1 or nwave < 10:
                    raise ValueError("Implausible nspec/nwave")

                # The nfeatures metadata rows follow the header.
                # After that comes the epoch-header line (starts with '0') and
                # then nwave data rows. We scan forward to find the epoch line.
                epoch_line_idx = None
                for i in range(1, len(lines_raw)):
                    parts = lines_raw[i].split()
                    if parts[0] == '0' and len(parts) == nspec + 1:
                        epoch_line_idx = i
                        break

                if epoch_line_idx is None:
                    raise ValueError("Could not find epoch header line")

                phases = [float(p) for p in lines_raw[epoch_line_idx].split()[1:]]

                # Read nwave data rows (wavelength + nspec flux values)
                data_start = epoch_line_idx + 1
                if len(lines_raw) < data_start + nwave:
                    raise ValueError("Not enough data rows")

                wavelengths = np.zeros(nwave)
                flux_matrix = np.zeros((nwave, nspec))
                for i in range(nwave):
                    row = lines_raw[data_start + i].split()
                    wavelengths[i] = float(row[0])
                    for j in range(nspec):
                        flux_matrix[i, j] = float(row[j + 1])

                templates = []
                for j, phase in enumerate(phases):
                    flux = flux_matrix[:, j]
                    # Flux is already continuum-subtracted; skip zero-only epochs
                    if np.max(np.abs(flux)) == 0:
                        continue
                    templates.append({
                        'wavelength': wavelengths,
                        'flux': flux,
                        'type': sn_type,
                        'phase': float(phase),
                        'name': f"{obj_name}_p{phase:+.1f}",
                        'source': 'super_snid',
                        'pre_flattened': True,
                    })

                if templates:
                    return templates

            except (ValueError, IndexError):
                pass   # Fall through to two-column ASCII fallback

        # --- Fallback: two-column ASCII (wavelength, flux) ---
        # Manual parser: skips comment lines and malformed rows, uses only
        # the first two numeric columns (extra columns are ignored).
        wave_list, flux_list = [], []
        with open(file_path, 'r') as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped or stripped.startswith('#'):
                    continue
                parts = stripped.split()
                if len(parts) < 2:
                    continue  # skip single-value or empty lines
                try:
                    w = float(parts[0])
                    f = float(parts[1])
                    wave_list.append(w)
                    flux_list.append(f)
                except ValueError:
                    continue  # skip non-numeric lines

        if len(wave_list) < 2:
            raise ValueError(
                f"Could not parse {file_path}: fewer than 2 valid data rows found"
            )

        wavelengths = np.array(wave_list)
        flux = np.array(flux_list)
        max_flux = np.max(np.abs(flux))
        if max_flux > 0:
            flux = flux / max_flux

        # Infer type/phase from filename (e.g. sn1999aa_Ia_+5.dat)
        sn_type = 'Unknown'
        phase = 0.0
        for part in name.split('_')[1:]:
            if part in ('Ia', 'Ib', 'Ic', 'II', 'IIn', 'IIP', 'IIL', 'Ic-BL', 'Ia-pec'):
                sn_type = part
            else:
                try:
                    phase = float(part)
                except ValueError:
                    pass

        # Override with comment metadata if present
        if comment_type is not None:
            sn_type = comment_type
        if comment_phase is not None:
            phase = comment_phase

        return {
            'wavelength': wavelengths,
            'flux': flux,
            'type': sn_type,
            'phase': phase,
            'name': name,
            'source': 'ascii',
        }

    @staticmethod
    def download_github_templates(repo_url: str,
                                   branch: str = "master",
                                   subdirectory: str = "",
                                   cache_dir: Optional[Union[str, Path]] = None) -> Path:
        """
        Download template files from a GitHub repository.

        :param repo_url: GitHub repository URL (e.g., 'https://github.com/metal-sn/SESNtemple')
        :param branch: Branch name (default: 'master')
        :param subdirectory: Subdirectory within repo containing templates
        :param cache_dir: Local directory to cache files. If None, uses ~/.redback/spectral_templates/
        :return: Path to downloaded template directory
        """
        import urllib.request
        import zipfile
        import tempfile

        if cache_dir is None:
            cache_dir = Path.home() / '.redback' / 'spectral_templates'
        else:
            cache_dir = Path(cache_dir).expanduser()

        cache_dir.mkdir(parents=True, exist_ok=True)

        # Parse repo URL to get owner and repo name
        parts = repo_url.rstrip('/').split('/')
        repo_name = parts[-1]
        owner = parts[-2]

        # Create unique cache directory for this repo
        repo_cache = cache_dir / f"{owner}_{repo_name}"

        if repo_cache.exists() and any(repo_cache.iterdir()):
            logger.info(f"Using cached templates from {repo_cache}")
            if subdirectory:
                return repo_cache / subdirectory
            return repo_cache

        # Download zip archive
        zip_url = f"https://github.com/{owner}/{repo_name}/archive/refs/heads/{branch}.zip"
        logger.info(f"Downloading templates from {zip_url}")

        try:
            with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_file:
                urllib.request.urlretrieve(zip_url, tmp_file.name)
                tmp_path = tmp_file.name

            # Extract zip
            with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
                zip_ref.extractall(cache_dir)

            # Rename extracted directory
            extracted_dir = cache_dir / f"{repo_name}-{branch}"
            if extracted_dir.exists():
                if repo_cache.exists():
                    import shutil
                    shutil.rmtree(repo_cache)
                extracted_dir.rename(repo_cache)

            Path(tmp_path).unlink()  # Clean up zip file

            logger.info(f"Templates downloaded to {repo_cache}")

            if subdirectory:
                return repo_cache / subdirectory
            return repo_cache

        except Exception as e:
            logger.error(f"Failed to download templates: {e}")
            raise

    @classmethod
    def from_sesn_templates(cls, cache_dir: Optional[Union[str, Path]] = None) -> 'SpectralTemplateMatcher':
        """
        Create matcher from METAL/SESNtemple stripped-envelope SN templates.

        Downloads templates from: https://github.com/metal-sn/SESNtemple

        :param cache_dir: Local cache directory (default: ~/.redback/spectral_templates/)
        :return: SpectralTemplateMatcher instance
        """
        template_dir = cls.download_github_templates(
            'https://github.com/metal-sn/SESNtemple',
            subdirectory='SNIDtemplates',
            cache_dir=cache_dir
        )
        return cls.from_snid_template_directory(template_dir, recursive=True)

    @classmethod
    def from_super_snid_templates(cls, cache_dir: Optional[Union[str, Path]] = None) -> 'SpectralTemplateMatcher':
        """
        Create matcher from the Super-SNID template library (Magill et al. 2025).

        Downloads the repository from https://github.com/dkjmagill/QUB-SNID-Templates,
        extracts the inner templates.zip, and loads all .lnw template files.

        Please cite: Magill et al. 2025 (Zenodo DOI: 10.5281/zenodo.15167198)

        :param cache_dir: Local cache directory (default: ~/.redback/spectral_templates/)
        :return: SpectralTemplateMatcher instance
        """
        import zipfile

        repo_dir = cls.download_github_templates(
            'https://github.com/dkjmagill/QUB-SNID-Templates',
            branch='main',
            cache_dir=cache_dir
        )

        templates_dir = repo_dir / 'templates'
        if not templates_dir.exists():
            # Extract the inner templates.zip
            inner_zip = repo_dir / 'templates.zip'
            if not inner_zip.exists():
                raise FileNotFoundError(
                    f"Could not find templates.zip in {repo_dir}. "
                    "The repository structure may have changed."
                )
            logger.info(f"Extracting {inner_zip} ...")
            with zipfile.ZipFile(inner_zip, 'r') as zf:
                zf.extractall(repo_dir)

        if not templates_dir.exists():
            raise FileNotFoundError(
                f"Expected a 'templates/' directory in {repo_dir} after extraction."
            )

        return cls.from_snid_template_directory(templates_dir)

    @classmethod
    def from_snid_template_directory(cls, directory: Union[str, Path],
                                      file_pattern: str = "*.lnw",
                                      recursive: bool = False) -> 'SpectralTemplateMatcher':
        """
        Create a SpectralTemplateMatcher from a directory of SNID template files.

        :param directory: Path to directory containing SNID template files
        :param file_pattern: Glob pattern for template files (default: "*.lnw")
        :param recursive: If True, search subdirectories recursively (default: False)
        :return: SpectralTemplateMatcher instance
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        glob_fn = directory.rglob if recursive else directory.glob

        template_files = list(glob_fn(file_pattern))
        if len(template_files) == 0:
            # Try other common extensions
            template_files = (list(glob_fn("*.lnw")) +
                              list(glob_fn("*.dat")) +
                              list(glob_fn("*.txt")))

        if len(template_files) == 0:
            raise ValueError(f"No template files found in {directory}")

        templates = []
        for file_path in template_files:
            try:
                result = cls.parse_snid_template_file(file_path)
                # parse_snid_template_file returns a list for .lnw files
                # and a single dict for ASCII files
                if isinstance(result, list):
                    templates.extend(result)
                    logger.info(f"Loaded {len(result)} epoch(s) from {file_path.name}")
                else:
                    templates.append(result)
                    logger.info(f"Loaded 1 template from {file_path.name}")
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
                continue

        logger.info(f"Loaded {len(templates)} total template epochs from {directory}")
        return cls(templates=templates)


    def save_templates(self, output_dir: Union[str, Path], format: str = 'csv') -> None:
        """
        Save current templates to disk for later use.

        :param output_dir: Directory to save templates
        :param format: Output format ('csv' or 'dat')
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for template in self.templates:
            safe_name = template['name'].replace('/', '-').replace('\\', '-')
            filename = f"{safe_name}.{format}"
            filepath = output_dir / filename

            data = np.column_stack([template['wavelength'], template['flux']])

            if format == 'csv':
                # Save with metadata as comments, then header, then data
                with open(filepath, 'w') as f:
                    f.write(f"# Type: {template['type']}\n")
                    f.write(f"# Phase: {template['phase']}\n")
                    f.write("wavelength,flux\n")
                    for row in data:
                        f.write(f"{row[0]},{row[1]}\n")
            else:
                np.savetxt(filepath, data,
                          header=f"Type: {template['type']}\nPhase: {template['phase']}")

        logger.info(f"Saved {len(self.templates)} templates to {output_dir}")

    def filter_templates(self, types: Optional[list] = None,
                         phase_range: Optional[tuple] = None) -> 'SpectralTemplateMatcher':
        """
        Create a new matcher with filtered templates.

        :param types: List of SN types to include (e.g., ['Ia', 'Ib'])
        :param phase_range: Tuple of (min_phase, max_phase) in days
        :return: New SpectralTemplateMatcher with filtered templates
        """
        filtered = self.templates.copy()

        if types is not None:
            filtered = [t for t in filtered if t['type'] in types]

        if phase_range is not None:
            min_phase, max_phase = phase_range
            filtered = [t for t in filtered if min_phase <= t['phase'] <= max_phase]

        logger.info(f"Filtered to {len(filtered)} templates")
        return SpectralTemplateMatcher(templates=filtered)


class PhotometricClassifier:
    """
    Classify transients from light curve shape using redback photometric models.

    Compares an observed normalised light curve against a set of representative
    model light curves using dynamic time warping (DTW), which is robust to
    10–20 day timing offsets between objects of the same type.

    Returns a :class:`ClassificationResult` with method='photometric'.
    """

    # Default model templates: (model_name, parameters, label)
    _DEFAULT_MODEL_PARAMS = [
        ('arnett', dict(f_nickel=0.6, mej=1.2, vej=10000, kappa=0.1,
                        kappa_gamma=10.0, temperature_floor=3000, redshift=0.01),
         'Ia'),
        ('arnett', dict(f_nickel=0.05, mej=5.0, vej=5000, kappa=0.07,
                        kappa_gamma=10.0, temperature_floor=3500, redshift=0.01),
         'IIP'),
        ('basic_magnetar_powered', dict(P0=2.0, Bp=1e14, Mns=1.4, chi=90.0,
                                        mej=5.0, vej=8000, kappa=0.1,
                                        kappa_gamma=10.0, redshift=0.05),
         'SLSN-I'),
        ('arnett', dict(f_nickel=0.2, mej=3.0, vej=15000, kappa=0.08,
                        kappa_gamma=10.0, temperature_floor=3000, redshift=0.02),
         'Ic-BL'),
    ]

    def __init__(self, model_templates: Optional[list] = None) -> None:
        """
        :param model_templates: List of (model_name, parameters_dict, label) tuples.
            If None, uses built-in defaults.
        """
        self.model_templates = model_templates or self._DEFAULT_MODEL_PARAMS
        self._lc_cache = {}

    @staticmethod
    def _dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute Dynamic Time Warping distance between two 1-D sequences.

        Uses a simple O(N*M) cumulative distance matrix without Sakoe-Chiba band.
        Both sequences should already be normalised (peak = 1).

        :param a: First sequence
        :param b: Second sequence
        :return: DTW distance (lower = more similar)
        """
        n, m = len(a), len(b)
        dtw = np.full((n + 1, m + 1), np.inf)
        dtw[0, 0] = 0.0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(a[i - 1] - b[j - 1])
                dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
        return float(dtw[n, m])

    def _evaluate_model_lc(self, model_name: str, params: dict,
                            time_grid: np.ndarray) -> Optional[np.ndarray]:
        """Evaluate a redback model on time_grid; returns normalised flux or None."""
        key = (model_name, tuple(sorted(params.items())))
        if key in self._lc_cache:
            return self._lc_cache[key]
        try:
            from redback.model_library import all_models_dict
            func = all_models_dict[model_name]
            flux = func(time_grid, **params)
            flux = np.asarray(flux, dtype=float)
            peak = np.max(flux)
            if peak > 0:
                flux = flux / peak
            self._lc_cache[key] = flux
            return flux
        except Exception as e:
            logger.warning(f"PhotometricClassifier: failed to evaluate model '{model_name}': {e}")
            self._lc_cache[key] = None
            return None

    def classify_from_lightcurve(self, transient, time_grid: Optional[np.ndarray] = None,
                                  top_n: int = 5) -> ClassificationResult:
        """
        Classify a transient from its bolometric or single-band light curve shape.

        :param transient: A redback transient object with ``time`` and a flux/
            luminosity attribute, or any object with ``time`` and ``flux_density``
            arrays.
        :param time_grid: Time grid (days) on which to evaluate models. If None,
            uses the transient's own time array.
        :param top_n: Number of top matches to use for probability estimation.
        :return: :class:`ClassificationResult` with method='photometric'.
        """
        # Extract observed LC
        obs_time = np.asarray(getattr(transient, 'time', None) or
                              getattr(transient, 'time_days', None))
        # Try common flux attributes
        for attr in ('flux_density', 'Lum50', 'magnitude', 'counts'):
            obs_flux = getattr(transient, attr, None)
            if obs_flux is not None:
                obs_flux = np.asarray(obs_flux, dtype=float)
                break

        if obs_time is None or obs_flux is None:
            return ClassificationResult(
                best_type='Unknown', best_phase=0.0, best_redshift=0.0,
                rlap=0.0, confidence='low', type_probabilities={},
                top_matches=[], best_template_name='',
                method='photometric', warnings=['Could not extract time/flux from transient'],
            )

        # Normalise observed flux
        peak = np.max(np.abs(obs_flux))
        if peak > 0:
            obs_norm = obs_flux / peak
        else:
            obs_norm = obs_flux

        if time_grid is None:
            time_grid = obs_time

        all_matches = []
        for model_name, params, label in self.model_templates:
            model_lc = self._evaluate_model_lc(model_name, params, time_grid)
            if model_lc is None:
                continue
            # Interpolate model onto observed time points
            from scipy.interpolate import interp1d
            f_interp = interp1d(time_grid, model_lc, bounds_error=False,
                                fill_value=(model_lc[0], model_lc[-1]))
            model_at_obs = f_interp(obs_time)
            dist = self._dtw_distance(obs_norm, model_at_obs)
            all_matches.append({
                'type': label,
                'phase': 0.0,
                'redshift': 0.0,
                'rlap': float(1.0 / (dist + 1e-6)),  # invert distance to rlap-like score
                'correlation': float(1.0 / (dist + 1e-6)),
                'template_name': f'{model_name}_{label}',
                'dtw_distance': dist,
            })

        if len(all_matches) == 0:
            return ClassificationResult(
                best_type='Unknown', best_phase=0.0, best_redshift=0.0,
                rlap=0.0, confidence='low', type_probabilities={},
                top_matches=[], best_template_name='',
                method='photometric', warnings=['No model templates could be evaluated'],
            )

        all_matches.sort(key=lambda x: x['dtw_distance'])
        top_matches = all_matches[:min(top_n, len(all_matches))]

        # Softmax over negative DTW distances (lower dist = better)
        from collections import defaultdict
        type_dists = defaultdict(list)
        for m in top_matches:
            type_dists[m['type']].append(m['dtw_distance'])
        type_mean_dist = {t: float(np.mean(v)) for t, v in type_dists.items()}
        min_dist = min(type_mean_dist.values())
        exp_scores = {t: np.exp(-(d - min_dist)) for t, d in type_mean_dist.items()}
        total = sum(exp_scores.values())
        type_probabilities = {t: float(v / total) for t, v in exp_scores.items()}

        best = top_matches[0]
        best_score = best['rlap']
        confidence = 'high' if best['dtw_distance'] < 0.5 else \
                     'medium' if best['dtw_distance'] < 2.0 else 'low'

        return ClassificationResult(
            best_type=best['type'],
            best_phase=best['phase'],
            best_redshift=best['redshift'],
            rlap=best_score,
            confidence=confidence,
            type_probabilities=type_probabilities,
            top_matches=top_matches,
            best_template_name=best['template_name'],
            method='photometric',
        )


def combine_classifications(spectral_result: ClassificationResult,
                            photometric_result: ClassificationResult,
                            spectral_weight: float = 0.7) -> ClassificationResult:
    """
    Combine spectral and photometric classification results into a single estimate.

    Type probabilities are computed as a weighted average:
    ``p_combined = spectral_weight * p_spectral + (1 - spectral_weight) * p_photometric``

    :param spectral_result: :class:`ClassificationResult` from
        :meth:`SpectralTemplateMatcher.classify_spectrum`
    :param photometric_result: :class:`ClassificationResult` from
        :meth:`PhotometricClassifier.classify_from_lightcurve`
    :param spectral_weight: Weight given to spectral classification (0–1).
        Default 0.7 reflects that spectral features are more discriminating.
    :return: Combined :class:`ClassificationResult` with method='combined'
    """
    photo_weight = 1.0 - spectral_weight

    # Merge type sets
    all_types = set(spectral_result.type_probabilities) | set(photometric_result.type_probabilities)
    combined_probs = {}
    for t in all_types:
        p_spec = spectral_result.type_probabilities.get(t, 0.0)
        p_phot = photometric_result.type_probabilities.get(t, 0.0)
        combined_probs[t] = spectral_weight * p_spec + photo_weight * p_phot

    # Normalise (in case the two probability dicts don't cover the same types)
    total = sum(combined_probs.values())
    if total > 0:
        combined_probs = {t: v / total for t, v in combined_probs.items()}

    best_type = max(combined_probs, key=combined_probs.get)
    # Take the best-redshift and best-phase from the spectral result (more precise)
    combined_rlap = (spectral_weight * spectral_result.rlap +
                     photo_weight * photometric_result.rlap)
    confidence = 'high' if combined_rlap > 8 else 'medium' if combined_rlap > 5 else 'low'

    warnings = spectral_result.warnings + photometric_result.warnings

    return ClassificationResult(
        best_type=best_type,
        best_phase=spectral_result.best_phase,
        best_redshift=spectral_result.best_redshift,
        rlap=combined_rlap,
        confidence=confidence,
        type_probabilities=combined_probs,
        top_matches=spectral_result.top_matches,
        best_template_name=spectral_result.best_template_name,
        best_template_source=spectral_result.best_template_source,
        method='combined',
        warnings=warnings,
    )