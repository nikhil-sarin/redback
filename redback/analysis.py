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



class SpectralTemplateMatcher(object):
    """
    Match spectra to template library (similar to SNID).

    This class provides functionality for spectral template matching,
    allowing classification of transients based on spectral features.
    Templates can be loaded from a custom library or generated from
    redback spectral models.
    """

    def __init__(self, template_library_path: Optional[Union[str, Path]] = None,
                 templates: Optional[list] = None) -> None:
        """
        Initialize the SpectralTemplateMatcher with a template library.

        :param template_library_path: Path to a directory containing template files.
            Each template file should be a CSV or text file with columns for
            wavelength (in Angstroms) and flux. If None, uses built-in templates.
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
        Load built-in default templates.

        Currently generates simple blackbody templates at different temperatures
        to serve as a baseline. Users should provide their own template library
        for production use.

        :return: List of template dictionaries
        """
        logger.info("Loading default blackbody templates")
        templates = []

        # Generate simple blackbody templates at different temperatures
        wavelengths = np.linspace(3000, 10000, 1000)  # Angstroms

        # Type Ia-like templates (hotter)
        for phase in [-10, -5, 0, 5, 10, 15, 20]:
            # Temperature evolution: peak around max light
            temp = 12000 - 200 * phase  # Simple temperature evolution
            temp = max(temp, 5000)  # Minimum temperature
            flux = self._blackbody_flux(wavelengths, temp)
            templates.append({
                'wavelength': wavelengths,
                'flux': flux / np.max(flux),  # Normalize
                'type': 'Ia',
                'phase': phase,
                'name': f'Ia_phase_{phase}'
            })

        # Type II-like templates (cooler)
        for phase in [0, 10, 20, 30, 50]:
            temp = 8000 - 50 * phase
            temp = max(temp, 4000)
            flux = self._blackbody_flux(wavelengths, temp)
            templates.append({
                'wavelength': wavelengths,
                'flux': flux / np.max(flux),
                'type': 'II',
                'phase': phase,
                'name': f'II_phase_{phase}'
            })

        # Type Ib/c-like templates
        for phase in [-5, 0, 5, 10, 15]:
            temp = 10000 - 150 * phase
            temp = max(temp, 5500)
            flux = self._blackbody_flux(wavelengths, temp)
            templates.append({
                'wavelength': wavelengths,
                'flux': flux / np.max(flux),
                'type': 'Ib/c',
                'phase': phase,
                'name': f'Ibc_phase_{phase}'
            })

        return templates

    def _blackbody_flux(self, wavelength: np.ndarray, temperature: float) -> np.ndarray:
        """
        Calculate blackbody flux for given wavelength and temperature.

        :param wavelength: Wavelength array in Angstroms
        :param temperature: Temperature in Kelvin
        :return: Flux array (arbitrary units, will be normalized)
        """
        from redback.constants import planck, speed_of_light, boltzmann_constant

        # Convert wavelength to cm
        wavelength_cm = wavelength * 1e-8

        # Planck function
        h = planck
        c = speed_of_light
        k = boltzmann_constant

        # B_lambda (erg/s/cm^2/cm/sr)
        exponent = (h * c) / (wavelength_cm * k * temperature)
        # Avoid overflow
        exponent = np.clip(exponent, None, 700)
        flux = (2 * h * c**2 / wavelength_cm**5) / (np.exp(exponent) - 1)

        return flux

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
                       method: str = 'correlation',
                       return_all_matches: bool = False) -> Union[dict, list]:
        """
        Find best-matching template for an observed spectrum.

        :param spectrum: Spectrum object with angstroms and flux_density attributes
        :param redshift_range: Tuple of (z_min, z_max) for redshift search
        :param n_redshift_points: Number of redshift values to try
        :param method: Matching method - 'correlation' (Pearson), 'chi2', or 'both'
        :param return_all_matches: If True, return sorted list of all matches
        :return: Best match dictionary with keys:
            - 'type': Supernova type classification
            - 'phase': Phase in days from maximum
            - 'redshift': Best-fit redshift
            - 'correlation': Pearson correlation coefficient
            - 'chi2': Chi-squared value (if method includes chi2)
            - 'template_name': Name of matched template
        """
        from scipy.interpolate import interp1d
        from scipy.stats import pearsonr

        if len(self.templates) == 0:
            raise ValueError("No templates loaded. Add templates before matching.")

        all_matches = []

        # Get observed spectrum data
        obs_wavelength = spectrum.angstroms
        obs_flux = spectrum.flux_density

        # Normalize observed spectrum
        obs_flux_norm = obs_flux / np.max(np.abs(obs_flux))

        # Check if we have errors
        has_errors = hasattr(spectrum, 'flux_density_err') and spectrum.flux_density_err is not None
        if has_errors:
            obs_flux_err = spectrum.flux_density_err

        # Try each template at different redshifts
        for template in self.templates:
            for z in np.linspace(redshift_range[0], redshift_range[1], n_redshift_points):
                # Redshift template wavelengths to observed frame
                template_wave_obs = template['wavelength'] * (1 + z)

                # Check for wavelength overlap
                min_overlap = max(np.min(template_wave_obs), np.min(obs_wavelength))
                max_overlap = min(np.max(template_wave_obs), np.max(obs_wavelength))

                if max_overlap <= min_overlap:
                    continue  # No overlap

                # Interpolate template to observed wavelengths
                interp_func = interp1d(template_wave_obs, template['flux'],
                                       bounds_error=False, fill_value=np.nan)
                template_flux_interp = interp_func(obs_wavelength)

                # Mask invalid values
                valid_mask = ~np.isnan(template_flux_interp) & ~np.isnan(obs_flux_norm)
                valid_mask &= (template_flux_interp != 0)

                if np.sum(valid_mask) < 10:
                    continue  # Not enough valid points

                obs_valid = obs_flux_norm[valid_mask]
                template_valid = template_flux_interp[valid_mask]

                # Calculate correlation
                match_result = {
                    'type': template['type'],
                    'phase': template['phase'],
                    'redshift': z,
                    'template_name': template.get('name', f"{template['type']}_p{template['phase']}"),
                    'n_valid_points': int(np.sum(valid_mask))
                }

                if method in ['correlation', 'both']:
                    try:
                        corr, p_value = pearsonr(obs_valid, template_valid)
                        match_result['correlation'] = corr
                        match_result['p_value'] = p_value
                    except Exception:
                        match_result['correlation'] = -1
                        match_result['p_value'] = 1.0

                if method in ['chi2', 'both']:
                    if has_errors:
                        obs_err_valid = obs_flux_err[valid_mask]
                        # Scale template to match observed flux
                        scale = np.sum(obs_valid * template_valid / obs_err_valid**2) / \
                                np.sum(template_valid**2 / obs_err_valid**2)
                        scaled_template = scale * template_valid
                        chi2 = np.sum(((obs_valid - scaled_template) / obs_err_valid)**2)
                        reduced_chi2 = chi2 / (len(obs_valid) - 1)
                        match_result['chi2'] = chi2
                        match_result['reduced_chi2'] = reduced_chi2
                        match_result['scale_factor'] = scale
                    else:
                        # Without errors, use variance
                        scale = np.sum(obs_valid * template_valid) / np.sum(template_valid**2)
                        scaled_template = scale * template_valid
                        residuals = obs_valid - scaled_template
                        chi2 = np.sum(residuals**2) / np.var(obs_valid)
                        match_result['chi2'] = chi2
                        match_result['scale_factor'] = scale

                all_matches.append(match_result)

        if len(all_matches) == 0:
            logger.warning("No valid matches found. Check wavelength coverage and templates.")
            return None

        # Sort by best match
        if method == 'chi2':
            all_matches.sort(key=lambda x: x.get('chi2', np.inf))
            best_match = all_matches[0]
        elif method == 'correlation':
            all_matches.sort(key=lambda x: -x.get('correlation', -1))
            best_match = all_matches[0]
        else:  # both
            # Rank by correlation primarily
            all_matches.sort(key=lambda x: -x.get('correlation', -1))
            best_match = all_matches[0]

        if return_all_matches:
            return all_matches
        else:
            return best_match

    def classify_spectrum(self, spectrum, redshift_range: tuple = (0, 0.5),
                          n_redshift_points: int = 50,
                          top_n: int = 5) -> dict:
        """
        Classify a spectrum and provide confidence metrics.

        :param spectrum: Spectrum object to classify
        :param redshift_range: Tuple of (z_min, z_max) for redshift search
        :param n_redshift_points: Number of redshift values to try
        :param top_n: Number of top matches to consider for classification
        :return: Classification result dictionary with:
            - 'best_type': Most likely type
            - 'best_phase': Phase of best match
            - 'best_redshift': Redshift of best match
            - 'correlation': Correlation of best match
            - 'type_probabilities': Dict of type likelihoods based on top matches
            - 'top_matches': List of top N matches
        """
        all_matches = self.match_spectrum(spectrum, redshift_range=redshift_range,
                                          n_redshift_points=n_redshift_points,
                                          method='correlation',
                                          return_all_matches=True)

        if all_matches is None or len(all_matches) == 0:
            return {'best_type': None, 'error': 'No valid matches found'}

        # Get top N matches
        top_matches = all_matches[:min(top_n, len(all_matches))]

        # Calculate type probabilities from correlations
        type_scores = {}
        for match in top_matches:
            sn_type = match['type']
            corr = match.get('correlation', 0)
            # Weight by correlation squared (higher correlation = more weight)
            weight = max(0, corr)**2
            if sn_type in type_scores:
                type_scores[sn_type] += weight
            else:
                type_scores[sn_type] = weight

        # Normalize to probabilities
        total_score = sum(type_scores.values())
        if total_score > 0:
            type_probabilities = {k: v / total_score for k, v in type_scores.items()}
        else:
            type_probabilities = {k: 0 for k in type_scores}

        best_match = top_matches[0]

        return {
            'best_type': best_match['type'],
            'best_phase': best_match['phase'],
            'best_redshift': best_match['redshift'],
            'correlation': best_match['correlation'],
            'type_probabilities': type_probabilities,
            'top_matches': top_matches
        }

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
        obs_flux = spectrum.flux_density / np.max(spectrum.flux_density)

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

        # Redshift template
        z = match_result['redshift']
        template_wave_obs = template['wavelength'] * (1 + z)

        # Interpolate template to observed wavelengths for comparison
        interp_func = interp1d(template_wave_obs, template['flux'],
                               bounds_error=False, fill_value=np.nan)
        template_flux_interp = interp_func(obs_wavelength)

        # Scale template to match observed flux
        valid_mask = ~np.isnan(template_flux_interp)
        if 'scale_factor' in match_result:
            scale = match_result['scale_factor']
        else:
            scale = np.nansum(obs_flux * template_flux_interp) / np.nansum(template_flux_interp**2)

        # Plot
        ax.plot(obs_wavelength, obs_flux, 'k-', label='Observed', alpha=0.8, lw=1.5)
        ax.plot(obs_wavelength, scale * template_flux_interp, 'r--',
                label=f"Template: {match_result['type']} (phase={match_result['phase']:.0f}d, z={z:.3f})",
                alpha=0.8, lw=1.5)

        ax.set_xlabel(r'Wavelength ($\mathrm{\AA}$)')
        ax.set_ylabel('Normalized Flux')
        title = f"Best Match: {match_result['type']}, r={match_result.get('correlation', 0):.3f}"
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
            'open_supernova_catalog': {
                'description': 'Open Supernova Catalog API',
                'url': 'https://sne.space/',
                'api': 'https://api.sne.space/',
                'citation': 'Guillochon et al. 2017'
            },
            'wiserep': {
                'description': 'Weizmann Interactive Supernova Data Repository',
                'url': 'https://www.wiserep.org/',
                'citation': 'Yaron & Gal-Yam 2012'
            }
        }

    @classmethod
    def download_templates_from_osc(cls, sn_types: list = None,
                                     max_per_type: int = 10,
                                     cache_dir: Optional[Union[str, Path]] = None) -> 'SpectralTemplateMatcher':
        """
        Download spectral templates from the Open Supernova Catalog.

        :param sn_types: List of SN types to download (e.g., ['Ia', 'II', 'Ib', 'Ic'])
            If None, downloads common types.
        :param max_per_type: Maximum number of spectra per type
        :param cache_dir: Directory to cache downloaded templates. If None, uses
            ~/.redback/spectral_templates/
        :return: SpectralTemplateMatcher instance with downloaded templates
        """
        import urllib.request
        import json

        if sn_types is None:
            sn_types = ['Ia', 'II', 'Ib', 'Ic', 'IIn', 'Ic-BL']

        if cache_dir is None:
            cache_dir = Path.home() / '.redback' / 'spectral_templates' / 'osc'
        else:
            cache_dir = Path(cache_dir)

        cache_dir.mkdir(parents=True, exist_ok=True)

        templates = []

        logger.info(f"Downloading templates from Open Supernova Catalog for types: {sn_types}")

        for sn_type in sn_types:
            logger.info(f"Fetching Type {sn_type} supernovae...")
            try:
                # Query OSC API for supernovae of this type
                # The API returns basic info; we need to get spectra separately
                api_url = f"https://api.sne.space/catalog?claimedtype={sn_type}&spectra&format=json"
                api_url = api_url.replace(' ', '%20')

                with urllib.request.urlopen(api_url, timeout=30) as response:
                    data = json.loads(response.read().decode())

                count = 0
                for sn_name, sn_data in data.items():
                    if count >= max_per_type:
                        break

                    if 'spectra' not in sn_data or len(sn_data['spectra']) == 0:
                        continue

                    # Get the first spectrum with data
                    for spec_entry in sn_data['spectra']:
                        if 'data' not in spec_entry:
                            continue

                        try:
                            spec_data = spec_entry['data']
                            wavelengths = []
                            fluxes = []

                            for point in spec_data:
                                if len(point) >= 2:
                                    wavelengths.append(float(point[0]))
                                    fluxes.append(float(point[1]))

                            if len(wavelengths) < 50:
                                continue

                            wavelengths = np.array(wavelengths)
                            fluxes = np.array(fluxes)

                            # Normalize
                            fluxes = fluxes / np.max(np.abs(fluxes))

                            # Extract phase if available
                            phase = 0.0
                            if 'time' in spec_entry:
                                try:
                                    phase = float(spec_entry['time'])
                                except (ValueError, TypeError):
                                    pass

                            templates.append({
                                'wavelength': wavelengths,
                                'flux': fluxes,
                                'type': sn_type,
                                'phase': phase,
                                'name': f"{sn_name}_{sn_type}_phase{phase:.0f}"
                            })

                            count += 1
                            logger.info(f"  Downloaded {sn_name} spectrum")
                            break  # Only take first spectrum per SN

                        except Exception as e:
                            logger.warning(f"  Failed to parse spectrum for {sn_name}: {e}")
                            continue

                logger.info(f"Downloaded {count} Type {sn_type} templates")

            except Exception as e:
                logger.warning(f"Failed to download Type {sn_type} templates: {e}")
                continue

        if len(templates) == 0:
            logger.warning("No templates downloaded. Using default templates.")
            return cls()

        return cls(templates=templates)

    @staticmethod
    def parse_snid_template_file(file_path: Union[str, Path]) -> dict:
        """
        Parse a SNID template file (.lnw format).

        SNID template files have a specific format:
        - Header lines starting with '#' or containing metadata
        - Data lines with wavelength and flux columns

        :param file_path: Path to .lnw or .dat template file
        :return: Dictionary with wavelength, flux, type, phase, and name
        """
        file_path = Path(file_path)

        wavelengths = []
        fluxes = []
        sn_type = 'Unknown'
        phase = 0.0
        name = file_path.stem

        # Parse filename for metadata (common SNID naming: sn1999aa_Ia_+5.lnw)
        parts = name.split('_')
        if len(parts) >= 2:
            # Try to extract type
            for part in parts[1:]:
                if part in ['Ia', 'Ib', 'Ic', 'II', 'IIn', 'IIP', 'IIL', 'Ic-BL', 'Ia-pec']:
                    sn_type = part
                elif part.startswith('+') or part.startswith('-') or part.replace('.', '').isdigit():
                    try:
                        phase = float(part.replace('+', ''))
                    except ValueError:
                        pass

        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
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
                    continue

                try:
                    parts = line.split()
                    if len(parts) >= 2:
                        wavelengths.append(float(parts[0]))
                        fluxes.append(float(parts[1]))
                except ValueError:
                    continue

        if len(wavelengths) == 0:
            raise ValueError(f"No valid data found in {file_path}")

        wavelengths = np.array(wavelengths)
        fluxes = np.array(fluxes)

        # Normalize
        fluxes = fluxes / np.max(np.abs(fluxes))

        return {
            'wavelength': wavelengths,
            'flux': fluxes,
            'type': sn_type,
            'phase': phase,
            'name': name
        }

    @classmethod
    def from_snid_template_directory(cls, directory: Union[str, Path],
                                      file_pattern: str = "*.lnw") -> 'SpectralTemplateMatcher':
        """
        Create a SpectralTemplateMatcher from a directory of SNID template files.

        :param directory: Path to directory containing SNID template files
        :param file_pattern: Glob pattern for template files (default: "*.lnw")
        :return: SpectralTemplateMatcher instance
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        template_files = list(directory.glob(file_pattern))
        if len(template_files) == 0:
            # Try other common extensions
            template_files = (list(directory.glob("*.lnw")) +
                            list(directory.glob("*.dat")) +
                            list(directory.glob("*.txt")))

        if len(template_files) == 0:
            raise ValueError(f"No template files found in {directory}")

        templates = []
        for file_path in template_files:
            try:
                template = cls.parse_snid_template_file(file_path)
                templates.append(template)
                logger.info(f"Loaded SNID template: {file_path.name}")
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
                continue

        logger.info(f"Loaded {len(templates)} SNID templates from {directory}")
        return cls(templates=templates)

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
            cache_dir = Path(cache_dir)

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

        return cls.from_snid_template_directory(template_dir)

    def save_templates(self, output_dir: Union[str, Path], format: str = 'csv') -> None:
        """
        Save current templates to disk for later use.

        :param output_dir: Directory to save templates
        :param format: Output format ('csv' or 'dat')
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for template in self.templates:
            filename = f"{template['name']}.{format}"
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