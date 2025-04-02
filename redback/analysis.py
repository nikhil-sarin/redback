import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

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


def plot_lightcurve(transient, parameters, model, model_kwargs=None, **kwargs: None):
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
                               save=False, show=False, **kwargs)


def plot_multiband_lightcurve(transient, parameters, model, model_kwargs=None, **kwargs: None):
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
                                         save=False, show=False, **kwargs)


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
        ax[0].plot(time, nn, "--", lw=1, color='red', alpha=2.5, zorder=-1)
        ax[1].plot(time, np.rad2deg(alpha), "--", lw=1, color='red', alpha=2.5, zorder=-1)
        ax[2].plot(time, mu, "--", lw=1, color='red', alpha=2.5, zorder=-1)
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

def plot_gp_lightcurves(transient, gp_output, axes=None, band_colors=None):
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
                f_new = np.ones_like(t_new) * bands_to_frequency([band])
                X_new = np.column_stack((f_new, t_new))
                gp = gp_output.gp
                y_pred, y_cov = gp.predict(gp_output.scaled_y, X_new, return_cov=True)
                y_std = np.sqrt(np.diag(y_cov))
                y_lower = y_pred - 0.5 * y_std
                y_upper = y_pred + 0.5 * y_std
                ax.plot(t_new - ref_date, y_pred * gp_output.y_scaler, color=band_colors[band])
                ax.fill_between(t_new - ref_date, y_lower * gp_output.y_scaler, y_upper * gp_output.y_scaler, alpha=0.5,
                                color=band_colors[band])
        else:
            for band in transient.unique_bands:
                gp = gp_output.gp[band]
                y_pred, y_cov = gp.predict(gp_output.scaled_y[band], t_new, return_cov=True)
                y_std = np.sqrt(np.diag(y_cov))
                y_lower = y_pred - 0.5 * y_std
                y_upper = y_pred + 0.5 * y_std
                ax.plot(t_new - ref_date, y_pred * gp_output.y_scaler, color=band_colors[band])
                ax.fill_between(t_new - ref_date, y_lower * gp_output.y_scaler, y_upper * gp_output.y_scaler, alpha=0.5,
                                color=band_colors[band])
    else:
        y_pred, y_cov = gp_output.gp.predict(gp_output.scaled_y, t_new, return_cov=True)
        y_std = np.sqrt(np.diag(y_cov))
        y_lower = y_pred - 0.5 * y_std
        y_upper = y_pred + 0.5 * y_std

        ax.plot(t_new, y_pred * gp_output.y_scaler, color='red')
        ax.fill_between(t_new, y_lower * gp_output.y_scaler, y_upper * gp_output.y_scaler, alpha=0.5, color='red')
    return ax

def estimate_blackbody_temperature_and_radius(transient, window_duration=1, ignore_epoch_duration=0.5,
                                              use_flux_density_approximation=True):
    """
    Estimate the temperature and radius as a function of time for any optical transient
    """
    logger.info("Using the blackbody SED to estimate temperature and radius time series")
    if transient.data_mode in ['flux', 'luminosity']:
        raise ValueError("This method only works for flux density or magnitude data modes")
    if transient.data_mode in ['magnitude']:
        if use_flux_density_approximation:
            logger.warning("Using the flux density at effective wavelength approximation")
            logger.warning("This approximation is not correct for bandpass magnitudes and fluxes and tends to "
                           "effect radius estimation. Use with caution")
            df = pd.DataFrame()
            df['time'] = transient.x
            df['band'] = transient.sncosmo_bands
            df['mag'] = transient.y
            df['mag_err'] = transient.y_err
            df['epoch'] = (transient.x // window_duration).astype(int)

            epoch_fits = []
            for epoch, group in df.groupby('epoch'):
                if group['band'].nunique() < 3:
                    continue

                # Use the average time as the epoch time.
                t_epoch = group['time'].mean()
                logger.info(f"Epoch Time: {t_epoch}")

                if t_epoch < ignore_epoch_duration:
                    logger.info('ignoring epochs before {} days'.format(ignore_epoch_duration))
                    continue
                else:
                    # For each observation in the group, convert the mag to flux
                    waves = []
                    fluxes = []
                    flux_errs = []
                    for idx, row in group.iterrows():
                        band = row['band']
                        # Extract the scalar effective wavelength from the returned array
                        lam_eff = redback.utils.nu_to_lambda(redback.utils.bands_to_frequency([band]))[0]
                        f_lambda = redback.utils.abmag_to_flambda(row['mag'], lam_eff)
                        # Propagate mag error into flux error:
                        f_lambda_err = redback.utils.flux_err_from_mag_err(f_lambda, row['mag_err'])

                        waves.append(lam_eff)  # should be scalar value in Å
                        fluxes.append(f_lambda)
                        flux_errs.append(f_lambda_err)

                    # Convert lists to 1D numpy arrays.
                    waves = np.array(waves).flatten()
                    fluxes = np.array(fluxes).flatten()
                    flux_errs = np.array(flux_errs).flatten()
                    log_y_data = np.log(fluxes)
                    log_y_err = flux_errs / fluxes

                    # Set initial guesses (in log-space)
                    # For example, if you expect T ~ 6000 K and R ~ 1e15 cm:
                    initial_logT = np.log(6000)
                    initial_logR = np.log(1e15)
                    initial_guess = [initial_logT, initial_logR]

                    # Perform the fit in log-space.
                    popt, pcov = curve_fit(
                        log_blackbody_model_logTR,
                        waves,  # independent variable: wavelengths
                        log_y_data,  # dependent variable: ln(flux)
                        p0=initial_guess,
                        sigma=log_y_err,
                        maxfev=10000,
                        absolute_sigma=True
                    )

                    # Extract the best-fit values for logT and logR.
                    logT_best, logR_best = popt
                    perr = np.sqrt(np.diag(pcov))
                    err_logT, err_logR = perr

                    # Convert the parameters back from log-space.
                    T_fit = np.exp(logT_best)
                    R_fit = np.exp(logR_best)

                    # Propagate the uncertainties:
                    # For T = exp(logT), dT/d(logT) = T, so the uncertainty becomes:
                    T_err = T_fit * err_logT
                    R_err = R_fit * err_logR

                    epoch_fits.append({'time': t_epoch, 'T': T_fit, 'T_err': T_err,
                                       'R': R_fit, 'R_err': R_err})

                    # Convert epoch fits to a DataFrame.
            df_bb = pd.DataFrame(epoch_fits)
            # print("\nBlackbody fit results per epoch:")
            print(df_bb)

            t_data = df_bb['time'].values
            T_data = df_bb['T'].values
            T_err = df_bb['T_err'].values
            R_data = df_bb['R'].values
            R_err = df_bb['R_err'].values

    pass
