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

    :param data:
    :param kernelT:
    :param kernelR:
    :param gp_kwargs:
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
        gp_T_err = T_err * 1.5
        gp_R_err = R_err * 1.5
    gp_T = george.GP(kernelT)
    gp_T.compute(t_data, gp_T_err + 1e-8)

    def neg_ln_like_T(p):
        gp_T.set_parameter_vector(p)
        return -gp_T.log_likelihood(temperature)

    def grad_neg_ln_like_T(p):
        gp_T.set_parameter_vector(p)
        return -gp_T.grad_log_likelihood(temperature)

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
        t_pred = np.linspace(t_data.min(), t_data.max(), 100)
        # Temperature prediction
        T_pred, T_pred_var = gp_T.predict(temperature, t_pred, return_var=True)
        T_pred_std = np.sqrt(T_pred_var)

        # Radius prediction
        R_pred, R_pred_var = gp_R.predict(radius, t_pred, return_var=True)
        R_pred_std = np.sqrt(R_pred_var)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
        ax1.errorbar(t_data, temperature, yerr=T_err, fmt='o', label='Data', color='blue')
        ax1.plot(t_pred, T_pred, label='GP Prediction', color='red')
        ax1.fill_between(t_pred, T_pred - T_pred_std, T_pred + T_pred_std, alpha=0.2, color='red',
                         label=r"$1\sigma$ GP uncertainty")
        ax2.errorbar(t_data, radius, yerr=R_err, fmt='o', label='Data', color='blue')
        ax2.plot(t_pred, R_pred, label='GP Prediction', color='red')
        ax2.fill_between(t_pred, R_pred - R_pred_std, R_pred + R_pred_std, alpha=0.2, color='red',
                         label=r"$1\sigma$ GP uncertainty")

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
