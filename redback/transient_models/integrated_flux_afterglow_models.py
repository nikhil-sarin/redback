from inspect import isfunction
import numpy as np

from scipy.integrate import simps

from redback.utils import logger

integrated_flux_base_models = ['tophat', 'cocoon', 'gaussian',
                               'kn_afterglow', 'cone_afterglow',
                               'gaussiancore', 'gaussian',
                               'smoothpowerlaw', 'powerlawcore',
                               'tophat']


def integrated_flux_afterglowpy_base_model(time, **kwargs):
    from ..model_library import modules_dict  # import model library in function to avoid circular dependency
    base_model = kwargs['base_model']

    if isfunction(base_model):
        function = base_model
    elif base_model not in integrated_flux_base_models:
        logger.warning('{} is not implemented as a base model'.format(base_model))
        raise ValueError('Please choose a different base model')
    elif isinstance(base_model, str):
        function = modules_dict['afterglow_models'][base_model]
    else:
        raise ValueError("Not a valid base model.")

    frequency_bounds = kwargs['frequency']  # should be 2 numbers that serve as start and end point
    nu_1d = np.linspace(frequency_bounds[0], frequency_bounds[1], 3)
    tt, nu = np.meshgrid(time, nu_1d)  # meshgrid makes 2D t and n
    tt = tt.flatten()
    nu = nu.flatten()
    kwargs['frequency'] = nu
    flux_density = function(tt, **kwargs)
    lightcurve_at_nu = flux_density.reshape(len(nu_1d), len(time))
    prefactor = 1e-26
    lightcurve_at_nu = prefactor * lightcurve_at_nu
    integrated_flux = simps(lightcurve_at_nu, axis=0, x=nu_1d)
    return integrated_flux


def integrated_flux_rate_model(time, **kwargs):
    """
    :param time:
    :param kwargs:
    :return: rate
    """
    prefactor = kwargs.get('prefactor', 0)
    dt = kwargs.get('dt', 1)
    background_rate = kwargs.get('background_rate', 0)
    integrated_flux = integrated_flux_afterglowpy_base_model(time, **kwargs)
    rate = (prefactor * integrated_flux + background_rate) * dt
    return rate
