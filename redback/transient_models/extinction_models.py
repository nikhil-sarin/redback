from inspect import isfunction
import numpy as np

from redback.transient_models.fireball_models import predeceleration
from redback.utils import logger, calc_ABmag_from_flux_density

extinction_base_models = ['tophat', 'cocoon', 'gaussian',
                          'kn_afterglow', 'cone_afterglow',
                          'gaussiancore', 'gaussian',
                          'smoothpowerlaw', 'powerlawcore',
                          'tophat']
import astropy.units as uu

def extinction_with_afterglow_base_model(time, av, **kwargs):
    """
    :param time: time in observer frame in seconds
    :param av: absolute mag extinction
    :param kwargs: Must be all the parameters required by the base_model specified using kwargs['base_model']
    :return: flux_density or magnitude depending on kwargs['output_format']
    """
    import extinction  # noqa
    from redback.model_library import modules_dict  # import model library in function to avoid circular dependency
    base_model = kwargs['base_model']

    if isfunction(base_model):
        function = base_model
    elif base_model not in extinction_base_models:
        logger.warning('{} is not implemented as a base model'.format(base_model))
        raise ValueError('Please choose a different base model')
    elif isinstance(base_model, str):
        function = modules_dict['afterglow_models'][base_model]
    else:
        raise ValueError("Not a valid base model.")

    frequency = kwargs['frequency']
    # convert to angstrom
    frequency = (frequency * uu.Hz).to(uu.Angstrom).value
    # logger.info('Using the fitzpatrick99 extinction law')
    mag_extinction = extinction.fitzpatrick99(frequency, av, r_v=3.1)
    # read the base_model dict
    # logger.info('Using {} as the base model for extinction'.format(base_model))
    flux_density = function(time, **kwargs)
    flux_density = extinction.apply(mag_extinction, flux_density)
    if kwargs['output_format'] == 'flux_density':
        return flux_density.value
    elif kwargs['output_format'] == 'magnitude':
        return calc_ABmag_from_flux_density(flux_density).value

def extinction_with_galactic_dust_to_gas_ratio(time, lognh, factor=2.21, **kwargs):
    """
    :param time: time in observer frame in seconds
    :param lognh: log10 hydrogen column density
    :param factor: factor to convert nh to av i.e., av = nh/factor
    :param kwargs: Must be all the parameters required by the base_model specified using kwargs['base_model']
    :return: flux_density or magnitude depending on kwargs['output_format']
    """
    factor = factor * 1e21
    nh = 10 ** lognh
    av = nh / factor
    output = extinction_with_afterglow_base_model(time=time, av=av, **kwargs)
    return output


def extinction_with_predeceleration(time, lognh, factor, **kwargs):
    """
    :param time: time in some unit.
    :param lognh: host galaxy column density
    :param factor: extinction factor
    :param kwargs: all params
    :return: flux or magnitude with extinction applied depending on kwargs
    """
    import extinction  # noqa
    lc = predeceleration(time, **kwargs)
    lc = np.nan_to_num(lc)
    factor = factor * 1e21
    nh = 10 ** lognh
    av = nh / factor
    frequency = kwargs['frequency']
    # convert to angstrom
    frequency = (frequency * uu.Hz).to(uu.Angstrom).value
    mag_extinction = extinction.fitzpatrick99(frequency, av, r_v=3.1)
    lc = extinction.apply(mag_extinction, lc, inplace=True)
    if kwargs['output_format'] == 'flux_density':
        return lc
    elif kwargs['output_format'] == 'magnitude':
        return calc_ABmag_from_flux_density(lc).value
