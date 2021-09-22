import extinction
from .fireball_models import predeceleration
from ..utils import logger, calc_ABmag_from_fluxdensity
from ..model_library import models_dict
import numpy as np


extinction_base_models = ['tophat', 'cocoon', 'gaussian',
                          'kn_afterglow', 'cone_afterglow',
                          'gaussiancore', 'gaussian',
                          'smoothpowerlaw', 'powerlawcore',
                          'tophat']


def extinction_with_afterglow_base_model(time, lognh, factor, **kwargs):
    base_model = kwargs['base_model']
    if base_model not in extinction_base_models:
        logger.warning('{} is not implemented as a base model'.format(base_model))
        raise ValueError('Please choose a different base model')

    if isinstance(base_model, str):
        function = models_dict['afterglow_models'][base_model]

    # logger.info('Using the extinction factor from Guver and Ozel 2009')
    factor = factor * 1e21
    nh = 10**lognh
    av = nh/factor
    frequency = kwargs['frequency']
    # logger.info('Using the fitzpatrick99 extinction law')
    mag_extinction = extinction.fitzpatrick99(frequency, av, r_v = 3.1)
    #read the base_model dict
    # logger.info('Using {} as the base model for extinction'.format(base_model))
    flux = function(time, **kwargs)
    flux = extinction.apply(mag_extinction, flux)
    output_magnitude = calc_ABmag_from_fluxdensity(flux).value
    return output_magnitude

def extinction_with_predeceleration(time, lognh, factor, **kwargs):
    """
    :param time: time in some unit.
    :param lognh: host galaxy column density
    :param factor: extinction factor
    :param kwargs: all params
    :return: flux or magnitude with extinction applied depending on kwargs
    """
    lc = predeceleration(time, **kwargs)
    lc = np.nan_to_num(lc)
    factor = factor * 1e21
    nh = 10 ** lognh
    av = nh/factor
    frequency = kwargs['frequency']
    mag_extinction = extinction.fitzpatrick99(frequency, av, r_v=3.1)
    lc = extinction.apply(mag_extinction, lc, inplace=True)
    if kwargs['output_format'] == 'flux_density':
        return lc
    elif kwargs['output_format'] == 'magnitude':
        return calc_ABmag_from_fluxdensity(lc).value
