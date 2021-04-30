import numpy as np
import extinction
from ..utils import logger, calc_ABmag_from_fluxdensity
from ..utils import get_functions_dict
from . import afterglow_models

_, modules_dict = get_functions_dict(afterglow_models)

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
        function = modules_dict['afterglow_models'][base_model]

    logger.info('Using the extinction factor from Guver and Ozel 2009')
    factor = factor * 1e21
    nh = 10**lognh
    av = nh/factor
    frequency = kwargs['frequency']
    logger.info('Using the fitzpatrick99 extinction law')
    mag_extinction = extinction.fitzpatrick99(frequency, av, r_v = 3.1)
    #read the base_model dict
    logger.info('Using {} as the base model for extinction')
    flux = function(time, **kwargs)
    flux = extinction.apply(mag_extinction, flux)
    output_magnitude = calc_ABmag_from_fluxdensity(flux).value
    return output_magnitude