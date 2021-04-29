import numpy as np
import scipy.special as ss
import extinction
from ..utils import logger, calc_ABmag_from_fluxdensity

extinction_base_models = ['tophat', 'cocoon', 'gaussian',
                          'kn_afterglow', 'cone_afterglow',
                          'gaussiancore', 'gaussian',
                          'smoothpowerlaw', 'powerlawcore',
                          'tophat']


def extinction_models(time, lognh, factor, **kwargs):
    base_model = kwargs['base_model']
    if base_model not in extinction_base_models:
        logger.warning('{} is not implemented as a base model'.format(base_model))
        raise ValueError('Please choose a different base model')

    if isinstance(base_model, str):
        function = model_dict[base_model]

    #from Guver and Ozel 2006
    factor = factor * 1e21
    nh = 10**lognh
    av = nh/factor
    frequency = kwargs['frequency']
    mag_extinction = extinction.fitzpatrick99(frequency, av, r_v = 3.1)
    #read the base_model dict
    flux = gaussiancore(time, **kwargs)
    intrinsic_mag = calc_ABmag_from_fluxdensity(flux).value
    output_magnitude = intrinsic_mag + mag_extinction
    return output_magnitude