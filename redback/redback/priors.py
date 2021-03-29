import os

from bilby.core.prior import PriorDict

from redback.redback.utils import logger


def get_priors(model, data_mode):
    priors = PriorDict()
    try:
        filename = os.path.join(os.path.dirname(__file__), 'priors', f'{model}.prior')
        priors.from_file(filename)
    except FileNotFoundError as e:
        logger.warning(e)
        logger.warning('Returning empty PriorDict.')
    return priors
