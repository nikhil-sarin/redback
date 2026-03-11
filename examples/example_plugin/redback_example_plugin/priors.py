"""
Prior loader for the redback example plugin.

Registered via the 'redback.model.priors' entry point group.
"""

import os
from bilby.core.prior import PriorDict

PRIOR_DIR = os.path.join(os.path.dirname(__file__), 'priors')


def get_prior(model_name):
    """
    Return a PriorDict for the given model name, or None if unknown.

    :param model_name: Name of the model (str)
    :return: PriorDict or None
    """
    path = os.path.join(PRIOR_DIR, f'{model_name}.prior')
    if not os.path.exists(path):
        return None
    p = PriorDict()
    p.from_file(path)
    return p
