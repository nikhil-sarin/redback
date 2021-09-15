# import gwemlightcurves.sampler.model as gwem
# import kilonovalightcurves
# import numpy as np
from ..utils import get_functions_dict, calc_ABmag_from_fluxdensity, calc_fluxdensity_from_ABmag, logger



# def simple_one_component_kilonova_model(time, mej, vej, beta, kappa_r, **kwargs):
#     """
#     A simple one component kilonova model with second order parameters
#     from GWEMlightcurves using Metzger kilonovae review Eq (14) as a guide
#     :param time:
#     :param mej:
#     :param vej:
#     :param beta:
#     :param kappa_r:
#     :param kwargs:
#     :return: flux density or magnitude
#     """
#     t_days, lbol, magnitudes = gwem.Me2017_model(mej=mej, vej=vej, beta=beta, kappa_r=kappa_r)
