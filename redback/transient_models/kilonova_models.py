from gwemlightcurves.sampler.model import generate_lightcurve
from gwemlightcurves.KNModels.io.model import _MODELS
from gwemlightcurves.KNModels.table import KNTable
from astropy.table import Table, Column
from scipy.interpolate import interp1d


# MODELS = {}
# MODEL_CLASS_DICT = {}
#
# for m, v in _MODELS.items():
#     MODELS[m[0]] = v[0]
#     MODEL_CLASS_DICT[m[0]] = m[1]


def generate_single_lightcurve(model, tini, tmax, dt, **parameters):
    t = Table()
    for key in parameters.keys():
        val = parameters[key]
        t.add_column(Column(data=[val], name=key))
    t.add_column(Column(data=[tini], name="tini"))
    t.add_column(Column(data=[tmax], name="tmax"))
    t.add_column(Column(data=[dt], name="dt"))
    model_table = KNTable.model(model, t)
    return model_table["t"][0], model_table["lbol"][0], model_table["mag"][0]
    # if len(model_table) == 0:
    #     return [], [], []
    # else:



def generate_single_lightcurve_at_times(times, model, **parameters):
    tini = times[0]
    tmax = times[-1]
    dt = (tmax - tini)/(len(times) - 1)
    new_times, lbol, mag = generate_single_lightcurve(model=model, tini=times[0], tmax=times[-1], dt=dt, **parameters)

    lbol = interp1d(new_times, lbol)(times)
    mag = interp1d(new_times, mag)(times)
    return lbol, mag


def Me2017_bolometric(times, **parameters):
    return generate_single_lightcurve_at_times(times=times, model="Me2017", **parameters)

# import gwemlightcurves.sampler.model as gwem
# import kilonovalightcurves
# import numpy as np
# from redback.utils import get_functions_dict, calc_ABmag_from_flux_density, calc_flux_density_from_ABmag, logger

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
