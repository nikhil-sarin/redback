import numpy as np
import pandas as pd

from astropy.table import Table, Column
from scipy.interpolate import interp1d


def _generate_single_lightcurve(model, t_ini, t_max, dt, **parameters):
    """
    Generates a single lightcurve for a given `gwemlightcurves` model

    Parameters
    ----------
    model: str
        The `gwemlightcurve` model, e.g. 'DiUj2017'
    t_ini: float
        Starting time of the time array `gwemlightcurves` will calculate values at.
    t_max: float
        End time of the time array `gwemlightcurves` will calculate values at.
    dt: float
        Spacing of time uniform time steps.
    parameters: dict
        Function parameters for the given model.
    Returns
    ----------
    func, func: A bolometric function and the magnitude function.
    """
    from gwemlightcurves.KNModels.table import KNTable

    t = Table()
    for key in parameters.keys():
        val = parameters[key]
        t.add_column(Column(data=[val], name=key))
    t.add_column(Column(data=[t_ini], name="tini"))
    t.add_column(Column(data=[t_max], name="tmax"))
    t.add_column(Column(data=[dt], name="dt"))
    model_table = KNTable.model(model, t)
    return model_table["t"][0], model_table["lbol"][0], model_table["mag"][0]


def _generate_single_lightcurve_at_times(model, times, **parameters):
    """
    Generates a single lightcurve for a given `gwemlightcurves` model

    Parameters
    ----------
    model: str
        The `gwemlightcurve` model, e.g. 'DiUj2017'
    times: array_like
        Times at which we interpolate the `gwemlightcurves` values
    parameters: dict
        Function parameters for the given model.
    Returns
    ----------
    array_like, array_like: bolometric and magnitude arrays.
    """

    tini = times[0]
    tmax = times[-1]
    dt = (tmax - tini)/(len(times) - 1)
    gwem_times, lbol, mag = _generate_single_lightcurve(model=model, t_ini=times[0], t_max=times[-1],
                                                        dt=dt, **parameters)

    lbol = interp1d(gwem_times, lbol)(times)
    new_mag = []
    for m in mag:
        new_mag.append(interp1d(gwem_times, m)(times))
    return lbol, np.array(new_mag)


def _gwemlightcurve_interface_factory(model):
    """
    Generates `bilby`-compatible functions from `gwemlightcurve` models.
    This is currently very inefficient as there is an interpolation step.

    Parameters
    ----------
    model: str
        The `gwemlightcurve` model, e.g. 'DiUj2017'

    Returns
    ----------
    func, func: A bolometric function and the magnitude function.
    """

    default_filters = ['u', 'g', 'r', 'i', 'z', 'y', 'J', 'H', 'K']

    def interface_bolometric(times, **parameters):
        return _generate_single_lightcurve_at_times(model=model, times=times, **parameters)[0]

    def interface_all_magnitudes(times, **parameters):
        magnitudes = _generate_single_lightcurve_at_times(model=model, times=times, **parameters)[1]
        return pd.DataFrame(magnitudes.T, columns=default_filters)

    def interface_filtered_magnitudes(times, **parameters):
        filters = parameters.get('filters', default_filters)
        all_magnitudes = interface_all_magnitudes(times, **parameters)
        if len(filters) == 1:
            return all_magnitudes[filters[0]]

        filtered_magnitudes = np.zeros(len(times))
        for i, f in enumerate(filters):
            filtered_magnitudes[i] = all_magnitudes[f][i]

        return filtered_magnitudes

    return interface_bolometric, interface_filtered_magnitudes


gwem_DiUj2017_bolometric, gwem_DiUj2017_magnitudes = _gwemlightcurve_interface_factory("DiUj2017")
gwem_SmCh2017_bolometric, gwem_SmCh2017_magnitudes = _gwemlightcurve_interface_factory("SmCh2017")
gwem_Me2017_bolometric, gwem_Me2017_magnitudes = _gwemlightcurve_interface_factory("Me2017")
gwem_KaKy2016_bolometric, gwem_KaKy2016_magnitudes = _gwemlightcurve_interface_factory("KaKy2016")
gwem_WoKo2017_bolometric, gwem_WoKo2017_magnitudes = _gwemlightcurve_interface_factory("WoKo2017")
gwem_BaKa2016_bolometric, gwem_BaKa2016_magnitudes = _gwemlightcurve_interface_factory("BaKa2016")
gwem_Ka2017_bolometric, gwem_Ka2017_magnitudes = _gwemlightcurve_interface_factory("Ka2017")
gwem_Ka2017x2_bolometric, gwem_Ka2017x2_magnitudes = _gwemlightcurve_interface_factory("Ka2017x2")
gwem_Ka2017inc_bolometric, gwem_Ka2017inc_magnitudes = _gwemlightcurve_interface_factory("Ka2017inc")
gwem_Ka2017x2inc_bolometric, gwem_Ka2017x2inc_magnitudes = _gwemlightcurve_interface_factory("Ka2017x2inc")
gwem_RoFe2017_bolometric, gwem_RoFe2017_magnitudes = _gwemlightcurve_interface_factory("RoFe2017")
gwem_Bu2019_bolometric, gwem_Bu2019_magnitudes = _gwemlightcurve_interface_factory("Bu2019")
gwem_Bu2019inc_bolometric, gwem_Bu2019inc_magnitudes = _gwemlightcurve_interface_factory("Bu2019inc")
gwem_Bu2019lf_bolometric, gwem_Bu2019lf_magnitudes = _gwemlightcurve_interface_factory("Bu2019lf")
gwem_Bu2019lr_bolometric, gwem_Bu2019lr_magnitudes = _gwemlightcurve_interface_factory("Bu2019lr")
gwem_Bu2019lm_bolometric, gwem_Bu2019lm_magnitudes = _gwemlightcurve_interface_factory("Bu2019lm")
gwem_Bu2019lw_bolometric, gwem_Bu2019lw_magnitudes = _gwemlightcurve_interface_factory("Bu2019lw")
gwem_Bu2019re_bolometric, gwem_Bu2019re_magnitudes = _gwemlightcurve_interface_factory("Bu2019re")
gwem_Bu2019bc_bolometric, gwem_Bu2019bc_magnitudes = _gwemlightcurve_interface_factory("Bu2019bc")
gwem_Bu2019op_bolometric, gwem_Bu2019op_magnitudes = _gwemlightcurve_interface_factory("Bu2019op")
gwem_Bu2019ops_bolometric, gwem_Bu2019ops_magnitudes = _gwemlightcurve_interface_factory("Bu2019ops")
gwem_Bu2019rp_bolometric, gwem_Bu2019rp_magnitudes = _gwemlightcurve_interface_factory("Bu2019rp")
gwem_Bu2019rps_bolometric, gwem_Bu2019rps_magnitudes = _gwemlightcurve_interface_factory("Bu2019rps")
gwem_Wo2020dyn_bolometric, gwem_Wo2020dyn_magnitudes = _gwemlightcurve_interface_factory("Wo2020dyn")
gwem_Wo2020dw_bolometric, gwem_Wo2020dw_magnitudes = _gwemlightcurve_interface_factory("Wo2020dw")
gwem_Bu2019nsbh_bolometric, gwem_Bu2019nsbh_magnitudes = _gwemlightcurve_interface_factory("Bu2019nsbh")
