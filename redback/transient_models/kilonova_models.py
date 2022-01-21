from gwemlightcurves.KNModels.table import KNTable
from astropy.table import Table, Column
from scipy.interpolate import interp1d


def generate_single_lightcurve(model, t_ini, t_max, dt, **parameters):
    t = Table()
    for key in parameters.keys():
        val = parameters[key]
        t.add_column(Column(data=[val], name=key))
    t.add_column(Column(data=[t_ini], name="tini"))
    t.add_column(Column(data=[t_max], name="tmax"))
    t.add_column(Column(data=[dt], name="dt"))
    model_table = KNTable.model(model, t)
    return model_table["t"][0], model_table["lbol"][0], model_table["mag"][0]


def generate_single_lightcurve_at_times(times, model, **parameters):
    tini = times[0]
    tmax = times[-1]
    dt = (tmax - tini)/(len(times) - 1)
    new_times, lbol, mag = generate_single_lightcurve(model=model, t_ini=times[0], t_max=times[-1], dt=dt, **parameters)

    lbol = interp1d(new_times, lbol)(times)
    mag = interp1d(new_times, mag)(times)
    return lbol, mag


def gwemlightcurve_interface_factory(model_name):

    def interface_bolometric(times, **parameters):
        model = model_name
        return generate_single_lightcurve_at_times(times=times, model=model, **parameters)[0]

    def interface_magnitudes(times, **parameters):
        model = model_name
        return generate_single_lightcurve_at_times(times=times, model=model, **parameters)[1]

    return interface_bolometric, interface_magnitudes


DiUj2017_bolometric, DiUj2017_magnitudes = gwemlightcurve_interface_factory("DiUj2017")
SmCh2017_bolometric, SmCh2017_magnitudes = gwemlightcurve_interface_factory("SmCh2017")
Me2017_bolometric, Me2017_magnitudes = gwemlightcurve_interface_factory("Me2017")
KaKy2016_bolometric, KaKy2016_magnitudes = gwemlightcurve_interface_factory("KaKy2016")
WoKo2017_bolometric, WoKo2017_magnitudes = gwemlightcurve_interface_factory("WoKo2017")
BaKa2016_bolometric, BaKa2016_magnitudes = gwemlightcurve_interface_factory("BaKa2016")
Ka2017_bolometric, Ka2017_magnitudes = gwemlightcurve_interface_factory("Ka2017")
Ka2017x2_bolometric, Ka2017x2_magnitudes = gwemlightcurve_interface_factory("Ka2017x2")
Ka2017inc_bolometric, Ka2017inc_magnitudes = gwemlightcurve_interface_factory("Ka2017inc")
Ka2017x2inc_bolometric, Ka2017x2inc_magnitudes = gwemlightcurve_interface_factory("Ka2017x2inc")
RoFe2017_bolometric, RoFe2017_magnitudes = gwemlightcurve_interface_factory("RoFe2017")
Bu2019_bolometric, Bu2019_magnitudes = gwemlightcurve_interface_factory("Bu2019")
Bu2019inc_bolometric, Bu2019inc_magnitudes = gwemlightcurve_interface_factory("Bu2019inc")
Bu2019lf_bolometric, Bu2019lf_magnitudes = gwemlightcurve_interface_factory("Bu2019lf")
Bu2019lr_bolometric, Bu2019lr_magnitudes = gwemlightcurve_interface_factory("Bu2019lr")
Bu2019lm_bolometric, Bu2019lm_magnitudes = gwemlightcurve_interface_factory("Bu2019lm")
Bu2019lw_bolometric, Bu2019lw_magnitudes = gwemlightcurve_interface_factory("Bu2019lw")
Bu2019re_bolometric, Bu2019re_magnitudes = gwemlightcurve_interface_factory("Bu2019re")
Bu2019bc_bolometric, Bu2019bc_magnitudes = gwemlightcurve_interface_factory("Bu2019bc")
Bu2019op_bolometric, Bu2019op_magnitudes = gwemlightcurve_interface_factory("Bu2019op")
Bu2019ops_bolometric, Bu2019ops_magnitudes = gwemlightcurve_interface_factory("Bu2019ops")
Bu2019rp_bolometric, Bu2019rp_magnitudes = gwemlightcurve_interface_factory("Bu2019rp")
Bu2019rps_bolometric, Bu2019rps_magnitudes = gwemlightcurve_interface_factory("Bu2019rps")
Wo2020dyn_bolometric, Wo2020dyn_magnitudes = gwemlightcurve_interface_factory("Wo2020dyn")
Wo2020dw_bolometric, Wo2020dw_magnitudes = gwemlightcurve_interface_factory("Wo2020dw")
Bu2019nsbh_bolometric, Bu2019nsbh_magnitudes = gwemlightcurve_interface_factory("Bu2019nsbh")

print(Me2017_bolometric)
