import numpy as np
from typing import Union
import regex as re
from sncosmo import TimeSeriesSource, Model, get_bandpass
import simsurvey
import redback
import astropy.units as uu


class SimulateOpticalTransient(object):
    """
    Simulate a single optical transient from a given observation dictionary
    """
    def __init__(self, model, parameters):
        """
        :param model:
        :param parameters:
        """




def SimulateOpticalTransient(model, parameters, pointings, dEng=1.0, energy_unit=uu.Hz):
    """
    :param model:
    :type model:
    :param energy_unit:
    :type energy_unit:
    :param parameters:
    :type parameters:
    :param pointings:
    :type pointings:
    :param dEng:
    :type dEng:
    :return:
    :rtype:
    """
    energy_unit = energy_unit.to(uu.Angstrom, equivalencies=uu.spectral())
    unique_filter_list = pointings.filter.unique()
    times = pointings.times
    min_wave_list = []
    max_wave_list = []
    for filter_name in unique_filter_list:
        bp = get_bandpass(filter_name)
        min_wave_list.append(bp.wave.min())
        max_wave_list.append(bp.wave.max())
    min_wave = min(min_wave_list)
    max_wave = max(max_wave_list)
    wavelengths = np.linspace(min_wave, max_wave, num=int((max_wave - min_wave)/dAng))
    # determine wavelength/frequency range based on spectral width of filters provided
    sncosmo_wrapped_model = sncosmo_function_wrapper(model, energy_unit_scale, energy_unit_base, parameters, wavelengths)


def sncosmo_function_wrapper(
    model: string,
    energy_unit_scale: string,
    energy_unit_base: astropy.units.UnitBase,
    parameters: dict,
    wavelengths: np.ndarray,
    dense_times: np.ndarray = None,
    **kwargs,
):
    model_function = redback.model_library.all_models_dict[model]

    # Setup case structure based on transient type to determine end time
    if dense_times is None:
        transient_type = model_function.transient_type
        if transient_type == "KNe":
            max_time = 10.0
        elif transient_type == "SNe":
            max_time = 100.0
        elif transient_type == "TDE":
            max_time = 50.0
        elif transient_type == "Afterglow":
            max_time = 1000.0
        dense_times = np.linspace(0.001, max_time, num=1000)

    if energy_unit_scale == 'frequency':
        energy_delimiter = (wavelengths*u.Angstrom).to(energy_unit_base).value
    else:
        energy_delimiter = wavelengths

    redshift = parameters.pop("z")

    model_output = model_function(dense_times, redshift, parameters, f"{energy_unit_scale}"=energy_delimiter, output_format='flux_density', kwargs)
    source = TimeSeriesSource(dense_times, wavelengths, model_output)
    sncosmo_model = Model(source=source)
    return sncosmo_model


class redback_to_sncosmo(sncosmo.Source):
    def __init__():
        
