# from redback.get_data import directory, open_data, swift, utils
import redback.get_data.directory
import redback.get_data.open_data
import redback.get_data.swift
import redback.get_data.utils


def _get_data_functions_dict():
    return {
            ("afterglow", "swift"): get_afterglow_data_from_swift,
            ("afterglow", "swift_xrt"): get_xrt_data_from_swift,
            ("prompt", "swift"): get_prompt_data_from_swift,}
            # ("prompt", "fermi"): get_prompt_data_from_fermi,
            # ("prompt", "konus"): get_prompt_data_from_konus,
            # ("prompt", "batse"): get_prompt_data_from_batse,
            # ("kilonova", "open_data"): get_kilonova_data_from_open_transient_catalog_data,
            # ("supernova", "open_data"): get_supernova_data_from_open_transient_catalog_data,
            # ("tidal_disruption_event", "open_data"): get_tidal_disruption_event_data_from_open_transient_catalog_data}


def get_xrt_data_from_swift(grb, data_mode=None, **kwargs):
    return get_swift_data(grb=grb, transient_type='afterglow', data_mode=data_mode, instrument="XRT")


def get_afterglow_data_from_swift(grb, data_mode, **kwargs):
    return get_swift_data(grb=grb, transient_type='afterglow', data_mode=data_mode, instrument="BAT+XRT")


def get_prompt_data_from_swift(grb, bin_size="1s", **kwargs):
    return get_swift_data(grb=grb, transient_type='prompt', data_mode='prompt', instrument="BAT+XRT", bin_size=bin_size)


def get_swift_data(grb, transient_type, data_mode='flux', instrument='BAT+XRT', bin_size=None):
    getter = swift.SwiftDataGetter(
        grb=grb, transient_type=transient_type, data_mode=data_mode,
        bin_size=bin_size, instrument=instrument)
    getter.get_data()
    return getter
