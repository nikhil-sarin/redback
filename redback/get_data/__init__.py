from __future__ import annotations

import redback.get_data.directory
import redback.get_data.open_data
import redback.get_data.swift
import redback.get_data.utils
from redback.get_data.swift import SwiftDataGetter


def get_xrt_data_from_swift(grb: str, data_mode: str = None, **kwargs: dict) -> SwiftDataGetter:
    return get_swift_data(grb=grb, transient_type='afterglow', data_mode=data_mode, instrument="XRT")


def get_afterglow_data_from_swift(grb: str, data_mode: str, **kwargs: dict) -> SwiftDataGetter:
    return get_swift_data(grb=grb, transient_type='afterglow', data_mode=data_mode, instrument="BAT+XRT")


def get_prompt_data_from_swift(grb: str, bin_size: str = "1s", **kwargs: dict) -> SwiftDataGetter:
    return get_swift_data(grb=grb, transient_type='prompt', data_mode='prompt', instrument="BAT+XRT", bin_size=bin_size)


def get_swift_data(
        grb: str, transient_type: str, data_mode: str = 'flux', instrument: str = 'BAT+XRT',
        bin_size: str = None) -> SwiftDataGetter:
    getter = swift.SwiftDataGetter(
        grb=grb, transient_type=transient_type, data_mode=data_mode,
        bin_size=bin_size, instrument=instrument)
    getter.get_data()
    return getter


_functions_dict = {
            ("afterglow", "swift"): get_afterglow_data_from_swift,
            ("afterglow", "swift_xrt"): get_xrt_data_from_swift,
            ("prompt", "swift"): get_prompt_data_from_swift}
            # ("prompt", "fermi"): get_prompt_data_from_fermi,
            # ("prompt", "konus"): get_prompt_data_from_konus,
            # ("prompt", "batse"): get_prompt_data_from_batse,
            # ("kilonova", "open_data"): get_kilonova_data_from_open_transient_catalog_data,
            # ("supernova", "open_data"): get_supernova_data_from_open_transient_catalog_data,
            # ("tidal_disruption_event", "open_data"): get_tidal_disruption_event_data_from_open_transient_catalog_data}


def get_data(data_mode: str, instrument: str, transient: str, **kwargs: dict) -> object:
    _functions_dict[(transient, instrument)](data_mode=data_mode, **kwargs)
