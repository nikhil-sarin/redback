from __future__ import annotations

import urllib

import redback.get_data.directory
import redback.get_data.open_data
import redback.get_data.swift
import redback.get_data.utils
from redback.get_data.swift import SwiftDataGetter
from redback.get_data.open_data import OpenDataGetter
from redback.get_data.batse import BATSEDataGetter
from redback.get_data.fermi import FermiDataGetter
from redback.get_data.konus import KonusDataGetter
from redback.utils import logger


SWIFT_PROMPT_BIN_SIZES = ['1s', '2ms', '8ms', '16ms', '64ms', '256ms']

DATA_SOURCES = ["swift", "swift_xrt", "fermi", "konus", "batse", "open_data"]
TRANSIENT_TYPES = ["afterglow", "prompt", "kilonova", "supernova", "tidal_disruption_event"]


def get_xrt_data_from_swift(grb: str, data_mode: str = None, **kwargs: dict) -> SwiftDataGetter:
    return get_swift_data(grb=grb, transient_type='afterglow', data_mode=data_mode, instrument="XRT")


def get_afterglow_data_from_swift(grb: str, data_mode: str, **kwargs: dict) -> SwiftDataGetter:
    return get_swift_data(grb=grb, transient_type='afterglow', data_mode=data_mode, instrument="BAT+XRT")


def get_prompt_data_from_swift(grb: str, bin_size: str = "1s", **kwargs: dict) -> SwiftDataGetter:
    return get_swift_data(grb=grb, transient_type='prompt', data_mode='prompt', instrument="BAT+XRT", bin_size=bin_size)


def get_prompt_data_from_batse(grb: str, **kwargs) -> BATSEDataGetter:
    getter = BATSEDataGetter(grb=grb)
    getter.get_data()
    return getter


def get_kilonova_data_from_open_transient_catalog_data(transient: str, **kwargs: dict) -> OpenDataGetter:
    return get_open_transient_catalog_data(transient, transient_type="kilonova")


def get_supernova_data_from_open_transient_catalog_data(transient: str, **kwargs: dict) -> OpenDataGetter:
    return get_open_transient_catalog_data(transient, transient_type="supernova")


def get_tidal_disruption_event_data_from_open_transient_catalog_data(transient: str, **kwargs: dict) -> OpenDataGetter:
    return get_open_transient_catalog_data(transient, transient_type="tidal_disruption_event")


def get_swift_data(
        grb: str, transient_type: str, data_mode: str = 'flux', instrument: str = 'BAT+XRT',
        bin_size: str = None) -> SwiftDataGetter:
    getter = SwiftDataGetter(
        grb=grb, transient_type=transient_type, data_mode=data_mode,
        bin_size=bin_size, instrument=instrument)
    getter.get_data()
    return getter


def get_prompt_data_from_fermi(*args: list, **kwargs: dict) -> FermiDataGetter:
    raise NotImplementedError()


def get_prompt_data_from_konus(*args: list, **kwargs: dict) -> KonusDataGetter:
    raise NotImplementedError()


def get_open_transient_catalog_data(
        transient: str, transient_type: str) -> OpenDataGetter:
    getter = OpenDataGetter(
        transient_type=transient_type, transient=transient)
    getter.get_data()
    return getter


def get_oac_metadata() -> None:
    url = 'https://api.astrocats.space/catalog?format=CSV'
    urllib.request.urlretrieve(url, 'metadata.csv')
    logger.info('Downloaded metadata for open access catalog transients')


_functions_dict = {
            ("afterglow", "swift"): get_afterglow_data_from_swift,
            ("afterglow", "swift_xrt"): get_xrt_data_from_swift,
            ("prompt", "swift"): get_prompt_data_from_swift,
            ("prompt", "fermi"): get_prompt_data_from_fermi,
            ("prompt", "konus"): get_prompt_data_from_konus,
            ("prompt", "batse"): get_prompt_data_from_batse,
            ("kilonova", "open_data"): get_kilonova_data_from_open_transient_catalog_data,
            ("supernova", "open_data"): get_supernova_data_from_open_transient_catalog_data,
            ("tidal_disruption_event", "open_data"): get_tidal_disruption_event_data_from_open_transient_catalog_data}


def get_data(transient: str, instrument: str, **kwargs: dict) -> object:
    return _functions_dict[(transient, instrument)](transient, **kwargs)
