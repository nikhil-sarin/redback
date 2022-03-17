from __future__ import annotations

from typing import Union
import urllib

import pandas as pd

from redback.get_data import batse, directory, fermi, konus, lasair, open_data, swift, utils
from redback.get_data.swift import SwiftDataGetter
from redback.get_data.open_data import OpenDataGetter
from redback.get_data.batse import BATSEDataGetter
from redback.get_data.fermi import FermiDataGetter
from redback.get_data.konus import KonusDataGetter
from redback.get_data.lasair import LasairDataGetter
from redback.utils import logger

SWIFT_PROMPT_BIN_SIZES = ['1s', '2ms', '8ms', '16ms', '64ms', '256ms']

DATA_SOURCES = ["swift", "swift_xrt", "fermi", "konus", "batse", "open_data"]
TRANSIENT_TYPES = ["afterglow", "prompt", "kilonova", "supernova", "tidal_disruption_event"]


def get_xrt_afterglow_data_from_swift(grb: str, data_mode: str = None, **kwargs: dict) -> pd.DataFrame:
    """
    Get XRT afterglow data from Swift. Creates a directory structure and saves the data.
    Returns the data, though no further action needs to be taken by the user.

    Parameters
    ----------
    grb: str
        Telephone number of GRB, e.g., 'GRB140903A' or '140903A' are valid inputs.
    data_mode: str
        Data mode must be from `redback.get_data.swift.SwiftDataGetter.VALID_DATA_MODES`.
    kwargs: dict
        Placeholder to prevent TypeErrors.

    Returns
    -------
    pandas.DataFrame: The processed data.

    """
    return get_swift_data(grb=grb, transient_type='afterglow', data_mode=data_mode, instrument="XRT")


def get_bat_xrt_afterglow_data_from_swift(grb: str, data_mode: str, **kwargs: dict) -> pd.DataFrame:
    """
    Get BAT+XRT afterglow data from Swift. Creates a directory structure and saves the data.
    Returns the data, though no further action needs to be taken by the user.

    Parameters
    ----------
    grb: str
        Telephone number of GRB, e.g., 'GRB140903A' or '140903A' are valid inputs.
    data_mode: str
        Data mode must be from `redback.get_data.swift.SwiftDataGetter.VALID_DATA_MODES`.
    kwargs: dict
        Placeholder to prevent TypeErrors.

    Returns
    -------
    pandas.DataFrame: The processed data.

    """
    return get_swift_data(grb=grb, transient_type='afterglow', data_mode=data_mode, instrument="BAT+XRT")


def get_prompt_data_from_swift(grb: str, bin_size: str = "1s", **kwargs: dict) -> pd.DataFrame:
    """
    Get prompt emission data from Swift. Creates a directory structure and saves the data.
    Returns the data, though no further action needs to be taken by the user.

    Parameters
    ----------
    grb: str
        Telephone number of GRB, e.g., 'GRB140903A' or '140903A' are valid inputs.
    bin_size: str, optional
        Bin size. Must be from `redback.get_data.swift.SwiftDataGetter.SWIFT_PROMPT_BIN_SIZES`.
    kwargs: dict
        Placeholder to prevent TypeErrors.

    Returns
    -------
    pandas.DataFrame: The processed data.

    """
    return get_swift_data(grb=grb, transient_type='prompt', data_mode='prompt', instrument="BAT+XRT", bin_size=bin_size)


def get_swift_data(
        grb: str, transient_type: str, data_mode: str = 'flux', instrument: str = 'BAT+XRT',
        bin_size: str = None, **kwargs: dict) -> pd.DataFrame:
    """
    Catch all data getting function for Swift.  Creates a directory structure and saves the data.
    Returns the data, though no further action needs to be taken by the user.

    Parameters
    ----------
    grb: str
        Telephone number of GRB, e.g., 'GRB140903A' or '140903A' are valid inputs.
    transient_type:
        Type of the transient. Should be 'prompt' or 'afterglow'.
    data_mode: str
        Data mode must be from `redback.get_data.swift.SwiftDataGetter.VALID_DATA_MODES`.
    instrument: str
        Instrument(s) to use. Must be from `redback.get_data.swift.SwiftDataGetter.VALID_INSTRUMENTS`.
    bin_size: str, optional
        Bin size. Must be from `redback.get_data.swift.SwiftDataGetter.SWIFT_PROMPT_BIN_SIZES`.
    kwargs: dict
        Placeholder to prevent TypeErrors.

    Returns
    -------
    pandas.DataFrame: The processed data.

    """
    getter = SwiftDataGetter(
        grb=grb, transient_type=transient_type, data_mode=data_mode,
        bin_size=bin_size, instrument=instrument)
    return getter.get_data()


def get_prompt_data_from_batse(grb: str, **kwargs) -> pd.DataFrame:
    """
    Get prompt emission data from BATSE. Creates a directory structure and saves the data.
    Returns the data, though no further action needs to be taken by the user.

    Parameters
    ----------
    grb: str
        Telephone number of GRB, e.g., 'GRB140903A' or '140903A' are valid inputs.
    kwargs: dict
        Placeholder to prevent TypeErrors.

    Returns
    -------
    pandas.DataFrame: The processed data.

    """
    getter = BATSEDataGetter(grb=grb)
    return getter.get_data()


def get_kilonova_data_from_open_transient_catalog_data(transient: str, **kwargs: dict) -> pd.DataFrame:
    """
    Get kilonova data from the Open Access Catalog. Creates a directory structure and saves the data.
    Returns the data, though no further action needs to be taken by the user.

    Parameters
    ----------
    transient: str
        The name of the transient, e.g. 'at2017gfo'.
    kwargs: dict
        Placeholder to prevent TypeErrors.

    Returns
    -------
    pandas.DataFrame: The processed data.

    """
    return get_open_transient_catalog_data(transient, transient_type="kilonova")


def get_supernova_data_from_open_transient_catalog_data(transient: str, **kwargs: dict) -> pd.DataFrame:
    """
    Get supernova data from the Open Access Catalog. Creates a directory structure and saves the data.
    Returns the data, though no further action needs to be taken by the user.

    Parameters
    ----------
    transient: str
        The name of the transient, e.g. 'SN2011kl'.
    kwargs: dict
        Placeholder to prevent TypeErrors.

    Returns
    -------
    pandas.DataFrame: The processed data.

    """
    return get_open_transient_catalog_data(transient, transient_type="supernova")


def get_tidal_disruption_event_data_from_open_transient_catalog_data(
        transient: str, **kwargs: dict) -> pd.DataFrame:
    """
    Get TDE data from the Open Access Catalog. Creates a directory structure and saves the data.
    Returns the data, though no further action needs to be taken by the user.

    Parameters
    ----------
    transient: str
        The name of the transient, e.g. 'PS18kh'.
    kwargs: dict
        Placeholder to prevent TypeErrors.

    Returns
    -------
    pandas.DataFrame: The processed data.

    """
    return get_open_transient_catalog_data(transient, transient_type="tidal_disruption_event")


def get_prompt_data_from_fermi(*args: list, **kwargs: dict) -> pd.DataFrame:
    """
    Get prompt emission data from Fermi. Creates a directory structure and saves the data.
    Returns the data, though no further action needs to be taken by the user.

    Parameters
    ----------
    args: list
        Placeholder
    kwargs: dict
        Placeholder
    Raises
    -------
    NotImplementedError: Functionality needs yet to be implemented.

    """
    raise NotImplementedError("This function is not yet implemented.")


def get_prompt_data_from_konus(*args: list, **kwargs: dict) -> pd.DataFrame:
    """
    Get prompt emission data from Konus. Creates a directory structure and saves the data.
    Returns the data, though no further action needs to be taken by the user.

    Parameters
    ----------
    args: list
        Placeholder
    kwargs: dict
        Placeholder
    Raises
    -------
    NotImplementedError: Functionality needs yet to be implemented.

    """
    raise NotImplementedError("This function is not yet implemented.")


def get_lasair_data(
        transient: str, transient_type: str, **kwargs) -> pd.DataFrame:
    """
    Catch all data getting function for Lasair data. Creates a directory structure and saves the data.
    Returns the data, though no further action needs to be taken by the user.

    Parameters
    ----------
    transient: str
        The name of the transient, e.g. 'ZTF19aagqkrq'.
    transient_type: str
        Type of the transient. Must be from `redback.get_data.lasair.LasairDataGetter.VALID_TRANSIENT_TYPES`.
    kwargs: dict
        Placeholder to prevent TypeErrors.

    Returns
    -------
    pandas.DataFrame: The processed data.

    """
    getter = LasairDataGetter(
        transient_type=transient_type, transient=transient)
    return getter.get_data()

def get_open_transient_catalog_data(
        transient: str, transient_type: str, **kwargs) -> pd.DataFrame:
    """
    Catch all data getting function for the Open Access Catalog. Creates a directory structure and saves the data.
    Returns the data, though no further action needs to be taken by the user.

    Parameters
    ----------
    transient: str
        The name of the transient, e.g. 'at2017gfo'.
    transient_type: str
        Type of the transient. Must be from `redback.get_data.open_data.OpenDataGetter.VALID_TRANSIENT_TYPES`.
    kwargs: dict
        Placeholder to prevent TypeErrors.

    Returns
    -------
    pandas.DataFrame: The processed data.

    """
    getter = OpenDataGetter(
        transient_type=transient_type, transient=transient)
    return getter.get_data()


def get_oac_metadata() -> None:
    """
    Retrieves Open Access Catalog metadata table.
    """
    url = 'https://api.astrocats.space/catalog?format=CSV'
    urllib.request.urlretrieve(url, 'metadata.csv')
    logger.info('Downloaded metadata for open access catalog transients.')


_functions_dict = {
    ("afterglow", "swift"): get_bat_xrt_afterglow_data_from_swift,
    ("afterglow", "swift_xrt"): get_xrt_afterglow_data_from_swift,
    ("prompt", "swift"): get_prompt_data_from_swift,
    ("prompt", "fermi"): get_prompt_data_from_fermi,
    ("prompt", "konus"): get_prompt_data_from_konus,
    ("prompt", "batse"): get_prompt_data_from_batse,
    ("kilonova", "open_data"): get_kilonova_data_from_open_transient_catalog_data,
    ("supernova", "open_data"): get_supernova_data_from_open_transient_catalog_data,
    ("tidal_disruption_event", "open_data"): get_tidal_disruption_event_data_from_open_transient_catalog_data}


def get_data(
        transient: str, instrument: str, **kwargs: dict)\
        -> pd.DataFrame:
    """
    Catch all data getter.

    Parameters
    ----------
    transient: str
        The name of the transient.
    instrument: str
        The name of the instrument.
    kwargs: dict
        Any other keyword arguments to be passed through.

    Returns
    -------
    pandas.DataFrame: The processed data.

    """
    return _functions_dict[(transient, instrument)](transient, **kwargs)
