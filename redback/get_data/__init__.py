from __future__ import annotations

from typing import Union
import urllib

import pandas as pd

from redback.get_data import batse, directory, fermi, getter, konus, lasair, fink, open_data, swift, utils
from redback.get_data.swift import SwiftDataGetter
from redback.get_data.open_data import OpenDataGetter
from redback.get_data.batse import BATSEDataGetter
from redback.get_data.fermi import FermiDataGetter
from redback.get_data.konus import KonusDataGetter
from redback.get_data.lasair import LasairDataGetter
from redback.get_data.fink import FinkDataGetter
from redback.utils import logger

SWIFT_PROMPT_BIN_SIZES = ['1s', '2ms', '8ms', '16ms', '64ms', '256ms']

DATA_SOURCES = ["swift", "swift_xrt", "fermi", "konus", "batse", "open_data"]
TRANSIENT_TYPES = ["afterglow", "prompt", "kilonova", "supernova", "tidal_disruption_event"]


def get_xrt_afterglow_data_from_swift(grb: str, data_mode: str = None, **kwargs: None) -> pd.DataFrame:
    """Get XRT afterglow data from Swift. Creates a directory structure and saves the data.
    Returns the data, though no further action needs to be taken by the user.

    :param grb: Telephone number of GRB, e.g., 'GRB140903A' or '140903A' are valid inputs.
    :type grb: str
    :param data_mode: Data mode must be from `redback.get_data.swift.SwiftDataGetter.VALID_DATA_MODES`.
    :type data_mode: str
    :param kwargs: Placeholder to prevent TypeErrors.

    :return: The processed data.
    :rtype: pandas.DataFrame
    """
    return get_swift_data(grb=grb, transient_type='afterglow', data_mode=data_mode, instrument="XRT")


def get_bat_xrt_afterglow_data_from_swift(grb: str, data_mode: str, **kwargs: None) -> pd.DataFrame:
    """Get BAT+XRT afterglow data from Swift. Creates a directory structure and saves the data.
    Returns the data, though no further action needs to be taken by the user.

    :param grb: Telephone number of GRB, e.g., 'GRB140903A' or '140903A' are valid inputs.
    :type grb: str
    :param data_mode: Data mode must be from `redback.get_data.swift.SwiftDataGetter.VALID_DATA_MODES`.
    :type data_mode: str
    :param kwargs: Placeholder to prevent TypeErrors.
    :type kwargs: dict
    
    :return: The processed data.
    :rtype: pandas.DataFrame
    """
    return get_swift_data(grb=grb, transient_type='afterglow', data_mode=data_mode, instrument="BAT+XRT")


def get_prompt_data_from_swift(grb: str, bin_size: str = "1s", **kwargs: None) -> pd.DataFrame:
    """Get prompt emission data from Swift. Creates a directory structure and saves the data.
    Returns the data, though no further action needs to be taken by the user.

    :param grb: Telephone number of GRB, e.g., 'GRB140903A' or '140903A' are valid inputs.
    :type grb: str
    :param bin_size: Bin size. Must be from `redback.get_data.swift.SwiftDataGetter.SWIFT_PROMPT_BIN_SIZES`. 
                     (Default value = "1s")
    :type bin_size: str, optional
    :param kwargs: Placeholder to prevent TypeErrors.
    :type kwargs: None

    :return: The processed data.
    :rtype: pandas.DataFrame
    """
    return get_swift_data(grb=grb, transient_type='prompt', data_mode='prompt', instrument="BAT+XRT", bin_size=bin_size)


def get_swift_data(
        grb: str, transient_type: str, data_mode: str = 'flux', instrument: str = 'BAT+XRT',
        bin_size: str = None, **kwargs: None) -> pd.DataFrame:
    """Catch all data getting function for Swift.  Creates a directory structure and saves the data.
    Returns the data, though no further action needs to be taken by the user.

    :param grb: Telephone number of GRB, e.g., 'GRB140903A' or '140903A' are valid inputs.
    :type grb: str
    :param transient_type: Type of the transient. Should be 'prompt' or 'afterglow'.
    :param data_mode: Data mode must be from `redback.get_data.swift.SwiftDataGetter.VALID_DATA_MODES`.
                      (Default value = 'flux')
    :type data_mode: str
    :param instrument: Instrument(s) to use. Must be from `redback.get_data.swift.SwiftDataGetter.VALID_INSTRUMENTS`.
                       (Default value = 'BAT+XRT')
    :type instrument: str
    :param bin_size: Bin size. Must be from `redback.get_data.swift.SwiftDataGetter.SWIFT_PROMPT_BIN_SIZES`.
                     (Default value = None)
    :type bin_size: str, optional
    :param kwargs: Placeholder to prevent TypeErrors.
    :type kwargs: None

    :return: The processed data.
    :rtype: pandas.DataFrame
    """
    getter = SwiftDataGetter(
        grb=grb, transient_type=transient_type, data_mode=data_mode,
        bin_size=bin_size, instrument=instrument)
    return getter.get_data()


def get_prompt_data_from_batse(grb: str, **kwargs: None) -> pd.DataFrame:
    """Get prompt emission data from BATSE. Creates a directory structure and saves the data.
    Returns the data, though no further action needs to be taken by the user.

    :param grb: Telephone number of GRB, e.g., 'GRB140903A' or '140903A' are valid inputs.
    :type grb: str
    :param kwargs: Placeholder to prevent TypeErrors.
    :type kwargs: None

    :return: The processed data.
    :rtype: pandas.DataFrame
    """
    getter = BATSEDataGetter(grb=grb)
    return getter.get_data()


def get_kilonova_data_from_open_transient_catalog_data(transient: str, **kwargs: None) -> pd.DataFrame:
    """Get kilonova data from the Open Access Catalog. Creates a directory structure and saves the data.
    Returns the data, though no further action needs to be taken by the user.

    :param transient: The name of the transient, e.g. 'at2017gfo'.
    :type transient: str
    :param kwargs: Placeholder to prevent TypeErrors.
    :type kwargs: None

    :return: The processed data.
    :rtype: pandas.DataFrame
    """
    return get_open_transient_catalog_data(transient, transient_type="kilonova")


def get_supernova_data_from_open_transient_catalog_data(transient: str, **kwargs: None) -> pd.DataFrame:
    """Get supernova data from the Open Access Catalog. Creates a directory structure and saves the data.
    Returns the data, though no further action needs to be taken by the user.

    :param transient: The name of the transient, e.g. 'SN2011kl'.
    :type transient: str
    :param kwargs: Placeholder to prevent TypeErrors.
    :type kwargs: None

    :return: The processed data.
    :rtype: pandas.DataFrame
    """
    return get_open_transient_catalog_data(transient, transient_type="supernova")


def get_tidal_disruption_event_data_from_open_transient_catalog_data(
        transient: str, **kwargs: None) -> pd.DataFrame:
    """Get TDE data from the Open Access Catalog. Creates a directory structure and saves the data.
    Returns the data, though no further action needs to be taken by the user.

    :param transient: The name of the transient, e.g. 'PS18kh'.
    :type transient: str
    :param kwargs: Placeholder to prevent TypeErrors.
    :type kwargs: None
    
    :return: The processed data.
    :rtype: pandas.DataFrame
    """
    return get_open_transient_catalog_data(transient, transient_type="tidal_disruption_event")


def get_prompt_data_from_fermi(*args: None, **kwargs: None) -> pd.DataFrame:
    """Get prompt emission data from Fermi. Creates a directory structure and saves the data.
    Returns the data, though no further action needs to be taken by the user.

    :param args: Placeholder
    :type args: None
    :param kwargs: 
    :type kwargs: None
    
    :return: The processed data.
    :rtype: pandas.DataFrame
    """
    raise NotImplementedError("This function is not yet implemented.")


def get_prompt_data_from_konus(*args: list, **kwargs: None) -> pd.DataFrame:
    """Get prompt emission data from Konus. Creates a directory structure and saves the data.
    Returns the data, though no further action needs to be taken by the user.

    :param args: Placeholder
    :type args: None
    :param kwargs: 
    :type kwargs: None
    
    :return: The processed data.
    :rtype: pandas.DataFrame
    """
    raise NotImplementedError("This function is not yet implemented.")


def get_lasair_data(
        transient: str, transient_type: str, **kwargs: None) -> pd.DataFrame:
    """Catch all data getting function for Lasair data. Creates a directory structure and saves the data.
    Returns the data, though no further action needs to be taken by the user.

    :param transient: The name of the transient, e.g. 'ZTF19aagqkrq'.
    :type transient: str
    :param transient_type: Type of the transient. Must be from `redback.get_data.lasair.LasairDataGetter.VALID_TRANSIENT_TYPES`.
    :type transient_type: str
    :param kwargs: Placeholder to prevent TypeErrors.
    :type kwargs: None

    :return: The processed data.
    :rtype: pandas.DataFrame
    """
    getter = LasairDataGetter(
        transient_type=transient_type, transient=transient)
    return getter.get_data()

def get_fink_data(
        transient: str, transient_type: str, **kwargs: None) -> pd.DataFrame:
    """Catch all data getting function for Fink data. Creates a directory structure and saves the data.
    Returns the data, though no further action needs to be taken by the user.

    :param transient: The name of the transient, e.g. 'ZTF19aagqkrq'.
    :type transient: str
    :param transient_type: Type of the transient. Must be from `redback.get_data.fink.FinkDataGetter.VALID_TRANSIENT_TYPES`.
    :type transient_type: str
    :param kwargs: Placeholder to prevent TypeErrors.
    :type kwargs: None

    :return: The processed data.
    :rtype: pandas.DataFrame
    """
    getter = FinkDataGetter(
        transient_type=transient_type, transient=transient)
    return getter.get_data()

def get_open_transient_catalog_data(
        transient: str, transient_type: str, **kwargs: None) -> pd.DataFrame:
    """Catch all data getting function for the Open Access Catalog. Creates a directory structure and saves the data.
    Returns the data, though no further action needs to be taken by the user.

    :param transient: The name of the transient, e.g. 'at2017gfo'.
    :type transient: str
    :param transient_type: Type of the transient. Must be from
                           `redback.get_data.open_data.OpenDataGetter.VALID_TRANSIENT_TYPES`.
    :type transient_type: str
    :param kwargs: Placeholder to prevent TypeErrors.
    :type kwargs: None
    
    :return: The processed data.
    :rtype: pandas.DataFrame
    """
    getter = OpenDataGetter(
        transient_type=transient_type, transient=transient)
    return getter.get_data()


def get_oac_metadata() -> None:
    """Retrieves Open Access Catalog metadata table."""
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
        transient: str, instrument: str, **kwargs: None)\
        -> pd.DataFrame:
    """Catch all data getter.

    :param transient: The name of the transient.
    :type transient: str
    :param instrument: The name of the instrument.
    :type instrument: str
    :param kwargs: Any other keyword arguments to be passed through.
    :type kwargs: None

    :return: The processed data.
    :rtype: pandas.DataFrame
    """
    return _functions_dict[(transient, instrument)](transient, **kwargs)
