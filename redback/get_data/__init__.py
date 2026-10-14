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
from redback.get_data.astrocolibri import AstroColibriDataGetter
from redback.utils import logger

try:
    from redback.get_data import otter
    from redback.get_data.otter import OtterDataGetter
    OTTER_AVAILABLE = otter.OTTER_INSTALLED
except ImportError:
    OTTER_AVAILABLE = False
    logger.debug("astro-otter is not installed; OTTER data access is unavailable.")

SWIFT_PROMPT_BIN_SIZES = ['1s', '2ms', '8ms', '16ms', '64ms', '256ms']

DATA_SOURCES = ["swift", "swift_xrt", "fermi", "konus", "batse", "open_data", "otter", "astrocolibri"]
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


def get_bat_xrt_afterglow_data_from_swift(
        grb: str, data_mode: str, snr: Union[int, str] = 4, force_download: bool = False, **kwargs: None) -> pd.DataFrame:
    """Get BAT+XRT afterglow data from Swift. Creates a directory structure and saves the data.
    Returns the data, though no further action needs to be taken by the user.

    :param grb: Telephone number of GRB, e.g., 'GRB140903A' or '140903A' are valid inputs.
    :type grb: str
    :param data_mode: Data mode must be from `redback.get_data.swift.SwiftDataGetter.VALID_DATA_MODES`.
    :type data_mode: str
    :param snr: BAT Burst Analyser SNR choice (e.g., 4, 5, 6, 7).
    :type snr: int or str
    :param force_download: If True, re-download data from API even if cached files exist. (Default value = False)
    :type force_download: bool, optional
    :param kwargs: Placeholder to prevent TypeErrors.
    :type kwargs: dict
    
    :return: The processed data.
    :rtype: pandas.DataFrame
    """
    return get_swift_data(
        grb=grb, transient_type='afterglow', data_mode=data_mode, instrument="BAT+XRT", 
        snr=snr, force_download=force_download)


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
        bin_size: str = None, snr: Union[int, str] = 4, force_download: bool = False, **kwargs: None) -> pd.DataFrame:
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
    :param snr: BAT Burst Analyser SNR choice (e.g., 4, 5, 6, 7).
    :type snr: int or str
    :param force_download: If True, re-download data from API even if cached files exist. (Default value = False)
    :type force_download: bool, optional
    :param kwargs: Placeholder to prevent TypeErrors.
    :type kwargs: None

    :return: The processed data.
    :rtype: pandas.DataFrame
    """
    getter = SwiftDataGetter(
        grb=grb, transient_type=transient_type, data_mode=data_mode,
        bin_size=bin_size, instrument=instrument, snr=snr, force_download=force_download)
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
    :param kwargs: Placeholder to prevent TypeErrors.
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
    :param kwargs: Placeholder to prevent TypeErrors.
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
        transient: str, transient_type: str, source: str = 'ztf', **kwargs: None) -> pd.DataFrame:
    """Catch all data getting function for Fink data. Creates a directory structure and saves the data.
    Returns the data, though no further action needs to be taken by the user.

    :param transient: The name of the transient, e.g. 'ZTF19aagqkrq' for ZTF or '170019717277810735' for LSST.
    :type transient: str
    :param transient_type: Type of the transient. Must be from `redback.get_data.fink.FinkDataGetter.VALID_TRANSIENT_TYPES`.
    :type transient_type: str
    :param source: The source of the data. Must be either 'ztf' or 'lsst'. Default is 'ztf'.
    :type source: str
    :param kwargs: Placeholder to prevent TypeErrors.
    :type kwargs: None

    :return: The processed data.
    :rtype: pandas.DataFrame
    """
    getter = FinkDataGetter(
        transient_type=transient_type, transient=transient, source=source)
    return getter.get_data()


def get_astrocolibri_data(
        transient: str, transient_type: str, uid: str = None,
        start: str = None, end: str = None,
        ztf: bool = True, lsst: bool = True, atlas: bool = True,
        asas_sn: bool = True, aavso: bool = True, aavso_all_filters: bool = False,
        rapas: bool = False, include_upper_limits: bool = False, **kwargs: None) -> pd.DataFrame:
    """Catch all data getting function for Astro-COLIBRI data. Creates a directory structure and
    saves the data. Returns the data, though no further action needs to be taken by the user.

    Downloads multi-survey optical photometry (ATLAS, ZTF, LSST, ASAS-SN and AAVSO)
    for any transient known to Astro-COLIBRI. Lightcurve requests count against
    the daily Astro-COLIBRI API quota of the account; the user id is required and
    can be obtained from https://astro-colibri.com/account. It can be passed as
    `uid` or set through the ASTROCOLIBRI_UID environment variable.

    :param transient: The name of the transient, e.g. 'AT2017gfo' or 'SN2026ejy'.
    :type transient: str
    :param transient_type: Type of the transient. Must be from
                           `redback.get_data.astrocolibri.AstroColibriDataGetter.VALID_TRANSIENT_TYPES`.
    :type transient_type: str
    :param uid: Astro-COLIBRI user id. If None, read from the ASTROCOLIBRI_UID environment variable.
    :type uid: str, optional
    :param start: Start of the requested time window, e.g. '2024-05-02'. Defaults to 5 days before the event.
    :type start: str, optional
    :param end: End of the requested time window, e.g. '2024-06-06'. Defaults to 30 days after the event.
    :type end: str, optional
    :param ztf: Whether ZTF data should be requested. (Default value = True)
    :type ztf: bool, optional
    :param lsst: Whether LSST data should be requested. (Default value = True)
    :type lsst: bool, optional
    :param atlas: Whether ATLAS data should be requested. (Default value = True)
    :type atlas: bool, optional
    :param asas_sn: Whether ASAS-SN data should be requested. (Default value = True)
    :type asas_sn: bool, optional
    :param aavso: Whether AAVSO data should be requested. (Default value = True)
    :type aavso: bool, optional
    :param aavso_all_filters: Whether to include all other AAVSO filters, including visual estimates.
                              (Default value = False)
    :type aavso_all_filters: bool, optional
    :param rapas: Whether RAPAS data should be requested (members only). (Default value = False)
    :type rapas: bool, optional
    :param include_upper_limits: If True, non-detections are kept in the processed data with
                                 `detected=0` and the limiting magnitude in `limiting_magnitude`.
                                 If False (default), non-detections are dropped.
    :type include_upper_limits: bool, optional
    :param kwargs: Placeholder to prevent TypeErrors.
    :type kwargs: None

    :return: The processed data.
    :rtype: pandas.DataFrame
    """
    getter = AstroColibriDataGetter(
        transient=transient, transient_type=transient_type, uid=uid,
        start=start, end=end, ztf=ztf, lsst=lsst, atlas=atlas, asas_sn=asas_sn, aavso=aavso,
        aavso_all_filters=aavso_all_filters, rapas=rapas, include_upper_limits=include_upper_limits)
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


def get_kilonova_data_from_otter(transient: str, obs_type: str = 'uvoir', **kwargs: None) -> pd.DataFrame:
    """Get kilonova data from OTTER. Creates a directory structure and saves the data.
    Returns the data, though no further action needs to be taken by the user.

    :param transient: The name of the transient, e.g. '19dsg' or 'AT2017gfo'.
    :type transient: str
    :param obs_type: Observation type: 'uvoir', 'radio', 'xray', or list of these. Default is 'uvoir'.
    :type obs_type: str or list
    :param kwargs: Placeholder to prevent TypeErrors.
    :type kwargs: None

    :return: The processed data.
    :rtype: pandas.DataFrame
    """
    if not OTTER_AVAILABLE:
        raise ImportError(
            "OTTER data access requires astro-otter; install redback[data].")
    getter = OtterDataGetter(transient=transient, transient_type='kilonova', obs_type=obs_type)
    return getter.get_data()


def get_supernova_data_from_otter(transient: str, obs_type: str = 'uvoir', **kwargs: None) -> pd.DataFrame:
    """Get supernova data from OTTER. Creates a directory structure and saves the data.
    Returns the data, though no further action needs to be taken by the user.

    :param transient: The name of the transient, e.g. 'SN2011fe'.
    :type transient: str
    :param obs_type: Observation type: 'uvoir', 'radio', 'xray', or list of these. Default is 'uvoir'.
    :type obs_type: str or list
    :param kwargs: Placeholder to prevent TypeErrors.
    :type kwargs: None

    :return: The processed data.
    :rtype: pandas.DataFrame
    """
    if not OTTER_AVAILABLE:
        raise ImportError(
            "OTTER data access requires astro-otter; install redback[data].")
    getter = OtterDataGetter(transient=transient, transient_type='supernova', obs_type=obs_type)
    return getter.get_data()


def get_tidal_disruption_event_data_from_otter(transient: str, obs_type: str = 'uvoir', **kwargs: None) -> pd.DataFrame:
    """Get TDE data from OTTER. Creates a directory structure and saves the data.
    Returns the data, though no further action needs to be taken by the user.

    :param transient: The name of the transient, e.g. 'ASASSN-14li'.
    :type transient: str
    :param obs_type: Observation type: 'uvoir', 'radio', 'xray', or list of these. Default is 'uvoir'.
    :type obs_type: str or list
    :param kwargs: Placeholder to prevent TypeErrors.
    :type kwargs: None
    
    :return: The processed data.
    :rtype: pandas.DataFrame
    """
    if not OTTER_AVAILABLE:
        raise ImportError(
            "OTTER data access requires astro-otter; install redback[data].")
    getter = OtterDataGetter(transient=transient, transient_type='tidal_disruption_event', obs_type=obs_type)
    return getter.get_data()


_functions_dict = {
    ("afterglow", "swift"): get_bat_xrt_afterglow_data_from_swift,
    ("afterglow", "swift_xrt"): get_xrt_afterglow_data_from_swift,
    ("prompt", "swift"): get_prompt_data_from_swift,
    ("prompt", "fermi"): get_prompt_data_from_fermi,
    ("prompt", "konus"): get_prompt_data_from_konus,
    ("prompt", "batse"): get_prompt_data_from_batse,
    ("kilonova", "open_data"): get_kilonova_data_from_open_transient_catalog_data,
    ("supernova", "open_data"): get_supernova_data_from_open_transient_catalog_data,
    ("tidal_disruption_event", "open_data"): get_tidal_disruption_event_data_from_open_transient_catalog_data,
    ("kilonova", "otter"): get_kilonova_data_from_otter,
    ("supernova", "otter"): get_supernova_data_from_otter,
    ("tidal_disruption_event", "otter"): get_tidal_disruption_event_data_from_otter,
    ("afterglow", "astrocolibri"): get_astrocolibri_data,
    ("kilonova", "astrocolibri"): get_astrocolibri_data,
    ("supernova", "astrocolibri"): get_astrocolibri_data,
    ("tidal_disruption_event", "astrocolibri"): get_astrocolibri_data,
    ("unknown", "astrocolibri"): get_astrocolibri_data,
}


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
