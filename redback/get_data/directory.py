from collections import namedtuple
import os

from bilby.core.utils.io import check_directory_exists_and_if_not_mkdir

from redback.get_data.utils import get_batse_trigger_from_grb
from redback.utils import logger

_dirname = os.path.dirname(__file__)
SWIFT_PROMPT_BIN_SIZES = ['1s', '2ms', '8ms', '16ms', '64ms', '256ms']


DirectoryStructure = namedtuple("DirectoryStructure", ['directory_path', 'raw_file_path', 'processed_file_path'])


def afterglow_directory_structure(grb: str, data_mode: str, instrument: str = 'BAT+XRT') -> DirectoryStructure:
    """
    Provides directory structure for Swift afterglow data.

    Parameters
    ----------
    grb: str
        Name of the GRB, e.g. GRB123456.
    data_mode: str
        Data mode.
    instrument: str, optional
        Must be in ['BAT+XRT', 'XRT'], default is 'BAT+XRT'

    Returns
    -------
    tuple: The directory, the raw data file name, and the processed file name.

    """
    grb = "GRB" + grb.lstrip("GRB")
    directory_path = f'GRBData/afterglow/{data_mode}/'
    check_directory_exists_and_if_not_mkdir(directory_path)

    path = f'{directory_path}{grb}'

    if instrument == 'XRT':
        raw_file_path = f'{path}_xrt_rawSwiftData.csv'
        processed_file_path = f'{path}_xrt.csv'
    else:
        raw_file_path = f'{path}_rawSwiftData.csv'
        processed_file_path = f'{path}.csv'

    return DirectoryStructure(
        directory_path=directory_path, raw_file_path=raw_file_path, processed_file_path=processed_file_path)


def swift_prompt_directory_structure(grb: str, bin_size: str = '2ms') -> DirectoryStructure:
    """
    Provides directory structure for Swift prompt data.


    Parameters
    ----------
    grb: str
        Name of the GRB, e.g. GRB123456.
    bin_size: str
        Bin size to use. Must be in `SWIFT_PROMPT_BIN_SIZES`. Default is '2ms'.

    Returns
    -------
    tuple: The directory, the raw data file name, and the processed file name.

    """
    if bin_size not in SWIFT_PROMPT_BIN_SIZES:
        raise ValueError(f'Bin size {bin_size} not in allowed bin sizes.\n'
                         f'Use one of the following: {SWIFT_PROMPT_BIN_SIZES}')
    directory_path = f'GRBData/prompt/flux/'
    check_directory_exists_and_if_not_mkdir(directory_path)

    raw_file_path = f'{directory_path}{grb}_{bin_size}_lc_ascii.dat'
    processed_file_path = f'{directory_path}{grb}_{bin_size}_lc.csv'
    return DirectoryStructure(
        directory_path=directory_path, raw_file_path=raw_file_path, processed_file_path=processed_file_path)


def batse_prompt_directory_structure(grb: str, trigger: str = None, **kwargs: object) -> DirectoryStructure:
    """
    Provides directory structure for BATSE prompt data.

    Parameters
    ----------
    grb: str
        Name of the GRB, e.g. GRB123456.
    trigger: str, optional
        The BATSE trigger number. Will be inferred from the GRB if not given.
    kwargs: object
        Add callable `get_batse_trigger_from_grb` for testing

    Returns
    -------
    tuple: The directory, the raw data file name, and the processed file name.
    """

    directory_path = f'GRBData/prompt/flux/'
    check_directory_exists_and_if_not_mkdir(directory_path)
    convert_grb_to_trigger = kwargs.get("get_batse_trigger_from_grb", get_batse_trigger_from_grb)
    if trigger is None:
        trigger = convert_grb_to_trigger(grb=grb)

    raw_file_path = f'{directory_path}tte_bfits_{trigger}.fits.gz'
    processed_file_path = f'{directory_path}{grb}_BATSE_lc.csv'
    return DirectoryStructure(
        directory_path=directory_path, raw_file_path=raw_file_path, processed_file_path=processed_file_path)


def open_access_directory_structure(transient: str, transient_type: str) -> DirectoryStructure:
    """
    Provides a general directory structure.

    Parameters
    ----------
    transient: str
        Name of the transient.
    transient_type: str
        Type of the transient.

    Returns
    -------
    tuple: The directory, the raw data file name, and the processed file name.
    """
    directory_path = f"{transient_type}/"
    check_directory_exists_and_if_not_mkdir(directory_path)
    raw_file_path = f"{directory_path}{transient}_rawdata.csv"
    processed_file_path = f"{directory_path}{transient}.csv"
    return DirectoryStructure(
        directory_path=directory_path, raw_file_path=raw_file_path, processed_file_path=processed_file_path)


def lasair_directory_structure(transient: str, transient_type: str) -> DirectoryStructure:
    """
    Provides a general directory structure.

    Parameters
    ----------
    transient: str
        Name of the transient.
    transient_type: str
        Type of the transient.

    Returns
    -------
    tuple: The directory, the raw data file name, and the processed file name.
    """
    if transient_type == "afterglow":
        directory_path = "GRBData/afterglow/"
    else:
        directory_path = f"{transient_type}/"
    check_directory_exists_and_if_not_mkdir(directory_path)
    raw_file_path = f"{directory_path}{transient}_rawdata.json"
    processed_file_path = f"{directory_path}{transient}.csv"
    return DirectoryStructure(
        directory_path=directory_path, raw_file_path=raw_file_path, processed_file_path=processed_file_path)

