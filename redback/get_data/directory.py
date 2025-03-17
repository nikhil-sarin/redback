from collections import namedtuple
import os

from bilby.core.utils.io import check_directory_exists_and_if_not_mkdir

from redback.get_data.utils import get_batse_trigger_from_grb

_dirname = os.path.dirname(__file__)
SWIFT_PROMPT_BIN_SIZES = ['1s', '2ms', '8ms', '16ms', '64ms', '256ms']


DirectoryStructure = namedtuple("DirectoryStructure", ['directory_path', 'raw_file_path', 'processed_file_path'])


def spectrum_directory_structure(transient: str) -> DirectoryStructure:
    """Provides directory structure for any spectrum data.

    :param transient: Name of the GRB, e.g. GRB123456.
    :type transient: str

    :return: The directory structure, with 'directory_path', 'raw_file_path', and 'processed_file_path'
    :rtype: namedtuple
    """
    directory_path = f'spectrum/'
    check_directory_exists_and_if_not_mkdir(directory_path)

    raw_file_path = f"{directory_path}{transient}_rawdata.csv"
    processed_file_path = f"{directory_path}{transient}.csv"

    return DirectoryStructure(
        directory_path=directory_path, raw_file_path=raw_file_path, processed_file_path=processed_file_path)

def afterglow_directory_structure(grb: str, data_mode: str, instrument: str = 'BAT+XRT') -> DirectoryStructure:
    """Provides directory structure for Swift afterglow data.

    :param grb: Name of the GRB, e.g. GRB123456.
    :type grb: str
    :param data_mode: Data mode.
    :type data_mode: str
    :param instrument: Must be in ['BAT+XRT', 'XRT'], default is 'BAT+XRT' (Default value = 'BAT+XRT')
    :type instrument: str, optional

    :return: The directory structure, with 'directory_path', 'raw_file_path', and 'processed_file_path'
    :rtype: namedtuple
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
    """Provides directory structure for Swift prompt data.

    :param grb: Name of the GRB, e.g. GRB123456.
    :type grb: str
    :param bin_size: Bin size to use. Must be in `SWIFT_PROMPT_BIN_SIZES`. (Default value = '2ms')
    :type bin_size: str

    :return: The directory structure, with 'directory_path', 'raw_file_path', and 'processed_file_path'
    :rtype: namedtuple
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


def batse_prompt_directory_structure(grb: str, trigger: str = None, **kwargs: None) -> DirectoryStructure:
    """Provides directory structure for BATSE prompt data.

    :param grb: Name of the GRB, e.g. GRB123456.
    :type grb: str
    :param trigger: The BATSE trigger number. Will be inferred from the GRB if not given. (Default value = None)
    :type trigger: str, optional
    :param kwargs: Add callable `get_batse_trigger_from_grb` for testing
    :type kwargs: None

    :return: The directory structure, with 'directory_path', 'raw_file_path', and 'processed_file_path'
    :rtype: namedtuple
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
    """Provides a general directory structure.

    :param transient: Name of the transient.
    :type transient: str
    :param transient_type: Type of the transient.
    :type transient_type: str

    :return: The directory structure, with 'directory_path', 'raw_file_path', and 'processed_file_path'
    :rtype: namedtuple
    """
    directory_path = f"{transient_type}/"
    check_directory_exists_and_if_not_mkdir(directory_path)
    raw_file_path = f"{directory_path}{transient}_rawdata.csv"
    processed_file_path = f"{directory_path}{transient}.csv"
    return DirectoryStructure(
        directory_path=directory_path, raw_file_path=raw_file_path, processed_file_path=processed_file_path)


def lasair_directory_structure(transient: str, transient_type: str) -> DirectoryStructure:
    """Provides a general directory structure.

    :param transient: Name of the transient.
    :type transient: str
    :param transient_type: Type of the transient.
    :type transient_type: str

    :return: The directory structure, with 'directory_path', 'raw_file_path', and 'processed_file_path'
    :rtype: namedtuple
    """
    if transient_type == "afterglow":
        directory_path = "GRBData/afterglow/"
    else:
        directory_path = f"{transient_type}/"
    check_directory_exists_and_if_not_mkdir(directory_path)
    raw_file_path = f"{directory_path}{transient}_rawdata.csv"
    processed_file_path = f"{directory_path}{transient}.csv"
    return DirectoryStructure(
        directory_path=directory_path, raw_file_path=raw_file_path, processed_file_path=processed_file_path)

def fink_directory_structure(transient: str, transient_type: str) -> DirectoryStructure:
    """Provides a general directory structure.

    :param transient: Name of the transient.
    :type transient: str
    :param transient_type: Type of the transient.
    :type transient_type: str

    :return: The directory structure, with 'directory_path', 'raw_file_path', and 'processed_file_path'
    :rtype: namedtuple
    """
    if transient_type == "afterglow":
        directory_path = "GRBData/afterglow/"
    else:
        directory_path = f"{transient_type}/"
    check_directory_exists_and_if_not_mkdir(directory_path)
    raw_file_path = f"{directory_path}{transient}_rawdata.csv"
    processed_file_path = f"{directory_path}{transient}.csv"
    return DirectoryStructure(
        directory_path=directory_path, raw_file_path=raw_file_path, processed_file_path=processed_file_path)

