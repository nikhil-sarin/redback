from bilby.core.utils.io import check_directory_exists_and_if_not_mkdir
from redback.utils import logger


SWIFT_PROMPT_BIN_SIZES = ['1s', '2ms', '8ms', '16ms', '64ms', '256ms']


def afterglow_directory_structure(grb: str, data_mode: str, instrument: str = 'BAT+XRT') -> tuple:
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
    grb_dir = f'GRBData/afterglow/{data_mode}/'
    check_directory_exists_and_if_not_mkdir(grb_dir)

    path = f'{grb_dir}{grb}'

    if instrument == 'XRT':
        rawfile = f'{path}_xrt_rawSwiftData.csv'
        fullfile = f'{path}_xrt.csv'
        logger.warning('You are only downloading XRT data, you may not capture the tail of the prompt emission.')
    else:
        logger.warning('You are downloading BAT and XRT data, you will need to truncate the data for some models.')
        rawfile = f'{path}_rawSwiftData.csv'
        fullfile = f'{path}.csv'

    return grb_dir, rawfile, fullfile


def prompt_directory_structure(grb: str, bin_size: str = '2ms') -> tuple:
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
    grb_dir = f'GRBData/prompt/flux/'
    check_directory_exists_and_if_not_mkdir(grb_dir)

    rawfile_path = f'{grb_dir}{grb}_{bin_size}_lc_ascii.dat'
    processed_file_path = f'{grb_dir}{grb}_{bin_size}_lc.csv'
    return grb_dir, rawfile_path, processed_file_path


def transient_directory_structure(transient: str, transient_type: str) -> tuple:
    """

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
    open_transient_dir = transient_type + '/' + transient + '/'
    rawfile_path = open_transient_dir + transient + '_rawdata.csv'
    fullfile_path = open_transient_dir + transient + '_data.csv'
    return open_transient_dir, rawfile_path, fullfile_path
