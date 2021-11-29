from bilby.core.utils.io import check_directory_exists_and_if_not_mkdir
from redback.utils import logger


SWIFT_PROMPT_BIN_SIZES = ['1s', '2ms', '8ms', '16ms', '64ms', '256ms']


def afterglow_directory_structure(grb, data_mode, instrument='BAT+XRT'):
    grb_dir = 'GRBData/GRB' + grb + '/afterglow/'
    grb_dir = grb_dir + data_mode + '/'
    check_directory_exists_and_if_not_mkdir(grb_dir)

    rawfile_path = grb_dir + 'GRB' + grb

    if instrument == 'xrt':
        rawfile = rawfile_path + '_xrt_rawSwiftData.csv'
        fullfile = rawfile_path + '_xrt_csv'
        logger.warning('You are only downloading XRT data, you may not capture the tail of the prompt emission.')
    else:
        logger.warning('You are downloading BAT and XRT data, you will need to truncate the data for some models.')
        rawfile = rawfile_path + '_rawSwiftData.csv'
        fullfile = rawfile_path + '.csv'

    return grb_dir, rawfile, fullfile


def prompt_directory_structure(grb, bin_size='2ms'):
    if bin_size not in SWIFT_PROMPT_BIN_SIZES:
        raise ValueError(f'Bin size {bin_size} not in allowed bin sizes.\n'
                         f'Use one of the following: {SWIFT_PROMPT_BIN_SIZES}')
    grb_dir = 'GRBData/GRB' + grb + '/'
    grb_dir = grb_dir + 'prompt/'
    check_directory_exists_and_if_not_mkdir(grb_dir)

    rawfile_path = grb_dir + f'{bin_size}_lc_ascii.dat'
    processed_file_path = grb_dir + f'{bin_size}_lc.csv'
    return grb_dir, rawfile_path, processed_file_path


def transient_directory_structure(transient, transient_type):
    open_transient_dir = transient_type + '/' + transient + '/'
    rawfile_path = open_transient_dir + transient + '_rawdata.csv'
    fullfile_path = open_transient_dir + transient + '_data.csv'
    return open_transient_dir, rawfile_path, fullfile_path
