"""
Code for reading GRB data from Swift website
"""
import os
import sys
import time
import urllib
import urllib.request
import requests
import numpy as np

import pandas as pd

from .utils import logger, fetch_driver, check_element
from .redback_errors import DataExists, WebsiteExist

from bilby.core.utils import check_directory_exists_and_if_not_mkdir

dirname = os.path.dirname(__file__)


def afterglow_directory_structure(grb, use_default_directory, data_mode):
    if use_default_directory:
        grb_dir = os.path.join(dirname, '../data/GRBData/GRB' + grb + '/')
    else:
        grb_dir = 'GRBData/GRB' + grb + '/'

    grb_dir = grb_dir + data_mode + '/'
    check_directory_exists_and_if_not_mkdir(grb_dir)

    rawfile_path = grb_dir + 'GRB' + grb + '_rawSwiftData.csv'
    fullfile_path = grb_dir + 'GRB' + grb + '.csv'
    return grb_dir, rawfile_path, fullfile_path


def prompt_directory_structure(grb, use_default_directory, binning='2ms'):
    if use_default_directory:
        grb_dir = os.path.join(dirname, '../data/GRBData/GRB' + grb + '/')
    else:
        grb_dir = 'GRBData/GRB' + grb + '/'

    grb_dir = grb_dir + 'prompt/'
    check_directory_exists_and_if_not_mkdir(grb_dir)

    rawfile_path = grb_dir + f'{binning}_lc_ascii.dat'
    processed_file_path = grb_dir + f'{binning}_lc.csv'
    return grb_dir, rawfile_path, processed_file_path


def process_fluxdensity_data(grb, rawfile):
    logger.info('Getting trigger number')
    trigger = get_trigger_number(grb)
    grb_website = 'http://www.swift.ac.uk/burst_analyser/00' + trigger + '/'
    logger.info('opening Swift website for GRB {}'.format(grb))

    # open the webdriver
    driver = fetch_driver()

    driver.get(grb_website)
    try:
        driver.find_element_by_xpath(".//*[@id='batxrt_XRTBAND_makeDownload']").click()
        time.sleep(20)

        grb_url = driver.current_url

        # Close the driver and all opened windows
        driver.quit()

        # scrape the data
        urllib.request.urlretrieve(grb_url, rawfile)
        logger.info('Congratulations, you now have raw data for GRB {}'.grb)
    except Exception:
        logger.warning('cannot load the website for GRB {}'.format(grb))


def process_integrated_flux_data(grb, rawfile):
    logger.info('Getting trigger number')
    trigger = get_trigger_number(grb)
    grb_website = 'http://www.swift.ac.uk/burst_analyser/00' + trigger + '/'
    logger.info('opening Swift website for GRB {}'.format(grb))

    # open the webdriver
    driver = fetch_driver()

    driver.get(grb_website)

    # celect option for BAT binning
    bat_binning = 'batxrtbin'
    if check_element(driver, bat_binning):
        driver.find_element_by_xpath("//select[@name='batxrtbin']/option[text()='SNR 4']").click()

    # select option for subplot
    subplot = "batxrtsub"
    if check_element(driver, subplot):
        driver.find_element_by_xpath("//select[@name='batxrtsub']/option[text()='no']").click()

    # Select option for flux density
    flux_density1 = "batxrtband1"
    flux_density0 = "batxrtband0"
    if (check_element(driver, flux_density1)) and (check_element(driver, flux_density0)):
        driver.find_element_by_xpath(".//*[@id='batxrtband1']").click()
        driver.find_element_by_xpath(".//*[@id='batxrtband0']").click()

    # Generate data file
    driver.find_element_by_xpath(".//*[@id='batxrt_XRTBAND_makeDownload']").click()
    time.sleep(20)

    grb_url = driver.current_url

    # Close the driver and all opened windows
    driver.quit()

    # scrape the data
    urllib.request.urlretrieve(grb_url, rawfile)
    logger.info('Congratulations, you now have raw data for GRB {}'.format(grb))


def sort_integrated_flux_data(rawfile, fullfile):
    xrtpcflag = 0
    xrtpc = []

    batflag = 0
    bat = []

    xrtwtflag = 0
    xrtwt = []

    try:
        with open(rawfile) as data:
            for num, line in enumerate(data):
                if line[0] == '!':
                    if line[2:13] == 'batSNR4flux':
                        xrtpcflag = 0
                        batflag = 1
                        xrtwtflag = 0
                    elif line[2:11] == 'xrtpcflux':
                        xrtpcflag = 1
                        batflag = 0
                        xrtwtflag = 0
                    elif line[2:11] == 'xrtwtflux':
                        xrtpcflag = 0
                        batflag = 0
                        xrtwtflag = 1
                    else:
                        xrtpcflag = 0
                        batflag = 0
                        xrtwtflag = 0

                if xrtpcflag == 1:
                    xrtpc.append(line)
                if batflag == 1:
                    bat.append(line)
                if xrtwtflag == 1:
                    xrtwt.append(line)

        with open(fullfile, 'w') as out:
            out.write('## BAT - batSNR4flux\n')
            for ii in range(len(bat)):
                try:
                    int(bat[ii][0])
                    out.write(bat[ii])
                except ValueError:
                    pass

            out.write('\n')
            out.write('## XRT - xrtwtflux\n')
            for ii in range(len(xrtwt)):
                try:
                    int(xrtwt[ii][0])
                    out.write(xrtwt[ii])
                except ValueError:
                    pass

            out.write('\n')
            out.write('## XRT - xrtpcflux\n')

            for ii in range(len(xrtpc)):
                try:
                    int(xrtpc[ii][0])
                    out.write(xrtpc[ii])
                except ValueError:
                    pass
    except IOError:
        try:
            logger.warning('There was an error opening the file')
            sys.exit()
        except SystemExit:
            pass
    logger.info('Congratulations, you now have a nice data file: {}'.format(fullfile))


def sort_fluxdensity_data(rawfile, fullfile):
    logger.info('Congratulations, you now have a nice data file: {}'.format(fullfile))


def collect_swift_data(grb, use_default_directory, data_mode):
    valid_data_modes = ['flux', 'flux_density']
    if data_mode not in valid_data_modes:
        raise ValueError("Swift does not have {} data".format(data_mode))

    grbdir, rawfile, fullfile = afterglow_directory_structure(grb, use_default_directory, data_mode)

    if os.path.isfile(rawfile):
        logger.warning('The raw data file already exists')
        return None

    logger.info('Getting trigger number')
    trigger = get_trigger_number(grb)
    grb_website = 'http://www.swift.ac.uk/burst_analyser/00' + trigger + '/'
    logger.info('opening Swift website for GRB {}'.format(grb))
    response = requests.get(grb_website)
    if not response.ok:
        logger.warning('Problem loading the website for GRB {}. Are you sure GRB {} has Swift data?'.format(grb, grb))
        raise WebsiteExist('Problem loading the website for GRB {}'.format(grb))
    else:
        if data_mode == 'flux':
            process_integrated_flux_data(grb, rawfile)
        if data_mode == 'flux_density':
            process_fluxdensity_data(grb, rawfile)

    return None


def sort_swift_data(grb, use_default_directory, data_mode):
    grbdir, rawfile, fullfile = afterglow_directory_structure(grb, use_default_directory, data_mode)

    if os.path.isfile(fullfile):
        logger.warning('The processed data file already exists')
        return pd.read_csv(fullfile, sep='\t')

    if not os.path.isfile(rawfile):
        logger.warning('The raw data does not exist.')
        raise DataExists('Raw data is missing.')

    if data_mode == 'flux':
        sort_integrated_flux_data(rawfile, fullfile)
    if data_mode == 'flux_density':
        sort_fluxdensity_data(rawfile, fullfile)
    return pd.read_csv(fullfile, sep='\t')


def get_afterglow_data_from_swift(grb, data_mode='flux', use_default_directory=False):
    collect_swift_data(grb, use_default_directory, data_mode)
    data = sort_swift_data(grb, use_default_directory, data_mode)
    return data


def collect_swift_prompt_data(grb, use_default_directory, binning='2ms'):
    grbdir, rawfile, processed_file = prompt_directory_structure(grb, use_default_directory)
    if os.path.isfile(rawfile):
        logger.warning('The raw data file already exists')
        return None

    if not grb.startswith('GRB'):
        grb = 'GRB' + grb

    grb_website = f"https://swift.gsfc.nasa.gov/results/batgrbcat/{grb}/data_product/"
    logger.info('opening Swift website for GRB {}'.format(grb))
    data_file = f"{binning}_lc_ascii.dat"

    # open the webdriver
    driver = fetch_driver()

    driver.get(grb_website)
    try:
        driver.find_element_by_partial_link_text("results").click()
        time.sleep(20)
        driver.find_element_by_link_text("lc/").click()
        time.sleep(20)
        grb_url = driver.current_url + data_file
        urllib.request.urlretrieve(grb_url, rawfile)

        # Close the driver and all opened windows
        driver.quit()

        logger.info(f'Congratulations, you now have raw data for GRB {grb}')
    except Exception:
        logger.warning(f'Cannot load the website for GRB {grb}')


def sort_swift_prompt_data(grb, use_default_directory):
    grbdir, rawfile_path, processed_file_path = prompt_directory_structure(grb, use_default_directory)

    if os.path.isfile(processed_file_path):
        logger.warning('The processed data file already exists')
        return pd.read_csv(processed_file_path, sep='\t')

    if not os.path.isfile(rawfile_path):
        logger.warning('The raw data does not exist.')
        raise DataExists('Raw data is missing.')

    data = np.loadtxt(rawfile_path)
    times = data[, :0]
    flux_15_25 = data[, :1]
    flux_15_25_err = data[, :2]
    flux_25_50 = data[, :3]
    flux_25_50_err = data[, :4]
    flux_50_100 = data[, :5]
    flux_50_100_err = data[, :6]
    flux_100_350 = data[, :7]
    flux_100_350_err = data[, :8]
    flux_15_350 = data[, :9]
    flux_15_350_err = data[, :10]
    df = pd.DataFrame(data=data, columns=[
        "Time [s]", "flux_15_25", "flux_15_25_err", "flux_25_50", "flux_25_50_err",
        "flux_50_100", "flux_50_100_err", "flux_100_350", "flux_100_350_err", "flux_15_350", "flux_15_350_err"])
    df.to_csv(processed_file_path, sep)
    return df



def get_prompt_data_from_swift(grb, binning='2ms', use_default_directory=True):
    collect_swift_prompt_data(grb=grb, use_default_directory=use_default_directory, binning=binning)


def get_prompt_data_from_fermi(grb):
    return None


def get_prompt_data_from_konus(grb):
    return None


def get_prompt_data_from_batse(grb):
    return None


def get_open_transient_catalog_data(transient, transient_type, use_default_directory=False):
    collect_open_catalog_data(transient, use_default_directory, transient_type)
    data = sort_open_access_data(transient, use_default_directory, transient_type)
    return data


def transient_directory_structure(transient, use_default_directory, transient_type):
    if use_default_directory:
        open_transient_dir = os.path.join(dirname, '../data/' + transient_type + 'Data/' + transient + '/')
    else:
        open_transient_dir = transient_type + '/' + transient + '/'
    rawfile_path = open_transient_dir + transient + '_rawdata.csv'
    fullfile_path = open_transient_dir + transient + '_data.csv'
    return open_transient_dir, rawfile_path, fullfile_path


def collect_open_catalog_data(transient, use_default_directory, transient_type):
    transient_dict = ['kilonova', 'supernova', 'tidal_disruption_event']

    open_transient_dir, raw_filename, full_filename = transient_directory_structure(transient, use_default_directory,
                                                                                    transient_type)

    check_directory_exists_and_if_not_mkdir(open_transient_dir)

    if os.path.isfile(raw_filename):
        logger.warning('The raw data file already exists')
        return None

    url = 'https://api.astrocats.space/' + transient + '/photometry/time+magnitude+e_magnitude+band?e_magnitude&band&time&format=csv&complete'
    response = requests.get(url)

    if transient_type not in transient_dict:
        logger.warning('Transient type does not have open access data')
        raise WebsiteExist()

    if 'not found' in response.text:
        logger.warning(
            'Transient {} does not exist in the catalog. Are you sure you are using the right alias?'.format(transient))
        raise WebsiteExist('Webpage does not exist')
    else:
        if os.path.isfile(full_filename):
            logger.warning('The processed data file already exists')
            return None
        else:
            urllib.request.urlretrieve(url, raw_filename)
            logger.info('Retrieved data for {}'.format(transient))


def sort_open_access_data(transient, use_default_directory, transient_type):
    directory, rawfilename, fullfilename = transient_directory_structure(transient, use_default_directory,
                                                                         transient_type)

    if os.path.isfile(fullfilename):
        logger.warning('processed data already exists')
        return pd.read_csv(fullfilename)

    if not os.path.isfile(rawfilename):
        logger.warning('The raw data does not exist.')
        raise DataExists('Raw data is missing.')
    else:
        rawdata = pd.read_csv(rawfilename)
        data = rawdata
        data.to_csv(fullfilename, sep=' ')
        logger.info(f'Congratulations, you now have a nice data file: {fullfilename}')

    return data


def get_trigger_number(grb):
    data = get_grb_table()
    trigger = data.query('GRB == @grb')['Trigger Number']
    trigger = trigger.values[0]
    return trigger


def get_grb_table():
    short_table = os.path.join(dirname, 'tables/SGRB_table.txt')
    sgrb = pd.read_csv(short_table, header=0,
                       error_bad_lines=False, delimiter='\t', dtype='str')
    long_table = os.path.join(dirname, 'tables/LGRB_table.txt')
    lgrb = pd.read_csv(long_table, header=0,
                       error_bad_lines=False, delimiter='\t', dtype='str')
    frames = [lgrb, sgrb]
    data = pd.concat(frames, ignore_index=True)
    return data
