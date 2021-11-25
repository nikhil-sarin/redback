import numpy as np
import os
import pandas as pd
from pathlib import Path
import re
import requests
import sqlite3
import time
import urllib
import urllib.request

import astropy.units as uu
from astropy.io import ascii, fits
from astropy.time import Time
from bilby.core.utils import check_directory_exists_and_if_not_mkdir

from redback.utils import logger, fetch_driver, check_element, calc_flux_density_from_ABmag, calc_flux_density_error
from redback.redback_errors import DataExists, WebsiteExist

dirname = os.path.dirname(__file__)

SWIFT_PROMPT_BIN_SIZES = ['1s', '2ms', '8ms', '16ms', '64ms', '256ms']

DATA_SOURCES = ["swift", "swift_xrt", "fermi", "konus", "batse", "open_data"]
TRANSIENT_TYPES = ["afterglow", "prompt", "kilonova", "supernova", "tidal_disruption_event"]


def get_data(transient_type, data_source, event_label, data_mode=None, **kwargs):
    if transient_type not in TRANSIENT_TYPES:
        raise ValueError(f"Transient type {transient_type} not allowed, "
                         f"must be one of the following: {TRANSIENT_TYPES}")
    if data_source not in DATA_SOURCES:
        raise ValueError(f"Data source {data_source} not allowed, "
                         f"must be one of the following: {DATA_SOURCES}")
    kwargs["bin_size"] = kwargs.get("bin_size", "2ms")
    func_dict = _get_data_functions_dict()
    try:
        func_dict[(transient_type.lower(), data_source.lower())](
            grb=event_label, data_mode=data_mode, **kwargs)
    except KeyError:
        raise ValueError(f"Combination of {transient_type} from {data_source} instrument not implemented or "
                         f"not available.")


def _get_data_functions_dict():
    return {("afterglow", "swift"): get_afterglow_data_from_swift,
            ("afterglow", "swift_xrt"): get_xrt_data_from_swift,
            ("prompt", "swift"): get_prompt_data_from_swift,
            ("prompt", "fermi"): get_prompt_data_from_fermi,
            ("prompt", "konus"): get_prompt_data_from_konus,
            ("prompt", "batse"): get_prompt_data_from_batse,
            ("kilonova", "open_data"): get_kilonova_data_from_open_transient_catalog_data,
            ("supernova", "open_data"): get_supernova_data_from_open_transient_catalog_data,
            ("tidal_disruption_event", "open_data"): get_tidal_disruption_event_data_from_open_transient_catalog_data}


def get_afterglow_data_from_swift(grb, data_mode='flux', **kwargs):
    rawfile, fullfile = collect_swift_data(grb, data_mode)
    sort_swift_data(rawfile, fullfile, data_mode)


def get_xrt_data_from_swift(grb, data_mode='flux', **kwargs):
    grbdir, rawfile, fullfile = afterglow_directory_structure(grb, data_mode, instrument='xrt')
    logger.info('Getting trigger number')
    trigger = get_trigger_number(grb)
    grb_website = 'https://www.swift.ac.uk/xrt_curves/00' + trigger + '/flux.qdp'
    response = requests.get(grb_website)
    if 'No Light curve available' in response.text:
        logger.warning('Problem loading the website for GRB {}. Are you sure GRB {} has Swift data?'.format(grb, grb))
        raise WebsiteExist('Problem loading the website for GRB {}'.format(grb))

    urllib.request.urlretrieve(grb_website, rawfile)
    logger.info('Congratulations, you now have raw XRT data for GRB {}'.format(grb))
    data = process_xrt_data(rawfile)
    data.to_csv(fullfile, sep=',', index=False)
    logger.info('Congratulations, you now have processed XRT data for GRB {}'.format(grb))


def get_prompt_data_from_swift(grb, bin_size='2ms', **kwargs):
    collect_swift_prompt_data(grb=grb, bin_size=bin_size)
    data = sort_swift_prompt_data(grb=grb)
    return data


def get_prompt_data_from_fermi(grb, **kwargs):
    return None


def get_prompt_data_from_konus(grb, **kwargs):
    return None


def get_prompt_data_from_batse(grb, **kwargs):
    trigger = get_batse_trigger_from_grb(grb=grb)
    trigger_filled = str(trigger).zfill(5)
    s = trigger - trigger % 200 + 1
    start = str(s).zfill(5)
    stop = str(s + 199).zfill(5)

    filename = f'tte_bfits_{trigger}.fits.gz'

    grb_dir = 'GRBData/GRB' + grb + '/'
    grb_dir = grb_dir + 'prompt/'
    Path(grb_dir).mkdir(exist_ok=True, parents=True)

    url = f"https://heasarc.gsfc.nasa.gov/FTP/compton/data/batse/trigger/{start}_{stop}/{trigger_filled}_burst/{filename}"
    file_path = f"{grb_dir}/{filename}"
    urllib.request.urlretrieve(url, file_path)

    with fits.open(file_path) as fits_data:
        data = fits_data[-1].data
        bin_left = np.array(data['TIMES'][:, 0])
        bin_right = np.array(data['TIMES'][:, 1])
        rates = np.array(data['RATES'][:, :])
        errors = np.array(data['ERRORS'][:, :])
        # counts = np.array([np.multiply(rates[:, i],
        #                                bin_right - bin_left) for i in range(4)]).T
        # count_err = np.sqrt(counts)
        # t90_st, end = bin_left[0], bin_right[-1]

    data = np.array([bin_left, bin_right, rates[:, 0], errors[:, 0], rates[:, 1], errors[:, 1],
                     rates[:, 2], errors[:, 2], rates[:, 3], errors[:, 3]]).T

    df = pd.DataFrame(data=data, columns=[
        "Time bin left [s]",
        "Time bin right [s]",
        "flux_20_50 [counts/s]",
        "flux_20_50_err [counts/s]",
        "flux_50_100 [counts/s]",
        "flux_50_100_err [counts/s]",
        "flux_100_300 [counts/s]",
        "flux_100_300_err [counts/s]",
        "flux_greater_300 [counts/s]",
        "flux_greater_300_err [counts/s]"])

    processed_file_path = f'{grb_dir}/BATSE_lc.csv'
    df.to_csv(processed_file_path, index=False)


def get_batse_trigger_from_grb(grb):
    ALPHABET = "ABCDEFGHIJKLMNOP"
    dat = ascii.read(f"{dirname}/tables/BATSE_trigger_table.txt")
    batse_triggers = list(dat['col1'])
    object_labels = list(dat['col2'])

    label_locations = dict()
    for i, label in enumerate(object_labels):
        if label in label_locations:
            label_locations[label].append(i)
        else:
            label_locations[label] = [i]

    for label, location in label_locations.items():
        if len(location) != 1:
            for i, loc in enumerate(location):
                object_labels[loc] = object_labels[loc] + ALPHABET[i]

    if grb[0].isnumeric():
        grb = 'GRB' + grb
    index = object_labels.index(grb)
    return int(batse_triggers[index])


def get_open_transient_catalog_data(transient, transient_type):
    collect_open_catalog_data(transient, transient_type)
    data = sort_open_access_data(transient, transient_type)
    return data


def get_kilonova_data_from_open_transient_catalog_data(transient):
    return get_open_transient_catalog_data(transient, transient_type="kilonova")


def get_supernova_data_from_open_transient_catalog_data(transient):
    return get_open_transient_catalog_data(transient, transient_type="supernova")


def get_tidal_disruption_event_data_from_open_transient_catalog_data(transient):
    return get_open_transient_catalog_data(transient, transient_type="tidal_disruption_event")


def get_oac_metadata():
    url = 'https://api.astrocats.space/catalog?format=CSV'
    urllib.request.urlretrieve(url, 'metadata.csv')
    logger.info('Downloaded metadata for open access catalog transients')
    return None


def get_grb_alias(transient):
    metadata = pd.read_csv('tables/OAC_metadata.csv')
    transient = metadata[metadata['event'] == transient]
    alias = transient['alias'].iloc[0]
    try:
        grb_alias = re.search('GRB (.+?),', alias).group(1)
    except AttributeError as e:
        logger.warning(e)
        logger.warning("Did not find a valid alias, returning None.")
        grb_alias = None
    return grb_alias


def sort_integrated_flux_data(rawfile, fullfile):
    keys = ["Time [s]", "Pos. time err [s]", "Neg. time err [s]", "Flux [erg cm^{-2} s^{-1}]",
            "Pos. flux err [erg cm^{-2} s^{-1}]", "Neg. flux err [erg cm^{-2} s^{-1}]", "Instrument"]

    data = {key: [] for key in keys}
    with open(rawfile) as f:
        instrument = None
        for num, line in enumerate(f.readlines()):
            if line == "!\n":
                continue
            elif "READ TERR 1 2" in line or "NO NO NO NO NO NO" in line:
                continue
            elif 'batSNR4flux' in line:
                instrument = 'batSNR4flux'
            elif 'xrtpcflux' in line:
                instrument = 'xrtpcflux'
                print(instrument)
            elif 'xrtwtflux' in line:
                instrument = 'xrtwtflux'
                print(instrument)
            else:
                line_items = line.split('\t')
                line_items.append(instrument)
                for key, item in zip(keys, line_items):
                    data[key].append(item.replace('\n', ''))

    df = pd.DataFrame(data=data)
    df.to_csv(fullfile, index=False)
    logger.info('Congratulations, you now have a nice data file: {}'.format(fullfile))


def sort_flux_density_data(rawfile, fullfile):
    data = np.loadtxt(rawfile, skiprows=2, delimiter='\t')
    df = pd.DataFrame(data=data, columns=['Time [s]', 'Time err plus [s]', 'Time err minus [s]',
                                          'Flux [mJy]', 'Flux err plus [mJy]', 'Flux err minus [mJy]'])
    df.to_csv(fullfile, index=False, sep=',')
    logger.info('Congratulations, you now have a nice data file: {}'.format(fullfile))


def sort_swift_data(rawfile, fullfile, data_mode):
    if os.path.isfile(fullfile):
        logger.warning('The processed data file already exists')
        return None

    if not os.path.isfile(rawfile):
        logger.warning('The raw data does not exist.')
        raise DataExists('Raw data is missing.')

    if data_mode == 'flux':
        sort_integrated_flux_data(rawfile, fullfile)
    if data_mode == 'flux_density':
        sort_flux_density_data(rawfile, fullfile)


def sort_swift_prompt_data(grb):
    grbdir, rawfile_path, processed_file_path = prompt_directory_structure(grb)

    if os.path.isfile(processed_file_path):
        logger.warning('The processed data file already exists')
        return pd.read_csv(processed_file_path, sep='\t')

    if not os.path.isfile(rawfile_path):
        logger.warning('The raw data does not exist.')
        raise DataExists('Raw data is missing.')

    data = np.loadtxt(rawfile_path)
    df = pd.DataFrame(data=data, columns=[
        "Time [s]", "flux_15_25 [counts/s/det]", "flux_15_25_err [counts/s/det]", "flux_25_50 [counts/s/det]",
        "flux_25_50_err [counts/s/det]", "flux_50_100 [counts/s/det]", "flux_50_100_err [counts/s/det]",
        "flux_100_350 [counts/s/det]", "flux_100_350_err [counts/s/det]", "flux_15_350 [counts/s/det]",
        "flux_15_350_err [counts/s/det]"])
    df.to_csv(processed_file_path, index=False)
    return df


def sort_open_access_data(transient, transient_type):
    directory, rawfilename, fullfilename = transient_directory_structure(transient, transient_type)
    if os.path.isfile(fullfilename):
        logger.warning('processed data already exists')
        return pd.read_csv(fullfilename, sep=',')

    if not os.path.isfile(rawfilename):
        logger.warning('The raw data does not exist.')
        raise DataExists('Raw data is missing.')
    else:
        rawdata = pd.read_csv(rawfilename, sep=',')
        if pd.isna(rawdata['system']).any():
            logger.warning("Some data points do not have system information. Assuming AB magnitude")
            rawdata['system'].fillna('AB', inplace=True)
        logger.info('Processing data for transient {}.'.format(transient))
        data = rawdata.copy()
        data = data[data['band'] != 'C']
        data = data[data['band'] != 'W']
        data = data[data['system'] == 'AB']
        logger.info('Keeping only AB magnitude data')

        data['flux_density(mjy)'] = calc_flux_density_from_ABmag(data['magnitude'].values)
        data['flux_density_error'] = calc_flux_density_error(magnitude=data['magnitude'].values,
                                                             magnitude_error=data['e_magnitude'].values,
                                                             reference_flux=3631,
                                                             magnitude_system='AB')

        metadata = directory + 'metadata.csv'
        metadata = pd.read_csv(metadata)
        metadata.replace(r'^\s+$', np.nan, regex=True)
        timeofevent = metadata['timeofmerger'].iloc[0]

        if np.isnan(timeofevent) and transient_type == 'kilonova':
            logger.warning('No timeofevent in metadata. Looking through associated GRBs')
            timeofevent = get_t0_from_grb(transient)

        if np.isnan(timeofevent) and transient_type != 'kilonova':
            logger.warning('No time of event in metadata.')
            logger.warning('Temporarily using the first data point as a start time')
            logger.warning(
                'Please run function fix_t0_of_transient before any further analysis. Or use models which sample with T0.')
            print(data['time'])
            timeofevent = data['time'].iloc[0]

        timeofevent = Time(timeofevent, format='mjd')
        tt = Time(np.asarray(data['time'], dtype=float), format='mjd')
        data['time (days)'] = (tt - timeofevent).to(uu.day)
        data.to_csv(fullfilename, sep=',', index=False)
        logger.info(f'Congratulations, you now have a nice data file: {fullfilename}')
    return data


def process_xrt_data(rawfile):
    data = np.loadtxt(rawfile, comments=['!', 'READ', 'NO'])
    time = data[:, 0]
    timepos = data[:, 1]
    timeneg = data[:, 2]
    flux = data[:, 3]
    fluxpos = data[:, 4]
    fluxneg = data[:, 5]
    data = {'time': time, 'timepos': timepos, 'timeneg': timeneg,
            'flux': flux, 'fluxpos': fluxpos, 'fluxneg': fluxneg}
    data = pd.DataFrame(data)
    processedfile = data[data['fluxpos'] != 0.]
    return processedfile


def process_flux_density_data(grb, rawfile):
    logger.info('Getting trigger number')
    trigger = get_trigger_number(grb)
    grb_website = 'http://www.swift.ac.uk/burst_analyser/00' + trigger + '/'
    logger.info('opening Swift website for GRB {}'.format(grb))

    # open the webdriver
    driver = fetch_driver()

    driver.get(grb_website)
    try:
        driver.find_element_by_xpath("//select[@name='xrtsub']/option[text()='no']").click()
        time.sleep(20)
        driver.find_element_by_id("xrt_DENSITY_makeDownload").click()
        time.sleep(20)
        grb_url = driver.current_url

        # Close the driver and all opened windows
        driver.quit()

        # scrape the data
        urllib.request.urlretrieve(grb_url, rawfile)
        logger.info(f'Congratulations, you now have raw data for GRB {grb}')
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

    # celect option for BAT bin_size
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


def collect_swift_data(grb, data_mode):
    valid_data_modes = ['flux', 'flux_density']
    if data_mode not in valid_data_modes:
        raise ValueError("Swift does not have {} data".format(data_mode))

    grbdir, rawfile, fullfile = afterglow_directory_structure(grb, data_mode)

    if os.path.isfile(rawfile):
        logger.warning('The raw data file already exists')
        return rawfile, fullfile

    logger.info('Getting trigger number')
    trigger = get_trigger_number(grb)
    grb_website = 'http://www.swift.ac.uk/burst_analyser/00' + trigger + '/'
    logger.info('opening Swift website for GRB {}'.format(grb))
    response = requests.get(grb_website)
    if 'No Light curve available' in response.text:
        logger.warning('Problem loading the website for GRB {}. Are you sure GRB {} has Swift data?'.format(grb, grb))
        raise WebsiteExist('Problem loading the website for GRB {}'.format(grb))
    else:
        if data_mode == 'flux':
            process_integrated_flux_data(grb, rawfile)
        if data_mode == 'flux_density':
            process_flux_density_data(grb, rawfile)

    return rawfile, fullfile


def collect_swift_prompt_data(grb, bin_size='2ms'):
    grbdir, rawfile, processed_file = prompt_directory_structure(
        grb=grb, bin_size=bin_size)
    if os.path.isfile(rawfile):
        logger.warning('The raw data file already exists')
        return None

    if not grb.startswith('GRB'):
        grb = 'GRB' + grb

    trigger = get_swift_trigger_from_grb(grb)
    data_file = f"{bin_size}_lc_ascii.dat"
    grb_url = f"https://swift.gsfc.nasa.gov/results/batgrbcat/{grb}/data_product/{trigger}-results/lc/{data_file}"
    try:
        urllib.request.urlretrieve(grb_url, rawfile)
        logger.info(f'Congratulations, you now have raw data for GRB {grb}')
    except Exception:
        logger.warning(f'Cannot load the website for GRB {grb}')


def collect_open_catalog_data(transient, transient_type):
    transient_dict = ['kilonova', 'supernova', 'tidal_disruption_event']

    open_transient_dir, raw_filename, full_filename = transient_directory_structure(transient, transient_type)

    check_directory_exists_and_if_not_mkdir(open_transient_dir)

    if os.path.isfile(raw_filename):
        logger.warning('The raw data file already exists')
        return None

    if transient_type not in transient_dict:
        logger.warning('Transient type does not have open access data')
        raise WebsiteExist()
    url = 'https://api.astrocats.space/' + transient + '/photometry/time+magnitude+e_magnitude+band+system?e_magnitude&band&time&format=csv'
    response = requests.get(url)

    if 'not found' in response.text:
        logger.warning(
            'Transient {} does not exist in the catalog. Are you sure you are using the right alias?'.format(transient))
        raise WebsiteExist('Webpage does not exist')
    else:
        if os.path.isfile(full_filename):
            logger.warning('The processed data file already exists')
            return None
        else:
            metadata = open_transient_dir + 'metadata.csv'
            urllib.request.urlretrieve(url, raw_filename)
            logger.info('Retrieved data for {}'.format(transient))
            metadataurl = 'https://api.astrocats.space/' + transient + '/timeofmerger+discoverdate+redshift+ra+dec+host+alias?format=CSV'
            urllib.request.urlretrieve(metadataurl, metadata)
            logger.info('Metdata for transient {} added'.format(transient))


def get_swift_trigger_from_grb(grb):
    data = ascii.read(f'{dirname}/tables/summary_general_swift_bat.txt')
    triggers = list(data['col2'])
    event_names = list(data['col1'])
    trigger = triggers[event_names.index(grb)]
    if len(trigger) == 6:
        trigger += "000"
        trigger = trigger.zfill(11)
    return trigger


def get_t0_from_grb(transient):
    grb_alias = get_grb_alias(transient)
    catalog = sqlite3.connect('tables/GRBcatalog.sqlite')
    summary_table = pd.read_sql_query("SELECT * from Summary", catalog)
    timeofevent = summary_table[summary_table['GRB_name'] == grb_alias]['mjd'].iloc[0]
    if np.isnan(timeofevent):
        logger.warning('Not found an associated GRB. Temporarily using the first data point as a start time')
        logger.warning(
            'Please run function fix_t0_of_transient before any further analysis. Or use models which sample with T0.')
    return timeofevent


def fix_t0_of_transient(timeofevent, transient, transient_type):
    """
    :param timeofevent: T0 of event in mjd
    :param transient: transient name
    :param transient_type:
    :return: None, but fixes the processed data file
    """
    directory, rawfilename, fullfilename = transient_directory_structure(transient, transient_type)
    data = pd.read_csv(fullfilename, sep=',')
    timeofevent = Time(timeofevent, format='mjd')
    tt = Time(np.asarray(data['time'], dtype=float), format='mjd')
    data['time (days)'] = (tt - timeofevent).to(uu.day)
    data.to_csv(fullfilename, sep=',', index=False)
    logger.info(f'Change input time : {fullfilename}')
    return None


def get_trigger_number(grb):
    data = get_grb_table()
    trigger = data.query('GRB == @grb')['Trigger Number']
    if len(trigger) == 0:
        trigger = '0'
    else:
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
