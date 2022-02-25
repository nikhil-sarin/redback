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
import redback

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
    return {
            ("prompt", "fermi"): get_prompt_data_from_fermi,
            ("prompt", "konus"): get_prompt_data_from_konus,
            ("prompt", "batse"): get_prompt_data_from_batse}


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




