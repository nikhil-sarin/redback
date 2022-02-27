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
            ("prompt", "konus"): get_prompt_data_from_konus}


def get_prompt_data_from_fermi(grb, **kwargs):
    return None


def get_prompt_data_from_konus(grb, **kwargs):
    return None









