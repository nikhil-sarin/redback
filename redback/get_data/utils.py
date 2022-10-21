import os
import pandas as pd
import numpy as np

import astropy.io.ascii

_dirname = os.path.dirname(__file__)


def get_trigger_number(grb: str) -> str:
    """Gets the trigger number from the GRB table.

    :param grb: Telephone number of GRB, e.g., 'GRB140903A' or '140903A' are valid inputs.
    :type grb: str
    :return: The GRB trigger number.
    :rtype: str
    """
    grb = grb.lstrip('GRB')
    grb_table = get_grb_table()
    trigger = grb_table.query('GRB == @grb')['Trigger Number']
    if len(trigger) == 0:
        raise TriggerNotFoundError(f"The trigger for {grb} does not exist in the table.")
    else:
        return trigger.values[0]


def get_grb_table() -> pd.DataFrame:
    """
    :return: The combined long and short GRB table.
    :rtype: pandas.DataFrame
    """
    short_table = os.path.join(_dirname, '../tables/SGRB_table.txt')
    long_table = os.path.join(_dirname, '../tables/LGRB_table.txt')
    sgrb = pd.read_csv(
        short_table, header=0, on_bad_lines='skip', delimiter='\t', dtype='str')
    lgrb = pd.read_csv(
        long_table, header=0, on_bad_lines='skip', delimiter='\t', dtype='str')
    return pd.concat([lgrb, sgrb], ignore_index=True)


def get_batse_trigger_from_grb(grb: str) -> int:
    """Gets the BATSE trigger from the BATSE trigger table. If the same trigger appears multiple times,
    successive alphabetical letters need to be appended to distinguish the triggers.

    :param grb: Telephone number of GRB, e.g., 'GRB910425A' or '910425A' are valid inputs. An alphabetical letter
                needs to be appended if the event is listed multiple times.
    :type grb: str
    :return: The BATSE trigger number.
    :rtype: int
    """
    grb = "GRB" + grb.lstrip("GRB")

    ALPHABET = "ABCDEFGHIJKLMNOP"
    dat = astropy.io.ascii.read(f"{_dirname}/../tables/BATSE_trigger_table.txt")
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

    index = object_labels.index(grb)
    return int(batse_triggers[index])

def convert_ztf_difference_magnitude_to_apparent_magnitude(filters, diff_mag, diff_mag_err,
                                                           status, ref_mag, ref_mag_err):
    """
    Convert ztf difference magnitudes. This code is modified from https://lasair-ztf.lsst.ac.uk/lasair/static/mag.py

    :param filter: filters name
    :param diff_mag: difference magnitude
    :param diff_mag_err: difference magnitude error
    :param status: "t" or "f" depending on whether difference is positive or negative
    :param ref_mag: reference image magnitude
    :param ref_mag_err: reference image magnitude error
    :return: apparent magnitude, apparent magnitude error
    """
    zero_point_mag_dict = {"g":26.325, "r":26.275, "i":25.660}
    zero_points = np.array([zero_point_mag_dict[filters] for _ in range(len(filters))])

    magdiff = zero_points - ref_mag
    if magdiff > 12.0:
        magdiff = 12.0
    ref_flux = 10 ** (0.4 * (magdiff))
    ref_sigflux = (ref_mag_err / 1.0857) * ref_flux

    magdiff = zero_points - diff_mag
    if magdiff > 12.0:
        magdiff = 12.0
    difference_flux = 10 ** (0.4 * (magdiff))
    difference_sigflux = (diff_mag_err / 1.0857) * difference_flux

    # add or subract difference flux based on status flag
    if status == 't':
        dc_flux = ref_flux + difference_flux
    elif status == 'f':
        dc_flux = ref_flux - difference_flux
    else:
        raise ValueError("status must be 't' or 'f'")

    dc_sigflux = np.sqrt(difference_sigflux ** 2 + ref_sigflux ** 2)

    # apparent mag and its error from fluxes
    if dc_flux > 0.0:
        dc_mag = zero_points - 2.5 * np.log10(dc_flux)
        dc_sigmag = dc_sigflux / dc_flux * 1.0857
    else:
        dc_mag = zero_points
        dc_sigmag = diff_mag_err

    return {'dc_mag': dc_mag, 'dc_sigmag': dc_sigmag}

class TriggerNotFoundError(Exception):
    """ Exceptions raised when trigger is not found."""
