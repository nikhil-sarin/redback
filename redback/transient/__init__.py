import numpy as np
import os

from . import afterglow, kilonova, prompt, supernova, tde


def from_file(filename, transient_type, name, data_mode='flux'):
    """
    Instantiate the object from a private datafile.
    You do not need all the attributes just some.
    If a user has their own data for a type of transient, they can input the
    :param data_mode: flux, flux_density, luminosity
    :return: transient object with corresponding data loaded into an object
    """
    return transient_object