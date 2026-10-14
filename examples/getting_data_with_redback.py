# In this example, we show the different methods to get data in redback.
# Redback provides a few different API's to get data and does some additional processing that is necessary for inference.
# The location where the data is fixed and format is standardised but as these are just csv files.
# One can read them in as pandas dataframes and use/modify them as they wish.
import redback
import pandas as pd
# First, let's show how to get data from FINK.

name = 'ZTF22abdjqlm'
data = redback.get_data.get_fink_data(transient=name, transient_type='supernova')

# Now LASAIR
transient = 'ZTF20aamdsjv'
data = redback.get_data.get_lasair_data(transient=transient, transient_type='supernova')

# OAC
tde = "PS18kh"
data = redback.get_data.get_tidal_disruption_event_data_from_open_transient_catalog_data(tde)

# BATSE
name = '910505'
data = redback.get_data.get_prompt_data_from_batse(grb=name)

# Swift
GRB = '070809'
data = redback.get_data.get_bat_xrt_afterglow_data_from_swift(grb=GRB, data_mode="flux")

# Otter, make sure you have astro-otter installed.
tde = 'at2019dsg'
data = redback.get_data.get_tidal_disruption_event_data_from_otter(transient=tde)

# you can also get radio and X-ray data from otter.
data = redback.get_data.get_tidal_disruption_event_data_from_otter(
 transient='at2019dsg',
 obs_type=['radio', 'xray'])

# Astro-COLIBRI: multi-survey optical photometry (ATLAS, ZTF, LSST, ASAS-SN, AAVSO).
# Needs your personal Astro-COLIBRI user id from https://astro-colibri.com/account;
# set it as the ASTROCOLIBRI_UID environment variable or pass it as uid=...
# Each lightcurve request counts against the daily API quota of your account.
kilonova = 'at2017gfo'
data = redback.get_data.get_astrocolibri_data(
    transient=kilonova, transient_type='kilonova', uid='your-user-id')

# you can select surveys, a time window, and keep non-detections as upper limits.
data = redback.get_data.get_astrocolibri_data(
    transient=kilonova, transient_type='kilonova', ztf=True, lsst=False, atlas=True,
    start='2017-08-01', end='2017-09-15', include_upper_limits=True)