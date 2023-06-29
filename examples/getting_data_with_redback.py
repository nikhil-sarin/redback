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
