# In this example, we show a few of the different ways to load transient objects in redback.
# Note this example will not run unless you have run some other examples/have data downloaded.
# It is just meant to show the interface and describe the differences.

# First, we can load data downloaded from the open access catalogue.
# Some of these methods will only work the Optical Transient class,
# or classes that inherit the Optical transient class. While others will work for any transient object.
# Please look through the documentation to see which methods work for which classes.
import redback
import numpy as np
import pandas as pd

# Let's load the data from the open access catalogue/FINK which have the same data structure.
# So the OAC class method works for both.
# Note you will need to have the data downloaded for this to work.

name = 'ZTF22abdjqlm'
supernova = redback.supernova.Supernova.from_open_access_catalogue(name=name,
                                                                   data_mode='flux')
# Here we have loaded the flux data for this transient by specifying data_mode='flux'.
# The data_mode could be changed to flux_density or magnitude for any optical transient for that data.

# We can also load data from LASAIR.
transient = 'ZTF20aamdsjv'
sn = redback.transient.Supernova.from_lasair_data(transient, use_phase_model=True,
                                                  data_mode='flux_density', active_bands=np.array(['ztfr']))

# Here we have loaded the flux density data for this transient by specifying data_mode='flux_density'.
# And set use_phase_model=True, which will load the time array as MJD values instead of a time since burst.
# We have also set active_bands to only load the ztf r band data. This will set the other bands to be inactive.

# We can also load data from our simulated data.
kn_object = redback.transient.Kilonova.from_simulated_optical_data(name='my_kilonova', data_mode='magnitude')

# Here we have loaded the magnitude data for this transient by specifying data_mode='magnitude'.
# The data for this transient was simulated using redback.

# We can also create a transient object from values in a csv file.
data = pd.read_csv('example_data/grb_afterglow.csv')
time_d = data['time'].values
flux_density = data['flux'].values
frequency = data['frequency'].values
flux_density_err = data['flux_err'].values
# set some other useful things as variables
name = '170817A'
redshift = 1e-2

afterglow = redback.transient.Afterglow(
    name=name, data_mode='flux_density', time=time_d,
    flux_density=flux_density, flux_density_err=flux_density_err, frequency=frequency)

# The latter method shows how the class be constructed generically.
# This could be done for any other combination of attributes. For example,
# if you wanted to load a transient but give bands and time_in_mjd instead of time and frequency.
# Note this is just a made up example. This transient is not real!

tde = redback.transient.TDE(name='my_tde', data_mode='flux_density', time_mjd=time_d,
                            flux_density=flux_density, flux_density_err=flux_density_err,frequency=frequency,
                            bands=np.repeat('ztfr', len(time_d)), redshift=redshift)
