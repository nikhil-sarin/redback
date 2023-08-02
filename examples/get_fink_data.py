#An example to get data from FINK and plot it
import redback

name = 'ZTF22abdjqlm'
data = redback.get_data.get_fink_data(transient=name, transient_type='supernova')

# FINK and OAC have the same data format, so we can just use the OAC class method to load this data
supernova = redback.supernova.Supernova.from_open_access_catalogue(name=name,
                                                                   data_mode='flux')
# lets now plot it and change some plotting attributes
supernova.plot_data(ylim_high=1e-12, ylim_low=1e-14)