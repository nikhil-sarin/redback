import numpy as np
from typing import Union
import regex as re
from sncosmo import TimeSeriesSource, Model, get_bandpass
import simsurvey
import redback
import pandas as pd
from collections import namedtuple
import astropy.units as uu


class SimulateOpticalTransient(object):
    """
    Simulate a single optical transient from a given observation dictionary
    """
    def __init__(self, model, pointings_database='rubin_baseline_nexp1_v1_7_10yrs.tar.gz', **kwargs):
        """
        :param model: str of redback model
        :param parameters:s
        :param start_mjd:
        """
        if isinstance(model, str):
            self.model = redback.model_library.all_models_dict[model]
        elif callable(model):
            self.model = model
        else:
            raise ValueError("The user needs to specify model as either a string or function.")

        self.pointing_database = pointings_database

    @property
    def _initialise_from_pointings(self):
        pointing_database = self.pointing_database
        df = pd.read_csv(self.pointing_database, compression='gzip')
        min_dec = np.min(df['_dec'])
        max_dec = np.max(df['_dec'])
        dec_dist = (np.arccos(2* np.random.uniform(low=(1 - np.sin(min_dec)) / 2,
                                                   high=(1 - np.sin(max_dec)) / 2,
                                                   size=1)- 1) - np.pi / 2)
        start_mjd = np.min(df['expMJD'])
        end_mjd = np.max(df['expMJD'])
        return min_dec, max_dec, dec_dist, start_mjd, end_mjd

    def _make_sncosmo_wrapper(self):
        pass

    def _sncosmo_function_wrapper(self, model, energy_unit_scale, energy_unit_base, parameters, wavelengths, dense_times, **kwargs):
        """
        Function to wrap redback models into sncosmo model format for added functionality.
        """
        # Setup case structure based on transient type to determine end time
        if dense_times is None:
            transient_type = model_function.transient_type
            if transient_type == "KNe":
                max_time = 10.0
            elif transient_type == "SNe":
                max_time = 100.0
            elif transient_type == "TDE":
                max_time = 50.0
            elif transient_type == "Afterglow":
                max_time = 1000.0
            dense_times = np.linspace(0.001, max_time, num=1000)

        if energy_unit_scale == 'frequency':
            energy_delimiter = (wavelengths * u.Angstrom).to(energy_unit_base).value
        else:
            energy_delimiter = wavelengths

        # may want to also have dense wavelengths so that bandpass fluxes are estimated correctly
        redshift = parameters.pop("z")

        model_output = model_function(dense_times, redshift, parameters, f"{energy_unit_scale}"=energy_delimiter, output_format='flux_density', kwargs)
        source = TimeSeriesSource(dense_times, wavelengths, model_output)
        sncosmo_model = Model(source=source)
        return sncosmo_model

    def _make_observations(self):
        pass

    @classmethod
    def simulate_transient(self, parameters, buffer_days=100, survey='Rubin', **kwargs):
        if isinstance(survey, str):
            # load the correspondiong file
        if isinstance(survey, pd.DataFrame):
            # initialise from the datafile.
        self.RA = parameters.get("RA", np.random.uniform(0, 2*np.pi))
        self.DEC = parameters.get("DEC", np.random.uniform(-np.pi/2, np.pi/2))
        self.mjd = parameters.get("mjd", np.random.uniform((self.start_mjd - buffer_days), (self.end_mjd)))
        #
        #make observations
        #save the file to disk
        pass

    @classmethod
    def simulate_transient_population(self, dataframe, outdir):
        """
        Simulate a population of transients from a dataframe of parameters.

        :param dataframe: pandas dataframe of parameters to make simulated observations for
        :param outdir: output directory where the simulated observations will be saved
        :return:
        """
        pass

def make_pointing_table_from_something():
    """
    Makes a pandas dataframe of pointings from specified settings.
    :return:
    """
    pass

class SimulateFullSurvey(object):
    """
    Do SimSurvey or SNANA
    """
    def __init__(self):
        pass

    # def SimulateOpticalTransient(self, dEng=1.0, energy_unit=uu.Hz):
    #     """
    #     :param model:
    #     :type model:
    #     :param energy_unit:
    #     :type energy_unit:
    #     :param parameters:
    #     :type parameters:
    #     :param pointings:
    #     :type pointings:
    #     :param dEng:
    #     :type dEng:
    #     :return:
    #     :rtype:
    #     """
    #     try:
    #         self.pointings
    #     except AttributeError:
    #         get_pointings()
    #     energy_unit = energy_unit.to(uu.Angstrom, equivalencies=uu.spectral())
    #     unique_filter_list = self.pointings.filter.unique()
    #     times = self.pointings.times
    #     min_wave_list = []
    #     max_wave_list = []
    #     for filter_name in unique_filter_list:
    #         bp = get_bandpass(filter_name)
    #         min_wave_list.append(bp.wave.min())
    #         max_wave_list.append(bp.wave.max())
    #     min_wave = min(min_wave_list)
    #     max_wave = max(max_wave_list)
    #     wavelengths = np.linspace(min_wave, max_wave, num=int((max_wave - min_wave)/dAng))
    #     # determine wavelength/frequency range based on spectral width of filters provided
    #     self.sncosmo_model = sncosmo_function_wrapper(model, energy_unit_scale, energy_unit_base, parameters, wavelengths)
    #
    #
    # def get_pointings(self, instrument=None, pointings=None, scheduler=None):
    #     """
    #     :param instrument: The name of the instrument that pointings
    #     :param pointings:
    #     :param scheduler:
    #     """
    #     pointings = pd.DataFrame(columns=['mjd', ])
    #
    #     if pointings and scheduler:
    #         pointings = pointings_from_scheduler(pointings, scheduler)
    #     elif pointings:
    #         pass # could verify provided pointings format here.
    #     else:
    #         if instrument == 'ZTF':
    #             pointings = create_ztf_pointings() # create list of pointings based on ZTF
    #         elif instrument == 'Rubin':
    #             pointings = create_rubin_pointings() # create list of pointings based on Rubin
    #         elif instrument == 'Roman':
    #             pointings = create_roman_pointings() # create list of pointings based on Roman
    #         else:
    #             print('The requested observatory does not have an implemented set of pointings.')
    #     return
    #
    # def pointings_from_scheduler(self, pointings, scheduler):
    #     """
    #
    #     """
    #     # placeholder to set instrument pointings on scheduler call OpSimSummary etc.
    #
    #     return
    #
    #
    # def SimulateMockObservations(self, bands, times, zpsys='ab'):
    #     """
    #     Function to generate mock observations from the given model and pointings, but can be reexecuted for different parameters
    #     """
    #
    #     bandfluxes = self.sncosmo_model.bandflux(bands, times, zpsys)
    #     noisy_bandfluxes = np.random.normal(loc=bandfluxes, scale=self.pointings.sky_noise, size=bandfluxes.shape)
    #     # Return observation table
    #     # What columns do we want to add, time, band, band-wave, bandflux, flux_err,
    #     pd.DataFrame(columns=['Time_mjd', 'Band', 'Bandflux', 'Flux_err'])
    #
    #     return
