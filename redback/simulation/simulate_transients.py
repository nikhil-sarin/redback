import numpy as np
from typing import Union
import regex as re
from sncosmo import TimeSeriesSource, Model, get_bandpass
import simsurvey
import redback
import pandas as pd
from redback.utils import logger
from collections import namedtuple
import astropy.units as uu


class SimulateOpticalTransient(object):
    """
    Simulate a single optical transient from a given observation dictionary
    """
    def __init__(self, model, parameters, pointings_database=None, survey='Rubin_10yr_baseline',
                 min_dec=None,max_dec=None, dec_dist=None, start_mjd=None, end_mjd=None, buffer_days=100, **kwargs):
        """
        :param model: str of redback model
        :param parameters:s
        :param start_mjd:
        """
        if isinstance(model, str):
            self.model = redback.model_library.all_models_dict[model]
        elif callable(model):
            self.model = model
            logger.info('Using custom model. Making a SNCosmo wrapper for this model')
            self._sncosmo_function_wrapper = self._sncosmo_function_wrapper()
        else:
            raise ValueError("The user needs to specify model as either a string or function.")

        if survey is not None:
            self.pointing_database = self._survey_to_table_name_lookup(survey)
            logger.info(f"Using {survey} as the pointing database.")
        else:
            self.pointing_database = pointings_database
            if isinstance(survey, pd.DataFrame):
                logger.info(f"Using the supplied {pointings_database} as the pointing database.")
            else:
                raise ValueError("The user needs to specify survey as either a string or a "
                                 "pointings_databse pandas DataFrame.")

        self.buffer_days = buffer_days
        self.parameters = parameters
        self.min_dec = min_dec
        self.max_dec = max_dec
        self.dec_dist = dec_dist
        self.start_mjd = start_mjd
        self.end_mjd = end_mjd

    @property
    def _initialise_from_pointings(self):
        df = pd.read_csv(self.pointing_database, compression='gzip')
        min_dec = np.min(df['_dec'])
        max_dec = np.max(df['_dec'])
        dec_dist = (np.arccos(2* np.random.uniform(low=(1 - np.sin(min_dec)) / 2,
                                                   high=(1 - np.sin(max_dec)) / 2,
                                                   size=1)- 1) - np.pi / 2)
        start_mjd = np.min(df['expMJD'])
        end_mjd = np.max(df['expMJD'])
        pointing_df = df
        return self.min_dec, max_dec, dec_dist, start_mjd, end_mjd, pointing_df

    def _make_sncosmo_wrapper(self,**kwargs):
        model_kwargs = {}
        model_kwargs['output_format'] = 'sncosmo_source'
        time = np.linspace(0, 100, 1000)
        full_kwargs = self.parameters.copy()
        full_kwargs.update(model_kwargs)
        source = self.model(time, **full_kwargs)
        return Model(source=source)

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



    def _initialise_from_parameters(self):
        self.RA = self.parameters.get("RA", np.random.uniform(0, 2*np.pi))
        self.DEC = self.parameters.get("DEC", np.random.uniform(-np.pi/2, np.pi/2))
        self.mjd = self.parameters.get("mjd", np.random.uniform((self.start_mjd - self.buffer_days), (self.end_mjd)))
        pass

    def _make_observations(self):
        pass

    def _save_transient(self):
        pass

    def _survey_to_table_name_lookup(self, survey):
        survey_to_table = {'Rubin_10yr_baseline': 'rubin_baseline_nexp1_v1_7_10yrs.tar.gz',
                           'Rubin_10yr_deep': 'rubin_deep_nexp1_v1_7_10yrs.tar.gz',
                           'Rubin_10yr_wfd': 'rubin_wfd_nexp1_v1_7_10yrs.tar.gz',
                           'Rubin_10yr_dcr': 'rubin_dcr_nexp1_v1_7_10yrs.tar.gz',
                           'ZTF_deep': 'ztf_deep_nexp1_v1_7_10yrs.tar.gz',
                           'ZTF_wfd': 'ztf_wfd_nexp1_v1_7_10yrs.tar.gz'}
        return survey_to_table[survey]

    # @classmethod
    # def simulate_transient(cls, model, parameters, buffer_days=100, survey='Rubin_10yr_baseline', **kwargs):
    #     if isinstance(survey, str):
    #         # load the corresponding file
    #         table = cls._survey_to_table_name_lookup(survey)
    #         simtransient = cls(model, pointings_database=pointings_database, **kwargs)
    #
    #     if isinstance(survey, pd.DataFrame):
    #         # initialise from the datafile.
    #     self.RA = parameters.get("RA", np.random.uniform(0, 2*np.pi))
    #     self.DEC = parameters.get("DEC", np.random.uniform(-np.pi/2, np.pi/2))
    #     self.mjd = parameters.get("mjd", np.random.uniform((self.start_mjd - buffer_days), (self.end_mjd)))
    #     #
    #     #make observations
    #     data = self._make_observations()
    #
    #     #save the file to disk
    #     self._save_transient()
    #     logger.info("Transient simulated and saved to disk.")
    #     return data

    @classmethod
    def simulate_transient_population(self, dataframe, outdir):
        """
        Simulate a population of transients from a dataframe of parameters.

        :param dataframe: pandas dataframe of parameters to make simulated observations for
        :param outdir: output directory where the simulated observations will be saved
        :return:
        """
        RA =
        DEC =
        mjd =
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
