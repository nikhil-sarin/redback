import numpy as np
from typing import Union
import regex as re
from sncosmo import TimeSeriesSource, Model, get_bandpass
import redback
import pandas as pd
from redback.utils import logger, calc_flux_density_error_from_monochromatic_magnitude, calc_flux_density_from_ABmag
from itertools import repeat
from collections import namedtuple
import astropy.units as uu
from scipy.spatial import KDTree
import os
datadir = os.path.join(os.path.dirname(redback.__file__), 'tables')


class SimulateOpticalTransient(object):
    """
    Simulate a single optical transient from a given observation dictionary
    """
    def __init__(self, model, parameters, pointings_database=None, survey='Rubin_10yr_baseline',
                 sncosmo_kwargs=None, buffer_days=1,
                 obs_buffer=5.0, end_transient_time=1000, population=False, model_kwargs=None, **kwargs):
        if isinstance(model, str):
            self.model = redback.model_library.all_models_dict[model]
            model_kwargs['output_format'] = 'sncosmo_source'
            _time_array = np.linspace(0.1, 3000.0, 10)
            self.sncosmo_model = self.model(_time_array, **parameters, **model_kwargs)
        elif callable(model):
            self.model = model
            logger.info('Using custom model. Making a SNCosmo wrapper for this model')
            self.sncosmo_model = self._make_sncosmo_wrapper_for_user_model()
        else:
            raise ValueError("The user needs to specify model as either a string or function.")

        if survey is not None:
            self.pointings_database_name = self._survey_to_table_name_lookup(survey)
            self.pointings_database = pd.read_csv(datadir + "/" + self.pointings_database_name, compression='gzip')
            logger.info(f"Using {self.pointings_database_name} as the pointing database corresponding to {survey}.")
        else:
            self.pointings_database = pointings_database
            self.pointings_database_name = 'user_defined'
            if isinstance(self.pointings_database, pd.DataFrame):
                logger.info(f"Using the supplied as the pointing database.")
            else:
                raise ValueError("The user needs to specify survey as either a string or a "
                                 "pointings_database pandas DataFrame.")

        self.buffer_days = buffer_days
        self.population = population
        if population:
            self.parameters = pd.DataFrame(parameters)
        else:
            self.parameters = pd.DataFrame(parameters, index=[0])
        self.sncosmo_kwargs = sncosmo_kwargs
        self.obs_buffer = obs_buffer
        self.end_transient_time = self.parameters['t0_mjd_transient'] + end_transient_time
        self.observations = self._make_observations()
        self.inference_observations = self._make_inference_dataframe()

    def _make_inference_dataframe(self):
        df = self.observations
        df = df[df.detected != 0]
        return df

    @property
    def RA(self):
        if 'ra' in self.parameters:
            RA = self.parameters['ra']
        else:
            RA = self.pointings_database['_ra'].sample(len(self.parameters))
        return RA

    @property
    def DEC(self):
        if 'dec' in self.parameters:
            dec = self.parameters['dec']
        else:
            dec = self.pointings_database['_dec'].sample(len(self.parameters))
        return dec

    @property
    def min_dec(self):
        df = self.pointings_database
        return np.min(df['_dec'])

    @property
    def max_dec(self):
        df = self.pointings_database
        return np.max(df['_dec'])

    @property
    def start_mjd(self):
        df = self.pointings_database
        return np.min(df['expMJD'])

    @property
    def end_mjd(self):
        df = self.pointings_database
        return np.max(df['expMJD'])

    def _get_unique_reference_fluxes(self):
        unique_bands = self.pointings_database.filters.unique()
        ref_flux = redback.utils.bands_to_reference_flux(unique_bands)
        return ref_flux

    def _make_sncosmo_wrapper_for_redback_model(self):
        model_kwargs = {}
        self.sncosmo_kwargs['max_time'] = self.sncosmo_kwargs.get('max_time', 100)
        model_kwargs['output_format'] = 'sncosmo_source'
        time = self.sncosmo_kwargs.get('dense_times', np.linspace(0, self.sncosmo_kwargs['max_time'], 200))
        full_kwargs = self.parameters.copy()
        full_kwargs.update(model_kwargs)
        source = self.model(time, **full_kwargs)
        return Model(source=source, **self.sncosmo_kwargs)

    def _make_sncosmo_wrapper_for_user_model(self):
        """
        Function to wrap redback models into sncosmo model format for added functionality.
        """
        self.sncosmo_kwargs['max_time'] = self.sncosmo_kwargs.get('max_time', 100)
        self.parameters['wavelength_observer_frame'] = self.parameters.get('wavelength_observer_frame',
                                                                          np.geomspace(100,60000,100))
        time = self.sncosmo_kwargs.get('dense_times', np.linspace(0, self.sncosmo_kwargs['max_time'], 200))
        fmjy = self.model(time, **self.parameters)
        spectra = fmjy.to(uu.mJy).to(uu.erg / uu.cm ** 2 / uu.s / uu.Angstrom,
                equivalencies=uu.spectral_density(wav=self.parameters['wavelength_observer_frame'] * uu.Angstrom))
        source = TimeSeriesSource(phase=time, wave=self.parameters['wavelength_observer_frame'],
                                  flux=spectra)
        return Model(source=source, **self.sncosmo_kwargs)

    def _survey_to_table_name_lookup(self, survey):
        survey_to_table = {'Rubin_10yr_baseline': 'rubin_baseline_v3.0_10yrs.tar.gz',
                           'Rubin_10yr_morez': 'rubin_morez.tar.gz',
                           'Rubin_10yr_lessweight': 'rubin_lessweight.tar.gz',
                           'ZTF_deep': 'ztf_deep_nexp1_v1_7_10yrs.tar.gz',
                           'ZTF_wfd': 'ztf_wfd_nexp1_v1_7_10yrs.tar.gz'}
        return survey_to_table[survey]

    def _make_observation_single(self, overlapping_database):
        times = overlapping_database['expMJD'].values - self.parameters['t0_mjd_transient'].values
        filters = overlapping_database['filter'].values
        magnitude = self.sncosmo_model.bandmag(phase=times, band=filters, magsys='AB')
        flux = redback.utils.bandpass_magnitude_to_flux(magnitude=magnitude, bands=filters)
        ref_flux = redback.utils.bands_to_reference_flux(filters)
        bandflux_errors = redback.utils.bandflux_error_from_limiting_mag(overlapping_database['fiveSigmaDepth'].values,
                                                                         ref_flux)
        # what can be preprocessed
        observed_flux = np.random.normal(loc=flux, scale=bandflux_errors)
        magnitudes = redback.utils.bandpass_flux_to_magnitude(observed_flux, filters)
        magnitude_errs = redback.utils.magnitude_error_from_flux_error(flux, bandflux_errors)
        flux_density = calc_flux_density_from_ABmag(magnitude).value

        observation_dataframe = pd.DataFrame()
        observation_dataframe['time'] = overlapping_database['expMJD'].values
        observation_dataframe['magnitude'] = magnitudes
        observation_dataframe['e_magnitude'] = magnitude_errs
        observation_dataframe['band'] = filters
        observation_dataframe['system'] = 'AB'
        observation_dataframe['flux_density(mjy)'] = flux_density
        observation_dataframe['flux_density_error'] = calc_flux_density_error_from_monochromatic_magnitude(
            magnitude=magnitude, magnitude_error=magnitude_errs, reference_flux=3631)
        observation_dataframe['flux(erg/cm2/s)'] = observed_flux
        observation_dataframe['flux_error'] = bandflux_errors
        observation_dataframe['time (days)'] = times
        mask = (observation_dataframe['time (days)'] <= 0.) | (np.isnan(observation_dataframe['magnitude']))
        snr = observed_flux/bandflux_errors
        mask_snr = snr < 8.
        detected = np.ones(len(observation_dataframe))
        detected[mask] = 0
        detected[mask_snr] = 0
        observation_dataframe['detected'] = detected
        observation_dataframe['limiting_magnitude'] = overlapping_database['fiveSigmaDepth'].values
        return observation_dataframe

    def _make_observations(self):
        overlapping_sky_indices = self._find_sky_overlaps(survey_fov_sqdeg=9.6)
        overlapping_time_indices = self._find_time_overlaps(self.obs_buffer)

        if self.population:
            space_set = set(overlapping_sky_indices)
        else:
            space_set = set(overlapping_sky_indices)

        time_set = set(overlapping_time_indices)

        time_space_overlap = list(space_set.intersection(time_set))

        overlapping_database_iter = self.pointings_database.iloc[time_space_overlap]
        overlapping_database_iter = overlapping_database_iter.sort_values(by=['expMJD'])
        dataframe = self._make_observation_single(overlapping_database_iter)
        return dataframe


    def _convert_circular_fov_to_radius(self, survey_fov_sqdeg):
        radius = np.sqrt(survey_fov_sqdeg*((np.pi/180.0)**2.0)/np.pi)
        return radius

    def _find_time_overlaps(self, obs_buffer):
        pointing_times = self.pointings_database[['expMJD']].values.flatten()
        condition_1 = pointing_times >= self.parameters['t0_mjd_transient'].values - obs_buffer
        condition_2 = pointing_times <= self.end_transient_time.values
        mask = np.logical_and(condition_1, condition_2)
        time_indices = np.where(mask)
        return time_indices[0]


    def _find_sky_overlaps(self, survey_fov_sqdeg=None):
        """
        Makes a pandas dataframe of pointings from specified settings.

        :param float: ra
        :param float: dec
        :param dict: num_obs
        :param dict: average_cadence
        :param dict: cadence_scatter
        :param dict: limiting_magnitudes

        :return dataframe: pandas dataframe of the mock pointings needed to simulate observations for
        given transient.
        """
        pointings_sky_pos = np.column_stack((self.pointings_database['_ra'].values, self.pointings_database['_dec'].values))
        transient_sky_pos = np.column_stack((self.parameters['ra'].values, self.parameters['dec'].values))
        survey_fov_radius = self._convert_circular_fov_to_radius(survey_fov_sqdeg=survey_fov_sqdeg)

        transient_sky_pos_3D = np.vstack([np.cos(transient_sky_pos[:,0]) * np.cos(transient_sky_pos[:,1]), np.sin(transient_sky_pos[:,0]) * np.cos(transient_sky_pos[:,1]), np.sin(transient_sky_pos[:,1])]).T
        pointings_sky_pos_3D = np.vstack([np.cos(pointings_sky_pos[:, 0]) * np.cos(pointings_sky_pos[:,1]), np.sin(pointings_sky_pos[:,0]) * np.cos(pointings_sky_pos[:,1]), np.sin(pointings_sky_pos[:,1])]).T
        # law of cosines to compute 3D distance
        max_3D_dist = np.sqrt(2. - 2. * np.cos(survey_fov_radius))
        survey_tree = KDTree(pointings_sky_pos_3D)
        overlap_indices = survey_tree.query_ball_point(x=transient_sky_pos_3D.T.flatten(), r=max_3D_dist)
        return overlap_indices



    def _save_transient(self):
        self.observations.to_csv()

    @classmethod
    def simulate_transient(cls, model, parameters, pointings_database=None, survey='Rubin_10yr_baseline',
                           buffer_days=100, population=False, model_kwargs=None, **kwargs):
        """
        Makes a pandas dataframe of pointings from specified settings.

        :param float: ra
        :param float: dec
        :param dict: num_obs
        :param dict: average_cadence
        :param dict: cadence_scatter
        :param dict: limiting_magnitudes

        :return dataframe: pandas dataframe of the mock pointings needed to simulate observations for
        given transient.
        """
        return cls(model, parameters, pointings_database=pointings_database, survey=survey,
                   buffer_days=buffer_days, population=population, model_kwargs=model_kwargs, **kwargs)

    @classmethod
    def simulate_transient_population(cls, model, parameters, pointings_database=None, survey='Rubin_10yr_baseline',
                           buffer_days=100, population=True, **kwargs):
        """
        Makes a pandas dataframe of pointings from specified settings.

        :param float: ra
        :param float: dec
        :param dict: num_obs
        :param dict: average_cadence
        :param dict: cadence_scatter
        :param dict: limiting_magnitudes

        :return dataframe: pandas dataframe of the mock pointings needed to simulate observations for
        given transient.
        """
        return cls(model, parameters, pointings_database=pointings_database, survey=survey,
                   buffer_days=buffer_days, population=population, **kwargs)

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

        #
        # :param dataframe: pandas dataframe of parameters to make simulated observations for
        # :param outdir: output directory where the simulated observations will be saved
        # :return:
        # """

def make_pointing_table_from_average_cadence(ra, dec, num_obs, average_cadence, cadence_scatter, limiting_magnitudes, **kwargs):
    """
    Makes a pandas dataframe of pointings from specified settings.

    :param float: ra
    :param float: dec
    :param dict: num_obs
    :param dict: average_cadence
    :param dict: cadence_scatter
    :param dict: limiting_magnitudes

    :return dataframe: pandas dataframe of the mock pointings needed to simulate observations for
    given transient.
    """
    initMJD = kwargs.get("initMJD", 59580)
    df_list = []
    for band in average_cadence.keys():
        expMJD = initMJD + np.cumsum(np.random.normal(loc=average_cadence[band], scale=cadence_scatter[band], size=num_obs[band]))
        filters = list(repeat(band, num_obs[band]))
        limiting_mag = list(repeat(limiting_magnitudes[band], num_obs[band]))
        ras = np.ones(num_obs[band])*ra
        decs = np.ones(num_obs[band])*dec
        band_pointings_dataframe = pd.DataFrame.from_dict({'expMJD': expMJD, '_ra': ras, '_dec': decs, 'filter': filters, 'fiveSigmaDepth': limiting_mag})
        df_list.append(band_pointings_dataframe)
    pointings_dataframe = pd.concat(df_list)
    pointings_dataframe.sort_values('expMJD', inplace=True)
    return pointings_dataframe

class SimulateFullOpticalSurvey(SimulateOpticalTransient):
    """
    Do SimSurvey or SNANA
    """
    def __init__(self, model, parameters, pointings_database=None, survey='Rubin_10yr_baseline',
                 min_dec=None,max_dec=None, start_mjd=None, end_mjd=None, buffer_days=100,
                 population=False, **kwargs):
        super().__init__(model=model, parameters=parameters, pointings_database=pointings_database, survey=survey,
                 min_dec=min_dec,max_dec=max_dec, start_mjd=start_mjd, end_mjd=end_mjd, buffer_days=buffer_days,
                 population=population, **kwargs)

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
