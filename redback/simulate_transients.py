import numpy as np
from sncosmo import TimeSeriesSource, Model, get_bandpass
import redback
import pandas as pd
from redback.utils import logger, calc_flux_density_error_from_monochromatic_magnitude, calc_flux_density_from_ABmag
from itertools import repeat
import astropy.units as uu
from scipy.spatial import KDTree
import os
import bilby
import random
from astropy.cosmology import Planck18 as cosmo
datadir = os.path.join(os.path.dirname(redback.__file__), 'tables')

class SimulateGenericTransient(object):
    def __init__(self, model, parameters, times, model_kwargs, data_points,
                 seed=1234, multiwavelength_transient=False, noise_term=0.2, noise_type='gaussianmodel', extra_scatter=0.0):
        """
        A generic interface to simulating transients

        :param model: String corresponding to redback model or a python function that can evaluate an SED.
        :param parameters: Dictionary of parameters describing a single transient
        :param times: Time values that the model is evaluated from
        :param model_kwargs: Additional keyword arguments, must include all the keyword arguments required by the model.
                Refer to the model documentation for details
        :param data_points: Number of data points to randomly sample.
                This will randomly sample data_points in time and in bands or frequency.
        :param seed: random seed for reproducibility
        :param multiwavelength_transient: Boolean.
                If True, the model is assumed to be a transient which has multiple bands/frequency
                and the data points are sampled in bands/frequency as well,
                rather than just corresponding to one wavelength/filter.
                This also allows the same time value to be sampled multiple times.
        :param noise_type: String. Type of noise to add to the model.
            Default is 'gaussianmodel' where sigma is noise_term * model.
            Another option is 'gaussian' i.e., a simple Gaussian noise with sigma = noise_term.
        :param noise_term: Float. Factor which is multiplied by the model flux/magnitude to give the sigma
            or is sigma itself for 'gaussian' noise. Or the SNR for 'SNRbased' noise.
        :param extra_scatter: Float. Sigma of normal added to output for additional scatter.
        """
        if model in redback.model_library.all_models_dict:
            self.model = redback.model_library.all_models_dict[model]
        else:
            self.model = model 
        self.parameters = parameters
        self.all_times = times
        self.model_kwargs = model_kwargs
        self.multiwavelength_transient = multiwavelength_transient
        self.data_points = data_points
        self.seed = seed
        self.random_state = np.random.RandomState(seed=self.seed)
        self.noise_term = noise_term
        random.seed(self.seed)

        self.all_bands = self.model_kwargs.get('bands', None)
        self.all_frequency = self.model_kwargs.get('frequency', None)
        if self.all_bands is None and self.all_frequency is None:
            raise ValueError('Must supply either bands or frequency to sample data points for an optical transient')
        else:
            if multiwavelength_transient:
                if self.all_bands is not None and self.all_frequency is None:
                    self.subset_bands = np.array(random.choices(self.all_bands, k=self.data_points))
                if self.all_bands is None and self.all_frequency is not None:
                    self.subset_frequency = np.array(random.choices(self.all_frequency, k=self.data_points))
                self.replacement = True
                # allow times to be chosen repeatedly
            else:
                if self.all_bands is not None and self.all_frequency is None:
                    self.subset_bands = self.data_points * [self.all_bands]
                if self.all_bands is None and self.all_frequency is not None:
                    self.subset_frequency = np.ones(self.data_points) * self.all_frequency
                # allow times to be chosen only once.
                self.replacement = False
        self.subset_times = np.sort(np.random.choice(self.all_times, size=self.data_points, replace=self.replacement))

        injection_kwargs = self.parameters.copy()
        if 'bands' in model_kwargs.keys():
            injection_kwargs['bands'] = self.subset_bands
            injection_kwargs['output_format'] = 'magnitude'
        if 'frequency' in model_kwargs.keys():
            injection_kwargs['frequency'] = self.subset_frequency
            injection_kwargs['output_format'] = 'flux_density'

        true_output = self.model(self.subset_times, **injection_kwargs)
        data = pd.DataFrame()
        data['time'] = self.subset_times
        if 'bands' in model_kwargs.keys():
            data['band'] = self.subset_bands
        if 'frequency' in model_kwargs.keys():
            data['frequency'] = self.subset_frequency
        data['true_output'] = true_output

        if noise_type == 'gaussianmodel':
            noise = np.random.normal(0, self.noise_term * true_output, len(true_output))
            output = true_output + noise
            output_error = self.noise_term * true_output
        elif noise_type == 'gaussian':
            noise = np.random.normal(0, self.noise_term, len(true_output))
            output = true_output + noise
            output_error = self.noise_term
        elif noise_type == 'SNRbased':
            sigma = np.sqrt(true_output + np.min(true_output)/self.noise_term)
            output_error = sigma
            output = true_output + np.random.normal(0, sigma, len(true_output))
        else:
            logger.warning(f"noise_type {noise_type} not implemented.")
            raise ValueError('noise_type must be either gaussianmodel, gaussian, or SNRBased')

        if extra_scatter > 0:
            extra_noise = np.random.normal(0, extra_scatter, len(true_output))
            output = output + extra_noise
            output_error = np.sqrt(output_error**2 + extra_noise**2)

        data['output'] = output
        data['output_error'] = output_error
        self.data = data

    def save_transient(self, name):
        """
        Save the transient observations to a csv file.
        This will save the full observational dataframe including non-detections etc.
        This will save the data to a folder called 'simulated'
        with the name of the transient and a csv file of the injection parameters

        :param name: name to save transient.
        """
        bilby.utils.check_directory_exists_and_if_not_mkdir('simulated')
        path = 'simulated/' + name + '.csv'
        injection_path = 'simulated/' + name + '_injection_parameters.csv'
        self.data.to_csv(path, index=False)
        self.parameters=pd.DataFrame.from_dict([self.parameters])
        self.parameters.to_csv(injection_path, index=False)

class SimulateOpticalTransient(object):
    def __init__(self, model, parameters, pointings_database=None,
                 survey='Rubin_10yr_baseline',sncosmo_kwargs=None, obs_buffer=5.0,
                 survey_fov_sqdeg=9.6,snr_threshold=5, end_transient_time=1000, add_source_noise=False,
                 population=False, model_kwargs=None, **kwargs):
        """
        Simulate an optical transient or transient population for an optical Survey like Rubin, ZTF, Roman etc

        :param model: String corresponding to redback model or a python function that can evaluate an SED.
        :param parameters: Dictionary of parameters describing a single transient or a transient population.
                This can either include RA and DEC or it is randomly drawn from the pointing database.
                Must include t0_mjd_transient or t0.
        :param pointings_database: A pandas DataFrame containing the pointings of the survey.
        :param survey: String corresponding to the survey name. This is used to look up the pointings database.
                Set to LSST 10 year baseline 3.0 by default. If None, the user must supply a pointings_database.
                Implemented surveys currently include a Rubin 10 year baseline 3.0 as 'Rubin_10yr_baseline, and ZTF as 'ztf'.
        :param sncosmo_kwargs: Any kwargs to be passed to SNcosmo.
                SNcosmo is used to evaluate the bandpass magnitudes in different bands.
        :param obs_buffer: A observation buffer in days to add to the start of the transient
                to allow for non-detections. Default is 5 days
        :param survey_fov_sqdeg: Survey field of view. Default is 9.6 sqdeg for Rubin.
                36" for ZTF as a circular approximation to the square FOV of ZTF.
        :param snr_threshold: SNR threshold for detection. Default is 5.
        :param end_transient_time: End time of the transient in days. Default is 1000 days.
                Note that SNCosmo will extrapolate past when the transient model evaluates the SED so these should really be the same.
        :param add_source_noise: Boolean. If True, add an extra noise in quadrature to the limiting mag noise.
                The factor is a multiple of the model flux i.e. noise = (skynoise**2 + (model_flux*source_noise)**2)**0.5
        :param population: Boolean. If True, the parameters are assumed to be for a population of transients.
        :param model_kwargs: Dictionary of kwargs to be passed to the model.
        :param kwargs: Dictionary of additional kwargs
        :param source_noise: Float. Factor to multiply the model flux by to add an extra noise
                in quadrature to the limiting mag noise. Default value is 0.02, disabled by default.
        """

        self.survey_fov_sqdeg = survey_fov_sqdeg
        self.snr_threshold = snr_threshold
        self.population = population
        self.source_noise_factor = kwargs.get('source_noise', 0.02)
        if population:
            self.parameters = pd.DataFrame(parameters)
        else:
            self.parameters = pd.DataFrame(parameters, index=[0])
        self.sncosmo_kwargs = sncosmo_kwargs
        self.obs_buffer = obs_buffer
        self.end_transient_time = self.t0_transient + end_transient_time
        self.add_source_noise = add_source_noise

        if isinstance(model, str):
            self.model = redback.model_library.all_models_dict[model]
            model_kwargs['output_format'] = 'sncosmo_source'
            _time_array = np.linspace(0.1, 3000.0, 10)
            if self.population:
                self.all_sncosmo_models = []
                for x in range(len(self.parameters)):
                    sncosmomodel = self.model(_time_array, **self.parameters.iloc[x], **model_kwargs)
                    self.all_sncosmo_models.append(sncosmomodel)
            else:
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

        self.parameters['ra'] = self.RA
        self.parameters['dec'] = self.DEC

        if population:
            self.list_of_observations = self._make_observations_for_population()
            self.list_of_inference_observations = self._make_inference_dataframe()
        else:
            self.observations = self._make_observations()
            self.inference_observations = self._make_inference_dataframe()
            self.inference_observations = self._make_inference_dataframe()

    @classmethod
    def simulate_transient(cls, model, parameters, pointings_database=None,
                 survey='Rubin_10yr_baseline',sncosmo_kwargs=None, obs_buffer=5.0, survey_fov_sqdeg=9.6,
                 snr_threshold=5, end_transient_time=1000, add_source_noise=False, model_kwargs=None, **kwargs):
        """
        Constructor method to build simulated transient object for a single transient.

        :param model: String corresponding to redback model or a python function that can evaluate an SED.
        :param parameters: Dictionary of parameters describing a single transient or a transient population.
            This can either include RA and DEC or it is randomly drawn from the pointing database.
            Must include t0_mjd_transient or t0.
        :param pointings_database: A pandas DataFrame containing the pointings of the survey.
        :param survey: String corresponding to the survey name. This is used to look up the pointings database.
            Set to LSST 10 year baseline 3.0 by default.
        :param sncosmo_kwargs: Any kwargs to be passed to SNcosmo.
            SNcosmo is used to evaluate the bandpass magnitudes in different bands.
        :param obs_buffer: A observation buffer in days to add to the start of the transient
            to allow for non-detections. Default is 5 days
        :param survey_fov_sqdeg: Survey field of view. Default is 9.6 sqdeg for Rubin.
        :param snr_threshold: SNR threshold for detection. Default is 5.
        :param end_transient_time: End time of the transient in days. Default is 1000 days.
            Note that SNCosmo will extrapolate past when the transient model evaluates the SED so these should really be the same.
        :param add_source_noise: Boolean. If True, add an extra noise in quadrature to the limiting mag noise.
            The factor is a multiple of the model flux i.e. noise = (skynoise**2 + (model_flux*source_noise)**2)**0.5
        :param population: Boolean. If True, the parameters are assumed to be for a population of transients.
        :param model_kwargs: Dictionary of kwargs to be passed to the model.
        :param kwargs: Dictionary of additional kwargs
        :param source_noise: Float. Factor to multiply the model flux by to add an extra noise
            in quadrature to the limiting mag noise. Default value is 0.02, disabled by default.
        """
        return cls(model=model, parameters=parameters, pointings_database=pointings_database, survey=survey,
                   sncosmo_kwargs=sncosmo_kwargs, obs_buffer=obs_buffer,
                   survey_fov_sqdeg=survey_fov_sqdeg, snr_threshold=snr_threshold,
                   end_transient_time=end_transient_time, add_source_noise=add_source_noise,
                   population=False, model_kwargs=model_kwargs, **kwargs)

    @classmethod
    def simulate_transient_in_rubin(cls, model, parameters, pointings_database=None,
                 survey='Rubin_10yr_baseline',sncosmo_kwargs=None, obs_buffer=5.0,
                snr_threshold=5, end_transient_time=1000, add_source_noise=False, model_kwargs=None, **kwargs):
        """
        Constructor method to build simulated transient object for a single transient with Rubin.

        :param model: String corresponding to redback model or a python function that can evaluate an SED.
        :param parameters: Dictionary of parameters describing a single transient or a transient population.
            This can either include RA and DEC or it is randomly drawn from the pointing database.
            Must include t0_mjd_transient or t0.
        :param pointings_database: A pandas DataFrame containing the pointings of the survey.
        :param survey: String corresponding to the survey name. This is used to look up the pointings database.
            Set to LSST 10 year baseline 3.0 by default.
        :param sncosmo_kwargs: Any kwargs to be passed to SNcosmo.
            SNcosmo is used to evaluate the bandpass magnitudes in different bands.
        :param obs_buffer: A observation buffer in days to add to the start of the transient
            to allow for non-detections. Default is 5 days
        :param snr_threshold: SNR threshold for detection. Default is 5.
        :param end_transient_time: End time of the transient in days. Default is 1000 days.
            Note that SNCosmo will extrapolate past when the transient model evaluates the SED so these should really be the same.
        :param add_source_noise: Boolean. If True, add an extra noise in quadrature to the limiting mag noise.
            The factor is a multiple of the model flux i.e. noise = (skynoise**2 + (model_flux*source_noise)**2)**0.5
        :param model_kwargs: Dictionary of kwargs to be passed to the model.
        :param kwargs: Dictionary of additional kwargs
        :param source_noise: Float. Factor to multiply the model flux by to add an extra noise
            in quadrature to the limiting mag noise. Default value is 0.02, disabled by default.
        """
        return cls(model=model, parameters=parameters, pointings_database=pointings_database, survey=survey,
                   sncosmo_kwargs=sncosmo_kwargs, obs_buffer=obs_buffer,
                   survey_fov_sqdeg=9.6, snr_threshold=snr_threshold,
                   end_transient_time=end_transient_time, add_source_noise=add_source_noise,
                   population=False, model_kwargs=model_kwargs, **kwargs)

    @classmethod
    def simulate_transient_in_ztf(cls, model, parameters, pointings_database=None,
                 survey='ztf',sncosmo_kwargs=None, obs_buffer=5.0,
                  snr_threshold=5, end_transient_time=1000, add_source_noise=False, model_kwargs=None, **kwargs):
        """
        Constructor method to build simulated transient object for a single transient with ZTF.

        :param model: String corresponding to redback model or a python function that can evaluate an SED.
        :param parameters: Dictionary of parameters describing a single transient or a transient population.
            This can either include RA and DEC or it is randomly drawn from the pointing database.
            Must include t0_mjd_transient or t0.
        :param pointings_database: A pandas DataFrame containing the pointings of the survey.
        :param survey: String corresponding to the survey name. This is used to look up the pointings database.
        :param sncosmo_kwargs: Any kwargs to be passed to SNcosmo.
            SNcosmo is used to evaluate the bandpass magnitudes in different bands.
        :param obs_buffer: A observation buffer in days to add to the start of the transient
            to allow for non-detections. Default is 5 days
        :param snr_threshold: SNR threshold for detection. Default is 5.
        :param end_transient_time: End time of the transient in days. Default is 1000 days.
            Note that SNCosmo will extrapolate past when the transient model evaluates the SED so these should really be the same.
        :param add_source_noise: Boolean. If True, add an extra noise in quadrature to the limiting mag noise.
            The factor is a multiple of the model flux i.e. noise = (skynoise**2 + (model_flux*source_noise)**2)**0.5
        :param model_kwargs: Dictionary of kwargs to be passed to the model.
        :param kwargs: Dictionary of additional kwargs
        :param source_noise: Float. Factor to multiply the model flux by to add an extra noise
            in quadrature to the limiting mag noise. Default value is 0.02, disabled by default.
        """
        return cls(model=model, parameters=parameters, pointings_database=pointings_database, survey=survey,
                   sncosmo_kwargs=sncosmo_kwargs, obs_buffer=obs_buffer,
                   survey_fov_sqdeg=36., snr_threshold=snr_threshold,
                   end_transient_time=end_transient_time,add_source_noise=add_source_noise,
                   population=False, model_kwargs=model_kwargs, **kwargs)

    @classmethod
    def simulate_transient_population(cls, model, parameters, pointings_database=None,
                 survey='Rubin_10yr_baseline',sncosmo_kwargs=None,obs_buffer=5.0, survey_fov_sqdeg=9.6,
                 snr_threshold=5, end_transient_time=1000, add_source_noise=False, model_kwargs=None, **kwargs):
        """
        Constructor method to build simulated transient object for a single transient.

        :param model: String corresponding to redback model or a python function that can evaluate an SED.
        :param parameters: Dictionary of parameters describing a single transient or a transient population.
            This can either include RA and DEC or it is randomly drawn from the pointing database.
            Must include t0_mjd_transient or t0.
        :param pointings_database: A pandas DataFrame containing the pointings of the survey.
        :param survey: String corresponding to the survey name. This is used to look up the pointings database.
            Set to LSST 10 year baseline 3.0 by default.
        :param sncosmo_kwargs: Any kwargs to be passed to SNcosmo.
            SNcosmo is used to evaluate the bandpass magnitudes in different bands.
        :param obs_buffer: A observation buffer in days to add to the start of the transient
            to allow for non-detections. Default is 5 days
        :param survey_fov_sqdeg: Survey field of view. Default is 9.6 sqdeg for Rubin.
        :param snr_threshold: SNR threshold for detection. Default is 5.
        :param end_transient_time: End time of the transient in days. Default is 1000 days.
            Note that SNCosmo will extrapolate past when the transient model evaluates the SED so these should really be the same.
        :param add_source_noise: Boolean. If True, add an extra noise in quadrature to the limiting mag noise.
            The factor is a multiple of the model flux i.e. noise = (skynoise**2 + (model_flux*source_noise)**2)**0.5
        :param model_kwargs: Dictionary of kwargs to be passed to the model.
        :param kwargs: Dictionary of additional kwargs
        :param source_noise: Float. Factor to multiply the model flux by to add an extra noise
            in quadrature to the limiting mag noise. Default value is 0.02, disabled by default.
        """
        return cls(model=model, parameters=parameters, pointings_database=pointings_database, survey=survey,
                   sncosmo_kwargs=sncosmo_kwargs, obs_buffer=obs_buffer,
                   survey_fov_sqdeg=survey_fov_sqdeg, snr_threshold=snr_threshold,
                   end_transient_time=end_transient_time, add_source_noise=add_source_noise,
                   population=True, model_kwargs=model_kwargs, **kwargs)

    @classmethod
    def simulate_transient_population_in_rubin(cls, model, parameters, pointings_database=None,
                 survey='Rubin_10yr_baseline',sncosmo_kwargs=None, obs_buffer=5.0,
                 snr_threshold=5, end_transient_time=1000, add_source_noise=False, model_kwargs=None, **kwargs):
        """
        Constructor method to build simulated transient object for a single transient.

        :param model: String corresponding to redback model or a python function that can evaluate an SED.
        :param parameters: Dictionary of parameters describing a single transient or a transient population.
            This can either include RA and DEC or it is randomly drawn from the pointing database.
            Must include t0_mjd_transient or t0.
        :param pointings_database: A pandas DataFrame containing the pointings of the survey.
        :param survey: String corresponding to the survey name. This is used to look up the pointings database.
            Set to LSST 10 year baseline 3.0 by default.
        :param sncosmo_kwargs: Any kwargs to be passed to SNcosmo.
            SNcosmo is used to evaluate the bandpass magnitudes in different bands.
        :param obs_buffer: A observation buffer in days to add to the start of the transient
            to allow for non-detections. Default is 5 days
        :param snr_threshold: SNR threshold for detection. Default is 5.
        :param end_transient_time: End time of the transient in days. Default is 1000 days.
        :param add_source_noise: Boolean. If True, add an extra noise in quadrature to the limiting mag noise.
            The factor is a multiple of the model flux i.e. noise = (skynoise**2 + (model_flux*source_noise)**2)**0.5
        :param model_kwargs: Dictionary of kwargs to be passed to the model.
        :param kwargs: Dictionary of additional kwargs
        :param source_noise: Float. Factor to multiply the model flux by to add an extra noise
            in quadrature to the limiting mag noise. Default value is 0.02, disabled by default.
        """
        return cls(model=model, parameters=parameters, pointings_database=pointings_database, survey=survey,
                   sncosmo_kwargs=sncosmo_kwargs, obs_buffer=obs_buffer,
                   survey_fov_sqdeg=9.6, snr_threshold=snr_threshold,
                   end_transient_time=end_transient_time,add_source_noise=add_source_noise,
                   population=True, model_kwargs=model_kwargs, **kwargs)

    @classmethod
    def simulate_transient_population_in_ztf(cls, model, parameters, pointings_database=None,
                 survey='ztf',sncosmo_kwargs=None, obs_buffer=5.0,
                 snr_threshold=5, end_transient_time=1000, add_source_noise=False, model_kwargs=None, **kwargs):
        """
        Constructor method to build simulated transient object for a single transient.

        :param model: String corresponding to redback model or a python function that can evaluate an SED.
        :param parameters: Dictionary of parameters describing a single transient or a transient population.
            This can either include RA and DEC or it is randomly drawn from the pointing database.
            Must include t0_mjd_transient or t0.
        :param pointings_database: A pandas DataFrame containing the pointings of the survey.
        :param survey: String corresponding to the survey name. This is used to look up the pointings database.
        :param sncosmo_kwargs: Any kwargs to be passed to SNcosmo.
            SNcosmo is used to evaluate the bandpass magnitudes in different bands.
        :param obs_buffer: A observation buffer in days to add to the start of the transient
            to allow for non-detections. Default is 5 days
        :param snr_threshold: SNR threshold for detection. Default is 5.
        :param end_transient_time: End time of the transient in days. Default is 1000 days.
            Note that SNCosmo will extrapolate past when the transient model evaluates the SED so these should really be the same.
        :param add_source_noise: Boolean. If True, add an extra noise in quadrature to the limiting mag noise.
            The factor is a multiple of the model flux i.e. noise = (skynoise**2 + (model_flux*source_noise)**2)**0.5
        :param model_kwargs: Dictionary of kwargs to be passed to the model.
        :param kwargs: Dictionary of additional kwargs
        :param source_noise: Float. Factor to multiply the model flux by to add an extra noise
            in quadrature to the limiting mag noise. Default value is 0.02, disabled by default.
        """
        return cls(model=model, parameters=parameters, pointings_database=pointings_database, survey=survey,
                   sncosmo_kwargs=sncosmo_kwargs, obs_buffer=obs_buffer,
                   survey_fov_sqdeg=36., snr_threshold=snr_threshold,
                   end_transient_time=end_transient_time, add_source_noise=add_source_noise,
                   population=True, model_kwargs=model_kwargs, **kwargs)

    def _make_inference_dataframe(self):
        """
        Make a dataframe that can be used for inference.
        This removes all the non-detections from the observations dataframe.

        :return:
        """
        if self.population:
            all_data = self.list_of_observations
            events = len(self.parameters)
            dfs = []
            for x in range(events):
                df = all_data[x]
                df = df[df.detected != 0]
                dfs.append(df)
            return dfs
        else:
            df = self.observations
            df = df[df.detected != 0]
            return df

    @property
    def survey_radius(self):
        """
        Convert the circular field of view to a radius in radians.
        :return: survey_radius in radians
        """
        survey_fov_sqrad = self.survey_fov_sqdeg*(np.pi/180.0)**2
        survey_radius = np.sqrt(survey_fov_sqrad/np.pi)
        # survey_radius = np.sqrt(self.survey_fov_sqdeg*((np.pi/180.0)**2.0)/np.pi)
        return survey_radius

    @property
    def t0_transient(self):
        """
        :return: The start time of the transient in MJD
        """
        if 't0' in self.parameters:
            return self.parameters['t0']
        else:
            return self.parameters['t0_mjd_transient']

    @property
    def RA(self):
        """
        :return: The RA of the transient in radians. Draw randomly from the pointings database if not supplied.
        """
        if 'ra' in self.parameters:
            RA = self.parameters['ra'].values
        else:
            RA = self.pointings_database['_ra'].sample(len(self.parameters)).values
        return RA

    @property
    def DEC(self):
        """
        :return: The DEC of the transient in radians. Draw randomly from the pointings database if not supplied.
        """
        if 'dec' in self.parameters:
            dec = self.parameters['dec'].values
        else:
            dec = self.pointings_database['_dec'].sample(len(self.parameters)).values
        return dec

    @property
    def min_dec(self):
        """
        :return: Minimum dec of the survey in radians
        """
        df = self.pointings_database
        return np.min(df['_dec'])

    @property
    def max_dec(self):
        """
        :return: Maximum dec of the survey in radians
        """
        df = self.pointings_database
        return np.max(df['_dec'])

    @property
    def start_mjd(self):
        """
        :return: Start of the survey in MJD
        """
        df = self.pointings_database
        return np.min(df['expMJD'])

    @property
    def end_mjd(self):
        """
        :return: End of the survey in MJD
        """
        df = self.pointings_database
        return np.max(df['expMJD'])

    def _get_unique_reference_fluxes(self):
        """
        :return: Get the unique reference fluxes for each filter in the survey
        """
        unique_bands = self.pointings_database.filters.unique()
        ref_flux = redback.utils.bands_to_reference_flux(unique_bands)
        return ref_flux

    def _make_sncosmo_wrapper_for_user_model(self):
        """

        Function to wrap user models into sncosmo model format for full functionality.
        :return: sncosmo source
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
        return source

    def _survey_to_table_name_lookup(self, survey):
        """

        A lookup table to get the name of the pointings database for a given survey.
        :param survey: name of the survey
        :return: path to the pointings database
        """
        survey_to_table = {'Rubin_10yr_baseline': 'rubin_baseline_v3.0_10yrs.tar.gz',
                           'Rubin_10yr_morez': 'rubin_morez.tar.gz',
                           'Rubin_10yr_lessweight': 'rubin_lessweight.tar.gz',
                           'ztf': 'ztf.tar.gz',
                           'roman': 'roman.tar.gz'}
        return survey_to_table[survey]

    def _find_time_overlaps(self):
        """
        Find the time indices of the pointings database that overlap with the transient.

        :return: indices of the pointings database that overlap with the transient.
        """
        pointing_times = self.pointings_database[['expMJD']].values.flatten()
        if self.population:
            condition_1 = pointing_times >= self.t0_transient.values[:, None] - self.obs_buffer
            condition_2 = pointing_times <= self.end_transient_time.values[:, None]
        else:
            condition_1 = pointing_times >= self.t0_transient.values - self.obs_buffer
            condition_2 = pointing_times <= self.end_transient_time.values
        mask = np.logical_and(condition_1, condition_2)
        if self.population:
            return mask
        else:
            time_indices = np.where(mask)
            return time_indices[0]


    def _find_sky_overlaps(self):
        """
        Find the sky indices of the pointings database that overlap with the transient.
        """
        pointings_sky_pos = np.column_stack((self.pointings_database['_ra'].values, self.pointings_database['_dec'].values))
        transient_sky_pos = np.column_stack((self.parameters['ra'].values, self.parameters['dec'].values))

        transient_sky_pos_3D = np.vstack([np.cos(transient_sky_pos[:,0]) * np.cos(transient_sky_pos[:,1]),
                                          np.sin(transient_sky_pos[:,0]) * np.cos(transient_sky_pos[:,1]),
                                          np.sin(transient_sky_pos[:,1])]).T
        pointings_sky_pos_3D = np.vstack([np.cos(pointings_sky_pos[:, 0]) * np.cos(pointings_sky_pos[:,1]),
                                          np.sin(pointings_sky_pos[:,0]) * np.cos(pointings_sky_pos[:,1]),
                                          np.sin(pointings_sky_pos[:,1])]).T
        # law of cosines to compute 3D distance
        max_3D_dist = np.sqrt(2. - 2. * np.cos(self.survey_radius))
        survey_tree = KDTree(pointings_sky_pos_3D)
        if self.population:
            overlap_indices = survey_tree.query_ball_point(x=transient_sky_pos_3D, r=max_3D_dist)
        else:
            overlap_indices = survey_tree.query_ball_point(x=transient_sky_pos_3D.T.flatten(), r=max_3D_dist)
        return overlap_indices


    def _make_observation_single(self, overlapping_database, t0_transient, sncosmo_model):
        """
        Calculate properties of the transient at the overlapping pointings for a single transient.

        :param overlapping_database:
        :return: Dataframe of observations including non-detections/upper limits
        """
        times = overlapping_database['expMJD'].values - t0_transient
        filters = overlapping_database['filter'].values
        magnitude = sncosmo_model.bandmag(phase=times, band=filters, magsys='AB')
        flux = redback.utils.bandpass_magnitude_to_flux(magnitude=magnitude, bands=filters)
        ref_flux = redback.utils.bands_to_reference_flux(filters)
        bandflux_errors = redback.utils.bandflux_error_from_limiting_mag(overlapping_database['fiveSigmaDepth'].values,
                                                                         ref_flux)
        if self.add_source_noise:
            bandflux_errors = np.sqrt(bandflux_errors**2 + self.source_noise_factor*flux**2)
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
        mask_snr = snr < self.snr_threshold
        detected = np.ones(len(observation_dataframe))
        detected[mask] = 0
        detected[mask_snr] = 0
        observation_dataframe['detected'] = detected
        observation_dataframe['limiting_magnitude'] = overlapping_database['fiveSigmaDepth'].values
        return observation_dataframe

    def _make_observations(self):
        """
        Calculate properties of the transient at the overlapping pointings for a single transient.
        :return: Dataframe of observations including non-detections/upper limits
        """
        overlapping_sky_indices = self._find_sky_overlaps()
        overlapping_time_indices = self._find_time_overlaps()

        space_set = set(overlapping_sky_indices)
        time_set = set(overlapping_time_indices)
        time_space_overlap = list(space_set.intersection(time_set))
        overlapping_database_iter = self.pointings_database.iloc[time_space_overlap]
        overlapping_database_iter = overlapping_database_iter.sort_values(by=['expMJD'])
        dataframe = self._make_observation_single(overlapping_database_iter,
                                                  t0_transient=self.t0_transient.values,
                                                  sncosmo_model=self.sncosmo_model)
        return dataframe

    def _make_observations_for_population(self):
        """
        Calculate properties of the transient at the overlapping pointings for a transient population.
        :return: Dataframe of observations including non-detections/upper limits
        """
        dfs = []
        overlapping_sky_indices = self._find_sky_overlaps()
        time_mask = self._find_time_overlaps()
        for x in range(len(self.parameters)):
            overlapping_time_indices = np.where(time_mask[x])[0]
            space_set = set(overlapping_sky_indices[x])
            time_set = set(overlapping_time_indices)
            time_space_overlap = list(space_set.intersection(time_set))
            overlapping_database_iter = self.pointings_database.iloc[time_space_overlap]
            overlapping_database_iter = overlapping_database_iter.sort_values(by=['expMJD'])
            dataframe = self._make_observation_single(overlapping_database_iter,
                                                      t0_transient=self.t0_transient.iloc[x],
                                                      sncosmo_model=self.all_sncosmo_models[x])
            dfs.append(dataframe)
        return dfs


    def save_transient_population(self, transient_names=None, **kwargs):
        """
        Save the transient population to a csv file.
        This will save the full observational dataframe including non-detections etc.
        This will save the data to a folder called 'simulated'
        with the name of the transient and a csv file of the injection parameters

        :param transient_names: list of transient names. Default is None which will label transients as event_0, etc
        :param kwargs: kwargs for the save_transient function
        :param injection_file_path: path to save the injection file
        :return: None
        """
        injection_file_name = kwargs.get('injection_file_path', 'simulated/population_injection_parameters.csv')
        if transient_names is None:
            transient_names = ['event_' + str(x) for x in range(len(self.list_of_observations))]
        bilby.utils.check_directory_exists_and_if_not_mkdir('simulated')
        self.parameters.to_csv(injection_file_name, index=False)
        for ii, transient_name in enumerate(transient_names):
            transient = self.list_of_observations[ii]
            transient.to_csv('simulated/' + transient_name + '.csv', index=False)

    def save_transient(self, name):
        """
        Save the transient observations to a csv file.
        This will save the full observational dataframe including non-detections etc.
        This will save the data to a folder called 'simulated'
        with the name of the transient and a csv file of the injection parameters

        :param name: name to save transient.
        """
        bilby.utils.check_directory_exists_and_if_not_mkdir('simulated')
        path = 'simulated/' + name + '.csv'
        injection_path = 'simulated/' + name + '_injection_parameters.csv'
        self.observations.to_csv(path, index=False)
        self.parameters.to_csv(injection_path, index=False)

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
    def __init__(self, model, prior, rate, survey_start_date, survey_duration, pointings_database=None,
                 survey='Rubin_10yr_baseline',sncosmo_kwargs=None, obs_buffer=5.0, survey_fov_sqdeg=9.6,
                 snr_threshold=5, end_transient_time=1000, add_source_noise=False, model_kwargs=None, **kwargs):
        """
        Simulate a full optical survey. This requires a rate and a prior for the population.
        The rate is used to draw events in a period of time, placing them isotropically on the sky and uniform in comoving volume.
        The prior is used to draw the parameters of the individual events.
        We can then simulate observations of all these events and understand the rate of detections etc.

        :param model: String corresponding to redback model or a python function that can evaluate an SED.
        :param prior: A redback prior corresponding to the model.
            The prior on the redshift is forced to be uniform in comoving volume. With maximum what the user sets in their prior.
        :param rate: Rate of the population in Gpc^-3 yr^-1
        :param survey_start_date: Start date of the survey in MJD.
        :param survey_duration: Duration of the survey in years.
            This can be set arbitrarily high if one wants to look at detection efficiencies.
            Or to a real number if wanting to look at a volume/flux limited survey.
        :param pointings_database: A pandas DataFrame containing the pointings of the survey.
        :param survey: String corresponding to the survey name. This is used to look up the pointings database.
        :param sncosmo_kwargs: Any kwargs to be passed to SNcosmo.
            SNcosmo is used to evaluate the bandpass magnitudes in different bands.
        :param obs_buffer: A observation buffer in days to add to the start of the transient
            to allow for non-detections. Default is 5 days
        :param snr_threshold: SNR threshold for detection. Default is 5.
        :param end_transient_time: End time of the transient in days. Default is 1000 days.
            Note that SNCosmo will extrapolate past when the transient model evaluates the SED so these should really be the same.
        :param add_source_noise: Boolean. If True, add an extra noise in quadrature to the limiting mag noise.
            The factor is a multiple of the model flux i.e. noise = (skynoise**2 + (model_flux*source_noise)**2)**0.5
        :param model_kwargs: Dictionary of kwargs to be passed to the model.
        :param kwargs: Dictionary of additional kwargs
        :param cosmology: Cosmology to use. Default is Planck18.
            Users can pass their own cosmology class here as long as it works like astropy.cosmology.
            Users should ensure they use the same cosmology in the model. Or deliberately choose not to.
        :param source_noise: Float. Factor to multiply the model flux by to add an extra noise
            in quadrature to the limiting mag noise. Default value is 0.02, disabled by default.
        """
        self.rate = rate * uu.Gpc**-3 * uu.yr**-1
        self.prior = prior
        self.survey_start_date = survey_start_date
        self.survey_duration = survey_duration * uu.yr
        cosmology = kwargs.get('cosmology',cosmo)
        self.horizon_redshift = self.prior['redshift'].maximum
        self.horizon_distance = cosmology.luminosity_distance(self.horizon_redshift).to(uu.Mpc)
        self.number_of_events = np.random.poisson(self.rate_per_sec * self.survey_duration_seconds)
        self.prior['redshift'] = bilby.gw.prior.UniformSourceFrame(minimum=0, maximum=self.horizon_redshift,
                                                                   name='redshift', cosmology='Planck18')
        self.prior['ra'] = bilby.core.prior.Uniform(minimum=0, maximum=2*np.pi, name='ra', latex_label='$\\mathrm{RA}$',
                                   unit=None, boundary='periodic')
        self.prior['dec'] = bilby.core.prior.Cosine(minimum=-np.pi/2, maximum=np.pi/2., name='dec',
                                  latex_label='$\\mathrm{DEC}$', unit=None, boundary=None)
        parameters = prior.sample(self.number_of_events)
        _ra = self.prior['ra'].sample(self.number_of_events)
        _dec = self.prior['dec'].sample(self.number_of_events)
        _event_times = self.get_event_times()
        parameters['ra'] = _ra
        parameters['dec'] = _dec
        parameters['t0_mjd_transient'] = _event_times
        super().__init__(model=model, parameters=parameters, pointings_database=pointings_database,
                         survey=survey, sncosmo_kwargs=sncosmo_kwargs,
                         obs_buffer=obs_buffer,survey_fov_sqdeg=survey_fov_sqdeg,
                         snr_threshold=snr_threshold, end_transient_time=end_transient_time,
                         add_source_noise=add_source_noise,
                         population=True, model_kwargs=model_kwargs, **kwargs)

    @property
    def rate_per_sec(self):
        rate_per_year = self.rate * self.horizon_distance.to(uu.Gpc)**3 * 4./3. * np.pi
        return rate_per_year.to(uu.s**-1)

    @property
    def survey_duration_seconds(self):
        return self.survey_duration.to(uu.s)

    @property
    def time_window(self):
        time_window = [self.survey_start_date, self.survey_start_date + self.survey_duration.to(uu.day).value]
        return time_window

    def get_event_times(self):
        if self.number_of_events == 1:
            event_id = random.random() * self.survey_duration_seconds.to(uu.day).value + self.survey_start_date
        elif self.number_of_events > 1:
            event_id = []
            for j in range(self.number_of_events):
                event_id.append(random.random())
            event_id.sort()
            event_id = np.array(event_id)
            for j in range(self.number_of_events):
                event_id[j] *= self.survey_duration_seconds.to(uu.day).value
                event_id[j] += self.survey_start_date
        else:
            event_id = [self.survey_start_date]
        return event_id

    def save_survey(self, survey=None, **kwargs):
        """
        Save the transient population to a csv file.
        This will save the full observational dataframe including non-detections etc.
        This will save the data to a folder called 'simulated'
        with the name of the transient and a csv file of the injection parameters

        :param transient_names: list of transient names. Default is None which will label transients as event_0, etc
        :param kwargs: kwargs for the save_transient function
        :param injection_file_path: path to save the injection file
        :return: None
        """
        injection_file_name = kwargs.get('injection_file_path', 'simulated_survey/population_injection_parameters.csv')
        if survey is None:
            survey_name = 'survey'
        else:
            survey_name = survey
        transient_names = [survey_name + '_event_' + str(x) for x in range(len(self.list_of_observations))]
        bilby.utils.check_directory_exists_and_if_not_mkdir('simulated_survey')
        self.parameters.to_csv(injection_file_name, index=False)
        for ii, transient_name in enumerate(transient_names):
            transient = self.list_of_observations[ii]
            transient.to_csv('simulated_survey/' + transient_name + '.csv', index=False)
