import numpy as np
from redback.sed import RedbackTimeSeriesSource
import redback
import pandas as pd
from redback.utils import logger, calc_flux_density_error_from_monochromatic_magnitude, calc_flux_density_from_ABmag
from itertools import repeat
import astropy.units as uu
from scipy.spatial import KDTree
from scipy import stats, integrate, interpolate
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
            injection_kwargs['output_format'] = model_kwargs['output_format']
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
            if kwargs['redback_compatible_model']:
                model_kwargs['output_format'] = 'sncosmo_source'
                logger.info("Model is consistent with redback model format. Using simplified model wrapper.")
                model_end_time = model_kwargs.get('end_time', 400.0)
                _time_array = np.linspace(0.1, model_end_time, 1000)
                sncosmomodel = self.model(_time_array, **parameters, **model_kwargs)
                self.sncosmo_model = sncosmomodel
            else:
                logger.info('Model is inconsistent with redback model format. Making a custom wrapper for this model')
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
        # Ensure sncosmo_kwargs is a dictionary
        if self.sncosmo_kwargs is None:
            self.sncosmo_kwargs = {}

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

        :param overlapping_database: Database of overlapping pointings
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
        observation_dataframe['flux_limit'] = redback.utils.bandpass_magnitude_to_flux(
            overlapping_database['fiveSigmaDepth'].values, filters)
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


class TransientPopulation(object):
    """
    Container class for transient populations with analysis capabilities
    """
    def __init__(self, parameters, metadata=None, light_curves=None):
        """
        Initialize a TransientPopulation container

        :param parameters: DataFrame of transient parameters
        :param metadata: dict, Optional metadata about the population (rate, cosmology, etc.)
        :param light_curves: list, Optional list of light curve data for each transient
        """
        self.parameters = parameters
        self.metadata = metadata if metadata is not None else {}
        self.light_curves = light_curves
        self.n_transients = len(parameters)

    def __len__(self):
        return self.n_transients

    def __repr__(self):
        return f"TransientPopulation({self.n_transients} transients)"

    @property
    def redshifts(self):
        """Get redshift array"""
        if 'redshift' in self.parameters.columns:
            return self.parameters['redshift'].values
        return None

    @property
    def detected(self):
        """Get subset of detected transients"""
        if 'detected' in self.parameters.columns:
            return self.parameters[self.parameters['detected'] == True]
        return self.parameters

    @property
    def detection_fraction(self):
        """Calculate detection fraction"""
        if 'detected' in self.parameters.columns:
            return np.sum(self.parameters['detected']) / len(self.parameters)
        return 1.0  # Assume all detected if not specified

    def get_redshift_distribution(self, bins=20):
        """
        Get redshift distribution

        :param bins: Number of bins for histogram
        :return: tuple of (bin_edges, counts, bin_centers)
        """
        if self.redshifts is None:
            logger.warning("No redshift information in population")
            return None, None, None

        counts, edges = np.histogram(self.redshifts, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        return edges, counts, centers

    def get_parameter_distribution(self, param_name, bins=20):
        """
        Get distribution of any parameter

        :param param_name: Name of parameter
        :param bins: Number of bins
        :return: tuple of (bin_edges, counts, bin_centers)
        """
        if param_name not in self.parameters.columns:
            logger.warning(f"Parameter {param_name} not in population")
            return None, None, None

        values = self.parameters[param_name].values
        counts, edges = np.histogram(values, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        return edges, counts, centers

    def save(self, filename='population.csv', save_metadata=True):
        """
        Save population to file

        :param filename: Output filename
        :param save_metadata: Whether to save metadata as well
        """
        bilby.utils.check_directory_exists_and_if_not_mkdir('populations')
        path = f'populations/{filename}'
        self.parameters.to_csv(path, index=False)
        logger.info(f"Saved population to {path}")

        if save_metadata and self.metadata:
            import json
            metadata_path = path.replace('.csv', '_metadata.json')
            with open(metadata_path, 'w') as f:
                # Convert numpy types to python types for JSON serialization
                clean_metadata = {}
                for key, val in self.metadata.items():
                    if isinstance(val, (np.integer, np.floating)):
                        clean_metadata[key] = float(val)
                    elif isinstance(val, np.ndarray):
                        clean_metadata[key] = val.tolist()
                    else:
                        clean_metadata[key] = val
                json.dump(clean_metadata, f, indent=2)
            logger.info(f"Saved metadata to {metadata_path}")

    @classmethod
    def load(cls, filename='population.csv'):
        """
        Load population from file

        :param filename: Input filename
        :return: TransientPopulation object
        """
        path = f'populations/{filename}'
        parameters = pd.read_csv(path)

        # Try to load metadata
        metadata = None
        metadata_path = path.replace('.csv', '_metadata.json')
        if os.path.exists(metadata_path):
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

        return cls(parameters, metadata=metadata)


class PopulationSynthesizer(object):
    """
    Advanced population synthesis with realistic distributions and selection effects

    This class provides survey-agnostic population synthesis with:
    - Volumetric rate evolution with redshift
    - Realistic parameter sampling from priors
    - Optional selection effects and detection efficiency
    - Cosmology-aware redshift sampling
    - Rate inference capabilities
    """

    def __init__(self, model, prior=None, rate=1e-6, rate_evolution='constant',
                 cosmology='Planck18', seed=42):
        """
        Initialize PopulationSynthesizer

        :param model: String corresponding to redback model or callable function
        :param prior: bilby.core.prior.PriorDict or string name of prior file
            If None, will attempt to load default prior for the model
        :param rate: Volumetric rate in Gpc^-3 yr^-1 at z=0 (for constant rate)
            or callable function rate(z) for redshift-dependent rates
        :param rate_evolution: String specifying rate evolution model
            Options: 'constant', 'powerlaw', 'sfr_like', or callable
            - 'constant': R(z) = R0
            - 'powerlaw': R(z) = R0 * (1+z)^alpha (requires alpha in metadata)
            - 'sfr_like': R(z) follows star formation rate (Madau & Dickinson 2014)
        :param cosmology: Cosmology to use ('Planck18', 'Planck15', etc.) or astropy cosmology object
        :param seed: Random seed for reproducibility
        """
        # Set model
        if isinstance(model, str):
            if model in redback.model_library.all_models_dict:
                self.model = redback.model_library.all_models_dict[model]
                self.model_name = model
            else:
                raise ValueError(f"Model {model} not found in redback model library")
        elif callable(model):
            self.model = model
            self.model_name = 'custom'
        else:
            raise ValueError("Model must be string or callable")

        # Set prior
        if prior is None:
            # Try to load default prior for model
            if self.model_name != 'custom':
                self.prior = redback.priors.get_priors(self.model_name)
                logger.info(f"Loaded default prior for {self.model_name}")
            else:
                raise ValueError("Must provide prior for custom models")
        elif isinstance(prior, str):
            self.prior = redback.priors.get_priors(prior)
        elif isinstance(prior, bilby.core.prior.PriorDict):
            self.prior = prior
        else:
            raise ValueError("Prior must be None, string, or bilby.core.prior.PriorDict")

        # Set cosmology
        if isinstance(cosmology, str):
            from astropy.cosmology import default_cosmology
            self.cosmology = default_cosmology.get_cosmology_from_string(cosmology)
            self.cosmology_name = cosmology
        else:
            self.cosmology = cosmology
            self.cosmology_name = 'custom'

        # Set rate and rate evolution
        if callable(rate):
            self.rate_function = rate
            self.base_rate = None
        else:
            self.base_rate = rate * uu.Gpc**-3 * uu.yr**-1
            self.rate_function = self._create_rate_function(rate_evolution)

        self.rate_evolution = rate_evolution
        self.seed = seed
        self.rng = np.random.RandomState(seed=seed)
        random.seed(seed)

    def _create_rate_function(self, evolution_model):
        """
        Create rate evolution function

        :param evolution_model: String or callable defining evolution
        :return: Function R(z) in units of Gpc^-3 yr^-1
        """
        if evolution_model == 'constant':
            def rate_func(z):
                return self.base_rate.value * np.ones_like(z)
            return rate_func

        elif evolution_model == 'powerlaw':
            # R(z) = R0 * (1+z)^alpha
            # Default alpha=2.7 similar to GW merger rate evolution
            alpha = 2.7
            def rate_func(z):
                return self.base_rate.value * (1 + z)**alpha
            return rate_func

        elif evolution_model == 'sfr_like':
            # Star formation rate evolution from Madau & Dickinson 2014
            def rate_func(z):
                return self.base_rate.value * 0.015 * (1+z)**2.7 / (1 + ((1+z)/2.9)**5.6)
            return rate_func

        elif callable(evolution_model):
            return evolution_model

        else:
            raise ValueError(f"Unknown rate evolution model: {evolution_model}")

    def _sample_redshifts(self, n_events, z_max=None):
        """
        Sample redshifts from volumetric rate with cosmology

        :param n_events: Number of events to sample
        :param z_max: Maximum redshift (if None, use prior maximum)
        :return: Array of redshifts
        """
        if z_max is None:
            if 'redshift' in self.prior:
                z_max = self.prior['redshift'].maximum
            else:
                z_max = 2.0  # Default
                logger.warning(f"No redshift in prior, using z_max={z_max}")

        # Create PDF: dN/dz ∝ R(z) * dV_c/dz
        z_array = np.linspace(0, z_max, 1000)

        # Differential comoving volume
        dVc_dz = self.cosmology.differential_comoving_volume(z_array).value  # Mpc^3 sr^-1

        # Rate at each redshift
        rate_z = self.rate_function(z_array)  # Gpc^-3 yr^-1

        # PDF (unnormalized)
        pdf = rate_z * dVc_dz / (1 + z_array)  # Include (1+z)^-1 for time dilation
        pdf /= np.trapz(pdf, z_array)  # Normalize

        # Sample from this distribution
        cdf = np.cumsum(pdf)
        cdf = cdf / cdf[-1]

        # Inverse CDF sampling
        u = self.rng.uniform(0, 1, n_events)
        redshifts = np.interp(u, cdf, z_array)

        return redshifts

    def _sample_intrinsic_params(self, n_events):
        """
        Sample intrinsic parameters from prior

        :param n_events: Number of events
        :return: DataFrame of parameters
        """
        # Sample from prior, excluding redshift (we handle that separately)
        params = self.prior.sample(n_events)

        return params

    def _sample_sky_positions(self, n_events):
        """
        Sample isotropic sky positions (RA, DEC)

        :param n_events: Number of events
        :return: tuple of (ra, dec) arrays in radians
        """
        # RA: uniform [0, 2π]
        ra = self.rng.uniform(0, 2 * np.pi, n_events)

        # DEC: cos(dec) uniform [-π/2, π/2]
        # Sample from uniform in sin(dec) then take arcsin
        dec = np.arcsin(self.rng.uniform(-1, 1, n_events))

        return ra, dec

    def _sample_event_times(self, n_events, time_range):
        """
        Sample event times uniformly in time range

        For a Poisson process, event times are uniform in the observation window

        :param n_events: Number of events
        :param time_range: Tuple of (start_time, end_time) in MJD
        :return: Array of event times in MJD
        """
        start_time, end_time = time_range
        event_times = self.rng.uniform(start_time, end_time, n_events)
        return np.sort(event_times)

    def _calculate_expected_events(self, n_years, z_max):
        """
        Calculate expected number of events from volumetric rate

        The observed rate must account for cosmological time dilation via (1+z)^-1

        :param n_years: Observation time in years
        :param z_max: Maximum redshift
        :return: Expected number of events
        """
        # Integrate rate over volume and time
        z_array = np.linspace(0, z_max, 100)
        dVc_dz = self.cosmology.differential_comoving_volume(z_array).value
        rate_z = self.rate_function(z_array)

        # Total rate over all space (4π sr full sky)
        # Include (1+z)^-1 for time dilation - events at higher z occur slower in observer frame
        integrand = rate_z * dVc_dz / (1 + z_array) * 4 * np.pi / (1e9)**3  # Convert Mpc^3 to Gpc^3
        total_rate_per_year = np.trapz(integrand, z_array)  # Events per year in observer frame

        expected_events = total_rate_per_year * n_years
        return expected_events

    def generate_population(self, n_years=10, n_events=None, z_max=None,
                           time_range=None, include_sky_position=True):
        """
        Generate population parameters according to volumetric rate and priors

        This method generates a pure parameter DataFrame that can be passed to
        any simulation tool (redback SimulateOpticalTransient, custom tools, etc.)

        :param n_years: Number of years to observe (used with rate to calculate n_events)
        :param n_events: Fixed number of events (overrides n_years if specified)
        :param z_max: Maximum redshift (if None, use prior maximum)
        :param time_range: Tuple of (start_time_mjd, end_time_mjd) for event times
            If None, uses (60000, 60000 + n_years*365.25)
        :param include_sky_position: Whether to add RA/DEC (isotropic)
        :return: pandas DataFrame with all parameters (including redshift, ra, dec, t0_mjd_transient)
        """
        logger.info(f"Generating population parameters for {self.model_name}")

        # Determine redshift range
        if z_max is None:
            z_max = self.prior['redshift'].maximum if 'redshift' in self.prior else 2.0

        # Determine number of events
        if n_events is None:
            # Calculate from rate (Poisson draw)
            expected_events = self._calculate_expected_events(n_years, z_max)
            n_events = self.rng.poisson(expected_events)
            logger.info(f"Expected {expected_events:.1f} events in {n_years} years, drew {n_events}")
        else:
            logger.info(f"Generating fixed {n_events} events")

        if n_events == 0:
            logger.warning("No events generated!")
            return pd.DataFrame()

        # Sample redshifts (weighted by rate and volume)
        redshifts = self._sample_redshifts(n_events, z_max=z_max)

        # Sample intrinsic parameters from priors
        params_df = self._sample_intrinsic_params(n_events)

        # Add redshift
        params_df['redshift'] = redshifts

        # Add luminosity distance
        params_df['luminosity_distance'] = self.cosmology.luminosity_distance(redshifts).value  # Mpc

        # Add sky positions (isotropic)
        if include_sky_position:
            ra, dec = self._sample_sky_positions(n_events)
            params_df['ra'] = ra
            params_df['dec'] = dec

        # Add event times (t0)
        if time_range is None:
            # Default: start at MJD 60000, span n_years
            time_range = (60000.0, 60000.0 + n_years * 365.25)

        event_times = self._sample_event_times(n_events, time_range)
        params_df['t0_mjd_transient'] = event_times

        logger.info(f"Generated {n_events} events with redshifts [{redshifts.min():.3f}, {redshifts.max():.3f}]")

        return params_df

    def apply_detection_criteria(self, population, detection_function, **kwargs):
        """
        Apply custom detection criteria to a population

        This is a flexible post-processing method that allows users to define
        their own detection logic based on the population parameters.

        :param population: pandas DataFrame from generate_population() or TransientPopulation
        :param detection_function: Callable that takes (row, **kwargs) and returns bool or float [0,1]
            - If returns bool: True = detected, False = not detected
            - If returns float: Interpreted as detection probability, stochastically applied
            - Function signature: detection_function(row, **kwargs) -> bool or float
        :param kwargs: Additional arguments passed to detection_function
        :return: pandas DataFrame with added 'detected' and optionally 'detection_probability' columns
        """
        logger.info("Applying custom detection criteria...")

        # Extract DataFrame if TransientPopulation
        if isinstance(population, TransientPopulation):
            params_df = population.parameters.copy()
        elif isinstance(population, pd.DataFrame):
            params_df = population.copy()
        else:
            raise ValueError("population must be DataFrame or TransientPopulation")

        detected = []
        detection_probs = []

        for idx in range(len(params_df)):
            row = params_df.iloc[idx]
            result = detection_function(row, **kwargs)

            # Handle boolean vs probability return
            if isinstance(result, bool):
                detected.append(result)
                detection_probs.append(1.0 if result else 0.0)
            elif isinstance(result, (int, float)):
                # Interpret as probability
                detection_probs.append(float(result))
                # Stochastically apply
                is_detected = self.rng.uniform() < result
                detected.append(is_detected)
            else:
                raise ValueError(f"detection_function must return bool or float, got {type(result)}")

        params_df['detection_probability'] = detection_probs
        params_df['detected'] = detected

        n_detected = np.sum(detected)
        logger.info(f"Detected {n_detected}/{len(params_df)} transients ({100*n_detected/len(params_df):.1f}%)")

        return params_df

    def _calculate_detection_probability(self, params, survey_config):
        """
        Calculate detection probability for a transient

        :param params: Dictionary of transient parameters
        :param survey_config: Dictionary with survey parameters
            Must include 'limiting_mag' and 'bands' at minimum
        :return: Detection probability (0 to 1)
        """
        # This is a simplified model - can be extended
        # For now, use peak magnitude vs limiting magnitude

        if 'limiting_mag' not in survey_config:
            logger.warning("No limiting_mag in survey_config, assuming all detected")
            return 1.0

        # Generate light curve to find peak magnitude
        # This is expensive - in practice you might want to cache or approximate
        times = np.linspace(0, 100, 100)  # days

        try:
            # Try to get magnitude
            if 'bands' in survey_config:
                band = survey_config['bands'][0]
                model_kwargs = {'bands': np.array([band] * len(times)),
                              'output_format': 'magnitude'}
                params_with_kwargs = {**params, **model_kwargs}
                mags = self.model(times, **params_with_kwargs)
                peak_mag = np.min(mags[np.isfinite(mags)])
            else:
                # Assume flux output, convert to magnitude
                output = self.model(times, **params)
                # Simple approximation for detection
                peak_flux = np.max(output)
                if peak_flux <= 0:
                    return 0.0
                return 1.0  # Simplified
        except:
            # If we can't evaluate, assume detectable
            logger.debug("Could not evaluate model for detection probability")
            return 1.0

        # Sigmoid function for detection efficiency
        lim_mag = survey_config['limiting_mag']
        # 50% detection at limiting magnitude, sharp falloff
        detection_prob = 1.0 / (1.0 + np.exp(2.0 * (peak_mag - lim_mag)))

        return detection_prob

    def simulate_population(self, n_years=10, n_events=None, z_max=None,
                           time_range=None, include_selection_effects=False,
                           survey_config=None, include_lightcurves=False, model_kwargs=None):
        """
        Convenience method to generate and optionally filter a transient population

        This is a wrapper around generate_population() and apply_detection_criteria()
        that returns a TransientPopulation object. For more control, use the
        individual methods directly.

        :param n_years: Number of years to simulate (used to calculate expected events from rate)
        :param n_events: Fixed number of events (overrides n_years if specified)
        :param z_max: Maximum redshift to sample
        :param time_range: Tuple of (start_mjd, end_mjd) for event times
        :param include_selection_effects: Whether to apply detection efficiency cuts
        :param survey_config: Dictionary with survey parameters for selection effects
            Example: {'limiting_mag': 22.5, 'bands': ['lsstr'], 'area_sqdeg': 18000}
        :param include_lightcurves: Whether to generate light curves (expensive)
        :param model_kwargs: Additional kwargs to pass to model
        :return: TransientPopulation object
        """
        logger.info(f"Simulating transient population with {self.model_name}")

        # Generate population parameters
        params_df = self.generate_population(
            n_years=n_years,
            n_events=n_events,
            z_max=z_max,
            time_range=time_range,
            include_sky_position=True
        )

        if len(params_df) == 0:
            return TransientPopulation(pd.DataFrame(), metadata={'n_years': n_years})

        # Apply selection effects if requested
        if include_selection_effects and survey_config is not None:
            # Use built-in detection probability calculator
            def detection_func(row, config):
                return self._calculate_detection_probability(row.to_dict(), config)

            params_df = self.apply_detection_criteria(
                params_df,
                detection_function=detection_func,
                config=survey_config
            )

        # Generate light curves if requested (expensive!)
        light_curves = None
        if include_lightcurves:
            logger.info("Generating light curves (this may take a while)...")
            light_curves = []
            for idx in range(len(params_df)):
                params = params_df.iloc[idx].to_dict()
                # Generate light curve
                times = np.linspace(0, 100, 100)
                if model_kwargs is None:
                    model_kwargs = {}
                lc = self.model(times, **params, **model_kwargs)
                light_curves.append({'times': times, 'flux': lc})

        # Create metadata
        metadata = {
            'model': self.model_name,
            'n_years': n_years,
            'n_events': len(params_df),
            'rate': self.base_rate.value if self.base_rate is not None else 'custom',
            'rate_evolution': self.rate_evolution,
            'cosmology': self.cosmology_name,
            'include_selection_effects': include_selection_effects,
            'seed': self.seed
        }

        if survey_config is not None:
            metadata['survey_config'] = survey_config

        return TransientPopulation(params_df, metadata=metadata, light_curves=light_curves)

    def infer_rate(self, observed_sample, efficiency_function=None, z_bins=10,
                   prior_range=(1e-8, 1e-4), method='maximum_likelihood'):
        """
        Infer volumetric rate from observed sample

        This accounts for selection effects via the efficiency function

        :param observed_sample: DataFrame or TransientPopulation of observed transients
            Must have 'redshift' column
        :param efficiency_function: Callable epsilon(z) giving detection efficiency vs redshift
            If None, assumes perfect detection (epsilon=1)
        :param z_bins: Number of redshift bins for rate estimation
        :param prior_range: Tuple of (min, max) for rate prior in Gpc^-3 yr^-1
        :param method: 'maximum_likelihood' or 'bayesian'
        :return: Dictionary with rate estimate and uncertainties
        """
        logger.info("Inferring volumetric rate from observed sample...")

        # Extract redshifts
        if isinstance(observed_sample, TransientPopulation):
            redshifts = observed_sample.redshifts
        elif isinstance(observed_sample, pd.DataFrame):
            if 'redshift' not in observed_sample.columns:
                raise ValueError("observed_sample must have 'redshift' column")
            redshifts = observed_sample['redshift'].values
        else:
            redshifts = np.array(observed_sample)

        n_obs = len(redshifts)
        z_max = np.max(redshifts)

        # Create redshift bins
        z_edges = np.linspace(0, z_max, z_bins + 1)
        z_centers = (z_edges[:-1] + z_edges[1:]) / 2

        # Count observed events in each bin
        n_in_bins, _ = np.histogram(redshifts, bins=z_edges)

        # Calculate expected number in each bin for unit rate
        # N_expected(z) = R * integral[dVc/dz * epsilon(z) * T_obs, z_i, z_i+1]

        if efficiency_function is None:
            # Perfect detection
            efficiency_function = lambda z: np.ones_like(z)

        # Assume 1 year observation time (rate will scale)
        T_obs = 1.0  # years

        # Expected events per bin for R=1 Gpc^-3 yr^-1
        expected_per_unit_rate = np.zeros(z_bins)
        for i in range(z_bins):
            z_range = np.linspace(z_edges[i], z_edges[i+1], 20)
            if len(z_range) > 1:
                dVc_dz = self.cosmology.differential_comoving_volume(z_range).value
                eff = efficiency_function(z_range)
                # 4π sr for full sky, convert to Gpc^3
                integrand = dVc_dz * eff * 4 * np.pi / (1e9)**3 / (1 + z_range)
                expected_per_unit_rate[i] = np.trapz(integrand, z_range) * T_obs

        # Maximum likelihood estimate
        # For Poisson: R_ML = N_obs / sum(expected_per_unit_rate)
        total_expected_per_unit = np.sum(expected_per_unit_rate)
        rate_ml = n_obs / total_expected_per_unit if total_expected_per_unit > 0 else 0

        # Uncertainty (Poisson)
        rate_uncertainty = np.sqrt(n_obs) / total_expected_per_unit if total_expected_per_unit > 0 else 0

        results = {
            'rate_ml': rate_ml,
            'rate_uncertainty': rate_uncertainty,
            'n_observed': n_obs,
            'z_bins': z_centers,
            'n_in_bins': n_in_bins,
            'expected_per_unit_rate': expected_per_unit_rate
        }

        logger.info(f"Inferred rate: {rate_ml:.2e} ± {rate_uncertainty:.2e} Gpc^-3 yr^-1")

        return results
