import unittest
from unittest import mock
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
from redback.simulate_transients import (SimulateGenericTransient, SimulateOpticalTransient, SimulateFullOpticalSurvey,
    make_pointing_table_from_average_cadence, PopulationSynthesizer, TransientPopulation,
    SimulateTransientWithCadence, SimulateGammaRayTransient)
import bilby

class TestSimulateGenericTransient(unittest.TestCase):
    def setUp(self) -> None:
        self.model = lambda t, **kwargs: t * 2  # Simplified mock model
        self.parameters = {"param1": 1.0, "param2": 2.0}
        self.times = np.linspace(0, 10, 100)
        self.model_kwargs = {"bands": ["g", "r", "i"], "output_format": "flux"}
        self.data_points = 50
        self.seed = 42

    def test_initialization(self):
        transient = SimulateGenericTransient(
            model=self.model,
            parameters=self.parameters,
            times=self.times,
            model_kwargs=self.model_kwargs,
            data_points=self.data_points,
            seed=self.seed
        )
        self.assertEqual(transient.data_points, self.data_points)
        self.assertEqual(transient.seed, self.seed)
        self.assertEqual(len(transient.subset_times), self.data_points)
        self.assertTrue("time" in transient.data.columns)

    def test_multiwavelength_transient(self):
        transient = SimulateGenericTransient(
            model=self.model,
            parameters=self.parameters,
            times=self.times,
            model_kwargs=self.model_kwargs,
            data_points=self.data_points,
            seed=self.seed,
            multiwavelength_transient=True,
        )
        self.assertTrue("band" in transient.data.columns)
        self.assertEqual(len(transient.data), self.data_points)

    def test_gaussian_noise(self):
        transient = SimulateGenericTransient(
            model=self.model,
            parameters=self.parameters,
            times=self.times,
            model_kwargs=self.model_kwargs,
            data_points=self.data_points,
            seed=self.seed,
            noise_type="gaussian",
            noise_term=0.5,
        )
        self.assertTrue((transient.data["output_error"] == 0.5).all())

    def test_gaussianmodel_noise(self):
        transient = SimulateGenericTransient(
            model=self.model,
            parameters=self.parameters,
            times=self.times,
            model_kwargs=self.model_kwargs,
            data_points=self.data_points,
            seed=self.seed,
            noise_type="gaussianmodel",
            noise_term=0.1,
        )
        self.assertTrue("output_error" in transient.data.columns)
        self.assertTrue((transient.data["output"] != transient.data["true_output"]).any())

    def test_invalid_noise_type(self):
        with self.assertRaises(ValueError):
            SimulateGenericTransient(
                model=self.model,
                parameters=self.parameters,
                times=self.times,
                model_kwargs=self.model_kwargs,
                data_points=self.data_points,
                seed=self.seed,
                noise_type="invalid_noise_type",
            )

    def test_extra_scatter(self):
        transient = SimulateGenericTransient(
            model=self.model,
            parameters=self.parameters,
            times=self.times,
            model_kwargs=self.model_kwargs,
            data_points=self.data_points,
            seed=self.seed,
            extra_scatter=0.5,
        )
        self.assertTrue("output_error" in transient.data.columns)
        self.assertTrue((transient.data["output_error"] > 0).all())

    def test_save_transient(self):
        transient = SimulateGenericTransient(
            model=self.model,
            parameters=self.parameters,
            times=self.times,
            model_kwargs=self.model_kwargs,
            data_points=self.data_points,
            seed=self.seed,
        )
        with unittest.mock.patch("bilby.utils.check_directory_exists_and_if_not_mkdir"), \
                unittest.mock.patch("pandas.DataFrame.to_csv") as mock_to_csv:
            transient.save_transient("test_name")
            self.assertEqual(mock_to_csv.call_count, 2)
            mock_to_csv.assert_any_call("simulated/test_name.csv", index=False)
            mock_to_csv.assert_any_call("simulated/test_name_injection_parameters.csv", index=False)

class TestSimulateOpticalTransient(unittest.TestCase):

    def setUp(self) -> None:
        self.model = 'arnett'
        self.parameters = {"param1": 1.0, "param2": 2.0, 't0':60500, 'f_nickel':0.1,
                           'mej':2, 'redshift':0.1, 'vej':2000, 'kappa':0.1,
                           'kappa_gamma':0.6, 'temperature_floor':3000}
        self.model_kwargs = {}
        self.pointings_database = MagicMock()

    def tearDown(self) -> None:
        pass

    def test_simulate_transient(self):
        with patch("redback.simulate_transients.SimulateOpticalTransient.simulate_transient") as mock_simulate:
            mock_simulate.return_value = {"transient": True}
            result = SimulateOpticalTransient.simulate_transient(
                model=self.model,
                parameters=self.parameters,
                pointings_database=self.pointings_database,
                survey="Rubin_10yr_baseline"
            )
            self.assertEqual(result, {"transient": True})
            mock_simulate.assert_called_once()

    def test_simulate_transient_in_rubin(self):
        with patch("redback.simulate_transients.SimulateOpticalTransient.simulate_transient_in_rubin") as mock_simulate:
            mock_simulate.return_value = {"rubin_transient": True}
            result = SimulateOpticalTransient.simulate_transient_in_rubin(
                model=self.model,
                parameters=self.parameters,
                pointings_database=self.pointings_database,
            )
            self.assertEqual(result, {"rubin_transient": True})
            mock_simulate.assert_called_once()

    def test_simulate_transient_population(self):
        with patch(
                "redback.simulate_transients.SimulateOpticalTransient.simulate_transient_population") as mock_simulate:
            mock_simulate.return_value = {"population": True}
            result = SimulateOpticalTransient.simulate_transient_population(
                model=self.model,
                parameters=self.parameters,
                pointings_database=self.pointings_database,
                survey="Rubin_10yr_baseline"
            )
            self.assertEqual(result, {"population": True})
            mock_simulate.assert_called_once()

    def test_survey_radius_property(self):
        params_with_wavelength = {**self.parameters, "wavelength_observer_frame": 5500}
        instance = SimulateOpticalTransient(
            model=self.model,
            parameters=params_with_wavelength,
            survey_fov_sqdeg=9.6,
            model_kwargs=self.model_kwargs,
        )
        radius = instance.survey_radius
        self.assertAlmostEqual(radius, np.sqrt(9.6 * (np.pi / 180.0) ** 2 / np.pi))

    def test_save_transient_called(self):
        instance = SimulateOpticalTransient(
            model=self.model,
            parameters=self.parameters,
            model_kwargs=self.model_kwargs,
        )
        instance.save_transient = MagicMock()
        instance.save_transient("test_transient")
        instance.save_transient.assert_called_once_with("test_transient")

    def test_save_transient_population_called(self):
        instance = SimulateOpticalTransient(
            model=self.model,
            parameters=self.parameters,
            model_kwargs=self.model_kwargs,
        )
        instance.save_transient_population = MagicMock()
        instance.save_transient_population(transient_names=["entry1", "entry2"])
        instance.save_transient_population.assert_called_once_with(transient_names=["entry1", "entry2"])

class TestMakePointingTableFromAverageCadence(unittest.TestCase):
    def test_table_creation(self):
        ra = 10.0
        dec = -5.0
        num_obs = {'u': 5, 'g': 3}
        average_cadence = {'u': 2.0, 'g': 2.0}
        cadence_scatter = {'u': 0.5, 'g': 0.5}
        limiting_magnitudes = {'u': 23.5, 'g': 24.0}

        df = make_pointing_table_from_average_cadence(
            ra, dec, num_obs, average_cadence, cadence_scatter, limiting_magnitudes, initMJD=59580
        )

        # Check number of rows (sum of all observations)
        self.assertEqual(len(df), sum(num_obs.values()))

        # Check expected columns
        expected_cols = ['expMJD', '_ra', '_dec', 'filter', 'fiveSigmaDepth']
        for col in expected_cols:
            self.assertIn(col, df.columns)

        # Verify that RA and DEC columns match the input
        self.assertTrue(np.allclose(df['_ra'], ra))
        self.assertTrue(np.allclose(df['_dec'], dec))

        # Check that the dataframe is sorted by 'expMJD'
        self.assertTrue(df['expMJD'].is_monotonic_increasing)

class TestSimulateFullOpticalSurvey(unittest.TestCase):

    def setUp(self):
        self.model = MagicMock()
        self.prior = bilby.core.prior.PriorDict()
        self.prior['redshift'] = bilby.core.prior.Uniform(0, 2, "redshift")
        self.rate = 10
        self.survey_start_date = 60000
        self.survey_duration = 1
        self.pointings_database = MagicMock()
        self.survey = "Rubin_10yr_baseline"
        self.sncosmo_kwargs = {}
        self.obs_buffer = 5.0
        self.survey_fov_sqdeg = 9.6
        self.snr_threshold = 5
        self.end_transient_time = 1000
        self.add_source_noise = False
        self.model_kwargs = {}
        self.kwargs = {}

    @patch("redback.simulate_transients.SimulateOpticalTransient.__init__", return_value=None)
    def test_initialization(self, mock_base_init):
        survey = SimulateFullOpticalSurvey(
            model=self.model,
            prior=self.prior,
            rate=self.rate,
            survey_start_date=self.survey_start_date,
            survey_duration=self.survey_duration,
            pointings_database=self.pointings_database,
            survey=self.survey,
            sncosmo_kwargs=self.sncosmo_kwargs,
            obs_buffer=self.obs_buffer,
            survey_fov_sqdeg=self.survey_fov_sqdeg,
            snr_threshold=self.snr_threshold,
            end_transient_time=self.end_transient_time,
            add_source_noise=self.add_source_noise,
            model_kwargs=self.model_kwargs,
            **self.kwargs
        )
        self.assertEqual(survey.rate.value, self.rate)
        self.assertEqual(survey.survey_start_date, self.survey_start_date)

    @patch("redback.simulate_transients.np.random.poisson", return_value=100)
    @patch("redback.simulate_transients.SimulateOpticalTransient.__init__", return_value=None)
    def test_rate_per_sec_property(self, mock_base_init, mock_poisson):
        survey = SimulateFullOpticalSurvey(
            model=self.model,
            prior=self.prior,
            rate=self.rate,
            survey_start_date=self.survey_start_date,
            survey_duration=self.survey_duration,
            pointings_database=self.pointings_database,
            survey=self.survey,
            sncosmo_kwargs=self.sncosmo_kwargs,
            obs_buffer=self.obs_buffer,
            survey_fov_sqdeg=self.survey_fov_sqdeg,
            snr_threshold=self.snr_threshold,
            end_transient_time=self.end_transient_time,
            add_source_noise=self.add_source_noise,
            model_kwargs=self.model_kwargs,
            **self.kwargs
        )
        self.assertTrue(hasattr(survey, "rate_per_sec"))
        self.assertGreater(survey.rate_per_sec.value, 0)

    @patch("redback.simulate_transients.SimulateOpticalTransient.__init__", return_value=None)
    def test_survey_duration_seconds_property(self, mock_base_init):
        survey = SimulateFullOpticalSurvey(
            model=self.model,
            prior=self.prior,
            rate=self.rate,
            survey_start_date=self.survey_start_date,
            survey_duration=self.survey_duration,
            pointings_database=self.pointings_database,
            survey=self.survey,
            sncosmo_kwargs=self.sncosmo_kwargs,
            obs_buffer=self.obs_buffer,
            survey_fov_sqdeg=self.survey_fov_sqdeg,
            snr_threshold=self.snr_threshold,
            end_transient_time=self.end_transient_time,
            add_source_noise=self.add_source_noise,
            model_kwargs=self.model_kwargs,
            **self.kwargs
        )
        self.assertEqual(
            survey.survey_duration_seconds.value,
            self.survey_duration * 365.25 * 24 * 3600,
        )

    @patch("redback.simulate_transients.SimulateOpticalTransient.__init__", return_value=None)
    def test_time_window_property(self, mock_base_init):
        survey = SimulateFullOpticalSurvey(
            model=self.model,
            prior=self.prior,
            rate=self.rate,
            survey_start_date=self.survey_start_date,
            survey_duration=self.survey_duration,
            pointings_database=self.pointings_database,
            survey=self.survey,
            sncosmo_kwargs=self.sncosmo_kwargs,
            obs_buffer=self.obs_buffer,
            survey_fov_sqdeg=self.survey_fov_sqdeg,
            snr_threshold=self.snr_threshold,
            end_transient_time=self.end_transient_time,
            add_source_noise=self.add_source_noise,
            model_kwargs=self.model_kwargs,
            **self.kwargs
        )
        self.assertEqual(
            survey.time_window,
            [self.survey_start_date, self.survey_start_date + 365.25],
        )

    @patch("redback.simulate_transients.random.random", return_value=0.5)
    @patch("redback.simulate_transients.SimulateOpticalTransient.__init__", return_value=None)
    def test_get_event_times(self, mock_base_init, mock_random):
        survey = SimulateFullOpticalSurvey(
            model=self.model,
            prior=self.prior,
            rate=self.rate,
            survey_start_date=self.survey_start_date,
            survey_duration=self.survey_duration,
            pointings_database=self.pointings_database,
            survey=self.survey,
            sncosmo_kwargs=self.sncosmo_kwargs,
            obs_buffer=self.obs_buffer,
            survey_fov_sqdeg=self.survey_fov_sqdeg,
            snr_threshold=self.snr_threshold,
            end_transient_time=self.end_transient_time,
            add_source_noise=self.add_source_noise,
            model_kwargs=self.model_kwargs,
            **self.kwargs
        )
        event_times = survey.get_event_times()
        self.assertTrue(len(event_times) >= 1)

    @patch("redback.simulate_transients.bilby.utils.check_directory_exists_and_if_not_mkdir")
    @patch("redback.simulate_transients.pd.DataFrame.to_csv")
    @patch("redback.simulate_transients.SimulateOpticalTransient.__init__", return_value=None)
    def test_save_survey(self, mock_base_init, mock_to_csv, mock_check_directory):
        # Create the survey instance
        survey = SimulateFullOpticalSurvey(
            model=self.model,
            prior=self.prior,
            rate=self.rate,
            survey_start_date=self.survey_start_date,
            survey_duration=self.survey_duration,
            pointings_database=self.pointings_database,
            survey=self.survey,
            sncosmo_kwargs=self.sncosmo_kwargs,
            obs_buffer=self.obs_buffer,
            survey_fov_sqdeg=self.survey_fov_sqdeg,
            snr_threshold=self.snr_threshold,
            end_transient_time=self.end_transient_time,
            add_source_noise=self.add_source_noise,
            model_kwargs=self.model_kwargs,
            **self.kwargs
        )

        # Mock the `parameters` attribute to mimic a DataFrame
        survey.parameters = MagicMock()
        survey.parameters.to_csv = MagicMock()

        # Mock the `list_of_observations` attribute
        survey.list_of_observations = [
            MagicMock(to_csv=MagicMock()) for _ in range(3)
        ]  # Simulate 3 mock observations

        # Call save_survey()
        survey.save_survey()

        # Assert that the injection file is saved to CSV
        survey.parameters.to_csv.assert_called_once_with("simulated_survey/population_injection_parameters.csv",
                                                         index=False)

        # Assert that the directory creation is called
        mock_check_directory.assert_called_once_with("simulated_survey")

        # Assert `to_csv` is called for each observation
        for obs in survey.list_of_observations:
            obs.to_csv.assert_called()


class TestPopulationSynthesizer(unittest.TestCase):
    """Test suite for PopulationSynthesizer class"""

    def setUp(self) -> None:
        self.model = lambda t, **kwargs: t * 2  # Mock model
        self.prior = bilby.core.prior.PriorDict()
        self.prior['param1'] = bilby.core.prior.Uniform(0, 1, 'param1')
        self.prior['param2'] = bilby.core.prior.Uniform(0, 10, 'param2')
        self.rate = 1e-6  # Gpc^-3 yr^-1
        self.seed = 42

    def test_initialization_constant_rate(self):
        """Test initialization with constant rate evolution"""
        synth = PopulationSynthesizer(
            model=self.model,
            prior=self.prior,
            rate=self.rate,
            rate_evolution='constant',
            seed=self.seed
        )
        self.assertEqual(synth.rate, self.rate)
        self.assertEqual(synth.seed, self.seed)
        self.assertIsNotNone(synth.rate_function)

    def test_initialization_powerlaw_rate(self):
        """Test initialization with power-law rate evolution"""
        synth = PopulationSynthesizer(
            model=self.model,
            prior=self.prior,
            rate=self.rate,
            rate_evolution='powerlaw',
            rate_params={'alpha': 2.7},
            seed=self.seed
        )
        # Test that rate function works
        rate_at_z1 = synth.rate_function(1.0)
        rate_at_z0 = synth.rate_function(0.0)
        self.assertGreater(rate_at_z1, rate_at_z0)  # Rate increases with z

    def test_initialization_sfr_rate(self):
        """Test initialization with SFR-like rate evolution"""
        synth = PopulationSynthesizer(
            model=self.model,
            prior=self.prior,
            rate=self.rate,
            rate_evolution='sfr_like',
            seed=self.seed
        )
        rate_at_z = synth.rate_function(2.0)
        self.assertGreater(rate_at_z, 0)

    def test_initialization_custom_rate_function(self):
        """Test initialization with custom rate function"""
        custom_func = lambda z: self.rate * np.exp(-z / 0.5)
        synth = PopulationSynthesizer(
            model=self.model,
            prior=self.prior,
            rate=self.rate,
            rate_evolution=custom_func,
            seed=self.seed
        )
        self.assertEqual(synth.rate_function(0), self.rate)

    def test_generate_population_fixed_n(self):
        """Test population generation with fixed number of events"""
        synth = PopulationSynthesizer(
            model=self.model,
            prior=self.prior,
            rate=self.rate,
            rate_evolution='constant',
            seed=self.seed
        )
        n_events = 10
        params = synth.generate_population(
            n_events=n_events,
            z_max=0.3,
            rate_weighted_redshifts=True
        )
        self.assertIsInstance(params, pd.DataFrame)
        self.assertEqual(len(params), n_events)
        self.assertIn('redshift', params.columns)

    def test_generate_population_has_sky_positions(self):
        """Test that population includes sky positions when requested"""
        synth = PopulationSynthesizer(
            model=self.model,
            prior=self.prior,
            rate=self.rate,
            seed=self.seed
        )
        params = synth.generate_population(
            n_events=5,
            z_max=0.3,
            include_sky_position=True
        )
        self.assertIn('ra', params.columns)
        self.assertIn('dec', params.columns)
        # Check RA is in [0, 360)
        self.assertTrue((params['ra'] >= 0).all())
        self.assertTrue((params['ra'] < 360).all())
        # Check DEC is in [-90, 90]
        self.assertTrue((params['dec'] >= -90).all())
        self.assertTrue((params['dec'] <= 90).all())

    def test_generate_population_has_distances(self):
        """Test that population includes distance calculations"""
        synth = PopulationSynthesizer(
            model=self.model,
            prior=self.prior,
            rate=self.rate,
            seed=self.seed
        )
        params = synth.generate_population(n_events=5, z_max=0.3)
        self.assertIn('luminosity_distance', params.columns)
        self.assertIn('comoving_distance', params.columns)
        # Check distances are positive
        self.assertTrue((params['luminosity_distance'] > 0).all())
        self.assertTrue((params['comoving_distance'] > 0).all())

    def test_generate_population_has_event_times(self):
        """Test that population includes event times"""
        synth = PopulationSynthesizer(
            model=self.model,
            prior=self.prior,
            rate=self.rate,
            seed=self.seed
        )
        params = synth.generate_population(
            n_events=5,
            z_max=0.3,
            time_range=(60000, 60365.25)  # One year MJD range
        )
        self.assertIn('t0_mjd_transient', params.columns)
        # Check times are within range
        self.assertTrue((params['t0_mjd_transient'] >= 60000).all())
        self.assertTrue((params['t0_mjd_transient'] <= 60365.25).all())

    def test_generate_population_rate_weighted_false(self):
        """Test population generation with rate_weighted_redshifts=False"""
        synth = PopulationSynthesizer(
            model=self.model,
            prior=self.prior,
            rate=self.rate,
            seed=self.seed
        )
        # Add redshift to prior
        prior_with_z = self.prior.copy()
        prior_with_z['redshift'] = bilby.core.prior.Uniform(0.01, 0.5, 'redshift')
        synth.prior = prior_with_z

        params = synth.generate_population(
            n_events=10,
            rate_weighted_redshifts=False
        )
        self.assertEqual(len(params), 10)
        self.assertIn('redshift', params.columns)

    def test_apply_detection_criteria_simple(self):
        """Test applying simple detection criteria"""
        synth = PopulationSynthesizer(
            model=self.model,
            prior=self.prior,
            rate=self.rate,
            seed=self.seed
        )
        params = synth.generate_population(n_events=10, z_max=0.5)

        # Simple detection: z < 0.3
        def simple_det(row):
            return row['redshift'] < 0.3

        detected = synth.apply_detection_criteria(params, simple_det)
        self.assertIn('detected', detected.columns)
        self.assertIn('detection_probability', detected.columns)

        # Check that detection is applied correctly
        for idx, row in detected.iterrows():
            if row['redshift'] < 0.3:
                self.assertTrue(row['detected'])
            else:
                self.assertFalse(row['detected'])

    def test_apply_detection_criteria_probabilistic(self):
        """Test applying probabilistic detection criteria"""
        synth = PopulationSynthesizer(
            model=self.model,
            prior=self.prior,
            rate=self.rate,
            seed=self.seed
        )
        params = synth.generate_population(n_events=20, z_max=0.5)

        # Probabilistic detection
        def prob_det(row):
            return 0.5  # 50% detection probability

        detected = synth.apply_detection_criteria(params, prob_det)
        # With 20 events at 50%, we should have some detected and some not
        # (probabilistic, but statistically very unlikely to have 0 or 20)
        n_detected = detected['detected'].sum()
        self.assertGreater(n_detected, 0)
        self.assertLess(n_detected, 20)

    def test_redshift_sampling_within_bounds(self):
        """Test that redshifts are sampled within specified bounds"""
        synth = PopulationSynthesizer(
            model=self.model,
            prior=self.prior,
            rate=self.rate,
            seed=self.seed
        )
        z_max = 0.2
        params = synth.generate_population(n_events=50, z_max=z_max)
        self.assertTrue((params['redshift'] > 0).all())
        self.assertTrue((params['redshift'] <= z_max).all())

    def test_infer_rate_basic(self):
        """Test rate inference from observed sample"""
        synth = PopulationSynthesizer(
            model=self.model,
            prior=self.prior,
            rate=self.rate,
            seed=self.seed
        )
        # Create fake observed sample
        observed = pd.DataFrame({'redshift': [0.1, 0.15, 0.2, 0.25, 0.3]})
        result = synth.infer_rate(observed)
        self.assertIn('rate_ml', result)
        self.assertIn('rate_uncertainty', result)
        self.assertIn('n_observed', result)
        self.assertEqual(result['n_observed'], 5)


class TestTransientPopulation(unittest.TestCase):
    """Test suite for TransientPopulation container class"""

    def setUp(self) -> None:
        self.params = pd.DataFrame({
            'redshift': [0.1, 0.2, 0.3, 0.4, 0.5],
            'ra': [10, 20, 30, 40, 50],
            'dec': [-10, -5, 0, 5, 10],
            'luminosity_distance': [450, 950, 1500, 2100, 2800],
            'detected': [True, True, True, False, False]
        })
        self.metadata = {'survey': 'LSST', 'model': 'kilonova'}

    def test_initialization(self):
        """Test TransientPopulation initialization"""
        pop = TransientPopulation(self.params, self.metadata)
        self.assertEqual(pop.n_transients, 5)
        self.assertEqual(pop.metadata, self.metadata)

    def test_redshifts_property(self):
        """Test redshifts property"""
        pop = TransientPopulation(self.params, self.metadata)
        redshifts = pop.redshifts
        np.testing.assert_array_equal(redshifts, self.params['redshift'].values)

    def test_sky_positions_property(self):
        """Test sky_positions property"""
        pop = TransientPopulation(self.params, self.metadata)
        ra, dec = pop.sky_positions
        np.testing.assert_array_equal(ra, self.params['ra'].values)
        np.testing.assert_array_equal(dec, self.params['dec'].values)

    def test_detection_fraction_property(self):
        """Test detection_fraction property"""
        pop = TransientPopulation(self.params, self.metadata)
        frac = pop.detection_fraction
        self.assertAlmostEqual(frac, 3/5)  # 3 out of 5 detected

    def test_detection_fraction_no_column(self):
        """Test detection_fraction when no 'detected' column exists"""
        params_no_det = self.params.drop(columns=['detected'])
        pop = TransientPopulation(params_no_det, self.metadata)
        frac = pop.detection_fraction
        self.assertEqual(frac, 1.0)  # Default to all detected

    def test_summary_stats(self):
        """Test summary_stats method"""
        pop = TransientPopulation(self.params, self.metadata)
        stats = pop.summary_stats()
        self.assertIn('n_transients', stats)
        self.assertIn('median_redshift', stats)
        self.assertIn('min_redshift', stats)
        self.assertIn('max_redshift', stats)
        self.assertEqual(stats['n_transients'], 5)

    def test_filter_by_redshift(self):
        """Test filter_by_redshift method"""
        pop = TransientPopulation(self.params, self.metadata)
        filtered = pop.filter_by_redshift(z_min=0.15, z_max=0.35)
        self.assertIsInstance(filtered, TransientPopulation)
        self.assertEqual(filtered.n_transients, 2)  # z=0.2 and z=0.3

    @patch("bilby.utils.check_directory_exists_and_if_not_mkdir")
    @patch("pandas.DataFrame.to_csv")
    def test_save_without_metadata(self, mock_to_csv, mock_makedirs):
        """Test save method without metadata"""
        pop = TransientPopulation(self.params)
        pop.save('test_pop.csv', save_metadata=False)
        mock_makedirs.assert_called()
        mock_to_csv.assert_called_once()

    @patch("bilby.utils.check_directory_exists_and_if_not_mkdir")
    @patch("builtins.open", new_callable=mock.mock_open)
    @patch("pandas.DataFrame.to_csv")
    def test_save_with_metadata(self, mock_to_csv, mock_open, mock_makedirs):
        """Test save method with metadata"""
        pop = TransientPopulation(self.params, self.metadata)
        pop.save('test_pop.csv', save_metadata=True)
        mock_makedirs.assert_called()
        mock_to_csv.assert_called_once()
        mock_open.assert_called()  # For JSON metadata


class TestSimulateTransientWithCadence(unittest.TestCase):
    """Test suite for SimulateTransientWithCadence class"""

    def setUp(self) -> None:
        # Mock model that returns simple magnitudes
        self.model = lambda t, **kwargs: np.full_like(t, 20.0)  # Constant magnitude
        self.parameters = {
            'redshift': 0.1,
            'luminosity_distance': 450,
            'param1': 1.0
        }
        self.optical_cadence = {
            'bands': ['g', 'r'],
            'cadence_days': 2.0,
            'duration_days': 10,
            'limiting_mags': {'g': 22.5, 'r': 23.0}
        }
        self.seed = 42

    def test_optical_initialization(self):
        """Test initialization for optical observations"""
        sim = SimulateTransientWithCadence(
            model=self.model,
            parameters=self.parameters,
            cadence_config=self.optical_cadence,
            observation_mode='optical',
            seed=self.seed
        )
        self.assertEqual(sim.observation_mode, 'optical')
        self.assertIsNotNone(sim.observations)

    def test_auto_detect_optical_mode(self):
        """Test auto-detection of optical mode from config keys"""
        sim = SimulateTransientWithCadence(
            model=self.model,
            parameters=self.parameters,
            cadence_config=self.optical_cadence,
            seed=self.seed
        )
        self.assertEqual(sim.observation_mode, 'optical')

    def test_radio_initialization(self):
        """Test initialization for radio observations"""
        radio_cadence = {
            'frequencies': [1.4e9, 5e9],
            'cadence_days': 7,
            'duration_days': 30,
            'sensitivity': 0.05
        }
        # Mock model for radio (returns flux density)
        radio_model = lambda t, **kwargs: np.full_like(t, 1.0)  # 1 Jy

        sim = SimulateTransientWithCadence(
            model=radio_model,
            parameters=self.parameters,
            cadence_config=radio_cadence,
            observation_mode='radio',
            seed=self.seed
        )
        self.assertEqual(sim.observation_mode, 'radio')

    def test_auto_detect_radio_mode(self):
        """Test auto-detection of radio mode from config keys"""
        radio_cadence = {
            'frequencies': [1.4e9, 5e9],
            'cadence_days': 7,
            'duration_days': 30,
            'sensitivity': 0.05
        }
        radio_model = lambda t, **kwargs: np.full_like(t, 1.0)

        sim = SimulateTransientWithCadence(
            model=radio_model,
            parameters=self.parameters,
            cadence_config=radio_cadence,
            seed=self.seed
        )
        self.assertEqual(sim.observation_mode, 'radio')

    def test_observations_dataframe_structure_optical(self):
        """Test that observations DataFrame has correct structure for optical"""
        sim = SimulateTransientWithCadence(
            model=self.model,
            parameters=self.parameters,
            cadence_config=self.optical_cadence,
            seed=self.seed
        )
        obs = sim.observations
        self.assertIsInstance(obs, pd.DataFrame)
        self.assertIn('time_since_t0', obs.columns)
        self.assertIn('band', obs.columns)
        self.assertIn('magnitude', obs.columns)
        self.assertIn('snr', obs.columns)
        self.assertIn('detected', obs.columns)

    def test_detected_observations_filtering(self):
        """Test that detected_observations filters correctly"""
        sim = SimulateTransientWithCadence(
            model=self.model,
            parameters=self.parameters,
            cadence_config=self.optical_cadence,
            snr_threshold=5,
            seed=self.seed
        )
        detected = sim.detected_observations
        self.assertIsInstance(detected, pd.DataFrame)
        # All detected should have SNR >= threshold
        if len(detected) > 0:
            self.assertTrue((detected['snr'] >= 5).all())

    def test_per_band_cadence(self):
        """Test per-band cadence configuration"""
        cadence = {
            'bands': ['g', 'r', 'i'],
            'cadence_days': {'g': 3, 'r': 1, 'i': 5},
            'duration_days': 15,
            'limiting_mags': {'g': 22.5, 'r': 23.0, 'i': 22.5}
        }
        sim = SimulateTransientWithCadence(
            model=self.model,
            parameters=self.parameters,
            cadence_config=cadence,
            seed=self.seed
        )
        # r-band has cadence of 1 day, so should have more observations
        r_obs = sim.observations[sim.observations['band'] == 'r']
        g_obs = sim.observations[sim.observations['band'] == 'g']
        self.assertGreater(len(r_obs), len(g_obs))

    def test_delayed_start(self):
        """Test delayed observation start"""
        cadence = {
            'bands': ['g'],
            'cadence_days': 2,
            'duration_days': 10,
            'limiting_mags': {'g': 22.5},
            'start_offset_days': 5
        }
        sim = SimulateTransientWithCadence(
            model=self.model,
            parameters=self.parameters,
            cadence_config=cadence,
            seed=self.seed
        )
        # First observation should be at offset
        min_time = sim.observations['time_since_t0'].min()
        self.assertGreaterEqual(min_time, 5)

    def test_snr_threshold_effect(self):
        """Test that SNR threshold affects detections"""
        # Model with faint magnitude (lower SNR)
        faint_model = lambda t, **kwargs: np.full_like(t, 23.5)  # Fainter than limiting

        sim = SimulateTransientWithCadence(
            model=faint_model,
            parameters=self.parameters,
            cadence_config=self.optical_cadence,
            snr_threshold=5,
            seed=self.seed
        )
        # Should have fewer detections because source is faint
        detected = sim.detected_observations
        self.assertLessEqual(len(detected), len(sim.observations))

    @patch("bilby.utils.check_directory_exists_and_if_not_mkdir")
    @patch("pandas.DataFrame.to_csv")
    def test_save_transient(self, mock_to_csv, mock_makedirs):
        """Test saving transient data"""
        sim = SimulateTransientWithCadence(
            model=self.model,
            parameters=self.parameters,
            cadence_config=self.optical_cadence,
            seed=self.seed
        )
        with patch("builtins.open", new_callable=mock.mock_open):
            sim.save_transient('test_cadence')
        mock_makedirs.assert_called()
        # Should save observations, parameters, and config
        self.assertGreaterEqual(mock_to_csv.call_count, 2)


class TestSimulateGammaRayTransient(unittest.TestCase):
    """Test suite for SimulateGammaRayTransient class"""

    def setUp(self) -> None:
        # Mock model that returns flux density (Jy)
        self.model = lambda t, **kwargs: np.full_like(t, 1e-3)  # 1 mJy
        self.parameters = {'luminosity': 1e50, 'redshift': 0.5}
        self.energy_edges = [10, 50, 100, 300]  # keV
        self.time_range = (0, 10)  # seconds
        self.effective_area = 100  # cm^2
        self.background_rate = 0.1  # counts/s/keV
        self.seed = 42

    def test_initialization(self):
        """Test SimulateGammaRayTransient initialization"""
        sim = SimulateGammaRayTransient(
            model=self.model,
            parameters=self.parameters,
            energy_edges=self.energy_edges,
            time_range=self.time_range,
            effective_area=self.effective_area,
            background_rate=self.background_rate,
            seed=self.seed
        )
        self.assertEqual(sim.n_energy_bins, 3)  # 3 bins from 4 edges
        self.assertEqual(sim.t_start, 0)
        self.assertEqual(sim.t_end, 10)
        self.assertEqual(sim.seed, self.seed)

    def test_energy_centers_geometric_mean(self):
        """Test that energy centers are geometric means of bin edges"""
        sim = SimulateGammaRayTransient(
            model=self.model,
            parameters=self.parameters,
            energy_edges=self.energy_edges,
            time_range=self.time_range,
            seed=self.seed
        )
        # Check first bin: sqrt(10 * 50) = 22.36
        expected_center = np.sqrt(10 * 50)
        self.assertAlmostEqual(sim.energy_centers[0], expected_center, places=2)

    def test_constant_effective_area(self):
        """Test constant effective area setup"""
        sim = SimulateGammaRayTransient(
            model=self.model,
            parameters=self.parameters,
            energy_edges=self.energy_edges,
            time_range=self.time_range,
            effective_area=100,
            seed=self.seed
        )
        # Should return 100 for any energy
        self.assertEqual(sim.effective_area_func(50), 100)
        self.assertEqual(sim.effective_area_func(200), 100)

    def test_dict_effective_area(self):
        """Test dictionary effective area setup"""
        area_dict = {10: 50, 100: 120, 1000: 40}
        sim = SimulateGammaRayTransient(
            model=self.model,
            parameters=self.parameters,
            energy_edges=self.energy_edges,
            time_range=self.time_range,
            effective_area=area_dict,
            seed=self.seed
        )
        # Should interpolate
        area_at_100 = sim.effective_area_func(100)
        self.assertAlmostEqual(area_at_100, 120, places=1)

    def test_callable_effective_area(self):
        """Test callable effective area setup"""
        area_func = lambda e: 100 * np.exp(-e / 500)
        sim = SimulateGammaRayTransient(
            model=self.model,
            parameters=self.parameters,
            energy_edges=self.energy_edges,
            time_range=self.time_range,
            effective_area=area_func,
            seed=self.seed
        )
        area_at_50 = sim.effective_area_func(50)
        expected = 100 * np.exp(-50 / 500)
        self.assertAlmostEqual(area_at_50, expected, places=5)

    def test_constant_background_rate(self):
        """Test constant background rate setup"""
        sim = SimulateGammaRayTransient(
            model=self.model,
            parameters=self.parameters,
            energy_edges=self.energy_edges,
            time_range=self.time_range,
            background_rate=0.1,
            seed=self.seed
        )
        self.assertEqual(sim.background_rate_func(50), 0.1)

    def test_generate_binned_counts_structure(self):
        """Test binned counts generation structure"""
        sim = SimulateGammaRayTransient(
            model=self.model,
            parameters=self.parameters,
            energy_edges=self.energy_edges,
            time_range=self.time_range,
            seed=self.seed
        )
        time_bins = np.linspace(0, 10, 6)  # 5 time bins
        binned = sim.generate_binned_counts(time_bins, energy_integrated=False)

        self.assertIsInstance(binned, pd.DataFrame)
        self.assertIn('time_center', binned.columns)
        self.assertIn('counts', binned.columns)
        self.assertIn('counts_error', binned.columns)
        self.assertIn('count_rate', binned.columns)
        self.assertIn('energy_channel', binned.columns)

        # Should have 5 time bins * 3 energy channels = 15 rows
        self.assertEqual(len(binned), 15)

    def test_generate_binned_counts_energy_integrated(self):
        """Test energy-integrated binned counts"""
        sim = SimulateGammaRayTransient(
            model=self.model,
            parameters=self.parameters,
            energy_edges=self.energy_edges,
            time_range=self.time_range,
            seed=self.seed
        )
        time_bins = np.linspace(0, 10, 6)
        binned = sim.generate_binned_counts(time_bins, energy_integrated=True)

        # Should have only 5 rows (one per time bin)
        self.assertEqual(len(binned), 5)
        self.assertNotIn('energy_channel', binned.columns)

    def test_counts_are_positive(self):
        """Test that counts are non-negative"""
        sim = SimulateGammaRayTransient(
            model=self.model,
            parameters=self.parameters,
            energy_edges=self.energy_edges,
            time_range=self.time_range,
            seed=self.seed
        )
        time_bins = np.linspace(0, 10, 11)
        binned = sim.generate_binned_counts(time_bins)
        self.assertTrue((binned['counts'] >= 0).all())

    def test_count_rate_error_calculation(self):
        """Test that count rate error is correctly calculated"""
        sim = SimulateGammaRayTransient(
            model=self.model,
            parameters=self.parameters,
            energy_edges=self.energy_edges,
            time_range=self.time_range,
            seed=self.seed
        )
        time_bins = np.array([0, 2, 4, 6, 8, 10])  # dt = 2s
        binned = sim.generate_binned_counts(time_bins, energy_integrated=True)

        # Check error = sqrt(counts) / dt
        for idx, row in binned.iterrows():
            expected_error = np.sqrt(max(row['counts'], 1)) / row['dt']
            self.assertAlmostEqual(row['count_rate_error'], expected_error, places=10)

    def test_time_tagged_events_structure(self):
        """Test time-tagged events generation structure"""
        # Use constant flux model for predictable behavior
        constant_model = lambda t, **kwargs: np.full_like(t, 1e-5)  # Weak flux

        sim = SimulateGammaRayTransient(
            model=constant_model,
            parameters=self.parameters,
            energy_edges=[10, 50, 100],  # 2 channels
            time_range=(0, 1),  # Short time
            effective_area=10,  # Small area
            background_rate=1.0,  # Dominate by background
            time_resolution=0.01,
            seed=self.seed
        )
        events = sim.generate_time_tagged_events(max_events=10000)

        self.assertIsInstance(events, pd.DataFrame)
        self.assertIn('time', events.columns)
        self.assertIn('energy', events.columns)
        self.assertIn('energy_channel', events.columns)
        self.assertIn('is_background', events.columns)

    def test_time_tagged_events_sorted(self):
        """Test that time-tagged events are sorted by time"""
        constant_model = lambda t, **kwargs: np.full_like(t, 1e-5)
        sim = SimulateGammaRayTransient(
            model=constant_model,
            parameters=self.parameters,
            energy_edges=[10, 50, 100],
            time_range=(0, 1),
            effective_area=10,
            background_rate=1.0,
            seed=self.seed
        )
        events = sim.generate_time_tagged_events()

        if len(events) > 1:
            times = events['time'].values
            self.assertTrue((np.diff(times) >= 0).all())

    def test_events_within_time_range(self):
        """Test that events are within specified time range"""
        constant_model = lambda t, **kwargs: np.full_like(t, 1e-5)
        sim = SimulateGammaRayTransient(
            model=constant_model,
            parameters=self.parameters,
            energy_edges=[10, 50, 100],
            time_range=(2, 8),
            effective_area=10,
            background_rate=1.0,
            seed=self.seed
        )
        events = sim.generate_time_tagged_events()

        if len(events) > 0:
            self.assertTrue((events['time'] >= 2).all())
            self.assertTrue((events['time'] <= 8).all())

    def test_events_energies_within_channels(self):
        """Test that event energies are within their channel bounds"""
        constant_model = lambda t, **kwargs: np.full_like(t, 1e-5)
        sim = SimulateGammaRayTransient(
            model=constant_model,
            parameters=self.parameters,
            energy_edges=[10, 50, 100],
            time_range=(0, 1),
            effective_area=10,
            background_rate=1.0,
            seed=self.seed
        )
        events = sim.generate_time_tagged_events()

        if len(events) > 0:
            for idx, row in events.iterrows():
                ch = int(row['energy_channel'])
                e_low = sim.energy_edges[ch]
                e_high = sim.energy_edges[ch + 1]
                self.assertGreaterEqual(row['energy'], e_low)
                self.assertLess(row['energy'], e_high)

    @patch("os.makedirs")
    @patch("pandas.DataFrame.to_csv")
    @patch("builtins.open", new_callable=mock.mock_open)
    def test_save_time_tagged_events(self, mock_open, mock_to_csv, mock_makedirs):
        """Test saving time-tagged events"""
        constant_model = lambda t, **kwargs: np.full_like(t, 1e-5)
        sim = SimulateGammaRayTransient(
            model=constant_model,
            parameters=self.parameters,
            energy_edges=[10, 50, 100],
            time_range=(0, 1),
            effective_area=10,
            background_rate=1.0,
            seed=self.seed
        )
        sim.generate_time_tagged_events()
        sim.save_time_tagged_events('test_events')

        mock_makedirs.assert_called()
        mock_to_csv.assert_called_once()
        mock_open.assert_called()  # For JSON metadata

    @patch("os.makedirs")
    @patch("pandas.DataFrame.to_csv")
    def test_save_binned_counts(self, mock_to_csv, mock_makedirs):
        """Test saving binned counts"""
        sim = SimulateGammaRayTransient(
            model=self.model,
            parameters=self.parameters,
            energy_edges=self.energy_edges,
            time_range=self.time_range,
            seed=self.seed
        )
        time_bins = np.linspace(0, 10, 6)
        sim.generate_binned_counts(time_bins)
        sim.save_binned_counts('test_binned')

        mock_makedirs.assert_called()
        mock_to_csv.assert_called_once()

    def test_save_without_generation_raises_error(self):
        """Test that saving without generating data raises error"""
        sim = SimulateGammaRayTransient(
            model=self.model,
            parameters=self.parameters,
            energy_edges=self.energy_edges,
            time_range=self.time_range,
            seed=self.seed
        )
        with self.assertRaises(ValueError):
            sim.save_time_tagged_events('test')

        with self.assertRaises(ValueError):
            sim.save_binned_counts('test')

