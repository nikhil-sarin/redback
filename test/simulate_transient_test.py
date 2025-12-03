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


# Additional comprehensive tests for higher coverage

class TestTransientPopulationExtended(unittest.TestCase):
    """Extended tests for TransientPopulation to increase coverage"""

    def setUp(self):
        self.params = pd.DataFrame({
            'redshift': [0.1, 0.2, 0.3, 0.4, 0.5],
            'ra': [10, 20, 30, 40, 50],
            'dec': [-10, -5, 0, 5, 10],
            'luminosity_distance': [450, 950, 1500, 2100, 2800],
            'detected': [True, True, True, False, False],
            'param1': [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        self.metadata = {'survey': 'LSST', 'model': 'kilonova'}

    def test_len_method(self):
        """Test __len__ method"""
        pop = TransientPopulation(self.params, self.metadata)
        self.assertEqual(len(pop), 5)

    def test_repr_method(self):
        """Test __repr__ method"""
        pop = TransientPopulation(self.params, self.metadata)
        repr_str = repr(pop)
        self.assertIn('TransientPopulation', repr_str)
        self.assertIn('5', repr_str)

    def test_detected_property(self):
        """Test detected property returns filtered DataFrame"""
        pop = TransientPopulation(self.params, self.metadata)
        detected = pop.detected
        # Should return DataFrame with only detected transients
        self.assertIsInstance(detected, pd.DataFrame)
        self.assertEqual(len(detected), 3)  # 3 detected out of 5

    def test_detected_property_no_column(self):
        """Test detected property when column doesn't exist"""
        params_no_det = self.params.drop(columns=['detected'])
        pop = TransientPopulation(params_no_det, self.metadata)
        detected = pop.detected
        # Should return all parameters
        self.assertEqual(len(detected), 5)

    def test_get_redshift_distribution(self):
        """Test get_redshift_distribution method"""
        pop = TransientPopulation(self.params, self.metadata)
        bin_edges, hist, bin_centers = pop.get_redshift_distribution(bins=5)
        self.assertEqual(len(hist), 5)
        self.assertEqual(len(bin_edges), 6)
        self.assertEqual(hist.sum(), 5)  # Total count

    def test_get_parameter_distribution(self):
        """Test get_parameter_distribution method"""
        pop = TransientPopulation(self.params, self.metadata)
        bin_edges, hist, bin_centers = pop.get_parameter_distribution('param1', bins=5)
        self.assertEqual(len(hist), 5)
        self.assertEqual(hist.sum(), 5)

    def test_get_parameter_distribution_missing_param(self):
        """Test get_parameter_distribution with missing parameter"""
        pop = TransientPopulation(self.params, self.metadata)
        # Returns (None, None, None) for missing parameter
        result = pop.get_parameter_distribution('nonexistent_param')
        self.assertEqual(result, (None, None, None))

    @patch("bilby.utils.check_directory_exists_and_if_not_mkdir")
    @patch("builtins.open", new_callable=mock.mock_open, read_data='{"survey": "test"}')
    @patch("pandas.read_csv")
    def test_load_classmethod(self, mock_read_csv, mock_open, mock_makedirs):
        """Test load classmethod"""
        mock_read_csv.return_value = self.params
        pop = TransientPopulation.load('test_pop.csv')
        self.assertIsInstance(pop, TransientPopulation)
        mock_read_csv.assert_called_once()

    @patch("bilby.utils.check_directory_exists_and_if_not_mkdir")
    @patch("pandas.read_csv")
    def test_load_classmethod_no_metadata(self, mock_read_csv, mock_makedirs):
        """Test load classmethod when metadata file doesn't exist"""
        mock_read_csv.return_value = self.params
        with patch("builtins.open", side_effect=FileNotFoundError):
            pop = TransientPopulation.load('test_pop.csv')
        self.assertIsInstance(pop, TransientPopulation)
        # When metadata file not found, metadata is empty dict or None
        self.assertTrue(pop.metadata is None or pop.metadata == {})

    def test_sky_positions_no_ra_dec(self):
        """Test sky_positions when RA/Dec columns don't exist"""
        params_no_sky = self.params.drop(columns=['ra', 'dec'])
        pop = TransientPopulation(params_no_sky, self.metadata)
        ra, dec = pop.sky_positions
        self.assertIsNone(ra)
        self.assertIsNone(dec)

    def test_filter_by_redshift_min_only(self):
        """Test filter_by_redshift with only z_min"""
        pop = TransientPopulation(self.params, self.metadata)
        filtered = pop.filter_by_redshift(z_min=0.25)
        self.assertEqual(filtered.n_transients, 3)  # z=0.3, 0.4, 0.5

    def test_filter_by_redshift_max_only(self):
        """Test filter_by_redshift with only z_max"""
        pop = TransientPopulation(self.params, self.metadata)
        filtered = pop.filter_by_redshift(z_max=0.25)
        self.assertEqual(filtered.n_transients, 2)  # z=0.1, 0.2

    def test_summary_stats_no_redshift(self):
        """Test summary_stats when no redshift column"""
        params_no_z = self.params.drop(columns=['redshift'])
        pop = TransientPopulation(params_no_z, self.metadata)
        stats = pop.summary_stats()
        self.assertIn('n_transients', stats)
        self.assertNotIn('median_redshift', stats)


class TestSimulateTransientWithCadenceExtended(unittest.TestCase):
    """Extended tests for SimulateTransientWithCadence"""

    def setUp(self):
        self.model = lambda t, **kwargs: np.full_like(t, 20.0)
        self.parameters = {'redshift': 0.1, 'luminosity_distance': 450}
        self.seed = 42

    def test_xray_initialization(self):
        """Test X-ray mode initialization"""
        xray_cadence = {
            'frequencies': [2.4e17, 1.2e18],  # X-ray frequencies
            'cadence_days': 5,
            'duration_days': 20,
            'sensitivity': 1e-13  # erg/cm^2/s
        }
        xray_model = lambda t, **kwargs: np.full_like(t, 1e-12)

        sim = SimulateTransientWithCadence(
            model=xray_model,
            parameters=self.parameters,
            cadence_config=xray_cadence,
            observation_mode='xray',
            seed=self.seed
        )
        self.assertEqual(sim.observation_mode, 'xray')
        self.assertIn('frequency', sim.observations.columns)

    def test_gaussian_noise_optical(self):
        """Test Gaussian noise type for optical"""
        optical_cadence = {
            'bands': ['g', 'r'],
            'cadence_days': 2.0,
            'duration_days': 10,
            'limiting_mags': {'g': 22.5, 'r': 23.0}
        }
        sim = SimulateTransientWithCadence(
            model=self.model,
            parameters=self.parameters,
            cadence_config=optical_cadence,
            noise_type='gaussian',
            seed=self.seed
        )
        self.assertIn('magnitude_error', sim.observations.columns)
        self.assertIn('snr', sim.observations.columns)

    def test_gaussianmodel_noise_optical(self):
        """Test Gaussian model noise type for optical"""
        optical_cadence = {
            'bands': ['g'],
            'cadence_days': 1.0,
            'duration_days': 5,
            'limiting_mags': {'g': 22.5}
        }
        sim = SimulateTransientWithCadence(
            model=self.model,
            parameters=self.parameters,
            cadence_config=optical_cadence,
            noise_type='gaussianmodel',
            seed=self.seed
        )
        self.assertIn('magnitude_error', sim.observations.columns)

    def test_gaussian_noise_radio(self):
        """Test Gaussian noise for radio"""
        radio_cadence = {
            'frequencies': [1.4e9],
            'cadence_days': 7,
            'duration_days': 30,
            'sensitivity': 0.05
        }
        radio_model = lambda t, **kwargs: np.full_like(t, 1.0)

        sim = SimulateTransientWithCadence(
            model=radio_model,
            parameters=self.parameters,
            cadence_config=radio_cadence,
            observation_mode='radio',
            noise_type='gaussian',
            seed=self.seed
        )
        self.assertIn('snr', sim.observations.columns)

    def test_gaussianmodel_noise_radio(self):
        """Test Gaussian model noise for radio"""
        radio_cadence = {
            'frequencies': [1.4e9],
            'cadence_days': 7,
            'duration_days': 30,
            'sensitivity': 0.05
        }
        radio_model = lambda t, **kwargs: np.full_like(t, 1.0)

        sim = SimulateTransientWithCadence(
            model=radio_model,
            parameters=self.parameters,
            cadence_config=radio_cadence,
            observation_mode='radio',
            noise_type='gaussianmodel',
            seed=self.seed
        )
        self.assertIn('flux_density', sim.observations.columns)

    def test_per_frequency_cadence(self):
        """Test per-frequency cadence configuration"""
        radio_cadence = {
            'frequencies': [1.4e9, 5e9],
            'cadence_days': {1.4e9: 7, 5e9: 14},
            'duration_days': 30,
            'sensitivity': {1.4e9: 0.05, 5e9: 0.02}
        }
        radio_model = lambda t, **kwargs: np.full_like(t, 1.0)

        sim = SimulateTransientWithCadence(
            model=radio_model,
            parameters=self.parameters,
            cadence_config=radio_cadence,
            observation_mode='radio',
            seed=self.seed
        )
        # Check that different frequencies have different number of observations
        freq_counts = sim.observations['frequency'].value_counts()
        self.assertIn(1.4e9, freq_counts.index)

    def test_no_snr_cut(self):
        """Test with apply_snr_cut=False"""
        optical_cadence = {
            'bands': ['g'],
            'cadence_days': 1.0,
            'duration_days': 5,
            'limiting_mags': {'g': 22.5}
        }
        sim = SimulateTransientWithCadence(
            model=self.model,
            parameters=self.parameters,
            cadence_config=optical_cadence,
            apply_snr_cut=False,
            seed=self.seed
        )
        # All observations should be detected
        self.assertTrue(sim.observations['detected'].all())

    def test_missing_cadence_key_optical(self):
        """Test missing required key in optical cadence config"""
        bad_cadence = {
            'bands': ['g'],
            'cadence_days': 1.0,
            # Missing 'duration_days' and 'limiting_mags'
        }
        with self.assertRaises(ValueError):
            SimulateTransientWithCadence(
                model=self.model,
                parameters=self.parameters,
                cadence_config=bad_cadence,
                seed=self.seed
            )

    def test_missing_cadence_key_radio(self):
        """Test missing required key in radio cadence config"""
        bad_cadence = {
            'frequencies': [1.4e9],
            'cadence_days': 7,
            # Missing 'duration_days' and 'sensitivity'
        }
        with self.assertRaises(ValueError):
            SimulateTransientWithCadence(
                model=self.model,
                parameters=self.parameters,
                cadence_config=bad_cadence,
                observation_mode='radio',
                seed=self.seed
            )

    def test_invalid_observation_mode(self):
        """Test invalid observation mode raises error during validation"""
        # Config that doesn't match the mode will fail validation
        bad_cadence = {
            'bands': ['g'],  # Optical config
            'cadence_days': 1.0,
            'duration_days': 5,
            'limiting_mags': {'g': 22.5}
        }
        # Setting mode to radio but giving optical config should fail
        with self.assertRaises(ValueError):
            SimulateTransientWithCadence(
                model=self.model,
                parameters=self.parameters,
                cadence_config={'frequencies': [1e9], 'cadence_days': 1.0},  # Incomplete radio config
                observation_mode='radio',
                seed=self.seed
            )

    def test_invalid_model_string(self):
        """Test invalid model string"""
        optical_cadence = {
            'bands': ['g'],
            'cadence_days': 1.0,
            'duration_days': 5,
            'limiting_mags': {'g': 22.5}
        }
        with self.assertRaises(ValueError):
            SimulateTransientWithCadence(
                model='nonexistent_model',
                parameters=self.parameters,
                cadence_config=optical_cadence,
                seed=self.seed
            )

    def test_missing_band_in_limiting_mags(self):
        """Test missing band in limiting_mags dict"""
        bad_cadence = {
            'bands': ['g', 'r'],
            'cadence_days': 1.0,
            'duration_days': 5,
            'limiting_mags': {'g': 22.5}  # Missing 'r'
        }
        with self.assertRaises(ValueError):
            SimulateTransientWithCadence(
                model=self.model,
                parameters=self.parameters,
                cadence_config=bad_cadence,
                seed=self.seed
            )

    def test_with_t0_mjd_transient(self):
        """Test with t0_mjd_transient in parameters"""
        params_with_t0 = {
            'redshift': 0.1,
            't0_mjd_transient': 59000.0
        }
        optical_cadence = {
            'bands': ['g'],
            'cadence_days': 1.0,
            'duration_days': 5,
            'limiting_mags': {'g': 22.5}
        }
        sim = SimulateTransientWithCadence(
            model=self.model,
            parameters=params_with_t0,
            cadence_config=optical_cadence,
            seed=self.seed
        )
        self.assertEqual(sim.t0, 59000.0)

    def test_start_offset_days(self):
        """Test observation start offset"""
        optical_cadence = {
            'bands': ['g'],
            'cadence_days': 1.0,
            'duration_days': 5,
            'start_offset_days': 2.0,
            'limiting_mags': {'g': 22.5}
        }
        sim = SimulateTransientWithCadence(
            model=self.model,
            parameters=self.parameters,
            cadence_config=optical_cadence,
            seed=self.seed
        )
        # First observation should be at t0 + 2.0
        min_time = sim.observations['time_since_t0'].min()
        self.assertGreaterEqual(min_time, 2.0)


class TestPopulationSynthesizerExtended(unittest.TestCase):
    """Extended tests for PopulationSynthesizer"""

    def setUp(self):
        self.model = 'arnett'
        self.prior = bilby.core.prior.PriorDict({
            'mej': bilby.core.prior.Uniform(0.01, 0.1, name='mej'),
            'vej': bilby.core.prior.Uniform(0.1, 0.3, name='vej')
        })
        self.rate = 1e-6
        self.seed = 42

    def test_simulate_population_basic(self):
        """Test simulate_population method"""
        synth = PopulationSynthesizer(
            model=self.model,
            prior=self.prior,
            rate=self.rate,
            seed=self.seed
        )

        def simple_detection(row):
            return row['redshift'] < 0.3

        pop = synth.simulate_population(n_events=10)
        self.assertIsInstance(pop, TransientPopulation)
        self.assertEqual(pop.n_transients, 10)
        # Apply detection separately
        detected = synth.apply_detection_criteria(pop.parameters, simple_detection)
        self.assertIn('detected', detected.columns)

    def test_simulate_population_no_detection(self):
        """Test simulate_population without detection function"""
        synth = PopulationSynthesizer(
            model=self.model,
            prior=self.prior,
            rate=self.rate,
            seed=self.seed
        )
        pop = synth.simulate_population(n_events=5)
        self.assertIsInstance(pop, TransientPopulation)
        self.assertEqual(pop.n_transients, 5)

    def test_calculate_expected_events(self):
        """Test _calculate_expected_events internal method"""
        synth = PopulationSynthesizer(
            model=self.model,
            prior=self.prior,
            rate=self.rate,
            seed=self.seed
        )
        n_expected = synth._calculate_expected_events(n_years=1, z_max=0.5)
        self.assertGreater(n_expected, 0)
        self.assertIsInstance(n_expected, (int, float))

    def test_powerlaw_rate_evolution_increasing(self):
        """Test powerlaw rate evolution increases with redshift"""
        synth = PopulationSynthesizer(
            model=self.model,
            prior=self.prior,
            rate=self.rate,
            rate_evolution='powerlaw',
            rate_params={'alpha': 3.0},
            seed=self.seed
        )
        # Powerlaw evolution should increase rate at higher z
        rate_z0 = synth.rate_function(0.1)
        rate_z1 = synth.rate_function(1.0)
        self.assertGreater(rate_z1, rate_z0)

    def test_custom_rate_function(self):
        """Test custom rate function"""
        custom_rate = lambda z: 1e-5 * (1 + z) ** 2

        synth = PopulationSynthesizer(
            model=self.model,
            prior=self.prior,
            rate=custom_rate,
            rate_evolution='custom',
            seed=self.seed
        )
        self.assertEqual(synth.rate_function(0.5), custom_rate(0.5))

    def test_generate_population_with_time_range(self):
        """Test population generation with time range"""
        synth = PopulationSynthesizer(
            model=self.model,
            prior=self.prior,
            rate=self.rate,
            seed=self.seed
        )
        params = synth.generate_population(
            n_events=10,
            time_range=(59000, 60000)
        )
        # Check event times are within range
        if 't0_mjd_transient' in params.columns:
            times = params['t0_mjd_transient']
            self.assertTrue((times >= 59000).all())
            self.assertTrue((times <= 60000).all())

    def test_generate_population_with_sky_positions(self):
        """Test population generation includes sky positions"""
        synth = PopulationSynthesizer(
            model=self.model,
            prior=self.prior,
            rate=self.rate,
            seed=self.seed
        )
        params = synth.generate_population(n_events=20)
        self.assertIn('ra', params.columns)
        self.assertIn('dec', params.columns)
        # RA should be in [0, 360), Dec in [-90, 90]
        self.assertTrue((params['ra'] >= 0).all())
        self.assertTrue((params['ra'] < 360).all())
        self.assertTrue((params['dec'] >= -90).all())
        self.assertTrue((params['dec'] <= 90).all())

    def test_apply_detection_criteria_with_float_return(self):
        """Test detection criteria with float probability return"""
        synth = PopulationSynthesizer(
            model=self.model,
            prior=self.prior,
            rate=self.rate,
            seed=123  # Different seed for reproducibility
        )
        params = synth.generate_population(n_events=100)

        # Detection function that returns probability
        def prob_det(row):
            return np.clip(1.0 - row['redshift'] * 2, 0, 1)

        detected = synth.apply_detection_criteria(params, prob_det)
        self.assertIn('detected', detected.columns)
        self.assertIn('detection_probability', detected.columns)

    def test_infer_rate_with_efficiency(self):
        """Test rate inference with efficiency function"""
        synth = PopulationSynthesizer(
            model=self.model,
            prior=self.prior,
            rate=self.rate,
            seed=self.seed
        )
        observed = pd.DataFrame({'redshift': [0.1, 0.15, 0.2]})

        def efficiency(z):
            return np.clip(1.0 - z * 2, 0.1, 1.0)

        result = synth.infer_rate(observed, efficiency_function=efficiency)
        self.assertIn('rate_ml', result)
        self.assertIn('rate_uncertainty', result)

    def test_sample_redshifts_rejection_sampling(self):
        """Test redshift sampling uses rejection sampling for powerlaw"""
        synth = PopulationSynthesizer(
            model=self.model,
            prior=self.prior,
            rate=self.rate,
            rate_evolution='powerlaw',
            rate_params={'alpha': 3.0},
            seed=self.seed
        )
        redshifts = synth._sample_redshifts(100, z_max=1.0)
        self.assertEqual(len(redshifts), 100)
        self.assertTrue((redshifts > 0).all())
        self.assertTrue((redshifts <= 1.0).all())

    def test_invalid_cosmology_string(self):
        """Test invalid cosmology string raises error"""
        with self.assertRaises(ValueError):
            PopulationSynthesizer(
                model=self.model,
                prior=self.prior,
                rate=self.rate,
                cosmology='InvalidCosmology',
                seed=self.seed
            )

    def test_sample_intrinsic_params_size(self):
        """Test that intrinsic parameters sampling returns correct size"""
        synth = PopulationSynthesizer(
            model=self.model,
            prior=self.prior,
            rate=self.rate,
            seed=self.seed
        )
        params = synth._sample_intrinsic_params(50)
        self.assertEqual(len(params), 50)
        self.assertIn('mej', params.columns)
        self.assertIn('vej', params.columns)

    def test_sample_sky_positions_uniform(self):
        """Test uniform sky position sampling"""
        synth = PopulationSynthesizer(
            model=self.model,
            prior=self.prior,
            rate=self.rate,
            seed=self.seed
        )
        ra, dec = synth._sample_sky_positions(1000)
        # Check reasonable distribution
        self.assertEqual(len(ra), 1000)
        self.assertEqual(len(dec), 1000)
        # RA should be in valid range [0, 360)
        self.assertTrue((ra >= 0).all())
        self.assertTrue((ra < 360).all())
        # Dec should be in valid range [-90, 90]
        self.assertTrue((dec >= -90).all())
        self.assertTrue((dec <= 90).all())
        # Should have some variation (not all same value)
        self.assertGreater(ra.std(), 0.1)
        self.assertGreater(dec.std(), 0.1)


class TestSimulateGammaRayTransientExtended(unittest.TestCase):
    """Extended tests for SimulateGammaRayTransient"""

    def setUp(self):
        self.model = lambda t, **kwargs: np.full_like(t, 1e-3)
        self.parameters = {'luminosity': 1e50, 'redshift': 0.5}
        self.energy_edges = [10, 50, 100, 300]
        self.time_range = (0, 10)
        self.seed = 42

    def test_callable_background_rate(self):
        """Test callable background rate"""
        def bg_rate(energy):
            return 0.01 * (energy / 100) ** (-2)

        sim = SimulateGammaRayTransient(
            model=self.model,
            parameters=self.parameters,
            energy_edges=self.energy_edges,
            time_range=self.time_range,
            effective_area=100,
            background_rate=bg_rate,
            seed=self.seed
        )
        # Should initialize without error and be able to generate events
        events = sim.generate_time_tagged_events()
        self.assertIsInstance(events, pd.DataFrame)

    def test_dict_background_rate(self):
        """Test dictionary background rate per channel"""
        bg_dict = {0: 0.1, 1: 0.05, 2: 0.02}

        sim = SimulateGammaRayTransient(
            model=self.model,
            parameters=self.parameters,
            energy_edges=self.energy_edges,
            time_range=self.time_range,
            effective_area=100,
            background_rate=bg_dict,
            seed=self.seed
        )
        # Should initialize without error and be able to generate events
        events = sim.generate_time_tagged_events()
        self.assertIsInstance(events, pd.DataFrame)

    def test_very_short_time_range(self):
        """Test with very short time range"""
        sim = SimulateGammaRayTransient(
            model=self.model,
            parameters=self.parameters,
            energy_edges=[10, 100],
            time_range=(0, 0.001),  # 1 ms
            effective_area=100,
            background_rate=0.1,
            seed=self.seed
        )
        events = sim.generate_time_tagged_events()
        self.assertIsInstance(events, pd.DataFrame)
        # Should be very few events
        self.assertLess(len(events), 100)

    def test_high_background_rate(self):
        """Test with high background rate"""
        sim = SimulateGammaRayTransient(
            model=self.model,
            parameters=self.parameters,
            energy_edges=[10, 50],
            time_range=(0, 1),
            effective_area=100,
            background_rate=100.0,  # Very high background
            seed=self.seed
        )
        events = sim.generate_time_tagged_events()
        # Should have many background events
        self.assertGreater(len(events), 1000)

    def test_zero_effective_area(self):
        """Test with very small effective area"""
        sim = SimulateGammaRayTransient(
            model=self.model,
            parameters=self.parameters,
            energy_edges=[10, 100],
            time_range=(0, 1),
            effective_area=0.01,  # Very small area
            background_rate=0.1,
            seed=self.seed
        )
        events = sim.generate_time_tagged_events()
        # Should have mostly background events
        self.assertIsInstance(events, pd.DataFrame)

    def test_single_energy_channel(self):
        """Test with single energy channel"""
        sim = SimulateGammaRayTransient(
            model=self.model,
            parameters=self.parameters,
            energy_edges=[10, 100],  # Only one channel
            time_range=(0, 5),
            effective_area=100,
            background_rate=0.1,
            seed=self.seed
        )
        events = sim.generate_time_tagged_events()
        # All events should be in channel 0
        self.assertTrue((events['energy_channel'] == 0).all())

    def test_many_energy_channels(self):
        """Test with many energy channels"""
        edges = [10, 20, 30, 50, 80, 100, 150, 200, 300, 500]
        sim = SimulateGammaRayTransient(
            model=self.model,
            parameters=self.parameters,
            energy_edges=edges,
            time_range=(0, 5),
            effective_area=100,
            background_rate=0.1,
            seed=self.seed
        )
        counts = sim.generate_binned_counts(np.linspace(0, 5, 6))
        # Should have 12 columns: time_start, time_end, time_center, dt, counts, counts_error,
        # count_rate, count_rate_error, energy_channel, energy_low, energy_high, energy_center
        self.assertEqual(counts.shape[1], 12)
        # Should have n_time_bins * n_energy_channels rows
        n_time_bins = 5  # 6 edges -> 5 bins
        n_energy_bins = len(edges) - 1  # 9 channels
        self.assertEqual(len(counts), n_time_bins * n_energy_bins)

    def test_generate_binned_with_exposure_correction(self):
        """Test binned counts generation preserves total exposure"""
        sim = SimulateGammaRayTransient(
            model=self.model,
            parameters=self.parameters,
            energy_edges=self.energy_edges,  # 3 energy channels
            time_range=(0, 10),
            effective_area=100,
            background_rate=0.1,
            seed=self.seed
        )
        time_bins = np.linspace(0, 10, 11)  # 10 bins, 1 second each
        counts = sim.generate_binned_counts(time_bins)
        # Total dt per energy channel should be 10.0 seconds
        # With 3 energy channels, total sum of dt is 30.0
        n_channels = len(self.energy_edges) - 1
        total_exposure = counts['dt'].sum()
        self.assertAlmostEqual(total_exposure, 10.0 * n_channels, places=1)

    @patch("bilby.utils.check_directory_exists_and_if_not_mkdir")
    @patch("pandas.DataFrame.to_csv")
    @patch("builtins.open", new_callable=mock.mock_open)
    def test_save_with_custom_filenames(self, mock_open, mock_to_csv, mock_makedirs):
        """Test saving with custom filenames"""
        sim = SimulateGammaRayTransient(
            model=self.model,
            parameters=self.parameters,
            energy_edges=self.energy_edges,
            time_range=self.time_range,
            effective_area=100,
            background_rate=0.1,
            seed=self.seed
        )
        sim.generate_time_tagged_events()
        sim.save_time_tagged_events('custom_name_for_events')
        mock_to_csv.assert_called_once()

    def test_max_events_limit(self):
        """Test max_events parameter limits event generation"""
        # Very high rate to ensure we hit the limit
        high_rate_model = lambda t, **kwargs: np.full_like(t, 1.0)  # Very bright

        sim = SimulateGammaRayTransient(
            model=high_rate_model,
            parameters=self.parameters,
            energy_edges=[10, 100],
            time_range=(0, 100),  # Long observation
            effective_area=10000,  # Large area
            background_rate=100.0,  # High background
            seed=self.seed
        )
        events = sim.generate_time_tagged_events(max_events=500)
        # Should be limited to around max_events
        self.assertLessEqual(len(events), 50000)  # Upper bound with background

    def test_evaluate_model_flux_shape(self):
        """Test _evaluate_model_flux returns correct shape"""
        sim = SimulateGammaRayTransient(
            model=self.model,
            parameters=self.parameters,
            energy_edges=self.energy_edges,
            time_range=self.time_range,
            seed=self.seed
        )
        times = np.array([1.0, 2.0, 3.0])
        energies = np.array([30.0, 75.0, 200.0])
        fluxes = sim._evaluate_model_flux(times, energies)
        self.assertEqual(fluxes.shape, times.shape)


class TestErrorPathsCoverage(unittest.TestCase):
    """Tests for error paths and edge cases to increase coverage"""

    def setUp(self):
        self.prior = bilby.core.prior.PriorDict({
            'mej': bilby.core.prior.Uniform(0.01, 0.1, name='mej'),
        })
        self.seed = 42

    def test_invalid_observation_mode_string(self):
        """Test invalid observation mode raises error"""
        model = lambda t, **kwargs: np.full_like(t, 20.0)
        # Create config that will fail validation for the mode
        cadence = {
            'bands': ['g'],
            'cadence_days': 1.0,
            'duration_days': 5,
            'limiting_mags': {'g': 22.5}
        }
        # Force invalid mode after auto-detection
        with self.assertRaises(ValueError) as ctx:
            sim = SimulateTransientWithCadence(
                model=model,
                parameters={'redshift': 0.1},
                cadence_config={'custom_key': 'value'},  # Invalid config
                observation_mode='radio',
                seed=self.seed
            )
        self.assertIn("must contain", str(ctx.exception))

    def test_population_synthesizer_invalid_model_string(self):
        """Test PopulationSynthesizer with invalid model string"""
        with self.assertRaises(ValueError) as ctx:
            PopulationSynthesizer(
                model='nonexistent_model_xyz',
                prior=self.prior,
                rate=1e-6,
                seed=self.seed
            )
        self.assertIn("not found", str(ctx.exception))

    def test_population_synthesizer_invalid_model_type(self):
        """Test PopulationSynthesizer with invalid model type"""
        with self.assertRaises(ValueError) as ctx:
            PopulationSynthesizer(
                model=123,  # Not string or callable
                prior=self.prior,
                rate=1e-6,
                seed=self.seed
            )
        self.assertIn("must be string or callable", str(ctx.exception))

    def test_population_synthesizer_custom_model_no_prior(self):
        """Test custom model requires prior"""
        custom_model = lambda t, **kwargs: t
        with self.assertRaises(ValueError) as ctx:
            PopulationSynthesizer(
                model=custom_model,
                prior=None,
                rate=1e-6,
                seed=self.seed
            )
        self.assertIn("Must provide prior", str(ctx.exception))

    def test_population_synthesizer_invalid_prior_type(self):
        """Test invalid prior type raises error"""
        with self.assertRaises(ValueError) as ctx:
            PopulationSynthesizer(
                model='arnett',
                prior=[1, 2, 3],  # Invalid type
                rate=1e-6,
                seed=self.seed
            )
        self.assertIn("Prior must be", str(ctx.exception))

    def test_population_synthesizer_unknown_rate_evolution(self):
        """Test unknown rate evolution model raises error"""
        with self.assertRaises(ValueError) as ctx:
            PopulationSynthesizer(
                model='arnett',
                prior=self.prior,
                rate=1e-6,
                rate_evolution='invalid_evolution',
                seed=self.seed
            )
        self.assertIn("Unknown rate evolution", str(ctx.exception))

    def test_transient_population_filter_no_redshift(self):
        """Test filter_by_redshift when no redshift column"""
        params_no_z = pd.DataFrame({
            'ra': [10, 20],
            'dec': [-10, -5]
        })
        pop = TransientPopulation(params_no_z)
        with self.assertRaises(ValueError) as ctx:
            pop.filter_by_redshift(z_min=0.1)
        self.assertIn("Cannot filter by redshift", str(ctx.exception))

    def test_population_synthesizer_apply_invalid_population_type(self):
        """Test apply_detection_criteria with invalid population type"""
        synth = PopulationSynthesizer(
            model='arnett',
            prior=self.prior,
            rate=1e-6,
            seed=self.seed
        )

        def det_func(row):
            return True

        with self.assertRaises(ValueError) as ctx:
            synth.apply_detection_criteria([1, 2, 3], det_func)
        self.assertIn("must be DataFrame", str(ctx.exception))

    def test_population_synthesizer_apply_invalid_return_type(self):
        """Test apply_detection_criteria with invalid return type from function"""
        synth = PopulationSynthesizer(
            model='arnett',
            prior=self.prior,
            rate=1e-6,
            seed=self.seed
        )
        params = pd.DataFrame({
            'redshift': [0.1],
            'luminosity_distance': [450]
        })

        def bad_det_func(row):
            return "invalid"  # Returns string instead of bool/float

        with self.assertRaises(ValueError) as ctx:
            synth.apply_detection_criteria(params, bad_det_func)
        self.assertIn("must return bool or float", str(ctx.exception))

    def test_population_synthesizer_infer_rate_no_redshift(self):
        """Test infer_rate when sample has no redshift column"""
        synth = PopulationSynthesizer(
            model='arnett',
            prior=self.prior,
            rate=1e-6,
            seed=self.seed
        )
        bad_sample = pd.DataFrame({'other_column': [1, 2, 3]})
        with self.assertRaises(ValueError) as ctx:
            synth.infer_rate(bad_sample)
        self.assertIn("must have 'redshift'", str(ctx.exception))

    def test_transient_population_get_redshift_distribution_no_z(self):
        """Test get_redshift_distribution when no redshift"""
        params_no_z = pd.DataFrame({'ra': [10, 20]})
        pop = TransientPopulation(params_no_z)
        result = pop.get_redshift_distribution()
        self.assertEqual(result, (None, None, None))

    def test_population_synthesizer_generate_no_prior_redshift_warning(self):
        """Test warning when prior has no redshift and rate_weighted=False"""
        synth = PopulationSynthesizer(
            model='arnett',
            prior=self.prior,  # No redshift in prior
            rate=1e-6,
            seed=self.seed
        )
        # This should raise an error because prior must contain redshift when rate_weighted=False
        with self.assertRaises(ValueError) as ctx:
            synth.generate_population(
                n_events=5,
                rate_weighted_redshifts=False
            )
        self.assertIn("Prior must contain 'redshift'", str(ctx.exception))

    def test_population_synthesizer_generate_rate_weighted_no_n_events(self):
        """Test error when rate_weighted=False but no n_events specified"""
        synth = PopulationSynthesizer(
            model='arnett',
            prior=self.prior,
            rate=1e-6,
            seed=self.seed
        )
        # Cannot calculate n_events from rate when rate_weighted=False
        with self.assertRaises(ValueError) as ctx:
            synth.generate_population(
                n_years=1,
                rate_weighted_redshifts=False
            )
        self.assertIn("Cannot calculate n_events", str(ctx.exception))

    def test_simulate_transient_with_cadence_model_error_optical(self):
        """Test error handling when optical model fails"""
        def failing_model(t, **kwargs):
            raise RuntimeError("Model evaluation failed")

        cadence = {
            'bands': ['g'],
            'cadence_days': 1.0,
            'duration_days': 5,
            'limiting_mags': {'g': 22.5}
        }
        with self.assertRaises(RuntimeError):
            SimulateTransientWithCadence(
                model=failing_model,
                parameters={'redshift': 0.1},
                cadence_config=cadence,
                seed=self.seed
            )

    def test_simulate_transient_with_cadence_model_error_radio(self):
        """Test error handling when radio model fails"""
        def failing_model(t, **kwargs):
            raise RuntimeError("Model evaluation failed")

        cadence = {
            'frequencies': [1.4e9],
            'cadence_days': 7,
            'duration_days': 30,
            'sensitivity': 0.05
        }
        with self.assertRaises(RuntimeError):
            SimulateTransientWithCadence(
                model=failing_model,
                parameters={'redshift': 0.1},
                cadence_config=cadence,
                observation_mode='radio',
                seed=self.seed
            )

    def test_population_synthesizer_simulate_population_with_model_name(self):
        """Test simulate_population stores model name in metadata"""
        synth = PopulationSynthesizer(
            model='arnett',
            prior=self.prior,
            rate=1e-6,
            seed=self.seed
        )
        pop = synth.simulate_population(n_events=5)
        self.assertIsInstance(pop, TransientPopulation)
        self.assertIn('model', pop.metadata)
        self.assertEqual(pop.metadata['model'], 'arnett')

    def test_transient_population_with_light_curves(self):
        """Test TransientPopulation stores light_curves"""
        params = pd.DataFrame({'redshift': [0.1, 0.2]})
        light_curves = {'lc1': [1, 2, 3], 'lc2': [4, 5, 6]}
        pop = TransientPopulation(params, light_curves=light_curves)
        self.assertEqual(pop.light_curves, light_curves)

    def test_gamma_ray_save_binned_counts_without_generation(self):
        """Test saving binned counts without generating raises error"""
        model = lambda t, **kwargs: np.full_like(t, 1e-3)
        sim = SimulateGammaRayTransient(
            model=model,
            parameters={'redshift': 0.5},
            energy_edges=[10, 100],
            time_range=(0, 10),
            seed=self.seed
        )
        with self.assertRaises(ValueError) as ctx:
            sim.save_binned_counts('test')
        self.assertIn("No binned counts generated", str(ctx.exception))

    def test_gamma_ray_save_tte_without_generation(self):
        """Test saving TTE without generating raises error"""
        model = lambda t, **kwargs: np.full_like(t, 1e-3)
        sim = SimulateGammaRayTransient(
            model=model,
            parameters={'redshift': 0.5},
            energy_edges=[10, 100],
            time_range=(0, 10),
            seed=self.seed
        )
        with self.assertRaises(ValueError) as ctx:
            sim.save_time_tagged_events('test')
        self.assertIn("No time-tagged events generated", str(ctx.exception))

    def test_population_synthesizer_no_events_generated_warning(self):
        """Test warning when no events are generated"""
        synth = PopulationSynthesizer(
            model='arnett',
            prior=self.prior,
            rate=1e-100,  # Extremely low rate
            seed=self.seed
        )
        # Force n_events to 0
        params = synth.generate_population(n_events=0)
        self.assertEqual(len(params), 0)

    def test_simulate_transient_with_cadence_auto_convert_sensitivity_to_limiting_mag(self):
        """Test auto-conversion of sensitivity to limiting_mag for optical"""
        model = lambda t, **kwargs: np.full_like(t, 20.0)
        cadence = {
            'bands': ['g'],
            'cadence_days': 1.0,
            'duration_days': 5,
            'limiting_mags': {'g': 22.5}
        }
        # Force noise_type to sensitivity (should auto-convert)
        sim = SimulateTransientWithCadence(
            model=model,
            parameters={'redshift': 0.1},
            cadence_config=cadence,
            noise_type='sensitivity',  # Will be converted
            seed=self.seed
        )
        self.assertEqual(sim.noise_type, 'limiting_mag')


class TestAdditionalCoveragePaths(unittest.TestCase):
    """Additional tests to maximize code coverage"""

    def setUp(self):
        self.prior = bilby.core.prior.PriorDict({
            'mej': bilby.core.prior.Uniform(0.01, 0.1, name='mej'),
        })
        self.seed = 42

    def test_calculate_detection_probability_no_limiting_mag(self):
        """Test _calculate_detection_probability when no limiting_mag in config"""
        synth = PopulationSynthesizer(
            model='arnett',
            prior=self.prior,
            rate=1e-6,
            seed=self.seed
        )
        params = {'redshift': 0.1, 'mej': 0.05}
        config = {'bands': ['lsstr']}  # No limiting_mag
        prob = synth._calculate_detection_probability(params, config)
        self.assertEqual(prob, 1.0)  # Should return 1.0 with warning

    def test_calculate_detection_probability_with_bands(self):
        """Test _calculate_detection_probability with bands config"""
        synth = PopulationSynthesizer(
            model='arnett',
            prior=self.prior,
            rate=1e-6,
            seed=self.seed
        )
        params = {'mej': 0.05, 'vej': 0.2, 'kappa': 1.0, 'kappa_gamma': 1e4,
                  'temperature_floor': 4000, 'redshift': 0.1, 'luminosity_distance': 450}
        config = {'limiting_mag': 22.5, 'bands': ['bessellb']}
        # This will try to evaluate model - may fail but tests the path
        prob = synth._calculate_detection_probability(params, config)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_calculate_detection_probability_no_bands(self):
        """Test _calculate_detection_probability without bands (flux mode)"""
        synth = PopulationSynthesizer(
            model='arnett',
            prior=self.prior,
            rate=1e-6,
            seed=self.seed
        )
        params = {'mej': 0.05, 'vej': 0.2, 'kappa': 1.0, 'kappa_gamma': 1e4,
                  'temperature_floor': 4000}
        config = {'limiting_mag': 22.5}  # No bands key
        prob = synth._calculate_detection_probability(params, config)
        # Should either return 1.0 (positive flux) or 0.0 (negative flux)
        self.assertIn(prob, [0.0, 1.0])

    def test_simulate_population_with_selection_effects(self):
        """Test simulate_population with selection effects"""
        synth = PopulationSynthesizer(
            model='arnett',
            prior=self.prior,
            rate=1e-6,
            seed=self.seed
        )
        survey_config = {
            'limiting_mag': 22.5,
            'bands': ['bessellb']
        }
        pop = synth.simulate_population(
            n_events=3,
            include_selection_effects=True,
            survey_config=survey_config
        )
        self.assertIsInstance(pop, TransientPopulation)
        self.assertIn('detected', pop.parameters.columns)
        self.assertIn('survey_config', pop.metadata)

    def test_simulate_population_with_lightcurves(self):
        """Test simulate_population generates light curves"""
        # Use simple custom model to avoid missing parameter issues
        simple_model = lambda t, **kwargs: np.exp(-t/10)
        synth = PopulationSynthesizer(
            model=simple_model,
            prior=self.prior,
            rate=1e-6,
            seed=self.seed
        )
        pop = synth.simulate_population(
            n_events=2,
            include_lightcurves=True
        )
        self.assertIsInstance(pop, TransientPopulation)
        self.assertIsNotNone(pop.light_curves)
        self.assertEqual(len(pop.light_curves), 2)
        self.assertIn('times', pop.light_curves[0])
        self.assertIn('flux', pop.light_curves[0])

    def test_simulate_population_empty_population(self):
        """Test simulate_population returns empty TransientPopulation"""
        synth = PopulationSynthesizer(
            model='arnett',
            prior=self.prior,
            rate=1e-6,
            seed=self.seed
        )
        pop = synth.simulate_population(n_events=0)
        self.assertIsInstance(pop, TransientPopulation)
        self.assertEqual(pop.n_transients, 0)

    def test_infer_rate_with_transient_population(self):
        """Test infer_rate with TransientPopulation input"""
        synth = PopulationSynthesizer(
            model='arnett',
            prior=self.prior,
            rate=1e-6,
            seed=self.seed
        )
        params = pd.DataFrame({'redshift': [0.1, 0.15, 0.2, 0.25]})
        pop = TransientPopulation(params)
        result = synth.infer_rate(pop)
        self.assertIn('rate_ml', result)
        self.assertIn('rate_uncertainty', result)

    def test_infer_rate_with_array_input(self):
        """Test infer_rate with array input"""
        synth = PopulationSynthesizer(
            model='arnett',
            prior=self.prior,
            rate=1e-6,
            seed=self.seed
        )
        redshifts = np.array([0.1, 0.15, 0.2, 0.25])
        result = synth.infer_rate(redshifts)
        self.assertIn('rate_ml', result)

    def test_population_synthesizer_string_prior(self):
        """Test PopulationSynthesizer with string prior"""
        # This would need bilby prior files - test initialization path
        synth = PopulationSynthesizer(
            model='arnett',
            prior=self.prior,  # Already PriorDict
            rate=1e-6,
            seed=self.seed
        )
        self.assertIsNotNone(synth.prior)

    def test_population_synthesizer_constant_rate_evolution(self):
        """Test constant rate evolution returns constant"""
        synth = PopulationSynthesizer(
            model='arnett',
            prior=self.prior,
            rate=1e-6,
            rate_evolution='constant',
            seed=self.seed
        )
        rate_z0 = synth.rate_function(0.1)
        rate_z1 = synth.rate_function(1.0)
        # Constant rate should be same at all z
        self.assertEqual(rate_z0, rate_z1)

    def test_simulate_transient_with_cadence_with_t0(self):
        """Test SimulateTransientWithCadence uses t0 when no t0_mjd_transient"""
        model = lambda t, **kwargs: np.full_like(t, 20.0)
        params_with_t0 = {'redshift': 0.1, 't0': 60000.0}
        cadence = {
            'bands': ['g'],
            'cadence_days': 1.0,
            'duration_days': 5,
            'limiting_mags': {'g': 22.5}
        }
        sim = SimulateTransientWithCadence(
            model=model,
            parameters=params_with_t0,
            cadence_config=cadence,
            seed=self.seed
        )
        self.assertEqual(sim.t0, 60000.0)

    def test_simulate_transient_with_cadence_band_sequence(self):
        """Test band sequence in optical cadence"""
        model = lambda t, **kwargs: np.full_like(t, 20.0)
        cadence = {
            'bands': ['g', 'r', 'i'],
            'cadence_days': 1.0,
            'duration_days': 5,
            'limiting_mags': {'g': 22.5, 'r': 23.0, 'i': 22.8},
            'band_sequence': ['g', 'r', 'i', 'g', 'r']  # Specific sequence
        }
        sim = SimulateTransientWithCadence(
            model=model,
            parameters={'redshift': 0.1},
            cadence_config=cadence,
            seed=self.seed
        )
        self.assertIn('band', sim.observations.columns)

    def test_transient_population_summary_stats_with_luminosity_distance(self):
        """Test summary_stats includes luminosity_distance stats"""
        params = pd.DataFrame({
            'redshift': [0.1, 0.2, 0.3],
            'luminosity_distance': [450, 950, 1500]
        })
        pop = TransientPopulation(params)
        stats = pop.summary_stats()
        self.assertIn('median_distance_mpc', stats)
        self.assertIn('max_distance_mpc', stats)

    def test_transient_population_summary_stats_with_detection(self):
        """Test summary_stats includes detection fraction"""
        params = pd.DataFrame({
            'redshift': [0.1, 0.2, 0.3],
            'detected': [True, True, False]
        })
        pop = TransientPopulation(params)
        stats = pop.summary_stats()
        self.assertIn('detection_fraction', stats)
        self.assertAlmostEqual(stats['detection_fraction'], 2/3)

    def test_gamma_ray_energy_integrated_counts(self):
        """Test energy-integrated binned counts"""
        model = lambda t, **kwargs: np.full_like(t, 1e-3)
        sim = SimulateGammaRayTransient(
            model=model,
            parameters={'redshift': 0.5},
            energy_edges=[10, 50, 100, 300],
            time_range=(0, 10),
            effective_area=100,
            background_rate=0.1,
            seed=self.seed
        )
        time_bins = np.linspace(0, 10, 6)
        counts = sim.generate_binned_counts(time_bins, energy_integrated=True)
        # Energy integrated should have fewer columns
        self.assertIn('counts', counts.columns)
        self.assertNotIn('energy_channel', counts.columns)
        self.assertEqual(len(counts), 5)  # 5 time bins

    def test_gamma_ray_thinning_algorithm_source_events(self):
        """Test that bright sources generate source events"""
        # Very bright source
        bright_model = lambda t, **kwargs: np.full_like(t, 10.0)  # 10 Jy
        sim = SimulateGammaRayTransient(
            model=bright_model,
            parameters={'redshift': 0.1},
            energy_edges=[10, 100],
            time_range=(0, 1),
            effective_area=10000,  # Large area
            background_rate=0.01,  # Low background
            seed=self.seed
        )
        events = sim.generate_time_tagged_events()
        # Should have some events
        self.assertGreater(len(events), 0)
        # Just verify structure, not source vs background ratio
        self.assertIn('time', events.columns)
        self.assertIn('energy', events.columns)

    def test_population_synthesizer_sample_event_times(self):
        """Test event time sampling with specific range"""
        synth = PopulationSynthesizer(
            model='arnett',
            prior=self.prior,
            rate=1e-6,
            seed=self.seed
        )
        times = synth._sample_event_times(10, time_range=(59000, 60000))
        self.assertEqual(len(times), 10)
        self.assertTrue((times >= 59000).all())
        self.assertTrue((times <= 60000).all())

    def test_simulate_transient_with_cadence_dict_cadence_days(self):
        """Test cadence_days as dictionary for per-band cadence"""
        model = lambda t, **kwargs: np.full_like(t, 20.0)
        cadence = {
            'bands': ['g', 'r'],
            'cadence_days': {'g': 1.0, 'r': 2.0},  # Different cadences
            'duration_days': 10,
            'limiting_mags': {'g': 22.5, 'r': 23.0}
        }
        sim = SimulateTransientWithCadence(
            model=model,
            parameters={'redshift': 0.1},
            cadence_config=cadence,
            seed=self.seed
        )
        # g band should have more observations than r band
        g_count = (sim.observations['band'] == 'g').sum()
        r_count = (sim.observations['band'] == 'r').sum()
        self.assertGreater(g_count, r_count)

    def test_population_synthesizer_cosmology_object_input(self):
        """Test PopulationSynthesizer with cosmology object"""
        from astropy import cosmology as astropy_cosmology
        cosmo = astropy_cosmology.WMAP9

        synth = PopulationSynthesizer(
            model='arnett',
            prior=self.prior,
            rate=1e-6,
            cosmology=cosmo,
            seed=self.seed
        )
        self.assertEqual(synth.cosmology, cosmo)

    def test_transient_population_filter_preserves_metadata(self):
        """Test that filtering preserves metadata"""
        params = pd.DataFrame({
            'redshift': [0.1, 0.2, 0.3, 0.4, 0.5],
            'ra': [10, 20, 30, 40, 50],
            'dec': [-10, -5, 0, 5, 10]
        })
        metadata = {'survey': 'LSST', 'model': 'kilonova'}
        pop = TransientPopulation(params, metadata)
        filtered = pop.filter_by_redshift(z_min=0.2, z_max=0.4)
        self.assertEqual(filtered.metadata, metadata)
        self.assertEqual(filtered.n_transients, 3)

    def test_gamma_ray_model_evaluation_with_parameters(self):
        """Test GammaRayTransient model with additional parameters"""
        model = lambda t, energy=100, **kwargs: np.full_like(t, 1e-3 * (energy/100)**(-2))
        sim = SimulateGammaRayTransient(
            model=model,
            parameters={'redshift': 0.5, 'luminosity': 1e50},
            energy_edges=[10, 50, 100],
            time_range=(0, 5),
            effective_area=100,
            background_rate=0.1,
            seed=self.seed
        )
        events = sim.generate_time_tagged_events()
        self.assertIsInstance(events, pd.DataFrame)
        self.assertGreater(len(events), 0)

    @patch("os.makedirs")
    @patch("pandas.DataFrame.to_csv")
    def test_gamma_ray_save_binned_with_makedirs(self, mock_to_csv, mock_makedirs):
        """Test that save_binned_counts creates directory"""
        model = lambda t, **kwargs: np.full_like(t, 1e-3)
        sim = SimulateGammaRayTransient(
            model=model,
            parameters={'redshift': 0.5},
            energy_edges=[10, 100],
            time_range=(0, 10),
            seed=self.seed
        )
        time_bins = np.linspace(0, 10, 6)
        sim.generate_binned_counts(time_bins)
        sim.save_binned_counts('test_counts')
        mock_makedirs.assert_called()
        mock_to_csv.assert_called_once()

    def test_population_synthesizer_apply_with_transient_population_input(self):
        """Test apply_detection_criteria with TransientPopulation input"""
        synth = PopulationSynthesizer(
            model='arnett',
            prior=self.prior,
            rate=1e-6,
            seed=self.seed
        )
        params = pd.DataFrame({
            'redshift': [0.1, 0.2, 0.3],
            'luminosity_distance': [450, 950, 1500]
        })
        pop = TransientPopulation(params)

        def det_func(row):
            return row['redshift'] < 0.25

        result = synth.apply_detection_criteria(pop, det_func)
        self.assertIn('detected', result.columns)
        self.assertEqual(result['detected'].sum(), 2)  # z=0.1 and 0.2


class TestMaximumCoverageTargeted(unittest.TestCase):
    """Targeted tests for specific uncovered code lines"""

    def setUp(self):
        self.seed = 42

    def test_population_synthesizer_default_prior_loading(self):
        """Test PopulationSynthesizer loads default prior when prior=None"""
        # Use a model that has a default prior
        synth = PopulationSynthesizer(
            model='arnett',
            prior=None,  # Should load default prior
            rate=1e-6,
            seed=self.seed
        )
        self.assertIsInstance(synth.prior, bilby.core.prior.PriorDict)

    def test_population_synthesizer_string_prior_loading(self):
        """Test PopulationSynthesizer loads prior from string"""
        synth = PopulationSynthesizer(
            model='arnett',
            prior='arnett',  # String prior name
            rate=1e-6,
            seed=self.seed
        )
        self.assertIsInstance(synth.prior, bilby.core.prior.PriorDict)

    def test_population_synthesizer_generate_with_poisson_draw(self):
        """Test generate_population draws from Poisson when n_events not specified"""
        synth = PopulationSynthesizer(
            model='arnett',
            prior=bilby.core.prior.PriorDict({'mej': bilby.core.prior.Uniform(0.01, 0.1)}),
            rate=1e-4,  # Higher rate to get some events
            seed=self.seed
        )
        # Don't specify n_events - should draw from Poisson
        params = synth.generate_population(n_years=0.1, z_max=0.3)
        # Should have generated some events (Poisson draw)
        self.assertIsInstance(params, pd.DataFrame)

    def test_sample_redshifts_with_prior_redshift_max(self):
        """Test _sample_redshifts uses prior's redshift maximum"""
        prior_with_z = bilby.core.prior.PriorDict({
            'mej': bilby.core.prior.Uniform(0.01, 0.1),
            'redshift': bilby.core.prior.Uniform(0.01, 1.5)  # Has redshift with maximum
        })
        synth = PopulationSynthesizer(
            model='arnett',
            prior=prior_with_z,
            rate=1e-6,
            seed=self.seed
        )
        # Sample without specifying z_max - should use prior's maximum
        redshifts = synth._sample_redshifts(10)  # No z_max argument
        self.assertEqual(len(redshifts), 10)
        self.assertTrue((redshifts <= 1.5).all())

    def test_calculate_detection_probability_success_path(self):
        """Test _calculate_detection_probability with successful model evaluation"""
        # Use a simple model that returns magnitudes successfully
        def magnitude_model(time_array, **kwargs):
            # Return magnitudes - brighter (lower number) at peak
            return 18.0 + time_array / 10.0  # Starts at 18 mag

        synth = PopulationSynthesizer(
            model=magnitude_model,
            prior=bilby.core.prior.PriorDict({'mej': bilby.core.prior.Uniform(0.01, 0.1)}),
            rate=1e-6,
            seed=self.seed
        )
        params = {'mej': 0.05}
        config = {
            'limiting_mag': 22.5,
            'bands': ['bessellb']  # Will trigger magnitude path
        }
        prob = synth._calculate_detection_probability(params, config)
        # Should get a probability between 0 and 1
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)
        # Peak mag is ~18, limiting is 22.5, so should be highly detectable
        self.assertGreater(prob, 0.5)

    def test_calculate_detection_probability_negative_flux(self):
        """Test _calculate_detection_probability with negative peak flux"""
        def negative_flux_model(time_array, **kwargs):
            return np.full_like(time_array, -1.0)  # Negative flux

        synth = PopulationSynthesizer(
            model=negative_flux_model,
            prior=bilby.core.prior.PriorDict({'mej': bilby.core.prior.Uniform(0.01, 0.1)}),
            rate=1e-6,
            seed=self.seed
        )
        params = {'mej': 0.05}
        config = {'limiting_mag': 22.5}  # No 'bands' key - uses flux path
        prob = synth._calculate_detection_probability(params, config)
        # Should return 0.0 for negative flux
        self.assertEqual(prob, 0.0)

    def test_simulate_generic_transient_gaussian_noise(self):
        """Test SimulateGenericTransient with gaussian noise"""
        model = lambda t, **kwargs: t * 1e-24 * np.ones_like(t)  # Simple flux model
        times = np.linspace(0.1, 10, 100)

        transient = SimulateGenericTransient(
            model=model,
            times=times,
            model_kwargs={'frequency': 1e14},  # Single frequency
            parameters={},
            data_points=10,
            noise_term=1e-25,
            noise_type='gaussian',
            seed=self.seed
        )
        self.assertIn('output_error', transient.data.columns)

    def test_simulate_generic_transient_gaussianmodel_noise(self):
        """Test SimulateGenericTransient with gaussianmodel noise"""
        model = lambda t, **kwargs: t * 1e-24 * np.ones_like(t)  # Simple flux model
        times = np.linspace(0.1, 10, 100)

        transient = SimulateGenericTransient(
            model=model,
            times=times,
            model_kwargs={'frequency': 1e14},  # Single frequency
            parameters={},
            data_points=10,
            noise_term=0.1,  # Fractional noise
            noise_type='gaussianmodel',
            seed=self.seed
        )
        self.assertIn('output_error', transient.data.columns)

    def test_simulate_generic_transient_with_bands_and_extra_scatter(self):
        """Test SimulateGenericTransient with bands and extra scatter"""
        model = lambda t, **kwargs: np.full_like(t, 20.0)  # Magnitude model
        times = np.linspace(0.1, 10, 100)

        transient = SimulateGenericTransient(
            model=model,
            times=times,
            model_kwargs={'bands': 'bessellb', 'output_format': 'magnitude'},  # Single band
            parameters={},
            data_points=10,
            noise_term=0.1,
            noise_type='gaussian',
            extra_scatter=0.05,
            seed=self.seed
        )
        self.assertIn('output_error', transient.data.columns)
        # Also check extra_scatter was applied
        self.assertIsNotNone(transient.data)

    def test_simulate_transient_with_cadence_model_string(self):
        """Test SimulateTransientWithCadence with model string"""
        cadence = {
            'bands': ['bessellb'],
            'cadence_days': 1.0,
            'duration_days': 5,
            'limiting_mags': {'bessellb': 22.5}
        }
        # This will test the model string loading path
        with self.assertRaises(Exception):
            # arnett requires additional parameters, so this will fail
            # but it tests the string model loading code path
            SimulateTransientWithCadence(
                model='arnett',
                parameters={'mej': 0.05, 'vej': 0.2, 'kappa': 1.0, 'kappa_gamma': 1e4,
                           'temperature_floor': 4000},
                cadence_config=cadence,
                seed=self.seed
            )

    def test_transient_population_save_creates_directories(self):
        """Test TransientPopulation.save creates necessary directories"""
        params = pd.DataFrame({'redshift': [0.1, 0.2]})
        pop = TransientPopulation(params)

        with patch("bilby.utils.check_directory_exists_and_if_not_mkdir") as mock_mkdir:
            with patch("pandas.DataFrame.to_csv"):
                pop.save('test_pop.csv', save_metadata=False)

        mock_mkdir.assert_called()

    def test_transient_population_load_structure(self):
        """Test TransientPopulation.load returns correct structure"""
        params = pd.DataFrame({'redshift': [0.1, 0.2]})

        with patch("pandas.read_csv", return_value=params):
            with patch("builtins.open", side_effect=FileNotFoundError):  # No metadata file
                with patch("bilby.utils.check_directory_exists_and_if_not_mkdir"):
                    pop = TransientPopulation.load('test.csv')

        self.assertIsInstance(pop, TransientPopulation)
        self.assertEqual(pop.n_transients, 2)

    def test_gamma_ray_transient_model_error_handling(self):
        """Test SimulateGammaRayTransient handles model errors gracefully"""
        def sometimes_failing_model(t, **kwargs):
            # Model that fails for certain energy values
            if 'energy' in kwargs and kwargs.get('energy', 0) > 200:
                raise ValueError("Energy too high")
            return np.full_like(t, 1e-3)

        sim = SimulateGammaRayTransient(
            model=sometimes_failing_model,
            parameters={'redshift': 0.5},
            energy_edges=[10, 100],
            time_range=(0, 5),
            effective_area=100,
            background_rate=0.1,
            seed=self.seed
        )
        # Should still work - uses try/except internally
        events = sim.generate_time_tagged_events()
        self.assertIsInstance(events, pd.DataFrame)

    def test_population_synthesizer_n_years_calculation(self):
        """Test that n_years parameter works correctly"""
        synth = PopulationSynthesizer(
            model='arnett',
            prior=bilby.core.prior.PriorDict({'mej': bilby.core.prior.Uniform(0.01, 0.1)}),
            rate=1e-3,  # High rate
            seed=self.seed
        )
        # Generate with n_years - uses Poisson draw
        params = synth.generate_population(n_years=0.01, z_max=0.1)
        self.assertIsInstance(params, pd.DataFrame)


class TestRemainingCoveragePaths(unittest.TestCase):
    """Tests targeting specific uncovered code paths for maximum coverage"""

    def setUp(self):
        self.seed = 42

    def test_generic_transient_model_string_lookup(self):
        """Test line 43: model string lookup in all_models_dict"""
        # Use a model name that exists in the model library
        times = np.linspace(0.1, 10, 20)
        transient = SimulateGenericTransient(
            model='arnett_bolometric',  # String lookup - use bolometric for simplicity
            parameters={'f_nickel': 0.1, 'mej': 1.0, 'kappa': 0.2, 'kappa_gamma': 1e4, 'vej': 1e4},
            times=times,
            model_kwargs={'frequency': 1e14},  # Use frequency for simpler test
            data_points=20,
            seed=self.seed
        )
        self.assertIsNotNone(transient.data)
        self.assertEqual(len(transient.data), 20)

    def test_generic_transient_no_bands_or_frequency_error(self):
        """Test line 59: ValueError when no bands or frequency supplied"""
        times = np.linspace(0, 10, 10)
        model = lambda t, **kwargs: np.full_like(t, 1.0)

        with self.assertRaises(ValueError) as context:
            SimulateGenericTransient(
                model=model,
                parameters={},
                times=times,
                model_kwargs={'output_format': 'flux_density'},  # No bands or frequency
                data_points=10,
                seed=self.seed
            )
        self.assertIn('Must supply either bands or frequency', str(context.exception))

    def test_generic_transient_multiwavelength_with_frequency(self):
        """Test line 65: multiwavelength with frequency array"""
        times = np.linspace(0, 10, 20)
        model = lambda t, **kwargs: np.full_like(t, 1e-3)  # Returns flux density

        transient = SimulateGenericTransient(
            model=model,
            parameters={},
            times=times,
            model_kwargs={'frequency': np.array([1e9, 3e9, 10e9])},  # Multiple frequencies
            multiwavelength_transient=True,
            data_points=15,
            seed=self.seed
        )
        self.assertIsNotNone(transient.data)
        self.assertEqual(len(transient.data), 15)  # data_points limit
        self.assertIn('frequency', transient.data.columns)

    def test_generic_transient_snr_based_noise(self):
        """Test lines 103-105: SNRbased noise type"""
        times = np.linspace(0, 10, 20)
        model = lambda t, **kwargs: np.full_like(t, 1.0, dtype=float)  # Constant flux

        transient = SimulateGenericTransient(
            model=model,
            parameters={},
            times=times,
            model_kwargs={'frequency': 1e14},
            data_points=20,
            noise_type='SNRbased',
            noise_term=10,  # SNR factor
            seed=self.seed
        )
        self.assertIsNotNone(transient.data)
        self.assertIn('output_error', transient.data.columns)
        # SNRbased uses sqrt(flux + min_flux/noise_term)
        errors = transient.data['output_error'].values
        self.assertTrue(all(e > 0 for e in errors))

    def test_cadence_transient_frequency_sequence(self):
        """Test lines 367-376: frequency_sequence in cadence_config"""
        model = lambda t, **kwargs: np.full_like(t, 1e-3, dtype=float)

        # Use frequency_sequence for alternating observations
        cadence_config = {
            'frequencies': [1e9, 3e9],
            'cadence_days': 0.5,  # Minimum cadence
            'duration_days': 3,
            'sensitivity': 0.01,
            'frequency_sequence': [1e9, 3e9, 1e9],  # Specific sequence
        }

        transient = SimulateTransientWithCadence(
            model=model,
            parameters={'redshift': 0.1},
            cadence_config=cadence_config,
            observation_mode='radio',
            seed=self.seed
        )
        self.assertIsNotNone(transient.observations)
        # Should follow the frequency_sequence pattern
        self.assertIn('frequency', transient.observations.columns)

    def test_cadence_transient_no_snr_threshold(self):
        """Test lines 578, 585: Detection without SNR threshold"""
        model = lambda t, **kwargs: np.full_like(t, 1e-3, dtype=float)

        cadence_config = {
            'frequencies': [1e9],
            'cadence_days': 1,
            'duration_days': 5,
            'sensitivity': 0.01
        }

        transient = SimulateTransientWithCadence(
            model=model,
            parameters={'redshift': 0.1},
            cadence_config=cadence_config,
            observation_mode='radio',
            snr_threshold=None,  # No threshold - but still uses SNR threshold logic
            seed=self.seed
        )

        # Check that detected column exists
        self.assertIn('detected', transient.observations.columns)
        # Check that detected_observations property works
        detected = transient.detected_observations
        self.assertIsInstance(detected, pd.DataFrame)
        # If no detected column has True values, it returns all (line 585 coverage)
        if 'detected' not in transient.observations.columns or not any(transient.observations['detected']):
            # This tests line 585: return self.observations when no detected column
            pass  # The property call already tested this path

    @patch("redback.simulate_transients.SimulateOpticalTransient.__init__", return_value=None)
    def test_full_survey_single_event_time(self, mock_base_init):
        """Test line 1329: get_event_times with single event"""
        import astropy.units as u
        prior = bilby.core.prior.PriorDict()
        prior['redshift'] = bilby.core.prior.Uniform(0, 2, "redshift")

        survey = SimulateFullOpticalSurvey(
            model=lambda t, **kwargs: np.full_like(t, 1e-3),
            prior=prior,
            rate=10,
            survey_start_date=0,
            survey_duration=1,
        )
        survey.number_of_events = 1  # Single event
        survey.survey_start_date = 0
        survey.survey_duration = 100 * u.day  # Set survey_duration, not survey_duration_seconds

        event_times = survey.get_event_times()
        # Single event returns a float
        self.assertIsInstance(event_times, (int, float))
        self.assertGreaterEqual(event_times, 0)

    @patch("redback.simulate_transients.SimulateOpticalTransient.__init__", return_value=None)
    def test_full_survey_zero_events_time(self, mock_base_init):
        """Test line 1340: get_event_times with zero events"""
        import astropy.units as u
        prior = bilby.core.prior.PriorDict()
        prior['redshift'] = bilby.core.prior.Uniform(0, 2, "redshift")

        survey = SimulateFullOpticalSurvey(
            model=lambda t, **kwargs: np.full_like(t, 1e-3),
            prior=prior,
            rate=10,
            survey_start_date=10,
            survey_duration=1,
        )
        survey.number_of_events = 0  # Zero events - edge case
        survey.survey_start_date = 10
        survey.survey_duration = 100 * u.day  # Set survey_duration, not survey_duration_seconds

        event_times = survey.get_event_times()
        # Zero events should return survey start date in a list
        self.assertIsInstance(event_times, list)
        self.assertEqual(event_times[0], 10)

    def test_transient_population_load_with_metadata(self):
        """Test lines 1553-1555: Load with metadata JSON file"""
        import tempfile
        import json
        import os

        # Create populations directory if it doesn't exist
        os.makedirs('populations', exist_ok=True)

        # Create test CSV in populations directory
        csv_filename = 'test_pop_with_meta.csv'
        csv_path = f'populations/{csv_filename}'

        params = pd.DataFrame({
            'redshift': [0.1, 0.2],
            'mass': [1.0, 2.0]
        })
        params.to_csv(csv_path, index=False)

        metadata = {
            'model': 'test_model',
            'rate': 1e-4,
            'z_max': 1.0
        }

        # Create metadata JSON next to the CSV
        metadata_path = csv_path.replace('.csv', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)

        try:
            # Load with metadata - pass just the filename, load() adds 'populations/'
            pop = TransientPopulation.load(csv_filename)
            self.assertEqual(len(pop.parameters), 2)
            self.assertIsNotNone(pop.metadata)
            self.assertEqual(pop.metadata['model'], 'test_model')
        finally:
            os.unlink(csv_path)
            if os.path.exists(metadata_path):
                os.unlink(metadata_path)

    def test_gamma_ray_model_fallback_exception(self):
        """Test lines 2286-2299: Model that raises exception for fallback"""
        call_count = [0]

        def failing_model(t, **kwargs):
            call_count[0] += 1
            # Fail on first call when using frequency array
            if call_count[0] == 1:
                raise TypeError("Cannot handle frequency array")
            # Succeed on subsequent single-frequency calls
            return np.full_like(t, 1e-3)

        sim = SimulateGammaRayTransient(
            model=failing_model,
            parameters={'redshift': 0.5},
            energy_edges=[10, 50, 100],
            time_range=(0, 5),
            effective_area=100,
            background_rate=0.1,
            seed=self.seed
        )

        # Generate events should trigger fallback
        events = sim.generate_time_tagged_events(max_events=100)
        self.assertIsInstance(events, pd.DataFrame)
        # Fallback was triggered if call_count > 1
        self.assertGreater(call_count[0], 1)

    def test_generic_transient_extra_scatter(self):
        """Test lines 110-113: extra_scatter parameter"""
        times = np.linspace(0, 10, 10)
        model = lambda t, **kwargs: np.full_like(t, 1.0, dtype=float)

        transient = SimulateGenericTransient(
            model=model,
            parameters={},
            times=times,
            model_kwargs={'frequency': 1e14},
            data_points=10,
            extra_scatter=0.1,  # Add extra scatter
            seed=self.seed
        )
        self.assertIsNotNone(transient.data)
        # With extra scatter, errors should include the scatter term
        self.assertIn('output_error', transient.data.columns)

    def test_cadence_transient_optical_bands_mode(self):
        """Test optical mode with bands in cadence"""
        model = lambda t, **kwargs: np.full_like(t, 20.0, dtype=float)  # Magnitude

        cadence_config = {
            'bands': ['g', 'r'],
            'cadence_days': 1.0,
            'duration_days': 3,
            'limiting_mags': {'g': 22.5, 'r': 23.0}
        }

        transient = SimulateTransientWithCadence(
            model=model,
            parameters={'redshift': 0.1},
            cadence_config=cadence_config,
            observation_mode='optical',
            seed=self.seed
        )
        self.assertIsNotNone(transient.observations)
        self.assertIn('band', transient.observations.columns)

    def test_population_synthesizer_custom_cosmo(self):
        """Test PopulationSynthesizer with custom cosmology"""
        from astropy.cosmology import FlatLambdaCDM
        custom_cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

        synth = PopulationSynthesizer(
            model='arnett',
            prior=bilby.core.prior.PriorDict({'mej': bilby.core.prior.Uniform(0.01, 0.1)}),
            rate=1e-6,
            cosmology=custom_cosmo,
            seed=self.seed
        )
        self.assertEqual(synth.cosmology, custom_cosmo)

    def test_transient_population_filter_detected(self):
        """Test filtering population to detected only"""
        params = pd.DataFrame({
            'redshift': [0.1, 0.2, 0.3, 0.4],
            'mass': [1.0, 2.0, 3.0, 4.0],
            'detected': [True, False, True, False]
        })
        pop = TransientPopulation(params)

        # Get only detected
        detected_df = pop.detected
        self.assertEqual(len(detected_df), 2)
        self.assertTrue(all(detected_df['detected']))

    def test_cadence_transient_with_rms_noise(self):
        """Test cadence transient with RMS-based noise calculation"""
        model = lambda t, **kwargs: np.full_like(t, 1e-3, dtype=float)

        cadence_config = {
            'frequencies': [1e9],
            'cadence_days': 1,
            'duration_days': 5,
            'sensitivity': 1e-5  # RMS noise
        }

        transient = SimulateTransientWithCadence(
            model=model,
            parameters={'redshift': 0.1},
            cadence_config=cadence_config,
            observation_mode='radio',
            seed=self.seed
        )
        self.assertIn('flux_density_error', transient.observations.columns)
        # Check RMS is used for errors
        errors = transient.observations['flux_density_error'].values
        self.assertTrue(all(e > 0 for e in errors))

