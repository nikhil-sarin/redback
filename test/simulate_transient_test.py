import unittest
from unittest import mock
from unittest.mock import MagicMock, patch
import numpy as np
import requests
from redback.simulate_transients import (SimulateGenericTransient, SimulateOpticalTransient, SimulateFullOpticalSurvey,
    make_pointing_table_from_average_cadence)
import bilby


def _network_available():
    """Check if network access is available."""
    try:
        # Try to access a network resource to check availability
        response = requests.get("https://www.google.com/", timeout=5)
        return response.status_code != 403
    except Exception:
        return False


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


@unittest.skipUnless(_network_available(), "Network access required for sncosmo data")
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

