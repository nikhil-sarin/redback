import unittest
from os import listdir
from os.path import dirname
from unittest import mock
from unittest.mock import patch, MagicMock

import astropy.units as uu
from collections import namedtuple
from scipy.interpolate import interp1d
import bilby
import numpy as np

import redback.model_library

_dirname = dirname(__file__)


class TestModels(unittest.TestCase):

    def setUp(self) -> None:
        self.path_to_files = f"{_dirname}/../redback/priors/"
        self.prior_files = listdir(self.path_to_files)

    def tearDown(self) -> None:
        pass

    def get_prior(self, file):
        prior_dict = bilby.prior.PriorDict()
        prior_dict.from_file(f"{self.path_to_files}{file}")
        return prior_dict

    def test_models(self):
        kwargs = dict(frequency=6e14)
        times = np.array([1, 2, 3])
        for f in self.prior_files:
            print(f)
            model_name = f.replace(".prior", "")
            if model_name == 'trapped_magnetar':
                kwargs['output_format'] = 'luminosity'
            else:
                kwargs['output_format'] = 'flux_density'
            prior = self.get_prior(file=f)
            sample = prior.sample()
            if model_name == 'polytrope_eos_two_component_bns':
                sample['gamma_1'] = 4.04
                sample['gamma_2'] = 2.159
                sample['gamma_3'] = 3.688
                sample['log_p'] = 33.72
            function = redback.model_library.all_models_dict[model_name]
            ys = function(times, **sample, **kwargs)
            self.assertEqual(len(times), len(ys))

class TestPhaseModels(unittest.TestCase):
    def setUp(self) -> None:
        self.path_to_files = f"{_dirname}/../redback/priors/"
        self.prior_files = listdir(self.path_to_files)

    def tearDown(self) -> None:
        pass

    def get_prior(self, file):
        prior_dict = bilby.prior.PriorDict()
        prior_dict.from_file(f"{self.path_to_files}{file}")
        return prior_dict

    def test_models(self):
        kwargs = dict(frequency=2e14)
        times = np.array([1, 2, 3]) + 55855
        for f in self.prior_files:
            print(f)
            model_name = f.replace(".prior", "")
            skip_dict = ['bazin_sne', 'villar_sne', 'blackbody_spectrum_with_absorption_and_emission_lines',
                         'powerlaw_spectrum_with_absorption_and_emission_lines', 'exp_rise_powerlaw_decline', 'salt2']
            if model_name in skip_dict:
                print('Skipping {}'.format(model_name))
                pass
            else:
                if model_name == 'trapped_magnetar':
                    kwargs['output_format'] = 'luminosity'
                else:
                    kwargs['output_format'] = 'flux_density'
                prior = self.get_prior(file=f)
                function = redback.model_library.all_models_dict['t0_base_model']
                kwargs['base_model'] = model_name
                kwargs['t0'] = 55855
                sample = prior.sample()
                if model_name == 'polytrope_eos_two_component_bns':
                    sample['gamma_1'] = 4.04
                    sample['gamma_2'] = 2.159
                    sample['gamma_3'] = 3.688
                    sample['log_p'] = 33.72
                ys = function(times, **sample, **kwargs)
                self.assertEqual(len(times), len(ys))

class TestMagnitudeOutput(unittest.TestCase):
    def setUp(self) -> None:
        self.path_to_files = f"{_dirname}/../redback/priors/"
        self.prior_files = listdir(self.path_to_files)

    def tearDown(self) -> None:
        pass

    def get_prior(self, file):
        prior_dict = bilby.prior.PriorDict()
        prior_dict.from_file(f"{self.path_to_files}{file}")
        return prior_dict

    def test_models(self):
        kwargs = dict(frequency=2e14, bands='ztfg')
        times = np.array([1, 2, 3])
        for f in self.prior_files:
            print(f)
            model_name = f.replace(".prior", "")
            if model_name == 'trapped_magnetar':
                kwargs['output_format'] = 'luminosity'
            elif model_name in ['tophat_and_twocomponent', 'tophat_and_twolayerstratified',
                                'tophat_and_arnett', 'tophatredback_and_twolayerstratified']:
                pass
            else:
                kwargs['output_format'] = 'magnitude'
            prior = self.get_prior(file=f)
            sample = prior.sample()
            if model_name == 'polytrope_eos_two_component_bns':
                sample['gamma_1'] = 4.04
                sample['gamma_2'] = 2.159
                sample['gamma_3'] = 3.688
                sample['log_p'] = 33.72
            function = redback.model_library.all_models_dict[model_name]
            ys = function(times, **sample, **kwargs)
            self.assertEqual(len(times), len(ys))

class TestFluxOutput(unittest.TestCase):
    def setUp(self) -> None:
        self.path_to_files = f"{_dirname}/../redback/priors/"
        self.prior_files = listdir(self.path_to_files)

    def tearDown(self) -> None:
        pass

    def get_prior(self, file):
        prior_dict = bilby.prior.PriorDict()
        prior_dict.from_file(f"{self.path_to_files}{file}")
        return prior_dict

    def test_models(self):
        kwargs = dict(frequency=2e14, bands='ztfg')
        times = np.array([1, 2, 3])
        valid_models = ['arnett', 'one_component_kilonova_model', 'slsn',
                        'tde_analytical', 'basic_mergernova', 'csm_nickel']
        for f in self.prior_files:
            model_name = f.replace(".prior", "")
            if model_name in valid_models:
                print(f)
                kwargs['output_format'] = 'flux'
                prior = self.get_prior(file=f)
                sample = prior.sample()
                if model_name == 'polytrope_eos_two_component_bns':
                    sample['gamma_1'] = 4.04
                    sample['gamma_2'] = 2.159
                    sample['gamma_3'] = 3.688
                    sample['log_p'] = 33.72
                function = redback.model_library.all_models_dict[model_name]
                ys = function(times, **sample, **kwargs)
            else:
                ys = np.ones(len(times))
            self.assertEqual(len(times), len(ys))

class TestIntegratedFluxModelFlux(unittest.TestCase):
    def setUp(self) -> None:
        self.path_to_files = f"{_dirname}/../redback/priors/"
        self.prior_files = listdir(self.path_to_files)

    def tearDown(self) -> None:
        pass

    def get_prior(self, file):
        prior_dict = bilby.prior.PriorDict()
        prior_dict.from_file(f"{self.path_to_files}{file}")
        return prior_dict

    def test_models(self):
        kwargs = dict(frequency=np.array([2e14,6e14]), bands='ztfg')
        times = np.array([1, 2, 3])
        valid_models = ['gaussiancore']
        for f in self.prior_files:
            model_name = f.replace(".prior", "")
            if model_name in valid_models:
                print(f)
                prior = self.get_prior(file=f)
                kwargs['base_model'] = model_name
                sample = prior.sample()
                function = redback.model_library.all_models_dict['integrated_flux_afterglowpy_base_model']
                ys = function(times, **sample, **kwargs)
            else:
                ys = np.ones(len(times))
            self.assertEqual(len(times), len(ys))

class TestIntegratedFluxModelRate(unittest.TestCase):
    def setUp(self) -> None:
        self.path_to_files = f"{_dirname}/../redback/priors/"
        self.prior_files = listdir(self.path_to_files)

    def tearDown(self) -> None:
        pass

    def get_prior(self, file):
        prior_dict = bilby.prior.PriorDict()
        prior_dict.from_file(f"{self.path_to_files}{file}")
        return prior_dict

    def test_models(self):
        kwargs = dict(frequency=np.array([2e14,6e14]), bands='ztfg')
        times = np.array([1, 2, 3])
        valid_models = ['smoothpowerlaw']
        for f in self.prior_files:
            model_name = f.replace(".prior", "")
            if model_name in valid_models:
                print(f)
                prior = self.get_prior(file=f)
                kwargs['base_model'] = model_name
                function = redback.model_library.all_models_dict['integrated_flux_rate_model']
                ys = function(times, **prior.sample(), **kwargs)
            else:
                ys = np.ones(len(times))
            self.assertEqual(len(times), len(ys))

class TestExtinctionModelsFluxDensity(unittest.TestCase):
    def setUp(self) -> None:
        self.path_to_files = f"{_dirname}/../redback/priors/"
        self.prior_files = listdir(self.path_to_files)

    def tearDown(self) -> None:
        pass

    def get_prior(self, file):
        prior_dict = bilby.prior.PriorDict()
        prior_dict.from_file(f"{self.path_to_files}{file}")
        return prior_dict

    def test_models(self):
        kwargs = dict(frequency=2e14, bands='ztfg')
        times = np.array([1, 2, 3]) + 58555
        valid_models = ['arnett', 'slsn','one_component_kilonova_model'
                        'tde_analytical', 'basic_mergernova', 'gaussiancore',
                        'smoothpowerlaw', 'shock_cooling']
        supernova_models = ['arnett', 'slsn']
        kilonova_models = ['one_component_kilonova_model']
        tde_models = ['tde_analytical']
        magnetardriven_models = ['basic_mergernova']
        for f in self.prior_files:
            model_name = f.replace(".prior", "")
            if model_name in valid_models:
                print(f)
                kwargs['output_format'] = 'flux_density'
                kwargs['base_model'] = model_name
                prior = self.get_prior(file=f)
                prior['av_host'] = 0.5
                prior['t0'] = 58555
                if model_name in supernova_models:
                    function = 't0_supernova_extinction'
                elif model_name in kilonova_models:
                    function = 't0_kilonova_extinction'
                elif model_name in tde_models:
                    function = 't0_tde_extinction'
                elif model_name in magnetardriven_models:
                    function = 't0_magnetar_driven_extinction'
                if model_name == 'smoothpowerlaw':
                    function = 't0_afterglow_extinction'
                if model_name == 'gaussiancore':
                    function = 't0_afterglow_extinction_model_d2g'
                    kwargs['lognh'] = 22
                    kwargs['factor'] = 5
                    prior.pop('av_host')
                if model_name == 'shock_cooling':
                    function = 't0_shock_powered_extinction'
                function = redback.model_library.all_models_dict[function]
                ys = function(times, **prior.sample(), **kwargs)
            else:
                ys = np.ones(len(times))
            self.assertEqual(len(times), len(ys))

class TestExtinctionModelsMagnitude(unittest.TestCase):
    def setUp(self) -> None:
        self.path_to_files = f"{_dirname}/../redback/priors/"
        self.prior_files = listdir(self.path_to_files)

    def tearDown(self) -> None:
        pass

    def get_prior(self, file):
        prior_dict = bilby.prior.PriorDict()
        prior_dict.from_file(f"{self.path_to_files}{file}")
        return prior_dict

    def test_models(self):
        kwargs = dict(frequency=2e14, bands='ztfg')
        times = np.array([1, 2, 3]) + 58555
        valid_models = ['arnett', 'slsn','one_component_kilonova_model'
                        'tde_analytical', 'basic_mergernova', 'shock_cooling']
        supernova_models = ['arnett', 'slsn']
        kilonova_models = ['one_component_kilonova_model']
        tde_models = ['tde_analytical']
        magnetardriven_models = ['basic_mergernova']
        for f in self.prior_files:
            model_name = f.replace(".prior", "")
            if model_name in valid_models:
                print(f)
                kwargs['output_format'] = 'magnitude'
                kwargs['base_model'] = model_name
                prior = self.get_prior(file=f)
                prior['av_host'] = 0.5
                prior['t0'] = 58555
                if model_name in supernova_models:
                    function = 't0_supernova_extinction'
                elif model_name in kilonova_models:
                    function = 't0_kilonova_extinction'
                elif model_name in tde_models:
                    function = 't0_tde_extinction'
                elif model_name in magnetardriven_models:
                    function = 't0_magnetar_driven_extinction'
                if model_name == 'shock_cooling':
                    function = 't0_shock_powered_extinction'
                function = redback.model_library.all_models_dict[function]
                ys = function(times, **prior.sample(), **kwargs)
            else:
                ys = np.ones(len(times))
            self.assertEqual(len(times), len(ys))

class TestHomologousExpansion(unittest.TestCase):
    def setUp(self):
        self.time = np.array([1, 2, 3])
        self.base_model = 'arnett_bolometric'

    def tearDown(self) -> None:
        pass

    def get_prior(self):
        prior_dict = redback.priors.get_priors(self.base_model)
        return prior_dict

    def test_base_model(self):
        with mock.patch('redback.transient_models.supernova_models.homologous_expansion_supernova') as m:
            kwargs = dict(frequency=2e14, bands='ztfg')
            prior = self.get_prior()
            kwargs['output_format'] = 'flux_density'
            prior.pop('vej')
            prior['ek'] =  bilby.core.prior.LogUniform(1e50, 1e51, 'ek')
            prior['redshift'] = 0.075
            prior['temperature_floor'] = bilby.core.prior.LogUniform(1e3,1e5,name = 'temperature_floor')
            function = redback.transient_models.supernova_models.homologous_expansion_supernova
            m.base_model = self.base_model
            sample = prior.sample()
            actual = function(self.time, base_model=self.base_model, **sample, **kwargs)
            m.assert_called_with(self.time, base_model=self.base_model, **sample, **kwargs)

    def test_fluxdensity_output(self):
        kwargs = dict(frequency=2e14, bands='ztfg')
        prior = self.get_prior()
        kwargs['base_model'] = self.base_model
        kwargs['output_format'] = 'flux_density'
        function = redback.model_library.all_models_dict['homologous_expansion_supernova']
        prior.pop('vej')
        prior['ek'] =  bilby.core.prior.LogUniform(1e50, 1e51, 'ek')
        prior['redshift'] = 0.075
        prior['temperature_floor'] = bilby.core.prior.LogUniform(1e3,1e5,name = 'temperature_floor')
        ys = function(self.time, **prior.sample(), **kwargs)
        self.assertEqual(len(self.time), len(ys))

    def test_magnitude_output(self):
        kwargs = dict(frequency=2e14, bands='ztfg')
        prior = self.get_prior()
        kwargs['base_model'] = self.base_model
        kwargs['output_format'] = 'magnitude'
        function = redback.model_library.all_models_dict['homologous_expansion_supernova']
        prior.pop('vej')
        prior['ek'] =  bilby.core.prior.LogUniform(1e50, 1e51, 'ek')
        prior['redshift'] = 0.075
        prior['temperature_floor'] = bilby.core.prior.LogUniform(1e3,1e5,name = 'temperature_floor')
        ys = function(self.time, **prior.sample(), **kwargs)
        self.assertEqual(len(self.time), len(ys))

class TestThinShellExpansion(unittest.TestCase):
    def setUp(self):
        self.time = np.array([1, 2, 3])
        self.base_model = 'arnett_bolometric'

    def tearDown(self) -> None:
        pass

    def get_prior(self):
        prior_dict = redback.priors.get_priors(self.base_model)
        return prior_dict

    def test_base_model(self):
        with mock.patch('redback.transient_models.supernova_models.thin_shell_supernova') as m:
            kwargs = dict(frequency=2e14, bands='ztfg')
            prior = self.get_prior()
            kwargs['output_format'] = 'flux_density'
            prior.pop('vej')
            prior['ek'] =  bilby.core.prior.LogUniform(1e50, 1e51, 'ek')
            prior['redshift'] = 0.075
            prior['temperature_floor'] = bilby.core.prior.LogUniform(1e3,1e5,name = 'temperature_floor')
            function = redback.transient_models.supernova_models.thin_shell_supernova
            m.base_model = self.base_model
            sample = prior.sample()
            actual = function(self.time, base_model=self.base_model, **sample, **kwargs)
            m.assert_called_with(self.time, base_model=self.base_model, **sample, **kwargs)

    def test_fluxdensity_output(self):
        kwargs = dict(frequency=2e14, bands='ztfg')
        prior = self.get_prior()
        kwargs['base_model'] = self.base_model
        kwargs['output_format'] = 'flux_density'
        function = redback.model_library.all_models_dict['thin_shell_supernova']
        prior.pop('vej')
        prior['ek'] =  bilby.core.prior.LogUniform(1e50, 1e51, 'ek')
        prior['redshift'] = 0.075
        prior['temperature_floor'] = bilby.core.prior.LogUniform(1e3,1e5,name = 'temperature_floor')
        ys = function(self.time, **prior.sample(), **kwargs)
        self.assertEqual(len(self.time), len(ys))

    def test_magnitude_output(self):
        kwargs = dict(frequency=2e14, bands='ztfg')
        prior = self.get_prior()
        kwargs['base_model'] = self.base_model
        kwargs['output_format'] = 'magnitude'
        function = redback.model_library.all_models_dict['thin_shell_supernova']
        prior.pop('vej')
        prior['ek'] =  bilby.core.prior.LogUniform(1e50, 1e51, 'ek')
        prior['redshift'] = 0.075
        prior['temperature_floor'] = bilby.core.prior.LogUniform(1e3,1e5,name = 'temperature_floor')
        ys = function(self.time, **prior.sample(), **kwargs)
        self.assertEqual(len(self.time), len(ys))

class TestTypeIIWrapperFunctions(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures."""
        # Import the functions under test - note the correct function name
        from redback.transient_models.supernova_models import typeII_bolometric, typeII_photosphere_properties, typeII_surrogate_sarin25
        self.typeII_bolometric = typeII_bolometric
        self.typeII_photosphere_properties = typeII_photosphere_properties  # This is the actual function name
        self.typeII = typeII_surrogate_sarin25

    def test_typeII_bolometric_single_time(self):
        """Test typeII_bolometric with single time value."""
        # Setup mock return values
        mock_times = np.array([0.1, 1.0, 10.0, 100.0])
        mock_lbol = np.array([1e42, 1e43, 1e42, 1e41])

        # Test parameters
        time = 5.0
        progenitor = 15.0
        ni_mass = 0.05
        log10_mdot = -3.0
        beta = 1.0
        rcsm = 5.0
        esn = 1.0

        # Create a mock module with the function
        mock_typeII_lbol = MagicMock(return_value=(mock_times, mock_lbol))

        # Patch the import by replacing the module in sys.modules temporarily
        mock_module = MagicMock()
        mock_module.typeII_lbol = mock_typeII_lbol

        with patch.dict('sys.modules', {'redback_surrogates.supernovamodels': mock_module}):
            # Call function
            result = self.typeII_bolometric(
                time=time, progenitor=progenitor, ni_mass=ni_mass,
                log10_mdot=log10_mdot, beta=beta, rcsm=rcsm, esn=esn
            )

        # Verify mock was called with correct parameters
        mock_typeII_lbol.assert_called_once_with(
            time=time, progenitor=progenitor, ni_mass=ni_mass,
            log10_mdot=log10_mdot, beta=beta, rcsm=rcsm, esn=esn
        )

        # Verify result is a scalar (could be float, np.floating, or 0-d array)
        self.assertTrue(np.isscalar(result) or (isinstance(result, np.ndarray) and result.ndim == 0))

        # Convert to scalar if it's a 0-d array
        scalar_result = float(result) if hasattr(result, 'item') else result

        # Verify result is reasonable (interpolated between mock values)
        self.assertGreater(scalar_result, 0)

    def test_typeII_bolometric_array_time(self):
        """Test typeII_bolometric with array of time values."""
        # Setup mock return values
        mock_times = np.array([0.1, 1.0, 10.0, 100.0])
        mock_lbol = np.array([1e42, 1e43, 1e42, 1e41])

        # Test parameters
        time = np.array([0.5, 5.0, 50.0])
        progenitor = 15.0
        ni_mass = 0.05
        log10_mdot = -3.0
        beta = 1.0
        rcsm = 5.0
        esn = 1.0

        # Create a mock module with the function
        mock_typeII_lbol = MagicMock(return_value=(mock_times, mock_lbol))
        mock_module = MagicMock()
        mock_module.typeII_lbol = mock_typeII_lbol

        with patch.dict('sys.modules', {'redback_surrogates.supernovamodels': mock_module}):
            # Call function
            result = self.typeII_bolometric(
                time=time, progenitor=progenitor, ni_mass=ni_mass,
                log10_mdot=log10_mdot, beta=beta, rcsm=rcsm, esn=esn
            )

        # Verify result is an array with same length as input time
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), len(time))

        # Verify all results are positive
        self.assertTrue(np.all(result > 0))

    def test_typeII_photosphere_properties_single_time(self):  # Updated test name
        """Test typeII_photosphere_properties with single time value."""
        # Setup mock return values
        mock_times = np.array([0.1, 1.0, 10.0, 100.0])
        mock_temp = np.array([10000, 8000, 6000, 4000])
        mock_rad = np.array([1e14, 2e14, 3e14, 4e14])

        # Test parameters
        time = 5.0
        progenitor = 15.0
        ni_mass = 0.05
        log10_mdot = -3.0
        beta = 1.0
        rcsm = 5.0
        esn = 1.0

        # Create a proper mock that returns the expected tuple
        def mock_photosphere_func(*args, **kwargs):
            return (mock_times, mock_temp, mock_rad)

        mock_module = MagicMock()
        # The redback function imports typeII_photosphere from surrogates
        mock_module.typeII_photosphere = mock_photosphere_func

        with patch.dict('sys.modules', {'redback_surrogates.supernovamodels': mock_module}):
            # Call function - note the correct function name
            temp, rad = self.typeII_photosphere_properties(
                time=time, progenitor=progenitor, ni_mass=ni_mass,
                log10_mdot=log10_mdot, beta=beta, rcsm=rcsm, esn=esn
            )

        # Verify results are scalars (could be float, np.floating, or 0-d array)
        self.assertTrue(np.isscalar(temp) or (isinstance(temp, np.ndarray) and temp.ndim == 0))
        self.assertTrue(np.isscalar(rad) or (isinstance(rad, np.ndarray) and rad.ndim == 0))

        # Convert to scalars if they're 0-d arrays
        scalar_temp = float(temp) if hasattr(temp, 'item') else temp
        scalar_rad = float(rad) if hasattr(rad, 'item') else rad

        # Verify results are reasonable
        self.assertGreater(scalar_temp, 0)
        self.assertGreater(scalar_rad, 0)

    def test_typeII_photosphere_properties_array_time(self):  # Updated test name
        """Test typeII_photosphere_properties with array of time values."""
        # Setup mock return values
        mock_times = np.array([0.1, 1.0, 10.0, 100.0])
        mock_temp = np.array([10000, 8000, 6000, 4000])
        mock_rad = np.array([1e14, 2e14, 3e14, 4e14])

        # Test parameters
        time = np.array([0.5, 5.0, 50.0])
        progenitor = 15.0
        ni_mass = 0.05
        log10_mdot = -3.0
        beta = 1.0
        rcsm = 5.0
        esn = 1.0

        # Create a proper mock that returns the expected tuple
        def mock_photosphere_func(*args, **kwargs):
            return (mock_times, mock_temp, mock_rad)

        mock_module = MagicMock()
        mock_module.typeII_photosphere = mock_photosphere_func

        with patch.dict('sys.modules', {'redback_surrogates.supernovamodels': mock_module}):
            # Call function
            temp, rad = self.typeII_photosphere_properties(
                time=time, progenitor=progenitor, ni_mass=ni_mass,
                log10_mdot=log10_mdot, beta=beta, rcsm=rcsm, esn=esn
            )

        # Verify results are arrays with same length as input time
        self.assertIsInstance(temp, np.ndarray)
        self.assertIsInstance(rad, np.ndarray)
        self.assertEqual(len(temp), len(time))
        self.assertEqual(len(rad), len(time))

        # Verify all results are positive
        self.assertTrue(np.all(temp > 0))
        self.assertTrue(np.all(rad > 0))

    def test_typeII_interpolation_behavior(self):
        """Test that interpolation functions behave correctly with edge cases."""
        # Test extrapolation behavior
        times = np.array([1.0, 2.0, 3.0])
        values = np.array([10.0, 20.0, 30.0])

        # Create interpolation function like in the actual functions
        interp_func = interp1d(times, values, bounds_error=False, fill_value='extrapolate')

        # Test interpolation within bounds
        result_interp = interp_func(2.5)
        self.assertAlmostEqual(float(result_interp), 25.0)

        # Test extrapolation beyond bounds
        result_extrap_low = interp_func(0.5)
        result_extrap_high = interp_func(4.0)
        self.assertLess(float(result_extrap_low), 10.0)  # Should extrapolate to lower value
        self.assertGreater(float(result_extrap_high), 30.0)  # Should extrapolate to higher value

    def test_typeII_bolometric_parameter_passing(self):
        """Test that all parameters are correctly passed to typeII_lbol."""
        # Setup mock
        mock_times = np.array([1.0, 2.0])
        mock_lbols = np.array([1e42, 2e42])

        # Test with various parameter combinations
        test_cases = [
            {
                'time': 1.0,
                'progenitor': 15.0,
                'ni_mass': 0.05,
                'log10_mdot': -3.0,
                'beta': 1.0,
                'rcsm': 5.0,
                'esn': 1.0
            },
            {
                'time': np.array([1.0, 2.0]),
                'progenitor': 20.0,
                'ni_mass': 0.1,
                'log10_mdot': -2.5,
                'beta': 1.5,
                'rcsm': 10.0,
                'esn': 2.0
            }
        ]

        for params in test_cases:
            with self.subTest(params=params):
                # Create a mock module with the function
                mock_typeII_lbol = MagicMock(return_value=(mock_times, mock_lbols))
                mock_module = MagicMock()
                mock_module.typeII_lbol = mock_typeII_lbol

                with patch.dict('sys.modules', {'redback_surrogates.supernovamodels': mock_module}):
                    result = self.typeII_bolometric(**params)

                    # Verify the mock was called with exactly the same parameters
                    mock_typeII_lbol.assert_called_once_with(**params)

    def test_typeII_photosphere_properties_parameter_passing(self):
        """Test that all parameters are correctly passed to typeII_photosphere."""
        # Setup mock
        mock_times = np.array([1.0, 2.0])
        mock_temp = np.array([5000, 4000])
        mock_rad = np.array([1e14, 2e14])

        # Test parameters
        params = {
            'time': 1.0,
            'progenitor': 15.0,
            'ni_mass': 0.05,
            'log10_mdot': -3.0,
            'beta': 1.0,
            'rcsm': 5.0,
            'esn': 1.0
        }

        # Create a mock function that captures the arguments
        call_log = []

        def mock_photosphere_func(*args, **kwargs):
            call_log.append((args, kwargs))
            return (mock_times, mock_temp, mock_rad)

        mock_module = MagicMock()
        mock_module.typeII_photosphere = mock_photosphere_func

        with patch.dict('sys.modules', {'redback_surrogates.supernovamodels': mock_module}):
            result = self.typeII_photosphere_properties(**params)

            # Verify the function was called
            self.assertEqual(len(call_log), 1)
            args, kwargs = call_log[0]

            # Check that the correct parameters were passed (excluding 'time' which is used for interpolation)
            expected_params = {k: v for k, v in params.items() if k != 'time'}
            for key, value in expected_params.items():
                self.assertIn(key, kwargs)
                self.assertEqual(kwargs[key], value)

    def test_typeII_photosphere_properties_error_handling(self):
        """Test error handling in typeII_photosphere_properties."""
        time = 1.0
        params = {
            'progenitor': 15.0, 'ni_mass': 0.05, 'log10_mdot': -3.0,
            'beta': 1.0, 'rcsm': 5.0, 'esn': 1.0
        }

        # Create a mock function that returns empty arrays
        def mock_photosphere_func_empty(*args, **kwargs):
            return (np.array([]), np.array([]), np.array([]))

        mock_module = MagicMock()
        mock_module.typeII_photosphere = mock_photosphere_func_empty

        with patch.dict('sys.modules', {'redback_surrogates.supernovamodels': mock_module}):
            # This should handle empty arrays gracefully or raise appropriate error
            with self.assertRaises((ValueError, IndexError)):
                self.typeII_photosphere_properties(time=time, **params)

    def test_typeII_bolometric_kwargs_handling(self):
        """Test that kwargs are properly passed through."""
        mock_times = np.array([1.0, 2.0])
        mock_lbols = np.array([1e42, 2e42])

        # Create a mock module with the function
        mock_typeII_lbol = MagicMock(return_value=(mock_times, mock_lbols))
        mock_module = MagicMock()
        mock_module.typeII_lbol = mock_typeII_lbol

        with patch.dict('sys.modules', {'redback_surrogates.supernovamodels': mock_module}):
            # Call with extra kwargs
            result = self.typeII_bolometric(
                time=1.0, progenitor=15.0, ni_mass=0.05, log10_mdot=-3.0,
                beta=1.0, rcsm=5.0, esn=1.0, extra_param='test'
            )

            # Verify kwargs were passed through
            mock_typeII_lbol.assert_called_once()
            call_kwargs = mock_typeII_lbol.call_args[1]
            self.assertIn('extra_param', call_kwargs)
            self.assertEqual(call_kwargs['extra_param'], 'test')

    def test_typeII_photosphere_properties_kwargs_handling(self):
        """Test that kwargs are properly passed through in photosphere function."""
        mock_times = np.array([1.0, 2.0])
        mock_temp = np.array([5000, 4000])
        mock_rad = np.array([1e14, 2e14])

        # Create a mock function that captures the arguments
        call_log = []

        def mock_photosphere_func(*args, **kwargs):
            call_log.append((args, kwargs))
            return (mock_times, mock_temp, mock_rad)

        mock_module = MagicMock()
        mock_module.typeII_photosphere = mock_photosphere_func

        with patch.dict('sys.modules', {'redback_surrogates.supernovamodels': mock_module}):
            # Call with extra kwargs
            result = self.typeII_photosphere_properties(
                time=1.0, progenitor=15.0, ni_mass=0.05, log10_mdot=-3.0,
                beta=1.0, rcsm=5.0, esn=1.0, extra_param='test'
            )

            # Verify the function was called and kwargs were passed
            self.assertEqual(len(call_log), 1)
            args, kwargs = call_log[0]
            self.assertIn('extra_param', kwargs)
            self.assertEqual(kwargs['extra_param'], 'test')

class TestTypeIIEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def setUp(self):
        from redback.transient_models.supernova_models import typeII_bolometric, typeII_photosphere_properties, typeII_surrogate_sarin25
        self.typeII_bolometric = typeII_bolometric
        self.typeII_photosphere_properties = typeII_photosphere_properties
        self.typeII = typeII_surrogate_sarin25

    def test_typeII_bolometric_zero_time(self):
        """Test behavior with zero time input."""
        mock_times = np.array([0.1, 1.0, 10.0])
        mock_lbols = np.array([1e42, 1e43, 1e41])

        # Create a mock module with the function
        mock_typeII_lbol = MagicMock(return_value=(mock_times, mock_lbols))
        mock_module = MagicMock()
        mock_module.typeII_lbol = mock_typeII_lbol

        with patch.dict('sys.modules', {'redback_surrogates.supernovamodels': mock_module}):
            result = self.typeII_bolometric(
                time=0.0, progenitor=15.0, ni_mass=0.05, log10_mdot=-3.0,
                beta=1.0, rcsm=5.0, esn=1.0
            )

        # Should return a finite value (extrapolated)
        scalar_result = float(result) if hasattr(result, 'item') else result
        self.assertTrue(np.isfinite(scalar_result))

    def test_typeII_photosphere_properties_zero_time(self):
        """Test photosphere behavior with zero time input."""
        mock_times = np.array([0.1, 1.0, 10.0])
        mock_temp = np.array([10000, 8000, 6000])
        mock_rad = np.array([1e14, 2e14, 3e14])

        # Create a mock function
        def mock_photosphere_func(*args, **kwargs):
            return (mock_times, mock_temp, mock_rad)

        mock_module = MagicMock()
        mock_module.typeII_photosphere = mock_photosphere_func

        with patch.dict('sys.modules', {'redback_surrogates.supernovamodels': mock_module}):
            temp, rad = self.typeII_photosphere_properties(
                time=0.0, progenitor=15.0, ni_mass=0.05, log10_mdot=-3.0,
                beta=1.0, rcsm=5.0, esn=1.0
            )

        # Should return finite values (extrapolated)
        scalar_temp = float(temp) if hasattr(temp, 'item') else temp
        scalar_rad = float(rad) if hasattr(rad, 'item') else rad
        self.assertTrue(np.isfinite(scalar_temp))
        self.assertTrue(np.isfinite(scalar_rad))


class TestJetsimpyModels(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures."""
        # Import the functions under test
        from redback.transient_models.afterglow_models import jetsimpy_tophat, jetsimpy_gaussian, jetsimpy_powerlaw
        self.jetsimpy_tophat = jetsimpy_tophat
        self.jetsimpy_gaussian = jetsimpy_gaussian
        self.jetsimpy_powerlaw = jetsimpy_powerlaw

        # Common test parameters
        self.time = np.array([1.0, 2.0, 3.0])
        self.redshift = 0.1
        self.thv = 0.2
        self.loge0 = 52.0
        self.thc = 0.1
        self.nism = 1.0
        self.A = 1.0
        self.p = 2.2
        self.logepse = -1.0
        self.logepsb = -2.0
        self.g0 = 100.0
        self.s = 2.0  # For powerlaw model

        # Mock cosmology
        self.mock_cosmology = MagicMock()
        self.mock_dl = MagicMock()
        self.mock_dl.cgs.value = 1e28
        self.mock_cosmology.luminosity_distance.return_value = self.mock_dl

    def test_jetsimpy_tophat_flux_density(self):
        """Test jetsimpy_tophat with flux_density output."""
        # Create mock jetsimpy module
        mock_jetsimpy = MagicMock()
        mock_flux_density = np.array([1e-26, 2e-26, 3e-26])
        mock_jetsimpy.FluxDensity_tophat.return_value = mock_flux_density

        # Test parameters
        frequency = 5e14
        kwargs = {
            'output_format': 'flux_density',
            'frequency': frequency
        }

        # Patch the import and cosmology
        with patch.dict('sys.modules', {'jetsimpy': mock_jetsimpy}), \
                patch('redback.transient_models.afterglow_models.cosmo', self.mock_cosmology):
            # Call function
            result = self.jetsimpy_tophat(
                time=self.time, redshift=self.redshift, thv=self.thv, loge0=self.loge0,
                thc=self.thc, nism=self.nism, A=self.A, p=self.p,
                logepse=self.logepse, logepsb=self.logepsb, g0=self.g0, **kwargs
            )

        # Verify jetsimpy was called with correct parameters
        mock_jetsimpy.FluxDensity_tophat.assert_called_once()
        call_args = mock_jetsimpy.FluxDensity_tophat.call_args

        # Check time conversion (days to seconds)
        expected_time_s = self.time * 86400  # day_to_s
        np.testing.assert_array_almost_equal(call_args[0][0], expected_time_s)

        # Check frequency
        self.assertEqual(call_args[0][1], frequency)

        # Check parameter dictionary structure
        param_dict = call_args[0][2]
        self.assertIn('Eiso', param_dict)
        self.assertEqual(param_dict['Eiso'], 10 ** self.loge0)
        self.assertEqual(param_dict['lf'], self.g0)
        self.assertEqual(param_dict['theta_c'], self.thc)
        self.assertEqual(param_dict['theta_v'], self.thv)

        # Verify result
        np.testing.assert_array_equal(result, mock_flux_density)

    def test_jetsimpy_tophat_magnitude(self):
        """Test jetsimpy_tophat with magnitude output."""
        # Create mock jetsimpy module
        mock_jetsimpy = MagicMock()
        mock_flux_density = np.array([1e-26, 2e-26, 3e-26])
        mock_jetsimpy.FluxDensity_tophat.return_value = mock_flux_density

        mock_frequency = 5e14
        mock_mag_result = MagicMock()
        mock_mag_result.value = np.array([20.0, 21.0, 22.0])

        # Test parameters
        kwargs = {
            'output_format': 'magnitude',
            'bands': ['g']
        }

        # Patch everything needed
        with patch.dict('sys.modules', {'jetsimpy': mock_jetsimpy}), \
                patch('redback.transient_models.afterglow_models.cosmo', self.mock_cosmology), \
                patch('redback.transient_models.afterglow_models.bands_to_frequency', return_value=mock_frequency), \
                patch('redback.transient_models.afterglow_models.calc_ABmag_from_flux_density',
                      return_value=mock_mag_result):
            # Call function
            result = self.jetsimpy_tophat(
                time=self.time, redshift=self.redshift, thv=self.thv, loge0=self.loge0,
                thc=self.thc, nism=self.nism, A=self.A, p=self.p,
                logepse=self.logepse, logepsb=self.logepsb, g0=self.g0, **kwargs
            )

        # Verify jetsimpy was called
        mock_jetsimpy.FluxDensity_tophat.assert_called_once()

        # Verify result
        np.testing.assert_array_equal(result, mock_mag_result.value)

    def test_jetsimpy_gaussian_flux_density(self):
        """Test jetsimpy_gaussian with flux_density output."""
        # Create mock jetsimpy module
        mock_jetsimpy = MagicMock()
        mock_flux_density = np.array([1e-26, 2e-26, 3e-26])
        mock_jetsimpy.FluxDensity_gaussian.return_value = mock_flux_density

        # Test parameters
        frequency = 5e14
        kwargs = {
            'output_format': 'flux_density',
            'frequency': frequency
        }

        # Patch the import and cosmology
        with patch.dict('sys.modules', {'jetsimpy': mock_jetsimpy}), \
                patch('redback.transient_models.afterglow_models.cosmo', self.mock_cosmology):
            # Call function
            result = self.jetsimpy_gaussian(
                time=self.time, redshift=self.redshift, thv=self.thv, loge0=self.loge0,
                thc=self.thc, nism=self.nism, A=self.A, p=self.p,
                logepse=self.logepse, logepsb=self.logepsb, g0=self.g0, **kwargs
            )

        # Verify jetsimpy was called
        mock_jetsimpy.FluxDensity_gaussian.assert_called_once()

        # Verify result
        np.testing.assert_array_equal(result, mock_flux_density)

    def test_jetsimpy_powerlaw_flux_density(self):
        """Test jetsimpy_powerlaw with flux_density output."""
        # Create mock jetsimpy module
        mock_jetsimpy = MagicMock()
        mock_flux_density = np.array([1e-26, 2e-26, 3e-26])
        mock_jetsimpy.FluxDensity_powerlaw.return_value = mock_flux_density

        # Test parameters
        frequency = 5e14
        kwargs = {
            'output_format': 'flux_density',
            'frequency': frequency
        }

        # Patch the import and cosmology
        with patch.dict('sys.modules', {'jetsimpy': mock_jetsimpy}), \
                patch('redback.transient_models.afterglow_models.cosmo', self.mock_cosmology):
            # Call function
            result = self.jetsimpy_powerlaw(
                time=self.time, redshift=self.redshift, thv=self.thv, loge0=self.loge0,
                thc=self.thc, nism=self.nism, A=self.A, p=self.p,
                logepse=self.logepse, logepsb=self.logepsb, g0=self.g0, s=self.s, **kwargs
            )

        # Verify jetsimpy was called
        mock_jetsimpy.FluxDensity_powerlaw.assert_called_once()

        # Check that 's' parameter is included in the parameter dictionary
        call_args = mock_jetsimpy.FluxDensity_powerlaw.call_args
        param_dict = call_args[0][2]
        self.assertIn('s', param_dict)
        self.assertEqual(param_dict['s'], self.s)

        # Verify result
        np.testing.assert_array_equal(result, mock_flux_density)

    def test_parameter_dictionary_construction(self):
        """Test that the parameter dictionary is constructed correctly."""
        # Create mock jetsimpy module
        mock_jetsimpy = MagicMock()
        mock_jetsimpy.FluxDensity_tophat.return_value = np.array([1e-26])

        kwargs = {
            'output_format': 'flux_density',
            'frequency': 5e14
        }

        # Patch the import and cosmology
        with patch.dict('sys.modules', {'jetsimpy': mock_jetsimpy}), \
                patch('redback.transient_models.afterglow_models.cosmo', self.mock_cosmology):
            self.jetsimpy_tophat(
                time=np.array([1.0]), redshift=self.redshift, thv=self.thv, loge0=self.loge0,
                thc=self.thc, nism=self.nism, A=self.A, p=self.p,
                logepse=self.logepse, logepsb=self.logepsb, g0=self.g0, **kwargs
            )

        # Get the parameter dictionary that was passed
        call_args = mock_jetsimpy.FluxDensity_tophat.call_args
        param_dict = call_args[0][2]

        # Verify all expected parameters
        expected_params = {
            'Eiso': 10 ** self.loge0,
            'lf': self.g0,
            'theta_c': self.thc,
            'n0': self.nism,
            'A': self.A,
            'eps_e': 10 ** self.logepse,
            'eps_b': 10 ** self.logepsb,
            'p': self.p,
            'theta_v': self.thv,
            'd': self.mock_dl.cgs.value * 3.24078e-25,
            'z': self.redshift
        }

        for key, expected_value in expected_params.items():
            self.assertIn(key, param_dict)
            self.assertAlmostEqual(param_dict[key], expected_value)

    def test_time_conversion(self):
        """Test that time is correctly converted from days to seconds."""
        # Create mock jetsimpy module
        mock_jetsimpy = MagicMock()
        mock_jetsimpy.FluxDensity_tophat.return_value = np.array([1e-26, 2e-26, 3e-26])

        kwargs = {
            'output_format': 'flux_density',
            'frequency': 5e14
        }

        input_time = np.array([1.0, 2.0, 3.0])  # days

        # Patch the import, cosmology, and day_to_s
        with patch.dict('sys.modules', {'jetsimpy': mock_jetsimpy}), \
                patch('redback.transient_models.afterglow_models.cosmo', self.mock_cosmology), \
                patch('redback.transient_models.afterglow_models.day_to_s', 86400):
            self.jetsimpy_tophat(
                time=input_time, redshift=self.redshift, thv=self.thv, loge0=self.loge0,
                thc=self.thc, nism=self.nism, A=self.A, p=self.p,
                logepse=self.logepse, logepsb=self.logepsb, g0=self.g0, **kwargs
            )

        # Verify time was converted correctly
        call_args = mock_jetsimpy.FluxDensity_tophat.call_args
        passed_time = call_args[0][0]
        expected_time = input_time * 86400  # Convert days to seconds

        np.testing.assert_array_almost_equal(passed_time, expected_time)

    def test_custom_cosmology(self):
        """Test that custom cosmology is used when provided."""
        custom_cosmology = MagicMock()
        custom_dl = MagicMock()
        custom_dl.cgs.value = 2e28
        custom_cosmology.luminosity_distance.return_value = custom_dl

        # Create mock jetsimpy module
        mock_jetsimpy = MagicMock()
        mock_jetsimpy.FluxDensity_tophat.return_value = np.array([1e-26])

        kwargs = {
            'output_format': 'flux_density',
            'frequency': 5e14,
            'cosmology': custom_cosmology
        }

        # Patch the import (no need to patch cosmo since we're providing custom cosmology)
        with patch.dict('sys.modules', {'jetsimpy': mock_jetsimpy}):
            self.jetsimpy_tophat(
                time=np.array([1.0]), redshift=self.redshift, thv=self.thv, loge0=self.loge0,
                thc=self.thc, nism=self.nism, A=self.A, p=self.p,
                logepse=self.logepse, logepsb=self.logepsb, g0=self.g0, **kwargs
            )

        # Verify custom cosmology was used
        custom_cosmology.luminosity_distance.assert_called_once_with(self.redshift)

        # Verify the custom distance was used in parameter dictionary
        call_args = mock_jetsimpy.FluxDensity_tophat.call_args
        param_dict = call_args[0][2]
        expected_d = custom_dl.cgs.value * 3.24078e-25
        self.assertEqual(param_dict['d'], expected_d)

    def test_all_three_models_called_correctly(self):
        """Test that all three models call their respective jetsimpy functions."""
        # Create mock jetsimpy module
        mock_jetsimpy = MagicMock()
        mock_jetsimpy.FluxDensity_tophat.return_value = np.array([1e-26])
        mock_jetsimpy.FluxDensity_gaussian.return_value = np.array([2e-26])
        mock_jetsimpy.FluxDensity_powerlaw.return_value = np.array([3e-26])

        kwargs = {
            'output_format': 'flux_density',
            'frequency': 5e14
        }

        # Patch the import and cosmology
        with patch.dict('sys.modules', {'jetsimpy': mock_jetsimpy}), \
                patch('redback.transient_models.afterglow_models.cosmo', self.mock_cosmology):
            # Test tophat
            result1 = self.jetsimpy_tophat(
                time=np.array([1.0]), redshift=self.redshift, thv=self.thv, loge0=self.loge0,
                thc=self.thc, nism=self.nism, A=self.A, p=self.p,
                logepse=self.logepse, logepsb=self.logepsb, g0=self.g0, **kwargs
            )

            # Test gaussian
            result2 = self.jetsimpy_gaussian(
                time=np.array([1.0]), redshift=self.redshift, thv=self.thv, loge0=self.loge0,
                thc=self.thc, nism=self.nism, A=self.A, p=self.p,
                logepse=self.logepse, logepsb=self.logepsb, g0=self.g0, **kwargs
            )

            # Test powerlaw
            result3 = self.jetsimpy_powerlaw(
                time=np.array([1.0]), redshift=self.redshift, thv=self.thv, loge0=self.loge0,
                thc=self.thc, nism=self.nism, A=self.A, p=self.p,
                logepse=self.logepse, logepsb=self.logepsb, g0=self.g0, s=self.s, **kwargs
            )

        # Verify each function was called once
        mock_jetsimpy.FluxDensity_tophat.assert_called_once()
        mock_jetsimpy.FluxDensity_gaussian.assert_called_once()
        mock_jetsimpy.FluxDensity_powerlaw.assert_called_once()

        # Verify results
        np.testing.assert_array_equal(result1, np.array([1e-26]))
        np.testing.assert_array_equal(result2, np.array([2e-26]))
        np.testing.assert_array_equal(result3, np.array([3e-26]))



