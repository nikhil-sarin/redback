"""
Comprehensive regression tests for ALL redback models.

Tests all models against reference data stored in a single compressed file.
If tests fail, regenerate reference data with:
    python test/reference_results/generate_all_model_ref_data.py
"""

import os
import unittest
import numpy as np
import redback
import pickle
import gzip
import warnings

_dirname = os.path.dirname(__file__)
REFERENCE_FILE = os.path.join(_dirname, "reference_results", "all_models_reference.pkl.gz")


class TestAllModelsRegression(unittest.TestCase):
    """Test all models against comprehensive reference data."""
    
    @classmethod
    def setUpClass(cls):
        """Load all reference data once."""
        if not os.path.exists(REFERENCE_FILE):
            raise FileNotFoundError(
                f"Reference data not found at {REFERENCE_FILE}. "
                "Generate it with: python test/reference_results/generate_all_model_ref_data.py"
            )
        
        with gzip.open(REFERENCE_FILE, 'rb') as f:
            cls.reference_data = pickle.load(f)
        
        print(f"\nLoaded reference data for {len(cls.reference_data)} models")
    
    def _test_model(self, model_key):
        """Generic test method for any model."""
        # Parse model key
        category, model_name = model_key.split('.')
        
        # Get model function
        module_map = {
            'kilonova': redback.transient_models.kilonova_models,
            'supernova': redback.transient_models.supernova_models,
            'magnetar': redback.transient_models.magnetar_driven_ejecta_models,
            'tde': redback.transient_models.tde_models,
            'afterglow': redback.transient_models.afterglow_models,
            'shock_powered': redback.transient_models.shock_powered_models,
        }
        
        model_func = getattr(module_map[category], model_name)
        
        # Get reference data
        ref_data = self.reference_data[model_key]
        params_list = ref_data['params']
        results_list = ref_data['results']
        metadata = ref_data['metadata']
        
        # Test each parameter combination
        for params, ref_result in zip(params_list, results_list):
            current_kwargs = {
                'time': metadata['time_array'],
                **metadata['eval_kwargs'],
                **params
            }
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = model_func(**current_kwargs)
            
            # Compare results
            try:
                np.testing.assert_allclose(result, ref_result, rtol=1e-10, atol=1e-12)
            except AssertionError as e:
                self.fail(
                    f"{model_key} output changed for params {params}\n"
                    f"Max difference: {np.max(np.abs(result - ref_result))}\n"
                    f"Original error: {e}"
                )


# Dynamically create test methods for each model
def _create_test_methods():
    """Create a test method for each model in the reference data."""
    # Load reference data to get model list
    if os.path.exists(REFERENCE_FILE):
        with gzip.open(REFERENCE_FILE, 'rb') as f:
            reference_data = pickle.load(f)
        
        # Create a test method for each model
        for model_key in reference_data.keys():
            test_name = f"test_{model_key.replace('.', '_')}"
            
            # Create test method
            def make_test(key):
                def test_method(self):
                    self._test_model(key)
                return test_method
            
            # Add to test class
            setattr(TestAllModelsRegression, test_name, make_test(model_key))
        
        print(f"Created {len(reference_data)} test methods")


# Create test methods when module is imported
_create_test_methods()


if __name__ == '__main__':
    unittest.main()
