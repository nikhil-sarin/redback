"""
Generate reference data for ALL redback models with their default priors.

This comprehensive test will catch any changes to model outputs across the entire codebase.
All reference data is stored in a single pickle file for efficiency.

Run this to regenerate all reference data:
    python test/reference_results/generate_all_model_ref_data.py --quick  # Test a few models
    python test/reference_results/generate_all_model_ref_data.py          # Test all models
"""

import numpy as np
import redback
import inspect
import warnings
import argparse
import os
import pickle
import gzip

# Common time arrays
TIME_SHORT = np.array([0.1, 1.0, 5.0])  # days
TIME_LONG = np.array([1.0, 10.0, 100.0])  # days

# Output file for all reference data
REFERENCE_FILE = 'test/reference_results/all_models_reference.pkl.gz'


def get_model_functions(module):
    """Get all model functions from a module."""
    # Skip these - they're utilities not models
    skip_names = ['calc_kcorrected_properties', 'afterglow_models_sed', 
                  'afterglow_models_with_energy_injection', 'afterglow_models_with_jet_spread',
                  'csm_shock_breakout_bolometric', 'integrated_flux_afterglow',
                  'get_correct_output_format_from_spectra']
    
    functions = []
    for name in dir(module):
        if name.startswith('_') or name in skip_names:
            continue
        obj = getattr(module, name)
        if callable(obj) and not inspect.isclass(obj):
            # Check if it looks like a model (has 'time' parameter)
            try:
                sig = inspect.signature(obj)
                if 'time' in sig.parameters:
                    functions.append((name, obj))
            except:
                pass
    return functions


def sample_from_prior(prior, n_samples=3):
    """Sample n parameter sets from prior, spread across the range."""
    param_sets = []
    
    for i in range(n_samples):
        params = {}
        fraction = i / max(1, n_samples - 1)  # 0.0, 0.5, 1.0 for n=3
        
        for key in prior.keys():
            param_prior = prior[key]
            
            # Get value at this fraction of the range
            if hasattr(param_prior, 'minimum') and hasattr(param_prior, 'maximum'):
                low = param_prior.minimum
                high = param_prior.maximum
                params[key] = low + fraction * (high - low)
            elif hasattr(param_prior, 'mu') and hasattr(param_prior, 'sigma'):
                mu = param_prior.mu
                sigma = param_prior.sigma
                params[key] = mu + (fraction - 0.5) * sigma
            else:
                # Direct sampling
                try:
                    sample = prior.sample(1)
                    params[key] = sample[key].values[0]
                except:
                    params[key] = 1.0 + fraction  # Fallback
        
        param_sets.append(params)
    
    return param_sets


def generate_ref_data_for_model(category, model_name, model_func, time_array):
    """Generate reference data for a single model.
    
    Returns:
        (success, error, ref_data_dict) where ref_data_dict contains:
            - params: list of 3 parameter dicts
            - results: list of 3 model outputs
            - metadata: time_array, kwargs used
    """
    try:
        # Get prior
        prior = redback.priors.get_priors(model=model_name)
        
        # Sample 3 parameter sets from prior (low, middle, high)
        param_sets = sample_from_prior(prior, n_samples=3)
        
        # Remove redshift from params (we'll set it in kwargs)
        for params in param_sets:
            if 'redshift' in params:
                del params['redshift']
        
        # Set up evaluation kwargs
        eval_kwargs = {
            'time': time_array,
            'redshift': 0.01,
            'output_format': 'flux' if model_name == 'trapped_magnetar' else 'flux_density',
            'frequency': 1e14,
        }
        
        # Add photon_index for trapped_magnetar flux output
        if model_name == 'trapped_magnetar':
            eval_kwargs['photon_index'] = 2.0
        
        # Evaluate model for each parameter set
        params_list = []
        results_list = []
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for params in param_sets:
                current_kwargs = {**eval_kwargs, **params}
                result = model_func(**current_kwargs)
                params_list.append(params)
                results_list.append(result)
        
        ref_data = {
            'params': params_list,
            'results': results_list,
            'metadata': {
                'time_array': time_array,
                'eval_kwargs': {k: v for k, v in eval_kwargs.items() if k != 'time'}
            }
        }
        
        return True, None, ref_data
        
    except Exception as e:
        return False, str(e), None


def generate_ref_data_for_category(category, module, time_array, quick_mode=False):
    """Generate reference data for all models in a category."""
    functions = get_model_functions(module)
    
    # Models to skip (require extra packages not in requirements)
    skip_models = {
        'vegas_powerlaw', 'vegas_gaussian', 'vegas_tophat',
        'jetsimpy_powerlaw', 'jetsimpy_gaussian', 'jetsimpy_tophat',
    }
    
    # Filter out models that should be skipped
    functions = [(name, func) for name, func in functions if name not in skip_models]
    
    if quick_mode:
        functions = functions[:3]  # Just test first 3
    
    print(f"\n{'='*70}")
    print(f"Processing {category.upper()} models ({len(functions)} models)")
    print(f"{'='*70}")
    
    successful = {}
    failed = []
    
    for model_name, model_func in functions:
        success, error, ref_data = generate_ref_data_for_model(category, model_name, model_func, time_array)
        
        if success:
            successful[f"{category}.{model_name}"] = ref_data
            print(f"  ✓ {model_name}")
        else:
            failed.append((model_name, error))
            print(f"  ✗ {model_name}: {error[:60] if error else 'Unknown error'}")
    
    return successful, failed


def main():
    parser = argparse.ArgumentParser(description='Generate model reference data')
    parser.add_argument('--quick', action='store_true', help='Quick mode: test only 3 models per category')
    args = parser.parse_args()
    
    print("="*70)
    print("GENERATING COMPREHENSIVE MODEL REFERENCE DATA")
    print("="*70)
    if args.quick:
        print("QUICK MODE: Testing 3 models per category")
    else:
        print("FULL MODE: Testing ALL models")
    print()
    
    all_reference_data = {}
    all_failed = []
    
    # Process each category
    categories = [
        ('kilonova', redback.transient_models.kilonova_models, TIME_SHORT),
        ('supernova', redback.transient_models.supernova_models, TIME_LONG),
        ('magnetar', redback.transient_models.magnetar_driven_ejecta_models, TIME_LONG),
        ('tde', redback.transient_models.tde_models, TIME_LONG),
        ('afterglow', redback.transient_models.afterglow_models, TIME_SHORT),
        ('shock_powered', redback.transient_models.shock_powered_models, TIME_LONG),
    ]
    
    for category, module, time_array in categories:
        successful, failed = generate_ref_data_for_category(category, module, time_array, args.quick)
        all_reference_data.update(successful)
        all_failed.extend([(category, m, e) for m, e in failed])
    
    # Save all reference data to a single compressed file
    print()
    print("="*70)
    print("Saving reference data...")
    with gzip.open(REFERENCE_FILE, 'wb') as f:
        pickle.dump(all_reference_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Get file size
    file_size_kb = os.path.getsize(REFERENCE_FILE) / 1024
    
    # Summary
    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"✓ Successful: {len(all_reference_data)} models")
    print(f"✗ Failed:     {len(all_failed)} models")
    print(f"📦 File size:  {file_size_kb:.1f} KB")
    
    if all_reference_data:
        success_rate = 100*len(all_reference_data)/(len(all_reference_data)+len(all_failed))
        print(f"📊 Success rate: {success_rate:.1f}%")
    
    if all_failed and len(all_failed) <= 20:
        print("\nFailed models:")
        for category, model, error in all_failed:
            print(f"  • {category}.{model}: {error[:60] if error else ''}")
    elif all_failed:
        print(f"\n{len(all_failed)} models failed (run with --quick to see details)")
    
    print()
    print("="*70)
    print(f"✓ Saved reference data to {REFERENCE_FILE}")
    print(f"✓ Ready for CI testing!")
    print("="*70)


if __name__ == '__main__':
    main()
