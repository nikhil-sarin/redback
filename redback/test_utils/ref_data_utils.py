"""A helper function to generate reference data for a given model."""

import numpy as np
from astropy.table import Table
from itertools import product


def make_ref_data(model, param_grid, other_kwargs, filename):
    """Generate reference data for a given model.

    Parameters
    ----------
    model : callable
        The model function to generate data for.
    required_args : dict
        A list of the names of the required arguments for the model function.
    param_grid : dict
        A dictionary where keys are parameter names and values are lists of parameter values to sample.
    other_kwargs : dict
        Additional keyword arguments to pass to the model function.
    filename : str
        The name of the file to save the reference data to (ECSV format).
    """
    # Create a list of all combinations of parameters
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in product(*values)]

    # Generate a table of all inputs to the function and the results.
    all_data = {k: [] for k in keys}
    all_data.update({k: [] for k in other_kwargs.keys()})
    all_data["results"] = []

    for current_params in param_combinations:
        # Save these sets of keyword arguments.
        current_kwargs = {**other_kwargs, **current_params}
        for k, v in current_kwargs.items():
            all_data[k].append(v)

        # Evaluate the model and save the results.
        result = model(**current_kwargs)
        all_data["results"].append(result)

    # Convert to Table and save as an ECSV file.
    table = Table(all_data)
    table.write(filename, format='ascii.ecsv', overwrite=True)


def test_on_ref_data(model, filename):
    """Test a model against reference data.

    Parameters
    ----------
    model : callable
        The model function to test.
    filename : str
        The name of the file containing the reference data (ECSV format).
    other_kwargs : dict
        Additional keyword arguments to pass to the model function.
    """
    # Load the reference data
    table = Table.read(filename, format='ascii.ecsv')

    # Extract parameters and results
    param_names = [col for col in table.colnames if col != "results"]
    params = table[param_names].as_array()
    reference_results = [np.array(x) for x in table["results"]]

    # Test the model against the reference results
    for idx in range(len(table)):
        current_params = {name: params[idx][name] for name in param_names}
        ref_result = reference_results[idx]
        model_result = model(**current_params)
        assert np.allclose(model_result, ref_result), f"Model output does not match reference data for parameters: {current_params}"
