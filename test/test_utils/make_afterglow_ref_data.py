"""A helper script to generate reference data for the afterglow models.

To regenerate the data run the following command from this directory:
>> python make_afterglow_ref_data.py
"""
import numpy as np
import redback

from ref_data_utils import make_ref_data, test_on_ref_data


time = np.linspace(0.1, 50, 100)
param_grid = dict(
    g0 = [60.0],
    loge0 = [48.0, 50.0, 52.0],
    logepsb = [-2.0],
    logepse = [-1.0],
    logn0 = [-1.0, 0.0, 1.0],
    p = [2.9],
    redshift = [0.1, 0.2],
    thc = [0.03, 0.05, 0.07],
    thv = [0.0, 0.5],
    xiN = [0.1],
)
other_kwargs = {'time': time, 'output_format': 'flux_density', 'frequency': 2e17}

make_ref_data(
    redback.transient_models.afterglow_models.tophat_redback,
    param_grid,
    other_kwargs,
    '../reference_results/tophat_redback_ref_data.ecsv'
)

# Confirm that the data works.
test_on_ref_data(
    redback.transient_models.afterglow_models.tophat_redback,
    '../reference_results/tophat_redback_ref_data.ecsv',
)
