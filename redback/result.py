import os
import warnings

import numpy as np
from bilby.core.result import Result
from bilby.core.result import _determine_file_name # noqa

from .transient import TRANSIENT_DICT
from .utils import MetaDataAccessor

warnings.simplefilter(action='ignore')


class RedbackResult(Result):
    model = MetaDataAccessor('model')
    transient_type = MetaDataAccessor('transient_type')
    name = MetaDataAccessor('name')
    path = MetaDataAccessor('path')

    def __init__(self, label='no_label', outdir='.', sampler=None, search_parameter_keys=None,
                 fixed_parameter_keys=None, constraint_parameter_keys=None, priors=None, sampler_kwargs=None,
                 injection_parameters=None, meta_data=None, posterior=None, samples=None, nested_samples=None,
                 log_evidence=np.nan, log_evidence_err=np.nan, information_gain=np.nan, log_noise_evidence=np.nan,
                 log_bayes_factor=np.nan,
                 log_likelihood_evaluations=None, log_prior_evaluations=None, sampling_time=None, nburn=None,
                 num_likelihood_evaluations=None, walkers=None, max_autocorrelation_time=None, use_ratio=None,
                 parameter_labels=None, parameter_labels_with_unit=None, version=None):
        super(RedbackResult, self).__init__(
            label=label, outdir=outdir, sampler=sampler,
            search_parameter_keys=search_parameter_keys, fixed_parameter_keys=fixed_parameter_keys,
            constraint_parameter_keys=constraint_parameter_keys, priors=priors,
            sampler_kwargs=sampler_kwargs, injection_parameters=injection_parameters,
            meta_data=meta_data, posterior=posterior, samples=samples,
            nested_samples=nested_samples, log_evidence=log_evidence,
            log_evidence_err=log_evidence_err, information_gain=information_gain,
            log_noise_evidence=log_noise_evidence, log_bayes_factor=log_bayes_factor,
            log_likelihood_evaluations=log_likelihood_evaluations,
            log_prior_evaluations=log_prior_evaluations, sampling_time=sampling_time, nburn=nburn,
            num_likelihood_evaluations=num_likelihood_evaluations, walkers=walkers,
            max_autocorrelation_time=max_autocorrelation_time, use_ratio=use_ratio,
            parameter_labels=parameter_labels, parameter_labels_with_unit=parameter_labels_with_unit,
            version=version)

    @property
    def transient(self):
        return TRANSIENT_DICT[self.transient_type](**self.meta_data)

    def plot_lightcurve(self, **kwargs):
        self.transient.plot_lightcurve(**kwargs)

    def plot_data(self, **kwargs):
        self.transient.plot_data(**kwargs)

    def plot_multiband(self, **kwargs):
        self.transient.plot_multiband(**kwargs)


def read_in_result(filename=None, outdir=None, label=None, extension='json', gzip=False):
    filename = _determine_file_name(filename, outdir, label, extension, gzip)

    # Get the actual extension (may differ from the default extension if the filename is given)
    extension = os.path.splitext(filename)[1].lstrip('.')
    if extension == 'gz':  # gzipped file
        extension = os.path.splitext(os.path.splitext(filename)[0])[1].lstrip('.')

    if 'json' in extension:
        result = RedbackResult.from_json(filename=filename)
    elif ('hdf5' in extension) or ('h5' in extension):
        result = RedbackResult.from_hdf5(filename=filename)
    elif ("pkl" in extension) or ("pickle" in extension):
        result = RedbackResult.from_pickle(filename=filename)
    elif extension is None:
        raise ValueError("No filetype extension provided")
    else:
        raise ValueError("Filetype {} not understood".format(extension))
    return result
