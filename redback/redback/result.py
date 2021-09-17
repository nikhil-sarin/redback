import warnings
from pathlib import Path

import bilby
import matplotlib.pyplot as plt
import numpy as np
from bilby.core.result import Result

from . import afterglow as tools
from .utils import MetaDataAccessor

warnings.simplefilter(action='ignore')


class RedbackResult(Result):

    model = MetaDataAccessor('model')
    transient = MetaDataAccessor('transient')
    path = MetaDataAccessor('path')
    use_photon_index_prior = MetaDataAccessor('use_photon_index_prior')
    truncate = MetaDataAccessor('truncate')
    truncate_method = MetaDataAccessor('truncate_method')
    luminosity_data = MetaDataAccessor('luminosity_data')
    save_format = MetaDataAccessor('save_format')

    def __init__(self, label='no_label', outdir='.', sampler=None, search_parameter_keys=None,
                 fixed_parameter_keys=None, constraint_parameter_keys=None, priors=None, sampler_kwargs=None,
                 injection_parameters=None, meta_data=None, posterior=None, samples=None, nested_samples=None,
                 log_evidence=np.nan, log_evidence_err=np.nan, information_gain=np.nan, log_noise_evidence=np.nan, log_bayes_factor=np.nan,
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
    def plot_directory_structure(self):
        """
        :param data: instantiated GRB class
        :param model: model name
        :return: plot_base_directory
        """
        # set up plotting directory structure
        directory = f"{self.transient.path}/{self.transient.name}"
        plots_base_directory = f"{directory}/{self.model}/plots/"
        Path(plots_base_directory).mkdir(parents=True, exist_ok=True)
        return plots_base_directory

    def save_to_file(self, filename=None, overwrite=False, outdir=None,
                     extension='json', gzip=False):
        pass

    def plot_corner(self, parameters=None, priors=None, titles=True, save=True, dpi=300, fontsize=22, **kwargs):
        filename = self.plot_directory_structure + self.model
        if self.use_photon_index_prior:
            filename += '_photon_index_corner.png'
        else:
            filename += '_corner.png'


        super().plot_corner(parameters=parameters, priors=priors, filename=filename, titles=titles,
                            save=save, label_kwargs = dict(fontsize = fontsize), dpi=dpi, **kwargs)
        if kwargs.get('plot_show', False):
            plt.show()

    def plot_lightcurve(self, model, axes=None, plot_save=True,
                        plot_show=True, random_models=1000, outdir=None, **kwargs):
        """
        :param model:
        :param axes:
        :param plot_save:
        :param plot_show:
        :param random_models:

        plots the lightcurve
        model = model to plot
        path = path to GRB folder

        """
        outdir = self.plot_directory_structure if outdir is None else outdir
        self.transient.plot_lightcurve(model=model, axes=axes, plot_save=plot_save, plot_show=plot_show,
                                       random_models=random_models, posterior=self.posterior,
                                       outdir=outdir, use_photon_index_prior=self.use_photon_index_prior, **kwargs)


def read_in_grb_result(model, grb, path='.', truncate=True, use_photon_index_prior=False,
                       truncate_method='prompt_time_error',
                       data_mode='flux', save_format='json'):
    """
    :param model: model to analyse
    :param grb: telephone number of GRB
    :param path: path to GRB
    :param truncate: flag to truncate or not
    :param use_photon_index_prior:
    :param truncate_method:
    :param data_mode:
    :param save_format:
    :return: bilby result object and data object
    """
    result_path = f"{path}/GRB{grb}/{model}/"
    Path(result_path).mkdir(parents=True, exist_ok=True)

    label = data_mode
    if use_photon_index_prior:
        label += '_photon_index'

    result = bilby.result.read_in_result(filename=f"{result_path}{label}_result.{save_format}")
    data = tools.GRB.from_path_and_grb_with_truncation(
        grb=grb, truncate=truncate, path=path, truncate_method=truncate_method,
        data_mode=data_mode)
    return result, data


def calculate_bf(model1, model2, grb, path='.', use_photon_index_prior=False, data_mode='flux',
                 save_format='json'):
    model1, data = read_in_grb_result(model=model1, grb=grb, path=path, use_photon_index_prior=use_photon_index_prior,
                                      data_mode=data_mode, save_format=save_format)
    model2, data = read_in_grb_result(model=model2, grb=grb, path=path, use_photon_index_prior=use_photon_index_prior,
                                      data_mode=data_mode, save_format=save_format)
    log_bf = model1.log_evidence - model2.log_evidence
    return log_bf
