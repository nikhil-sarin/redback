import unittest
import bilby
from os import listdir
from os.path import dirname
import pandas as pd
from pathlib import Path
from shutil import rmtree

_dirname = dirname(__file__)


class TestLoadPriors(unittest.TestCase):

    def setUp(self) -> None:
        self.path_to_files = f"{_dirname}/../redback/priors/"
        self.prior_files = listdir(self.path_to_files)

    def tearDown(self) -> None:
        pass

    def test_load_priors(self):
        for f in self.prior_files:
            prior_dict = bilby.prior.PriorDict()
            prior_dict.from_file(f"{self.path_to_files}{f}")


class TestCornerPlotPriorSamples(unittest.TestCase):
    outdir = "testing_corner"

    @classmethod
    def setUpClass(cls) -> None:
        Path(cls.outdir).mkdir(exist_ok=True, parents=True)

    @classmethod
    def tearDownClass(cls) -> None:
        rmtree(cls.outdir)

    def setUp(self) -> None:
        self.path_to_files = f"{_dirname}/../redback/priors/"
        self.prior_files = listdir(self.path_to_files)

    def tearDown(self) -> None:
        pass

    def get_prior(self, file):
        prior_dict = bilby.prior.PriorDict()
        prior_dict.from_file(f"{self.path_to_files}{file}")
        return prior_dict

    def get_posterior(self, file):
        return pd.DataFrame.from_dict(self.get_prior(file=file).sample(100))

    def get_result(self, file):
        prior = self.get_prior(file=file)
        posterior = self.get_posterior(file=file)
        search_parameter_keys = [k for k, v in prior.items() if
                                 not isinstance(v, (bilby.core.prior.DeltaFunction, bilby.core.prior.Constraint, float, int))]
        fixed_parameter_keys = [k for k, v in prior.items() if isinstance(v, (bilby.core.prior.DeltaFunction, float, int))]
        constraint_parameter_keys = [k for k, v in prior.items() if isinstance(v, bilby.core.prior.Constraint)]
        return bilby.result.Result(label=file, outdir=self.outdir,
                                   search_parameter_keys=search_parameter_keys,
                                   fixed_parameter_keys=fixed_parameter_keys,
                                   constraint_parameter_keys=constraint_parameter_keys, priors=prior,
                                   sampler_kwargs=dict(), injection_parameters=None,
                                   meta_data=None, posterior=posterior, samples=None,
                                   nested_samples=None, log_evidence=0,
                                   log_evidence_err=0, information_gain=0,
                                   log_noise_evidence=0, log_bayes_factor=0,
                                   log_likelihood_evaluations=0,
                                   log_prior_evaluations=0, sampling_time=0, nburn=0,
                                   num_likelihood_evaluations=0, walkers=0,
                                   max_autocorrelation_time=0, use_ratio=False,
                                   version=None)

    def test_plot_priors(self):
        for f in self.prior_files:
            print(f)
            res = self.get_result(file=f)
            res.plot_corner()
