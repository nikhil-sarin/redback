import unittest
from os import listdir
from os.path import dirname

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
        kwargs = dict(frequency=2e14, output_format="flux_density")
        times = np.array([1, 2, 3])
        # for k in redback.model_library.all_models_dict:
        #     print(k)
        for f in self.prior_files:
            print(f)
            prior = self.get_prior(file=f)
            function = redback.model_library.all_models_dict[f.replace(".prior", "")]
            ys = function(times, **prior.sample(), **kwargs)
            self.assertEqual(len(times), len(ys))
