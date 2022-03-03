import unittest
import bilby
from os import listdir


class TestLoadPriors(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_load_priors(self):
        path_to_files = "../redback/priors/"
        prior_files = listdir(path_to_files)

        for f in prior_files:
            print(f)
            prior_dict = bilby.prior.PriorDict()
            prior_dict.from_file(f"{path_to_files}{f}")
