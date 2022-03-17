import unittest
import bilby
from os import listdir


class TestLoadPriors(unittest.TestCase):

    def setUp(self) -> None:
        self.path_to_files = "../redback/priors/"
        self.prior_files = listdir(self.path_to_files)

    def tearDown(self) -> None:
        pass

    def test_load_priors(self):
        for f in self.prior_files:
            prior_dict = bilby.prior.PriorDict()
            prior_dict.from_file(f"{self.path_to_files}{f}")

    # def test_dollar_signs(self):
    #     for f in self.prior_files:
    #         print()
    #         print(f)
    #         prior_dict = bilby.prior.PriorDict()
    #         prior_dict.from_file(f"{self.path_to_files}{f}")
    #         for k, p in prior_dict.items():
    #             print(k)
    #             occurences = p.latex_label.count("$")
    #             assert occurences % 2 == 0



