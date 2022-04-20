import unittest

import redback.plotting


class PlottingTests(unittest.TestCase):

    def setUp(self) -> None:
        self.plotter = redback.plotting.Plotter(transient=None)

    def tearDown(self) -> None:
        del self.plotter

    def test_all_kwargs_documented(self):
        doc_dict_entries = set(self.plotter.doc_dict.keys())
        plotter_attributes = {a for a in dir(self.plotter) if not a.startswith("_")}
        print(doc_dict_entries)
        self.assertSetEqual(doc_dict_entries, plotter_attributes)
