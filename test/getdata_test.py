import unittest
from grb_bilby import getdata


class TestGetTriggerNumber(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_trigger_number(self):
        trigger = getdata.get_trigger_number("041223")
        self.assertEqual("100585", trigger)