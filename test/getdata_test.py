import unittest
from grb_bilby import getdata


class TestGetTriggerNumber(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_trigger_number(self):
        trigger = getdata.get_trigger_number("GRB041223")
        print(trigger)