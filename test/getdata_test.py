import unittest
from redback import getdata


class TestGetTriggerNumber(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_trigger_number(self):
        trigger = getdata.get_trigger_number("041223")
        self.assertEqual("100585", trigger)


class TestGetGRBTable(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass


class TestCheckElement(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass


class TestGetGRBFile(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_get_grb_file(self):
        grb_file = getdata.get_grb_file(grb="GRB041223", use_default_directory=True)
        self.assertIsNotNone(grb_file)


class TestSortData(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass


class TestRetrieveAndProcessData(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass
