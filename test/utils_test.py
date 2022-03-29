import unittest

import redback


class TestTimeConversion(unittest.TestCase):

    def setUp(self) -> None:
        self.mjd = 0
        self.jd = 2400000.5

        self.year = 1858
        self.month = 11
        self.day = 17

    def tearDown(self) -> None:
        del self.mjd
        del self.jd

        del self.year
        del self.month
        del self.day

    def test_mjd_to_jd(self):
        jd = redback.utils.mjd_to_jd(mjd=self.mjd)
        self.assertEqual(self.jd, jd)

    def test_jd_to_mjd(self):
        mjd = redback.utils.jd_to_mjd(jd=self.jd)
        self.assertEqual(self.mjd, mjd)

    def test_jd_to_date(self):
        year, month, day = redback.utils.jd_to_date(jd=self.jd)
        self.assertEqual(self.year, year)
        self.assertEqual(self.month, month)
        self.assertEqual(self.day, day)

    def test_mjd_to_date(self):
        year, month, day = redback.utils.mjd_to_date(mjd=self.mjd)
        self.assertEqual(self.year, year)
        self.assertEqual(self.month, month)
        self.assertEqual(self.day, day)

    def test_date_to_jd(self):
        jd = redback.utils.date_to_jd(year=self.year, month=self.month, day=self.day)
        self.assertEqual(self.jd, jd)

    def test_date_to_mjd(self):
        mjd = redback.utils.date_to_mjd(year=self.year, month=self.month, day=self.day)
        self.assertEqual(self.mjd, mjd)
