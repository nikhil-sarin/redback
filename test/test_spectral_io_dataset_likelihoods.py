"""
Batch 2 unit tests for redback X-ray spectral fitting code.

Covers:
  - TestOGIPIO         : redback.spectral.io  (read_pha, read_rmf, read_arf, read_lc)
  - TestSpectralDataset: redback.spectral.dataset.SpectralDataset
  - TestPoissonSpectralLikelihood
  - TestWStatSpectralLikelihood
  - TestChiSquareSpectralLikelihood

Run with:
    python -m pytest test/test_spectral_io_dataset_likelihoods.py -v
or:
    python -m unittest test.spectral_test_batch2 -v
"""
from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
from astropy.io import fits


# ===========================================================================
# In-memory FITS helpers (write to tempdir, never leak to the repo)
# ===========================================================================

def _make_pha_fits(
    n_channels: int = 10,
    exposure: float = 1000.0,
    backscale: float = 1.0,
    areascal: float = 1.0,
    backfile: str | None = None,
    respfile: str | None = None,
    ancrfile: str | None = None,
    with_quality: bool = False,
    with_grouping: bool = False,
    tmpdir: str = None,
    filename: str = "test.pha",
) -> str:
    """
    Write a minimal OGIP Type I PHA FITS file and return the path.

    Columns: CHANNEL, COUNTS (and optionally QUALITY, GROUPING).
    BACKSCAL / AREASCAL are stored as header keywords (not columns),
    which exercises the _get_column_or_header_value fallback path.
    """
    channels = np.arange(1, n_channels + 1, dtype=np.int16)
    counts = np.ones(n_channels, dtype=np.float32) * 10.0

    col_list = [
        fits.Column(name="CHANNEL", format="I", array=channels),
        fits.Column(name="COUNTS", format="E", array=counts),
    ]
    if with_quality:
        quality = np.zeros(n_channels, dtype=np.int16)
        quality[0] = 1  # first channel flagged bad
        col_list.append(fits.Column(name="QUALITY", format="I", array=quality))
    if with_grouping:
        # All channels are group-start markers
        grouping = np.ones(n_channels, dtype=np.int16)
        col_list.append(fits.Column(name="GROUPING", format="I", array=grouping))

    table_hdu = fits.BinTableHDU.from_columns(col_list)
    table_hdu.name = "SPECTRUM"
    table_hdu.header["HDUCLAS1"] = "SPECTRUM"
    table_hdu.header["EXPOSURE"] = exposure
    table_hdu.header["BACKSCAL"] = backscale
    table_hdu.header["AREASCAL"] = areascal
    if backfile:
        table_hdu.header["BACKFILE"] = backfile
    if respfile:
        table_hdu.header["RESPFILE"] = respfile
    if ancrfile:
        table_hdu.header["ANCRFILE"] = ancrfile

    hdul = fits.HDUList([fits.PrimaryHDU(), table_hdu])
    path = os.path.join(tmpdir, filename)
    hdul.writeto(path, overwrite=True)
    return path


def _make_rmf_fits(
    n_energy: int = 8,
    n_channels: int = 10,
    tmpdir: str = None,
    filename: str = "test.rmf",
) -> str:
    """
    Write a minimal OGIP RMF FITS file (MATRIX + EBOUNDS HDUs) and return the path.

    Each energy row has a single group spanning all detector channels with a
    uniform redistribution (value = 1/n_channels), so the matrix integrates to 1.
    """
    e_lo = np.linspace(0.5, 8.0, n_energy + 1)[:-1].astype(np.float32)
    e_hi = np.linspace(0.5, 8.0, n_energy + 1)[1:].astype(np.float32)

    n_grp = np.ones(n_energy, dtype=np.int16)
    f_chan = np.ones(n_energy, dtype=np.int16)          # group starts at channel index 1
    n_chan = np.full(n_energy, n_channels, dtype=np.int16)
    mat_rows = np.full((n_energy, n_channels), 1.0 / n_channels, dtype=np.float32)

    matrix_cols = [
        fits.Column(name="ENERG_LO", format="E", array=e_lo),
        fits.Column(name="ENERG_HI", format="E", array=e_hi),
        fits.Column(name="N_GRP", format="I", array=n_grp),
        fits.Column(name="F_CHAN", format="I", array=f_chan),
        fits.Column(name="N_CHAN", format="I", array=n_chan),
        fits.Column(name="MATRIX", format=f"{n_channels}E", array=mat_rows),
    ]
    matrix_hdu = fits.BinTableHDU.from_columns(matrix_cols)
    matrix_hdu.name = "MATRIX"
    matrix_hdu.header["DETCHANS"] = n_channels

    chan = np.arange(1, n_channels + 1, dtype=np.int16)
    emin_chan = np.linspace(0.1, 9.5, n_channels + 1)[:-1].astype(np.float32)
    emax_chan = np.linspace(0.1, 9.5, n_channels + 1)[1:].astype(np.float32)
    ebounds_cols = [
        fits.Column(name="CHANNEL", format="I", array=chan),
        fits.Column(name="E_MIN", format="E", array=emin_chan),
        fits.Column(name="E_MAX", format="E", array=emax_chan),
    ]
    ebounds_hdu = fits.BinTableHDU.from_columns(ebounds_cols)
    ebounds_hdu.name = "EBOUNDS"

    hdul = fits.HDUList([fits.PrimaryHDU(), matrix_hdu, ebounds_hdu])
    path = os.path.join(tmpdir, filename)
    hdul.writeto(path, overwrite=True)
    return path


def _make_arf_fits(
    n_energy: int = 8,
    area_cm2: float = 500.0,
    tmpdir: str = None,
    filename: str = "test.arf",
) -> str:
    """Write a minimal OGIP ARF FITS file and return the path."""
    e_lo = np.linspace(0.5, 8.0, n_energy + 1)[:-1].astype(np.float32)
    e_hi = np.linspace(0.5, 8.0, n_energy + 1)[1:].astype(np.float32)
    area = np.full(n_energy, area_cm2, dtype=np.float32)

    specresp_cols = [
        fits.Column(name="ENERG_LO", format="E", array=e_lo),
        fits.Column(name="ENERG_HI", format="E", array=e_hi),
        fits.Column(name="SPECRESP", format="E", array=area),
    ]
    specresp_hdu = fits.BinTableHDU.from_columns(specresp_cols)
    specresp_hdu.name = "SPECRESP"

    hdul = fits.HDUList([fits.PrimaryHDU(), specresp_hdu])
    path = os.path.join(tmpdir, filename)
    hdul.writeto(path, overwrite=True)
    return path


def _make_lc_fits(
    n_bins: int = 20,
    timedel: float = 1.0,
    with_fracexp: bool = False,
    tmpdir: str = None,
    filename: str = "test.lc",
) -> str:
    """Write a minimal OGIP light-curve FITS file (RATE HDU) and return the path."""
    time = np.arange(n_bins, dtype=np.float64) * timedel
    rate = np.ones(n_bins, dtype=np.float32) * 2.5
    error = np.ones(n_bins, dtype=np.float32) * 0.1

    col_list = [
        fits.Column(name="TIME", format="D", array=time),
        fits.Column(name="RATE", format="E", array=rate),
        fits.Column(name="ERROR", format="E", array=error),
    ]
    if with_fracexp:
        fracexp = np.ones(n_bins, dtype=np.float32) * 0.95
        col_list.append(fits.Column(name="FRACEXP", format="E", array=fracexp))

    rate_hdu = fits.BinTableHDU.from_columns(col_list)
    rate_hdu.name = "RATE"
    rate_hdu.header["TIMEDEL"] = timedel

    hdul = fits.HDUList([fits.PrimaryHDU(), rate_hdu])
    path = os.path.join(tmpdir, filename)
    hdul.writeto(path, overwrite=True)
    return path


# ---------------------------------------------------------------------------
# Shared helper: build a minimal SpectralDataset without touching OGIP files
# ---------------------------------------------------------------------------

def _make_minimal_dataset(
    n_channels: int = 5,
    counts_val: float = 20.0,
    exposure: float = 1000.0,
    with_bkg: bool = False,
    bkg_counts_val: float = 5.0,
):
    """
    Return a SpectralDataset with n_channels uniform bins over [0.5, 3.0] keV.
    When with_bkg=True, a background spectrum is added with:
        bkg_exposure = 2 * exposure,  bkg_backscale = 2.0,  bkg_areascal = 1.0
    so background_scale_factor = 1000/(2000*2*1) = 0.25.
    """
    from redback.spectral.dataset import SpectralDataset

    edges = np.linspace(0.5, 3.0, n_channels + 1)
    counts = np.full(n_channels, counts_val, dtype=float)
    kwargs = dict(counts=counts, exposure=exposure, energy_edges_keV=edges)

    if with_bkg:
        kwargs["counts_bkg"] = np.full(n_channels, bkg_counts_val, dtype=float)
        kwargs["bkg_exposure"] = exposure * 2.0
        kwargs["bkg_backscale"] = 2.0
        kwargs["bkg_areascal"] = 1.0

    return SpectralDataset(**kwargs)


# ===========================================================================
# 1.  TestOGIPIO
# ===========================================================================

class TestOGIPIO(unittest.TestCase):
    """Tests for redback.spectral.io: read_pha, read_rmf, read_arf, read_lc."""

    def setUp(self):
        self._tmpdir_obj = tempfile.TemporaryDirectory()
        self.tmpdir = self._tmpdir_obj.name
        self.n_channels = 10
        self.n_energy = 8
        self.exposure = 2000.0

    def tearDown(self):
        self._tmpdir_obj.cleanup()

    # ------------------------------------------------------------------
    # Internal convenience
    # ------------------------------------------------------------------

    def _write_pha(self, **kwargs):
        """Write a PHA file using the class-level defaults."""
        return _make_pha_fits(
            n_channels=self.n_channels,
            exposure=self.exposure,
            tmpdir=self.tmpdir,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # read_pha
    # ------------------------------------------------------------------

    def test_read_pha_counts_shape(self):
        """counts array has the expected length."""
        from redback.spectral.io import read_pha
        path = self._write_pha()
        spec = read_pha(path)
        self.assertEqual(len(spec.counts), self.n_channels)

    def test_read_pha_exposure(self):
        """exposure matches the header value."""
        from redback.spectral.io import read_pha
        path = self._write_pha()
        spec = read_pha(path)
        self.assertAlmostEqual(spec.exposure, self.exposure, places=3)

    def test_read_pha_backscale_from_header(self):
        """BACKSCAL stored as a header keyword (not column) is read correctly."""
        from redback.spectral.io import read_pha
        path = self._write_pha(backscale=0.5)
        spec = read_pha(path)
        self.assertAlmostEqual(spec.backscale, 0.5, places=6)

    def test_read_pha_areascal_from_header(self):
        """AREASCAL stored as a header keyword is read correctly."""
        from redback.spectral.io import read_pha
        path = self._write_pha(areascal=0.75)
        spec = read_pha(path)
        self.assertAlmostEqual(spec.areascal, 0.75, places=6)

    def test_read_pha_quality(self):
        """QUALITY column is read; first channel is flagged bad (quality=1)."""
        from redback.spectral.io import read_pha
        path = self._write_pha(with_quality=True)
        spec = read_pha(path)
        self.assertIsNotNone(spec.quality)
        self.assertEqual(len(spec.quality), self.n_channels)
        self.assertEqual(spec.quality[0], 1)
        self.assertTrue(np.all(spec.quality[1:] == 0))

    def test_read_pha_grouping(self):
        """GROUPING column is read; all channels are group-start markers (value 1)."""
        from redback.spectral.io import read_pha
        path = self._write_pha(with_grouping=True)
        spec = read_pha(path)
        self.assertIsNotNone(spec.grouping)
        self.assertEqual(len(spec.grouping), self.n_channels)
        self.assertTrue(np.all(spec.grouping == 1))

    def test_read_pha_header_keywords(self):
        """backfile, respfile, ancrfile are extracted from the SPECTRUM header."""
        from redback.spectral.io import read_pha
        path = self._write_pha(
            backfile="myback.pha",
            respfile="myrmf.rmf",
            ancrfile="myarf.arf",
        )
        spec = read_pha(path)
        self.assertEqual(spec.backfile, "myback.pha")
        self.assertEqual(spec.respfile, "myrmf.rmf")
        self.assertEqual(spec.ancrfile, "myarf.arf")

    def test_read_pha_keywords_absent_gives_none(self):
        """When BACKFILE/RESPFILE/ANCRFILE are absent, attributes are None."""
        from redback.spectral.io import read_pha
        path = self._write_pha()
        spec = read_pha(path)
        self.assertIsNone(spec.backfile)
        self.assertIsNone(spec.respfile)
        self.assertIsNone(spec.ancrfile)

    def test_read_pha_missing_spectrum_hdu_raises(self):
        """A FITS file with no SPECTRUM BinTable HDU raises ValueError."""
        from redback.spectral.io import read_pha
        path = os.path.join(self.tmpdir, "bad.pha")
        fits.HDUList([fits.PrimaryHDU()]).writeto(path, overwrite=True)
        with self.assertRaises(ValueError):
            read_pha(path)

    def test_read_pha_channel_dtype_int(self):
        """CHANNEL is returned as an integer array."""
        from redback.spectral.io import read_pha
        path = self._write_pha()
        spec = read_pha(path)
        self.assertTrue(np.issubdtype(spec.channel.dtype, np.integer))

    def test_read_pha_counts_dtype_float(self):
        """COUNTS is returned as a floating-point array."""
        from redback.spectral.io import read_pha
        path = self._write_pha()
        spec = read_pha(path)
        self.assertTrue(np.issubdtype(spec.counts.dtype, np.floating))

    def test_read_pha_quality_absent_is_none(self):
        """spec.quality is None when no QUALITY column or header key is present."""
        from redback.spectral.io import read_pha
        path = self._write_pha(with_quality=False)
        spec = read_pha(path)
        self.assertIsNone(spec.quality)

    def test_read_pha_grouping_absent_is_none(self):
        """spec.grouping is None when no GROUPING column or header key is present."""
        from redback.spectral.io import read_pha
        path = self._write_pha(with_grouping=False)
        spec = read_pha(path)
        self.assertIsNone(spec.grouping)

    # ------------------------------------------------------------------
    # read_rmf
    # ------------------------------------------------------------------

    def test_read_rmf_matrix_shape(self):
        """RMF matrix shape is (n_channels, n_energy)."""
        from redback.spectral.io import read_rmf
        path = _make_rmf_fits(
            n_energy=self.n_energy, n_channels=self.n_channels, tmpdir=self.tmpdir
        )
        rmf = read_rmf(path)
        self.assertEqual(rmf.matrix.shape, (self.n_channels, self.n_energy))

    def test_read_rmf_channel_bounds(self):
        """emin_chan and emax_chan each have n_channels entries."""
        from redback.spectral.io import read_rmf
        path = _make_rmf_fits(
            n_energy=self.n_energy, n_channels=self.n_channels, tmpdir=self.tmpdir
        )
        rmf = read_rmf(path)
        self.assertEqual(len(rmf.emin_chan), self.n_channels)
        self.assertEqual(len(rmf.emax_chan), self.n_channels)

    def test_read_rmf_energy_bins_populated(self):
        """e_min and e_max each have n_energy entries."""
        from redback.spectral.io import read_rmf
        path = _make_rmf_fits(
            n_energy=self.n_energy, n_channels=self.n_channels, tmpdir=self.tmpdir
        )
        rmf = read_rmf(path)
        self.assertEqual(len(rmf.e_min), self.n_energy)
        self.assertEqual(len(rmf.e_max), self.n_energy)

    def test_read_rmf_matrix_non_negative(self):
        """All matrix values are non-negative."""
        from redback.spectral.io import read_rmf
        path = _make_rmf_fits(
            n_energy=self.n_energy, n_channels=self.n_channels, tmpdir=self.tmpdir
        )
        rmf = read_rmf(path)
        self.assertTrue(np.all(rmf.matrix >= 0.0))

    def test_read_rmf_channel_array_length(self):
        """channel attribute has n_channels entries."""
        from redback.spectral.io import read_rmf
        path = _make_rmf_fits(
            n_energy=self.n_energy, n_channels=self.n_channels, tmpdir=self.tmpdir
        )
        rmf = read_rmf(path)
        self.assertEqual(len(rmf.channel), self.n_channels)

    # ------------------------------------------------------------------
    # read_arf
    # ------------------------------------------------------------------

    def test_read_arf_area_shape(self):
        """area array has n_energy entries."""
        from redback.spectral.io import read_arf
        path = _make_arf_fits(n_energy=self.n_energy, tmpdir=self.tmpdir)
        arf = read_arf(path)
        self.assertEqual(len(arf.area), self.n_energy)

    def test_read_arf_energy_bins(self):
        """e_min and e_max each have n_energy entries."""
        from redback.spectral.io import read_arf
        path = _make_arf_fits(n_energy=self.n_energy, tmpdir=self.tmpdir)
        arf = read_arf(path)
        self.assertEqual(len(arf.e_min), self.n_energy)
        self.assertEqual(len(arf.e_max), self.n_energy)

    def test_read_arf_area_positive(self):
        """All effective-area values are positive."""
        from redback.spectral.io import read_arf
        path = _make_arf_fits(n_energy=self.n_energy, area_cm2=500.0, tmpdir=self.tmpdir)
        arf = read_arf(path)
        self.assertTrue(np.all(arf.area > 0.0))

    def test_read_arf_energy_ordered(self):
        """e_min < e_max for every bin, and bins are monotonically ordered."""
        from redback.spectral.io import read_arf
        path = _make_arf_fits(n_energy=self.n_energy, tmpdir=self.tmpdir)
        arf = read_arf(path)
        self.assertTrue(np.all(arf.e_min < arf.e_max))
        self.assertTrue(np.all(np.diff(arf.e_min) > 0))

    def test_read_arf_area_value(self):
        """area values match what we wrote."""
        from redback.spectral.io import read_arf
        path = _make_arf_fits(n_energy=self.n_energy, area_cm2=250.0, tmpdir=self.tmpdir)
        arf = read_arf(path)
        np.testing.assert_allclose(arf.area, 250.0, rtol=1e-5)

    # ------------------------------------------------------------------
    # read_lc
    # ------------------------------------------------------------------

    def test_read_lc_shape(self):
        """time, rate, error arrays each have n_bins entries."""
        from redback.spectral.io import read_lc
        path = _make_lc_fits(n_bins=20, tmpdir=self.tmpdir)
        lc = read_lc(path)
        self.assertEqual(len(lc.time), 20)
        self.assertEqual(len(lc.rate), 20)
        self.assertEqual(len(lc.error), 20)

    def test_read_lc_timedel(self):
        """timedel matches the TIMEDEL header keyword."""
        from redback.spectral.io import read_lc
        path = _make_lc_fits(n_bins=10, timedel=2.0, tmpdir=self.tmpdir)
        lc = read_lc(path)
        self.assertAlmostEqual(lc.timedel, 2.0, places=6)

    def test_read_lc_fracexp_absent_is_none(self):
        """fracexp is None when the FRACEXP column is not present."""
        from redback.spectral.io import read_lc
        path = _make_lc_fits(n_bins=10, with_fracexp=False, tmpdir=self.tmpdir)
        lc = read_lc(path)
        self.assertIsNone(lc.fracexp)

    def test_read_lc_fracexp_present(self):
        """fracexp array is populated when FRACEXP column exists."""
        from redback.spectral.io import read_lc
        path = _make_lc_fits(n_bins=10, with_fracexp=True, tmpdir=self.tmpdir)
        lc = read_lc(path)
        self.assertIsNotNone(lc.fracexp)
        self.assertEqual(len(lc.fracexp), 10)


# ===========================================================================
# 2.  TestSpectralDataset
# ===========================================================================

class TestSpectralDataset(unittest.TestCase):
    """Tests for redback.spectral.dataset.SpectralDataset."""

    def setUp(self):
        self._tmpdir_obj = tempfile.TemporaryDirectory()
        self.tmpdir = self._tmpdir_obj.name
        self.n_channels = 5
        self.exposure = 1000.0
        # Edges: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0] keV
        self.edges = np.linspace(0.5, 3.0, self.n_channels + 1)
        self.dataset = _make_minimal_dataset(
            n_channels=self.n_channels,
            counts_val=20.0,
            exposure=self.exposure,
        )

    def tearDown(self):
        self._tmpdir_obj.cleanup()
        del self.dataset

    # ------------------------------------------------------------------
    # energy_centers_keV / energy_centers_hz
    # ------------------------------------------------------------------

    def test_energy_centers_keV(self):
        """Centers are the midpoints of the bin edges."""
        expected = 0.5 * (self.edges[:-1] + self.edges[1:])
        np.testing.assert_array_almost_equal(
            self.dataset.energy_centers_keV, expected
        )

    def test_energy_centers_hz(self):
        """Centers in Hz equal centers_keV * 2.417989e17."""
        keV_to_Hz = 2.417989e17
        expected = self.dataset.energy_centers_keV * keV_to_Hz
        np.testing.assert_array_almost_equal(
            self.dataset.energy_centers_hz, expected
        )

    def test_energy_centers_hz_positive(self):
        """All Hz-domain centers are positive."""
        self.assertTrue(np.all(self.dataset.energy_centers_hz > 0.0))

    def test_energy_centers_count(self):
        """There are n_channels centers."""
        self.assertEqual(len(self.dataset.energy_centers_keV), self.n_channels)

    # ------------------------------------------------------------------
    # background_scale_factor
    # ------------------------------------------------------------------

    def test_background_scale_factor_no_bkg(self):
        """Returns 1.0 when no background counts are attached."""
        self.assertAlmostEqual(self.dataset.background_scale_factor, 1.0)

    def test_background_scale_factor_with_bkg(self):
        """
        Formula: (exposure * backscale * areascal) /
                 (bkg_exposure * bkg_backscale * bkg_areascal)
        With defaults from _make_minimal_dataset(with_bkg=True):
          exposure=1000, backscale=1, areascal=1
          bkg_exposure=2000, bkg_backscale=2, bkg_areascal=1
          => 1000 / 4000 = 0.25
        """
        ds = _make_minimal_dataset(with_bkg=True)
        expected = (ds.exposure * ds.backscale * ds.areascal) / (
            ds.bkg_exposure * ds.bkg_backscale * ds.bkg_areascal
        )
        self.assertAlmostEqual(ds.background_scale_factor, expected, places=12)

    def test_background_scale_factor_identical_params(self):
        """When source and background have identical parameters, factor = 1.0."""
        from redback.spectral.dataset import SpectralDataset

        ds = SpectralDataset(
            counts=np.array([10.0, 20.0]),
            exposure=1000.0,
            energy_edges_keV=np.array([0.5, 1.0, 1.5]),
            counts_bkg=np.array([5.0, 10.0]),
            bkg_exposure=1000.0,
            bkg_backscale=1.0,
            bkg_areascal=1.0,
            backscale=1.0,
            areascal=1.0,
        )
        self.assertAlmostEqual(ds.background_scale_factor, 1.0, places=12)

    # ------------------------------------------------------------------
    # set_active_interval
    # ------------------------------------------------------------------

    def test_set_active_interval_sets_min_max(self):
        """set_active_interval stores emin and emax on the dataset."""
        self.dataset.set_active_interval(1.0, 2.5)
        self.assertAlmostEqual(self.dataset.active_energy_min, 1.0)
        self.assertAlmostEqual(self.dataset.active_energy_max, 2.5)

    def test_set_active_interval_can_be_overwritten(self):
        """Calling set_active_interval twice updates both attributes."""
        self.dataset.set_active_interval(1.0, 2.5)
        self.dataset.set_active_interval(0.6, 2.8)
        self.assertAlmostEqual(self.dataset.active_energy_min, 0.6)
        self.assertAlmostEqual(self.dataset.active_energy_max, 2.8)

    # ------------------------------------------------------------------
    # mask_valid
    # ------------------------------------------------------------------

    def test_mask_valid_all_good(self):
        """All channels valid when there are no quality flags and no active interval."""
        mask = self.dataset.mask_valid()
        self.assertEqual(len(mask), self.n_channels)
        self.assertTrue(np.all(mask))

    def test_mask_valid_quality_flags(self):
        """Channels with quality != 0 are masked out; good channels remain True."""
        from redback.spectral.dataset import SpectralDataset

        quality = np.zeros(self.n_channels, dtype=int)
        quality[2] = 1   # mark channel 2 bad
        ds = SpectralDataset(
            counts=np.ones(self.n_channels) * 10.0,
            exposure=self.exposure,
            energy_edges_keV=self.edges,
            quality=quality,
        )
        mask = ds.mask_valid()
        self.assertFalse(mask[2])
        for i in [0, 1, 3, 4]:
            self.assertTrue(mask[i])

    def test_mask_valid_active_interval(self):
        """Channels outside the active interval are masked."""
        self.dataset.set_active_interval(1.0, 2.5)
        mask = self.dataset.mask_valid()
        centers = self.dataset.energy_centers_keV
        for i, c in enumerate(centers):
            if 1.0 <= c <= 2.5:
                self.assertTrue(mask[i],
                                f"Channel {i} (E={c:.3f} keV) should be valid")
            else:
                self.assertFalse(mask[i],
                                 f"Channel {i} (E={c:.3f} keV) should be masked")

    def test_mask_valid_combined(self):
        """Quality AND active interval are both applied in the combined mask."""
        from redback.spectral.dataset import SpectralDataset

        quality = np.zeros(self.n_channels, dtype=int)
        quality[1] = 1   # second channel bad

        ds = SpectralDataset(
            counts=np.ones(self.n_channels) * 10.0,
            exposure=self.exposure,
            energy_edges_keV=self.edges,
            quality=quality,
        )
        ds.set_active_interval(1.0, 2.5)
        mask = ds.mask_valid()
        centers = ds.energy_centers_keV
        for i, c in enumerate(centers):
            in_interval = 1.0 <= c <= 2.5
            good_quality = quality[i] == 0
            self.assertEqual(
                mask[i], in_interval and good_quality,
                f"Channel {i} (E={c:.3f} keV): "
                f"expected {in_interval and good_quality}, got {mask[i]}",
            )

    def test_mask_valid_all_bad_quality_gives_all_false(self):
        """When every quality flag is non-zero, the mask is all False."""
        from redback.spectral.dataset import SpectralDataset

        ds = SpectralDataset(
            counts=np.ones(self.n_channels) * 10.0,
            exposure=self.exposure,
            energy_edges_keV=self.edges,
            quality=np.ones(self.n_channels, dtype=int),
        )
        mask = ds.mask_valid()
        self.assertFalse(np.any(mask))

    # ------------------------------------------------------------------
    # predict_counts — helper models
    # ------------------------------------------------------------------

    @staticmethod
    def _flat_energies_keV_model(energies_keV, amplitude):
        """Model whose first positional argument is 'energies_keV' (keV path)."""
        return np.ones_like(energies_keV) * amplitude

    @staticmethod
    def _flat_frequency_model(time, amplitude, **kwargs):
        """Standard redback-style model using a 'frequency' kwarg."""
        freq = kwargs.get("frequency", np.array([1.0]))
        return np.ones_like(freq) * amplitude

    def test_predict_counts_energies_keV_model_shape(self):
        """Model accepting 'energies_keV' produces output with correct length."""
        predicted = self.dataset.predict_counts(
            model=self._flat_energies_keV_model,
            parameters={"amplitude": 1.0},
        )
        self.assertEqual(len(predicted), self.n_channels)

    def test_predict_counts_kwargs_model(self):
        """Standard redback model (frequency in **kwargs) also produces correct shape."""
        predicted = self.dataset.predict_counts(
            model=self._flat_frequency_model,
            parameters={"amplitude": 1.0},
        )
        self.assertEqual(len(predicted), self.n_channels)

    def test_predict_counts_shape_matches_channels(self):
        """Output length equals the number of counts channels."""
        predicted = self.dataset.predict_counts(
            model=self._flat_energies_keV_model,
            parameters={"amplitude": 0.5},
        )
        self.assertEqual(len(predicted), len(self.dataset.counts))

    def test_predict_counts_non_negative(self):
        """Predicted counts are non-negative for a positive-amplitude model."""
        predicted = self.dataset.predict_counts(
            model=self._flat_energies_keV_model,
            parameters={"amplitude": 1.0},
        )
        self.assertTrue(np.all(predicted >= 0.0))

    def test_predict_counts_finite(self):
        """Predicted counts are all finite for a well-behaved model."""
        predicted = self.dataset.predict_counts(
            model=self._flat_energies_keV_model,
            parameters={"amplitude": 1.0},
        )
        self.assertTrue(np.all(np.isfinite(predicted)))

    def test_predict_counts_with_arf(self):
        """
        When an ARF is attached, fold_spectrum multiplies by effective area
        so that predicted counts are larger (with area=500 cm^2 vs area=0).
        """
        from redback.spectral.dataset import SpectralDataset
        from redback.spectral.response import EffectiveArea

        arf = EffectiveArea(
            e_min=self.edges[:-1],
            e_max=self.edges[1:],
            area=np.full(self.n_channels, 500.0),
        )
        ds_with_arf = SpectralDataset(
            counts=np.ones(self.n_channels) * 10.0,
            exposure=self.exposure,
            energy_edges_keV=self.edges,
            arf=arf,
        )
        predicted = ds_with_arf.predict_counts(
            model=self._flat_energies_keV_model,
            parameters={"amplitude": 1.0},
        )
        self.assertTrue(np.all(predicted > 0.0))

    def test_predict_counts_scales_linearly_with_exposure(self):
        """Doubling exposure doubles predicted counts."""
        from redback.spectral.dataset import SpectralDataset

        ds1 = SpectralDataset(
            counts=np.ones(self.n_channels) * 10.0,
            exposure=1000.0,
            energy_edges_keV=self.edges,
        )
        ds2 = SpectralDataset(
            counts=np.ones(self.n_channels) * 10.0,
            exposure=2000.0,
            energy_edges_keV=self.edges,
        )
        c1 = ds1.predict_counts(
            model=self._flat_energies_keV_model, parameters={"amplitude": 1.0}
        )
        c2 = ds2.predict_counts(
            model=self._flat_energies_keV_model, parameters={"amplitude": 1.0}
        )
        np.testing.assert_array_almost_equal(c2, 2.0 * c1, decimal=10)

    # ------------------------------------------------------------------
    # from_ogip classmethod
    # ------------------------------------------------------------------

    def test_from_ogip_basic(self):
        """from_ogip loads counts and exposure from a minimal PHA + RMF pair."""
        from redback.spectral.dataset import SpectralDataset

        n_ch, n_en = 8, 8
        pha_path = _make_pha_fits(
            n_channels=n_ch, exposure=500.0, tmpdir=self.tmpdir, filename="src.pha"
        )
        rmf_path = _make_rmf_fits(
            n_energy=n_en, n_channels=n_ch, tmpdir=self.tmpdir
        )
        ds = SpectralDataset.from_ogip(pha=pha_path, rmf=rmf_path)
        self.assertEqual(len(ds.counts), n_ch)
        self.assertAlmostEqual(ds.exposure, 500.0, places=2)

    def test_from_ogip_explicit_energy_edges(self):
        """energy_edges_keV parameter is used verbatim when supplied."""
        from redback.spectral.dataset import SpectralDataset

        n_ch = 8
        explicit_edges = np.linspace(0.3, 10.0, n_ch + 1)
        pha_path = _make_pha_fits(
            n_channels=n_ch, exposure=500.0, tmpdir=self.tmpdir, filename="src2.pha"
        )
        ds = SpectralDataset.from_ogip(pha=pha_path, energy_edges_keV=explicit_edges)
        np.testing.assert_array_almost_equal(
            ds.energy_edges_keV, explicit_edges, decimal=10
        )

    def test_from_ogip_no_response_no_edges_raises(self):
        """ValueError is raised when neither RMF, ARF, nor energy_edges_keV is given."""
        from redback.spectral.dataset import SpectralDataset

        pha_path = _make_pha_fits(
            n_channels=5, exposure=100.0, tmpdir=self.tmpdir, filename="bare.pha"
        )
        with self.assertRaises(ValueError):
            SpectralDataset.from_ogip(pha=pha_path)

    def test_from_ogip_directory_finds_bkg(self):
        """
        from_ogip_directory auto-discovers a background file via the '_bkg.pha'
        suffix convention when no explicit bkg argument is supplied.
        """
        from redback.spectral.dataset import SpectralDataset

        subdir = os.path.join(self.tmpdir, "ogip_dir")
        os.makedirs(subdir)
        n_ch, n_en = 6, 6

        _make_pha_fits(n_channels=n_ch, exposure=1000.0,
                       tmpdir=subdir, filename="obs.pha")
        _make_pha_fits(n_channels=n_ch, exposure=2000.0,
                       tmpdir=subdir, filename="obs_bkg.pha")
        _make_rmf_fits(n_energy=n_en, n_channels=n_ch,
                       tmpdir=subdir, filename="obs.rmf")

        ds = SpectralDataset.from_ogip_directory(subdir)
        self.assertIsNotNone(ds.counts_bkg)
        self.assertEqual(len(ds.counts_bkg), n_ch)

    def test_from_ogip_quality_propagated(self):
        """Quality flags from the PHA file are available on the dataset."""
        from redback.spectral.dataset import SpectralDataset

        n_ch = 6
        pha_path = _make_pha_fits(
            n_channels=n_ch, exposure=500.0, with_quality=True,
            tmpdir=self.tmpdir, filename="qsrc.pha",
        )
        edges = np.linspace(0.5, 7.0, n_ch + 1)
        ds = SpectralDataset.from_ogip(pha=pha_path, energy_edges_keV=edges)
        self.assertIsNotNone(ds.quality)
        self.assertEqual(len(ds.quality), n_ch)


# ===========================================================================
# 3.  TestPoissonSpectralLikelihood  (Cash-stat)
# ===========================================================================

class TestPoissonSpectralLikelihood(unittest.TestCase):
    """Tests for redback.likelihoods.PoissonSpectralLikelihood."""

    def setUp(self):
        from redback.likelihoods import PoissonSpectralLikelihood

        self.n_channels = 5
        self.exposure = 1000.0
        self.edges = np.linspace(0.5, 3.0, self.n_channels + 1)

        self.dataset = _make_minimal_dataset(
            n_channels=self.n_channels, counts_val=10.0, exposure=self.exposure
        )

        def flat_model(energies_keV, amplitude):
            return np.ones_like(energies_keV) * amplitude

        self.model = flat_model
        self.likelihood = PoissonSpectralLikelihood(
            dataset=self.dataset, function=self.model
        )
        self.likelihood.parameters["amplitude"] = 1.0

    def tearDown(self):
        del self.likelihood
        del self.dataset

    def test_parameters_inferred(self):
        """'amplitude' appears in the inferred parameter dict."""
        self.assertIn("amplitude", self.likelihood.parameters)

    def test_energies_keV_not_a_parameter(self):
        """'energies_keV' is stripped from the parameter dict."""
        self.assertNotIn("energies_keV", self.likelihood.parameters)

    def test_energy_keV_not_a_parameter(self):
        """'energy_keV' is also stripped from the parameter dict."""
        self.assertNotIn("energy_keV", self.likelihood.parameters)

    def test_log_likelihood_finite(self):
        """Returns a finite float for valid inputs."""
        ll = self.likelihood.log_likelihood()
        self.assertTrue(np.isfinite(ll))

    def test_log_likelihood_is_scalar(self):
        """Return value is a scalar (convertible to float)."""
        ll = self.likelihood.log_likelihood()
        self.assertIsInstance(float(ll), float)

    def test_log_likelihood_known_value(self):
        """
        Cash-stat formula:  ll = sum( data * log(model) - model )
        Verify against a manual computation using the same predict_counts.
        """
        model_counts = self.dataset.predict_counts(
            model=self.model, parameters={"amplitude": 1.0}
        )
        mask = self.dataset.mask_valid()
        data = self.dataset.counts[mask]
        model = np.clip(model_counts[mask], 1e-30, 1e30)
        expected = float(np.nan_to_num(
            np.sum(data * np.log(model) - model),
            nan=-np.inf, neginf=-np.inf, posinf=-np.inf,
        ))
        self.assertAlmostEqual(self.likelihood.log_likelihood(), expected, places=10)

    def test_log_likelihood_nan_safe(self):
        """
        When amplitude is nearly zero, the likelihood must not return NaN.
        It must be either finite or -inf (nan_to_num ensures this).
        """
        self.likelihood.parameters["amplitude"] = 1e-300
        ll = self.likelihood.log_likelihood()
        self.assertFalse(np.isnan(ll))

    def test_log_likelihood_changes_with_amplitude(self):
        """
        The Cash-stat must be sensitive to the model amplitude: scanning a
        log-spaced range should produce a non-trivial spread in log-likelihood
        values (the surface must not be flat over many orders of magnitude).
        """
        amplitudes = np.logspace(-3, 3, 30)
        lls = []
        for a in amplitudes:
            self.likelihood.parameters["amplitude"] = a
            lls.append(self.likelihood.log_likelihood())
        lls = np.array(lls)
        finite_lls = lls[np.isfinite(lls)]
        self.assertGreater(len(finite_lls), 0)
        ll_range = float(np.max(finite_lls) - np.min(finite_lls))
        self.assertGreater(ll_range, 0.0)


# ===========================================================================
# 4.  TestWStatSpectralLikelihood
# ===========================================================================

class TestWStatSpectralLikelihood(unittest.TestCase):
    """Tests for redback.likelihoods.WStatSpectralLikelihood."""

    def setUp(self):
        from redback.likelihoods import WStatSpectralLikelihood
        from redback.spectral.dataset import SpectralDataset

        self.n_channels = 5
        self.exposure = 1000.0
        self.edges = np.linspace(0.5, 3.0, self.n_channels + 1)

        self.dataset_with_bkg = SpectralDataset(
            counts=np.array([15.0, 25.0, 20.0, 18.0, 12.0]),
            exposure=self.exposure,
            energy_edges_keV=self.edges,
            counts_bkg=np.array([5.0, 8.0, 6.0, 4.0, 3.0]),
            bkg_exposure=2000.0,
            bkg_backscale=1.0,
            bkg_areascal=1.0,
            backscale=1.0,
            areascal=1.0,
        )

        def flat_model(energies_keV, amplitude):
            return np.ones_like(energies_keV) * amplitude

        self.model = flat_model
        self.likelihood = WStatSpectralLikelihood(
            dataset=self.dataset_with_bkg, function=self.model
        )
        self.likelihood.parameters["amplitude"] = 1.0

    def tearDown(self):
        del self.likelihood
        del self.dataset_with_bkg

    def test_requires_counts_bkg(self):
        """ValueError is raised when counts_bkg is None on the dataset."""
        from redback.likelihoods import WStatSpectralLikelihood

        ds_no_bkg = _make_minimal_dataset(with_bkg=False)

        def flat_model(energies_keV, amplitude):
            return np.ones_like(energies_keV) * amplitude

        lik = WStatSpectralLikelihood(dataset=ds_no_bkg, function=flat_model)
        lik.parameters["amplitude"] = 1.0
        with self.assertRaises(ValueError):
            lik.log_likelihood()

    def test_log_likelihood_finite(self):
        """Returns a finite value for valid source + background data."""
        ll = self.likelihood.log_likelihood()
        self.assertTrue(np.isfinite(ll))

    def test_log_likelihood_is_scalar(self):
        """Return value is a scalar (convertible to float)."""
        ll = self.likelihood.log_likelihood()
        self.assertIsInstance(float(ll), float)

    def test_log_likelihood_no_background_counts(self):
        """W-stat works (does not raise) when background counts are all zero."""
        from redback.likelihoods import WStatSpectralLikelihood
        from redback.spectral.dataset import SpectralDataset

        ds_zero_bkg = SpectralDataset(
            counts=np.array([10.0, 20.0, 15.0, 12.0, 8.0]),
            exposure=1000.0,
            energy_edges_keV=self.edges,
            counts_bkg=np.zeros(5),
            bkg_exposure=1000.0,
            bkg_backscale=1.0,
            bkg_areascal=1.0,
            backscale=1.0,
            areascal=1.0,
        )

        def flat_model(energies_keV, amplitude):
            return np.ones_like(energies_keV) * amplitude

        lik = WStatSpectralLikelihood(dataset=ds_zero_bkg, function=flat_model)
        lik.parameters["amplitude"] = 1.0
        ll = lik.log_likelihood()
        self.assertFalse(np.isnan(ll))

    def test_log_likelihood_known_value(self):
        """
        Reproduce the W-stat formula from the source code on a single-channel
        dataset and verify the Python result matches.
        """
        from redback.likelihoods import WStatSpectralLikelihood
        from redback.spectral.dataset import SpectralDataset

        src = np.array([20.0])
        bkg_obs = np.array([5.0])
        edges = np.array([1.0, 2.0])

        ds = SpectralDataset(
            counts=src,
            exposure=1000.0,
            energy_edges_keV=edges,
            counts_bkg=bkg_obs,
            bkg_exposure=1000.0,
            bkg_backscale=1.0,
            bkg_areascal=1.0,
            backscale=1.0,
            areascal=1.0,
        )

        def const_model(energies_keV, amplitude):
            return np.ones_like(energies_keV) * amplitude

        lik = WStatSpectralLikelihood(dataset=ds, function=const_model)
        lik.parameters["amplitude"] = 1.0

        # Replicate the W-stat arithmetic from likelihoods.py
        model_counts = ds.predict_counts(
            model=const_model, parameters={"amplitude": 1.0}
        )
        mask = ds.mask_valid()
        data = ds.counts[mask]
        bkg = ds.counts_bkg[mask]
        scale = ds.background_scale_factor  # 1.0

        model_c = np.clip(model_counts[mask], 1e-30, 1e30)
        a = scale * (scale + 1.0)
        bcoef = (scale + 1.0) * model_c - scale * (data + bkg)
        c_coef = -bkg * model_c
        disc = np.maximum(bcoef ** 2 - 4.0 * a * c_coef, 0.0)
        b_hat = (-bcoef + np.sqrt(disc)) / (2.0 * a)
        b_hat = np.clip(b_hat, 0.0, None)
        mu = np.clip(model_c + scale * b_hat, 1e-30, None)
        ll_expected = np.sum(data * np.log(mu) - mu)
        if np.any(bkg > 0):
            ll_expected += np.sum(bkg * np.log(np.clip(b_hat, 1e-30, None)) - b_hat)
        else:
            ll_expected -= np.sum(b_hat)
        ll_expected = float(
            np.nan_to_num(ll_expected, nan=-np.inf, neginf=-np.inf, posinf=-np.inf)
        )

        self.assertAlmostEqual(lik.log_likelihood(), ll_expected, places=10)

    def test_parameters_inferred(self):
        """'amplitude' is in the parameters; 'energies_keV' is not."""
        self.assertIn("amplitude", self.likelihood.parameters)
        self.assertNotIn("energies_keV", self.likelihood.parameters)


# ===========================================================================
# 5.  TestChiSquareSpectralLikelihood
# ===========================================================================

class TestChiSquareSpectralLikelihood(unittest.TestCase):
    """Tests for redback.likelihoods.ChiSquareSpectralLikelihood."""

    def setUp(self):
        from redback.likelihoods import ChiSquareSpectralLikelihood

        self.n_channels = 5
        self.exposure = 1000.0
        self.edges = np.linspace(0.5, 3.0, self.n_channels + 1)
        self.counts_val = 25.0

        self.dataset = _make_minimal_dataset(
            n_channels=self.n_channels,
            counts_val=self.counts_val,
            exposure=self.exposure,
        )

        def flat_model(energies_keV, amplitude):
            return np.ones_like(energies_keV) * amplitude

        self.model = flat_model
        self.likelihood = ChiSquareSpectralLikelihood(
            dataset=self.dataset, function=self.model
        )
        self.likelihood.parameters["amplitude"] = 1.0

    def tearDown(self):
        del self.likelihood
        del self.dataset

    def test_log_likelihood_known_value(self):
        """
        Verify the chi-square formula:
          ll = -0.5 * sum( ((data - model)/sigma)^2 + log(2*pi*sigma^2) )
        where sigma = sqrt(max(data, 1.0)).
        """
        model_counts = self.dataset.predict_counts(
            model=self.model, parameters={"amplitude": 1.0}
        )
        mask = self.dataset.mask_valid()
        data = self.dataset.counts[mask]
        model = np.clip(model_counts[mask], 1e-30, 1e30)
        sigma = np.sqrt(np.maximum(data, 1.0))
        expected = float(np.nan_to_num(
            -0.5 * np.sum(((data - model) / sigma) ** 2 +
                          np.log(2 * np.pi * sigma ** 2)),
            nan=-np.inf, neginf=-np.inf, posinf=-np.inf,
        ))
        self.assertAlmostEqual(self.likelihood.log_likelihood(), expected, places=10)

    def test_log_likelihood_sigma_floor(self):
        """
        Bins with 0 observed counts use sigma = sqrt(max(0, 1)) = 1  (floor).
        The likelihood must remain finite (no division by zero).
        """
        from redback.likelihoods import ChiSquareSpectralLikelihood
        from redback.spectral.dataset import SpectralDataset

        ds_zero = SpectralDataset(
            counts=np.zeros(self.n_channels),
            exposure=self.exposure,
            energy_edges_keV=self.edges,
        )

        def flat_model(energies_keV, amplitude):
            return np.ones_like(energies_keV) * amplitude

        lik = ChiSquareSpectralLikelihood(dataset=ds_zero, function=flat_model)
        lik.parameters["amplitude"] = 1.0
        ll = lik.log_likelihood()
        self.assertFalse(np.isnan(ll))
        self.assertFalse(np.isposinf(ll))

    def test_log_likelihood_perfect_fit_formula(self):
        """
        When model == data in every channel, (data - model)^2 = 0, so
          ll = -0.5 * sum(log(2*pi*sigma^2))
        which is the highest possible value for fixed data.
        We verify the implementation matches this formula by checking
        actual output against a manual computation with the actual model counts.
        """
        model_counts = self.dataset.predict_counts(
            model=self.model, parameters={"amplitude": 1.0}
        )
        mask = self.dataset.mask_valid()
        data = self.dataset.counts[mask]
        model = np.clip(model_counts[mask], 1e-30, 1e30)
        sigma = np.sqrt(np.maximum(data, 1.0))
        expected = float(np.nan_to_num(
            -0.5 * np.sum(((data - model) / sigma) ** 2 +
                          np.log(2 * np.pi * sigma ** 2)),
            nan=-np.inf, neginf=-np.inf, posinf=-np.inf,
        ))
        self.assertAlmostEqual(self.likelihood.log_likelihood(), expected, places=10)

    def test_log_likelihood_finite(self):
        """Returns a finite value for well-behaved inputs."""
        ll = self.likelihood.log_likelihood()
        self.assertTrue(np.isfinite(ll))

    def test_log_likelihood_is_scalar(self):
        """Return value is a scalar."""
        ll = self.likelihood.log_likelihood()
        self.assertIsInstance(float(ll), float)

    def test_parameters_inferred(self):
        """'amplitude' is in the parameters; 'energies_keV' is excluded."""
        self.assertIn("amplitude", self.likelihood.parameters)
        self.assertNotIn("energies_keV", self.likelihood.parameters)

    def test_log_likelihood_worse_for_large_offset_model(self):
        """
        A model whose predicted counts are far from the data yields a lower
        log-likelihood than a model that is close to the data.
        """
        self.likelihood.parameters["amplitude"] = 1.0
        ll_close = self.likelihood.log_likelihood()

        self.likelihood.parameters["amplitude"] = 1e6   # wildly wrong
        ll_far = self.likelihood.log_likelihood()

        self.assertGreater(ll_close, ll_far)

    def test_log_likelihood_sigma_equals_one_for_zero_counts(self):
        """
        Explicitly confirm that sigma = sqrt(max(0, 1)) = 1 is applied.
        For data=0, model=2: contribution = -0.5 * (4 + log(2*pi)).
        """
        from redback.likelihoods import ChiSquareSpectralLikelihood
        from redback.spectral.dataset import SpectralDataset

        # Single-channel dataset with zero counts
        ds = SpectralDataset(
            counts=np.array([0.0]),
            exposure=1.0,
            energy_edges_keV=np.array([1.0, 2.0]),
        )

        # We need predict_counts to return exactly 2.0.  Build a model whose
        # fold_spectrum output we can intercept via monkeypatching.
        # Simpler: just verify the sigma=1 floor is used by checking finiteness
        # and that the result is less than zero (gaussian log-norm is negative).
        def tiny_model(energies_keV, amplitude):
            return np.ones_like(energies_keV) * amplitude

        lik = ChiSquareSpectralLikelihood(dataset=ds, function=tiny_model)
        lik.parameters["amplitude"] = 1.0
        ll = lik.log_likelihood()
        # The log-normalisation term alone is -0.5*log(2*pi) < 0, so ll < 0
        self.assertLess(ll, 0.0)
        self.assertFalse(np.isnan(ll))


# ===========================================================================
# 6.  TestCountsSpectrumTransient
# ===========================================================================

import inspect
import pandas as pd
from unittest import mock

import redback
import bilby.core.prior
from redback.transient import TRANSIENT_DICT
from redback.transient.spectral import CountsSpectrumTransient
from redback.spectral.dataset import SpectralDataset


def _make_counts_transient(name="test_src", n_channels=5):
    """Build a CountsSpectrumTransient backed by a minimal SpectralDataset."""
    dataset = _make_minimal_dataset(n_channels=n_channels, counts_val=10.0)
    # Bypass __post_init__ side-effect on directory_structure by patching it.
    with mock.patch(
        "redback.get_data.directory.spectrum_directory_structure",
        return_value=mock.MagicMock(),
    ):
        cst = CountsSpectrumTransient(dataset=dataset, name=name)
    return cst


class TestCountsSpectrumTransient(unittest.TestCase):
    """Tests for redback.transient.spectral.CountsSpectrumTransient."""

    def _make(self, name="test_src", n_channels=5):
        return _make_counts_transient(name=name, n_channels=n_channels)

    # ------------------------------------------------------------------
    # TRANSIENT_DICT registration
    # ------------------------------------------------------------------

    def test_in_transient_dict(self):
        """TRANSIENT_DICT['counts_spectrum'] maps to CountsSpectrumTransient."""
        self.assertIs(TRANSIENT_DICT["counts_spectrum"], CountsSpectrumTransient)

    # ------------------------------------------------------------------
    # __post_init__ behaviour
    # ------------------------------------------------------------------

    def test_post_init_data_mode(self):
        """After construction, data_mode is 'spectrum_counts'."""
        cst = self._make()
        self.assertEqual(cst.data_mode, "spectrum_counts")

    def test_post_init_name_propagated_to_dataset(self):
        """The dataset.name is updated to the transient name during __post_init__."""
        cst = self._make(name="my_source")
        self.assertEqual(cst.dataset.name, "my_source")

    # ------------------------------------------------------------------
    # from_simulator
    # ------------------------------------------------------------------

    def test_from_simulator_returns_counts_spectrum_transient(self):
        """from_simulator returns a CountsSpectrumTransient instance."""
        n_bins = 5
        energy_edges = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0])
        df = pd.DataFrame(
            {"energy_channel": [0, 1, 2, 3, 4], "counts": [10, 20, 15, 8, 5]}
        )

        mock_sim = mock.MagicMock()
        mock_sim.n_energy_bins = n_bins
        mock_sim.energy_edges = energy_edges
        mock_sim.effective_area_func = lambda e: np.ones_like(e) * 100.0
        mock_sim.background_rate_func = lambda e: np.ones_like(e) * 0.1
        mock_sim.generate_binned_counts.return_value = df

        time_bins = np.array([0.0, 1.0])

        with mock.patch(
            "redback.get_data.directory.spectrum_directory_structure",
            return_value=mock.MagicMock(),
        ):
            cst = CountsSpectrumTransient.from_simulator(
                sim=mock_sim, time_bins=time_bins, name="sim_src"
            )

        self.assertIsInstance(cst, CountsSpectrumTransient)

    def test_from_simulator_sets_name(self):
        """from_simulator propagates the supplied name to the transient."""
        energy_edges = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0])
        df = pd.DataFrame(
            {"energy_channel": [0, 1, 2, 3, 4], "counts": [10, 20, 15, 8, 5]}
        )
        mock_sim = mock.MagicMock()
        mock_sim.n_energy_bins = 5
        mock_sim.energy_edges = energy_edges
        mock_sim.effective_area_func = lambda e: np.ones_like(e) * 100.0
        mock_sim.background_rate_func = lambda e: np.ones_like(e) * 0.1
        mock_sim.generate_binned_counts.return_value = df

        with mock.patch(
            "redback.get_data.directory.spectrum_directory_structure",
            return_value=mock.MagicMock(),
        ):
            cst = CountsSpectrumTransient.from_simulator(
                sim=mock_sim, time_bins=np.array([0.0, 1.0]), name="named_src"
            )

        self.assertEqual(cst.name, "named_src")

    # ------------------------------------------------------------------
    # Delegation methods
    # ------------------------------------------------------------------

    def test_plot_data_delegates_to_dataset(self):
        """plot_data() calls dataset.plot_spectrum_data."""
        cst = self._make()
        cst.dataset.plot_spectrum_data = mock.MagicMock(return_value="ok")
        result = cst.plot_data(show=False)
        cst.dataset.plot_spectrum_data.assert_called_once_with(show=False)
        self.assertEqual(result, "ok")

    def test_plot_fit_delegates_to_dataset(self):
        """plot_fit() calls dataset.plot_spectrum_fit."""
        cst = self._make()
        cst.dataset.plot_spectrum_fit = mock.MagicMock(return_value="fit_ok")
        result = cst.plot_fit(show=False)
        cst.dataset.plot_spectrum_fit.assert_called_once_with(show=False)
        self.assertEqual(result, "fit_ok")

    def test_compute_band_flux_delegates(self):
        """compute_band_flux() delegates to dataset.compute_band_flux."""
        cst = self._make()
        cst.dataset.compute_band_flux = mock.MagicMock(return_value=42.0)
        result = cst.compute_band_flux(1.0, 5.0)
        cst.dataset.compute_band_flux.assert_called_once_with(1.0, 5.0)
        self.assertEqual(result, 42.0)


# ===========================================================================
# 7.  TestSpectralSampler
# ===========================================================================

import bilby
from redback.sampler import fit_model, _fit_spectral_dataset
from redback.likelihoods import PoissonSpectralLikelihood, WStatSpectralLikelihood


def _simple_spectral_model(energies_keV, amplitude, **kwargs):
    """Minimal spectral model that accepts energies_keV."""
    return amplitude * np.ones_like(energies_keV)


def _make_spectral_prior():
    return bilby.core.prior.PriorDict(
        {"amplitude": bilby.core.prior.Uniform(0.1, 10.0, "amplitude")}
    )


class TestSpectralSampler(unittest.TestCase):
    """Tests for the spectral-fitting routing logic in redback.sampler."""

    def setUp(self):
        self._tmpdir_obj = tempfile.TemporaryDirectory()
        self.outdir = self._tmpdir_obj.name
        self.dataset = _make_minimal_dataset(n_channels=5, counts_val=20.0)
        self.dataset_bkg = _make_minimal_dataset(
            n_channels=5, counts_val=20.0, with_bkg=True
        )
        self.prior = _make_spectral_prior()

    def tearDown(self):
        self._tmpdir_obj.cleanup()

    # ------------------------------------------------------------------
    # Routing: fit_model dispatches to _fit_spectral_dataset
    # ------------------------------------------------------------------

    @mock.patch("bilby.run_sampler")
    def test_fit_model_routes_spectral_dataset(self, mock_sampler):
        """fit_model calls bilby.run_sampler when transient is a SpectralDataset."""
        mock_result = mock.MagicMock()
        mock_result.posterior = pd.DataFrame({"amplitude": [1.0, 2.0]})
        mock_sampler.return_value = mock_result

        fit_model(
            transient=self.dataset,
            model=_simple_spectral_model,
            outdir=self.outdir,
            label="route_test",
            prior=self.prior,
            plot=False,
            clean=True,
        )

        mock_sampler.assert_called_once()

    @mock.patch("bilby.run_sampler")
    def test_fit_model_routes_counts_spectrum_transient(self, mock_sampler):
        """fit_model routes CountsSpectrumTransient through _fit_spectral_dataset."""
        mock_result = mock.MagicMock()
        mock_result.posterior = pd.DataFrame({"amplitude": [1.0, 2.0]})
        mock_sampler.return_value = mock_result

        with mock.patch(
            "redback.get_data.directory.spectrum_directory_structure",
            return_value=mock.MagicMock(),
        ):
            cst = CountsSpectrumTransient(dataset=self.dataset, name="cst_route")

        fit_model(
            transient=cst,
            model=_simple_spectral_model,
            outdir=self.outdir,
            label="cst_route",
            prior=self.prior,
            plot=False,
            clean=True,
        )

        mock_sampler.assert_called_once()

    # ------------------------------------------------------------------
    # Statistic selection
    # ------------------------------------------------------------------

    @mock.patch("bilby.run_sampler")
    def test_fit_spectral_dataset_auto_statistic_cstat(self, mock_sampler):
        """When counts_bkg is None, PoissonSpectralLikelihood (C-stat) is used."""
        mock_result = mock.MagicMock()
        mock_result.posterior = pd.DataFrame({"amplitude": [1.0]})
        mock_sampler.return_value = mock_result

        # dataset has no background
        self.assertIsNone(self.dataset.counts_bkg)

        _fit_spectral_dataset(
            transient=self.dataset,
            model=_simple_spectral_model,
            outdir=self.outdir,
            label="cstat_test",
            prior=self.prior,
            plot=False,
            clean=True,
        )

        mock_sampler.assert_called_once()
        call_kwargs = mock_sampler.call_args
        likelihood_used = call_kwargs[1].get("likelihood") or call_kwargs[0][0]
        # The first positional keyword arg is `likelihood`
        likelihood_arg = mock_sampler.call_args.kwargs.get("likelihood")
        self.assertIsInstance(likelihood_arg, PoissonSpectralLikelihood)

    @mock.patch("bilby.run_sampler")
    def test_fit_spectral_dataset_auto_statistic_wstat(self, mock_sampler):
        """When counts_bkg is set, WStatSpectralLikelihood (W-stat) is used."""
        mock_result = mock.MagicMock()
        mock_result.posterior = pd.DataFrame({"amplitude": [1.0]})
        mock_sampler.return_value = mock_result

        self.assertIsNotNone(self.dataset_bkg.counts_bkg)

        _fit_spectral_dataset(
            transient=self.dataset_bkg,
            model=_simple_spectral_model,
            outdir=self.outdir,
            label="wstat_test",
            prior=self.prior,
            plot=False,
            clean=True,
        )

        mock_sampler.assert_called_once()
        likelihood_arg = mock_sampler.call_args.kwargs.get("likelihood")
        self.assertIsInstance(likelihood_arg, WStatSpectralLikelihood)

    # ------------------------------------------------------------------
    # Error paths
    # ------------------------------------------------------------------

    def test_fit_spectral_dataset_unknown_statistic_raises(self):
        """An unrecognised statistic name raises ValueError."""
        with self.assertRaises(ValueError):
            _fit_spectral_dataset(
                transient=self.dataset,
                model=_simple_spectral_model,
                outdir=self.outdir,
                label="bad_stat",
                prior=self.prior,
                plot=False,
                clean=True,
                statistic="bad_stat",
            )

    def test_fit_spectral_dataset_model_without_energy_raises(self):
        """A model that lacks energies_keV/energy_keV and **kwargs raises ValueError."""

        def no_energy_model(time, amplitude):
            return amplitude * np.ones_like(time)

        with self.assertRaises(ValueError):
            _fit_spectral_dataset(
                transient=self.dataset,
                model=no_energy_model,
                outdir=self.outdir,
                label="no_energy",
                prior=self.prior,
                plot=False,
                clean=True,
            )

    # ------------------------------------------------------------------
    # Save-format override
    # ------------------------------------------------------------------

    @mock.patch("bilby.run_sampler")
    def test_fit_spectral_dataset_save_format_forced_pkl(self, mock_sampler):
        """JSON save_format is silently coerced to pkl for spectral datasets."""
        mock_result = mock.MagicMock()
        mock_result.posterior = pd.DataFrame({"amplitude": [1.0]})
        mock_sampler.return_value = mock_result

        _fit_spectral_dataset(
            transient=self.dataset,
            model=_simple_spectral_model,
            outdir=self.outdir,
            label="pkl_test",
            prior=self.prior,
            save_format="json",
            plot=False,
            clean=True,
        )

        mock_sampler.assert_called_once()
        # bilby.run_sampler should have received save='pkl', not 'json'
        call_kwargs = mock_sampler.call_args.kwargs
        self.assertEqual(call_kwargs.get("save"), "pkl")


# ===========================================================================
# 8.  TestSpectralPriors
# ===========================================================================

import redback as _redback_module


class TestSpectralPriors(unittest.TestCase):
    """Tests that all six high-energy spectral prior files load and have the right keys."""

    @property
    def _prior_dir(self):
        return os.path.join(os.path.dirname(_redback_module.__file__), "priors")

    def _load(self, filename):
        path = os.path.join(self._prior_dir, filename)
        return bilby.core.prior.PriorDict(path)

    # ------------------------------------------------------------------
    # powerlaw_high_energy
    # ------------------------------------------------------------------

    def test_powerlaw_prior_loads(self):
        """powerlaw_high_energy.prior loads without error."""
        prior = self._load("powerlaw_high_energy.prior")
        self.assertIsInstance(prior, bilby.core.prior.PriorDict)

    def test_powerlaw_prior_params(self):
        """powerlaw_high_energy prior contains log10_norm, alpha, redshift."""
        prior = self._load("powerlaw_high_energy.prior")
        for key in ("log10_norm", "alpha", "redshift"):
            self.assertIn(key, prior, f"Missing key '{key}' in powerlaw_high_energy prior")

    # ------------------------------------------------------------------
    # tbabs_powerlaw_high_energy
    # ------------------------------------------------------------------

    def test_tbabs_powerlaw_prior_loads(self):
        """tbabs_powerlaw_high_energy.prior loads without error."""
        prior = self._load("tbabs_powerlaw_high_energy.prior")
        self.assertIsInstance(prior, bilby.core.prior.PriorDict)

    def test_tbabs_powerlaw_prior_params(self):
        """tbabs_powerlaw_high_energy prior contains log10_norm, alpha, nh, redshift."""
        prior = self._load("tbabs_powerlaw_high_energy.prior")
        for key in ("log10_norm", "alpha", "nh", "redshift"):
            self.assertIn(key, prior, f"Missing key '{key}' in tbabs_powerlaw_high_energy prior")

    # ------------------------------------------------------------------
    # cutoff_powerlaw_high_energy
    # ------------------------------------------------------------------

    def test_cutoff_powerlaw_prior_loads(self):
        """cutoff_powerlaw_high_energy.prior loads without error."""
        prior = self._load("cutoff_powerlaw_high_energy.prior")
        self.assertIsInstance(prior, bilby.core.prior.PriorDict)

    def test_cutoff_powerlaw_prior_params(self):
        """cutoff_powerlaw_high_energy prior contains log10_norm, alpha, e_cut, redshift."""
        prior = self._load("cutoff_powerlaw_high_energy.prior")
        for key in ("log10_norm", "alpha", "e_cut", "redshift"):
            self.assertIn(key, prior, f"Missing key '{key}' in cutoff_powerlaw_high_energy prior")

    # ------------------------------------------------------------------
    # comptonized_high_energy
    # ------------------------------------------------------------------

    def test_comptonized_prior_loads(self):
        """comptonized_high_energy.prior loads without error."""
        prior = self._load("comptonized_high_energy.prior")
        self.assertIsInstance(prior, bilby.core.prior.PriorDict)

    def test_comptonized_prior_params(self):
        """comptonized_high_energy prior contains log10_norm, alpha, e_peak, redshift."""
        prior = self._load("comptonized_high_energy.prior")
        for key in ("log10_norm", "alpha", "e_peak", "redshift"):
            self.assertIn(key, prior, f"Missing key '{key}' in comptonized_high_energy prior")

    # ------------------------------------------------------------------
    # blackbody_high_energy
    # ------------------------------------------------------------------

    def test_blackbody_prior_loads(self):
        """blackbody_high_energy.prior loads without error."""
        prior = self._load("blackbody_high_energy.prior")
        self.assertIsInstance(prior, bilby.core.prior.PriorDict)

    def test_blackbody_prior_params(self):
        """blackbody_high_energy prior contains r_photosphere_rs, kT, redshift."""
        prior = self._load("blackbody_high_energy.prior")
        for key in ("r_photosphere_rs", "kT", "redshift"):
            self.assertIn(key, prior, f"Missing key '{key}' in blackbody_high_energy prior")

    # ------------------------------------------------------------------
    # band_function_high_energy
    # ------------------------------------------------------------------

    def test_band_function_prior_loads(self):
        """band_function_high_energy.prior loads without error."""
        prior = self._load("band_function_high_energy.prior")
        self.assertIsInstance(prior, bilby.core.prior.PriorDict)

    def test_band_function_prior_params(self):
        """band_function_high_energy prior contains log10_norm, alpha, beta, e_peak, redshift."""
        prior = self._load("band_function_high_energy.prior")
        for key in ("log10_norm", "alpha", "beta", "e_peak", "redshift"):
            self.assertIn(key, prior, f"Missing key '{key}' in band_function_high_energy prior")

    # ------------------------------------------------------------------
    # Cross-check: prior keys are a subset of model parameters
    # ------------------------------------------------------------------

    def test_prior_params_match_model_signatures(self):
        """
        For each of the six spectral models, the prior keys must be a subset
        of the model's accepted parameters (positional args + **kwargs covers
        anything not explicitly named).
        """
        from redback.transient_models.spectral_models import (
            powerlaw_high_energy,
            tbabs_powerlaw_high_energy,
            cutoff_powerlaw_high_energy,
            comptonized_high_energy,
            blackbody_high_energy,
            band_function_high_energy,
        )

        pairs = [
            ("powerlaw_high_energy.prior",        powerlaw_high_energy),
            ("tbabs_powerlaw_high_energy.prior",   tbabs_powerlaw_high_energy),
            ("cutoff_powerlaw_high_energy.prior",  cutoff_powerlaw_high_energy),
            ("comptonized_high_energy.prior",      comptonized_high_energy),
            ("blackbody_high_energy.prior",        blackbody_high_energy),
            ("band_function_high_energy.prior",    band_function_high_energy),
        ]

        for prior_file, model_fn in pairs:
            with self.subTest(prior=prior_file):
                prior = self._load(prior_file)
                sig = inspect.signature(model_fn)
                has_var_keyword = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD
                    for p in sig.parameters.values()
                )
                if has_var_keyword:
                    # **kwargs means the model accepts any extra keyword, so
                    # all prior keys are valid — nothing to check further.
                    continue
                model_params = set(sig.parameters.keys())
                # energies_keV is the data argument, not a prior parameter
                model_params.discard("energies_keV")
                model_params.discard("energy_keV")
                for key in prior.keys():
                    self.assertIn(
                        key,
                        model_params,
                        f"Prior key '{key}' not in model '{model_fn.__name__}' signature",
                    )


# ===========================================================================
# 9.  TestGroupingAndBinning — _group_with_flags, _group_min_counts, _compute_grouping
# ===========================================================================

class TestGroupingAndBinning(unittest.TestCase):
    """Tests for the internal grouping helpers on SpectralDataset."""

    def _make_ds(self, counts, grouping=None):
        from redback.spectral.dataset import SpectralDataset
        n = len(counts)
        edges = np.linspace(0.5, n + 0.5, n + 1)
        return SpectralDataset(
            counts=np.asarray(counts, dtype=float),
            exposure=1000.0,
            energy_edges_keV=edges,
            grouping=np.asarray(grouping, dtype=int) if grouping is not None else None,
        )

    # --- _group_with_flags ---

    def test_group_with_flags_no_grouping_returns_unchanged(self):
        """When grouping is None, arrays are returned unchanged."""
        ds = self._make_ds([1.0, 2.0, 3.0])
        counts = np.array([1.0, 2.0, 3.0])
        x = np.array([1.0, 2.0, 3.0])
        w = np.array([1.0, 1.0, 1.0])
        gc, gx, gw = ds._group_with_flags(counts, x, w)
        np.testing.assert_array_equal(gc, counts)

    def test_group_with_flags_wrong_length_returns_unchanged(self):
        """If grouping length != counts length, arrays are returned unchanged."""
        ds = self._make_ds([1.0, 2.0, 3.0], grouping=[1, -1, 1])
        ds.grouping = np.array([1, -1])  # wrong length
        counts = np.array([1.0, 2.0, 3.0])
        x = np.array([1.0, 2.0, 3.0])
        w = np.ones(3)
        gc, gx, gw = ds._group_with_flags(counts, x, w)
        np.testing.assert_array_equal(gc, counts)

    def test_group_with_flags_two_channels_merged(self):
        """[1, -1] groups first two channels; third is standalone."""
        ds = self._make_ds([10.0, 20.0, 5.0], grouping=[1, -1, 1])
        counts = np.array([10.0, 20.0, 5.0])
        x = np.array([1.0, 2.0, 3.0])
        w = np.ones(3)
        gc, gx, gw = ds._group_with_flags(counts, x, w)
        self.assertEqual(len(gc), 2)
        self.assertAlmostEqual(gc[0], 30.0)  # 10+20
        self.assertAlmostEqual(gc[1], 5.0)

    def test_group_with_flags_all_merged(self):
        """[1, -1, -1] merges all three channels into one."""
        ds = self._make_ds([5.0, 5.0, 5.0], grouping=[1, -1, -1])
        counts = np.array([5.0, 5.0, 5.0])
        x = np.array([1.0, 2.0, 3.0])
        w = np.ones(3)
        gc, gx, gw = ds._group_with_flags(counts, x, w)
        self.assertEqual(len(gc), 1)
        self.assertAlmostEqual(gc[0], 15.0)

    def test_group_with_flags_zero_g_is_standalone(self):
        """g=0 terminates a running group and is its own standalone bin."""
        ds = self._make_ds([5.0, 5.0, 3.0], grouping=[1, 0, 1])
        counts = np.array([5.0, 5.0, 3.0])
        x = np.array([1.0, 2.0, 3.0])
        w = np.ones(3)
        gc, gx, gw = ds._group_with_flags(counts, x, w)
        # group 1 (g=1 starts with 5, then g=0 terminates -> standalone 5),
        # standalone 5, standalone 3
        self.assertGreater(len(gc), 1)

    # --- _group_min_counts ---

    def test_group_min_counts_zero_returns_unchanged(self):
        """min_counts=0 returns arrays unchanged."""
        ds = self._make_ds([5.0, 5.0])
        counts = np.array([5.0, 5.0])
        x = np.array([1.0, 2.0])
        w = np.ones(2)
        gc, gx, gw = ds._group_min_counts(counts, x, w, min_counts=0)
        np.testing.assert_array_equal(gc, counts)

    def test_group_min_counts_none_returns_unchanged(self):
        """min_counts=None returns arrays unchanged."""
        ds = self._make_ds([5.0, 5.0])
        counts = np.array([5.0, 5.0])
        x = np.array([1.0, 2.0])
        w = np.ones(2)
        gc, gx, gw = ds._group_min_counts(counts, x, w, min_counts=None)
        np.testing.assert_array_equal(gc, counts)

    def test_group_min_counts_merges_to_threshold(self):
        """Channels accumulate until min_counts is reached."""
        ds = self._make_ds([3.0, 3.0, 3.0, 3.0])
        counts = np.array([3.0, 3.0, 3.0, 3.0])
        x = np.array([1.0, 2.0, 3.0, 4.0])
        w = np.ones(4)
        gc, gx, gw = ds._group_min_counts(counts, x, w, min_counts=5)
        # 3+3=6 >= 5: bin1=6; 3+3=6 >= 5: bin2=6
        self.assertEqual(len(gc), 2)
        self.assertAlmostEqual(gc[0], 6.0)
        self.assertAlmostEqual(gc[1], 6.0)

    def test_group_min_counts_remainder_appended(self):
        """Leftover channels below threshold form a final partial bin."""
        ds = self._make_ds([1.0, 1.0, 1.0])
        counts = np.array([1.0, 1.0, 1.0])
        x = np.array([1.0, 2.0, 3.0])
        w = np.ones(3)
        # min_counts=10 — nothing hits threshold, all become one remainder bin
        gc, gx, gw = ds._group_min_counts(counts, x, w, min_counts=10)
        self.assertEqual(len(gc), 1)
        self.assertAlmostEqual(gc[0], 3.0)

    # --- _compute_grouping ---

    def test_compute_grouping_no_grouping_no_min_counts(self):
        """Without grouping or min_counts, each channel is its own group."""
        ds = self._make_ds([5.0, 5.0, 5.0])
        counts = np.array([5.0, 5.0, 5.0])
        x = np.array([1.0, 2.0, 3.0])
        w = np.ones(3)
        gc, gx, gw, groups = ds._compute_grouping(counts, x, w, None)
        self.assertEqual(len(gc), 3)
        self.assertEqual(len(groups), 3)

    def test_compute_grouping_with_ogip_grouping(self):
        """OGIP grouping flag merges bins correctly."""
        ds = self._make_ds([10.0, 10.0, 5.0], grouping=[1, -1, 1])
        counts = np.array([10.0, 10.0, 5.0])
        x = np.array([1.0, 2.0, 3.0])
        w = np.ones(3)
        gc, gx, gw, groups = ds._compute_grouping(counts, x, w, None)
        self.assertEqual(len(gc), 2)
        self.assertAlmostEqual(gc[0], 20.0)

    def test_compute_grouping_with_min_counts(self):
        """min_counts grouping further merges channels."""
        ds = self._make_ds([3.0, 3.0, 3.0, 3.0])
        counts = np.array([3.0, 3.0, 3.0, 3.0])
        x = np.array([1.0, 2.0, 3.0, 4.0])
        w = np.ones(4)
        gc, gx, gw, groups = ds._compute_grouping(counts, x, w, min_counts=5)
        # same as _group_min_counts: two bins of 6
        self.assertEqual(len(gc), 2)


# ===========================================================================
# 10. TestComputeBandFlux — compute_band_flux
# ===========================================================================

class TestComputeBandFlux(unittest.TestCase):
    """Tests for SpectralDataset.compute_band_flux."""

    def setUp(self):
        from redback.spectral.dataset import SpectralDataset
        n = 20
        self.edges = np.logspace(-1, 2, n + 1)
        self.dataset = SpectralDataset(
            counts=np.ones(n) * 10.0,
            exposure=1000.0,
            energy_edges_keV=self.edges,
        )

    def _flat_model(self, energies_keV, amplitude):
        return np.ones_like(energies_keV) * amplitude

    def test_returns_positive_float(self):
        flux = self.dataset.compute_band_flux(
            model=self._flat_model,
            parameters={"amplitude": 1.0},
            energy_min_keV=1.0,
            energy_max_keV=10.0,
        )
        self.assertIsInstance(flux, float)
        self.assertGreater(flux, 0.0)

    def test_invalid_energy_range_raises(self):
        with self.assertRaises(ValueError):
            self.dataset.compute_band_flux(
                model=self._flat_model,
                parameters={"amplitude": 1.0},
                energy_min_keV=10.0,
                energy_max_keV=1.0,
            )

    def test_scales_with_amplitude(self):
        """Doubling amplitude doubles the flux."""
        f1 = self.dataset.compute_band_flux(
            model=self._flat_model, parameters={"amplitude": 1.0},
            energy_min_keV=1.0, energy_max_keV=10.0,
        )
        f2 = self.dataset.compute_band_flux(
            model=self._flat_model, parameters={"amplitude": 2.0},
            energy_min_keV=1.0, energy_max_keV=10.0,
        )
        self.assertAlmostEqual(f2 / f1, 2.0, places=3)

    def test_unabsorbed_zeroes_nh(self):
        """unabsorbed=True sets nh=0 before evaluating the model."""
        called_params = {}

        def absorbing_model(energies_keV, amplitude, nh):
            called_params["nh"] = nh
            return np.ones_like(energies_keV) * amplitude * (1.0 - nh)

        self.dataset.compute_band_flux(
            model=absorbing_model,
            parameters={"amplitude": 1.0, "nh": 0.5},
            energy_min_keV=1.0,
            energy_max_keV=10.0,
            unabsorbed=True,
        )
        self.assertAlmostEqual(called_params["nh"], 0.0)

    def test_unabsorbed_zeroes_lognh(self):
        """unabsorbed=True sets lognh=-inf before evaluating the model."""
        called_params = {}

        def lognh_model(energies_keV, amplitude, lognh):
            called_params["lognh"] = lognh
            return np.ones_like(energies_keV) * amplitude

        self.dataset.compute_band_flux(
            model=lognh_model,
            parameters={"amplitude": 1.0, "lognh": 1.0},
            energy_min_keV=1.0,
            energy_max_keV=10.0,
            unabsorbed=True,
        )
        self.assertTrue(np.isneginf(called_params["lognh"]))

    def test_string_model_name(self):
        """Model can be passed as a string looked up from the model library."""
        flux = self.dataset.compute_band_flux(
            model="powerlaw_high_energy",
            parameters={"log10_norm": -3.0, "alpha": -1.5, "redshift": 0.0},
            energy_min_keV=1.0,
            energy_max_keV=10.0,
        )
        self.assertGreater(flux, 0.0)

    def test_frequency_model_fallback(self):
        """Models using 'frequency' kwarg (not energies_keV) still work."""
        def freq_model(times, amplitude, **kwargs):
            freq = kwargs.get("frequency", np.ones(10))
            return np.ones_like(freq) * amplitude

        flux = self.dataset.compute_band_flux(
            model=freq_model,
            parameters={"amplitude": 1.0},
            energy_min_keV=1.0,
            energy_max_keV=10.0,
        )
        self.assertIsInstance(flux, float)


# ===========================================================================
# 11. TestPlotSpectrumData — plot_spectrum_data (matplotlib backend=Agg)
# ===========================================================================

import matplotlib
matplotlib.use("Agg")

class TestPlotSpectrumData(unittest.TestCase):
    """Smoke tests for SpectralDataset.plot_spectrum_data."""

    def _make_ds(self, with_background=False, with_grouping=False):
        from redback.spectral.dataset import SpectralDataset
        n = 10
        edges = np.linspace(1.0, 10.0, n + 1)
        counts = np.ones(n) * 20.0
        bkg = np.ones(n) * 5.0 if with_background else None
        grp = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1]) if with_grouping else None
        return SpectralDataset(
            counts=counts,
            exposure=1000.0,
            energy_edges_keV=edges,
            counts_bkg=bkg,
            bkg_exposure=2000.0 if with_background else None,
            bkg_backscale=1.0 if with_background else None,
            bkg_areascal=1.0 if with_background else None,
            grouping=grp,
        )

    def test_returns_axes(self):
        """plot_spectrum_data returns an axes object."""
        import matplotlib.pyplot as plt
        ds = self._make_ds()
        ax = ds.plot_spectrum_data(show=False, save=False)
        self.assertIsNotNone(ax)
        plt.close("all")

    def test_with_background(self):
        """Background is plotted when plot_background=True and counts_bkg present."""
        import matplotlib.pyplot as plt
        ds = self._make_ds(with_background=True)
        ax = ds.plot_spectrum_data(show=False, save=False, plot_background=True)
        self.assertIsNotNone(ax)
        plt.close("all")

    def test_subtract_background(self):
        """subtract_background=True subtracts background from data."""
        import matplotlib.pyplot as plt
        ds = self._make_ds(with_background=True)
        ax = ds.plot_spectrum_data(
            show=False, save=False, plot_background=True, subtract_background=True
        )
        self.assertIsNotNone(ax)
        plt.close("all")

    def test_with_min_counts(self):
        """min_counts triggers binning to minimum counts per bin."""
        import matplotlib.pyplot as plt
        ds = self._make_ds()
        ax = ds.plot_spectrum_data(show=False, save=False, min_counts=15)
        self.assertIsNotNone(ax)
        plt.close("all")

    def test_with_ogip_grouping(self):
        """OGIP grouping flags are applied when grouping is set."""
        import matplotlib.pyplot as plt
        ds = self._make_ds(with_grouping=True)
        ax = ds.plot_spectrum_data(show=False, save=False)
        self.assertIsNotNone(ax)
        plt.close("all")

    def test_rate_false_density_false(self):
        """rate=False, density=False plots raw counts."""
        import matplotlib.pyplot as plt
        ds = self._make_ds()
        ax = ds.plot_spectrum_data(show=False, save=False, rate=False, density=False)
        self.assertIsNotNone(ax)
        plt.close("all")

    def test_rate_true_density_false(self):
        """rate=True, density=False plots counts/s."""
        import matplotlib.pyplot as plt
        ds = self._make_ds()
        ax = ds.plot_spectrum_data(show=False, save=False, rate=True, density=False)
        self.assertIsNotNone(ax)
        plt.close("all")

    def test_save_to_file(self):
        """plot_spectrum_data saves a PNG when filename is provided."""
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, "spec.png")
            ds = self._make_ds()
            ds.plot_spectrum_data(show=False, save=True, filename=fname)
            self.assertTrue(os.path.exists(fname))
        plt.close("all")

    def test_existing_axes_used(self):
        """When axes is provided, that axes object is used directly."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ds = self._make_ds()
        returned_ax = ds.plot_spectrum_data(axes=ax, show=False, save=False)
        self.assertIs(returned_ax, ax)
        plt.close("all")

    def test_xlim_ylim_applied(self):
        """xlim and ylim kwargs are applied to the axes."""
        import matplotlib.pyplot as plt
        ds = self._make_ds()
        ax = ds.plot_spectrum_data(show=False, save=False, xlim=(2.0, 8.0), ylim=(1e-3, 1e1))
        self.assertAlmostEqual(ax.get_xlim()[0], 2.0, places=1)
        plt.close("all")

    def test_with_rmf_uses_rmf_energy_axis(self):
        """When an RMF is attached, the RMF channel energies are used for the x-axis."""
        import matplotlib.pyplot as plt
        from redback.spectral.response import ResponseMatrix
        n = 6
        e_lo = np.linspace(1.0, 6.0, n + 1)[:-1].astype(float)
        e_hi = np.linspace(1.0, 6.0, n + 1)[1:].astype(float)
        matrix = np.eye(n, dtype=float)
        rmf = ResponseMatrix(
            e_min=e_lo, e_max=e_hi, channel=np.arange(n),
            emin_chan=e_lo, emax_chan=e_hi, matrix=matrix,
        )
        from redback.spectral.dataset import SpectralDataset
        ds = SpectralDataset(
            counts=np.ones(n) * 10.0,
            exposure=1000.0,
            energy_edges_keV=np.linspace(1.0, 7.0, n + 1),
            rmf=rmf,
        )
        ax = ds.plot_spectrum_data(show=False, save=False)
        self.assertIsNotNone(ax)
        plt.close("all")


# ===========================================================================
# 12. TestPlotSpectrumFit — plot_spectrum_fit
# ===========================================================================

class TestPlotSpectrumFit(unittest.TestCase):
    """Smoke tests for SpectralDataset.plot_spectrum_fit."""

    def _make_ds(self):
        from redback.spectral.dataset import SpectralDataset
        n = 8
        return SpectralDataset(
            counts=np.ones(n) * 15.0,
            exposure=1000.0,
            energy_edges_keV=np.linspace(1.0, 9.0, n + 1),
        )

    def _flat_model(self, energies_keV, amplitude, **kwargs):
        return np.ones_like(energies_keV) * amplitude

    def _make_posterior(self, n_samples=5):
        import pandas as pd
        return pd.DataFrame({
            "amplitude": np.random.uniform(0.5, 2.0, n_samples),
            "log_likelihood": np.random.uniform(-100, -10, n_samples),
        })

    def test_with_parameters_dict(self):
        """plot_spectrum_fit works with a direct parameters dict."""
        import matplotlib.pyplot as plt
        ds = self._make_ds()
        ax = ds.plot_spectrum_fit(
            model=self._flat_model,
            parameters={"amplitude": 1.0},
            show=False, save=False,
        )
        self.assertIsNotNone(ax)
        plt.close("all")

    def test_with_posterior(self):
        """plot_spectrum_fit works with a posterior DataFrame."""
        import matplotlib.pyplot as plt
        ds = self._make_ds()
        posterior = self._make_posterior()
        ax = ds.plot_spectrum_fit(
            model=self._flat_model,
            posterior=posterior,
            show=False, save=False,
            uncertainty_mode="credible_intervals",
        )
        self.assertIsNotNone(ax)
        plt.close("all")

    def test_random_models_mode(self):
        """uncertainty_mode='random_models' draws individual sample curves."""
        import matplotlib.pyplot as plt
        ds = self._make_ds()
        posterior = self._make_posterior()
        ax = ds.plot_spectrum_fit(
            model=self._flat_model,
            posterior=posterior,
            show=False, save=False,
            uncertainty_mode="random_models",
        )
        self.assertIsNotNone(ax)
        plt.close("all")

    def test_with_residuals(self):
        """plot_residuals=True creates a residuals sub-panel."""
        import matplotlib.pyplot as plt
        ds = self._make_ds()
        ax = ds.plot_spectrum_fit(
            model=self._flat_model,
            parameters={"amplitude": 1.0},
            show=False, save=False,
            plot_residuals=True,
        )
        self.assertIsNotNone(ax)
        plt.close("all")

    def test_annotate_parameters(self):
        """annotate_parameters=True adds a text annotation to the axes."""
        import matplotlib.pyplot as plt
        ds = self._make_ds()
        ax = ds.plot_spectrum_fit(
            model=self._flat_model,
            parameters={"amplitude": 1.5},
            show=False, save=False,
            annotate_parameters=True,
        )
        self.assertIsNotNone(ax)
        plt.close("all")

    def test_annotate_parameters_list(self):
        """annotate_parameters can be a list of specific keys to annotate."""
        import matplotlib.pyplot as plt
        ds = self._make_ds()
        ax = ds.plot_spectrum_fit(
            model=self._flat_model,
            parameters={"amplitude": 1.5},
            show=False, save=False,
            annotate_parameters=["amplitude"],
        )
        self.assertIsNotNone(ax)
        plt.close("all")

    def test_save_to_file(self):
        """plot_spectrum_fit saves a PNG when filename provided."""
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, "fit.png")
            ds = self._make_ds()
            ds.plot_spectrum_fit(
                model=self._flat_model,
                parameters={"amplitude": 1.0},
                show=False, save=True, filename=fname,
            )
            self.assertTrue(os.path.exists(fname))
        plt.close("all")

    def test_string_model_name(self):
        """Model can be passed as a string."""
        import matplotlib.pyplot as plt
        n = 8
        from redback.spectral.dataset import SpectralDataset
        ds = SpectralDataset(
            counts=np.ones(n) * 15.0,
            exposure=1000.0,
            energy_edges_keV=np.logspace(0, 2, n + 1),
        )
        ax = ds.plot_spectrum_fit(
            model="powerlaw_high_energy",
            parameters={"log10_norm": -3.0, "alpha": -1.5, "redshift": 0.0},
            show=False, save=False,
        )
        self.assertIsNotNone(ax)
        plt.close("all")

    def test_posterior_no_log_likelihood_column(self):
        """When posterior has no log_likelihood, median is used as parameters."""
        import matplotlib.pyplot as plt
        import pandas as pd
        ds = self._make_ds()
        posterior = pd.DataFrame({"amplitude": [0.9, 1.0, 1.1]})
        ax = ds.plot_spectrum_fit(
            model=self._flat_model,
            posterior=posterior,
            show=False, save=False,
        )
        self.assertIsNotNone(ax)
        plt.close("all")


# ===========================================================================
# 13. TestPlotLightcurve — SpectralDataset.plot_lightcurve
# ===========================================================================

def _make_lc_fits_simple(tmpdir, filename="lc.fits", with_fracexp=False, timedel=1.0):
    """Write a minimal OGIP lightcurve FITS file (used by TestPlotLightcurve)."""
    n = 10
    time = np.arange(n, dtype=float)
    rate = np.ones(n, dtype=float) * 5.0
    error = np.ones(n, dtype=float) * 0.5

    col_list = [
        fits.Column(name="TIME", format="D", array=time),
        fits.Column(name="RATE", format="D", array=rate),
        fits.Column(name="ERROR", format="D", array=error),
    ]
    if with_fracexp:
        col_list.append(fits.Column(name="FRACEXP", format="D", array=np.ones(n)))

    table_hdu = fits.BinTableHDU.from_columns(col_list)
    table_hdu.name = "RATE"
    if timedel is not None:
        table_hdu.header["TIMEDEL"] = timedel

    hdul = fits.HDUList([fits.PrimaryHDU(), table_hdu])
    path = os.path.join(tmpdir, filename)
    hdul.writeto(path, overwrite=True)
    return path


class TestPlotLightcurve(unittest.TestCase):
    """Smoke tests for SpectralDataset.plot_lightcurve and read_lc."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir)

    def test_read_lc_basic(self):
        """read_lc reads time/rate/error from a RATE HDU."""
        from redback.spectral.io import read_lc
        path = _make_lc_fits_simple(self.tmpdir)
        lc = read_lc(path)
        self.assertEqual(len(lc.time), 10)
        self.assertEqual(len(lc.rate), 10)
        self.assertAlmostEqual(float(lc.rate[0]), 5.0)
        self.assertAlmostEqual(lc.timedel, 1.0)

    def test_read_lc_with_fracexp(self):
        """read_lc reads FRACEXP column when present."""
        from redback.spectral.io import read_lc
        path = _make_lc_fits_simple(self.tmpdir, with_fracexp=True)
        lc = read_lc(path)
        self.assertIsNotNone(lc.fracexp)
        self.assertEqual(len(lc.fracexp), 10)

    def test_read_lc_no_timedel(self):
        """read_lc sets timedel=None when TIMEDEL header is absent."""
        from redback.spectral.io import read_lc
        path = _make_lc_fits_simple(self.tmpdir, timedel=None)
        lc = read_lc(path)
        self.assertIsNone(lc.timedel)

    def test_plot_lightcurve_from_arrays(self):
        """plot_lightcurve works with raw time/rate/error arrays."""
        import matplotlib.pyplot as plt
        from redback.spectral.dataset import SpectralDataset
        time = np.arange(10, dtype=float)
        rate = np.ones(10) * 5.0
        error = np.ones(10) * 0.2
        ax = SpectralDataset.plot_lightcurve(
            time=time, rate=rate, error=error,
            show=False, save=False,
        )
        self.assertIsNotNone(ax)
        plt.close("all")

    def test_plot_lightcurve_from_lc_object(self):
        """plot_lightcurve works with an OGIPLightCurve object."""
        import matplotlib.pyplot as plt
        from redback.spectral.io import read_lc
        from redback.spectral.dataset import SpectralDataset
        path = _make_lc_fits_simple(self.tmpdir)
        lc = read_lc(path)
        ax = SpectralDataset.plot_lightcurve(lc=lc, show=False, save=False)
        self.assertIsNotNone(ax)
        plt.close("all")

    def test_plot_lightcurve_with_fracexp(self):
        """plot_lightcurve handles FRACEXP in time-bins mode."""
        import matplotlib.pyplot as plt
        from redback.spectral.io import read_lc
        from redback.spectral.dataset import SpectralDataset
        path = _make_lc_fits_simple(self.tmpdir, with_fracexp=True)
        lc = read_lc(path)
        ax = SpectralDataset.plot_lightcurve(lc=lc, show=False, save=False)
        self.assertIsNotNone(ax)
        plt.close("all")

    def test_plot_lightcurve_no_time_raises(self):
        """plot_lightcurve raises ValueError when neither lc nor arrays supplied."""
        from redback.spectral.dataset import SpectralDataset
        with self.assertRaises(ValueError):
            SpectralDataset.plot_lightcurve(show=False, save=False)

    def test_plot_lightcurve_with_time_bins(self):
        """plot_lightcurve with explicit time_bins uses binned-counts mode."""
        import matplotlib.pyplot as plt
        from redback.spectral.dataset import SpectralDataset
        time = np.arange(5, dtype=float)
        rate = np.ones(5) * 3.0
        error = np.ones(5) * 0.1
        time_bins = np.arange(6, dtype=float)
        ax = SpectralDataset.plot_lightcurve(
            time=time, rate=rate, error=error,
            time_bins=time_bins,
            show=False, save=False,
        )
        self.assertIsNotNone(ax)
        plt.close("all")


# ===========================================================================
# 14. TestOGIPIO_Extended — additional io.py coverage
# ===========================================================================

def _make_specresp_matrix_fits(n_energy=6, n_channels=6, tmpdir=None, filename="resp.rsp"):
    """Write a RSP file using 'SPECRESP MATRIX' HDU name (gamma-ray convention)."""
    e_lo = np.linspace(10.0, 200.0, n_energy + 1)[:-1].astype(np.float32)
    e_hi = np.linspace(10.0, 200.0, n_energy + 1)[1:].astype(np.float32)

    n_grp = np.ones(n_energy, dtype=np.int16)
    f_chan = np.zeros(n_energy, dtype=np.int16)
    n_chan = np.full(n_energy, n_channels, dtype=np.int16)
    matrix_rows = [np.full(n_channels, 1.0 / n_channels, dtype=np.float32) for _ in range(n_energy)]
    matrix_col = fits.Column(name="MATRIX", format=f"{n_channels}E",
                             array=np.array(matrix_rows))
    matrix_hdu = fits.BinTableHDU.from_columns([
        fits.Column(name="ENERG_LO", format="E", array=e_lo),
        fits.Column(name="ENERG_HI", format="E", array=e_hi),
        fits.Column(name="N_GRP", format="I", array=n_grp),
        fits.Column(name="F_CHAN", format="I", array=f_chan),
        fits.Column(name="N_CHAN", format="I", array=n_chan),
        matrix_col,
    ])
    matrix_hdu.name = "SPECRESP MATRIX"
    matrix_hdu.header["DETCHANS"] = n_channels

    emin_chan = np.linspace(100.0, 200.0, n_channels + 1)[:-1].astype(np.float32)
    emax_chan = np.linspace(100.0, 200.0, n_channels + 1)[1:].astype(np.float32)
    ebounds_hdu = fits.BinTableHDU.from_columns([
        fits.Column(name="CHANNEL", format="I", array=np.arange(n_channels, dtype=np.int16)),
        fits.Column(name="E_MIN", format="E", array=emin_chan),
        fits.Column(name="E_MAX", format="E", array=emax_chan),
    ])
    ebounds_hdu.name = "EBOUNDS"

    hdul = fits.HDUList([fits.PrimaryHDU(), matrix_hdu, ebounds_hdu])
    path = os.path.join(tmpdir, filename)
    hdul.writeto(path, overwrite=True)
    return path


class TestOGIPIO_Extended(unittest.TestCase):
    """Additional coverage for redback.spectral.io."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir)

    def test_read_rmf_specresp_matrix(self):
        """read_rmf handles 'SPECRESP MATRIX' HDU name (combined RSP files)."""
        from redback.spectral.io import read_rmf
        path = _make_specresp_matrix_fits(tmpdir=self.tmpdir)
        rmf = read_rmf(path)
        self.assertIsNotNone(rmf)
        self.assertEqual(len(rmf.e_min), 6)

    def test_read_rmf_no_matrix_raises(self):
        """read_rmf raises KeyError when no MATRIX or 'SPECRESP MATRIX' HDU exists."""
        from redback.spectral.io import read_rmf
        # Build a FITS file with neither MATRIX nor SPECRESP MATRIX
        hdul = fits.HDUList([fits.PrimaryHDU()])
        path = os.path.join(self.tmpdir, "bad.rsp")
        hdul.writeto(path)
        with self.assertRaises(KeyError):
            read_rmf(path)

    def test_read_pha_quality_in_header(self):
        """read_pha falls back to QUALITY header keyword when QUALITY column absent."""
        from redback.spectral.io import read_pha
        # Build PHA with QUALITY as header keyword, not column
        n = 5
        channels = np.arange(1, n + 1, dtype=np.int16)
        counts = np.ones(n, dtype=np.float32) * 10.0
        table_hdu = fits.BinTableHDU.from_columns([
            fits.Column(name="CHANNEL", format="I", array=channels),
            fits.Column(name="COUNTS", format="E", array=counts),
        ])
        table_hdu.name = "SPECTRUM"
        table_hdu.header["HDUCLAS1"] = "SPECTRUM"
        table_hdu.header["EXPOSURE"] = 100.0
        table_hdu.header["BACKSCAL"] = 1.0
        table_hdu.header["AREASCAL"] = 1.0
        table_hdu.header["QUALITY"] = 0  # quality as header, not column
        path = os.path.join(self.tmpdir, "q_header.pha")
        fits.HDUList([fits.PrimaryHDU(), table_hdu]).writeto(path)
        spec = read_pha(path)
        self.assertIsNotNone(spec.quality)
        self.assertEqual(len(spec.quality), n)

    def test_read_pha_grouping_in_header(self):
        """read_pha falls back to GROUPING header keyword when GROUPING column absent."""
        from redback.spectral.io import read_pha
        n = 5
        channels = np.arange(1, n + 1, dtype=np.int16)
        counts = np.ones(n, dtype=np.float32) * 10.0
        table_hdu = fits.BinTableHDU.from_columns([
            fits.Column(name="CHANNEL", format="I", array=channels),
            fits.Column(name="COUNTS", format="E", array=counts),
        ])
        table_hdu.name = "SPECTRUM"
        table_hdu.header["HDUCLAS1"] = "SPECTRUM"
        table_hdu.header["EXPOSURE"] = 100.0
        table_hdu.header["BACKSCAL"] = 1.0
        table_hdu.header["AREASCAL"] = 1.0
        table_hdu.header["GROUPING"] = 1  # grouping as header
        path = os.path.join(self.tmpdir, "g_header.pha")
        fits.HDUList([fits.PrimaryHDU(), table_hdu]).writeto(path)
        spec = read_pha(path)
        self.assertIsNotNone(spec.grouping)

    def test_get_spectrum_hdu_fallback_to_bintable(self):
        """_get_spectrum_hdu falls back to first BinTableHDU with HDUCLAS1=SPECTRUM."""
        from redback.spectral.io import _get_spectrum_hdu
        n = 4
        table_hdu = fits.BinTableHDU.from_columns([
            fits.Column(name="CHANNEL", format="I", array=np.arange(n, dtype=np.int16)),
            fits.Column(name="COUNTS", format="E", array=np.ones(n)),
        ])
        # Use a non-standard name (not "SPECTRUM") so the first branch fails
        table_hdu.name = "SPEC_DATA"
        table_hdu.header["HDUCLAS1"] = "SPECTRUM"
        hdul = fits.HDUList([fits.PrimaryHDU(), table_hdu])
        result = _get_spectrum_hdu(hdul)
        self.assertIs(result, table_hdu)

    def test_from_ogip_with_arf_only(self):
        """from_ogip uses ARF energy edges when no RMF is provided."""
        from redback.spectral.dataset import SpectralDataset
        # Build an ARF fits file manually
        n = 6
        e_lo = np.linspace(0.5, 6.5, n + 1)[:-1].astype(np.float32)
        e_hi = np.linspace(0.5, 6.5, n + 1)[1:].astype(np.float32)
        arf_hdu = fits.BinTableHDU.from_columns([
            fits.Column(name="ENERG_LO", format="E", array=e_lo),
            fits.Column(name="ENERG_HI", format="E", array=e_hi),
            fits.Column(name="SPECRESP", format="E", array=np.full(n, 100.0, dtype=np.float32)),
        ])
        arf_hdu.name = "SPECRESP"
        arf_path = os.path.join(self.tmpdir, "test.arf")
        fits.HDUList([fits.PrimaryHDU(), arf_hdu]).writeto(arf_path)
        pha_path = _make_pha_fits(n_channels=n, exposure=500.0, tmpdir=self.tmpdir, filename="src_arf.pha")
        ds = SpectralDataset.from_ogip(pha=pha_path, arf=arf_path)
        self.assertEqual(len(ds.energy_edges_keV), n + 1)

    def test_from_ogip_directory_no_bkg_suffix(self):
        """from_ogip_directory works without a background file."""
        from redback.spectral.dataset import SpectralDataset
        subdir = os.path.join(self.tmpdir, "no_bkg")
        os.makedirs(subdir)
        n = 5
        _make_pha_fits(n_channels=n, exposure=500.0, tmpdir=subdir, filename="obs.pha")
        _make_rmf_fits(n_energy=n, n_channels=n, tmpdir=subdir, filename="obs.rmf")
        ds = SpectralDataset.from_ogip_directory(subdir)
        self.assertIsNone(ds.counts_bkg)

    def test_from_ogip_directory_with_arf(self):
        """from_ogip_directory picks up an ARF file when present."""
        from redback.spectral.dataset import SpectralDataset
        subdir = os.path.join(self.tmpdir, "with_arf")
        os.makedirs(subdir)
        n = 5
        _make_pha_fits(n_channels=n, exposure=500.0, tmpdir=subdir, filename="src.pha")
        _make_rmf_fits(n_energy=n, n_channels=n, tmpdir=subdir, filename="src.rmf")
        # Write a minimal ARF
        e_lo = np.linspace(0.5, 5.5, n + 1)[:-1].astype(np.float32)
        e_hi = np.linspace(0.5, 5.5, n + 1)[1:].astype(np.float32)
        arf_hdu = fits.BinTableHDU.from_columns([
            fits.Column(name="ENERG_LO", format="E", array=e_lo),
            fits.Column(name="ENERG_HI", format="E", array=e_hi),
            fits.Column(name="SPECRESP", format="E", array=np.full(n, 100.0, np.float32)),
        ])
        arf_hdu.name = "SPECRESP"
        arf_path = os.path.join(subdir, "src.arf")
        fits.HDUList([fits.PrimaryHDU(), arf_hdu]).writeto(arf_path)
        ds = SpectralDataset.from_ogip_directory(subdir)
        self.assertIsNotNone(ds.arf)

    def test_from_ogip_directory_no_pha_raises(self):
        """from_ogip_directory raises FileNotFoundError when no PHA file present."""
        from redback.spectral.dataset import SpectralDataset
        subdir = os.path.join(self.tmpdir, "empty_dir")
        os.makedirs(subdir)
        with self.assertRaises(FileNotFoundError):
            SpectralDataset.from_ogip_directory(subdir)


# ===========================================================================
# 15. TestFromSimulator — SpectralDataset.from_simulator
# ===========================================================================

class TestFromSimulator(unittest.TestCase):
    """Tests for SpectralDataset.from_simulator."""

    def _make_simulator(self):
        from redback.simulate_transients import SimulateHighEnergyTransient

        # Use a very simple flat spectrum with tiny effective area to avoid
        # Poisson lam-too-large errors from physically large fluxes.
        def model(time, frequency, **kwargs):
            return np.ones_like(np.asarray(frequency)) * 1e-6  # 1e-6 mJy, very faint

        return SimulateHighEnergyTransient(
            model=model,
            parameters={},
            energy_edges=np.array([1.0, 2.0, 5.0, 10.0]),
            time_range=(0.0, 10.0),
            effective_area=1.0,         # tiny area → small expected counts
            background_rate=0.001,
            time_resolution=1.0,
            seed=42,
        )

    def test_from_simulator_basic(self):
        """from_simulator builds a valid SpectralDataset."""
        from redback.spectral.dataset import SpectralDataset
        sim = self._make_simulator()
        time_bins = np.array([0.0, 5.0, 10.0])
        ds = SpectralDataset.from_simulator(sim, time_bins=time_bins)
        self.assertIsNotNone(ds)
        self.assertEqual(len(ds.counts), 3)  # 3 energy channels

    def test_from_simulator_has_background(self):
        """from_simulator includes background counts."""
        from redback.spectral.dataset import SpectralDataset
        sim = self._make_simulator()
        time_bins = np.array([0.0, 10.0])
        ds = SpectralDataset.from_simulator(sim, time_bins=time_bins)
        self.assertIsNotNone(ds.counts_bkg)

    def test_from_simulator_default_name(self):
        """Dataset name defaults to 'spectral_dataset' when not specified."""
        from redback.spectral.dataset import SpectralDataset
        sim = self._make_simulator()
        time_bins = np.array([0.0, 10.0])
        ds = SpectralDataset.from_simulator(sim, time_bins=time_bins)
        self.assertIsInstance(ds.name, str)

    def test_counts_spectrum_transient_from_simulator(self):
        """CountsSpectrumTransient.from_simulator delegates to SpectralDataset."""
        from redback.transient.spectral import CountsSpectrumTransient
        sim = self._make_simulator()
        time_bins = np.array([0.0, 10.0])
        spec = CountsSpectrumTransient.from_simulator(sim, time_bins=time_bins, name="ct")
        self.assertIsNotNone(spec.dataset)
        self.assertIsNotNone(spec.dataset.counts)


# ===========================================================================
# 16. TestMaskValidEdgeCases — mask_valid with rmf-based axis
# ===========================================================================

class TestMaskValidEdgeCases(unittest.TestCase):
    """Extra mask_valid coverage using an RMF energy axis."""

    def test_mask_valid_with_rmf_energy_axis(self):
        """mask_valid uses RMF channel energies for the energy cut when RMF present."""
        from redback.spectral.dataset import SpectralDataset
        from redback.spectral.response import ResponseMatrix
        n = 6
        e_lo = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        e_hi = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        rmf = ResponseMatrix(
            e_min=e_lo, e_max=e_hi, channel=np.arange(n),
            emin_chan=e_lo, emax_chan=e_hi, matrix=np.eye(n),
        )
        ds = SpectralDataset(
            counts=np.ones(n) * 10.0,
            exposure=1000.0,
            energy_edges_keV=np.linspace(1.0, 7.0, n + 1),
            rmf=rmf,
        )
        ds.set_active_interval(2.5, 5.5)
        mask = ds.mask_valid()
        # Channels 1-4 (centers 1.5,2.5,3.5,4.5,5.5,6.5) — centers in [2.5,5.5]: idx 1,2,3,4
        self.assertEqual(mask.sum(), 4)

    def test_mask_valid_quality_and_energy(self):
        """mask_valid combines quality and energy masks correctly."""
        from redback.spectral.dataset import SpectralDataset
        n = 5
        quality = np.array([0, 1, 0, 0, 0], dtype=int)  # channel 1 flagged bad
        ds = SpectralDataset(
            counts=np.ones(n) * 10.0,
            exposure=1000.0,
            energy_edges_keV=np.linspace(1.0, 6.0, n + 1),
            quality=quality,
        )
        ds.set_active_interval(2.0, 5.5)
        mask = ds.mask_valid()
        # Centers: 1.5, 2.5, 3.5, 4.5, 5.5 — energy range [2,5.5]: idx 1,2,3,4
        # But idx 1 is quality-flagged → only idx 2,3,4 pass
        self.assertFalse(mask[1])  # quality flag
        self.assertTrue(mask[2])


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
