"""
Comprehensive unit tests for redback X-ray spectral fitting code.

Covers:
    - redback/spectral/conversions.py
    - redback/spectral/response.py
    - redback/spectral/folding.py
    - redback/transient_models/spectral_models.py  (high-energy models added in this branch)
"""

from __future__ import annotations

import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal


# ---------------------------------------------------------------------------
# Helpers shared across test modules
# ---------------------------------------------------------------------------

def _make_identity_rmf(n):
    """Return a ResponseMatrix whose matrix is the n x n identity."""
    from redback.spectral.response import ResponseMatrix
    e_min = np.arange(n, dtype=float)
    e_max = e_min + 1.0
    channels = np.arange(n, dtype=float)
    return ResponseMatrix(
        e_min=e_min,
        e_max=e_max,
        channel=channels,
        emin_chan=e_min,
        emax_chan=e_max,
        matrix=np.eye(n),
    )


def _make_uniform_arf(n, area=100.0, e_start=1.0):
    """Return an EffectiveArea with constant area over n bins.

    By default the bins start at e_start=1.0 keV so they align with
    the energy_edges used in TestFoldSpectrum (1..n+1 keV).
    """
    from redback.spectral.response import EffectiveArea
    e_min = np.arange(n, dtype=float) + e_start
    e_max = e_min + 1.0
    return EffectiveArea(
        e_min=e_min,
        e_max=e_max,
        area=np.full(n, area),
    )


# ===========================================================================
# 1.  TestConversions
# ===========================================================================

class TestConversions(unittest.TestCase):
    """Tests for redback/spectral/conversions.py"""

    # ------------------------------------------------------------------
    # Module-level constants used for hand-calculations
    # ------------------------------------------------------------------
    _MJY_TO_FNU = 1e-26        # erg / s / cm^2 / Hz
    _PLANCK_ERG_S = 6.62607015e-27  # erg * s
    _KEV_TO_HZ = 2.417989e17   # Hz / keV

    def setUp(self):
        from redback.spectral import conversions
        self.conv = conversions
        self.energies = np.array([0.5, 1.0, 2.0, 5.0, 10.0])   # keV
        self.flux_mjy = np.array([1.0, 2.0, 0.5, 3.0, 0.1])    # mJy

    def tearDown(self):
        del self.conv
        del self.energies
        del self.flux_mjy

    # --- mjy_to_fnu ---

    def test_mjy_to_fnu_scalar_known_value(self):
        """1 mJy must equal 1e-26 erg/s/cm^2/Hz."""
        result = self.conv.mjy_to_fnu(1.0)
        self.assertAlmostEqual(result, 1e-26, places=36)

    def test_mjy_to_fnu_array_shape(self):
        result = self.conv.mjy_to_fnu(self.flux_mjy)
        self.assertEqual(result.shape, self.flux_mjy.shape)

    def test_mjy_to_fnu_array_values(self):
        result = self.conv.mjy_to_fnu(self.flux_mjy)
        expected = self.flux_mjy * self._MJY_TO_FNU
        assert_allclose(result, expected, rtol=1e-12)

    def test_mjy_to_fnu_zero_input(self):
        result = self.conv.mjy_to_fnu(0.0)
        self.assertEqual(result, 0.0)

    def test_mjy_to_fnu_zero_array(self):
        result = self.conv.mjy_to_fnu(np.zeros(5))
        assert_array_equal(result, np.zeros(5))

    def test_mjy_to_fnu_linearity(self):
        """Doubling input must double output."""
        r1 = self.conv.mjy_to_fnu(self.flux_mjy)
        r2 = self.conv.mjy_to_fnu(self.flux_mjy * 2)
        assert_allclose(r2, r1 * 2, rtol=1e-12)

    def test_mjy_to_fnu_returns_numpy_type(self):
        """Scalar input must return a numpy numeric type (not a plain Python float)."""
        result = self.conv.mjy_to_fnu(5.0)
        self.assertIsInstance(result, (np.ndarray, np.floating))

    # --- mjy_to_photon_flux_per_keV ---

    def test_mjy_to_photon_flux_shape(self):
        result = self.conv.mjy_to_photon_flux_per_keV(self.flux_mjy, self.energies)
        self.assertEqual(result.shape, self.flux_mjy.shape)

    def test_mjy_to_photon_flux_positive(self):
        result = self.conv.mjy_to_photon_flux_per_keV(self.flux_mjy, self.energies)
        self.assertTrue(np.all(result > 0))

    def test_mjy_to_photon_flux_units_consistency(self):
        """
        N_E = F_nu / (h * E_keV).
        Verify h * E_keV * N_E == F_nu.
        """
        result = self.conv.mjy_to_photon_flux_per_keV(self.flux_mjy, self.energies)
        fnu_expected = self.conv.mjy_to_fnu(self.flux_mjy)
        fnu_recovered = result * self._PLANCK_ERG_S * self.energies
        assert_allclose(fnu_recovered, fnu_expected, rtol=1e-10)

    def test_mjy_to_photon_flux_scalar(self):
        result = self.conv.mjy_to_photon_flux_per_keV(1.0, 1.0)
        expected = self._MJY_TO_FNU / self._PLANCK_ERG_S
        assert_allclose(result, expected, rtol=1e-10)

    def test_mjy_to_photon_flux_inverse_energy_scaling(self):
        """N_E is inversely proportional to energy at fixed flux."""
        n_1keV = self.conv.mjy_to_photon_flux_per_keV(1.0, 1.0)
        n_2keV = self.conv.mjy_to_photon_flux_per_keV(1.0, 2.0)
        assert_allclose(n_1keV / n_2keV, 2.0, rtol=1e-10)

    # --- mjy_to_energy_flux_per_keV ---

    def test_mjy_to_energy_flux_shape(self):
        result = self.conv.mjy_to_energy_flux_per_keV(self.flux_mjy)
        self.assertEqual(result.shape, self.flux_mjy.shape)

    def test_mjy_to_energy_flux_known_conversion_factor(self):
        """
        F_keV = F_nu * (Hz/keV) = F_mJy * 1e-26 * 2.417989e17
        """
        result = self.conv.mjy_to_energy_flux_per_keV(self.flux_mjy)
        expected = self.flux_mjy * self._MJY_TO_FNU * self._KEV_TO_HZ
        assert_allclose(result, expected, rtol=1e-10)

    def test_mjy_to_energy_flux_scalar_known_value(self):
        result = self.conv.mjy_to_energy_flux_per_keV(1.0)
        expected = 1e-26 * 2.417989e17
        assert_allclose(result, expected, rtol=1e-10)

    def test_mjy_to_energy_flux_positive_for_positive_input(self):
        result = self.conv.mjy_to_energy_flux_per_keV(self.flux_mjy)
        self.assertTrue(np.all(result > 0))

    def test_mjy_to_energy_flux_linearity(self):
        r1 = self.conv.mjy_to_energy_flux_per_keV(self.flux_mjy)
        r2 = self.conv.mjy_to_energy_flux_per_keV(self.flux_mjy * 3)
        assert_allclose(r2, r1 * 3, rtol=1e-12)


# ===========================================================================
# 2.  TestResponseMatrix
# ===========================================================================

class TestResponseMatrix(unittest.TestCase):
    """Tests for redback/spectral/response.ResponseMatrix"""

    def setUp(self):
        from redback.spectral.response import ResponseMatrix
        self.n = 5
        self.e_min = np.array([0.5, 1.0, 2.0, 3.0, 4.0])
        self.e_max = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.channel = np.arange(self.n, dtype=float)
        # Simple redistribution matrix: diagonal with known off-diagonal leakage
        self.matrix = np.diag([1.0, 0.9, 0.8, 0.7, 0.6])
        self.rmf = ResponseMatrix(
            e_min=self.e_min,
            e_max=self.e_max,
            channel=self.channel,
            emin_chan=self.e_min,
            emax_chan=self.e_max,
            matrix=self.matrix,
        )

    def tearDown(self):
        del self.rmf

    def test_energy_centers_correct_midpoints(self):
        expected = 0.5 * (self.e_min + self.e_max)
        assert_allclose(self.rmf.energy_centers, expected, rtol=1e-12)

    def test_energy_centers_shape(self):
        self.assertEqual(self.rmf.energy_centers.shape, (self.n,))

    def test_apply_identity_matrix(self):
        """An identity RMF must leave the photon flux unchanged."""
        identity_rmf = _make_identity_rmf(self.n)
        flux = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = identity_rmf.apply(flux)
        assert_allclose(result, flux, rtol=1e-12)

    def test_apply_known_redistribution(self):
        """Output must equal matrix @ flux."""
        flux = np.array([10.0, 5.0, 3.0, 2.0, 1.0])
        result = self.rmf.apply(flux)
        expected = self.matrix @ flux
        assert_allclose(result, expected, rtol=1e-12)

    def test_apply_output_shape(self):
        flux = np.ones(self.n)
        result = self.rmf.apply(flux)
        self.assertEqual(result.shape, (self.n,))

    def test_apply_zero_flux(self):
        """Zero photon flux must produce zero counts."""
        flux = np.zeros(self.n)
        result = self.rmf.apply(flux)
        assert_array_equal(result, np.zeros(self.n))

    def test_apply_single_channel_selects_correct_row(self):
        """
        A matrix where every row is zero except the first should map all
        flux onto channel 0 only.
        """
        from redback.spectral.response import ResponseMatrix
        mat = np.zeros((self.n, self.n))
        mat[0, :] = 1.0          # every photon ends up in channel 0
        rmf = ResponseMatrix(
            e_min=self.e_min, e_max=self.e_max,
            channel=self.channel, emin_chan=self.e_min, emax_chan=self.e_max,
            matrix=mat,
        )
        flux = np.ones(self.n)
        result = rmf.apply(flux)
        self.assertAlmostEqual(result[0], float(self.n))
        assert_allclose(result[1:], 0.0, atol=1e-15)

    def test_apply_linearity(self):
        """Doubling flux must double output."""
        flux = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r1 = self.rmf.apply(flux)
        r2 = self.rmf.apply(flux * 2)
        assert_allclose(r2, r1 * 2, rtol=1e-12)


# ===========================================================================
# 3.  TestEffectiveArea
# ===========================================================================

class TestEffectiveArea(unittest.TestCase):
    """Tests for redback/spectral/response.EffectiveArea"""

    def setUp(self):
        from redback.spectral.response import EffectiveArea
        # 10 bins from 0.5 to 10.5 keV
        self.n = 10
        self.e_min = np.arange(self.n, dtype=float) + 0.5
        self.e_max = self.e_min + 1.0
        self.area_values = np.linspace(100.0, 500.0, self.n)
        self.arf = EffectiveArea(
            e_min=self.e_min,
            e_max=self.e_max,
            area=self.area_values,
        )

    def tearDown(self):
        del self.arf

    def test_energy_centers_correct_midpoints(self):
        expected = 0.5 * (self.e_min + self.e_max)
        assert_allclose(self.arf.energy_centers, expected, rtol=1e-12)

    def test_energy_centers_shape(self):
        self.assertEqual(self.arf.energy_centers.shape, (self.n,))

    def test_evaluate_at_known_center_energy(self):
        """evaluate() at the first bin centre must return the first area value."""
        centers = self.arf.energy_centers
        result = self.arf.evaluate(np.array([centers[0]]))
        assert_allclose(result, np.array([self.area_values[0]]), rtol=1e-10)

    def test_evaluate_at_all_centers(self):
        centers = self.arf.energy_centers
        result = self.arf.evaluate(centers)
        assert_allclose(result, self.area_values, rtol=1e-10)

    def test_evaluate_interpolates_between_bins(self):
        """At a midpoint between two centre energies, result should be between the two areas."""
        centers = self.arf.energy_centers
        mid = 0.5 * (centers[3] + centers[4])
        result = self.arf.evaluate(np.array([mid]))
        lo = min(self.area_values[3], self.area_values[4])
        hi = max(self.area_values[3], self.area_values[4])
        self.assertGreaterEqual(result[0], lo)
        self.assertLessEqual(result[0], hi)

    def test_evaluate_extrapolation_left_returns_zero(self):
        """Below the grid range evaluate() must return 0."""
        result = self.arf.evaluate(np.array([0.01]))
        self.assertEqual(result[0], 0.0)

    def test_evaluate_extrapolation_right_returns_zero(self):
        """Above the grid range evaluate() must return 0."""
        result = self.arf.evaluate(np.array([1000.0]))
        self.assertEqual(result[0], 0.0)

    def test_evaluate_output_shape_matches_input(self):
        query = np.linspace(1.0, 5.0, 20)
        result = self.arf.evaluate(query)
        self.assertEqual(result.shape, query.shape)

    def test_evaluate_uniform_arf(self):
        """For a constant area, every in-range query must return that constant."""
        arf = _make_uniform_arf(self.n, area=250.0)
        centers = arf.energy_centers
        result = arf.evaluate(centers)
        assert_allclose(result, 250.0, rtol=1e-10)


# ===========================================================================
# 4.  TestFoldSpectrum
# ===========================================================================

class TestFoldSpectrum(unittest.TestCase):
    """Tests for redback/spectral/folding.py"""

    def setUp(self):
        self.n = 8
        # Energy edges: 1, 2, 3, ... 9 keV  (8 bins)
        self.energy_edges = np.arange(1, self.n + 2, dtype=float)
        self.energy_centers = 0.5 * (self.energy_edges[:-1] + self.energy_edges[1:])
        # Flat spectrum in mJy
        self.flat_flux = np.ones(self.n) * 2.0

    def tearDown(self):
        del self.energy_edges
        del self.energy_centers
        del self.flat_flux

    # --- fold_spectrum with no rmf, no arf ---

    def test_no_rmf_no_arf_output_shape(self):
        from redback.spectral.folding import fold_spectrum
        result = fold_spectrum(self.flat_flux, self.energy_edges)
        self.assertEqual(result.shape, (self.n,))

    def test_no_rmf_no_arf_values_positive(self):
        from redback.spectral.folding import fold_spectrum
        result = fold_spectrum(self.flat_flux, self.energy_edges)
        self.assertTrue(np.all(result > 0))

    def test_no_rmf_no_arf_exposure_one(self):
        """With exposure=1 the result is the raw photon-flux-per-bin."""
        from redback.spectral.folding import fold_spectrum
        result = fold_spectrum(self.flat_flux, self.energy_edges, exposure=1.0)
        self.assertTrue(np.all(result > 0))

    # --- fold_spectrum with arf only ---

    def test_arf_only_scaling_applied(self):
        """
        With a uniform ARF and no RMF, the output should equal
        photon_flux_per_bin * area * exposure.
        """
        from redback.spectral.folding import fold_spectrum, _integrate_photon_flux_per_bin
        from redback.spectral.conversions import mjy_to_photon_flux_per_keV
        area = 200.0
        exposure = 5.0
        arf = _make_uniform_arf(self.n, area=area)

        result = fold_spectrum(self.flat_flux, self.energy_edges,
                               arf=arf, exposure=exposure)

        photon_flux_per_keV = mjy_to_photon_flux_per_keV(self.flat_flux, self.energy_centers)
        expected_per_bin = _integrate_photon_flux_per_bin(photon_flux_per_keV, self.energy_edges)
        expected = expected_per_bin * area * exposure
        assert_allclose(result, expected, rtol=1e-8)

    def test_arf_only_output_shape(self):
        from redback.spectral.folding import fold_spectrum
        arf = _make_uniform_arf(self.n, area=100.0)
        result = fold_spectrum(self.flat_flux, self.energy_edges, arf=arf)
        self.assertEqual(result.shape, (self.n,))

    def test_arf_only_larger_area_gives_larger_counts(self):
        from redback.spectral.folding import fold_spectrum
        arf_small = _make_uniform_arf(self.n, area=10.0)
        arf_large = _make_uniform_arf(self.n, area=1000.0)
        r_small = fold_spectrum(self.flat_flux, self.energy_edges, arf=arf_small)
        r_large = fold_spectrum(self.flat_flux, self.energy_edges, arf=arf_large)
        self.assertTrue(np.all(r_large > r_small))

    # --- fold_spectrum with rmf only ---

    def test_rmf_only_output_shape(self):
        from redback.spectral.folding import fold_spectrum
        rmf = _make_identity_rmf(self.n)
        result = fold_spectrum(self.flat_flux, self.energy_edges, rmf=rmf)
        self.assertEqual(result.shape, (self.n,))

    def test_rmf_only_identity_preserves_values(self):
        """Identity RMF: result must equal no-RMF result."""
        from redback.spectral.folding import fold_spectrum
        rmf = _make_identity_rmf(self.n)
        result_rmf = fold_spectrum(self.flat_flux, self.energy_edges, rmf=rmf)
        result_no_rmf = fold_spectrum(self.flat_flux, self.energy_edges)
        assert_allclose(result_rmf, result_no_rmf, rtol=1e-10)

    def test_rmf_only_redistribution_applied(self):
        """A matrix that sums all flux into channel 0 should produce large channel-0 count."""
        from redback.spectral.response import ResponseMatrix
        from redback.spectral.folding import fold_spectrum
        mat = np.zeros((self.n, self.n))
        mat[0, :] = 1.0
        e_min = self.energy_edges[:-1]
        e_max = self.energy_edges[1:]
        rmf = ResponseMatrix(e_min=e_min, e_max=e_max,
                             channel=np.arange(self.n, dtype=float),
                             emin_chan=e_min, emax_chan=e_max,
                             matrix=mat)
        result = fold_spectrum(self.flat_flux, self.energy_edges, rmf=rmf)
        self.assertGreater(result[0], 0.0)
        assert_allclose(result[1:], 0.0, atol=1e-20)

    # --- fold_spectrum with both rmf and arf ---

    def test_both_rmf_and_arf_output_shape(self):
        from redback.spectral.folding import fold_spectrum
        rmf = _make_identity_rmf(self.n)
        arf = _make_uniform_arf(self.n, area=300.0)
        result = fold_spectrum(self.flat_flux, self.energy_edges, rmf=rmf, arf=arf)
        self.assertEqual(result.shape, (self.n,))

    def test_both_rmf_and_arf_combined_effect(self):
        """
        Identity RMF + uniform ARF of 300 cm^2: result should be 300x
        the result with no ARF and no RMF.
        """
        from redback.spectral.folding import fold_spectrum
        rmf = _make_identity_rmf(self.n)
        area = 300.0
        arf = _make_uniform_arf(self.n, area=area)
        result_combined = fold_spectrum(self.flat_flux, self.energy_edges,
                                        rmf=rmf, arf=arf)
        result_bare = fold_spectrum(self.flat_flux, self.energy_edges)
        assert_allclose(result_combined, result_bare * area, rtol=1e-8)

    # --- exposure scaling ---

    def test_exposure_scaling_doubles_counts(self):
        """Doubling exposure must double the expected counts."""
        from redback.spectral.folding import fold_spectrum
        r1 = fold_spectrum(self.flat_flux, self.energy_edges, exposure=1.0)
        r2 = fold_spectrum(self.flat_flux, self.energy_edges, exposure=2.0)
        assert_allclose(r2, r1 * 2.0, rtol=1e-10)

    def test_exposure_scaling_with_arf(self):
        from redback.spectral.folding import fold_spectrum
        arf = _make_uniform_arf(self.n, area=100.0)
        r1 = fold_spectrum(self.flat_flux, self.energy_edges, arf=arf, exposure=10.0)
        r2 = fold_spectrum(self.flat_flux, self.energy_edges, arf=arf, exposure=20.0)
        assert_allclose(r2, r1 * 2.0, rtol=1e-10)

    def test_areascal_scaling(self):
        from redback.spectral.folding import fold_spectrum
        r1 = fold_spectrum(self.flat_flux, self.energy_edges, areascal=1.0)
        r2 = fold_spectrum(self.flat_flux, self.energy_edges, areascal=0.5)
        assert_allclose(r2, r1 * 0.5, rtol=1e-10)

    # --- _integrate_photon_flux_per_bin ---

    def test_integrate_flat_spectrum_approx_flux_times_width(self):
        """
        For a perfectly flat photon-flux spectrum, the integral over each bin
        should be very close to flux * bin_width.
        """
        from redback.spectral.folding import _integrate_photon_flux_per_bin
        flux_val = 5.0
        flat_flux = np.full(self.n, flux_val)
        result = _integrate_photon_flux_per_bin(flat_flux, self.energy_edges)
        # Each bin has width 1 keV; result should be ~flux_val * 1.0
        widths = self.energy_edges[1:] - self.energy_edges[:-1]
        expected = flat_flux * widths
        assert_allclose(result, expected, rtol=1e-8)

    def test_integrate_output_shape(self):
        from redback.spectral.folding import _integrate_photon_flux_per_bin
        from redback.spectral.conversions import mjy_to_photon_flux_per_keV
        photon_flux = mjy_to_photon_flux_per_keV(self.flat_flux, self.energy_centers)
        result = _integrate_photon_flux_per_bin(photon_flux, self.energy_edges)
        self.assertEqual(result.shape, (self.n,))

    def test_integrate_positive_for_positive_flux(self):
        from redback.spectral.folding import _integrate_photon_flux_per_bin
        flux = np.linspace(1.0, 10.0, self.n)
        result = _integrate_photon_flux_per_bin(flux, self.energy_edges)
        self.assertTrue(np.all(result > 0))

    def test_integrate_uses_log_log_path(self):
        """If all flux values are positive, the log-log branch is taken; result is positive."""
        from redback.spectral.folding import _integrate_photon_flux_per_bin
        flux = np.array([10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.5, 1.0])
        result = _integrate_photon_flux_per_bin(flux, self.energy_edges)
        self.assertTrue(np.all(result > 0))

    def test_integrate_fallback_path_for_nonpositive_flux(self):
        """A flux array with a zero entry must trigger the linear-interp fallback without error."""
        from redback.spectral.folding import _integrate_photon_flux_per_bin
        flux = np.array([1.0, 0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        result = _integrate_photon_flux_per_bin(flux, self.energy_edges)
        self.assertEqual(result.shape, (self.n,))
        self.assertTrue(np.all(np.isfinite(result)))


# ===========================================================================
# 5.  TestHighEnergySpectralModels
# ===========================================================================

class TestHighEnergySpectralModels(unittest.TestCase):
    """Tests for the high-energy spectral models added in this branch."""

    def setUp(self):
        from redback.transient_models import spectral_models
        self.sm = spectral_models
        # 20 log-spaced energies from 0.3 to 100 keV
        self.energies = np.logspace(np.log10(0.3), np.log10(100.0), 20)
        self.log10_norm = 0.0      # 1 photon/cm^2/s/keV at 100 keV
        self.alpha = -1.7          # typical power-law index
        self.redshift = 0.1

    def tearDown(self):
        del self.sm
        del self.energies

    # ------------------------------------------------------------------
    # powerlaw_high_energy
    # ------------------------------------------------------------------

    def test_powerlaw_output_shape(self):
        result = self.sm.powerlaw_high_energy(
            self.energies, self.log10_norm, self.alpha)
        self.assertEqual(result.shape, self.energies.shape)

    def test_powerlaw_all_positive(self):
        result = self.sm.powerlaw_high_energy(
            self.energies, self.log10_norm, self.alpha)
        self.assertTrue(np.all(result > 0))

    def test_powerlaw_spectral_slope_in_loglog(self):
        """
        In log-log space the output flux density should be a straight line
        with slope = alpha + 1 (because the model converts photon flux E^alpha
        to energy flux E^(alpha+1), then to mJy via constant factor).
        """
        result = self.sm.powerlaw_high_energy(
            self.energies, self.log10_norm, self.alpha, redshift=0.0)
        log_e = np.log10(self.energies)
        log_f = np.log10(result)
        slope = np.polyfit(log_e, log_f, 1)[0]
        expected_slope = self.alpha + 1.0
        self.assertAlmostEqual(slope, expected_slope, delta=0.01)

    def test_powerlaw_redshift_zero_consistent(self):
        """redshift=0 must give the same result as no redshift kwarg."""
        result_z0 = self.sm.powerlaw_high_energy(
            self.energies, self.log10_norm, self.alpha, redshift=0.0)
        result_default = self.sm.powerlaw_high_energy(
            self.energies, self.log10_norm, self.alpha)
        assert_allclose(result_z0, result_default, rtol=1e-12)

    def test_powerlaw_norm_scaling(self):
        """Increasing log10_norm by 1 (factor 10 in norm) must scale flux by 10."""
        r1 = self.sm.powerlaw_high_energy(
            self.energies, 0.0, self.alpha)
        r2 = self.sm.powerlaw_high_energy(
            self.energies, 1.0, self.alpha)
        assert_allclose(r2 / r1, 10.0, rtol=1e-8)

    def test_powerlaw_finite(self):
        result = self.sm.powerlaw_high_energy(
            self.energies, self.log10_norm, self.alpha)
        self.assertTrue(np.all(np.isfinite(result)))

    # ------------------------------------------------------------------
    # tbabs_powerlaw_high_energy
    # ------------------------------------------------------------------

    def test_tbabs_powerlaw_nh_zero_equals_unabsorbed(self):
        """
        When nh=0 the tbabs transmission is exp(0)=1, so the absorbed
        model must equal the unabsorbed power-law.
        The tbabs table may not be present in CI; skip gracefully.
        """
        try:
            result_abs = self.sm.tbabs_powerlaw_high_energy(
                self.energies, self.log10_norm, self.alpha, nh=0.0)
        except FileNotFoundError:
            self.skipTest("TBabs cross-section table not available.")
        result_pl = self.sm.powerlaw_high_energy(
            self.energies, self.log10_norm, self.alpha)
        assert_allclose(result_abs, result_pl, rtol=1e-8)

    def test_tbabs_powerlaw_output_shape(self):
        try:
            result = self.sm.tbabs_powerlaw_high_energy(
                self.energies, self.log10_norm, self.alpha, nh=0.1)
        except FileNotFoundError:
            self.skipTest("TBabs cross-section table not available.")
        self.assertEqual(result.shape, self.energies.shape)

    def test_tbabs_powerlaw_positive(self):
        try:
            result = self.sm.tbabs_powerlaw_high_energy(
                self.energies, self.log10_norm, self.alpha, nh=0.05)
        except FileNotFoundError:
            self.skipTest("TBabs cross-section table not available.")
        self.assertTrue(np.all(result >= 0))

    def test_tbabs_powerlaw_large_nh_suppresses_soft_more_than_hard(self):
        """
        Large NH suppresses soft X-rays more than hard X-rays.
        The ratio absorbed/unabsorbed at 0.5 keV must be smaller than at 10 keV.
        """
        try:
            energies_soft = np.array([0.5])
            energies_hard = np.array([10.0])
            pl_soft = self.sm.powerlaw_high_energy(
                energies_soft, self.log10_norm, self.alpha)
            pl_hard = self.sm.powerlaw_high_energy(
                energies_hard, self.log10_norm, self.alpha)
            abs_soft = self.sm.tbabs_powerlaw_high_energy(
                energies_soft, self.log10_norm, self.alpha, nh=1.0)
            abs_hard = self.sm.tbabs_powerlaw_high_energy(
                energies_hard, self.log10_norm, self.alpha, nh=1.0)
        except FileNotFoundError:
            self.skipTest("TBabs cross-section table not available.")
        ratio_soft = (abs_soft / pl_soft)[0]
        ratio_hard = (abs_hard / pl_hard)[0]
        self.assertLess(ratio_soft, ratio_hard)

    def test_tbabs_powerlaw_monotone_suppression_with_nh(self):
        """Greater NH must lead to smaller or equal flux at soft energies."""
        soft = np.array([0.5])
        try:
            r_low = self.sm.tbabs_powerlaw_high_energy(
                soft, self.log10_norm, self.alpha, nh=0.1)
            r_high = self.sm.tbabs_powerlaw_high_energy(
                soft, self.log10_norm, self.alpha, nh=2.0)
        except FileNotFoundError:
            self.skipTest("TBabs cross-section table not available.")
        self.assertLessEqual(r_high[0], r_low[0])

    # ------------------------------------------------------------------
    # cutoff_powerlaw_high_energy
    # ------------------------------------------------------------------

    def test_cutoff_powerlaw_output_shape(self):
        result = self.sm.cutoff_powerlaw_high_energy(
            self.energies, self.log10_norm, self.alpha, e_cut=10.0)
        self.assertEqual(result.shape, self.energies.shape)

    def test_cutoff_powerlaw_all_positive(self):
        result = self.sm.cutoff_powerlaw_high_energy(
            self.energies, self.log10_norm, self.alpha, e_cut=10.0)
        self.assertTrue(np.all(result > 0))

    def test_cutoff_powerlaw_finite(self):
        result = self.sm.cutoff_powerlaw_high_energy(
            self.energies, self.log10_norm, self.alpha, e_cut=10.0)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_cutoff_powerlaw_flux_drops_above_ecut(self):
        """
        Flux at energies well above e_cut must be less than for a model
        with a very large e_cut (approximating an uncut power law).
        """
        e_cut = 5.0
        high_energies = np.array([50.0, 80.0, 100.0])
        result_cut = self.sm.cutoff_powerlaw_high_energy(
            high_energies, self.log10_norm, self.alpha, e_cut=e_cut)
        result_nocut = self.sm.cutoff_powerlaw_high_energy(
            high_energies, self.log10_norm, self.alpha, e_cut=1e6)
        self.assertTrue(np.all(result_cut < result_nocut))

    def test_cutoff_powerlaw_large_ecut_approaches_powerlaw(self):
        """Very large e_cut should reproduce the uncut power law to good accuracy."""
        result_cut = self.sm.cutoff_powerlaw_high_energy(
            self.energies, self.log10_norm, self.alpha, e_cut=1e8)
        result_pl = self.sm.powerlaw_high_energy(
            self.energies, self.log10_norm, self.alpha)
        assert_allclose(result_cut, result_pl, rtol=1e-4)

    # ------------------------------------------------------------------
    # comptonized_high_energy
    # ------------------------------------------------------------------

    def test_comptonized_output_shape(self):
        result = self.sm.comptonized_high_energy(
            self.energies, self.log10_norm, self.alpha, e_peak=100.0)
        self.assertEqual(result.shape, self.energies.shape)

    def test_comptonized_all_positive(self):
        result = self.sm.comptonized_high_energy(
            self.energies, self.log10_norm, self.alpha, e_peak=100.0)
        self.assertTrue(np.all(result > 0))

    def test_comptonized_finite(self):
        result = self.sm.comptonized_high_energy(
            self.energies, self.log10_norm, self.alpha, e_peak=100.0)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_comptonized_ecut_relationship(self):
        """
        The Comptonized model uses e_cut = e_peak / (2 + alpha).
        A manually constructed cutoff_powerlaw with that e_cut should
        match the comptonized model exactly.
        """
        e_peak = 200.0
        alpha = -1.0
        e_cut = e_peak / (2.0 + alpha)
        result_comp = self.sm.comptonized_high_energy(
            self.energies, self.log10_norm, alpha, e_peak=e_peak)
        result_cpl = self.sm.cutoff_powerlaw_high_energy(
            self.energies, self.log10_norm, alpha, e_cut=e_cut)
        assert_allclose(result_comp, result_cpl, rtol=1e-10)

    def test_comptonized_higher_epeak_shifts_spectrum_harder(self):
        """
        Higher e_peak should give more flux at high energies relative to low energies,
        i.e. the ratio (high energy flux) / (low energy flux) is larger.
        """
        alpha = -1.0
        energies_lo = np.array([1.0])
        energies_hi = np.array([50.0])
        r_lo_low_ep = self.sm.comptonized_high_energy(
            energies_lo, self.log10_norm, alpha, e_peak=10.0)
        r_hi_low_ep = self.sm.comptonized_high_energy(
            energies_hi, self.log10_norm, alpha, e_peak=10.0)
        r_lo_high_ep = self.sm.comptonized_high_energy(
            energies_lo, self.log10_norm, alpha, e_peak=200.0)
        r_hi_high_ep = self.sm.comptonized_high_energy(
            energies_hi, self.log10_norm, alpha, e_peak=200.0)
        ratio_low_ep = (r_hi_low_ep / r_lo_low_ep)[0]
        ratio_high_ep = (r_hi_high_ep / r_lo_high_ep)[0]
        self.assertGreater(ratio_high_ep, ratio_low_ep)

    # ------------------------------------------------------------------
    # blackbody_high_energy
    # ------------------------------------------------------------------

    def test_blackbody_output_shape(self):
        result = self.sm.blackbody_high_energy(
            self.energies, redshift=self.redshift,
            r_photosphere_rs=1e6, kT=1.0)
        self.assertEqual(result.shape, self.energies.shape)

    def test_blackbody_all_positive(self):
        result = self.sm.blackbody_high_energy(
            self.energies, redshift=self.redshift,
            r_photosphere_rs=1e6, kT=1.0)
        self.assertTrue(np.all(result > 0))

    def test_blackbody_finite(self):
        result = self.sm.blackbody_high_energy(
            self.energies, redshift=self.redshift,
            r_photosphere_rs=1e6, kT=1.0)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_blackbody_hotter_kT_shifts_peak_to_higher_energies(self):
        """
        For a hotter blackbody the peak of E*F_E (the nu*F_nu proxy in keV)
        must shift to higher energy.  We use a dense energy grid to find the peak.
        """
        fine_energies = np.logspace(-1, 2, 200)
        r_phot = 1e7
        z = 0.01
        result_cool = self.sm.blackbody_high_energy(
            fine_energies, redshift=z, r_photosphere_rs=r_phot, kT=0.3)
        result_hot = self.sm.blackbody_high_energy(
            fine_energies, redshift=z, r_photosphere_rs=r_phot, kT=3.0)
        peak_cool = fine_energies[np.argmax(fine_energies * result_cool)]
        peak_hot = fine_energies[np.argmax(fine_energies * result_hot)]
        self.assertGreater(peak_hot, peak_cool)

    def test_blackbody_larger_radius_gives_larger_flux(self):
        """Larger photosphere radius must produce larger flux at all energies."""
        r_small = 1e5
        r_large = 1e8
        result_small = self.sm.blackbody_high_energy(
            self.energies, redshift=self.redshift,
            r_photosphere_rs=r_small, kT=1.0)
        result_large = self.sm.blackbody_high_energy(
            self.energies, redshift=self.redshift,
            r_photosphere_rs=r_large, kT=1.0)
        self.assertTrue(np.all(result_large > result_small))

    # ------------------------------------------------------------------
    # Model registration and citation
    # ------------------------------------------------------------------

    def test_all_models_registered_in_all_models_dict(self):
        """All five high-energy models must appear in the global model registry."""
        from redback.model_library import all_models_dict
        model_names = [
            "powerlaw_high_energy",
            "tbabs_powerlaw_high_energy",
            "cutoff_powerlaw_high_energy",
            "comptonized_high_energy",
            "blackbody_high_energy",
        ]
        for name in model_names:
            with self.subTest(model=name):
                self.assertIn(name, all_models_dict,
                              msg=f"'{name}' not found in all_models_dict")

    def test_all_models_have_citation_attribute(self):
        """Each model must carry a citation URL via the @citation_wrapper decorator."""
        model_names = [
            "powerlaw_high_energy",
            "tbabs_powerlaw_high_energy",
            "cutoff_powerlaw_high_energy",
            "comptonized_high_energy",
            "blackbody_high_energy",
        ]
        for name in model_names:
            with self.subTest(model=name):
                func = getattr(self.sm, name)
                self.assertTrue(
                    hasattr(func, "citation"),
                    msg=f"'{name}' has no .citation attribute",
                )
                self.assertIsInstance(func.citation, str)
                self.assertGreater(len(func.citation), 0)

    def test_powerlaw_citation_is_url(self):
        self.assertIn("http", self.sm.powerlaw_high_energy.citation)

    def test_cutoff_powerlaw_citation_is_url(self):
        self.assertIn("http", self.sm.cutoff_powerlaw_high_energy.citation)

    def test_comptonized_citation_is_url(self):
        self.assertIn("http", self.sm.comptonized_high_energy.citation)

    def test_blackbody_citation_is_url(self):
        self.assertIn("http", self.sm.blackbody_high_energy.citation)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    unittest.main()
