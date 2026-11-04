"""Cluster counts with the survey area as a function of redshift.

A volume-limited cluster sample loses the parts of the footprint that
are too shallow to detect clusters at redshift z (the DES redMaPPer
zmask), so its effective area falls at high z. The optional dataset key
cluster_area_file hands cosmolike a table of area(z) in deg^2
(set_cluster_area), and the counts (eq. 16 of arXiv 2503.13631)
integrate it:

    N_iA = int dz Omega_s(z) dV/dz <phi_i|z> n_A(z).

Without the key, Omega_s is the constant survey area at every z (the
forecast setting). Only the counts read the table: every two-point
block keeps its volume-weighted kernels.

The checks, on 4x2pt + N at the fiducial point of arXiv 2503.13631
(reference_cluster.FIDUCIAL), all entries computed (all-ones mask):
  1. An empty table leaves the data vector exactly as the constant-area
     default computes it.
  2. A table equal to the survey area at every z gives the default
     counts to 1e-12 (the area multiplies inside the z integral instead
     of after it: different rounding), and every two-point entry
     exactly unchanged.
  3. A table at half the survey area halves every count (1e-12).
  4. A smooth taper (full area at low z, falling around z = 0.6, the
     shape of the Cardinal footprint) scales each count by the fraction
     of its integrand weighted by area(z)/area(0). That fraction is
     computed here independently, in numpy, from the library's own
     ingredients: n_A(z) (ncl_richness), <phi_i|z> (phi_cluster) and the
     comoving distance of the cosmology tables, on a 4000-node
     Gauss-Legendre rule (1e-4: the two quadratures).
  5. Removing the table restores the default data vector exactly (the
     counts refill on the cache key cluster.random_area).
  6. A table that does not cover a cluster bin's support stops the
     process with a message (no extrapolation), checked in a forked
     child process.

The test feeds cosmolike without cobaya, through the likelihood's init
chain in tests/validation/compare_reference.py (init_c) and one CAMB
run of the Python reference; it reads the live project files, not the
frozen copies. The body runs in a python process of its own
(@own_process), which keeps this file's C state out of pytest's
process, where test_cache_consistency.py builds its cobaya models.

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):
    python -m pytest projects/des_cluster/tests/data_vector/test_cluster_area.py
"""

import os

# OpenMP reads OMP_NUM_THREADS when the compiled libraries load: set it
# before any cosmolike import (4 = the thread count of every test)
os.environ["OMP_NUM_THREADS"] = "4"

import sys
import tempfile
import unittest

# tests/ for the shim, validation/ for the init chain, reference/ for the
# Python reference, and the core folder for the shared abort check
TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TESTS_DIR)
sys.path.insert(0, os.path.join(TESTS_DIR, "validation"))
sys.path.insert(0, os.path.join(TESTS_DIR, "reference"))
sys.path.insert(0, os.path.abspath(os.path.join(
    TESTS_DIR, "..", "..", "..", "external_modules", "code", "cosmolike_core")))
import cocoa_test_utils as u
from cocoa_testing import assert_aborts

# Checks 2-3: the area enters inside the z sum instead of multiplying the
# finished sum, so the counts agree to rounding, not bit by bit.
REORDER_RTOL = 1.0e-12
# Check 4: the C counts (the Limber support rule of cosmo2D_cluster.c)
# against a 4000-node Gauss-Legendre integral in numpy.
TAPER_RTOL = 1.0e-4
# The taper of check 4: area(z) = A0 / (1 + exp((z - TAPER_Z)/TAPER_W)),
# full area at low z, half at TAPER_Z; it cuts into the 0.55-0.65 bin
# and barely touches the 0.2-0.4 one, like the Cardinal footprint.
TAPER_Z = 0.62
TAPER_W = 0.03


def build_state():
    """Initialize cosmolike for 4x2pt + N at the fiducial point.

    The likelihood's init chain (compare_reference.init_c) with dummy
    data and covariance, an all-ones mask so every entry is computed,
    the fiducial nuisance parameters, and the cosmology tables of one
    CAMB run of the Python reference.

    Returns:
      (ci, R, area0): the compiled interface, the cosmology tables (keys
      cin_*, among them cin_z_1D and cin_chi, the comoving distance in
      Mpc/h) and the dataset's survey area in deg^2.
    """
    import numpy as np
    import compare_reference as cr
    from reference_cluster import FIDUCIAL, COSMO_KEYS
    from ref_cosmology import Cosmology

    cfg = cr.load_config()
    workdir = tempfile.mkdtemp(prefix="des_cluster_area_")
    ci, info = cr.init_c(cfg, workdir, threads=4)
    ones = np.ones(info["ndata"], dtype=int)
    ci.init_data_cluster(info["covf"], cr.write_mask(workdir, "ones.mask", ones),
                         info["dvf"])
    cr.set_nuisance_c(ci, cfg, dict(FIDUCIAL))
    params = {key: FIDUCIAL[key] for key in COSMO_KEYS}
    R = cr.cosmology_inputs(Cosmology(params), cfg, "production")
    cr.set_cosmology_c(ci, R)
    return ci, R, float(cfg["dataset"]["survey_area_deg2"])


def evaluate(ci):
    """Return (data vector, counts [cluster z bin, richness bin])."""
    import numpy as np
    dv = np.array(ci.compute_data_vector_cluster_masked())
    counts = np.array(ci.N_cluster_tomo()).T
    return dv, counts


def counts_slice(ci):
    """Index range of the counts block in the joint vector (block 4: N)."""
    sizes = ci.compute_data_vector_cluster_sizes()
    starts = ci.compute_data_vector_cluster_starts()
    return slice(int(starts[4]), int(starts[4] + sizes[4]))


def taper_fraction(ci, R, area_fn):
    """Fraction of each count's integrand weighted by area_fn(z)/area_fn(0).

    N_iA ~ int dz w(z) area(z) with w = chi^2 dchi/dz <phi_i|z> n_A(z); the
    normalization of w cancels in the fraction, so chi may stay in Mpc/h.

    Returns:
      array [cluster z bin, richness bin].
    """
    import numpy as np
    from scipy.interpolate import CubicSpline

    chi = CubicSpline(R["cin_z_1D"], R["cin_chi"])
    x, wx = np.polynomial.legendre.leggauss(4000)
    nbin = len(np.array(ci.N_cluster_tomo())[0])
    nrich = len(np.array(ci.N_cluster_tomo()))
    out = np.zeros((nbin, nrich))
    for ni in range(nbin):
        z0 = max(1.0 / ci.amax_cluster(ni) - 1.0, 1.0e-5)
        z1 = 1.0 / ci.amin_cluster(ni) - 1.0
        z = 0.5 * (z1 - z0) * x + 0.5 * (z1 + z0)
        w = 0.5 * (z1 - z0) * wx * chi(z) ** 2 * chi(z, 1)
        w *= np.array([ci.phi_cluster(float(zz), ni) for zz in z])
        for nl in range(nrich):
            n = np.array([ci.ncl_richness(float(1.0 / (1.0 + zz)), nl) for zz in z])
            out[ni, nl] = np.sum(w * n * area_fn(z)) / (area_fn(0.0) * np.sum(w * n))
    return out


class TestClusterArea(unittest.TestCase):
    """The optional area(z) table of the cluster counts (checks 1-6)."""

    @classmethod
    def setUpClass(cls):
        """Refuse to run without start_cocoa.sh (ROOTDIR)."""
        u.require_cocoa_environment()

    @u.own_process(__file__)
    def test_cluster_area(self):
        """Run checks 1-6 of the module docstring on one cosmolike state."""
        import numpy as np

        ci, R, area0 = build_state()
        nslice = counts_slice(ci)
        others = np.ones(int(np.array(ci.compute_data_vector_cluster_sizes()).sum()), bool)
        others[nslice] = False
        z = np.linspace(0.0, 3.0, 3001)

        # --- 1. EMPTY TABLE = DEFAULT ---
        dv0, N0 = evaluate(ci)
        ci.set_cluster_area(area=np.zeros((0, 2)))
        dv, N = evaluate(ci)
        self.assertTrue(np.array_equal(dv, dv0), "empty table changed the vector")

        # --- 2. CONSTANT TABLE = SURVEY AREA ---
        ci.set_cluster_area(area=np.column_stack([z, np.full(z.size, area0)]))
        dv, N = evaluate(ci)
        print(f"  constant table: max |N/N0 - 1| = {np.abs(N / N0 - 1).max():.1e}")
        np.testing.assert_allclose(N, N0, rtol=REORDER_RTOL, atol=0.0)
        self.assertTrue(np.array_equal(dv[others], dv0[others]),
                        "a two-point entry moved with the area table")

        # --- 3. HALF THE AREA HALVES THE COUNTS ---
        ci.set_cluster_area(area=np.column_stack([z, np.full(z.size, 0.5 * area0)]))
        dv, N = evaluate(ci)
        np.testing.assert_allclose(N, 0.5 * N0, rtol=REORDER_RTOL, atol=0.0)

        # --- 4. SMOOTH TAPER AGAINST AN INDEPENDENT INTEGRAL ---
        def taper(zz):
            return area0 / (1.0 + np.exp((zz - TAPER_Z) / TAPER_W))
        ci.set_cluster_area(area=np.column_stack([z, taper(z)]))
        dv, N = evaluate(ci)
        expected = taper_fraction(ci, R, taper)
        print("  taper: N/N0 =", np.array2string(N / N0, precision=5),
              "\n  numpy fraction =", np.array2string(expected, precision=5))
        self.assertLess(float(np.min(N / N0)), 0.9, "the taper did not reach the counts")
        np.testing.assert_allclose(N / N0, expected, rtol=TAPER_RTOL, atol=0.0)
        self.assertTrue(np.array_equal(dv[others], dv0[others]),
                        "a two-point entry moved with the area table")

        # --- 5. REMOVING THE TABLE RESTORES THE DEFAULT ---
        ci.set_cluster_area(area=np.zeros((0, 2)))
        dv, N = evaluate(ci)
        self.assertTrue(np.array_equal(dv, dv0), "the default did not come back")

        # --- 6. A TABLE THAT MISSES THE SUPPORT STOPS THE PROCESS ---
        # The table starts at z = 0.3, inside the support of the lowest
        # cluster bin, so the counts must stop at their first node there.
        # The forked child inherits this process's state; the counts fill
        # runs serially and stops before any OpenMP region starts.
        z_short = np.linspace(0.3, 3.0, 300)
        short = np.column_stack([z_short, np.full(z_short.size, area0)])
        def uncovered():
            ci.set_cluster_area(area=short)
            ci.N_cluster_tomo()
        assert_aborts(uncovered)


if __name__ == "__main__":
    unittest.main()
