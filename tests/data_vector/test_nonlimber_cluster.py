"""Non-Limber cluster clustering: w_cc and w_cg with redshift-space distortions.

Below limits.LMAX_NOLIMBER = 150 the Limber approximation misses two
effects that matter for the thin cluster redshift bins on large angular
scales: the exact projection (a spherical Bessel function in each leg
instead of Limber's single wavenumber) and the redshift-space
distortions (the Kaiser term of every leg). With cluster_adopt_limber_cc
= 0 and cluster_adopt_limber_cg = 0 the library computes w_cc and w_cg
from the FFTLog projection of the radial rows of every (cluster bin,
richness bin) and lens bin, combined with the Limber spectra through
the FKEM split (cosmo2D_cluster.c, section NON-LIMBER; the C_cl_tomo
design of cosmo2D.c). Against lighthouse at the Cardinal point the two
blocks go from a diagonal-covariance Delta chi2 of 13.9 (w_cg) and 2.2
(w_cc) in Limber to 0.09 and 0.13.

The checks, on 4x2pt + N at the fiducial point of arXiv 2503.13631
(reference_cluster.FIDUCIAL), all entries computed (all-ones mask):
  1. The switch moves only the w_cg and w_cc blocks: every other entry
     stays bitwise.
  2. At four angular bins (8, 12, 16, 18 of the 2.5-250 arcmin grid:
     about 11, 28, 70 and 177 arcmin) of three rows of each block, the
     ratio non-Limber/Limber - 1 matches the values MEASURED below to 5%
     of each value (plus 1e-4, for the entries where the ratio crosses
     zero). The rows span the redshift bins and richness: w_cg rows 0,
     7, 11 = (cg pair 0, richness 0), (pair 1, richness 3), (pair 2,
     richness 3); w_cc rows 0, 14, 29 = (bin 0, pair (0,0)), (bin 1,
     pair (1,1)), (bin 2, pair (3,3)).
  3. At 177 arcmin every measured ratio is positive: the Kaiser term
     raises the correlation of thin bins on large scales.
  4. The non-Limber blocks are bitwise the same at 1, 8 and 4 OpenMP
     threads.
  5. Returning to Limber restores the Limber vector exactly (the real-
     space caches key on the switch).

MEASURED was recorded with this test's setup on 2026-11-04 (cosmolike_core
bugfix, des_cluster bugfix, macOS arm64). The 5% band is far wider than
the numerical noise (boost 1 vs 2 moves these blocks by Delta chi2
<= 0.003) and far narrower than the effect (up to +67% at 177 arcmin).
Re-measure, never copy, when the model of the rows changes on purpose.

The test feeds cosmolike without cobaya, through the likelihood's init
chain (tests/validation/compare_reference.py), in a python process of its
own (@own_process); see test_cluster_area.py for that setup.

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):
    python -m pytest projects/des_cluster/tests/data_vector/test_nonlimber_cluster.py
"""

import os

# OpenMP reads OMP_NUM_THREADS when the compiled libraries load: set it
# before any cosmolike import (4 = the thread count of every test)
os.environ["OMP_NUM_THREADS"] = "4"

import sys
import tempfile
import unittest

TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TESTS_DIR)
sys.path.insert(0, os.path.join(TESTS_DIR, "validation"))
sys.path.insert(0, os.path.join(TESTS_DIR, "reference"))
import cocoa_test_utils as u

# angular bins of check 2 (20 log bins over 2.5-250 arcmin)
THETA_BINS = (8, 12, 16, 18)
# non-Limber/Limber - 1 at THETA_BINS, per (block, row)
MEASURED = {
    ("w_cg", 0):  (0.00474, 0.01049, 0.03593, 0.08820),
    ("w_cg", 7):  (0.00600, 0.01543, 0.07493, 0.26429),
    ("w_cg", 11): (0.00924, 0.02647, 0.16257, 0.67301),
    ("w_cc", 0):  (0.00348, 0.00786, 0.02943, 0.08132),
    ("w_cc", 14): (0.00362, 0.00972, 0.05808, 0.25560),
    ("w_cc", 29): (-0.00278, -0.00197, 0.09257, 0.50187),
}
RTOL = 0.05
ATOL = 1.0e-4
# block positions in compute_data_vector_cluster_sizes/starts
BLOCK_INDEX = {"w_cg": 3, "w_cc": 5}
NTHETA = 20


def build_state():
    """Initialize cosmolike for 4x2pt + N at the fiducial point.

    The init chain of test_cluster_area.py: the likelihood's
    (compare_reference.init_c, which reads the switch from the likelihood
    yaml; vector() then sets it explicitly), an all-ones mask, the
    fiducial nuisance parameters and one CAMB run of the Python
    reference.

    Returns:
      the compiled interface.
    """
    import numpy as np
    import compare_reference as cr
    from reference_cluster import FIDUCIAL, COSMO_KEYS
    from ref_cosmology import Cosmology

    cfg = cr.load_config()
    workdir = tempfile.mkdtemp(prefix="des_cluster_nonlimber_")
    ci, info = cr.init_c(cfg, workdir, threads=4)
    ones = np.ones(info["ndata"], dtype=int)
    ci.init_data_cluster(info["covf"], cr.write_mask(workdir, "ones.mask", ones),
                         info["dvf"])
    cr.set_nuisance_c(ci, cfg, dict(FIDUCIAL))
    params = {key: FIDUCIAL[key] for key in COSMO_KEYS}
    cr.set_cosmology_c(ci, cr.cosmology_inputs(Cosmology(params), cfg, "production"))
    return ci


def vector(ci, limber):
    """The full data vector with w_cc and w_cg in Limber (1) or not (0)."""
    import numpy as np
    ci.init_cluster_adopt_limber(adopt_limber_cc=limber, adopt_limber_cg=limber)
    return np.array(ci.compute_data_vector_cluster_masked())


class TestNonLimberCluster(unittest.TestCase):
    """Non-Limber w_cc and w_cg (checks 1-5)."""

    @classmethod
    def setUpClass(cls):
        """Refuse to run without start_cocoa.sh (ROOTDIR)."""
        u.require_cocoa_environment()

    @u.own_process(__file__)
    def test_nonlimber_cluster(self):
        """Run checks 1-5 of the module docstring on one cosmolike state."""
        import numpy as np

        ci = build_state()
        sizes = ci.compute_data_vector_cluster_sizes()
        starts = ci.compute_data_vector_cluster_starts()
        moved = np.zeros(int(np.array(sizes).sum()), bool)
        for b in BLOCK_INDEX.values():
            moved[int(starts[b]):int(starts[b] + sizes[b])] = True

        dv_lim = vector(ci, 1)
        dv_nl = vector(ci, 0)

        # --- 1. ONLY w_cg AND w_cc MOVE ---
        self.assertTrue(np.array_equal(dv_nl[~moved], dv_lim[~moved]),
                        "the switch moved an entry outside w_cg and w_cc")

        # --- 2. AND 3. THE MEASURED RATIOS ---
        for (block, row), values in MEASURED.items():
            s = int(starts[BLOCK_INDEX[block]]) + NTHETA * row
            ratio = np.array([dv_nl[s + t] / dv_lim[s + t] - 1 for t in THETA_BINS])
            print(f"  {block} row {row:2d}: " + " ".join(f"{r:+.5f}" for r in ratio))
            np.testing.assert_allclose(ratio, values, rtol=RTOL, atol=ATOL,
                                       err_msg=f"{block} row {row}")
            self.assertGreater(ratio[-1], 0.0, f"{block} row {row}: no RSD boost")

        # --- 4. THE SAME AT 1, 8 AND 4 THREADS ---
        for threads in (1, 8, 4):
            ci.set_omp_threads(threads)
            vector(ci, 1)  # back to Limber, so the next call refills
            dv = vector(ci, 0)
            self.assertTrue(np.array_equal(dv, dv_nl),
                            f"non-Limber blocks differ at {threads} threads")

        # --- 5. BACK TO LIMBER ---
        self.assertTrue(np.array_equal(vector(ci, 1), dv_lim),
                        "the Limber vector did not come back")


if __name__ == "__main__":
    unittest.main()
