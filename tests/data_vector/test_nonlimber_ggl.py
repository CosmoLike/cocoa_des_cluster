"""Unit test: non-Limber galaxy-galaxy lensing (adopt_limber_gs).

The galaxy-galaxy lensing (ggl) spectrum C_l^gs enters the data vector
through gamma_t(theta). The likelihood yaml key adopt_limber_gs
chooses how it is computed:

  adopt_limber_gs: 1 (the default) - Limber approximation at every
      multipole.
  adopt_limber_gs: 0 - below l = 150 the exact projection, computed by
      cosmolike's C_gs_tomo with the split of Fang, Krause, Eifler &
      MacCrann (arXiv:1911.11947): an FFTLog integral of the linear
      power spectrum plus, in Limber, what linear theory misses.

ggl defaults to Limber because its lensing kernel is broad (galaxy
clustering has its own key, adopt_limber_gg; see
test_nonlimber_gg.py). The Limber approximation fails at low l for
the lens-source pairs whose kernels overlap in redshift (lens bin =
source bin, or the source bin in front of the lens bin, where the
signal is the intrinsic alignment of the sources times the lens
density). This test measures what the Limber default costs. The
gamma_t block of 6x2pt + N reads the same key, and no other test of
this project flips it.

It evaluates the frozen 3x2pt fiducial (example4, NLA) three times IN
ONE PROCESS: Limber, non-Limber, Limber again, and computes

    delta chi2 = delta^T C^-1 delta,
    delta = dv(non-Limber) - dv(Limber),

with C^-1 the masked inverse covariance: the chi2 the Limber model
would score against a data set generated with non-Limber ggl. It prints
the total and the contribution of each lens-source pair (the pair's own
block of delta, cross-covariance with other pairs ignored).

Assertions:
  1. delta chi2 is above a dead-flag floor: the flag reaches the C code
     and the ggl cache notices the change (a stale cache gives zero);
  2. only ggl entries change: cosmic shear and clustering are bitwise
     equal between the two evaluations;
  3. switching back to Limber reproduces the first data vector bitwise;
  4. delta chi2 matches the value measured for this project
     (DCHI2_MEASURED below) to 5%: a change in the non-Limber code, the
     kernels, or the covariance shows up here (skipped while no value
     is recorded);
  5. the Limber evaluation reproduces the frozen reference chi2
     (checked last, so a stale snapshot cannot hide checks 1-4).

The three evaluations run in a python process of their own
(cocoa_test_utils.own_process): under pytest the collecting process
also holds the cluster ladder's models, and cosmolike aborts a
process that initializes two data sets of different dimensions.

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/des_cluster/tests/data_vector/test_nonlimber_ggl.py
"""

import os

# OpenMP reads OMP_NUM_THREADS when the compiled libraries load, so
# this must run before ANY cobaya/cosmolike import in the process.
os.environ["OMP_NUM_THREADS"] = "4"

import sys
import time
import unittest

# The harness stays in the parent tests/ folder. Add it explicitly so
# direct execution and worker processes resolve this project's stored inputs.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cocoa_test_utils as u

EXAMPLE = "example4"
REFERENCE_KEY = "example4_nla"

# (report tag, adopt_limber_gs)
SETTINGS = (
    ("Limber (default)", 1),
    ("non-Limber", 0),
    ("Limber again (round trip)", 1),
)

# A stale ggl cache gives delta chi2 = 0 exactly; the floor is orders of
# magnitude below the value the des_y3 project measures on its copy of
# the DES-Y3 data set (0.011), so it only catches a dead flag.
DCHI2_FLOOR = 1.0e-6

# delta chi2 of this project's example4, and the relative band
# assertion 4 allows around it. None = not measured yet: the value of
# the des_y3 donor does not carry over (another evaluation point and
# other numerical settings), so it must come from a run on this
# project's frozen state. While it is None the test prints the value
# it measures and assertion 4 is skipped; record the printed value
# here (with the date and platform) to arm it.
# measured on the frozen state of tests/frozen: arm64 macOS (Apple M2),
# clang, default build, OMP_NUM_THREADS = 4, 2026-10-01, after the data and
# covariance were regenerated with the cb halo field (the 2026-09-30 value,
# 0.010849, belonged to the covariance before)
DCHI2_MEASURED = 0.012494
DCHI2_RTOL = 0.05


class TestNonLimberGGL(unittest.TestCase):
    """Limber vs non-Limber ggl on the frozen fiducial."""

    @classmethod
    def setUpClass(cls):
        u.require_cocoa_environment()
        u.verify_frozen()
        cls.reference = u.load_reference()

    # the three models are built in process: see the module docstring
    # for why they must not share pytest's process
    @u.own_process(__file__)
    def test_nonlimber_ggl(self):
        import numpy as np
        import cosmolike_des_cluster_interface as ci

        vectors = {}
        chi2s = {}
        icov = None
        sizes = None
        nlen = None
        pairs = None
        for tag, flag in SETTINGS:
            print(f"  building model ({EXAMPLE}, NLA, {tag}) ...",
                  flush=True)
            info = u.load_frozen_info(EXAMPLE, tatt=False)
            name = u.EXAMPLES[EXAMPLE]["likelihood"]
            info["likelihood"][name]["adopt_limber_gs"] = flag
            model = u.make_model(info)
            point = u.build_point(model, EXAMPLE, tatt=False)
            start = time.perf_counter()
            chi2s[tag] = u.evaluate_chi2(model, point)
            elapsed = time.perf_counter() - start
            print(f"    chi2 = {chi2s[tag]:.6f}  ({elapsed:.1f} s, "
                  "first evaluation of the model)", flush=True)
            # full-precision model vector (the printed one keeps 9 digits)
            vectors[tag] = np.array(ci.compute_data_vector_masked())
            if icov is None:
                # all settings share one data set (mask, covariance), so
                # the first build's masked inverse covariance serves all
                icov = np.array(ci.get_inv_cov_masked())
                like = model.likelihood[name]
                sizes = ci.compute_data_vector_3x2pt_real_sizes()
                nlen = int(like.ntheta)
                # the ggl pairs in data-vector order: lens-major, the
                # pairs listed in the yaml key ggl_exclude left out
                excluded = {(int(zl), int(zs)) for zl, zs in
                            (getattr(like, "ggl_exclude", None) or [])}
                pairs = [(zl, zs) for zl in range(int(like.lens_ntomo))
                                  for zs in range(int(like.source_ntomo))
                                  if (zl, zs) not in excluded]

        dv_limber = vectors[SETTINGS[0][0]]
        dv_nonlimber = vectors[SETTINGS[1][0]]
        delta = dv_nonlimber - dv_limber
        dchi2 = float(delta @ icov @ delta)

        # the ggl block follows the cosmic shear block: entries
        # [ggl0, ggl1)
        ggl0 = int(sizes[0])
        ggl1 = ggl0 + int(sizes[1])
        npairs = int(sizes[1]) // nlen

        # the ternary picks the note printed next to the value: the
        # recorded measurement, or the reminder that none exists yet
        measured = ("not recorded yet: set DCHI2_MEASURED to this value"
                    if DCHI2_MEASURED is None
                    else f"measured {DCHI2_MEASURED:.6f}")
        print(f"\n  delta chi2 report ({EXAMPLE}, NLA):")
        print(f"    Limber:     chi2 = {chi2s[SETTINGS[0][0]]:.6f} "
              f"(frozen reference {self.reference[REFERENCE_KEY]:.6f})")
        print(f"    non-Limber: chi2 = {chi2s[SETTINGS[1][0]]:.6f}")
        print(f"    delta^T C^-1 delta = {dchi2:.6f} ({measured})")
        print("    per lens-source pair (the pair's block alone):")
        rows = []
        for p in range(npairs):
            block = np.zeros_like(delta)
            sl = slice(ggl0 + p*nlen, ggl0 + (p + 1)*nlen)
            block[sl] = delta[sl]
            rows.append((float(block @ icov @ block), p))
        for contribution, p in sorted(rows, reverse=True):
            if contribution < 1.0e-3*max(dchi2, DCHI2_FLOOR):
                break
            label = (f"(lens {pairs[p][0]}, source {pairs[p][1]})"
                     if len(pairs) == npairs else f"pair {p}")
            print(f"      {label:24s} {contribution:.6f}")

        self.assertGreater(
            dchi2, DCHI2_FLOOR,
            "non-Limber ggl did not change the data vector: the "
            "adopt_limber_gs flag did not reach the C code, or the ggl "
            "cache did not rebuild")

        outside = np.concatenate((delta[:ggl0], delta[ggl1:]))
        self.assertTrue(
            np.all(outside == 0.0),
            "entries outside the ggl block changed with adopt_limber_gs")

        self.assertTrue(
            np.array_equal(vectors[SETTINGS[-1][0]], dv_limber),
            "returning to Limber did not reproduce the first data vector "
            "bit for bit; the ggl cache did not rebuild cleanly")

        if DCHI2_MEASURED is not None:
            self.assertLess(
                abs(dchi2/DCHI2_MEASURED - 1.0), DCHI2_RTOL,
                f"delta chi2 = {dchi2:.6f} differs from the measured "
                f"{DCHI2_MEASURED:.6f} by more than {DCHI2_RTOL:.0%}")

        # last, so that a stale frozen snapshot does not hide the four
        # checks above
        self.assertLess(
            abs(chi2s[SETTINGS[0][0]] - self.reference[REFERENCE_KEY]),
            u.CHI2_TOLERANCE,
            f"{SETTINGS[0][0]}: chi2 = {chi2s[SETTINGS[0][0]]:.6f} vs frozen "
            f"reference {self.reference[REFERENCE_KEY]:.6f}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
