"""Unit test: non-Limber galaxy-galaxy lensing (adopt_limber_gs).

The galaxy-galaxy lensing (ggl) spectrum C_l^gs, the cross spectrum of
the lens galaxy density and the shear of the source galaxies, enters
the data vector through gamma_t(theta), the tangential shear of the
sources around the lens galaxies (a sum over l of C_l^gs times
bin-averaged associated Legendre polynomials). The exact projection of
the 3D power spectrum P(k) onto the sky is, for one lens-source pair
(schematically: the redshift dependence of P is left out),

    C_l = (2/pi) int dk k^2 P(k) I_l^g(k) I_l^s(k),
    I_l^X(k) = int dchi W_X(chi) j_l(k chi),

with W_g the kernel of the lens bin (its n(z) times the galaxy bias),
W_s the kernel of the source bin (lensing and the intrinsic alignment
of the sources), chi the comoving distance and j_l the spherical
Bessel function. The Limber approximation replaces the Bessel
integrals by their value at k = (l + 1/2)/chi, which leaves one
line-of-sight integral,

    C_l = int dchi W_g(chi) W_s(chi) / chi^2 P((l + 1/2)/chi).

It is accurate when one of the kernels is broad compared with the
oscillations of j_l. The likelihood yaml key adopt_limber_gs chooses
how C_l^gs is computed:

  adopt_limber_gs: 1 (the default) - Limber approximation at every
      multipole.
  adopt_limber_gs: 0 - below l = 150 the exact projection, computed by
      cosmolike's C_gs_tomo with the split of Fang, Krause, Eifler &
      MacCrann (arXiv:1911.11947): an FFTLog integral of the linear
      power spectrum plus, in Limber, what linear theory misses, that
      is exact[P_lin] + Limber[P_nl] - Limber[P_lin]. FFTLog evaluates
      the Bessel integrals with a fast Fourier transform on a
      logarithmic k grid; the nonlinear part matters at small scales,
      where Limber is accurate. A lens-source pair switches to its
      Limber values at the multipole where the two agree within 1%,
      and every pair uses Limber from l = 150 up.

ggl defaults to Limber because its lensing kernel is broad (galaxy
clustering has its own key, adopt_limber_gg; see
test_nonlimber_gg.py). The Limber approximation fails at low l for
the lens-source pairs whose redshift distributions overlap (a source
bin at the redshift of the lens bin, or in front of it): there the
signal includes the intrinsic alignment of the sources times the lens
density, a product of two kernels that follow redshift distributions,
both narrower than the lensing kernel. This test measures what the
Limber default costs. The gamma_t block of 6x2pt + N reads the same
key, and no other test of this project flips it.

Terms used below: frozen means stored once under tests/frozen/ and
never edited by hand (the complete configuration with its evaluation
point, the fiducial point, the tests' own copy of the data files, and
the reference chi2 values), with every file pinned by its SHA-256 hash
(a fingerprint that changes when any byte changes) in
tests/manifest_sha256.json. The data vector is the list of every
measured value the likelihood compares with its model, and the
scale-cut mask marks the entries the chi2 uses; the other entries are
masked.

It evaluates the frozen 3x2pt fiducial (example4, NLA) three times IN
ONE PROCESS: Limber, non-Limber, Limber again, and computes

    delta chi2 = delta^T C^-1 delta,
    delta = dv(non-Limber) - dv(Limber),

with dv the 900-entry model data vector (blocks ss, gs, gg, zero at
the masked entries) and C^-1 the masked inverse covariance (the
inverse of the covariance of the kept entries, set in a 900 x 900
matrix whose masked rows and columns are zero): the chi2 the Limber
model would score against a data set generated with non-Limber ggl.
It prints the total and, for each lens-source pair p,
delta_p^T (C^-1)_pp delta_p, the quadratic form restricted to the
pair's entries ((C^-1)_pp is the pair's diagonal block of the full
inverse covariance; the cross terms between pairs are left out, so the
rows need not add up to the total).

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
     (checked last, so a frozen reference that no longer matches the
     code cannot hide checks 1-4).

A dead flag is a yaml key that never reaches the C code. A cache is a
table cosmolike keeps in C static memory: it outlives one model and is
rebuilt only when a value in its key changes. The adopt_limber_gs flag
is part of the key of the ggl table, so one process can switch between
the two paths.

The three evaluations run in a python process of their own
(cocoa_test_utils.own_process): under pytest the collecting process
also holds the cluster models of test_cache_consistency.py (the
sector ladder, which builds its 2812-entry models in pytest's own
process), and cosmolike aborts a process that initializes two data
sets of different dimensions (here 2812 and 900 entries). In the
pytest process the decorated test method runs this file again
with the same python executable,

    python <this file> TestNonLimberGGL.test_nonlimber_ggl

with COCOA_TESTS_OWN_PROCESS=1 and OMP_NUM_THREADS=4 in the child's
environment, and passes when the child exits with code 0; in the
child the method body runs.

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/des_cluster/tests/data_vector/test_nonlimber_ggl.py
"""

import os

# os.environ is the table of environment variables of this process
# (text settings that the processes it starts inherit). OpenMP, the
# library that spreads cosmolike's C loops over CPU threads, reads
# OMP_NUM_THREADS when the compiled libraries load, so this must run
# before ANY cobaya/cosmolike import in the process. "4" is the
# REQUIRED_OMP_THREADS of the harness: the thread count of the frozen
# references and of the DCHI2_MEASURED record below.
os.environ["OMP_NUM_THREADS"] = "4"

import sys
import time
import unittest

# The harness cocoa_test_utils.py sits one folder up, in tests/
# (dirname twice: this file -> data_vector/ -> tests/). Python searches
# the folders listed in sys.path for modules, a list that holds this
# file's own folder but not tests/, so insert(0, ...) puts tests/
# first; the import then works both when pytest imports this file and
# when python runs it directly. The harness is imported as u.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cocoa_test_utils as u

# The frozen 3x2pt configuration (des_cluster.combo_3x2pt on the DES Y3
# placeholder data set: 5 lens bins, 4 source bins, 20 angular bins),
# evaluated against its generated NLA vector, and its entry in
# reference_chi2.json.
EXAMPLE = "example4"
REFERENCE_KEY = "example4_nla"

# (report tag, adopt_limber_gs): the default 1 (Limber, the value of the
# likelihood yamls of this project and of the likelihood code when the
# key is absent), the other setting, and the default again
SETTINGS = (
    ("Limber (default)", 1),
    ("non-Limber", 0),
    ("Limber again (round trip)", 1),
)

# A dead flag or a stale ggl cache gives delta chi2 = 0 exactly; the
# floor sits about four orders of magnitude below the value measured
# here (DCHI2_MEASURED, 0.0125), so it catches only those two.
DCHI2_FLOOR = 1.0e-6

# DCHI2_MEASURED = the delta chi2 this test prints, recorded on the
# frozen state of tests/frozen (arm64 macOS on an Apple M2, clang,
# default build, OMP_NUM_THREADS = 4); DCHI2_RTOL = the relative band
# assertion 4 allows around it (5%). delta is the difference of two
# model vectors, so the value depends on the model (cosmology, n(z),
# galaxy bias, intrinsic alignment, the non-Limber numerics) and on the
# masked covariance, not on the data vector. The des_y3 project records
# 0.012 for its own frozen 3x2pt configuration (another evaluation
# point, lmax and kmax_boltzmann), a value that does not carry over.
# None switches assertion 4 off: the test then prints the value it
# measures, and recording it here, with the platform, arms the
# assertion.
DCHI2_MEASURED = 0.012494
DCHI2_RTOL = 0.05


class TestNonLimberGGL(unittest.TestCase):
    """Limber vs non-Limber ggl on the frozen fiducial.

    unittest.TestCase is the base class of Python's built-in test
    framework, and pytest, the runner of the command in the module
    docstring, collects such classes too: every method whose name
    starts with test_ is one test.
    """

    # @classmethod is a decorator (a function applied to the method
    # defined just below it): it hands the method the class itself
    # (cls), not an instance; unittest calls setUpClass once before
    # the first test of the class
    @classmethod
    def setUpClass(cls):
        """Check the environment and the frozen state once, before the test.

        u.require_cocoa_environment stops with an instruction when
        start_cocoa.sh was not sourced (ROOTDIR unset) and moves the
        process to ROOTDIR, the folder the relative paths of the frozen
        configuration start from. u.verify_frozen hashes the frozen
        files under tests/frozen/ and compares the hashes with
        tests/manifest_sha256.json, so an edited, missing or added file
        fails the test before a model is built. u.load_reference reads
        reference_chi2.json. All of this runs in the pytest process and
        again in the child process of own_process.

        Arguments:
          cls = the test class (see the comment above); an attribute
                set on it is visible in the test as self.<name>.

        Returns:
          nothing; stores the reference table, {configuration key:
          chi2}, as cls.reference, and leaves the process in ROOTDIR.

        Raises:
          RuntimeError when ROOTDIR is not set; AssertionError naming
          every frozen file that is CHANGED, MISSING or EXTRA.
        """
        u.require_cocoa_environment()
        u.verify_frozen()
        cls.reference = u.load_reference()

    # the three models are built in process: see the module docstring
    # for why they must not share pytest's process; the decorator
    # u.own_process(__file__) wraps the method so that, in the pytest
    # process, it runs this file again in a child process
    @u.own_process(__file__)
    def test_nonlimber_ggl(self):
        """Non-Limber ggl moves gamma_t alone, by the measured amount.

        Builds three models of the frozen 3x2pt configuration in this
        process, one per entry of SETTINGS (adopt_limber_gs = 1, 0, 1),
        evaluates each at the frozen point, and reads its full-precision
        model vector from the compiled interface. The models are built
        one after the other in one process on purpose: cosmolike's
        tables live in C static memory, which survives from one model
        to the next, so the flag must reach the C code, the ggl table
        must be rebuilt with it, and the third model must reproduce the
        first model's vector. The five assertions of the module
        docstring follow, the frozen-reference check last; its limit
        CHI2_TOLERANCE = 0.2 is the bound of every frozen chi2 check
        (no physics change is detectable below a chi2 shift of 0.2).

        Arguments:
          none (unittest passes the test case itself as self).

        Returns:
          nothing; prints the chi2 of each setting, delta chi2 and the
          contributions of the lens-source pairs.

        Raises:
          AssertionError when one of the five assertions fails. In the
          pytest process the decorator reports a nonzero exit of the
          child process as one AssertionError naming the test.
        """
        # numpy (the array library) and the compiled interface are
        # imported here, so only the child process loads them; ci is the
        # python module of the compiled cosmolike interface (C++
        # functions callable from python)
        import numpy as np
        import cosmolike_des_cluster_interface as ci

        # vectors and chi2s collect one entry per report tag; icov,
        # sizes, nlen and pairs are read once, from the first model
        # (None = not read yet)
        vectors = {}
        chi2s = {}
        icov = None
        sizes = None
        nlen = None
        pairs = None
        for tag, flag in SETTINGS:
            print(f"  building model ({EXAMPLE}, NLA, {tag}) ...",
                  flush=True)
            # load_frozen_info returns the frozen configuration as a
            # dictionary (absolute data path, NLA, the generated NLA
            # data vector, printing off); name is the likelihood's key
            # in it, des_cluster.combo_3x2pt, and the flag is written
            # into that likelihood's block before the model is built
            info = u.load_frozen_info(EXAMPLE, tatt=False)
            name = u.EXAMPLES[EXAMPLE]["likelihood"]
            info["likelihood"][name]["adopt_limber_gs"] = flag
            # make_model builds the cobaya model (CAMB plus the
            # likelihood, which hands the flag to cosmolike);
            # build_point returns the frozen point after checking that
            # it names exactly the parameters the model samples
            model = u.make_model(info)
            point = u.build_point(model, EXAMPLE, tatt=False)
            # perf_counter is a clock in seconds: the report shows the
            # time of the first evaluation of each model
            start = time.perf_counter()
            chi2s[tag] = u.evaluate_chi2(model, point)
            elapsed = time.perf_counter() - start
            print(f"    chi2 = {chi2s[tag]:.6f}  ({elapsed:.1f} s, "
                  "first evaluation of the model)", flush=True)
            # full-precision model vector (the printed one keeps 9 digits):
            # compute_data_vector_masked returns the 900 entries at the
            # state of the evaluation just made, zero at masked entries,
            # and np.array turns that list into a numpy array
            vectors[tag] = np.array(ci.compute_data_vector_masked())
            if icov is None:
                # all settings share one data set (mask, covariance), so
                # the first build's masked inverse covariance serves all
                # (900 x 900, masked rows and columns zero); like is the
                # likelihood instance inside the cobaya model, sizes the
                # lengths of the ss, gs and gg blocks ([400, 400, 100]
                # here), nlen the number of angular bins of one
                # two-point function (20)
                icov = np.array(ci.get_inv_cov_masked())
                like = model.likelihood[name]
                sizes = ci.compute_data_vector_3x2pt_real_sizes()
                nlen = int(like.ntheta)
                # the ggl pairs in data-vector order: lens-major, the
                # pairs listed in the yaml key ggl_exclude left out.
                # getattr(like, "ggl_exclude", None) gives None when the
                # likelihood has no such attribute, and "or []" turns
                # None into an empty list; the set comprehension turns
                # each listed (lens, source) pair into a tuple of ints.
                # The likelihoods of this project declare no
                # ggl_exclude, so all 5 x 4 = 20 pairs enter.
                excluded = {(int(zl), int(zs)) for zl, zs in
                            (getattr(like, "ggl_exclude", None) or [])}
                # the list comprehension loops over lens bins (outer)
                # and source bins (inner), the order in which cosmolike
                # numbers the pairs, and keeps the pairs not excluded
                pairs = [(zl, zs) for zl in range(int(like.lens_ntomo))
                                  for zs in range(int(like.source_ntomo))
                                  if (zl, zs) not in excluded]

        # SETTINGS[i][0] is the report tag of setting i: 0 = Limber,
        # 1 = non-Limber. @ is the matrix product, so
        # delta @ icov @ delta = delta^T C^-1 delta, one number.
        dv_limber = vectors[SETTINGS[0][0]]
        dv_nonlimber = vectors[SETTINGS[1][0]]
        delta = dv_nonlimber - dv_limber
        dchi2 = float(delta @ icov @ delta)

        # the ggl block follows the cosmic shear block: entries
        # [ggl0, ggl1) ([400, 800) here), one block of nlen entries per
        # lens-source pair (20 pairs of 20 angular bins here; // is
        # integer division)
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
        # per pair p: block = delta with every entry outside the pair's
        # nlen entries set to zero (np.zeros_like makes an array of
        # zeros with delta's shape; slice(start, stop) selects the
        # entries start to stop - 1), and block @ icov @ block is the
        # pair's quadratic form of the module docstring
        rows = []
        for p in range(npairs):
            block = np.zeros_like(delta)
            sl = slice(ggl0 + p*nlen, ggl0 + (p + 1)*nlen)
            block[sl] = delta[sl]
            rows.append((float(block @ icov @ block), p))
        # sorted(rows, reverse=True) orders the (contribution, pair)
        # tuples by contribution, largest first; the loop stops at the
        # first row below 1/1000 of the total (max with the floor keeps
        # that cut positive when the total is zero)
        for contribution, p in sorted(rows, reverse=True):
            if contribution < 1.0e-3*max(dchi2, DCHI2_FLOOR):
                break
            # the label names the two bins (counted from 0) when the
            # python pair list has as many pairs as the C code counted,
            # and falls back to the pair index otherwise; {label:24s}
            # pads the label to 24 characters
            label = (f"(lens {pairs[p][0]}, source {pairs[p][1]})"
                     if len(pairs) == npairs else f"pair {p}")
            print(f"      {label:24s} {contribution:.6f}")

        # assertion 1: the flag reached the C code (dead-flag floor)
        self.assertGreater(
            dchi2, DCHI2_FLOOR,
            "non-Limber ggl did not change the data vector: the "
            "adopt_limber_gs flag did not reach the C code, or the ggl "
            "cache did not rebuild")

        # assertion 2: np.concatenate joins the entries before and after
        # the ggl block into one array; every one of them must be
        # exactly zero
        outside = np.concatenate((delta[:ggl0], delta[ggl1:]))
        self.assertTrue(
            np.all(outside == 0.0),
            "entries outside the ggl block changed with adopt_limber_gs")

        # assertion 3: np.array_equal is True when the two vectors agree
        # in every entry; SETTINGS[-1] is the last setting, Limber again
        self.assertTrue(
            np.array_equal(vectors[SETTINGS[-1][0]], dv_limber),
            "returning to Limber did not reproduce the first data vector "
            "bit for bit; the ggl cache did not rebuild cleanly")

        # assertion 4, active once DCHI2_MEASURED holds a value; in the
        # message, {DCHI2_RTOL:.0%} prints 0.05 as 5%
        if DCHI2_MEASURED is not None:
            self.assertLess(
                abs(dchi2/DCHI2_MEASURED - 1.0), DCHI2_RTOL,
                f"delta chi2 = {dchi2:.6f} differs from the measured "
                f"{DCHI2_MEASURED:.6f} by more than {DCHI2_RTOL:.0%}")

        # assertion 5, last, so that a frozen reference that no longer
        # matches the code does not hide the four checks above
        self.assertLess(
            abs(chi2s[SETTINGS[0][0]] - self.reference[REFERENCE_KEY]),
            u.CHI2_TOLERANCE,
            f"{SETTINGS[0][0]}: chi2 = {chi2s[SETTINGS[0][0]]:.6f} vs frozen "
            f"reference {self.reference[REFERENCE_KEY]:.6f}")


# __name__ is "__main__" only when this file runs as a script: the
# own_process child runs it so, with the test name as its argument, and
# unittest.main runs that one test (every test of the file when no name
# is given). pytest imports the module instead, so this block stays
# idle under pytest.
if __name__ == "__main__":
    unittest.main(verbosity=2)
