"""Unit test: non-Limber galaxy clustering (adopt_limber_gg).

The galaxy clustering (gg) spectrum C_l^gg enters the data vector
through w(theta), the angular correlation function of the lens
galaxies (a sum over l of C_l^gg times bin-averaged Legendre
polynomials). The exact projection of the 3D power spectrum P(k) onto
the sky is, for one lens bin (schematically: the redshift dependence
of P is left out),

    C_l = (2/pi) int dk k^2 P(k) I_l(k)^2,
    I_l(k) = int dchi W(chi) j_l(k chi),

with W the radial kernel of the bin (its n(z) times the galaxy bias;
the code adds the redshift-space-distortion and magnification terms),
chi the comoving distance and j_l the spherical Bessel function. The
Limber approximation replaces the Bessel integrals by their value at
k = (l + 1/2)/chi, which leaves one line-of-sight integral,

    C_l = int dchi W(chi)^2 / chi^2 P((l + 1/2)/chi).

It is accurate when the kernel is broad compared with the oscillations
of j_l, and it fails at low l for narrow kernels. The likelihood yaml
key adopt_limber_gg chooses how C_l^gg is computed:

  adopt_limber_gg: 0 - below l = 150 the exact projection, computed by
      cosmolike's C_cl_tomo with the split of Fang, Krause, Eifler &
      MacCrann (arXiv:1911.11947): an FFTLog integral of the linear
      power spectrum plus, in Limber, what linear theory misses, that
      is exact[P_lin] + Limber[P_nl] - Limber[P_lin]. FFTLog evaluates
      the Bessel integrals with a fast Fourier transform on a
      logarithmic k grid; the nonlinear part matters at small scales,
      where Limber is accurate. A lens bin switches to its Limber
      values at the multipole where the two agree within 0.2%, and
      every bin uses Limber from l = 150 up.
  adopt_limber_gg: 1 - Limber approximation at every multipole.

This project's default is adopt_limber_gg: 0 (non-Limber), in the
galaxy-only likelihoods and in the cluster combinations alike (the
w_gg block of 4x2pt + N and 6x2pt + N reads the same key). The lens
galaxy redshift distributions are narrow, so the Limber approximation
fails at low l for the clustering auto spectra; this test measures
by how much. No other test of this project flips the key.

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
ONE PROCESS: the default, the other setting, the default again, and
computes

    delta chi2 = delta^T C^-1 delta,
    delta = dv(non-Limber) - dv(Limber),

with dv the 900-entry model data vector (blocks ss, gs, gg, zero at
the masked entries) and C^-1 the masked inverse covariance (the
inverse of the covariance of the kept entries, set in a 900 x 900
matrix whose masked rows and columns are zero): the chi2 a Limber
model would score against a data set generated with non-Limber
clustering. It prints the total and, for each lens bin b,
delta_b^T (C^-1)_bb delta_b, the quadratic form restricted to the
bin's entries ((C^-1)_bb is the bin's diagonal block of the full
inverse covariance; the cross terms between bins are left out, so the
rows need not add up to the total).

Assertions:
  1. delta chi2 is above a dead-flag floor: the flag reaches the C code
     and the clustering cache notices the change (a stale cache gives
     zero);
  2. only clustering entries change: cosmic shear and galaxy-galaxy
     lensing are bitwise equal between the two evaluations;
  3. switching back to the default reproduces the first data vector
     bitwise;
  4. delta chi2 matches the value measured for this project
     (DCHI2_MEASURED below) to 5%: a change in the non-Limber code, the
     kernels, or the covariance shows up here (skipped while no value
     is recorded);
  5. the default evaluation reproduces the frozen reference chi2
     (checked last, so a frozen reference that no longer matches the
     code cannot hide checks 1-4).

A dead flag is a yaml key that never reaches the C code. A cache is a
table cosmolike keeps in C static memory: it outlives one model and is
rebuilt only when a value in its key changes. The adopt_limber_gg flag
is part of the key of the clustering table, so one process can switch
between the two paths.

The three evaluations run in a python process of their own
(cocoa_test_utils.own_process): under pytest the collecting process
also holds the cluster models of test_cache_consistency.py (the
sector ladder, which builds its 2812-entry models in pytest's own
process), and cosmolike aborts a process that initializes two data
sets of different dimensions (here 2812 and 900 entries). In the
pytest process the decorated test method runs this file again
with the same python executable,

    python <this file> TestNonLimberGG.test_nonlimber_gg

with COCOA_TESTS_OWN_PROCESS=1 and OMP_NUM_THREADS=4 in the child's
environment, and passes when the child exits with code 0; in the
child the method body runs.

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/des_cluster/tests/data_vector/test_nonlimber_gg.py
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

# The value of adopt_limber_gg in the likelihood yamls of this project,
# also the default of the likelihood code when the key is absent:
# 0 = non-Limber.
DEFAULT = 0

# (report tag, adopt_limber_gg): the default, the other setting, the
# default again. _NAME maps a flag value to its printed name, and
# 1 - DEFAULT is the other value of the 0/1 flag; with DEFAULT = 0 the
# tags read "non-Limber (default)", "Limber" and "non-Limber again
# (round trip)".
_NAME = {0: "non-Limber", 1: "Limber"}
SETTINGS = (
    (f"{_NAME[DEFAULT]} (default)", DEFAULT),
    (_NAME[1 - DEFAULT], 1 - DEFAULT),
    (f"{_NAME[DEFAULT]} again (round trip)", DEFAULT),
)

# A dead flag or a stale clustering cache gives delta chi2 = 0 exactly;
# the floor sits about seven orders of magnitude below the value
# measured here (DCHI2_MEASURED, 7.12), so it catches only those two.
DCHI2_FLOOR = 1.0e-6

# DCHI2_MEASURED = the delta chi2 this test prints, recorded on the
# frozen state of tests/frozen (arm64 macOS on an Apple M2, clang,
# default build, OMP_NUM_THREADS = 4); DCHI2_RTOL = the relative band
# assertion 4 allows around it (5%). delta is the difference of two
# model vectors, so the value depends on the model (cosmology, lens
# n(z), galaxy bias, the non-Limber numerics) and on the masked
# covariance, not on the data vector. The des_y3 project records 9.49
# for its own frozen 3x2pt configuration (another evaluation point,
# lmax and kmax_boltzmann), a value that does not carry over. None
# switches assertion 4 off: the test then prints the value it
# measures, and recording it here, with the platform, arms the
# assertion.
DCHI2_MEASURED = 7.123750
DCHI2_RTOL = 0.05


class TestNonLimberGG(unittest.TestCase):
    """Limber vs non-Limber galaxy clustering on the frozen fiducial.

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
    def test_nonlimber_gg(self):
        """Non-Limber clustering moves w(theta) alone, by the measured amount.

        Builds three models of the frozen 3x2pt configuration in this
        process, one per entry of SETTINGS (adopt_limber_gg = 0, 1, 0),
        evaluates each at the frozen point, and reads its full-precision
        model vector from the compiled interface. The models are built
        one after the other in one process on purpose: cosmolike's
        tables live in C static memory, which survives from one model
        to the next, so the flag must reach the C code, the clustering
        table must be rebuilt with it, and the third model must
        reproduce the first model's vector. The five assertions of the
        module docstring follow, the frozen-reference check last; its
        limit CHI2_TOLERANCE = 0.2 is the bound of every frozen chi2
        check (no physics change is detectable below a chi2 shift of
        0.2).

        Arguments:
          none (unittest passes the test case itself as self).

        Returns:
          nothing; prints the chi2 of each setting, delta chi2 and the
          contributions of the lens bins.

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
        # sizes and nlen are read once, from the first model (None =
        # not read yet)
        vectors = {}
        chi2s = {}
        icov = None
        sizes = None
        nlen = None
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
            info["likelihood"][name]["adopt_limber_gg"] = flag
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

        # SETTINGS[:2] = the first two settings (the default and the
        # other one); the dict comprehension maps each flag value to its
        # report tag, so tags[0] names the non-Limber vector and tags[1]
        # the Limber one whatever DEFAULT is. SETTINGS[0][0] is the tag
        # of the first setting. @ is the matrix product, so
        # delta @ icov @ delta = delta^T C^-1 delta, one number.
        tags = {flag: tag for tag, flag in SETTINGS[:2]}
        dv_default = vectors[SETTINGS[0][0]]
        delta = vectors[tags[0]] - vectors[tags[1]]
        dchi2 = float(delta @ icov @ delta)

        # the clustering block follows cosmic shear and galaxy-galaxy
        # lensing: entries [gg0, gg1), one block of nlen entries per
        # lens bin ([800, 900), 5 lens bins of 20 angular bins here; //
        # is integer division)
        gg0 = int(sizes[0]) + int(sizes[1])
        gg1 = gg0 + int(sizes[2])
        nbins = int(sizes[2]) // nlen

        # the ternary picks the note printed next to the value: the
        # recorded measurement, or the reminder that none exists yet
        measured = ("not recorded yet: set DCHI2_MEASURED to this value"
                    if DCHI2_MEASURED is None
                    else f"measured {DCHI2_MEASURED:.4f}")
        print(f"\n  delta chi2 report ({EXAMPLE}, NLA):")
        print(f"    {SETTINGS[0][0]}: chi2 = {chi2s[SETTINGS[0][0]]:.6f} "
              f"(frozen reference {self.reference[REFERENCE_KEY]:.6f})")
        print(f"    {SETTINGS[1][0]}: chi2 = {chi2s[SETTINGS[1][0]]:.6f}")
        print(f"    delta^T C^-1 delta = {dchi2:.4f} ({measured})")
        print("    per lens bin (the bin's block alone):")
        # per lens bin b: block = delta with every entry outside the
        # bin's nlen entries set to zero (np.zeros_like makes an array
        # of zeros with delta's shape; slice(start, stop) selects the
        # entries start to stop - 1), and block @ icov @ block is the
        # bin's quadratic form of the module docstring
        rows = []
        for b in range(nbins):
            block = np.zeros_like(delta)
            sl = slice(gg0 + b*nlen, gg0 + (b + 1)*nlen)
            block[sl] = delta[sl]
            rows.append((float(block @ icov @ block), b))
        # sorted(rows, reverse=True) orders the (contribution, bin)
        # tuples by contribution, largest first; the loop stops at the
        # first row below 1/1000 of the total (max with the floor keeps
        # that cut positive when the total is zero); {b:<14d} prints the
        # bin number left-aligned in 14 characters
        for contribution, b in sorted(rows, reverse=True):
            if contribution < 1.0e-3*max(dchi2, DCHI2_FLOOR):
                break
            print(f"      lens bin {b:<14d} {contribution:.4f}")

        # assertion 1: the flag reached the C code (dead-flag floor)
        self.assertGreater(
            dchi2, DCHI2_FLOOR,
            "non-Limber clustering did not change the data vector: the "
            "adopt_limber_gg flag did not reach the C code, or the "
            "clustering cache did not rebuild")

        # assertion 2: np.concatenate joins the entries before and after
        # the clustering block into one array; every one of them must be
        # exactly zero
        outside = np.concatenate((delta[:gg0], delta[gg1:]))
        self.assertTrue(
            np.all(outside == 0.0),
            "entries outside the clustering block changed with "
            "adopt_limber_gg")

        # assertion 3: np.array_equal is True when the two vectors agree
        # in every entry; SETTINGS[-1] is the last setting, the default
        # again
        self.assertTrue(
            np.array_equal(vectors[SETTINGS[-1][0]], dv_default),
            "returning to the default did not reproduce the first data "
            "vector bit for bit; the clustering cache did not rebuild "
            "cleanly")

        # assertion 4, active once DCHI2_MEASURED holds a value; in the
        # message, {DCHI2_RTOL:.0%} prints 0.05 as 5%
        if DCHI2_MEASURED is not None:
            self.assertLess(
                abs(dchi2/DCHI2_MEASURED - 1.0), DCHI2_RTOL,
                f"delta chi2 = {dchi2:.4f} differs from the measured "
                f"{DCHI2_MEASURED:.4f} by more than {DCHI2_RTOL:.0%}")

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
