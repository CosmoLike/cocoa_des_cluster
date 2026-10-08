"""Unit tests 3-4: the 6x2pt + N likelihood on the frozen test data.

6x2pt + N is the full cluster-plus-3x2pt combination (CL+3x2pt of
arXiv 2503.13631): the 4x2pt + N blocks of example1 (cluster number
counts N, cluster lensing, cluster clustering, the cluster x galaxy
cross-correlation) joined with the galaxy 3x2pt (cosmic shear,
galaxy-galaxy lensing, and galaxy clustering of all six MagLim lens
bins). Here it is the des_cluster.combo_6x2pt_N likelihood, evaluated
on the frozen copy of example2's configuration.

Terms used below:

  frozen       stored once under tests/frozen/ and never edited by
               hand: the complete configuration with its evaluation
               point, the fiducial point of these tests
               (frozen_config_example2.py, read by cobaya, the sampler
               framework that loads the likelihoods), the tests' own
               copy of the data files (data/), and the reference chi2
               values (reference_chi2.json).
               tests/manifest_sha256.json holds the SHA-256 hash of
               each frozen file (a fingerprint that changes when any
               byte changes); the tests refuse to run on a mismatch.
  data vector  every measured value the likelihood compares with its
               model, in one fixed order: here the 2812 entries ss,
               gs, gg, cg, N, cc, cs of the synthetic DES Y6-like data
               set. The scale-cut mask named by
               des_cluster_y6_6x2ptN.dataset keeps entries of all
               seven blocks (gg of all six lens bins).
  chi2         (d - m)^T C^-1 (d - m) over the kept entries, with d
               the data vector, m the model vector and C their
               covariance.
  worker       a separate python process the harness starts for one
               evaluation, waiting for its result: the cosmolike C
               library keeps the data-vector size in global variables
               and aborts a process that initializes a second data set
               of another size, and this suite uses two (2812 and 900
               entries).

The frozen data vector is the synthetic one written by
scripts/make_synthetic_data.py: the model at the fiducial point of
Table I of arXiv 2503.13631, which is also the frozen point. The vector
is not regenerated with every change of the halo model, and the current
model differs from it by chi2 = 0.1518 at that point, the stored
reference (the project README, section Cluster options, gives
|Delta chi2| < 0.152 for both cluster combinations). The chi2 thus sits
close to its minimum at zero, where a small numerical change of the
model moves it much less than at a point far from the data. The two
tests:

  3. chi2 at the frozen fiducial point, within CHI2_TOLERANCE (0.2) of
     the frozen reference value.
  4. race check: on one model, the fiducial evaluated fresh and again
     as the 10th of 10 cosmologies in a row must agree to
     RACE_TOLERANCE (1e-4). A disagreement means state leaked between
     evaluations (a table kept from an earlier cosmology) or OpenMP
     threads raced (two of the CPU threads that share cosmolike's
     loops wrote the same memory).

Both run the NLA (nonlinear alignment) model of the intrinsic alignment
of source galaxy shapes, the only one the cluster lensing code carries
(there are no TATT variants of these tests).

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/des_cluster/tests/data_vector/test_example2.py

The folder path ./projects/des_cluster/tests/data_vector in place of
the file runs the whole data-vector sector.
"""

import os

# os.environ is the table of environment variables of this process
# (text settings that the processes it starts inherit). OpenMP, the
# library that spreads cosmolike's C loops over CPU threads, reads
# OMP_NUM_THREADS when the compiled libraries load, so this must run
# before ANY cobaya/cosmolike import in the process. "4" is the
# REQUIRED_OMP_THREADS of the harness: the race test needs several
# threads (with one there is no race to find), and the frozen
# references were computed with 4.
os.environ["OMP_NUM_THREADS"] = "4"

import sys
import unittest

# The harness cocoa_test_utils.py sits one folder up, in tests/
# (dirname twice: this file -> data_vector/ -> tests/). Python searches
# the folders listed in sys.path for modules, a list that holds this
# file's own folder but not tests/, so insert(0, ...) puts tests/
# first; the import then works both when pytest imports this file and
# when python runs it directly. The harness is imported as u.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cocoa_test_utils as u

# The key of cocoa_test_utils.EXAMPLES this file tests: it selects the
# frozen module (frozen_config_example2.py), the likelihood
# (des_cluster.combo_6x2pt_N) and the reference entry (example2_nla).
EXAMPLE = "example2"


class TestExample2Combo6x2ptN(unittest.TestCase):
    """Tests 3-4, sharing one frozen-state verification.

    unittest.TestCase is the base class of Python's built-in test
    framework, and pytest, the runner of the command in the module
    docstring, collects such classes too: every method whose name
    starts with test_ is one test, and the tests run in the
    alphabetical order of their names.

    setUpClass runs once before the tests: it moves to ROOTDIR,
    verifies every frozen file against the SHA-256 manifest (an edited
    frozen state must fail loudly before any physics runs), and loads
    the frozen reference chi2 values.
    """

    # @classmethod is a decorator (a function applied to the method
    # defined just below it): it hands the method the class itself
    # (cls), not an instance; unittest calls setUpClass once before
    # the first test of the class
    @classmethod
    def setUpClass(cls):
        """Check the environment and the frozen state once, before any test.

        u.require_cocoa_environment stops with an instruction when
        start_cocoa.sh was not sourced (ROOTDIR unset) and moves the
        process to ROOTDIR, the folder the relative paths of the frozen
        configuration start from. u.verify_frozen hashes the frozen
        files under tests/frozen/ and compares the hashes with
        tests/manifest_sha256.json, so an edited, missing or added file
        fails every test before a model is built. u.load_reference reads
        reference_chi2.json.

        Arguments:
          cls = the test class (see the comment above); an attribute
                set on it is visible in every test as self.<name>.

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

    def test_3_chi2_matches_frozen_reference(self):
        """chi2 at the frozen NLA point stays within 0.2 of the reference.

        u.single_model_chi2 starts one worker, which builds the cobaya
        model (CAMB plus the likelihood) from the frozen configuration,
        checks that the frozen point names exactly the parameters the
        model samples, and returns chi2 = -2 ln L at that point. The
        pass limit CHI2_TOLERANCE = 0.2 is the project's bound for every
        frozen chi2: compiler and library-version noise stays below it,
        and no physics change is detectable below a chi2 shift of 0.2,
        so a failure means the code or the frozen inputs changed the
        prediction.

        Arguments:
          none (unittest passes the test case itself as self).

        Returns:
          nothing; prints the report block of u.report_chi2_test.

        Raises:
          AssertionError when |chi2 - reference| >= 0.2; RuntimeError
          when the worker exits before writing its result (a cosmolike
          abort prints its reason just above).
        """
        # tatt=False selects the NLA model (the harness flag), the only
        # IA model of the cluster combinations
        chi2 = u.single_model_chi2(EXAMPLE, tatt=False)
        # reference_chi2.json names each entry <example>_<IA model>; the
        # f-string (a string literal prefixed by f, whose {...} fields
        # are replaced by their values) builds "example2_nla"
        ref = self.reference[f"{EXAMPLE}_nla"]
        # report_chi2_test prints chi2, the reference, their difference
        # and the limit; it never asserts (the assertion follows)
        u.report_chi2_test(
            3, "des_cluster example2 (6x2pt + N, NLA) chi2 vs frozen "
               "reference",
            chi2, ref, u.CHI2_TOLERANCE)
        # assertLess(a, b) passes when a < b and fails with msg
        # otherwise; in that message, :.6f prints fixed six decimals
        self.assertLess(
            abs(chi2 - ref), u.CHI2_TOLERANCE,
            msg=f"chi2 = {chi2:.6f} vs frozen reference {ref:.6f} "
                f"(|delta| >= {u.CHI2_TOLERANCE})")

    def test_4_no_race_condition_ten_in_a_row(self):
        """The fiducial as 10th of 10 cosmologies matches a fresh run.

        u.ten_in_a_row_chi2 starts one worker that builds one model and
        evaluates on it, in order: the frozen fiducial point (the fresh
        value), the nine cosmologies of RACE_PERTURBATIONS (each moves
        one or two cosmological parameters: As_1e9, omegam, H0, ns, w
        with w0pwa, or omegab with mnu), and the fiducial again, the
        10th point of the row. Correct code gives the same chi2 both
        times. A table kept from an earlier cosmology (leaked state) or
        two OpenMP threads writing the same memory (a race) shift the
        second value. RACE_TOLERANCE = 1e-4: both values come from the
        same code on the same inputs, so only floating-point noise is
        allowed, while leaked state shifts the chi2 by much more.

        Arguments:
          none (unittest passes the test case itself as self).

        Returns:
          nothing; prints the report block of u.report_race_test.

        Raises:
          RuntimeError from u.assert_omp_threads when OMP_NUM_THREADS
          is not "4", or when the worker exits before writing its
          result; AssertionError when the two chi2 values differ by
          1e-4 or more.
        """
        # with one thread no race can occur: a process that did not get
        # the 4 threads set at the top of this file stops here
        u.assert_omp_threads()
        # the function returns a (fresh, tenth) pair; the assignment
        # unpacks it into the two names
        fresh, tenth = u.ten_in_a_row_chi2(EXAMPLE, tatt=False)
        u.report_race_test(
            4, "des_cluster example2 (6x2pt + N, NLA) race check: "
               "10 cosmologies in a row",
            fresh, tenth, u.RACE_TOLERANCE)
        self.assertLess(
            abs(tenth - fresh), u.RACE_TOLERANCE,
            msg=f"10th-in-a-row chi2 = {tenth:.8f} vs fresh {fresh:.8f}")


# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest. unittest.main runs the tests of this file
# (verbosity=2 prints one line per test).
if __name__ == "__main__":
    unittest.main(verbosity=2)
