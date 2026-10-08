"""Accuracy advisory checks A1-A6: default vs high-accuracy settings.

Every reference chi2 of this suite is computed with the examples'
default numerical settings. These checks measure how much numerical
error those defaults carry. Each one re-evaluates a frozen
configuration at its frozen point with the numerical settings pushed
far beyond the defaults (cosmolike: accuracyboost 3,
internal_accuracyboost 2, integration_accuracy 10, lmax 200000,
kmax_boltzmann 40; CAMB: AccuracyBoost 2.0, k_per_logint 50, kmax 50;
the exact values live in cocoa_test_utils.HIGH_ACCURACY_LIKELIHOOD and
cocoa_testing.HIGH_ACCURACY_CAMB_EXTRA_ARGS) and reports

    delta chi2 = chi2(high accuracy) - chi2(default, frozen)

Terms used in this file:

  data vector  d, the measured correlation functions and cluster
               counts, one number per entry (in these tests a
               synthetic vector); the theory vector t is the model's
               prediction for the same entries.
  mask         one 0/1 flag per entry; 0 removes the entry (the scale
               cuts, and the blocks a combination does not use).
  chi2         (d - t)^T C^-1 (d - t), with C^-1 the inverse
               covariance of the entries the mask keeps.
  frozen       the tests' own copy, under tests/frozen/, of each
               example's cobaya configuration with every option and
               parameter written out (cobaya is the sampler framework
               that loads the likelihoods), of its evaluation point (the
               frozen point), of its data files and of the reference
               chi2 values; the SHA-256 hashes of
               tests/manifest_sha256.json pin every byte of it.
  NLA, TATT    the two intrinsic-alignment models: nonlinear linear
               alignment, and tidal alignment and tidal torquing (NLA
               plus second-order tidal terms).

There is NO pass/fail: how much numerical error an analysis tolerates
is a judgment call. The first two checks cover the two cluster
combinations of arXiv 2503.13631, with the NLA intrinsic-alignment
model (the only one the cluster lensing code carries); the other four
cover the galaxy-only likelihoods on the DES-Y3 placeholder data set
(the DES Y3 measurement, copied from the des_y3 project):

  A1. 4x2pt + N (example1), NLA: cluster counts N, cluster lensing,
      cluster clustering, cluster x galaxy, galaxy clustering of
      MagLim lens bins 1-3
  A2. 6x2pt + N (example2), NLA: the blocks of A1 plus cosmic shear,
      galaxy-galaxy lensing and the galaxy clustering of all six lens
      bins
  A3. cosmic shear (example3), NLA
  A4. 2x2pt (example4_2x2pt), NLA: galaxy-galaxy lensing and galaxy
      clustering
  A5. 3x2pt (example4), NLA: cosmic shear, galaxy-galaxy lensing and
      galaxy clustering
  A6. 3x2pt (example4), TATT

TATT gets one check, on 3x2pt: the TATT tables enter cosmic shear and
galaxy-galaxy lensing, and the 3x2pt vector holds both blocks, so the
cosmic-shear and 2x2pt TATT checks would re-measure parts of A6 at
minutes each.

Every check evaluates against a data vector that is, or is close to,
the default-settings model at the frozen point. The galaxy-only checks
use the NLA or TATT vector generated at freeze time
(cocoa_test_utils.SYNTHETIC_VECTORS). The cluster checks use the
frozen copy of the synthetic data vector that
scripts/make_synthetic_data.py wrote as the default-settings model;
those data predate the evolving cb variance of the halo model (project
README), so the frozen cluster references are small but not zero. At
the minimum (t = d), a small theory change dt moves the chi2 by the
quadratic term dt^T C^-1 dt alone. Away from it, the linear term
-2 (d - t)^T C^-1 dt adds up to 2 sqrt(chi2 x dt^T C^-1 dt) of either
sign: small for the cluster references, dominant for a chi2 far from
its minimum, where it mixes the numerical error with the distance to
the data.

The numbers are those of the scale cuts in each example's mask: the
masks hide the smallest scales, where the numerical error is largest
(tests/validation/knob_sweep.py scans the knobs one by one on the
cluster combinations with the small scales visible).

Each evaluation runs in a worker subprocess: a separate python process
that the harness starts for one model and that hands its chi2 back
through a temporary json file. cosmolike keeps the data-vector
dimensions in C global variables and aborts a process that initializes
two configurations with different dimensions, and the examples use two
data sets (2812 entries for the cluster combinations, 900 for the
galaxy-only likelihoods).

The checks are methods of a unittest.TestCase class (python's built-in
test framework: each method whose name starts with test_ is one test);
pytest, the test runner of the command below, collects such classes
too. A high-accuracy evaluation takes minutes, not seconds. To run
this file (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/des_cluster/tests/data_vector/test_accuracy.py
"""

import os

# OpenMP (the threading library of the compiled C code) reads the
# environment variable OMP_NUM_THREADS when the compiled libraries
# load, so this must run before ANY cobaya/cosmolike import in the
# process. os.environ is this process's environment, which the
# processes it starts inherit. 4 = cocoa_testing.REQUIRED_OMP_THREADS,
# the thread count of every test process (the harness gives its worker
# subprocesses the same value).
os.environ["OMP_NUM_THREADS"] = "4"

import sys
import unittest

# sys.path is the list of folders python searches on import, position 0
# first. dirname applied twice to this file's absolute path gives the
# tests/ folder, which holds this project's harness cocoa_test_utils.py
# (imported as u). Searching tests/ first makes the import find this
# project's harness both under pytest and under `python
# test_accuracy.py` (every Cocoa project names its harness
# cocoa_test_utils).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cocoa_test_utils as u


class TestAccuracyAdvisory(unittest.TestCase):
    """The advisory checks A1-A6, one test method each.

    They share one frozen-state verification (setUpClass) and one
    implementation (_accuracy_check), which evaluates a configuration
    in a worker subprocess and prints the report block.
    """

    # @classmethod is a decorator (a line starting with @ that wraps
    # the function defined below it): it hands the method the class
    # itself (cls), not an instance; unittest calls setUpClass once
    # before the first test of the class
    @classmethod
    def setUpClass(cls):
        """Check the environment and the frozen state once for all tests.

        Runs once, before the first test of the class: it moves to
        ROOTDIR (the Cocoa/ folder; the frozen configurations name
        paths relative to it), verifies every frozen file against the
        SHA-256 manifest (an edited frozen state must fail loudly
        before any physics runs), and loads the frozen reference chi2
        values.

        Arguments:
          cls = this test class (the classmethod decorator passes the
                class itself, not an instance).

        Returns:
          nothing; cls.reference holds the dictionary of
          frozen/reference_chi2.json: {"<example>_nla" or
          "<example>_tatt": reference chi2} plus a "_meta" entry.

        Raises:
          RuntimeError when ROOTDIR is not set (start_cocoa.sh was not
          sourced); AssertionError when a frozen file differs from the
          manifest.
        """
        u.require_cocoa_environment()
        u.verify_frozen()
        cls.reference = u.load_reference()

    def _accuracy_check(self, name, example, tatt, label):
        """Evaluate one configuration at high accuracy and print the delta.

        u.single_model_chi2(..., high_accuracy=True) starts a worker
        subprocess (a fresh python process, so no cosmolike state of
        another model can reach it) that loads the frozen
        configuration, applies the high-accuracy settings on top of it,
        builds the cobaya model, checks that the frozen point names
        exactly the model's sampled parameters, evaluates the chi2
        there, and hands it back as a float. The default-settings value
        is the frozen reference of the same configuration.
        u.report_accuracy prints both values and their difference (high
        minus default) and returns that difference, which is not
        asserted: the check is advisory.

        Arguments:
          name    = the advisory label (A1-A6) for the report.
          example = a key of cocoa_test_utils.EXAMPLES.
          tatt    = True evaluates the TATT variant against the
                    TATT-generated data vector (galaxy-only examples
                    only), False the NLA one.
          label   = one line naming the example, combination, and IA
                    model.

        Returns:
          nothing; the report block is printed.

        Raises:
          RuntimeError when the worker exits without a result (a
          cosmolike abort, or a non-finite chi2 in the worker);
          KeyError when the frozen reference file has no entry for the
          configuration.
        """
        chi2_high = u.single_model_chi2(example, tatt, high_accuracy=True)
        # `"tatt" if tatt else "nla"` is python's one-line if-else (the
        # ternary operator of C): "tatt" when tatt is True, else "nla".
        # The frozen reference file names its entries <example>_nla and
        # <example>_tatt; the f-string f"{example}_{suffix}" below fills
        # each {name} with that variable's value, e.g. "example4_tatt"
        suffix = "tatt" if tatt else "nla"
        default_ref = self.reference[f"{example}_{suffix}"]
        u.report_accuracy(f"{name}: {label}", chi2_high, default_ref)

    def test_a1_4x2pt_N_nla(self):
        """A1 measures the numerical error of the 4x2pt + N defaults.

        example1 (des_cluster.combo_4x2pt_N, NLA) is evaluated at high
        accuracy in a worker and compared with its frozen
        default-settings reference (_accuracy_check). Nothing is
        asserted: the test fails only when the evaluation itself fails.

        Arguments:
          none.

        Returns:
          nothing; the report block is printed.
        """
        self._accuracy_check("A1", "example1", False,
                             "des_cluster example1 (4x2pt + N, NLA)")

    def test_a2_6x2pt_N_nla(self):
        """A2 measures the numerical error of the 6x2pt + N defaults.

        example2 (des_cluster.combo_6x2pt_N, NLA) is evaluated at high
        accuracy in a worker and compared with its frozen
        default-settings reference (_accuracy_check). Nothing is
        asserted: the test fails only when the evaluation itself fails.

        Arguments:
          none.

        Returns:
          nothing; the report block is printed.
        """
        self._accuracy_check("A2", "example2", False,
                             "des_cluster example2 (6x2pt + N, NLA)")

    def test_a3_cosmic_shear_nla(self):
        """A3 measures the numerical error of the cosmic-shear defaults.

        example3 (des_cluster.cosmic_shear, NLA) is evaluated at high
        accuracy in a worker, against the NLA vector generated at
        freeze time, and compared with its frozen default-settings
        reference (_accuracy_check). Nothing is asserted: the test
        fails only when the evaluation itself fails.

        Arguments:
          none.

        Returns:
          nothing; the report block is printed.
        """
        self._accuracy_check("A3", "example3", False,
                             "des_cluster example3 (cosmic shear, NLA)")

    def test_a4_2x2pt_nla(self):
        """A4 measures the numerical error of the 2x2pt defaults.

        example4_2x2pt (des_cluster.combo_2x2pt, NLA) is evaluated at
        high accuracy in a worker, against the NLA vector generated at
        freeze time, and compared with its frozen default-settings
        reference (_accuracy_check). Nothing is asserted: the test
        fails only when the evaluation itself fails.

        Arguments:
          none.

        Returns:
          nothing; the report block is printed.
        """
        self._accuracy_check("A4", "example4_2x2pt", False,
                             "des_cluster example4_2x2pt (2x2pt, NLA)")

    def test_a5_3x2pt_nla(self):
        """A5 measures the numerical error of the 3x2pt defaults (NLA).

        example4 (des_cluster.combo_3x2pt, NLA) is evaluated at high
        accuracy in a worker, against the NLA vector generated at
        freeze time, and compared with its frozen default-settings
        reference (_accuracy_check). Nothing is asserted: the test
        fails only when the evaluation itself fails.

        Arguments:
          none.

        Returns:
          nothing; the report block is printed.
        """
        self._accuracy_check("A5", "example4", False,
                             "des_cluster example4 (3x2pt, NLA)")

    def test_a6_3x2pt_tatt(self):
        """A6 measures the numerical error of the 3x2pt defaults (TATT).

        example4 (des_cluster.combo_3x2pt) is evaluated with TATT
        (IA_model 1 and the TATT_POINT values of cocoa_test_utils) at
        high accuracy in a worker, against the TATT vector generated at
        freeze time, and compared with its frozen default-settings
        TATT reference (_accuracy_check). Nothing is asserted: the test
        fails only when the evaluation itself fails.

        Arguments:
          none.

        Returns:
          nothing; the report block is printed.
        """
        self._accuracy_check("A6", "example4", True,
                             "des_cluster example4 (3x2pt, TATT)")


# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest. unittest.main runs the tests of this file
# (verbosity=2 prints one line per test)
if __name__ == "__main__":
    unittest.main(verbosity=2)
