"""Accuracy advisory checks A1-A2: default vs high-accuracy settings.

Every reference in this suite is computed with the examples' default
numerical settings. These checks answer: how much numerical error do
those defaults carry? Each one re-evaluates a frozen configuration at
its frozen point with the numerical knobs pushed far beyond the
defaults (cosmolike: accuracyboost 3, internal_accuracyboost 2,
integration_accuracy 10, lmax 200000, kmax_boltzmann 40; CAMB:
AccuracyBoost 2.0, k_per_logint 50, kmax 50; the exact values live in
cocoa_test_utils.HIGH_ACCURACY_LIKELIHOOD and
cocoa_testing.HIGH_ACCURACY_CAMB_EXTRA_ARGS) and reports

    delta chi2 = chi2(high accuracy) - chi2(default, frozen)

There is NO pass/fail: how much numerical error an analysis tolerates
is a judgment call. The two checks cover the two cluster
combinations, both with the NLA intrinsic-alignment model (the only
one the cluster lensing code carries):

  A1. 4x2pt + N (example1)
  A2. 6x2pt + N (example2)

Every check evaluates against the frozen copy of the synthetic data
vector, which is the default-settings model at the frozen point
(scripts/make_synthetic_data.py), so the chi2 sits at a minimum and
the delta is a stable, quadratic response instead of a linear one.
The numbers are those of the scale cuts in each example's mask: the
masks hide the smallest scales, where the numerical error is largest
(tests/validation/knob_sweep.py scans the knobs one by one with the
small scales visible).

A high-accuracy evaluation takes minutes, not seconds. To run this
file (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/des_cluster/tests/test_accuracy.py
"""

import os

# OpenMP reads OMP_NUM_THREADS when the compiled libraries load, so
# this must run before ANY cobaya/cosmolike import in the process.
os.environ["OMP_NUM_THREADS"] = "4"

import sys
import unittest

# The tests folder is not a package; put it on the import path so the
# shared harness resolves no matter where pytest was launched from.
# insert(0, ...) puts the folder FIRST in the search order, ahead of
# every other place a same-named module could hide.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cocoa_test_utils as u


class TestAccuracyAdvisory(unittest.TestCase):
    """Advisory checks A1-A2, sharing one frozen-state verification.

    setUpClass runs once: it moves to ROOTDIR, verifies every frozen
    file against the SHA-256 manifest (an edited frozen state must
    fail loudly before any physics runs), and loads the frozen
    reference chi2 values.
    """

    # the classmethod decorator hands the method the class itself
    # (cls), not an instance; unittest calls setUpClass once before
    # the first test of the class
    @classmethod
    def setUpClass(cls):
        u.require_cocoa_environment()
        u.verify_frozen()
        cls.reference = u.load_reference()

    def _accuracy_check(self, name, example, label):
        """Evaluate one configuration at high accuracy and report.

        Arguments:
          name    = the advisory label (A1-A2) for the report.
          example = a key of cocoa_test_utils.EXAMPLES.
          label   = one line naming the example and combination.
        """
        # False = the NLA model (the harness's tatt flag), the only IA
        # model this suite evaluates
        chi2_high = u.single_model_chi2(example, False, high_accuracy=True)
        # the "_nla" suffix is the naming the frozen reference file uses
        default_ref = self.reference[f"{example}_nla"]
        u.report_accuracy(f"{name}: {label}", chi2_high, default_ref)

    def test_a1_4x2pt_N_nla(self):
        """A1: 4x2pt + N, NLA, default vs high accuracy."""
        self._accuracy_check("A1", "example1",
                             "des_cluster example1 (4x2pt + N, NLA)")

    def test_a2_6x2pt_N_nla(self):
        """A2: 6x2pt + N, NLA, default vs high accuracy."""
        self._accuracy_check("A2", "example2",
                             "des_cluster example2 (6x2pt + N, NLA)")


# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest
if __name__ == "__main__":
    unittest.main(verbosity=2)
