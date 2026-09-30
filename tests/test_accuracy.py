"""Accuracy advisory checks A1-A3: default vs high-accuracy settings.

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
is a judgment call. The three checks cover the three probes, all with
the NLA intrinsic-alignment model:

  A1. cosmic shear (example1)
  A2. 2x2pt (example2_2x2pt)
  A3. 3x2pt (example2)

Every check evaluates against the synthetic NLA data vector written
at freeze time (cocoa_test_utils.NLA_DATASET), so the chi2 sits at a
minimum and the delta is a stable, quadratic response instead of a
linear one.

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
    """Advisory checks A1-A3, sharing one frozen-state verification.

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
          name    = the advisory label (A1-A3) for the report.
          example = a key of cocoa_test_utils.EXAMPLES.
          label   = one line naming the example and probe.
        """
        # False = the NLA model (the harness's tatt flag), the only IA
        # model this suite evaluates
        chi2_high = u.single_model_chi2(example, False, high_accuracy=True)
        # the "_nla" suffix is the naming the frozen reference file uses
        default_ref = self.reference[f"{example}_nla"]
        u.report_accuracy(f"{name}: {label}", chi2_high, default_ref)

    def test_a1_cosmic_shear_nla(self):
        """A1: cosmic shear, NLA, default vs high accuracy."""
        self._accuracy_check("A1", "example1",
                             "example1 (cosmic shear, NLA)")

    def test_a2_2x2pt_nla(self):
        """A2: 2x2pt, NLA, default vs high accuracy."""
        self._accuracy_check("A2", "example2_2x2pt",
                             "example2_2x2pt (2x2pt, NLA)")

    def test_a3_3x2pt_nla(self):
        """A3: 3x2pt, NLA, default vs high accuracy."""
        self._accuracy_check("A3", "example2",
                             "example2 (3x2pt, NLA)")


# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest
if __name__ == "__main__":
    unittest.main(verbosity=2)
