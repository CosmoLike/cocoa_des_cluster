"""Accuracy advisory checks A1-A6: default vs high-accuracy settings.

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
is a judgment call. The first two checks cover the two cluster
combinations, with the NLA intrinsic-alignment model (the only one
the cluster lensing code carries); the other four cover the
galaxy-only likelihoods on the DES-Y3 placeholder data set:

  A1. 4x2pt + N (example1), NLA
  A2. 6x2pt + N (example2), NLA
  A3. cosmic shear (example3), NLA
  A4. 2x2pt (example4_2x2pt), NLA
  A5. 3x2pt (example4), NLA
  A6. 3x2pt (example4), TATT

TATT gets one check, on 3x2pt: the TATT tables enter cosmic shear and
galaxy-galaxy lensing, and the 3x2pt vector holds both blocks, so the
cosmic-shear and 2x2pt TATT checks would re-measure parts of A6 at
minutes each.

The cluster checks evaluate against the frozen copy of the synthetic
data vector, which is the default-settings model at the frozen point
(scripts/make_synthetic_data.py); the galaxy-only checks evaluate
against the NLA or TATT vector generated at freeze time
(cocoa_test_utils.SYNTHETIC_VECTORS). Either way the chi2 sits at a
minimum and the delta is a stable, quadratic response instead of a
linear one. The numbers are those of the scale cuts in each example's
mask: the masks hide the smallest scales, where the numerical error
is largest (tests/validation/knob_sweep.py scans the knobs one by one
on the cluster combinations with the small scales visible).

A high-accuracy evaluation takes minutes, not seconds. To run this
file (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/des_cluster/tests/data_vector/test_accuracy.py
"""

import os

# OpenMP reads OMP_NUM_THREADS when the compiled libraries load, so
# this must run before ANY cobaya/cosmolike import in the process.
os.environ["OMP_NUM_THREADS"] = "4"

import sys
import unittest

# The harness stays in the parent tests/ folder. Add it explicitly so
# direct execution and worker processes resolve this project's stored inputs.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cocoa_test_utils as u


class TestAccuracyAdvisory(unittest.TestCase):
    """Advisory checks A1-A6, sharing one frozen-state verification.

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

    def _accuracy_check(self, name, example, tatt, label):
        """Evaluate one configuration at high accuracy and report.

        Arguments:
          name    = the advisory label (A1-A6) for the report.
          example = a key of cocoa_test_utils.EXAMPLES.
          tatt    = True evaluates the TATT variant against the
                    TATT-generated data vector (galaxy-only examples
                    only), False the NLA one.
          label   = one line naming the example, combination, and IA
                    model.
        """
        chi2_high = u.single_model_chi2(example, tatt, high_accuracy=True)
        # ternary: the reference key ends in "tatt" or "nla", the
        # naming the frozen reference file uses
        suffix = "tatt" if tatt else "nla"
        default_ref = self.reference[f"{example}_{suffix}"]
        u.report_accuracy(f"{name}: {label}", chi2_high, default_ref)

    def test_a1_4x2pt_N_nla(self):
        """A1: 4x2pt + N, NLA, default vs high accuracy."""
        self._accuracy_check("A1", "example1", False,
                             "des_cluster example1 (4x2pt + N, NLA)")

    def test_a2_6x2pt_N_nla(self):
        """A2: 6x2pt + N, NLA, default vs high accuracy."""
        self._accuracy_check("A2", "example2", False,
                             "des_cluster example2 (6x2pt + N, NLA)")

    def test_a3_cosmic_shear_nla(self):
        """A3: cosmic shear, NLA, default vs high accuracy."""
        self._accuracy_check("A3", "example3", False,
                             "des_cluster example3 (cosmic shear, NLA)")

    def test_a4_2x2pt_nla(self):
        """A4: 2x2pt, NLA, default vs high accuracy."""
        self._accuracy_check("A4", "example4_2x2pt", False,
                             "des_cluster example4_2x2pt (2x2pt, NLA)")

    def test_a5_3x2pt_nla(self):
        """A5: 3x2pt, NLA, default vs high accuracy."""
        self._accuracy_check("A5", "example4", False,
                             "des_cluster example4 (3x2pt, NLA)")

    def test_a6_3x2pt_tatt(self):
        """A6: 3x2pt, TATT, default vs high accuracy."""
        self._accuracy_check("A6", "example4", True,
                             "des_cluster example4 (3x2pt, TATT)")


# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest
if __name__ == "__main__":
    unittest.main(verbosity=2)
