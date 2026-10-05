"""Unit tests 3-4: the 6x2pt + N likelihood on the frozen test data.

6x2pt + N is the full cluster-plus-3x2pt combination (CL+3x2pt of
arXiv 2503.13631): the 4x2pt + N blocks of example1 (cluster number
counts N, cluster lensing, cluster clustering, the cluster x galaxy
cross-correlation) joined with the galaxy 3x2pt (cosmic shear,
galaxy-galaxy lensing, and galaxy clustering of all six MagLim lens
bins). Here it is the des_cluster.combo_6x2pt_N likelihood, evaluated
on the frozen copy of example2's configuration (see cocoa_test_utils
for what "frozen" means and why). The frozen data vector is the model
at the frozen fiducial point, so the reference chi2 sits at zero. The
two tests:

  3. chi2 at the frozen fiducial point, within CHI2_TOLERANCE (0.2) of
     the frozen reference value.
  4. race check: on one model, the fiducial evaluated fresh and again
     as the 10th of 10 cosmologies in a row must agree to
     RACE_TOLERANCE (1e-4). A disagreement means state leaked between
     evaluations or OpenMP threads raced.

Both run the NLA intrinsic-alignment model, the only one the cluster
lensing code carries (there are no TATT variants of these tests).

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/des_cluster/tests/data_vector
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

EXAMPLE = "example2"


class TestExample2Combo6x2ptN(unittest.TestCase):
    """Tests 3-4, sharing one frozen-state verification.

    setUpClass runs once before the tests: it moves to ROOTDIR,
    verifies every frozen file against the SHA-256 manifest (an edited
    frozen state must fail loudly before any physics runs), and loads
    the frozen reference chi2 values.
    """

    # the classmethod decorator hands the method the class itself
    # (cls), not an instance; unittest calls setUpClass once before
    # the first test of the class
    @classmethod
    def setUpClass(cls):
        u.require_cocoa_environment()
        u.verify_frozen()
        cls.reference = u.load_reference()

    def test_3_chi2_matches_frozen_reference(self):
        """chi2 at the frozen NLA point stays within 0.2 of the reference."""
        # tatt=False = the NLA model (the harness's flag), the only IA
        # model this suite evaluates
        chi2 = u.single_model_chi2(EXAMPLE, tatt=False)
        # the "_nla" suffix is the naming the frozen reference file uses
        ref = self.reference[f"{EXAMPLE}_nla"]
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
        """The fiducial as 10th of 10 cosmologies matches a fresh run."""
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
# idle under pytest
if __name__ == "__main__":
    unittest.main(verbosity=2)
