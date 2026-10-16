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

# OpenMP reads OMP_NUM_THREADS when the compiled libraries load, so
# this must run before any cobaya/cosmolike import in the process.
os.environ["OMP_NUM_THREADS"] = "4"

import sys
import unittest

# The tests folder is not a package; put it on the import path so the
# shared harness resolves no matter where pytest was launched from.
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
import cocoa_test_utils as u

# The shared factory (cosmolike_core/cocoa_testing_suites.py) builds
# the advisory class; assigning its return value to the usual
# module-level name is what makes pytest collect it here, with
# unchanged test ids. Each roster row is (method name, advisory
# label, example, TATT flag, report label); knob_scan binds the
# example the one-knob-at-a-time scan evaluates.
TestAccuracyAdvisory = u.accuracy_suite(
    u, name="TestAccuracyAdvisory",
    checks=(
        ("test_a1_4x2pt_N_nla", "A1", "example1",
         False, "des_cluster example1 (4x2pt + N, NLA)"),
        ("test_a2_6x2pt_N_nla", "A2", "example2",
         False, "des_cluster example2 (6x2pt + N, NLA)"),
        ("test_a3_cosmic_shear_nla", "A3", "example3",
         False, "des_cluster example3 (cosmic shear, NLA)"),
        ("test_a4_2x2pt_nla", "A4", "example4_2x2pt",
         False, "des_cluster example4_2x2pt (2x2pt, NLA)"),
        ("test_a5_3x2pt_nla", "A5", "example4",
         False, "des_cluster example4 (3x2pt, NLA)"),
        ("test_a6_3x2pt_tatt", "A6", "example4",
         True, "des_cluster example4 (3x2pt, TATT)"),
    ))

# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest
if __name__ == "__main__":
    unittest.main(verbosity=2)
