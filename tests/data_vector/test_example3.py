"""Unit tests 5-8: the cosmic-shear likelihood on the frozen test data.

Cosmic shear is the correlation of galaxy shape distortions produced by
weak gravitational lensing; here it is the des_cluster.cosmic_shear
likelihood, one of the galaxy-only likelihoods of this project,
evaluated on the frozen copy of example3's configuration.

Terms used below:

  frozen       stored once under tests/frozen/ and never edited by
               hand: the complete configuration with its evaluation
               point, the fiducial point of these tests
               (frozen_config_example3.py, read by cobaya, the sampler
               framework that loads the likelihoods), the tests' own
               copy of the data files (data/), and the reference chi2
               values (reference_chi2.json).
               tests/manifest_sha256.json holds the SHA-256 hash of
               each frozen file (a fingerprint that changes when any
               byte changes); the tests refuse to run on a mismatch.
  data vector  every measured value the likelihood compares with its
               model, in one fixed order: here the 900 entries ss, gs,
               gg (xi+ and xi-, gamma_t, w(theta)) of the DES Y3
               placeholder data set. The cosmic-shear likelihood keeps
               the ss block (cosmolike zeroes the mask of the two
               others), and the scale-cut mask 3x2pt_baseline.mask
               drops the smallest angular bins.
  chi2         (d - m)^T C^-1 (d - m) over the kept entries, with d
               the data vector, m the model vector and C their
               covariance.
  worker       a separate python process the harness starts for one
               evaluation, waiting for its result: the cosmolike C
               library keeps the data-vector size in global variables
               and aborts a process that initializes a second data set
               of another size, and this suite uses two (2812 and 900
               entries).

The shipped data vector of the DES Y3 placeholder is a real
measurement, and the frozen point is not a fit to it: there the chi2
sits far from its minimum, where a tiny numerical change of the model
moves it linearly and reads as a large shift. The tests therefore
evaluate against model vectors generated when the frozen state was
created, from the 3x2pt configuration (example4: same cosmology, source
parameters and numerical settings): the NLA tests against
synthetic_des_y3.dataset (NLA at the frozen point), the TATT tests
against tatt_des_y3.dataset (TATT at the TATT point). The stored
references are zero up to the rounding of those vectors, which keep 9
significant digits: chi2 = 1.8e-14 for NLA and for TATT. The four
tests:

  5. chi2 at the frozen fiducial point, within CHI2_TOLERANCE (0.2) of
     the frozen reference value.
  6. race check: on one model, the fiducial evaluated fresh and again
     as the 10th of 10 cosmologies in a row must agree to
     RACE_TOLERANCE (1e-4). A disagreement means state leaked between
     evaluations (a table kept from an earlier cosmology) or OpenMP
     threads raced (two of the CPU threads that share cosmolike's
     loops wrote the same memory).
  7. the same comparison as test 5 with the TATT intrinsic-alignment
     model (IA_model: 1) and DES_A2_1 = 0.05, DES_BTA_1 = 0.05,
     DES_A2_2 = -1.51541 replacing the NLA point's values.
  8. the same race check as test 6 with the TATT model.

NLA (nonlinear alignment) makes the intrinsic alignment of source
galaxy shapes linear in the tidal field (amplitude A1). TATT (tidal
alignment and tidal torquing) adds the torquing term, quadratic in the
tidal field (amplitude A2), and the density weighting of the alignment
term (BTA). The NLA point sets DES_A2_1 = DES_BTA_1 = 0, which switches
those terms off; the TATT tests write cocoa_test_utils.TATT_POINT over
the point instead.

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/des_cluster/tests/data_vector/test_example3.py

The folder path ./projects/des_cluster/tests/data_vector in place of
the file runs the whole data-vector sector.
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
# the chi2 + race test class for one frozen example; assigning its
# return value to the usual module-level name is what makes pytest
# collect it here, with unchanged test ids. The numbers name the
# methods and the printed reports; the label opens every report line.
TestExample3CosmicShear = u.example_suite(
    u, name="TestExample3CosmicShear", example="example3",
    label="des_cluster example3 (cosmic shear",
    numbers=("5", "6"), tatt_numbers=("7", "8"))

# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest
if __name__ == "__main__":
    unittest.main(verbosity=2)
