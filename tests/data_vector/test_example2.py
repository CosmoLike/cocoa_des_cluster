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
TestExample2Combo6x2ptN = u.example_suite(
    u, name="TestExample2Combo6x2ptN", example="example2",
    label="des_cluster example2 (6x2pt + N",
    numbers=("3", "4"))

# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest
if __name__ == "__main__":
    unittest.main(verbosity=2)
