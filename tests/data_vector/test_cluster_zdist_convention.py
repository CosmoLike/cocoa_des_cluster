"""The z convention of the cluster selection-kernel table.

The cluster table <phi_i|z> (dataset key nz_cluster_file) lists, per row,
a redshift and the probability that a cluster at that true redshift lands
in each cluster bin. Two conventions exist for what the redshift means:

  Z_MID (cluster_zmid_convention = 1, the default): the value belongs at
        the listed z. The des_cluster synthetic tables are written so.
  Z_LOW (cluster_zmid_convention = 0): the listed z is the left edge of a
        cell, and the value belongs at the cell centre, halfway to the next
        row (the last cell repeats the previous width). The DES and
        lighthouse tables are written so, like every galaxy n(z) file
        (photoz_zmid_convention).

The two readings differ by half a cell: for the Cardinal table of the
lighthouse comparison (dz = 0.01) they move the counts of the lowest
cluster bin by 1-2%. set_cluster_zdist applies the convention before it
stores the table (its zmid_convention argument).

The checks, on 4x2pt + N at the fiducial point of arXiv 2503.13631, all
entries computed (all-ones mask), with the dataset's cluster table T:
  1. Passing zmid_convention = 1 explicitly reproduces the default call
     exactly.
  2. T read as Z_LOW equals, entry by entry and exactly, T with its z
     column moved to the cell centres in python (the arithmetic of the C
     code) and read as Z_MID.
  3. The two readings of T differ: the counts move.
  4. Returning to the default restores the default data vector exactly.

The test feeds cosmolike without cobaya, through the likelihood's init
chain (tests/validation/compare_reference.py), in a python process of its
own (@own_process); see test_cluster_area.py for that setup.

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):
    python -m pytest projects/des_cluster/tests/data_vector/test_cluster_zdist_convention.py
"""

import os

# OpenMP reads OMP_NUM_THREADS when the compiled libraries load: set it
# before any cosmolike import (4 = the thread count of every test)
os.environ["OMP_NUM_THREADS"] = "4"

import sys
import tempfile
import unittest

TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TESTS_DIR)
sys.path.insert(0, os.path.join(TESTS_DIR, "validation"))
sys.path.insert(0, os.path.join(TESTS_DIR, "reference"))
import cocoa_test_utils as u

# the encoding of cluster_zmid_convention (structs_cluster.h)
ZLOW = 0
ZMID = 1


def build_state():
    """Initialize cosmolike for 4x2pt + N at the fiducial point.

    The init chain of test_cluster_area.py: the likelihood's
    (compare_reference.init_c), an all-ones mask, the fiducial nuisance
    parameters and one CAMB run of the Python reference.

    Returns:
      (ci, cfg): the compiled interface and the compare_reference
      configuration (its "dataset" keys name the cluster table and the
      nominal bin edges).
    """
    import numpy as np
    import compare_reference as cr
    from reference_cluster import FIDUCIAL, COSMO_KEYS
    from ref_cosmology import Cosmology

    cfg = cr.load_config()
    workdir = tempfile.mkdtemp(prefix="des_cluster_zdist_")
    ci, info = cr.init_c(cfg, workdir, threads=4)
    ones = np.ones(info["ndata"], dtype=int)
    ci.init_data_cluster(info["covf"], cr.write_mask(workdir, "ones.mask", ones),
                         info["dvf"])
    cr.set_nuisance_c(ci, cfg, dict(FIDUCIAL))
    params = {key: FIDUCIAL[key] for key in COSMO_KEYS}
    cr.set_cosmology_c(ci, cr.cosmology_inputs(Cosmology(params), cfg, "production"))
    return ci, cfg


class TestClusterZdistConvention(unittest.TestCase):
    """The Z_LOW / Z_MID reading of the cluster table (checks 1-4)."""

    @classmethod
    def setUpClass(cls):
        """Refuse to run without start_cocoa.sh (ROOTDIR)."""
        u.require_cocoa_environment()

    @u.own_process(__file__)
    def test_cluster_zdist_convention(self):
        """Run checks 1-4 of the module docstring on one cosmolike state."""
        import numpy as np

        ci, cfg = build_state()
        ds = cfg["dataset"]
        table = np.loadtxt(ds["nz_cluster_file"])
        edges = np.array(ds["cluster_zbin_edges"])
        zmin, zmax = edges[:-1].copy(), edges[1:].copy()

        def install(tab, convention=None):
            if convention is None:
                ci.set_cluster_zdist(nofz=tab, zbin_min=zmin, zbin_max=zmax)
            else:
                ci.set_cluster_zdist(nofz=tab, zbin_min=zmin, zbin_max=zmax,
                                     zmid_convention=convention)
            return (np.array(ci.compute_data_vector_cluster_masked()),
                    np.array(ci.N_cluster_tomo()))

        # --- 1. EXPLICIT Z_MID = DEFAULT ---
        dv0, N0 = install(table)
        dv, N = install(table, ZMID)
        self.assertTrue(np.array_equal(dv, dv0), "explicit Z_MID changed the vector")

        # --- 2. Z_LOW = Z_MID OF THE CELL CENTRES ---
        # cell i spans [z_i, z_(i+1)); the last cell repeats the previous
        # width (the formulas of set_cluster_zdist)
        z = table[:, 0]
        centres = np.empty_like(z)
        centres[:-1] = 0.5 * (z[:-1] + z[1:])
        centres[-1] = z[-1] + 0.5 * (z[-1] - z[-2])
        moved = table.copy()
        moved[:, 0] = centres
        dv_low, N_low = install(table, ZLOW)
        dv_mid, N_mid = install(moved, ZMID)
        self.assertTrue(np.array_equal(dv_low, dv_mid),
                        "Z_LOW is not the Z_MID reading of the cell centres")

        # --- 3. THE TWO READINGS DIFFER ---
        shift = np.abs(N_low / N0 - 1).max()
        print(f"  counts, Z_LOW vs Z_MID: max |N/N0 - 1| = {shift:.2e}")
        self.assertGreater(shift, 1.0e-4, "the convention did not reach the counts")

        # --- 4. BACK TO THE DEFAULT ---
        dv, N = install(table)
        self.assertTrue(np.array_equal(dv, dv0), "the default did not come back")


if __name__ == "__main__":
    unittest.main()
