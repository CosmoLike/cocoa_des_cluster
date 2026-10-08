"""Unit test: the runtime n(z) photo-z conventions.

The likelihood reads each tomographic redshift distribution n(z) from
a table file (a z column, then one column per bin) and turns it into
the smooth function the Limber integrals evaluate. Two likelihood yaml
keys, read at run time, control that step; both default to 0, the
value the likelihood yamls of this project declare and the default of
cosmolike itself (structs.c):

  photoz_interpolation_type - the stage-1 interpolant of the two-stage
      n(z) scheme of cosmolike's redshift_spline.c. Stage 1 runs when
      the table is (re)built: a GSL interpolant through the nodes of
      the file, resampled on a fine uniform grid; stage 2, the hot
      path, evaluates cubic pieces on that grid. 0 = cubic spline (the
      default), 1 = linear, 2+ = Steffen monotone (no overshoot:
      between two non-negative nodes the stage-1 interpolant cannot
      dip below zero, which a cubic spline can do next to a sharp
      feature in the table).
  photoz_zmid_convention - how the z column of the n(z) file is read:
      0 = Z_LOW (the default; the column holds left bin edges, so the
      tabulated value belongs at the cell center z + dz/2), 1 = Z_MID
      (the column holds the sample points themselves). The two
      readings differ by a rigid dz/2 shift of every distribution. The
      source n(z) file of this data set has dz = 0.01, a shift of
      0.005 in redshift; with the same n(z) file, covariance and mask,
      the des_y3 project records delta chi2 = 0.30 between the two
      readings (its tests/data_vector/README.md).

Every likelihood of this project reads the two keys, the cluster
combinations included (their lens and source n(z) files hold Z_LOW
columns; the cluster selection table des_y6_cluster.nz has its own
reader, which the keys do not touch), and no other test of this
project moves them.

Terms used below: frozen means stored once under tests/frozen/ and
never edited by hand (the complete configuration with its evaluation
point, the fiducial point, the tests' own copy of the data files, and
the reference chi2 values), with every file pinned by its SHA-256 hash
(a fingerprint that changes when any byte changes) in
tests/manifest_sha256.json. The data vector is the list of every
measured value the likelihood compares with its model, and the
scale-cut mask marks the entries the chi2 uses; the other entries are
masked.

This test evaluates the frozen cosmic-shear fiducial (example3) under
five settings IN ONE PROCESS: the default, each alternative, and the
default again. Cosmic shear reads only the source n(z), so the test
probes that table. For each alternative it measures

    delta chi2 = delta^T C^-1 delta,
    delta = dv(alternative) - dv(default),

with dv the 900-entry model data vector the likelihood prints (zero at
the masked entries) and C^-1 the masked inverse covariance from the
compiled interface: the chi2 the alternative would score against a
data set whose data vector is the default prediction. delta is the
difference of two model vectors, so the number does not depend on the
data vector the likelihood reads.

Running everything in one process is the point, not a convenience:
cosmolike keeps the n(z) tables, and the tables built from them, in C
static memory, which survives from one model to the next, so each
model must rebuild them with its own setting. A dead flag (a key that
never reaches the C code) or a stale table makes delta exactly zero,
which the floors DCHI2_FLOORS catch, and the final return to the
default must reproduce the first vector (the round-trip assertion),
which a table left in the state of an alternative fails. The vectors
are compared as the likelihood prints them (print_datavector, 9
significant digits), so the round trip is exact to that precision.

That one process is a python process of the test's own
(cocoa_test_utils.own_process): under pytest the collecting process
also holds the cluster models of test_cache_consistency.py (the
sector ladder, which builds its 2812-entry models in pytest's own
process), and cosmolike aborts a process that initializes two data
sets of different dimensions (here 2812 and 900 entries). In the
pytest process the decorated test method runs this file again
with the same python executable,

    python <this file> TestPhotozConventions.test_photoz_conventions

with COCOA_TESTS_OWN_PROCESS=1 and OMP_NUM_THREADS=4 in the child's
environment, and passes when the child exits with code 0; in the
child the method body runs.

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/des_cluster/tests/data_vector/test_photoz_conventions.py
"""

import os

# os.environ is the table of environment variables of this process
# (text settings that the processes it starts inherit). OpenMP, the
# library that spreads cosmolike's C loops over CPU threads, reads
# OMP_NUM_THREADS when the compiled libraries load, so this must run
# before ANY cobaya/cosmolike import in the process. "4" is the
# REQUIRED_OMP_THREADS of the harness, the thread count of the frozen
# references.
os.environ["OMP_NUM_THREADS"] = "4"

import shutil
import sys
import tempfile
import unittest

# The harness cocoa_test_utils.py sits one folder up, in tests/
# (dirname twice: this file -> data_vector/ -> tests/). Python searches
# the folders listed in sys.path for modules, a list that holds this
# file's own folder but not tests/, so insert(0, ...) puts tests/
# first; the import then works both when pytest imports this file and
# when python runs it directly. The harness is imported as u.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cocoa_test_utils as u

# The frozen cosmic-shear configuration (des_cluster.cosmic_shear on the
# DES Y3 placeholder data set, evaluated against its generated NLA
# vector) and its entry in reference_chi2.json.
EXAMPLE = "example3"
REFERENCE_KEY = "example3_nla"

# (report tag, photoz_interpolation_type, photoz_zmid_convention): the
# default first and last (the round trip), and between them one
# alternative per row, each changing one key from the default
SETTINGS = (
    ("cspline/Z_LOW (default)", 0, 0),
    ("linear", 1, 0),
    ("steffen", 2, 0),
    ("Z_MID", 0, 1),
    ("default again (round trip)", 0, 0),
)

# Each alternative must change the data vector: a dead flag or a stale
# n(z) table gives delta chi2 = 0 exactly. The floors sit a factor 30
# (Z_MID) to 31000 (linear) below the delta chi2 values the des_y3
# project records at its own frozen cosmic-shear point for the same
# source n(z) file, covariance and mask (3.1e-4 linear, 1.6e-5 Steffen,
# 0.30 Z_MID; its tests/data_vector/README.md), so only a change of a
# delta chi2 by more than that factor can trip them.
DCHI2_FLOORS = {"linear": 1.0e-8, "steffen": 1.0e-8, "Z_MID": 1.0e-2}


class TestPhotozConventions(unittest.TestCase):
    """The five-setting sweep, sharing one frozen-state verification.

    unittest.TestCase is the base class of Python's built-in test
    framework, and pytest, the runner of the command in the module
    docstring, collects such classes too: every method whose name
    starts with test_ is one test.
    """

    # @classmethod is a decorator (a function applied to the method
    # defined just below it): it hands the method the class itself
    # (cls), not an instance; unittest calls setUpClass once before
    # the first test of the class
    @classmethod
    def setUpClass(cls):
        """Check the environment and the frozen state once, before the test.

        u.require_cocoa_environment stops with an instruction when
        start_cocoa.sh was not sourced (ROOTDIR unset) and moves the
        process to ROOTDIR, the folder the relative paths of the frozen
        configuration start from. u.verify_frozen hashes the frozen
        files under tests/frozen/ and compares the hashes with
        tests/manifest_sha256.json, so an edited, missing or added file
        fails the test before a model is built. u.load_reference reads
        reference_chi2.json. All of this runs in the pytest process and
        again in the child process of own_process.

        Arguments:
          cls = the test class (see the comment above); an attribute
                set on it is visible in the test as self.<name>.

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

    # the five models are built in process: see the module docstring
    # for why they must not share pytest's process; the decorator
    # u.own_process(__file__) wraps the method so that, in the pytest
    # process, it runs this file again in a child process
    @u.own_process(__file__)
    def test_photoz_conventions(self):
        """Each photo-z setting changes cosmic shear; the default returns.

        Builds five models of the frozen cosmic-shear configuration in
        this process, one per entry of SETTINGS, evaluates each at the
        frozen point, and reads back the model vector the likelihood
        prints. The checks, in order: each alternative moves delta chi2
        above its floor in DCHI2_FLOORS; the default chi2 stays within
        CHI2_TOLERANCE = 0.2 of the frozen reference, the bound of every
        frozen chi2 check (no physics change is detectable below a chi2
        shift of 0.2); the round trip reproduces the default vector to
        the 9 printed significant digits.

        Arguments:
          none (unittest passes the test case itself as self).

        Returns:
          nothing; prints the chi2 of each setting and the delta chi2 of
          each alternative. The printed vectors go to a temporary
          directory that is deleted when the test ends.

        Raises:
          RuntimeError when the likelihood wrote no vector file, or
          AssertionError when a check fails (both in the child process);
          in the pytest process the decorator reports a nonzero exit of
          the child as one AssertionError naming the test.
        """
        # numpy (the array library) and the compiled interface are
        # imported here, so only the child process loads them; ci is the
        # python module of the compiled cosmolike interface (C++
        # functions callable from python)
        import numpy as np
        import cosmolike_des_cluster_interface as ci

        # mkdtemp creates a new empty directory with a unique name in the
        # system's temporary folder and returns its path, so the printed
        # vectors never land inside the project. addCleanup registers a
        # call that unittest makes when the test ends, passed or failed:
        # shutil.rmtree(vectors_dir, ignore_errors=True) deletes the
        # directory with its files, and a deletion error cannot replace
        # the test result.
        vectors_dir = tempfile.mkdtemp(prefix="photoz_conventions_")
        self.addCleanup(shutil.rmtree, vectors_dir, ignore_errors=True)

        # results maps a report tag to its (chi2, printed vector) pair;
        # icov is read once, from the first model (None = not read yet)
        results = {}
        icov = None
        for tag, interp, zmid in SETTINGS:
            print(f"  building model ({EXAMPLE}, NLA, {tag}) ...",
                  flush=True)
            # load_frozen_info returns the frozen configuration as a
            # dictionary (absolute data path, NLA, the generated NLA
            # data vector, printing off)
            info = u.load_frozen_info(EXAMPLE, tatt=False)
            # likelihood_block is the likelihood's own entry of info, the
            # same dictionary and not a copy, so the keys written into it
            # reach the model built below (the trailing backslash
            # continues the statement on the next line)
            likelihood_block = \
                info["likelihood"][u.EXAMPLES[EXAMPLE]["likelihood"]]
            likelihood_block["photoz_interpolation_type"] = interp
            likelihood_block["photoz_zmid_convention"] = zmid
            # the likelihood writes its masked model vector to this file
            # at every evaluation (index and value, the value as %1.8e:
            # 9 significant digits); the round trip reuses the default's
            # file name dv_0_0, which is harmless because each vector is
            # read back right after its evaluation
            vector_path = os.path.join(
                vectors_dir, f"dv_{interp}_{zmid}.modelvector")
            likelihood_block["print_datavector"] = True
            likelihood_block["print_datavector_file"] = vector_path
            # make_model builds the cobaya model (CAMB plus the
            # likelihood, which hands the two keys to cosmolike);
            # build_point returns the frozen point after checking that
            # it names exactly the parameters the model samples
            model = u.make_model(info)
            point = u.build_point(model, EXAMPLE, tatt=False)
            chi2 = u.evaluate_chi2(model, point)
            # a missing file means the likelihood did not print: stop
            # with its path rather than fail inside the loader
            if not os.path.isfile(vector_path):
                raise RuntimeError(
                    f"print_datavector wrote no file at {vector_path}")
            # _load_datavector returns the value column of the file as a
            # 1D numpy array (900 entries)
            results[tag] = (chi2, u._load_datavector(vector_path))
            if icov is None:
                # all five settings share one dataset (mask, covariance),
                # so the masked inverse covariance of the first build
                # serves every comparison
                icov = np.array(ci.get_inv_cov_masked())

        # SETTINGS[0][0] is the tag of the first setting, the default;
        # its stored (chi2, vector) pair unpacks into two names
        chi2_default, dv_default = results[SETTINGS[0][0]]

        print(f"\n  chi2 report ({EXAMPLE}, NLA):")
        print(f"    default: chi2 = {chi2_default:.6f} "
              f"(frozen reference {self.reference[REFERENCE_KEY]:.6f})")
        # the three alternatives: @ is the matrix product, so
        # delta @ icov @ delta = delta^T C^-1 delta, one number; {tag:8s}
        # pads the tag to 8 characters, and :.6e prints six decimals in
        # exponent notation
        for tag in ("linear", "steffen", "Z_MID"):
            chi2, dv = results[tag]
            delta = dv - dv_default
            dchi2 = float(delta @ icov @ delta)
            print(f"    {tag:8s}: chi2 = {chi2:.6f}, "
                  f"delta^T C^-1 delta vs default = {dchi2:.6e}")
            self.assertGreater(
                dchi2, DCHI2_FLOORS[tag],
                f"{tag}: the flag change was not seen by the n(z) "
                "cache (delta chi2 at or below the dead-flag floor)")

        # the default evaluation passes the same check as the
        # frozen-reference tests (test 5 of test_example3.py):
        # |chi2 - reference| < CHI2_TOLERANCE = 0.2
        self.assertLess(
            abs(chi2_default - self.reference[REFERENCE_KEY]),
            u.CHI2_TOLERANCE)

        # round trip: after the three alternatives, the default settings
        # must reproduce the first printed vector entry for entry
        # (np.array_equal compares every value exactly, here at the 9
        # printed significant digits): the n(z) tables were rebuilt back
        # to the same state. SETTINGS[-1] is the last setting, and _
        # discards its chi2.
        _, dv_return = results[SETTINGS[-1][0]]
        self.assertTrue(
            np.array_equal(dv_return, dv_default),
            "returning to the default settings did not reproduce the "
            "default data vector bit for bit; the n(z) cache did not "
            "rebuild cleanly on the way back")


# __name__ is "__main__" only when this file runs as a script: the
# own_process child runs it so, with the test name as its argument, and
# unittest.main runs that one test (every test of the file when no name
# is given). pytest imports the module instead, so this block stays
# idle under pytest.
if __name__ == "__main__":
    unittest.main(verbosity=2)
