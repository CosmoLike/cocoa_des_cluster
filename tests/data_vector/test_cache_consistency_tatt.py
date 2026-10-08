"""Unit test: sector-wise cache invalidation under TATT (the parameter ladder).

cosmolike keeps every expensive intermediate result in a table that
survives from one evaluation to the next (C variables declared static
or global, which keep their value between calls): cosmology
(distances, growth, power-spectrum tables, and under TATT the FAST-PT
tables, the perturbation-theory integrals of the linear power
spectrum that the second-order alignment terms need), intrinsic
alignment, photo-z shifts (n(z) splines, lens efficiencies). Each
group of inputs that feeds a table (a sector of this test) carries a
cache key, a 64-bit number that cosmolike draws anew whenever one of
the sector's input values changes; a table stores the keys of every
sector it was built from and refills when one of them differs. The
shear calibrations m have no table: they rescale the data vector. A
partial-invalidation bug (one sector's update path failing to rebuild
a table another sector consumes, or a table that leaves one of its
sectors out of its key list) produces silently wrong data vectors only
in MIXED update sequences, where one sector moves while the others
stay; the per-point tests (test_example*.py) never run those.

NLA (nonlinear linear alignment) and TATT (tidal alignment and tidal
torquing: NLA plus second-order tidal terms) are the two
intrinsic-alignment (IA) models of the source galaxies. The NLA
ladder of this project is test_cache_consistency.py: it walks the two
cluster combinations, whose 6x2pt + N vector holds the cosmic shear,
galaxy-galaxy lensing and galaxy clustering blocks. The cluster
lensing code has no TATT, so that ladder never reaches the FAST-PT
tables or the code that rebuilds them. This file is the TATT ladder:
the same walk on the galaxy-only 3x2pt likelihood
(des_cluster.combo_3x2pt, example4), the widest vector the TATT tables
enter, evaluated against the TATT data vector generated at freeze time
(tests/frozen/, the tests' own copy of configurations and data, pinned
by the SHA-256 hashes of tests/manifest_sha256.json).

The test walks a deterministic ladder IN ONE PROCESS, evaluating the
model after every step (each sector's later steps keep the earlier
sectors at their last values, so the ladder ends at one well-defined
point):

    3 x cosmology-only steps   (omegam, H0, As_1e9)
    3 x IA-only steps          (A1, A2 and BTA)
    3 x source-photo-z steps   (every DZ_S shift)
    3 x lens-photo-z steps     (every DZ_L shift and the DZ2_L stretch)
    3 x shear-calibration steps (every M)

(A1 = DES_A1_1 and DES_A1_2, amplitude and redshift power of the term
linear in the tidal field; A2 = DES_A2_1 and DES_A2_2, the same for
the quadratic, tidal-torquing term; BTA = DES_BTA_1, b_TA, the
amplitude of the density-weighting term.)

It records the final data vector, then evaluates one SCRAMBLE point
(every sector moved at once, galaxy bias and point mass included; the
chi2 is discarded) and returns to the ladder's final point: the
pipeline must reproduce the recorded vector bit for bit (every double
identical). A second model instance walks the MIRRORED ladder (M ->
DZ_L -> DZ_S -> IA -> cosmology) to the same final point: the answer
must depend on the point, never on the invalidation history.

Assertions:
  1. every ladder step changes the data vector (a dead sector flag
     would pass the later checks vacuously);
  2. each M-only step rescales the masked vector by the analytic
     (1+m_i)(1+m_j) block factors to 1e-12 relative: cosmic shear by
     both bins' factors, gamma_t by the source factor, galaxy
     clustering w(theta) by nothing;
  3. a no-op update (re-sending the current point) leaves the vector
     bitwise unchanged;
  4. after the scramble, returning to the ladder's final point
     reproduces the recorded vector and chi2 bit for bit;
  5. the mirrored-order instance lands on the same final vector bit
     for bit.

Assertion 1 checks the whole vector, not block by block, and this
ladder has no fresh-process check (assertion 7 of the cluster ladder).

Every evaluation forces a full recomputation: cobaya (the sampler
framework that runs CAMB and the likelihood) can hand back a stored
result when it sees the same parameters again, and u.evaluate_chi2
switches that memoization off (cached=False), so each assertion tests
cosmolike's own invalidation.

The ladder runs in a python process of its own: the decorator
cocoa_test_utils.own_process restarts this file in a fresh python that
runs only this test, and the test passes when that process exits with
code 0. Under pytest the collecting process also holds the cluster
ladder's models (the 2812-entry joint vector), and cosmolike aborts a
process that initializes two data sets of different dimensions (the
3x2pt vector has 900 entries).

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest projects/des_cluster/tests/data_vector/test_cache_consistency_tatt.py
"""

import os

# OpenMP (the threading library of the compiled C code) reads the
# environment variable OMP_NUM_THREADS when the compiled libraries
# load, so this must run before ANY cobaya/cosmolike import in the
# process. os.environ is this process's environment, which the
# processes it starts inherit. 4 = cocoa_testing.REQUIRED_OMP_THREADS,
# the thread count of every test process (own_process sets the same
# value for the process it starts).
os.environ["OMP_NUM_THREADS"] = "4"

import re
import sys
import unittest

# sys.path is the list of folders python searches on import, position 0
# first. dirname applied twice to this file's absolute path gives the
# tests/ folder, which holds this project's harness cocoa_test_utils.py
# (imported as u). Searching tests/ first makes the import find this
# project's harness under pytest and when own_process runs this file as
# a script; every Cocoa project names its harness cocoa_test_utils.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cocoa_test_utils as u

# The galaxy-only 3x2pt configuration (des_cluster.combo_3x2pt): cosmic
# shear, galaxy-galaxy lensing and galaxy clustering of the DES Y3
# placeholder data set.
EXAMPLE = "example4"

# Sector membership by sampled-parameter name. Every sampled parameter
# must fall in a named sector (checked at run time: a parameter landing
# in "other" fails the test, so a nuisance parameter added to a
# likelihood cannot stay outside the ladder unnoticed); the bias and
# point-mass sectors move only in the scramble step. _DZ2?_L matches
# the lens photo-z shifts (DZ_L) and the lens photo-z stretch (DZ2_L),
# which this project's redMaGiC parameters sample for bin 5. re.compile
# turns a regular expression (a text pattern) into a pattern object
# whose .search finds the pattern anywhere in a name: ^ and $ anchor
# the start and the end of the name, | separates alternatives, (...)
# groups them, [0-9]+ is one or more digits, ? makes the character
# before it optional, and . is any single character, so the last entry
# catches every name the others miss. _sector_of takes the first entry
# that matches, so the order matters.
SECTORS = (
    ("cosmo", re.compile(r"^(As_1e9|H0|ns|omegab|omegam|mnu|w|w0pwa)$")),
    ("ia", re.compile(r"_A1_|_A2_|_BTA_")),
    ("dz_source", re.compile(r"_DZ_S")),
    ("dz_lens", re.compile(r"_DZ2?_L")),
    ("m", re.compile(r"_M[0-9]+$")),
    ("bias", re.compile(r"_B1_|_B2_|_BMAG_")),
    ("pm", re.compile(r"_PM[0-9]+$")),
    ("other", re.compile(r".")),
)

# Ladder phases (sector order of the forward walk).
PHASES = ("cosmo", "ia", "dz_source", "dz_lens", "m")
# Per-step offsets: a moved parameter takes the value fiducial + step *
# offset (deterministic). A key is either a parameter name (a str) or a
# compiled pattern that every parameter of a family matches (all four
# DES_DZ_S, for instance); _deltas_for gives each sampled name the
# offset of the first key that matches it. Sampled parameters without an
# offset stay at the fiducial (ns, omegab, mnu, w and w0pwa of the
# cosmology). Each ladder offset must change the data vector (assertion
# 1), and the scramble point (SCRAMBLE_STEP offsets from the fiducial)
# stays inside every prior of the frozen configuration: a point outside
# a prior evaluates to -inf, which u.evaluate_chi2 turns into a test
# failure.
DELTAS = {
    "cosmo": {"omegam": 0.002, "H0": 0.2, "As_1e9": 0.02},
    "ia": {re.compile(r"_A1_1$"): 0.05, re.compile(r"_A1_2$"): 0.05,
           re.compile(r"_A2_1$"): 0.05, re.compile(r"_A2_2$"): 0.05,
           re.compile(r"_BTA_1$"): 0.05},
    "dz_source": {re.compile(r"_DZ_S"): 0.001},
    "dz_lens": {re.compile(r"_DZ2?_L"): 0.001},
    "m": {re.compile(r"_M[0-9]+$"): 0.005},
    "bias": {re.compile(r"_B1_"): 0.05},
    "pm": {re.compile(r"_PM[0-9]+$"): 0.05},
}
# Steps per sector. Each step must change the vector again, so a table
# that refills only on the first change after a build fails from the
# second step on.
NSTEP = 3
# The scramble moves every sector one step beyond the last ladder step
# NSTEP: a point no ladder step visited. cosmolike draws a new cache key
# whenever an input changes, so the way back to the final point refills
# every table, and the refilled values must equal the recorded ones.
SCRAMBLE_STEP = 4  # every sector at step 4, bias and point mass included
# Relative tolerance of the analytic M rescale check (assertion 2). The
# rescale is exact up to the rounding of a few multiplications and
# divisions (about 1e-16 each in double precision); a wrong bin factor
# or any response of the tables to M changes entries at the level of
# the step itself (0.5% for an M step).
RESCALE_RTOL = 1.0e-12


def _sector_of(name):
    """Name the sector a sampled parameter belongs to.

    The patterns of SECTORS are tried in order and the first match
    wins. The last pattern, ".", matches any non-empty name, so a
    parameter no other pattern claims lands in "other", which the
    ladder treats as a failure.

    Arguments:
      name = a sampled-parameter name of the frozen point, e.g.
             "DES_DZ_S1" or "omegam".

    Returns:
      the sector name (str), a first field of SECTORS.
    """
    # each SECTORS entry is a (name, pattern) pair, unpacked into the
    # two loop variables; pat.search returns a match object (true) when
    # the pattern occurs in name, and None (false) otherwise
    for sector, pat in SECTORS:
        if pat.search(name):
            return sector
    return "other"


def _deltas_for(sector, names):
    """Collect the per-step offsets of one sector's sampled parameters.

    DELTAS[sector] maps keys to offsets; a key is either an exact
    parameter name (a str) or a compiled pattern matching a family of
    names. Each name takes the offset of the first key that matches
    it; a name no key matches gets no entry and stays at its fiducial
    value for the whole ladder.

    Arguments:
      sector = a sector name (a first field of SECTORS); a sector
               without a DELTAS entry ("other") yields {}.
      names  = the sampled parameters _sector_of assigned to sector.

    Returns:
      {parameter name: offset per ladder step}, in the parameter's own
      units.
    """
    # .get returns its second argument, an empty table, when the
    # sector has no DELTAS entry
    table = DELTAS.get(sector, {})
    out = {}
    for n in names:
        for key, d in table.items():
            # a str key must equal the name; a pattern key must occur
            # in it. The one-line if-else picks the test by the key's
            # type, and break keeps the first key that matches
            if (key == n) if isinstance(key, str) else key.search(n):
                out[n] = d
                break
    return out


class TestCacheConsistencyTATT(unittest.TestCase):
    """Sector-ladder cache-invalidation check on the frozen TATT fiducial.

    unittest.TestCase is the base class of python's unittest framework:
    each method whose name starts with test_ is one test, and pytest
    collects such classes too. _run_ladder stores the per-sector
    offsets of the model it walks in self.sector_deltas, which
    _point_at reads.
    """

    # @classmethod is a decorator (a line starting with @ that wraps
    # the function defined below it): it hands the method the class
    # itself (cls), not an instance; unittest calls setUpClass once
    # before the first test of the class
    @classmethod
    def setUpClass(cls):
        """Check the environment and the frozen state once.

        Moves to ROOTDIR (the Cocoa/ folder, where the frozen
        configurations' relative paths start) and verifies every frozen
        file against the SHA-256 manifest before any model is built.
        The ladder compares the model with itself, so no reference
        chi2 is loaded.

        Arguments:
          cls = this test class (the classmethod decorator passes the
                class itself, not an instance).

        Returns:
          nothing.

        Raises:
          RuntimeError when ROOTDIR is not set (start_cocoa.sh was not
          sourced); AssertionError when a frozen file differs from the
          manifest.
        """
        u.require_cocoa_environment()
        u.verify_frozen()

    def _point_at(self, fid, steps):
        """Build the ladder point with each sector moved by its step count.

        Every moved parameter of a sector takes fiducial + step x offset
        (the offsets of self.sector_deltas); every other parameter keeps
        its fiducial value.

        Arguments:
          fid   = {parameter: value}, the frozen fiducial point.
          steps = {sector: step count}, 0 = the fiducial value.

        Returns:
          a new {parameter: value} dictionary.
        """
        # dict(fid) is a copy, so the fiducial table itself never changes
        point = dict(fid)
        for sector, step in steps.items():
            for n, d in self.sector_deltas[sector].items():
                point[n] = fid[n] + step * d
        return point

    def _mpairs(self, like, np_):
        """Assign each 3x2pt entry the source bins of its calibration.

        Rows follow cosmolike's 3x2pt layout: cosmic shear (xi+ of
        every source pair i <= j, then xi- of every pair: the xi_pm
        split doubles the shear block), galaxy-galaxy lensing (lens
        outer, source inner), galaxy clustering, each row ntheta long.
        Cosmic shear scales by both bins' factors (1 + m), gamma_t by
        the source bin's, clustering by none.

        Arguments:
          like = the cobaya likelihood instance (model.likelihood[name]):
                 ntheta, source_ntomo and lens_ntomo come from the
                 dataset file it read.
          np_  = the numpy module (this file imports numpy inside the
                 functions that use it).

        Returns:
          int array [ndata, 2], ndata = the length of the 3x2pt vector
          (900): the (i, j) source bins of each entry, -1 = no factor.

        Raises:
          AssertionError when the rebuilt cosmic-shear or
          galaxy-galaxy-lensing rows do not end where the interface's
          block sizes say.
        """
        import cosmolike_des_cluster_interface as ci
        # the sizes of the ss, gs and gg blocks; the comprehension turns
        # each returned number into a python int
        sizes = [int(x) for x in ci.compute_data_vector_3x2pt_real_sizes()]
        nlen = int(like.ntheta)
        nsrc = int(like.source_ntomo)
        # the source-bin pairs i <= j of cosmic shear in cosmolike's
        # order (i outer, j inner): a comprehension with two for clauses
        # runs them as nested loops
        sspairs = [(i, j) for i in range(nsrc) for j in range(i, nsrc)]
        # the gs block skips the (lens, source) pairs a likelihood
        # option ggl_exclude would list. The likelihoods of this project
        # define no such option, so getattr returns its default None,
        # `or []` turns that into an empty list, and every pair is kept.
        # The set comprehension collects the excluded pairs as int
        # tuples; the list comprehension keeps, in cosmolike's order
        # (lens outer, source inner), the pairs not excluded
        excluded = {(int(a), int(b)) for a, b in
                    (getattr(like, "ggl_exclude", None) or [])}
        gglpairs = [(zl, zs) for zl in range(int(like.lens_ntomo))
                    for zs in range(nsrc) if (zl, zs) not in excluded]
        # sum(sizes) = ndata; zeros(...) - 1 fills an array with -1, the
        # "no factor" marker, which the clustering rows keep
        fac = np_.zeros((sum(sizes), 2), dtype=int) - 1
        # k runs over the entries; sspairs + sspairs concatenates the
        # two lists (xi+ rows, then xi- rows)
        k = 0
        for (i, j) in sspairs + sspairs:  # xi_plus + xi_minus
            for t in range(nlen):
                fac[k] = (i, j)
                k += 1
        self.assertEqual(k, sizes[0], "cosmic shear block layout")
        for (zl, zs) in gglpairs:
            for t in range(nlen):
                fac[k] = (-1, zs)
                k += 1
        self.assertEqual(k, sizes[0] + sizes[1], "ggl block layout")
        return fac

    def _run_ladder(self, tatt):
        """Walk the forward and the mirrored ladder of the 3x2pt model.

        For each order: build a model, check that every sampled
        parameter belongs to a named sector and that every phase has
        sampled parameters, walk the phases NSTEP steps each
        (assertions 1-2 at every step), probe a no-op update (assertion
        3), evaluate the scramble and return to the final point
        (assertion 4). Last, the two orders must end on the same vector
        (assertion 5).

        After every evaluation the test reads the 3x2pt data vector from
        the compiled interface (the python module
        cosmolike_des_cluster_interface, built from
        interface/interface.cpp), which recomputes it from the inputs
        and tables the evaluation left in the C layer: the vector of
        the point just evaluated, with zeros at masked entries.

        Arguments:
          tatt = True walks the TATT configuration (IA_model 1, the
                 TATT_POINT values of cocoa_test_utils, the TATT data
                 vector), the only value the test passes; False would
                 walk the same ladder with NLA.

        Returns:
          nothing; progress lines with the final chi2 of each ladder
          are printed. The two models are built in this process, one
          after the other, so cosmolike's global state then holds the
          second one, and self.sector_deltas holds its offsets.

        Raises:
          AssertionError at the first failed check.
        """
        import numpy as np
        import cosmolike_des_cluster_interface as ci

        # the cobaya component name, "des_cluster.combo_3x2pt"
        name = u.EXAMPLES[EXAMPLE]["likelihood"]
        # {order: final data vector}, compared after both orders ran
        results = {}
        for order in ("forward", "mirrored"):
            # each order builds its own model instance: load_frozen_info
            # returns the cobaya input dictionary of the frozen
            # configuration (with tatt=True: IA_model 1, the TATT data
            # vector, the fixed TATT values), make_model the cobaya
            # model (CAMB plus the likelihood, whose set-up initializes
            # cosmolike)
            info = u.load_frozen_info(EXAMPLE, tatt=tatt)
            model = u.make_model(info)
            # build_point returns the frozen point after checking that
            # it names exactly the model's sampled parameters, with the
            # sampled TATT values replaced; dict() makes a private copy
            fid = dict(u.build_point(model, EXAMPLE, tatt=tatt))
            # a comprehension with a condition: the sampled names that
            # no named sector claims. An empty list counts as false, so
            # assertFalse passes only when every name has a sector
            stray = [n for n in fid if _sector_of(n) == "other"]
            self.assertFalse(
                stray, f"sampled parameters outside every sector: {stray}")
            # {sector: {parameter: offset}} for every SECTORS entry (the
            # _ discards the pattern); the inner comprehension lists the
            # sampled names of sector s, and a sector with no moved
            # parameter gets an empty table, which counts as false below
            self.sector_deltas = {
                s: _deltas_for(s, [n for n in fid if _sector_of(n) == s])
                for s, _ in SECTORS}
            for s in PHASES:
                self.assertTrue(self.sector_deltas[s],
                                f"no sampled parameters in sector {s}")

            phases = PHASES if order == "forward" else tuple(reversed(PHASES))
            # every sector at step 0: the fiducial point
            steps = {s: 0 for s in self.sector_deltas}
            u.evaluate_chi2(model, self._point_at(fid, steps))
            prev = np.array(ci.compute_data_vector_masked())
            mfac = self._mpairs(model.likelihood[name], np)

            for sector in phases:
                for r in range(1, NSTEP + 1):
                    # the M values of the point before this step;
                    # .items() yields the (name, offset) pairs
                    m_prev = {n: fid[n] + steps["m"] * d
                              for n, d in self.sector_deltas["m"].items()}
                    steps[sector] = r
                    point = self._point_at(fid, steps)
                    u.evaluate_chi2(model, point)
                    dv = np.array(ci.compute_data_vector_masked())
                    self.assertFalse(
                        np.array_equal(dv, prev),
                        f"{order}: {sector} step {r} left the data vector "
                        "unchanged (dead sector flag or stale cache)")
                    if sector == "m":
                        # the M values after the step (a dictionary
                        # comprehension: one name: value entry per M)
                        m_now = {n: point[n]
                                 for n in self.sector_deltas["m"]}
                        # mp[i] = the M parameter of source bin i: sorted
                        # on a dictionary sorts its keys, and alphabetical
                        # order is bin order for DES_M1 .. DES_M4
                        mp = sorted(m_prev)  # M1..M4 in bin order
                        # ratio = the expected per-entry factor dv/prev:
                        # each source leg of the entry (i, j; -1 = no
                        # leg) contributes (1 + m_now)/(1 + m_prev)
                        ratio = np.ones(dv.size)
                        for k in range(dv.size):
                            i, j = mfac[k]
                            if j >= 0:
                                ratio[k] *= ((1 + m_now[mp[j]]) /
                                             (1 + m_prev[mp[j]]))
                            if i >= 0:
                                ratio[k] *= ((1 + m_now[mp[i]]) /
                                             (1 + m_prev[mp[i]]))
                        # masked entries are 0 in both vectors and are
                        # skipped (the division would give nan there)
                        nz = prev != 0
                        rel = np.abs(dv[nz]/(prev[nz]*ratio[nz]) - 1.0)
                        self.assertLess(
                            rel.max(), RESCALE_RTOL,
                            f"{order}: M step {r} is not the analytic "
                            f"(1+m_i)(1+m_j) rescale (max {rel.max():.2e})")
                    # the next step is compared with this one
                    prev = dv

            # every phase sector at step NSTEP: the ladder's final point
            final_point = self._point_at(fid, steps)
            final_chi2 = u.evaluate_chi2(model, final_point)
            final_dv = np.array(ci.compute_data_vector_masked())

            # no-op probe: identical point again, bitwise; dict() hands
            # over a copy, a different dictionary with the same values
            u.evaluate_chi2(model, dict(final_point))
            self.assertTrue(
                np.array_equal(np.array(ci.compute_data_vector_masked()),
                               final_dv),
                f"{order}: a no-op re-evaluation changed the data vector")

            # scramble: every sector at once, bias and point mass included
            # (the dictionary comprehension puts every sector at
            # SCRAMBLE_STEP)
            scr = {s: SCRAMBLE_STEP for s in self.sector_deltas}
            u.evaluate_chi2(model, self._point_at(fid, scr))

            # return: the ladder's final point must reproduce bitwise
            back_chi2 = u.evaluate_chi2(model, final_point)
            back_dv = np.array(ci.compute_data_vector_masked())
            self.assertTrue(
                np.array_equal(back_dv, final_dv),
                f"{order}: returning after the scramble did not reproduce "
                "the data vector bit for bit (stale sector cache)")
            self.assertEqual(
                back_chi2, final_chi2,
                f"{order}: chi2 after the scramble return differs")
            results[order] = final_dv
            # the f-string field {'TATT' if tatt else 'NLA'} inserts the
            # name of the IA model the ladder ran
            print(f"  {order} ladder ({EXAMPLE}, "
                  f"{'TATT' if tatt else 'NLA'}): "
                  f"final chi2 = {final_chi2:.6f}", flush=True)

        self.assertTrue(
            np.array_equal(results["forward"], results["mirrored"]),
            "the mirrored-order ladder landed on a different data vector: "
            "the answer depends on the invalidation history")

    # the ladder builds its models in process: see the module docstring
    # for why it must not share pytest's process. The decorator
    # u.own_process(__file__) wraps the method: in pytest's process the
    # wrapper starts a fresh python on this file that runs only this
    # test, and passes when that process exits with code 0; inside that
    # child (marked by the environment variable that
    # cocoa_test_utils.OWN_PROCESS_FLAG names) the wrapper runs the body
    # below
    @u.own_process(__file__)
    def test_cache_consistency_3x2pt_tatt(self):
        """Run the TATT sector ladder on 3x2pt (example4).

        Assertions 1-5 of the module docstring, in a python process of
        its own (the decorator above).

        Arguments:
          none.

        Returns:
          nothing.

        Raises:
          AssertionError at the first failed check inside the child
          process; in pytest's process the wrapper raises an
          AssertionError naming the test when the child exits with a
          nonzero code (the child's report, printed above it, shows the
          failed check).
        """
        self._run_ladder(tatt=True)


# __name__ is "__main__" only when this file runs directly as a script
# (the process own_process starts, or a manual run); pytest imports the
# module instead, so this block stays idle under pytest. unittest.main
# runs the test named on the command line (own_process names one) or,
# without a name, every test of the file; verbosity=2 prints one line
# per test
if __name__ == "__main__":
    unittest.main(verbosity=2)
