"""Unit test: sector-wise cache invalidation (the parameter ladder).

THE CACHE UNDER TEST

cosmolike keeps every expensive intermediate result in a table that
survives from one evaluation to the next (C variables declared static
or global, which keep their value between calls): distances, growth and
power-spectrum tables, n(z) splines and lens efficiencies, the C_ell
tables of each probe, and the cluster tables (the richness-weighted
mass integrals, the one-halo lensing table, the cluster kernels, the
Limber spectra of every cluster pair). Each group of inputs that feeds
a table carries a cache key, a 64-bit number that cosmolike draws anew
whenever one of the group's input values changes: cosmology.random for
the cosmology, nuisance.random_ia for intrinsic alignment, the photo-z
and galaxy-bias keys, and keys of the cluster code for the
mass-observable relation (cluster.random_mor) and for the cluster
redshift kernels read from the n(z) file (cluster.random_zdist). A
table stores the keys of every group it was built from and refills
when one of them differs; the cluster tables combine the cluster keys
with the galaxy-side ones. The sectors of this test are groups of
sampled parameters.

A table that leaves one of its groups out of that list, or an update
path that forgets to draw a new key, keeps the values of an earlier
point: a stale table. Such a partial-invalidation bug produces silently
wrong data vectors only in MIXED update sequences, where one sector
moves while the others stay; the per-point tests (test_example*.py)
never run those. Three sectors have no table of their own: the shear
calibrations m and the selection bias of these examples (the Y6 model,
eq. 23 of arXiv 2503.13631) rescale the data vector after the tables
are read, and the point masses add their term to gamma_t there
(cluster.random_selection keys only the Y1 selection model, which sits
inside the bias mass integral).

THE DATA VECTOR

The joint data vector has seven blocks, in this order:

    ss  cosmic shear xi+ and xi-         gs  galaxy-galaxy lensing
    gg  galaxy clustering                cg  cluster x galaxy
    N   cluster counts                   cc  cluster clustering
    cs  cluster lensing (Sigma = Y gamma_t)

Each combination's mask (one 0/1 flag per entry, 0 = entry not used)
keeps its own blocks and scale cuts: 4x2pt + N has no ss and no gs.
cs holds the Y transform of gamma_t (eq. 15 of arXiv 2503.13631, a
fixed linear combination of the gamma_t angular bins); its last
angular bin is zero by construction and always masked.

THE LADDER

The test walks a deterministic ladder IN ONE PROCESS, evaluating the
model after every step (each sector's later steps keep the earlier
sectors at their last values, so the ladder ends at one well-defined
point). The sectors, and the blocks each one must move:

    3 x cosmology steps (omegam, H0, As_1e9)    every block
    3 x IA steps (A1 amplitude and z power)     ss gs cs
    3 x source-photo-z steps (every DZ_S)       ss gs cs
    3 x lens-photo-z steps (every DZ_L)         gs gg cg
    3 x shear-calibration steps (every M)       ss gs cs
    3 x galaxy-bias steps (every B1)            gs gg cg
    3 x point-mass steps (every PM)             gs (6x2pt + N only)
    3 x mass-observable steps (the four MOR)    cg N cc cs
    3 x selection-bias steps (b_s1, b_s2, r_0)  cg cc cs

(IA = the intrinsic alignment of the source galaxies, here the NLA
model; MOR = the mass-observable relation, the lognormal distribution
of the richness lambda at given halo mass and redshift, with the
parameters ln lambda_0, A_lambda, sigma_int and B_lambda.)

It records the final data vector, then evaluates one SCRAMBLE point
(every sector moved at once; the chi2 is discarded) and returns to
the ladder's final point, then moves each sector ALONE and returns
again: every return must reproduce the recorded vector bit for bit
(every double identical). A second model instance walks the MIRRORED
ladder (selection -> ... -> cosmology) to the same final point, and a
FRESH process evaluates that point as its first and only evaluation:
the answer must depend on the point, never on the invalidation
history.

Assertions, for each combination (4x2pt + N and 6x2pt + N):
  1. every ladder step changes EVERY unmasked entry of the blocks its
     sector enters, and leaves every other block bitwise unchanged (a
     dead sector flag, a table that misses one of its keys, and a
     table rebuilt from the wrong sector all show up here, block by
     block);
  2. each M-only step rescales the masked vector by the analytic
     (1+m_i)(1+m_j) block factors to 1e-12 relative: cosmic shear by
     both bins' factors, gamma_t and cluster lensing by the source
     factor, every other block by nothing;
  3. each selection-only step rescales it by the ratio of the
     selection factors B(theta) of eq (23) of arXiv 2503.13631 to
     1e-12 relative: cluster lensing and cluster x galaxy by one
     factor, cluster clustering by its square, the counts by nothing;
  4. a no-op update (re-sending the current point) leaves the vector
     bitwise unchanged;
  5. after the scramble, and after each single-sector excursion,
     returning to the ladder's final point reproduces the recorded
     vector and chi2 bit for bit;
  6. the mirrored-order instance lands on the same final vector bit
     for bit;
  7. the fresh process, which never saw another point, computes the
     same vector and chi2 bit for bit as both ladders (a table filled
     at the first point and never rebuilt passes 5 and 6, because
     every path of this process shares it; only a process that starts
     at the final point exposes it).

Every evaluation forces a full recomputation: cobaya (the sampler
framework that runs CAMB and the likelihood) can hand back a stored
result when it sees the same parameters again, and u.evaluate_chi2
switches that memoization off (cached=False), so each assertion tests
cosmolike's own invalidation.

The models are built in pytest's own process, not in worker
subprocesses: the ladder reads the data vector from the compiled
interface (the python module cosmolike_des_cluster_interface, built
from interface/interface.cpp) after every evaluation, and one model
must keep its tables across all of its evaluations. Both combinations
read the same 2812-entry joint vector, so their models can follow one
another in one process (cosmolike aborts a process that initializes
two data sets of different dimensions). The fresh evaluation of
assertion 7 is the only subprocess.

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/des_cluster/tests/data_vector/test_cache_consistency.py
"""

import os

# OpenMP (the threading library of the compiled C code) reads the
# environment variable OMP_NUM_THREADS when the compiled libraries
# load, so this must run before ANY cobaya/cosmolike import in the
# process. os.environ is this process's environment, which the
# processes it starts inherit (the fresh-evaluation worker of
# assertion 7 among them). 4 = cocoa_testing.REQUIRED_OMP_THREADS, the
# thread count of every test process.
os.environ["OMP_NUM_THREADS"] = "4"

import re
import sys
import unittest

# sys.path is the list of folders python searches on import, position 0
# first. dirname applied twice to this file's absolute path gives the
# tests/ folder, which holds this project's harness cocoa_test_utils.py
# (imported as u). Searching tests/ first makes the import find this
# project's harness under pytest and when this file runs as a script
# (the fresh-evaluation worker); every Cocoa project names its harness
# cocoa_test_utils.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cocoa_test_utils as u

# The blocks of the joint data vector, in the order of the compiled
# interface's compute_data_vector_cluster_sizes / _starts.
BLOCKS = ("ss", "gs", "gg", "cg", "N", "cc", "cs")

# Sector membership by sampled-parameter name. Every sampled parameter
# must fall in a named sector (checked at run time: a parameter landing
# in "other" fails the test, so a nuisance parameter added to a
# likelihood cannot stay outside the ladder unnoticed). re.compile
# turns a regular expression (a text pattern) into a pattern object
# whose .search finds the pattern anywhere in a name: ^ and $ anchor
# the start and the end of the name, | separates alternatives, (...)
# groups them, [0-9]+ is one or more digits, and . is any single
# character, so the last entry catches every name the others miss.
# _sector_of takes the first entry that matches, so the order matters.
SECTORS = (
    ("cosmo", re.compile(r"^(As_1e9|H0|ns|omegab|omegam|mnu|w|w0pwa)$")),
    ("ia", re.compile(r"_A1_|_A2_|_BTA_")),
    ("dz_source", re.compile(r"_DZ_S")),
    ("dz_lens", re.compile(r"_DZ_L")),
    ("m", re.compile(r"_M[0-9]+$")),
    ("bias", re.compile(r"_B1_|_B2_|_BMAG_")),
    ("pm", re.compile(r"_PM[0-9]+$")),
    ("mor", re.compile(r"_CL_(LNLAMBDA0|A_LAMBDA|SIGMA_INT|B_LAMBDA)$")),
    ("selection", re.compile(r"_CL_(BS1|BS2|R0|BSZ)$")),
    ("other", re.compile(r".")),
)

# The blocks each sector enters (assertion 1). The cluster blocks read
# the galaxy-side sectors through their second leg: cluster lensing
# carries the sources (photo-z, shear calibration, and intrinsic
# alignment through cluster_include_ia), cluster x galaxy carries the
# lenses (photo-z, bias). The counts read the cosmology and the
# mass-observable relation only; the selection bias multiplies the
# two-point cluster blocks on the data vector and leaves the counts
# alone.
RESPONSE = {
    "cosmo": ("ss", "gs", "gg", "cg", "N", "cc", "cs"),
    "ia": ("ss", "gs", "cs"),
    "dz_source": ("ss", "gs", "cs"),
    "dz_lens": ("gs", "gg", "cg"),
    "m": ("ss", "gs", "cs"),
    "bias": ("gs", "gg", "cg"),
    "pm": ("gs",),
    "mor": ("cg", "N", "cc", "cs"),
    "selection": ("cg", "cc", "cs"),
}

# Ladder phases (sector order of the forward walk). The cluster sectors
# come last in the forward walk and first in the mirrored one, so each
# cluster table is refilled both after and before every galaxy-side
# sector it depends on.
PHASES = ("cosmo", "ia", "dz_source", "dz_lens", "m", "bias", "pm",
          "mor", "selection")
# Sectors a combination does not sample, so its ladder has no such
# phase: 4x2pt + N has no gs block and fixes the point masses at zero
# (likelihood/combo_4x2pt_N.yaml). Every other sector must be sampled;
# a sector listed here must not be (either way a renamed or mislaid
# parameter fails the test instead of silently shortening the ladder).
UNSAMPLED = {"example1": ("pm",)}
# Per-step offsets: during the ladder a moved parameter takes the value
# fiducial + step * offset (deterministic). A key is either a parameter
# name (a str) or a compiled pattern that every parameter of a family
# matches (all four DES_DZ_S, for instance); _deltas_for gives each
# sampled name the offset of the first key that matches it. Sampled
# parameters without an offset stay at the fiducial (ns, omegab, mnu, w
# and w0pwa of the cosmology). Each offset must move every unmasked
# entry of the blocks its sector enters (assertion 1 compares bitwise,
# so any change counts), and the excursion points (SCRAMBLE_STEP
# offsets from the fiducial) stay inside every prior of the frozen
# configurations: a point outside a prior evaluates to -inf, which
# u.evaluate_chi2 turns into a test failure.
DELTAS = {
    "cosmo": {"omegam": 0.002, "H0": 0.2, "As_1e9": 0.02},
    # the amplitude and the redshift power move together: at the
    # fiducial amplitude (zero) the power alone changes nothing
    "ia": {re.compile(r"_A1_1$"): 0.05, re.compile(r"_A1_2$"): 0.05},
    "dz_source": {re.compile(r"_DZ_S"): 0.001},
    "dz_lens": {re.compile(r"_DZ_L"): 0.001},
    "m": {re.compile(r"_M[0-9]+$"): 0.005},
    "bias": {re.compile(r"_B1_"): 0.05},
    "pm": {re.compile(r"_PM[0-9]+$"): 0.05},
    "mor": {re.compile(r"_CL_LNLAMBDA0$"): 0.01,
            re.compile(r"_CL_A_LAMBDA$"): 0.005,
            re.compile(r"_CL_SIGMA_INT$"): 0.005,
            re.compile(r"_CL_B_LAMBDA$"): 0.01},
    "selection": {re.compile(r"_CL_BS1$"): 0.01,
                  re.compile(r"_CL_BS2$"): 0.01,
                  re.compile(r"_CL_R0$"): 0.5},
}
# Steps per sector. Each step must change the vector again, so a table
# that refills only on the first change after a build fails from the
# second step on.
NSTEP = 3
# The excursions move a sector (or, in the scramble, every sector) one
# step beyond the last ladder step NSTEP: a point no ladder step
# visited. cosmolike draws a new cache key whenever an input changes,
# so the way back to the final point refills every table the excursion
# touched, and the refilled values must equal the recorded ones.
SCRAMBLE_STEP = 4  # the excursions: a sector (or every sector) at step 4
# Relative tolerance of the analytic rescale checks (assertions 2 and
# 3). The rescale is exact up to the rounding of a few multiplications
# and divisions (about 1e-16 each in double precision); a wrong factor
# (a wrong bin, a wrong power of B) or any response of the tables to
# the moved parameters changes entries at the level of the step itself
# (0.5% for an M step).
RESCALE_RTOL = 1.0e-12

# The command-line flag that turns this file into the worker of the
# fresh evaluation (assertion 7); see _fresh_worker.
FRESH_FLAG = "--fresh-one"


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


def _fresh_worker(example, point_path, result_path):
    """Worker side of the fresh evaluation: one model, one point.

    Runs in its own python process (this file started with
    FRESH_FLAG), so no cosmolike table exists before the point is
    evaluated: every cache is filled AT that point, by its first and
    only evaluation. The parent verified the frozen state before it
    started this process, so the worker does not repeat that check.

    Arguments:
      example     = a key of cocoa_test_utils.EXAMPLES.
      point_path  = json file holding the {parameter: value} point
                    (json, the JavaScript Object Notation text format,
                    writes each float with the digits that round-trip
                    it exactly, so the worker evaluates the very same
                    doubles as the ladder).
      result_path = .npy file the worker writes: the chi2 followed by
                    the masked data vector.

    Returns:
      nothing; the result lands in result_path.

    Raises:
      whatever the model build or the evaluation raises; the process
      then exits with a nonzero code and writes no result, which
      _fresh_evaluation turns into a RuntimeError.
    """
    import json

    import numpy as np

    # refuse to run without start_cocoa.sh (ROOTDIR unset), then move
    # to ROOTDIR, where the configuration's relative paths start
    u.require_cocoa_environment()
    # the compiled interface: the python module built from
    # interface/interface.cpp that exposes cosmolike's C/C++ functions
    import cosmolike_des_cluster_interface as ci

    # the with block closes the file on every exit; json.load returns
    # the {parameter: value} dictionary the parent wrote
    with open(point_path) as f:
        point = json.load(f)
    # load_frozen_info returns the cobaya input dictionary of the
    # frozen NLA configuration; make_model builds the cobaya model from
    # it (CAMB plus the likelihood, whose set-up initializes cosmolike)
    model = u.make_model(u.load_frozen_info(example, tatt=False))
    chi2 = u.evaluate_chi2(model, point)
    # recomputed from the inputs and tables the evaluation left in the
    # C layer: the theory vector of the point just evaluated, with
    # zeros at masked entries
    dv = np.array(ci.compute_data_vector_cluster_masked())
    # one array [chi2, dv[0], dv[1], ...]; np.save writes it in numpy's
    # binary .npy format, every double stored exactly
    np.save(result_path, np.concatenate(([chi2], dv)))


def _fresh_evaluation(example, point):
    """Evaluate one point in a fresh process; hand back (chi2, vector).

    The parent side of _fresh_worker: the point travels through a
    temporary json file, the result through a temporary .npy file,
    both inside a temporary folder that is removed afterwards. The
    child is a subprocess (a separate operating-system process started
    from this one) running this same file with FRESH_FLAG; it inherits
    this process's environment, ROOTDIR and OMP_NUM_THREADS included.

    Arguments:
      example = a key of cocoa_test_utils.EXAMPLES.
      point   = {parameter: value} covering the sampled parameters.

    Returns:
      (chi2, masked data vector) of the fresh process: a float and a
      1D numpy array with one value per entry of the joint vector
      (zeros at masked entries).

    Raises:
      RuntimeError when the worker exits with a nonzero code, which
      means it wrote no result.
    """
    import json
    import subprocess
    import tempfile

    import numpy as np

    # the with block removes the temporary folder on every exit, an
    # exception included
    with tempfile.TemporaryDirectory(prefix="cocoa_cache_fresh_") as tmp:
        point_path = os.path.join(tmp, "point.json")
        result_path = os.path.join(tmp, "result.npy")
        with open(point_path, "w") as f:
            json.dump(point, f)
        # sys.executable = this same python; __file__ = this very
        # file, which sets OMP_NUM_THREADS at its first line in the
        # worker too. subprocess.run starts the child with this
        # command line and blocks until it exits
        completed = subprocess.run(
            [sys.executable, os.path.abspath(__file__), FRESH_FLAG,
             example, point_path, result_path])
        if completed.returncode != 0:
            raise RuntimeError(
                f"fresh-evaluation worker for {example} exited with "
                f"code {completed.returncode}")
        result = np.load(result_path)
    # result[0] is the chi2; result[1:] (every element after the
    # first) is the data vector
    return float(result[0]), result[1:]


class TestCacheConsistency(unittest.TestCase):
    """Sector-ladder cache-invalidation check on the frozen fiducial.

    unittest.TestCase is the base class of python's unittest framework:
    each method whose name starts with test_ is one test (here one per
    cluster combination), and pytest collects such classes too.
    _run_ladder stores the per-sector offsets of the model it walks in
    self.sector_deltas, which _point_at reads.
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

    def _layout(self, like, np_):
        """Rebuild the block slices and the per-entry rescale factors.

        The row orders are those of the block fillers in
        cosmolike_core/cosmolike/generic_interface_cluster.cpp: ss [xi+
        pairs, xi- pairs], gs [lens-source pair], gg [lens bin], cg
        [cluster-lens pair][richness], N [cluster bin][richness], cc
        [cluster bin][richness pair], cs [cluster-source pair]
        [richness], each row ntheta long except the counts. Every block
        rebuilt here must end exactly where the interface says it ends;
        a mismatch means the layout assumed below no longer matches the
        C code.

        Arguments:
          like = the cobaya likelihood instance (model.likelihood[name]):
                 ntheta, source_ntomo, lens_ntomo and richness_edges
                 come from the dataset file it read.
          np_  = the numpy module (this file imports numpy inside the
                 functions that use it).

        Returns:
          (blocks, mfac, selbin, selpow, seltheta), with ndata = the
          length of the joint vector (2812):
            blocks   = {block name: slice of the joint vector}.
            mfac     = int array [ndata, 2]: the (i, j) source bins
                       whose shear-calibration factors (1 + m) multiply
                       the entry (-1 = no factor): cosmic shear scales
                       by both bins, gamma_t and cluster lensing by the
                       source bin, every other block by nothing.
            selbin   = int array [ndata]: the cluster redshift bin whose
                       selection factor B(theta) multiplies the entry
                       (-1 = none).
            selpow   = int array [ndata]: the power of that factor: 1
                       for cluster lensing and cluster x galaxy (one
                       cluster leg), 2 for cluster clustering (two), 0
                       elsewhere.
            seltheta = int array [ndata]: the theta bin of the entry
                       inside its row, where selbin is set (B depends
                       on the angular bin).

        Raises:
          AssertionError when a rebuilt block does not end where the
          interface's block sizes say, or when the cg or cs block holds
          another number of pairs than the interface's pair table.
        """
        import cosmolike_des_cluster_interface as ci
        # one size and one start index per block, in BLOCKS order; the
        # comprehension turns each returned number into a python int
        sizes = [int(x) for x in ci.compute_data_vector_cluster_sizes()]
        starts = [int(x) for x in ci.compute_data_vector_cluster_starts()]
        # zip walks BLOCKS, starts and sizes in step, one (name, start,
        # size) triple per block; slice(s, s + n) is the index range
        # s .. s+n-1, so dv[blocks["gg"]] is the gg block of a vector dv
        blocks = {b: slice(s, s + n)
                  for b, s, n in zip(BLOCKS, starts, sizes)}
        # angular bins per row, source bins, and richness bins (the
        # dataset lists the richness bin edges, one more than the bins)
        ntheta = int(like.ntheta)
        nsrc = int(like.source_ntomo)
        nrich = len(like.richness_edges) - 1
        # sum(sizes) = ndata; zeros(...) - 1 fills an array with -1, the
        # "no factor" marker
        mfac = np_.zeros((sum(sizes), 2), dtype=int) - 1
        selbin = np_.zeros(sum(sizes), dtype=int) - 1
        selpow = np_.zeros(sum(sizes), dtype=int)
        seltheta = np_.zeros(sum(sizes), dtype=int)

        # the source-bin pairs i <= j of cosmic shear in cosmolike's
        # order (i outer, j inner): a comprehension with two for clauses
        # runs them as nested loops. The ss block holds xi+ of every
        # pair, then xi- of every pair (sspairs + sspairs concatenates
        # the two lists); assigning the pair (i, j) to mfac[k:k +
        # ntheta] writes it into each of the ntheta rows (numpy
        # broadcasting: the short right side is repeated along the rows)
        sspairs = [(i, j) for i in range(nsrc) for j in range(i, nsrc)]
        k = blocks["ss"].start
        for (i, j) in sspairs + sspairs:  # xi_plus + xi_minus
            mfac[k:k + ntheta] = (i, j)
            k += ntheta
        self.assertEqual(k, blocks["ss"].stop, "ss block layout")

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
        # a gamma_t entry carries the factor of its source bin only
        k = blocks["gs"].start
        for (zl, zs) in gglpairs:
            mfac[k:k + ntheta] = (-1, zs)
            k += ntheta
        self.assertEqual(k, blocks["gs"].stop, "gs block layout")

        # the pair tables of the interface, in the row order of the
        # blocks: row n holds (cluster z bin, lens bin) of cg pair n and
        # (cluster z bin, source bin) of cs pair n, stored as floats
        # (.astype(int) converts them to integer bin indices)
        cg_bins = np_.array(ci.get_cg_redshift_bins()).astype(int)
        cs_bins = np_.array(ci.get_cs_redshift_bins()).astype(int)

        # one cluster pair owns nrich rows: nper entries in a row of
        # rows. // is the integer division (block length / nper = the
        # number of pairs); arange(nper) % ntheta counts 0..ntheta-1
        # again and again, the theta bin along every row
        nper = ntheta * nrich
        ncg = sizes[BLOCKS.index("cg")] // nper
        self.assertEqual(ncg, cg_bins.shape[0],
                         "cg block pairs != rows of get_cg_redshift_bins")
        k = blocks["cg"].start
        for n in range(ncg):
            selbin[k:k + nper] = cg_bins[n, 0]
            selpow[k:k + nper] = 1
            seltheta[k:k + nper] = np_.arange(nper) % ntheta
            k += nper
        self.assertEqual(k, blocks["cg"].stop, "cg block layout")

        # cluster lensing: the same row layout as cluster x galaxy, plus
        # the shear calibration of the pair's source bin (cs_bins[n, 1])
        ncs = sizes[BLOCKS.index("cs")] // nper
        self.assertEqual(ncs, cs_bins.shape[0],
                         "cs block pairs != rows of get_cs_redshift_bins")
        k = blocks["cs"].start
        for n in range(ncs):
            selbin[k:k + nper] = cs_bins[n, 0]
            selpow[k:k + nper] = 1
            seltheta[k:k + nper] = np_.arange(nper) % ntheta
            mfac[k:k + nper] = (-1, cs_bins[n, 1])
            k += nper
        self.assertEqual(k, blocks["cs"].stop, "cs block layout")

        # cluster clustering: one cluster bin owns the rows of every
        # richness pair nl1 <= nl2 (auto redshift bins only), that is
        # nrich (nrich + 1) / 2 rows, 10 for the four richness bins
        nper = ntheta * (nrich * (nrich + 1) // 2)
        k = blocks["cc"].start
        for ni in range(sizes[BLOCKS.index("cc")] // nper):
            selbin[k:k + nper] = ni
            selpow[k:k + nper] = 2
            seltheta[k:k + nper] = np_.arange(nper) % ntheta
            k += nper
        self.assertEqual(k, blocks["cc"].stop, "cc block layout")
        return blocks, mfac, selbin, selpow, seltheta

    def _assert_response(self, np_, label, sector, dv, prev, blocks, mask):
        """Check assertion 1: the step moved its sector's blocks and no other.

        For each block with at least one unmasked entry: when the sector
        enters the block (RESPONSE), every unmasked entry must differ
        from the previous step; otherwise no entry of the block may
        differ. The comparison is bitwise, so any change counts, and a
        block the sector does not enter must reproduce its previous
        doubles exactly.

        Arguments:
          np_    = the numpy module (not used by the body).
          label  = text naming the order, sector and step for the
                   failure message, e.g. "forward: mor step 2".
          sector = the sector the step moved (a key of RESPONSE).
          dv     = the masked data vector after the step, 1D [ndata].
          prev   = the masked data vector before the step, 1D [ndata].
          blocks = {block name: slice of the joint vector} (_layout).
          mask   = the mask cosmolike applies, 1D int [ndata], 1 = entry
                   used.

        Returns:
          nothing.

        Raises:
          AssertionError naming the block and the number of entries
          that broke the rule.
        """
        # dv != prev compares element by element: a boolean array, True
        # where the step changed the entry
        changed = dv != prev
        for b in BLOCKS:
            # True at the entries of block b that the mask keeps
            active = mask[blocks[b]] == 1
            # a block the combination's mask removes entirely (ss and
            # gs in 4x2pt + N) is zero at every step: nothing to test
            if not active.any():
                continue
            moved = changed[blocks[b]]
            if b in RESPONSE[sector]:
                # moved[active] keeps the flags of the unmasked entries
                # (a boolean array used as an index selects its True
                # positions); ~ negates each flag, so the message counts
                # the unmasked entries the step left unchanged
                self.assertTrue(
                    moved[active].all(),
                    f"{label} left {int((~moved[active]).sum())} of "
                    f"{int(active.sum())} unmasked {b} entries unchanged "
                    "(dead sector flag or stale cache)")
            else:
                self.assertFalse(
                    moved.any(),
                    f"{label} moved {int(moved.sum())} {b} entries: the "
                    f"{sector} sector does not enter that block")

    def _run_ladder(self, example):
        """Walk the forward and the mirrored ladder of one combination.

        For each order: build a model, check that every sampled
        parameter belongs to a named sector and that the sampled
        sectors are the expected ones, walk the phases NSTEP steps each
        (assertions 1-3 at every step), probe a no-op update (assertion
        4), run the scramble and the single-sector excursions with a
        return to the final point after each (assertion 5), and compare
        the final point with the fresh process (assertion 7; the fresh
        evaluation runs once and serves both orders). Last, the two
        orders must end on the same vector (assertion 6).

        The two models are built in this process, one after the other;
        cosmolike's global state then holds the second one. After every
        evaluation the test reads the data vector from the compiled
        interface, which recomputes it from the inputs and tables the
        evaluation left in the C layer: the vector of the point just
        evaluated, with zeros at masked entries.

        Arguments:
          example = "example1" (4x2pt + N) or "example2" (6x2pt + N).

        Returns:
          nothing; progress lines with the final chi2 of each ladder
          and of the fresh process are printed, and self.sector_deltas
          holds the offsets of the second model.

        Raises:
          AssertionError at the first failed check; RuntimeError when
          the fresh-evaluation worker fails.
        """
        import numpy as np
        import cosmolike_des_cluster_interface as ci

        # the cobaya component name, e.g. "des_cluster.combo_4x2pt_N"
        name = u.EXAMPLES[example]["likelihood"]
        # {order: final data vector}, compared after both orders ran
        results = {}
        # None until the fresh process has run (once per combination)
        fresh_chi2, fresh_dv = None, None
        for order in ("forward", "mirrored"):
            # each order builds its own model instance: load_frozen_info
            # returns the cobaya input dictionary of the frozen NLA
            # configuration, make_model the cobaya model (CAMB plus the
            # likelihood, whose set-up initializes cosmolike)
            info = u.load_frozen_info(example, tatt=False)
            model = u.make_model(info)
            # build_point returns the frozen point after checking that
            # it names exactly the model's sampled parameters; dict()
            # makes a private copy
            fid = dict(u.build_point(model, example, tatt=False))
            # a comprehension with a condition: the sampled names that
            # no named sector claims. An empty list counts as false, so
            # assertFalse passes only when every name has a sector
            stray = [n for n in fid if _sector_of(n) == "other"]
            self.assertFalse(
                stray, f"sampled parameters outside every sector: {stray}")
            # {sector: {parameter: offset}} for every SECTORS entry (the
            # _ discards the pattern); the inner comprehension lists the
            # sampled names of sector s, and a sector with no moved
            # parameter gets an empty table
            self.sector_deltas = {
                s: _deltas_for(s, [n for n in fid if _sector_of(n) == s])
                for s, _ in SECTORS}
            # .get returns () when the example has no UNSAMPLED entry; an
            # empty table counts as false in the two assertions below
            unsampled = UNSAMPLED.get(example, ())
            for s in PHASES:
                if s in unsampled:
                    self.assertFalse(
                        self.sector_deltas[s],
                        f"sector {s} is sampled in {example}, which "
                        "UNSAMPLED says it does not sample")
                else:
                    self.assertTrue(self.sector_deltas[s],
                                    f"no sampled parameters in sector {s}")
            # the phases this combination walks, in PHASES order (the
            # generator inside tuple() skips the unsampled sectors)
            walk = tuple(s for s in PHASES if s not in unsampled)

            phases = walk if order == "forward" else tuple(reversed(walk))
            # every sector at step 0: the fiducial point
            steps = {s: 0 for s in self.sector_deltas}
            u.evaluate_chi2(model, self._point_at(fid, steps))
            prev = np.array(ci.compute_data_vector_cluster_masked())
            # the cobaya likelihood instance; its attributes hold the
            # values the likelihood read from the dataset file
            like = model.likelihood[name]
            blocks, mfac, selbin, selpow, seltheta = self._layout(like, np)
            # the mask cosmolike applies (1 = entry used): the file's
            # mask with the switched-off blocks and the last theta bin
            # of every cs row set to 0
            mask = np.array(ci.get_mask_cluster())

            for sector in phases:
                for r in range(1, NSTEP + 1):
                    # the M values of the point before this step;
                    # .items() yields the (name, offset) pairs
                    m_prev = {n: fid[n] + steps["m"] * d
                              for n, d in self.sector_deltas["m"].items()}
                    if sector == "selection":
                        # B(theta) per (cluster bin, theta bin) at the
                        # point the step leaves
                        sel_prev = np.array(
                            ci.get_cluster_selection_factor())
                    steps[sector] = r
                    point = self._point_at(fid, steps)
                    u.evaluate_chi2(model, point)
                    dv = np.array(ci.compute_data_vector_cluster_masked())
                    self._assert_response(
                        np, f"{order}: {sector} step {r}", sector, dv, prev,
                        blocks, mask)
                    # ratio = the expected per-entry factor dv/prev of
                    # an analytic rescale (assertions 2 and 3); None for
                    # the other sectors, whose response is not a rescale
                    ratio = None
                    if sector == "m":
                        # the M values after the step (a dictionary
                        # comprehension: one name: value entry per M)
                        m_now = {n: point[n]
                                 for n in self.sector_deltas["m"]}
                        # mp[i] = the M parameter of source bin i: sorted
                        # on a dictionary sorts its keys, and alphabetical
                        # order is bin order for DES_M1 .. DES_M4
                        mp = sorted(m_prev)  # M1..M4 in bin order
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
                    if sector == "selection":
                        # B(theta) after the step; it must differ from
                        # sel_prev, or the step was a dead flag
                        sel_now = np.array(ci.get_cluster_selection_factor())
                        self.assertFalse(
                            np.array_equal(sel_now, sel_prev),
                            f"{order}: selection step {r} left B(theta) "
                            "unchanged")
                        # np.nonzero returns one index array per dimension;
                        # [0] takes the indices of the entries that carry
                        # a selection factor, and ** is the power
                        ratio = np.ones(dv.size)
                        for k in np.nonzero(selbin >= 0)[0]:
                            ratio[k] = (sel_now[selbin[k], seltheta[k]] /
                                        sel_prev[selbin[k], seltheta[k]]
                                        ) ** selpow[k]
                    if ratio is not None:
                        # masked entries are 0 in both vectors and are
                        # skipped (the division would give nan there)
                        nz = prev != 0
                        rel = np.abs(dv[nz]/(prev[nz]*ratio[nz]) - 1.0)
                        self.assertLess(
                            rel.max(), RESCALE_RTOL,
                            f"{order}: {sector} step {r} is not the "
                            f"analytic rescale (max {rel.max():.2e})")
                    # the next step is compared with this one
                    prev = dv

            # every walked sector at step NSTEP: the ladder's final point
            final_point = self._point_at(fid, steps)
            final_chi2 = u.evaluate_chi2(model, final_point)
            final_dv = np.array(ci.compute_data_vector_cluster_masked())

            # no-op probe: identical point again, bitwise; dict() hands
            # over a copy, a different dictionary with the same values
            u.evaluate_chi2(model, dict(final_point))
            self.assertTrue(
                np.array_equal(
                    np.array(ci.compute_data_vector_cluster_masked()),
                    final_dv),
                f"{order}: a no-op re-evaluation changed the data vector")

            # excursions: every sector at once (the scramble), then
            # each sector alone; after each one, the ladder's final
            # point must reproduce bitwise. scr puts every sector at
            # SCRAMBLE_STEP (a dictionary comprehension); excursions is
            # a list of (description, step counts) pairs
            scr = {s: SCRAMBLE_STEP for s in self.sector_deltas}
            excursions = [("the scramble", scr)]
            for s in walk:
                # dict(steps, **{s: ...}) copies the final step counts
                # with sector s alone moved to the excursion step
                excursions.append((f"the {s} excursion",
                                   dict(steps, **{s: SCRAMBLE_STEP})))
            for what, away in excursions:
                u.evaluate_chi2(model, self._point_at(fid, away))
                back_chi2 = u.evaluate_chi2(model, final_point)
                back_dv = np.array(ci.compute_data_vector_cluster_masked())
                self.assertTrue(
                    np.array_equal(back_dv, final_dv),
                    f"{order}: returning after {what} did not reproduce "
                    "the data vector bit for bit (stale sector cache)")
                self.assertEqual(
                    back_chi2, final_chi2,
                    f"{order}: chi2 after the return from {what} differs")
            results[order] = final_dv
            print(f"  {order} ladder ({example}, NLA): "
                  f"final chi2 = {final_chi2:.6f}", flush=True)

            # the fresh process evaluates the final point once per
            # combination (both orders end at the same point); both
            # ladders are held against it
            if fresh_dv is None:
                fresh_chi2, fresh_dv = _fresh_evaluation(example, final_point)
                print(f"  fresh process ({example}, NLA): "
                      f"final chi2 = {fresh_chi2:.6f}", flush=True)
            self.assertTrue(
                np.array_equal(final_dv, fresh_dv),
                f"{order}: the ladder's final data vector differs from a "
                "fresh process evaluating the same point (a table survived "
                "from an earlier point)")
            self.assertEqual(
                final_chi2, fresh_chi2,
                f"{order}: the ladder's final chi2 differs from the fresh "
                "process")

        self.assertTrue(
            np.array_equal(results["forward"], results["mirrored"]),
            "the mirrored-order ladder landed on a different data vector: "
            "the answer depends on the invalidation history")

    def test_cache_consistency_4x2pt_N(self):
        """Run the sector ladder on 4x2pt + N (example1).

        4x2pt + N has no ss and no gs block and fixes the point masses
        at zero, so its ladder walks eight phases (no point-mass phase,
        see UNSAMPLED) and assertion 1 skips the two absent blocks.
        Assertions 1-7 of the module docstring apply.

        Arguments:
          none.

        Returns:
          nothing.

        Raises:
          AssertionError at the first failed check.
        """
        self._run_ladder("example1")

    def test_cache_consistency_6x2pt_N(self):
        """Run the sector ladder on 6x2pt + N (example2).

        6x2pt + N holds every block of the joint vector and samples a
        parameter of every sector of PHASES, so its ladder walks all
        nine phases. Assertions 1-7 of the module docstring apply.

        Arguments:
          none.

        Returns:
          nothing.

        Raises:
          AssertionError at the first failed check.
        """
        self._run_ladder("example2")


# __name__ is "__main__" only when this file runs directly as a
# script (the fresh-evaluation worker, or a manual run); pytest imports
# the module instead, so this block stays idle under pytest
if __name__ == "__main__":
    # worker mode first: FRESH_FLAG example point_path result_path
    # evaluates one point and exits (see _fresh_worker); .index
    # returns the flag's position, so the three values follow it, and
    # the * spreads the three-element slice into three arguments
    if FRESH_FLAG in sys.argv:
        at = sys.argv.index(FRESH_FLAG)
        _fresh_worker(*sys.argv[at + 1:at + 4])
        sys.exit(0)
    # otherwise run the tests of this file (verbosity=2 prints one line
    # per test)
    unittest.main(verbosity=2)
