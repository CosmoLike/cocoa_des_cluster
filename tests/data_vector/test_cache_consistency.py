"""Unit test: sector-wise cache invalidation (the parameter ladder).

cosmolike caches every expensive stage behind its own key: cosmology
(distances, growth, power-spectrum tables), intrinsic alignment,
photo-z shifts (n(z) splines, lens efficiencies), galaxy bias, and
the shear calibrations (a pure data-vector rescale). The cluster code
adds keys of its own - the mass-observable relation
(cluster.random_mor), the selection bias (cluster.random_selection),
the cluster redshift kernels (cluster.random_zdist) - and its tables
(the richness-weighted mass integrals, the one-halo lensing table,
the cluster kernels, the Limber spectra of every cluster pair) are
keyed on COMBINATIONS of them with the galaxy-side keys. A
partial-invalidation bug - one sector's update path failing to
rebuild a static another sector consumes - produces silently wrong
data vectors only in MIXED update sequences, which the per-point
suites never exercise.

The joint data vector has seven blocks, in this order:

    ss  cosmic shear xi+ and xi-         gs  galaxy-galaxy lensing
    gg  galaxy clustering                cg  cluster x galaxy
    N   cluster counts                   cc  cluster clustering
    cs  cluster lensing (Sigma = Y gamma_t)

(each combination's mask keeps its own blocks: 4x2pt + N has no ss
and no gs). The test walks a deterministic ladder IN ONE PROCESS,
evaluating the model after every step (each sector's later steps keep
the earlier sectors at their last values, so the ladder ends at one
well-defined point). The sectors, and the blocks each one must move:

    3 x cosmology steps (omegam, H0, As_1e9)    every block
    3 x IA steps (A1 amplitude and z power)     ss gs cs
    3 x source-photo-z steps (every DZ_S)       ss gs cs
    3 x lens-photo-z steps (every DZ_L)         gs gg cg
    3 x shear-calibration steps (every M)       ss gs cs
    3 x galaxy-bias steps (every B1)            gs gg cg
    3 x point-mass steps (every PM)             gs (6x2pt + N only)
    3 x mass-observable steps (the four MOR)    cg N cc cs
    3 x selection-bias steps (b_s1, b_s2, r_0)  cg cc cs

It records the final data vector, then evaluates one SCRAMBLE point
(every sector moved at once; the chi2 is discarded) and returns to
the ladder's final point, then moves each sector ALONE and returns
again: every return must reproduce the recorded vector bit for bit. A
second model instance walks the MIRRORED ladder (selection -> ... ->
cosmology) to the same final point, and a FRESH process evaluates
that point as its first and only evaluation: the answer must depend
on the point, never on the invalidation history.

Assertions, for each combination (4x2pt + N and 6x2pt + N):
  1. every ladder step changes EVERY unmasked entry of the blocks its
     sector enters, and leaves every other block bitwise unchanged (a
     dead sector flag, a table that misses one of its keys, and a
     table rebuilt from the wrong sector all show up here, block by
     block);
  2. each M-only step rescales the masked vector by the analytic
     (1+m_i)(1+m_j) block factors to 1e-12 relative - cosmic shear by
     both bins' factors, gamma_t and cluster lensing by the source
     factor, every other block by nothing;
  3. each selection-only step rescales it by the ratio of the
     selection factors B(theta) of eq (23) of arXiv 2503.13631 to
     1e-12 relative - cluster lensing and cluster x galaxy by one
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

Every evaluation forces a full recomputation (cobaya's cache is
bypassed), so each assertion tests cosmolike's own invalidation, not
cobaya's memoization.

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/des_cluster/tests/data_vector/test_cache_consistency.py
"""

import os

# OpenMP reads OMP_NUM_THREADS when the compiled libraries load, so
# this must run before ANY cobaya/cosmolike import in the process.
os.environ["OMP_NUM_THREADS"] = "4"

import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cocoa_test_utils as u

# The blocks of the joint data vector, in the order of the compiled
# interface's compute_data_vector_cluster_sizes / _starts.
BLOCKS = ("ss", "gs", "gg", "cg", "N", "cc", "cs")

# Sector membership by sampled-parameter name. Every sampled parameter
# must fall in exactly one sector (checked at run time: a parameter
# landing in "other" fails the test, so a new nuisance parameter
# cannot stay outside the ladder unnoticed).
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

# Ladder phases (sector order of the forward walk) and the per-step
# offsets: parameter value = fiducial + step * delta, deterministic.
# The cluster sectors come last in the forward walk and first in the
# mirrored one, so each cluster table is refilled both after and
# before every galaxy-side sector it depends on.
PHASES = ("cosmo", "ia", "dz_source", "dz_lens", "m", "bias", "pm",
          "mor", "selection")
# Sectors a combination does not sample, so its ladder has no such
# phase: 4x2pt + N has no gs block and fixes the point masses at zero
# (likelihood/combo_4x2pt_N.yaml). Every other sector must be sampled;
# a sector listed here must not be (either way a renamed or mislaid
# parameter fails the test instead of silently shortening the ladder).
UNSAMPLED = {"example1": ("pm",)}
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
NSTEP = 3
SCRAMBLE_STEP = 4  # the excursions: a sector (or every sector) at step 4
RESCALE_RTOL = 1.0e-12

# The command-line flag that turns this file into the worker of the
# fresh evaluation (assertion 7); see _fresh_worker.
FRESH_FLAG = "--fresh-one"


def _sector_of(name):
    for sector, pat in SECTORS:
        if pat.search(name):
            return sector
    return "other"


def _deltas_for(sector, names):
    """{parameter: per-step delta} for this sector's sampled names."""
    table = DELTAS.get(sector, {})
    out = {}
    for n in names:
        for key, d in table.items():
            if (key == n) if isinstance(key, str) else key.search(n):
                out[n] = d
                break
    return out


def _fresh_worker(example, point_path, result_path):
    """Worker side of the fresh evaluation: one model, one point.

    Runs in its own python process (this file started with
    FRESH_FLAG), so no cosmolike table exists before the point is
    evaluated: every cache is filled AT that point, by its first and
    only evaluation.

    Arguments:
      example     = a key of cocoa_test_utils.EXAMPLES.
      point_path  = json file holding the {parameter: value} point
                    (json writes each float with the digits that
                    round-trip it exactly, so the worker evaluates the
                    very same doubles as the ladder).
      result_path = .npy file the worker writes: the chi2 followed by
                    the masked data vector.

    Returns:
      nothing; the result lands in result_path.
    """
    import json

    import numpy as np

    u.require_cocoa_environment()
    import cosmolike_des_cluster_interface as ci

    with open(point_path) as f:
        point = json.load(f)
    model = u.make_model(u.load_frozen_info(example, tatt=False))
    chi2 = u.evaluate_chi2(model, point)
    dv = np.array(ci.compute_data_vector_cluster_masked())
    np.save(result_path, np.concatenate(([chi2], dv)))


def _fresh_evaluation(example, point):
    """Evaluate one point in a fresh process; hand back (chi2, vector).

    The parent side of _fresh_worker: the point travels through a
    temporary json file, the result through a temporary .npy file.

    Arguments:
      example = a key of cocoa_test_utils.EXAMPLES.
      point   = {parameter: value} covering the sampled parameters.

    Returns:
      (chi2, masked data vector) of the fresh process.

    Raises:
      RuntimeError when the worker exits without writing a result.
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
        # worker too
        completed = subprocess.run(
            [sys.executable, os.path.abspath(__file__), FRESH_FLAG,
             example, point_path, result_path])
        if completed.returncode != 0:
            raise RuntimeError(
                f"fresh-evaluation worker for {example} exited with "
                f"code {completed.returncode}")
        result = np.load(result_path)
    return float(result[0]), result[1:]


class TestCacheConsistency(unittest.TestCase):
    """Sector-ladder cache-invalidation check on the frozen fiducial."""

    @classmethod
    def setUpClass(cls):
        u.require_cocoa_environment()
        u.verify_frozen()

    def _point_at(self, fid, steps):
        """The ladder point with each sector at its given step count."""
        point = dict(fid)
        for sector, step in steps.items():
            for n, d in self.sector_deltas[sector].items():
                point[n] = fid[n] + step * d
        return point

    def _layout(self, like, np_):
        """The block slices and the per-entry data-vector-level factors.

        Returns (blocks, mfac, selbin, selpow, seltheta):
          blocks   = {block name: slice of the joint vector}.
          mfac     = (i, j) source-bin shear-calibration factors per
                     entry (-1 = no factor): cosmic shear scales by
                     both bins, gamma_t and cluster lensing by the
                     source bin, every other block by nothing.
          selbin   = the cluster redshift bin whose selection factor
                     B(theta) multiplies the entry (-1 = none).
          selpow   = the power of that factor: 1 for cluster lensing
                     and cluster x galaxy (one cluster leg), 2 for
                     cluster clustering (two), 0 elsewhere.
          seltheta = the theta bin of the entry inside its row, where
                     selbin is set (B depends on the angular bin).
        The row orders are those of the block fillers in
        generic_interface_cluster.cpp: ss [xi+ pairs, xi- pairs], gs
        [lens-source pair], gg [lens bin], cg [cluster-lens pair]
        [richness], N [cluster bin][richness], cc [cluster bin]
        [richness pair], cs [cluster-source pair][richness], each row
        ntheta long except the counts.
        """
        import cosmolike_des_cluster_interface as ci
        sizes = [int(x) for x in ci.compute_data_vector_cluster_sizes()]
        starts = [int(x) for x in ci.compute_data_vector_cluster_starts()]
        blocks = {b: slice(s, s + n)
                  for b, s, n in zip(BLOCKS, starts, sizes)}
        ntheta = int(like.ntheta)
        nsrc = int(like.source_ntomo)
        nrich = len(like.richness_edges) - 1
        mfac = np_.zeros((sum(sizes), 2), dtype=int) - 1
        selbin = np_.zeros(sum(sizes), dtype=int) - 1
        selpow = np_.zeros(sum(sizes), dtype=int)
        seltheta = np_.zeros(sum(sizes), dtype=int)

        sspairs = [(i, j) for i in range(nsrc) for j in range(i, nsrc)]
        k = blocks["ss"].start
        for (i, j) in sspairs + sspairs:  # xi_plus + xi_minus
            mfac[k:k + ntheta] = (i, j)
            k += ntheta
        self.assertEqual(k, blocks["ss"].stop, "ss block layout")

        excluded = {(int(a), int(b)) for a, b in
                    (getattr(like, "ggl_exclude", None) or [])}
        gglpairs = [(zl, zs) for zl in range(int(like.lens_ntomo))
                    for zs in range(nsrc) if (zl, zs) not in excluded]
        k = blocks["gs"].start
        for (zl, zs) in gglpairs:
            mfac[k:k + ntheta] = (-1, zs)
            k += ntheta
        self.assertEqual(k, blocks["gs"].stop, "gs block layout")

        # the pair tables of the interface, in the row order of the
        # blocks: row n holds (cluster z bin, lens bin) of cg pair n and
        # (cluster z bin, source bin) of cs pair n, stored as floats
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
        # richness pair nl1 <= nl2 (auto redshift bins only)
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
        """Assertion 1: the step moved its sector's blocks and no other."""
        changed = dv != prev
        for b in BLOCKS:
            active = mask[blocks[b]] == 1
            # a block the combination's mask removes entirely (ss and
            # gs in 4x2pt + N) is zero at every step: nothing to test
            if not active.any():
                continue
            moved = changed[blocks[b]]
            if b in RESPONSE[sector]:
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
        import numpy as np
        import cosmolike_des_cluster_interface as ci

        name = u.EXAMPLES[example]["likelihood"]
        results = {}
        fresh_chi2, fresh_dv = None, None
        for order in ("forward", "mirrored"):
            info = u.load_frozen_info(example, tatt=False)
            model = u.make_model(info)
            fid = dict(u.build_point(model, example, tatt=False))
            stray = [n for n in fid if _sector_of(n) == "other"]
            self.assertFalse(
                stray, f"sampled parameters outside every sector: {stray}")
            self.sector_deltas = {
                s: _deltas_for(s, [n for n in fid if _sector_of(n) == s])
                for s, _ in SECTORS}
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
            walk = tuple(s for s in PHASES if s not in unsampled)

            phases = walk if order == "forward" else tuple(reversed(walk))
            steps = {s: 0 for s in self.sector_deltas}
            u.evaluate_chi2(model, self._point_at(fid, steps))
            prev = np.array(ci.compute_data_vector_cluster_masked())
            like = model.likelihood[name]
            blocks, mfac, selbin, selpow, seltheta = self._layout(like, np)
            mask = np.array(ci.get_mask_cluster())

            for sector in phases:
                for r in range(1, NSTEP + 1):
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
                    ratio = None
                    if sector == "m":
                        m_now = {n: point[n]
                                 for n in self.sector_deltas["m"]}
                        mp = sorted(m_prev)  # M1..M4 in bin order
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
                        sel_now = np.array(ci.get_cluster_selection_factor())
                        self.assertFalse(
                            np.array_equal(sel_now, sel_prev),
                            f"{order}: selection step {r} left B(theta) "
                            "unchanged")
                        ratio = np.ones(dv.size)
                        for k in np.nonzero(selbin >= 0)[0]:
                            ratio[k] = (sel_now[selbin[k], seltheta[k]] /
                                        sel_prev[selbin[k], seltheta[k]]
                                        ) ** selpow[k]
                    if ratio is not None:
                        nz = prev != 0
                        rel = np.abs(dv[nz]/(prev[nz]*ratio[nz]) - 1.0)
                        self.assertLess(
                            rel.max(), RESCALE_RTOL,
                            f"{order}: {sector} step {r} is not the "
                            f"analytic rescale (max {rel.max():.2e})")
                    prev = dv

            final_point = self._point_at(fid, steps)
            final_chi2 = u.evaluate_chi2(model, final_point)
            final_dv = np.array(ci.compute_data_vector_cluster_masked())

            # no-op probe: identical point again, bitwise
            u.evaluate_chi2(model, dict(final_point))
            self.assertTrue(
                np.array_equal(
                    np.array(ci.compute_data_vector_cluster_masked()),
                    final_dv),
                f"{order}: a no-op re-evaluation changed the data vector")

            # excursions: every sector at once (the scramble), then
            # each sector alone; after each one, the ladder's final
            # point must reproduce bitwise
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
        self._run_ladder("example1")

    def test_cache_consistency_6x2pt_N(self):
        self._run_ladder("example2")


if __name__ == "__main__":
    # worker mode first: FRESH_FLAG example point_path result_path
    # evaluates one point and exits (see _fresh_worker); .index
    # returns the flag's position, so the three values follow it
    if FRESH_FLAG in sys.argv:
        at = sys.argv.index(FRESH_FLAG)
        _fresh_worker(*sys.argv[at + 1:at + 4])
        sys.exit(0)
    unittest.main(verbosity=2)
