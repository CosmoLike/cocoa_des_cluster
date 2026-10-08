"""Accuracy-knob sweep of the des_cluster data vector (4x2pt + N / 6x2pt + N).

Every setting runs in a fresh process (the compiled interface keeps global
state), fed the SAME cosmology tables (the Python reference's CAMB tables,
through compare_reference.py's C-side init chain), so only the numerical
knobs differ. Each setting's full joint data vector (every entry computed:
cluster blocks unmasked) is scored against the high-accuracy setting with
the joint Gaussian covariance

    delta chi2 = (d_setting - d_high)^T C^-1 (d_setting - d_high)

on two masks: the production CL+3x2pt mask and the most aggressive
positive-definite mask (small scales visible: a scale-cut mask hides the
scales where numerical errors are largest, so a sweep scored only under it
would rate the settings too well). The cosmolike time of one full
evaluation (cosmology tables rebuilt) is reported next to it. Target: the
fastest setting with delta chi2 < 0.2, the accuracy budget of the whole
code.

The reference tables come from the cache of compare_reference.py (its
--cache-dir; run that script first). Two environment variables restrict a
run: KNOB_SWEEP_ONLY = comma-separated labels of SETTINGS to run (the
"high" setting is always needed for the scores), KNOB_SWEEP_PROBE = the
probe name passed to the library (default 4x2pt_N).

Usage (cocoa environment active, from Cocoa/):
    OMP_NUM_THREADS=4 python projects/des_cluster/tests/validation/knob_sweep.py \\
        --cache-dir <compare_reference cache> \\
        --cov projects/des_cluster/data/des_cluster_y6_cov.npy
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# (label, accuracyboost, integration_accuracy, lmax). "high" is the
# reference every other setting is scored against; "default" is the
# likelihood default of the sweep's base configuration; each other row
# moves one knob ("hdi" = integration_accuracy).
SETTINGS = [
    ("high",           2.0, 2, 100000),
    ("default",        1.0, 0,  75000),
    ("boost 0.75",     0.75, 0, 75000),
    ("boost 1.5",      1.5, 0,  75000),
    ("hdi 1",          1.0, 1,  75000),
    ("lmax 50000",     1.0, 0,  50000),
    ("lmax 100000",    1.0, 0, 100000),
]


def child(label, boost, hdi, lmax, cache_dir, out, threads):
    """One setting: init, feed the reference cosmology, time one full
    evaluation, write the joint data vector.

    Runs in a child process. The library is initialized with
    compare_reference.init_c at the setting's accuracy, receives the
    matched reference's cosmology tables and the fiducial nuisance
    parameters, and computes the full joint vector under an all-ones mask.
    A second evaluation after a small change of every ln P table
    (lnP_shift = 1e-4) times the refill of every table.

    Arguments:
      label     = the setting's label (names the work folder).
      boost     = accuracyboost.
      hdi       = integration_accuracy.
      lmax      = lmax of the C_ell tables.
      cache_dir = the compare_reference cache folder.
      out       = output .npy path of the data vector.
      threads   = OpenMP threads of the library.

    Returns:
      nothing; writes out (the vector) and out + ".json" (the two timings
      in seconds, and the block sizes and starts).
    """
    import compare_reference as cr
    cr.PROBE = os.environ.get("KNOB_SWEEP_PROBE", cr.PROBE)
    cfg = cr.load_config()
    cfg["likelihood"]["accuracyboost"] = boost
    cfg["likelihood"]["integration_accuracy"] = hdi
    cfg["likelihood"]["lmax"] = lmax
    args = argparse.Namespace(cache_dir=cache_dir, recompute_reference=False)
    R, _ = cr.load_reference(args, cfg, "matched")
    workdir = os.path.join(cache_dir, "knob_" + label.replace(" ", "_"))
    os.makedirs(workdir, exist_ok=True)
    ci, info = cr.init_c(cfg, workdir, threads)
    from reference_cluster import FIDUCIAL
    cr.set_nuisance_c(ci, cfg, dict(FIDUCIAL))
    # every entry computed (all ones mask): the joint vector at full length
    ones = np.ones(info["ndata"], dtype=int)
    ci.init_data_cluster(info["covf"], cr.write_mask(workdir, "ones.mask", ones),
                         info["dvf"])
    cr.set_cosmology_c(ci, R)
    t0 = time.perf_counter()
    dv = np.array(ci.compute_data_vector_cluster_masked())
    t1 = time.perf_counter()
    # a second evaluation at a nearby cosmology: every table refills
    cr.set_cosmology_c(ci, R, lnP_shift=1e-4)
    t2 = time.perf_counter()
    ci.compute_data_vector_cluster_masked()
    t3 = time.perf_counter()
    np.save(out, dv)
    sizes = [int(x) for x in ci.compute_data_vector_cluster_sizes()]
    starts = [int(x) for x in ci.compute_data_vector_cluster_starts()]
    with open(out + ".json", "w") as f:
        json.dump({"label": label, "t_first": t1 - t0, "t_refill": t3 - t2,
                   "sizes": sizes, "starts": starts}, f)


def row_pd_mask(cr, C, prod, cfg, layout):
    """Small scales visible: re-admit the masked theta bins of each cluster
    row (block, pair, richness) as a group while the correlation matrix of
    the kept set stays positive definite (smallest eigenvalue >=
    compare_reference.MIN_CORR_EIG = 1e-4); the last theta bin of every cs
    row stays masked (the Y transform's null row: zero by construction).
    One eigen-decomposition per row, not per point.

    Arguments:
      cr     = the compare_reference module.
      C      = joint covariance matrix.
      prod   = production 0/1 mask.
      cfg    = compare_reference.load_config() output.
      layout = dict with the block "sizes" and "starts" of the joint vector.

    Returns:
      the widened 0/1 mask.
    """
    ds = cfg["dataset"]
    nt = int(ds["n_theta"])
    sizes, starts = layout["sizes"], layout["starts"]
    blocks = dict(zip(cr.BLOCKS, zip(starts, sizes)))
    keep = prod.copy()
    for b in ("cg", "cc", "cs"):
      start, size = blocks[b]
      for r0 in range(start, start + size, nt):
        row = np.arange(r0, r0 + nt)
        if b == "cs":
          row = row[:-1]
        add = row[keep[row] == 0]
        if add.size == 0:
          continue
        trial = keep.copy()
        trial[add] = 1
        idx = np.nonzero(trial)[0]
        if cr.min_corr_eig(C[np.ix_(idx, idx)]) >= cr.MIN_CORR_EIG:
          keep = trial
    return keep


def read_cov(path, n):
    """Read the joint covariance from a .npy packed triangle or a text file.

    The .npy file holds the upper triangle with the diagonal, row by row:
    np.triu_indices(n) lists those (row, column) positions in the same
    order, and writing through cov.T (a transposed view of cov) copies them
    into the lower triangle. A text file holds "i j value" lines.

    Arguments:
      path = covariance file path.
      n    = data-vector length.

    Returns:
      the full symmetric [n, n] covariance.
    """
    cov = np.zeros((n, n))
    if path.endswith(".npy"):
        # the packed upper triangle, row by row (scripts/make_synthetic_data.py)
        iu = np.triu_indices(n)
        cov[iu] = np.load(path)
        cov.T[iu] = cov[iu]
        return cov
    data = np.loadtxt(path)
    i, j, v = data[:, 0].astype(int), data[:, 1].astype(int), data[:, 2]
    cov[i, j] = v
    cov[j, i] = v
    return cov


def main():
    """Run every setting in a child process, then print the delta chi2 table.

    Arguments:
      none; reads the command line (and, with --child, acts as one child).

    Returns:
      nothing; prints one row per setting: delta chi2 under the production
      and the widened mask, and the first and refill times.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--cov", required=True, help="joint covariance (.npy packed upper triangle, or text i j cov)")
    ap.add_argument("--mask", default=None, help="production mask (default: dataset 6x2ptN)")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", 4)))
    ap.add_argument("--child", nargs=5, default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.child:
        label, boost, hdi, lmax, out = a.child
        child(label, float(boost), int(hdi), int(lmax), a.cache_dir, out, a.threads)
        return

    import compare_reference as cr
    cfg = cr.load_config()
    outdir = os.path.join(a.cache_dir, "knob_sweep")
    os.makedirs(outdir, exist_ok=True)
    dvs, times = {}, {}
    # the settings whose label is listed in KNOB_SWEEP_ONLY (all when unset)
    only = os.environ.get("KNOB_SWEEP_ONLY")
    settings = [x for x in SETTINGS if (not only) or x[0] in only.split(",")]
    for label, boost, hdi, lmax in settings:
        out = os.path.join(outdir, label.replace(" ", "_") + ".npy")
        subprocess.run([sys.executable, __file__, "--cache-dir", a.cache_dir, "--cov", a.cov,
                        "--threads", str(a.threads), "--child", label, str(boost), str(hdi),
                        str(lmax), out], check=True)
        dvs[label] = np.load(out)
        times[label] = json.load(open(out + ".json"))
    n = dvs["high"].size
    C = read_cov(a.cov, n)
    prjdata = os.path.dirname(cfg["dataset"]["mask_file"])
    mask_path = a.mask or os.path.join(prjdata, "des_cluster_y6_6x2ptN.mask")
    prod = np.loadtxt(mask_path)[:, 1].astype(int)
    aggressive = row_pd_mask(cr, C, prod, cfg, times["high"])
    print("%-14s %12s %12s %10s %10s" % ("setting", "dchi2 prod", "dchi2 aggr",
                                         "t first", "t refill"))
    # label, *_ keeps the first element of each setting and discards the rest
    for label, *_ in settings:
        d = dvs[label] - dvs["high"]
        c_prod, _ = cr.chi2(d, C, prod == 1)
        c_aggr, _ = cr.chi2(d, C, aggressive == 1)
        print("%-14s %12.2e %12.2e %9.3fs %9.3fs" % (label, c_prod, c_aggr,
              times[label]["t_first"], times[label]["t_refill"]))


if __name__ == "__main__":
    main()
