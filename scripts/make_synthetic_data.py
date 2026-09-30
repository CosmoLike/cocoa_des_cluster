#!/usr/bin/env python
"""Synthetic DES Y6-like data of the cluster analyses (des_cluster project,
arXiv 2503.13631): the joint data vector and its Gaussian covariance.

Run from Cocoa/ in the cocoa environment (start_cocoa.sh sourced once; the
workers inherit this process's environment):

    python projects/des_cluster/scripts/make_synthetic_data.py \
        [--threads 3] [--cache-dir DIR] [--selection-mode signal] \
        [--skip-datavector] [--skip-covariance] [--skip-chi2] [--json OUT]

Steps (each cobaya model is built in its own subprocess: cosmolike keeps
global state per process):

(a) MODEL DATA VECTOR. The compiled C code through cobaya (get_model with
    des_cluster.combo_6x2pt_N: ss, gs, gg, cg, N, cc, cs) on a temporary
    copy of data/des_cluster_y6.dataset whose mask is all ones (every
    entry computed; the likelihood always zeroes the last theta bin of
    every cs row, where Sigma = T gamma_t vanishes) and placeholder data
    and covariance files. Parameters: Table I of 2503.13631 (FIDUCIAL_*
    below), the cosmology parameterized as projects/des_cluster/
    EXAMPLE_EVALUATE2.yaml (whose camb block is reused): Omega_nu h^2 =
    0.00083 <=> mnu = 0.00083 x 94.0708/(3.046/3)^0.75 eV. Written to
    data/des_cluster_y6.datavector (columns: index, value).
(b) COVARIANCE. tests/reference/ref_covariance_full.py at the same point
    (the Python reference: CAMB, halo model, Limber spectra of every field
    pair, Gaussian covariance of the joint layout, Y transform and the
    selection factor on the cs, cc, cg rows, counts Poisson + sample
    variance). Settings aligned to the dataset and the likelihood yaml
    (n(z) files, <phi_i|z> table, binning, area, lmax, kernel mode, C_c,
    halo field halo_matter_field; CAMB with the likelihood's neutrino
    configuration). The counts N_iA
    of the covariance (Poisson, cluster shot noise Omega_s/N_iA, sample-
    variance amplitude) are those of the data vector of (a), so the file
    follows the C counts after any halo-model change (--reference-counts:
    the reference's). Selection bias: ref_covariance_full "signal" mode.
    Written to data/des_cluster_y6.cov (columns: i, j, cov; upper
    triangle with the diagonal; the full 2812 x 2812 matrix, zeros
    included; Git LFS via the *cov pattern of .gitattributes).
(c) CHECKS. Layout (sizes/starts of the compiled code vs the reference
    layout, T and B(theta) of the C code vs the reference), the C model
    vs the reference vector per block (delta^T C^-1 delta, uncut and
    under each mask), positive definiteness and condition numbers of the
    covariance restricted to each mask, and the signal-to-noise per block
    after the cuts (compared with the Y3 numbers of 2503.13632).
(d) CHI2. combo_4x2pt_N and combo_6x2pt_N through cobaya at the
    fiducial with the files of (a) and (b): chi2 < 1e-6 is asserted (the
    data vector is the model at that point).
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

# Thread caps of every numerical library, set before numpy loads (numpy's
# BLAS otherwise starts one thread per core); an explicit value in the
# environment wins. The workers get --threads for all of them.
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
               "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
DEFAULT_THREADS = 3
for _var in THREAD_VARS:
    os.environ.setdefault(_var, str(DEFAULT_THREADS))

import numpy as np                                                   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PRJ = os.path.dirname(HERE)
DATA = os.path.join(PRJ, "data")
REFDIR = os.path.join(PRJ, "tests", "reference")
LIKDIR = os.path.join(PRJ, "likelihood")
if REFDIR not in sys.path:
    sys.path.insert(0, REFDIR)

DATASET = "des_cluster_y6.dataset"
MASKS = {"4x2pt_N": "des_cluster_y6_4x2ptN.mask", "6x2pt_N": "des_cluster_y6_6x2ptN.mask"}
COMBO_DATASETS = {"4x2pt_N": "des_cluster_y6_4x2ptN.dataset",
                  "6x2pt_N": "des_cluster_y6_6x2ptN.dataset"}
DATAVECTOR_FILE = "des_cluster_y6.datavector"
COV_FILE = "des_cluster_y6.cov"
BLOCKS = ("ss", "gs", "gg", "cg", "N", "cc", "cs")
CHI2_TOL = 1e-6

# ----------------------------------------------------------------------------
# Table I of arXiv 2503.13631 (fiducial = prior mean where the prior is Gaussian)
# ----------------------------------------------------------------------------
OMEGA_NU_H2 = 0.00083
NNU = 3.046
NEUTRINO_MASS_FAC = 94.0708   # EXAMPLE_EVALUATE2.yaml omegach2 lambda
FIDUCIAL_COSMO = dict(
    As_1e9=2.19, ns=0.96859, H0=69.0, omegab=0.048, omegam=0.3,
    mnu=OMEGA_NU_H2 * NEUTRINO_MASS_FAC / (NNU / 3.0) ** 0.75,
    w=-1.0, wa=0.0, tau=0.0697186)
FIDUCIAL_NUISANCE = {
    # sources: photo-z shift (eq 29), shear calibration
    "DES_DZ_S1": 0.034, "DES_DZ_S2": 0.028, "DES_DZ_S3": 0.011, "DES_DZ_S4": -0.010,
    "DES_M1": 0.0, "DES_M2": 0.0, "DES_M3": 0.0, "DES_M4": 0.0,
    # NLA: a1, eta1
    "DES_A1_1": 0.0, "DES_A1_2": 0.0,
    # MagLim lenses: photo-z shift, linear bias, point mass
    "DES_DZ_L1": 0.005, "DES_DZ_L2": 0.003, "DES_DZ_L3": 0.001,
    "DES_DZ_L4": -0.002, "DES_DZ_L5": 0.001, "DES_DZ_L6": 0.008,
    "DES_B1_1": 1.42, "DES_B1_2": 1.66, "DES_B1_3": 1.70,
    "DES_B1_4": 1.62, "DES_B1_5": 1.78, "DES_B1_6": 1.75,
    "DES_PM1": 0.0, "DES_PM2": 0.0, "DES_PM3": 0.0,
    "DES_PM4": 0.0, "DES_PM5": 0.0, "DES_PM6": 0.0,
    # clusters: MOR (eqs 18-19), selection bias (eq 23)
    "DES_CL_LNLAMBDA0": 4.26, "DES_CL_A_LAMBDA": 0.943,
    "DES_CL_SIGMA_INT": 0.15, "DES_CL_B_LAMBDA": 0.207,
    "DES_CL_BS1": 1.1, "DES_CL_BS2": 0.2, "DES_CL_R0": 30.0,
}
# fixed in the likelihood yaml files; checked against Table I at build time
FIXED_EXPECTED = {
    "DES_BMAG_1": -1.57, "DES_BMAG_2": -1.70, "DES_BMAG_3": -0.25,
    "DES_BMAG_4": 1.50, "DES_BMAG_5": 2.22, "DES_BMAG_6": 2.80,
    "DES_DZ2_L1": 1.0, "DES_DZ2_L2": 1.0, "DES_DZ2_L3": 1.0,
    "DES_DZ2_L4": 1.0, "DES_DZ2_L5": 1.0, "DES_DZ2_L6": 1.0,
    "DES_CL_BSZ": 0.0,
}

# DES Y3 cluster analysis (arXiv 2503.13632): S/N per block after cuts
Y3_SN = {"N": 94.5, "cs": 31.8, "cc": 18.8, "cg": 39.6, "gg": 52.5}
Y3_NPOINTS = {"N": 12, "cs": 404, "cc": 149, "cg": 124, "gg": 31}


def log(msg):
    print(f"[make_synthetic_data] {msg}", flush=True)


# ============================================================================
# configuration: dataset and likelihood yaml
# ============================================================================
def load_yaml(path):
    """A yaml file, ignoring cobaya tags (!defaults)."""
    import yaml

    class Loader(yaml.SafeLoader):
        pass

    Loader.add_multi_constructor("!", lambda loader, suffix, node: None)
    with open(path) as f:
        return yaml.load(f, Loader=Loader)


def load_dataset(path=os.path.join(DATA, DATASET)):
    from getdist import IniFile
    ini = IniFile(path)

    def lst(key, tp):
        return [tp(x) for x in ini.string(key).replace(",", " ").split()]

    return dict(
        nz_lens_file=ini.relativeFileName("nz_lens_file"),
        nz_source_file=ini.relativeFileName("nz_source_file"),
        nz_cluster_file=ini.relativeFileName("nz_cluster_file"),
        lens_ntomo=ini.int("lens_ntomo"), source_ntomo=ini.int("source_ntomo"),
        n_theta=ini.int("n_theta"), theta_min_arcmin=ini.float("theta_min_arcmin"),
        theta_max_arcmin=ini.float("theta_max_arcmin"),
        cluster_ntomo=ini.int("cluster_ntomo"),
        cluster_zbin_edges=lst("cluster_zbin_edges", float),
        richness_edges=lst("richness_edges", float),
        survey_area_deg2=ini.float("survey_area_deg2"),
        cg_lens_bins=lst("cg_lens_bins", int))


def reference_settings(ds, lik):
    """ClusterReference settings aligned to the dataset and the likelihood
    (every lens bin: the covariance needs the six of CL+3x2pt)."""
    return dict(
        photoz="table", phi_table_file=ds["nz_cluster_file"],
        source_nz=ds["nz_source_file"], lens_nz=ds["nz_lens_file"],
        lmax=int(lik["lmax"]), area_deg2=ds["survey_area_deg2"],
        zc_edges=ds["cluster_zbin_edges"], lambda_edges=ds["richness_edges"],
        ntheta=ds["n_theta"], tmin_arcmin=ds["theta_min_arcmin"],
        tmax_arcmin=ds["theta_max_arcmin"], lens_bins=list(range(ds["lens_ntomo"])),
        kernel_mode=int(lik["cluster_kernel_mode"]), C_c=float(lik["cluster_magnification"]),
        # the halo field of the likelihood yaml: 0 = total matter, 1 = cold
        # dark matter + baryons (sigma(M) from the reference's P_cb, rho_cb
        # in R(M) and dn/dM); the data vector of step (a) follows the same
        # key through the likelihood
        hmf_matter=("cb" if int(lik["halo_matter_field"]) == 1 else "tot"),
        hmf_alpha_mode=int(lik["cluster_hmf_alpha_mode"]))


# ============================================================================
# cobaya workers (run in their own subprocess)
# ============================================================================
def cobaya_info(combo, data_path, data_file):
    """cobaya input: the combo likelihood on a dataset, CAMB as in
    EXAMPLE_EVALUATE2.yaml, the cosmology fixed at Table I."""
    ex = load_yaml(os.path.join(PRJ, "EXAMPLE_EVALUATE2.yaml"))
    c = FIDUCIAL_COSMO
    params = {
        "As_1e9": {"value": c["As_1e9"], "drop": True},
        "As": {"value": "lambda As_1e9: 1e-9 * As_1e9"},
        "ns": {"value": c["ns"]},
        "H0": {"value": c["H0"]},
        "omegab": {"value": c["omegab"], "drop": True},
        "omegam": {"value": c["omegam"], "drop": True},
        "mnu": {"value": c["mnu"]},
        "w": {"value": c["w"]},
        "wa": {"value": c["wa"]},
        "tau": {"value": c["tau"]},
        "omegabh2": {"value": "lambda omegab, H0: omegab*(H0/100)**2"},
        "omegach2": {"value": ("lambda omegam, omegab, mnu, H0: (omegam-omegab)*(H0/100)**2"
                               f"-(mnu*({NNU}/3)**0.75)/{NEUTRINO_MASS_FAC}")},
        "sigma8": {"latex": r"\sigma_8"},
        "omegan2": {"latex": r"\Omega_\nu h^2"},
    }
    return {
        "likelihood": {f"des_cluster.combo_{combo}": {
            "path": data_path, "data_file": data_file, "print_datavector": False}},
        "theory": {"camb": dict(ex["theory"]["camb"])},
        "params": params,
        "stop_at_error": True,
        "debug": False,
    }


def build_point(model):
    """Sampled parameters at the Table I fiducial; every one must be known."""
    sampled = list(model.parameterization.sampled_params())
    missing = [p for p in sampled if p not in FIDUCIAL_NUISANCE]
    if missing:
        raise RuntimeError(f"no Table I fiducial for sampled parameters {missing}")
    const = model.parameterization.constant_params()
    missing = [p for p in FIXED_EXPECTED if p not in const]
    if missing:
        raise RuntimeError(f"expected fixed parameters {missing} not in the model")
    # the combo's fixed parameters (e.g. CL+GC fixes lens bins 4-6) and the
    # yaml constants must sit at Table I too
    for p, v in list(FIXED_EXPECTED.items()) + list(FIDUCIAL_NUISANCE.items()):
        if p in const and abs(const[p] - v) > 1e-12:
            raise RuntimeError(f"fixed parameter {p} = {const[p]} (Table I: {v})")
    return {p: FIDUCIAL_NUISANCE[p] for p in sampled}


def evaluate(combo, data_path, data_file):
    """(model, likelihood, point, chi2, data vector, derived) at the fiducial.

    The posterior call runs CAMB and the likelihood, and leaves the cobaya
    provider at this cosmology. The masked model vector is then read with
    the likelihood's own get_datavector, which repeats the computation of
    logp (set_cosmo_related, the nuisance setters, the masked data vector)
    from that provider state and the likelihood's input parameters, and
    returns the vector instead of the log-likelihood.
    """
    from cobaya.model import get_model
    os.chdir(os.environ["ROOTDIR"])
    model = get_model(cobaya_info(combo, data_path, data_file))
    lik = model.likelihood[f"des_cluster.combo_{combo}"]
    point = build_point(model)
    post = model.logposterior(point, cached=False)
    chi2 = -2.0 * float(post.loglikes[0])
    derived = dict(zip(model.parameterization.derived_params(), post.derived))
    # to_input resolves every input parameter at this point (sampled,
    # fixed, and those computed from them); the likelihood takes its own
    # subset, lik.input_params, as logp does
    all_inputs = model.parameterization.to_input(point)
    lik_inputs = {}
    for name in lik.input_params:
        lik_inputs[name] = all_inputs[name]
    dv = np.array(lik.get_datavector(**lik_inputs), dtype=float)
    return model, lik, point, chi2, dv, derived


def write_placeholder_dataset(tmp):
    """A copy of data/des_cluster_y6.dataset in the folder tmp with an
    all-ones mask, a zero data vector and an identity covariance (the
    likelihood reads all three at initialization); the other files point
    back to data/. Returns the dataset file name (relative to tmp)."""
    from ref_covariance_full import joint_layout
    ds = load_dataset()
    lay = joint_layout(ds["cluster_ntomo"], len(ds["richness_edges"]) - 1,
                       ds["source_ntomo"], ds["lens_ntomo"], ds["n_theta"],
                       ds["cg_lens_bins"])
    n = lay["ndata"]
    idx = np.arange(n)
    np.savetxt(os.path.join(tmp, "ones.mask"), np.column_stack([idx, np.ones(n)]),
               fmt="%d %d")
    np.savetxt(os.path.join(tmp, "zeros.datavector"),
               np.column_stack([idx, np.zeros(n)]), fmt="%d %.1f")
    np.savetxt(os.path.join(tmp, "identity.cov"),
               np.column_stack([idx, idx, np.ones(n)]), fmt="%d %d %.1f")
    lines = []
    for line in open(os.path.join(DATA, DATASET)).read().splitlines():
        key = line.split("=")[0].strip() if not line.lstrip().startswith("#") else ""
        if key == "data_file":
            line = f"data_file = {os.path.join(tmp, 'zeros.datavector')}"
        elif key == "cov_file":
            line = f"cov_file = {os.path.join(tmp, 'identity.cov')}"
        elif key == "mask_file":
            line = f"mask_file = {os.path.join(tmp, 'ones.mask')}"
        elif key.endswith("_file") and "=" in line:
            line = f"{key} = {os.path.join(DATA, line.split('=', 1)[1].strip())}"
        lines.append(line)
    with open(os.path.join(tmp, "ones.dataset"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return "ones.dataset"


def worker_datavector(out):
    """(a): the all-ones-mask model vector and the layout seen by the C code."""
    import cosmolike_des_cluster_interface as ci
    tmp = tempfile.mkdtemp(prefix="des_cluster_synth_")
    try:
        name = write_placeholder_dataset(tmp)
        t0 = time.time()
        model, lik, point, chi2, dv, derived = evaluate("6x2pt_N", tmp, name)
        log(f"model vector (6x2pt_N, all-ones mask) in {time.time() - t0:.1f} s")
        np.savez(out, dv=dv, sizes=np.array(ci.compute_data_vector_cluster_sizes()),
                 starts=np.array(ci.compute_data_vector_cluster_starts()),
                 T=np.array(ci.get_cluster_ytransform_matrix()),
                 B=np.array(ci.get_cluster_selection_factor()),
                 mask=np.array(ci.get_mask_cluster()),
                 derived=json.dumps(derived), point=json.dumps(point))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def worker_chi2(combo, out):
    """(d): chi2 of a combo at the fiducial against the written files."""
    t0 = time.time()
    model, lik, point, chi2, dv, derived = evaluate(combo, DATA, COMBO_DATASETS[combo])
    log(f"{combo}: chi2 = {chi2:.3e} ({time.time() - t0:.1f} s)")
    with open(out, "w") as f:
        json.dump(dict(combo=combo, chi2=chi2, ndata=int(dv.size)), f)


# Sourcing start_cocoa.sh (in another shell) removes and re-creates the
# cobaya/likelihoods/<project> symlinks; a model built in that window fails
# with one of these messages. The workers inherit this process's environment
# (start_cocoa.sh is sourced once) and are retried when they hit the race.
SYMLINK_RACE = ("could not be found", "does not recognize some options")


def run_worker(args, threads, retries=2):
    env = dict(os.environ, **{var: str(threads) for var in THREAD_VARS})
    cmd = [sys.executable, os.path.abspath(__file__)] + args
    for attempt in range(retries + 1):
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        sys.stdout.write(r.stdout)
        sys.stderr.write(r.stderr)
        if r.returncode == 0:
            return
        race = any(msg in r.stdout + r.stderr for msg in SYMLINK_RACE)
        if not race or attempt == retries:
            break
        log(f"worker {args[1]}: likelihood not found (start_cocoa.sh symlink race?), retrying")
        time.sleep(5)
    raise RuntimeError(f"worker {args} failed (exit {r.returncode})")


# ============================================================================
# (b) covariance
# ============================================================================
def source_hash(settings, selection_mode):
    h = hashlib.sha256()
    h.update(json.dumps(settings, sort_keys=True, default=str).encode())
    h.update(selection_mode.encode())
    for name in sorted(os.listdir(REFDIR)):
        if name.endswith(".py"):
            h.update(open(os.path.join(REFDIR, name), "rb").read())
    for key in ("phi_table_file", "source_nz", "lens_nz"):
        h.update(open(settings[key], "rb").read())
    return h.hexdigest()[:16]


def camb_num_massive_neutrinos():
    """The number of massive neutrino eigenstates of the likelihood's CAMB
    (EXAMPLE_EVALUATE2.yaml camb extra_args; CAMB's default is 1)."""
    ex = load_yaml(os.path.join(PRJ, "EXAMPLE_EVALUATE2.yaml"))
    return int(ex["theory"]["camb"].get("extra_args", {}).get("num_massive_neutrinos", 1))


def compute_covariance(settings, cg_lens_bins, selection_mode, cache_dir=None, counts=None):
    """(cov, reference vector pieces) from the Python reference.

    counts: N_iA of the synthetic data vector (the C model). They set the
    Poisson term, the cluster shot noise 1/n_cA = Omega_s/N_iA and the
    sample-variance amplitude N_iA N_iB S_i,AB of the counts block, so the
    covariance follows the counts of the data file even when the C and
    reference halo models differ (None: the reference counts).

    The reference CAMB run carries the neutrino configuration of the
    likelihood's CAMB (one massive eigenstate at Omega_nu h^2 = 0.00083;
    ref_cosmology's default is three degenerate ones, which moves sigma_8
    by 0.3% and the counts by up to 2.6% at the same Omega_nu h^2)."""
    from reference_cluster import ClusterReference, COSMO_KEYS, FIDUCIAL
    from ref_cosmology import Cosmology
    from ref_covariance_full import full_covariance
    nmassive = camb_num_massive_neutrinos()
    key = source_hash(dict(settings, cg_lens_bins=list(cg_lens_bins), num_massive_nu=nmassive,
                           counts=None if counts is None else
                           np.asarray(counts, dtype=float).round(8).tolist()),
                      selection_mode)
    cache = os.path.join(cache_dir, f"synthetic_cov_{key}.npz") if cache_dir else None
    if cache and os.path.exists(cache):
        log(f"covariance from cache {cache}")
        z = np.load(cache, allow_pickle=True)
        return {k: z[k] for k in z.files}
    t0 = time.time()
    from reference_cluster import DEFAULT_SETTINGS
    s = dict(DEFAULT_SETTINGS, **settings)
    cosmo = Cosmology({k: FIDUCIAL[k] for k in COSMO_KEYS}, kmax=s["kmax"],
                      halofit=s["halofit"], k_per_logint=s["k_per_logint"],
                      nl_z_order=s["pk_nl_z_order"], num_massive_nu=nmassive)
    log(f"reference CAMB: {nmassive} massive neutrino(s), mnu = {cosmo.mnu:.5f} eV, "
        f"sigma_8 = {cosmo.results.get_sigma8_0():.6f}")
    ref = ClusterReference(None, settings, cosmo=cosmo, verbose=True)
    cov, info = full_covariance(ref, selection_mode=selection_mode, counts=counts,
                                cg_lens_bins=cg_lens_bins)
    ref_dv = reference_vector(ref, info["layout"])
    out = dict(cov=cov, counts=info["counts"], counts_ref=ref.counts(), S=info["S"],
               ref_dv=ref_dv,
               B=ref.selection(), T=ref.T, edges=ref.edges,
               sigma8=cosmo.results.get_sigma8_0())
    log(f"reference covariance in {time.time() - t0:.1f} s")
    if cache:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(cache, **out)
    return out


def reference_vector(ref, lay):
    """The reference model in the joint layout (ss, gs, gg by Limber
    projection of the reference spectra; the cluster blocks from
    ClusterReference.data_vector)."""
    from ref_covariance_full import kernels
    from ref_projection import cl_on_integers
    sp = ref.spectra()
    s = ref.settings
    K = kernels(ref.edges, s["lmax"])
    m = np.asarray(ref.params["shear_m"])
    dv = np.zeros(lay["ndata"])
    for o, (block, legs, kind, lab) in enumerate(lay["obs"]):
        if block == "ss":
            _, i, j = lab
            cl = cl_on_integers(sp["ells"], sp["C_ss"][i, j], s["lmax"])
            dv[lay["index"][o]] = K[kind] @ cl * (1 + m[i]) * (1 + m[j])
        elif block == "gs":
            k, j = lab
            cl = cl_on_integers(sp["ells"], sp["C_gs"][k, j], s["lmax"])
            dv[lay["index"][o]] = K["gt"] @ cl * (1 + m[j])
        elif block == "gg":
            (k,) = lab
            cl = cl_on_integers(sp["ells"], sp["C_gg"][k, k], s["lmax"])
            dv[lay["index"][o]] = K["w"] @ cl
    cdv, st = ref.data_vector(selected=True)      # N, cs, cc, cg (reference layout)
    for b in ("N", "cs", "cc", "cg"):
        end = {"N": "cs", "cs": "cc", "cc": "cg", "cg": "end"}[b]
        seg = cdv[st[b]:st[end]]
        if seg.size != lay["sizes"][b]:
            raise RuntimeError(f"reference block {b}: {seg.size} != {lay['sizes'][b]}")
        dv[lay["starts"][b]:lay["starts"][b] + seg.size] = seg
    return dv


def write_datavector(path, dv):
    np.savetxt(path, np.column_stack([np.arange(dv.size), dv]), fmt="%d %.16e")


def write_covariance(path, cov):
    iu = np.triu_indices(cov.shape[0])
    with open(path, "w") as f:
        np.savetxt(f, np.column_stack([iu[0], iu[1], cov[iu]]), fmt="%d %d %.15e")


def read_covariance(path, n):
    try:
        import pandas as pd
        t = pd.read_csv(path, sep=" ", header=None).to_numpy()
    except ImportError:
        t = np.loadtxt(path)
    cov = np.zeros((n, n))
    i, j = t[:, 0].astype(int), t[:, 1].astype(int)
    cov[i, j] = t[:, 2]
    cov[j, i] = t[:, 2]
    return cov


# ============================================================================
# (c) checks
# ============================================================================
def effective_mask(mask, lay):
    """The mask as IPCluster applies it: the last theta bin of every cs row
    is always dropped (the last row of T vanishes)."""
    m = np.asarray(mask, dtype=bool).copy()
    nt = lay["index"].shape[1]
    for o, ob in enumerate(lay["obs"]):
        if ob[0] == "cs":
            m[lay["index"][o][nt - 1]] = False
    return m


def mask_report(cov, dv, mask, lay, label):
    """PD, condition numbers and S/N per block under one mask."""
    idx = np.where(mask)[0]
    C = cov[np.ix_(idx, idx)]
    d = np.sqrt(np.diag(C))
    corr = C / np.outer(d, d)
    evc = np.linalg.eigvalsh(corr)
    try:
        np.linalg.cholesky(C)
        chol = True
    except np.linalg.LinAlgError:
        chol = False
    # PD is judged on the correlation matrix (scale invariant) and by a
    # Cholesky factorization: the variances span ~18 decades (counts vs
    # xi-), so the eigenvalues of the raw matrix below ~1e-16 lambda_max
    # are round-off, and a raw eigen-solver can return small negative ones
    ev = np.linalg.eigvalsh(C)
    rep = dict(label=label, npoints=int(idx.size), pd=bool(chol and evc[0] > 0),
               cholesky=chol, corr_eig_min=float(evc[0]), corr_eig_max=float(evc[-1]),
               cond_corr=float(evc[-1] / evc[0]) if evc[0] > 0 else float("inf"),
               raw_eig_min=float(ev[0]), raw_eig_max=float(ev[-1]),
               raw_eig_nonpositive=int(np.sum(ev <= 0)),
               variance_range=[float(d.min() ** 2), float(d.max() ** 2)])
    Cinv = np.linalg.inv(C)
    rep["SN_total"] = float(np.sqrt(dv[idx] @ Cinv @ dv[idx]))
    rep["blocks"] = {}
    for b in BLOCKS:
        sel = np.zeros(mask.size, dtype=bool)
        sel[lay["starts"][b]:lay["starts"][b] + lay["sizes"][b]] = True
        ib = np.where(sel & mask)[0]
        if ib.size == 0:
            continue
        Cb = cov[np.ix_(ib, ib)]
        rep["blocks"][b] = dict(npoints=int(ib.size),
                                SN=float(np.sqrt(dv[ib] @ np.linalg.solve(Cb, dv[ib]))))
    return rep


def delta_chi2(cov, d1, d2, mask, lay):
    """(d1 - d2)^T C^-1 (d1 - d2) per block and in total under a mask."""
    out = {}
    delta = d1 - d2
    for b in BLOCKS + ("total",):
        if b == "total":
            ib = np.where(mask)[0]
        else:
            sel = np.zeros(mask.size, dtype=bool)
            sel[lay["starts"][b]:lay["starts"][b] + lay["sizes"][b]] = True
            ib = np.where(sel & mask)[0]
        if ib.size:
            out[b] = float(delta[ib] @ np.linalg.solve(cov[np.ix_(ib, ib)], delta[ib]))
    return out


# ============================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--threads", type=int, default=DEFAULT_THREADS,
                    help="OpenMP and BLAS threads of the cobaya workers")
    ap.add_argument("--cache-dir", default=None, help="cache of the reference covariance")
    ap.add_argument("--selection-mode", default="signal",
                    choices=("signal", "jacobian", "none"))
    ap.add_argument("--skip-datavector", action="store_true",
                    help="reuse data/des_cluster_y6.datavector")
    ap.add_argument("--skip-covariance", action="store_true",
                    help="do not rewrite data/des_cluster_y6.cov (the checks read the file)")
    ap.add_argument("--skip-chi2", action="store_true")
    ap.add_argument("--reference-counts", action="store_true",
                    help="counts of the Python reference in the covariance (default: "
                         "the counts of the synthetic data vector)")
    ap.add_argument("--json", default=None, help="write the report here")
    ap.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--combo", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--out", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)

    if a.worker == "datavector":
        return worker_datavector(a.out)
    if a.worker == "chi2":
        return worker_chi2(a.combo, a.out)
    if "ROOTDIR" not in os.environ:
        raise SystemExit("source start_cocoa.sh first (ROOTDIR not set)")

    report = {}
    ds = load_dataset()
    lik = load_yaml(os.path.join(LIKDIR, "combo_6x2pt_N.yaml"))
    from ref_covariance_full import joint_layout
    lay = joint_layout(ds["cluster_ntomo"], len(ds["richness_edges"]) - 1,
                       ds["source_ntomo"], ds["lens_ntomo"], ds["n_theta"], ds["cg_lens_bins"])
    report["layout"] = dict(sizes=lay["sizes"], starts=lay["starts"], ndata=lay["ndata"])
    work = tempfile.mkdtemp(prefix="des_cluster_synth_main_")
    dv_path = os.path.join(DATA, DATAVECTOR_FILE)
    cov_path = os.path.join(DATA, COV_FILE)
    try:
        # --- (a) model data vector ---
        if not a.skip_datavector:
            out = os.path.join(work, "datavector.npz")
            run_worker(["--worker", "datavector", "--out", out], a.threads)
            z = np.load(out)
            dv = z["dv"]
            if dv.size != lay["ndata"]:
                raise RuntimeError(f"C vector has {dv.size} entries, layout {lay['ndata']}")
            write_datavector(dv_path, dv)
            log(f"wrote {dv_path}")
            sizes_c = dict(zip(BLOCKS, z["sizes"].tolist()))
            starts_c = dict(zip(BLOCKS, z["starts"].tolist()))
            report["layout"]["compiled_sizes_match"] = sizes_c == lay["sizes"]
            report["layout"]["compiled_starts_match"] = starts_c == lay["starts"]
            if not (sizes_c == lay["sizes"] and starts_c == lay["starts"]):
                raise RuntimeError(f"layout mismatch: C {sizes_c} {starts_c}, "
                                   f"reference {lay['sizes']} {lay['starts']}")
            report["mask_ones_unmasked"] = int(z["mask"].sum())
            report["derived"] = json.loads(str(z["derived"]))
            T_c, B_c = z["T"], z["B"]
        else:
            dv = np.loadtxt(dv_path)[:, 1]
            T_c = B_c = None

        # --- (b) covariance ---
        settings = reference_settings(ds, lik)
        iN = slice(lay["starts"]["N"], lay["starts"]["N"] + lay["sizes"]["N"])
        counts_dv = dv[iN].reshape(ds["cluster_ntomo"], len(ds["richness_edges"]) - 1)
        res = compute_covariance(settings, ds["cg_lens_bins"], a.selection_mode, a.cache_dir,
                                 counts=None if a.reference_counts else counts_dv)
        cov = res["cov"]
        if not a.skip_covariance:
            t0 = time.time()
            write_covariance(cov_path, cov)
            log(f"wrote {cov_path} ({os.path.getsize(cov_path) / 2**20:.1f} MiB, "
                f"{time.time() - t0:.1f} s)")
        cov_file = read_covariance(cov_path, lay["ndata"])
        scale = np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
        ok = scale > 0
        report["cov_file_roundtrip_max_rel"] = float(
            np.max(np.abs(cov_file - cov)[ok] / scale[ok]))
        report["cov_file_MiB"] = os.path.getsize(cov_path) / 2**20
        report["cov_symmetric_max"] = float(np.max(np.abs(cov - cov.T)))
        cov = cov_file

        # --- (c) checks ---
        if T_c is not None:
            report["T_max_abs_diff"] = float(np.max(np.abs(T_c - res["T"])))
            report["B_max_rel_diff"] = float(np.max(np.abs(B_c / res["B"] - 1.0)))
        report["sigma8_reference"] = float(res["sigma8"])
        report["counts_C"] = counts_dv.tolist()
        report["counts_ref"] = res["counts_ref"].tolist()
        report["counts_in_covariance"] = "reference" if a.reference_counts else "data vector (C)"
        report["counts_C_over_ref_max"] = float(np.max(np.abs(
            counts_dv / res["counts_ref"] - 1.0)))
        y3 = np.loadtxt(os.path.join(DATA, "y3_redmapper_counts.txt"))[:, 4]
        report["counts_C_over_Y3"] = (dv[iN] / y3).tolist()
        masks = {"none": np.ones(lay["ndata"], dtype=bool)}
        for combo, f in MASKS.items():
            masks[combo] = np.loadtxt(os.path.join(DATA, f))[:, 1] > 0.5
        report["masks"] = {}
        report["C_vs_reference_chi2"] = {}
        for label, m in masks.items():
            me = effective_mask(m, lay)
            report["masks"][label] = mask_report(cov, dv, me, lay, label)
            report["C_vs_reference_chi2"][label] = delta_chi2(cov, dv, res["ref_dv"], me, lay)
        print_report(report)
        for combo in MASKS:
            if not report["masks"][combo]["pd"]:
                raise AssertionError(f"covariance not positive definite on the {combo} mask")

        # --- (d) chi2 at the fiducial ---
        if not a.skip_chi2:
            report["chi2"] = {}
            for combo in COMBO_DATASETS:
                out = os.path.join(work, f"chi2_{combo}.json")
                run_worker(["--worker", "chi2", "--combo", combo, "--out", out], a.threads)
                r = json.load(open(out))
                report["chi2"][combo] = r["chi2"]
                log(f"chi2({combo}) at the fiducial = {r['chi2']:.3e}")
            bad = {k: v for k, v in report["chi2"].items() if not v < CHI2_TOL}
            if bad:
                raise AssertionError(f"chi2 at the fiducial above {CHI2_TOL}: {bad}")
    finally:
        shutil.rmtree(work, ignore_errors=True)
        if a.json:
            with open(a.json, "w") as f:
                json.dump(report, f, indent=1, default=float)
    return report


def print_report(r):
    lay = r["layout"]
    log(f"layout sizes {lay['sizes']} (ndata {lay['ndata']})")
    for k in ("compiled_sizes_match", "compiled_starts_match", "mask_ones_unmasked",
              "sigma8_reference", "counts_in_covariance",
              "T_max_abs_diff", "B_max_rel_diff", "counts_C_over_ref_max",
              "cov_file_roundtrip_max_rel", "cov_file_MiB", "cov_symmetric_max"):
        if k in r:
            log(f"{k}: {r[k]}")
    if "derived" in r:
        log(f"derived at the fiducial: {r['derived']}")
    for label, m in r["masks"].items():
        log(f"mask {label}: {m['npoints']} points, PD {m['pd']} (Cholesky {m['cholesky']}, "
            f"correlation eigenvalues [{m['corr_eig_min']:.3e}, {m['corr_eig_max']:.3e}], "
            f"cond(corr) {m['cond_corr']:.3e}); raw eigenvalues [{m['raw_eig_min']:.3e}, "
            f"{m['raw_eig_max']:.3e}] ({m['raw_eig_nonpositive']} <= 0 by round-off), "
            f"variances [{m['variance_range'][0]:.2e}, {m['variance_range'][1]:.2e}]; "
            f"S/N total {m['SN_total']:.1f}")
        for b, v in m["blocks"].items():
            # Y3 quotes the S/N of the measured (noisy) vector: E[d^T C^-1 d]
            # = (S/N)^2 + n_points, so sqrt(S/N^2 - n) is its noise-free analog
            y3 = (f" (Y3: {Y3_NPOINTS[b]} points, S/N {Y3_SN[b]}, noise-debiased "
                  f"{np.sqrt(Y3_SN[b]**2 - Y3_NPOINTS[b]):.1f})"
                  if label == "4x2pt_N" and b in Y3_SN else "")
            log(f"   {b:>2}: {v['npoints']:4d} points, S/N {v['SN']:7.1f}{y3}")
        log(f"   C model vs reference, delta^T C^-1 delta: "
            + ", ".join(f"{b} {v:.3g}" for b, v in r["C_vs_reference_chi2"][label].items()))


if __name__ == "__main__":
    main()
