#!/usr/bin/env python
"""Write the synthetic DES Y6-like data vector and covariance of des_cluster.

The synthetic data set of the cluster analyses of arXiv 2503.13631: the
joint data vector (the model value of every observable, blocks ss, gs,
gg, cg, N, cc, cs, 2812 entries; the table of the project README) at the
Table I fiducial point, without noise, and its Gaussian covariance C. A
likelihood evaluates chi2 = (d - m)^T C^-1 (d - m) over the entries its
mask keeps (the 0/1 files of scripts/make_cluster_mask.py), so at the
fiducial point chi2 vanishes by construction. cobaya is the sampler
framework that loads the likelihoods; a ".dataset" descriptor is the
small text file of `key = value` lines that names the data, mask and
covariance files and the binning.

Run from Cocoa/ in the cocoa environment (start_cocoa.sh sourced once; the
workers inherit this process's environment):

    python projects/des_cluster/scripts/make_synthetic_data.py \
        [--threads 3] [--cache-dir DIR] [--selection-mode signal] \
        [--skip-datavector] [--skip-covariance] [--skip-chi2] [--json OUT]

--threads sets the OpenMP and BLAS threads of the workers; --cache-dir
keeps the reference covariance in an .npz file named by a hash of all its
inputs (source_hash) and reuses it while the inputs stay the same; --json
OUT writes the report of the checks as JSON (a text format of nested
key-value objects), also when a check fails.

Steps (each cobaya model is built in its own subprocess, a worker: a
separate python that runs this file with the hidden --worker option,
because cosmolike keeps global state per process):

(a) MODEL DATA VECTOR. The compiled C code through cobaya (get_model with
    des_cluster.combo_6x2pt_N: ss, gs, gg, cg, N, cc, cs) on a temporary
    copy of data/des_cluster_y6.dataset whose mask is all ones (every
    entry computed; the likelihood always zeroes the last theta bin of
    every cs row, where the Y statistic Sigma = T gamma_t is
    Y(R_max) = Sigma(R_max) - Sigma(R_max) = 0) and placeholder data
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
    cold-matter halo prescription; CAMB with the likelihood's neutrino
    configuration). The counts N_iA of the covariance (Poisson, cluster
    shot noise Omega_s/N_iA, sample-variance amplitude) are those of the
    data vector of (a), so the file follows the C counts after any
    halo-model change (--reference-counts: the reference's). The counts
    are absolute numbers of clusters per (z bin, richness bin) over the
    survey area, and their Poisson variance is the count itself; each
    cluster lands in exactly one bin, so that term is diagonal.
    Selection bias: --selection-mode, passed to
    ref_covariance_full (default "signal": the selection factor multiplies
    the cluster signal, not the shot noise). The cs rows are y = T x of the
    gamma_t rows x, so the reference applies T C T^t within cs and T on one
    side of every cs cross block. The block between the counts and the
    two-point functions is zero, and there is no super-sample or
    trispectrum term.
    Written to data/des_cluster_y6_cov.npy: the upper triangle with the
    diagonal, row by row (cov[np.triu_indices(2812)], zeros included), as
    little-endian float64 in NumPy's .npy format, which cosmolike's
    IPCluster::set_inv_cov reads directly. The doubles are stored exactly;
    the file is 32 MB, a quarter of the 124 MB of a text table (i, j, cov)
    of the same numbers, and every regeneration adds one such file to the
    Git LFS storage (.gitattributes: *_cov.npy).
(c) CHECKS. Layout (sizes/starts of the compiled code vs the reference
    layout, T and B(theta) of the C code vs the reference), the C model
    vs the reference vector per block (delta^T C^-1 delta, uncut and
    under each mask), positive definiteness and condition numbers of the
    covariance restricted to each mask, and the signal-to-noise per block
    after the cuts (compared with the Y3 numbers of 2503.13632). A layout
    mismatch, or a covariance that is not positive definite under either
    combination mask, stops the script.
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
# environment wins. The workers get --threads for all of them. The
# variables are read by OpenMP (the C code's threads) and by the BLAS
# libraries (OpenBLAS, Apple's vecLib, MKL) and numexpr.
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
               "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
# the cap of this process and the default of --threads; the reason for
# the value 3 is not recorded
DEFAULT_THREADS = 3
# setdefault writes a variable only when the environment does not have it
for _var in THREAD_VARS:
    os.environ.setdefault(_var, str(DEFAULT_THREADS))

# numpy is imported after the caps above on purpose; "noqa: E402" tells
# the flake8 style checker that this late import is deliberate
import numpy as np                                                   # noqa: E402

# Folders, from this script's own path (__file__): scripts/ (HERE), the
# project (PRJ), its data/, the Python reference (tests/reference) and
# the likelihood folder. REFDIR goes first on the module search path
# (sys.path) so that `from ref_covariance_full import ...` and the other
# reference imports below find those modules.
HERE = os.path.dirname(os.path.abspath(__file__))
PRJ = os.path.dirname(HERE)
DATA = os.path.join(PRJ, "data")
REFDIR = os.path.join(PRJ, "tests", "reference")
LIKDIR = os.path.join(PRJ, "likelihood")
if REFDIR not in sys.path:
    sys.path.insert(0, REFDIR)

# File names inside data/: the base dataset descriptor, the two
# scale-cut masks and the two combination descriptors (keyed by the
# combination name used in des_cluster.combo_<name>), the data vector
# and the packed covariance this script writes
DATASET = "des_cluster_y6.dataset"
MASKS = {"4x2pt_N": "des_cluster_y6_4x2ptN.mask", "6x2pt_N": "des_cluster_y6_6x2ptN.mask"}
COMBO_DATASETS = {"4x2pt_N": "des_cluster_y6_4x2ptN.dataset",
                  "6x2pt_N": "des_cluster_y6_6x2ptN.dataset"}
DATAVECTOR_FILE = "des_cluster_y6.datavector"
COV_FILE = "des_cluster_y6_cov.npy"
# block order of the joint data vector (README table; the C block starts)
BLOCKS = ("ss", "gs", "gg", "cg", "N", "cc", "cs")
# chi2 bound of step (d): the data vector is the model at the fiducial
# point, written with 17 significant digits, so a likelihood that
# evaluates the same model there returns chi2 = 0 up to round-off. 1e-6 is
# that zero test: a value above it means the likelihood evaluated a
# different model, point or data file than the one written in (a).
CHI2_TOL = 1e-6

# ----------------------------------------------------------------------------
# Table I of arXiv 2503.13631 (fiducial = prior mean where the prior is Gaussian)
# ----------------------------------------------------------------------------
# Omega_nu h^2 = mnu (NNU/3)^0.75 / NEUTRINO_MASS_FAC (mnu in eV), the
# relation of the omegach2 lambda of EXAMPLE_EVALUATE2.yaml; NNU = 3.046
# is the effective number of neutrino species. Table I gives
# Omega_nu h^2 = 0.00083, so mnu = 0.0772 eV. tau is the value fixed in
# EXAMPLE_EVALUATE2.yaml.
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

# DES Y3 cluster analysis (arXiv 2503.13632): S/N per block after cuts and
# the number of points per block after cuts; print_report shows them next
# to the 4x2pt + N S/N of the synthetic data
Y3_SN = {"N": 94.5, "cs": 31.8, "cc": 18.8, "cg": 39.6, "gg": 52.5}
Y3_NPOINTS = {"N": 12, "cs": 404, "cc": 149, "cg": 124, "gg": 31}


def log(msg):
    """Print one progress line with the script's name as prefix.

    flush=True writes the line out at once instead of leaving it in the
    output buffer, so the progress shows while the run goes on, also when
    the output goes to a file or a pipe (as a worker's output does).

    Arguments:
      msg = the message.

    Returns:
      nothing.
    """
    print(f"[make_synthetic_data] {msg}", flush=True)


# ============================================================================
# configuration: dataset and likelihood yaml
# ============================================================================
def load_yaml(path):
    """A yaml file, ignoring cobaya tags (!defaults).

    A YAML tag is a word starting with "!" in front of a value; cobaya's
    !defaults pulls parameter files into a likelihood yaml (e.g. the
    params line of likelihood/combo_6x2pt_N.yaml). The plain PyYAML
    loader rejects unknown tags, so every tagged value is read as None:
    the script needs the plain keys of these files, not the merged
    parameter blocks.

    Arguments:
      path = the yaml file.

    Returns:
      the parsed content, a dict for these files (tagged values None).
    """
    import yaml

    class Loader(yaml.SafeLoader):
        """SafeLoader subclass: the tag rule below is registered on this
        class only, so yaml.SafeLoader itself is left unchanged."""
        pass

    # the anonymous function (lambda) receives the loader, the rest of the
    # tag and the YAML node, and returns None for every "!..." tag
    Loader.add_multi_constructor("!", lambda loader, suffix, node: None)
    # the with block closes the file when it ends, also after an error
    with open(path) as f:
        return yaml.load(f, Loader=Loader)


def load_dataset(path=os.path.join(DATA, DATASET)):
    """The keys of a .dataset descriptor that this script uses.

    The descriptor is read with getdist's IniFile, the reader of the
    `key = value` format that cobaya likelihoods use too; relativeFileName
    returns a file name joined to the descriptor's own folder.

    Arguments:
      path = the descriptor (default data/des_cluster_y6.dataset).

    Returns:
      dict with the n(z) and <phi_i|z> file paths (nz_lens_file,
      nz_source_file, nz_cluster_file), the numbers of lens, source,
      angular and cluster redshift bins, the angular range in arcmin,
      the cluster z_lambda and richness edges (lists of floats), the
      survey area in deg^2, and cg_lens_bins (list of ints: the lens bin
      paired with each cluster bin in w_cg).
    """
    from getdist import IniFile
    ini = IniFile(path)

    def lst(key, tp):
        """Read a comma- or space-separated value of the descriptor.

        Arguments:
          key = the descriptor key, e.g. "cluster_zbin_edges".
          tp = the conversion of each item, float or int.

        Returns:
          list of tp values.
        """
        # commas become spaces, split() cuts at every run of whitespace,
        # and the list comprehension converts each piece with tp
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
    (every lens bin: the covariance needs the six of CL+3x2pt).

    The reference (tests/reference/reference_cluster.py) takes its
    <phi_i|z> from the dataset's kernel table (photoz="table"), the n(z)
    files, the binning and the area from the dataset, and lmax, the
    cluster kernel mode, the magnification coefficient C_c and the
    mass-function amplitude mode from the likelihood yaml. Every other
    setting keeps the reference default (DEFAULT_SETTINGS).

    Arguments:
      ds = the dict of load_dataset.
      lik = the parsed likelihood yaml (load_yaml of
        likelihood/combo_6x2pt_N.yaml).

    Returns:
      dict of ClusterReference settings.
    """
    return dict(
        photoz="table", phi_table_file=ds["nz_cluster_file"],
        source_nz=ds["nz_source_file"], lens_nz=ds["nz_lens_file"],
        lmax=int(lik["lmax"]), area_deg2=ds["survey_area_deg2"],
        zc_edges=ds["cluster_zbin_edges"], lambda_edges=ds["richness_edges"],
        ntheta=ds["n_theta"], tmin_arcmin=ds["theta_min_arcmin"],
        tmax_arcmin=ds["theta_max_arcmin"], lens_bins=list(range(ds["lens_ntomo"])),
        kernel_mode=int(lik["cluster_kernel_mode"]), C_c=float(lik["cluster_magnification"]),
        # Halos always use P_cb(k,z) and rho_cb, as in the likelihood.
        # The reference integrates the variance independently at each z.
        hmf_matter="cb",
        hmf_alpha_mode=int(lik["cluster_hmf_alpha_mode"]))


# ============================================================================
# cobaya workers (run in their own subprocess)
# ============================================================================
def cobaya_info(combo, data_path, data_file):
    """cobaya input: the combo likelihood on a dataset, CAMB as in
    EXAMPLE_EVALUATE2.yaml, the cosmology fixed at Table I.

    The returned dict is what a cobaya YAML file holds, built in Python.
    The cosmological parameters are fixed values; the lambda strings
    (formulas cobaya evaluates) derive As, omegabh2 and omegach2 as in
    EXAMPLE_EVALUATE2.yaml, and "drop": True keeps As_1e9, omegab and
    omegam from being passed on as inputs of CAMB or the likelihood (they
    only feed the lambdas). sigma8 and omegan2 have no value and become
    derived outputs. The nuisance parameters come from the
    likelihood defaults (likelihood/params_*.yaml) and are set by
    build_point.

    Arguments:
      combo = "4x2pt_N" or "6x2pt_N" (the likelihood
        des_cluster.combo_<combo>).
      data_path = folder of the dataset descriptor.
      data_file = descriptor file name inside data_path.

    Returns:
      dict, the input of cobaya.model.get_model.
    """
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
        # dict(...) is a new dict with the example's camb settings
        "theory": {"camb": dict(ex["theory"]["camb"])},
        "params": params,
        "stop_at_error": True,
        "debug": False,
    }


def build_point(model):
    """Sampled parameters at the Table I fiducial; every one must be known.

    The cosmology of cobaya_info is fixed, so the sampled parameters are
    the nuisance parameters of the likelihood defaults. Each must have a
    value in FIDUCIAL_NUISANCE; the parameters of FIXED_EXPECTED must be
    fixed in the model; and every fixed parameter listed in either table
    must sit at its Table I value (to 1e-12).

    Arguments:
      model = the cobaya model of get_model(cobaya_info(...)).

    Returns:
      dict {sampled parameter name: Table I value}.

    Raises:
      RuntimeError naming the parameters that have no Table I value,
      that are expected fixed but are not, or that are fixed at another
      value.
    """
    sampled = list(model.parameterization.sampled_params())
    # comprehension with a condition: the sampled names without a value
    missing = [p for p in sampled if p not in FIDUCIAL_NUISANCE]
    if missing:
        raise RuntimeError(f"no Table I fiducial for sampled parameters {missing}")
    const = model.parameterization.constant_params()
    missing = [p for p in FIXED_EXPECTED if p not in const]
    if missing:
        raise RuntimeError(f"expected fixed parameters {missing} not in the model")
    # the combo's fixed parameters (e.g. CL+GC fixes lens bins 4-6) and the
    # yaml constants must sit at Table I too; the loop runs over the
    # (name, value) pairs of both tables, joined into one list
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

    Arguments:
      combo = "4x2pt_N" or "6x2pt_N".
      data_path = folder of the dataset descriptor.
      data_file = descriptor file name inside data_path.

    Returns:
      (model, lik, point, chi2, dv, derived): the cobaya model, the
      likelihood object, the point of build_point, chi2 = -2 log L (the
      likelihood's chi2 at the point), the masked model vector dv (numpy
      float array in the full joint layout, zeros at masked entries), and
      the derived parameters {name: value} (sigma8, ...).

    Side effects:
      changes the working directory of the process to ROOTDIR, the
      cocoa/Cocoa folder, from which relative paths such as the camb path
      ./external_modules/code/CAMB of EXAMPLE_EVALUATE2.yaml resolve.
    """
    from cobaya.model import get_model
    os.chdir(os.environ["ROOTDIR"])
    model = get_model(cobaya_info(combo, data_path, data_file))
    lik = model.likelihood[f"des_cluster.combo_{combo}"]
    point = build_point(model)
    post = model.logposterior(point, cached=False)
    chi2 = -2.0 * float(post.loglikes[0])
    # zip pairs the derived-parameter names with their values; dict(...)
    # turns the pairs into {name: value}
    derived = dict(zip(model.parameterization.derived_params(), post.derived))
    # to_input resolves every input parameter at this point (sampled,
    # fixed, and those computed from them); the likelihood takes its own
    # subset, lik.input_params, as logp does
    all_inputs = model.parameterization.to_input(point)
    lik_inputs = {}
    for name in lik.input_params:
        lik_inputs[name] = all_inputs[name]
    # **lik_inputs passes each dict entry as a keyword argument name=value
    dv = np.array(lik.get_datavector(**lik_inputs), dtype=float)
    return model, lik, point, chi2, dv, derived


def write_placeholder_dataset(tmp):
    """A copy of data/des_cluster_y6.dataset in the folder tmp with an
    all-ones mask, a zero data vector and an identity covariance (the
    likelihood reads all three at initialization); the other files point
    back to data/. Returns the dataset file name (relative to tmp).

    The length of the three placeholder files is the size of the joint
    layout of ref_covariance_full.joint_layout for the dataset's binning
    (2812). The identity covariance lists only its diagonal (i, i, 1.0):
    the three-column text format IPCluster::set_inv_cov reads, missing
    entries being zero. Every other *_file key of the copy is rewritten as
    an absolute path into data/, because a relative name would be looked
    up next to the copy, in tmp.

    Arguments:
      tmp = an existing folder that receives the four files.

    Returns:
      "ones.dataset", the name of the descriptor written in tmp.

    Side effects:
      writes ones.mask, zeros.datavector, identity.cov and ones.dataset
      in tmp.
    """
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
        # the key of a "key = value" line; "" for a comment line (the
        # conditional expression) and for a line without "="
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
    """(a): the all-ones-mask model vector and the layout seen by the C code.

    Runs inside a worker subprocess (main with --worker datavector). After
    evaluate, the compiled interface cosmolike_des_cluster_interface (the
    Python module built from interface/interface.cpp) holds the state the
    likelihood just set, and its functions return the C block sizes and
    starts, the Y transform matrix T (ntheta x ntheta), the selection
    factor B (cluster z bins x ntheta) and the mask IPCluster applies (the
    all-ones file with the last theta bin of every cs row set to 0).

    Arguments:
      out = path of the .npz archive to write (numpy's file of named
        arrays): dv, sizes, starts, T, B, mask, and the derived parameters
        and the point as JSON strings.

    Returns:
      nothing.

    Side effects:
      writes out; creates a temporary folder for the placeholder dataset
      and removes it in the finally block, also after an error.
    """
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
    """(d): chi2 of a combo at the fiducial against the written files.

    Runs inside a worker subprocess (main with --worker chi2) and loads
    the shipped descriptor data/des_cluster_y6_<combo>.dataset, so the
    likelihood reads the data vector and covariance just written.

    Arguments:
      combo = "4x2pt_N" or "6x2pt_N".
      out = path of the JSON file to write: {"combo", "chi2", "ndata"},
        ndata being the length of the full joint vector.

    Returns:
      nothing.
    """
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
    """Run this file in a worker subprocess and relay its output.

    A worker is a fresh python process (sys.executable is this same
    interpreter) running this script with the given --worker options; it
    builds its own cobaya model, so the global state cosmolike keeps per
    process never mixes two configurations. The call blocks until the
    worker exits. Its captured output is printed here; when it fails with
    one of the SYMLINK_RACE messages it is started again after 5 s.

    Arguments:
      args = the worker's command-line options, e.g. ["--worker",
        "datavector", "--out", path].
      threads = value of every THREAD_VARS variable in the worker (the
        OpenMP threads of cosmolike and the BLAS threads).
      retries = the number of extra attempts after a symlink-race failure.

    Returns:
      nothing, when the worker exits with code 0.

    Raises:
      RuntimeError naming the arguments and the exit code when the
      worker fails for another reason, or still fails after the retries.
    """
    # dict(os.environ, **{...}) is a copy of the environment with every
    # thread variable set to threads (the dict comprehension builds those
    # entries); only the worker sees it
    env = dict(os.environ, **{var: str(threads) for var in THREAD_VARS})
    cmd = [sys.executable, os.path.abspath(__file__)] + args
    for attempt in range(retries + 1):
        # subprocess.run blocks until the worker exits; capture_output and
        # text collect its stdout and stderr as strings
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        sys.stdout.write(r.stdout)
        sys.stderr.write(r.stderr)
        if r.returncode == 0:
            return
        # generator inside any: True when one of the race messages appears
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
    """Cache key of the reference covariance: a hash of every input.

    SHA-256 (a cryptographic hash: any change of the input bytes changes
    the digest) over the settings written as JSON with sorted keys, the
    selection mode, the bytes of every tests/reference/*.py file and the
    bytes of the three input tables (<phi_i|z>, source and lens n(z)).
    Any edit of the reference code, even of a comment, or of an input
    therefore gives a new key, and the next run recomputes the covariance
    instead of reusing a stale cache file.

    Arguments:
      settings = dict of the reference settings plus the extra entries
        compute_covariance adds (cg_lens_bins, num_massive_nu, counts);
        values that JSON cannot write are written as str(value).
      selection_mode = the --selection-mode string.

    Returns:
      the first 16 hexadecimal characters of the digest, a string.
    """
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
    (EXAMPLE_EVALUATE2.yaml camb extra_args; CAMB's default is 1).

    Returns:
      int, the num_massive_neutrinos of the example's camb extra_args, or
      1 when the key is absent (dict.get with a default).
    """
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
    by 0.3% and the counts by up to 2.6% at the same Omega_nu h^2).

    The reference runs at its own Table I values (reference_cluster.FIDUCIAL),
    a second copy of the values of FIDUCIAL_COSMO and FIDUCIAL_NUISANCE
    above; a difference between the two copies shows only in the
    C-versus-reference comparison of step (c).
    With cache_dir, the result is stored in synthetic_cov_<key>.npz (key
    from source_hash) and read back on the next run with the same inputs.

    Arguments:
      settings = the reference settings of reference_settings.
      cg_lens_bins = the lens bin paired with each cluster bin in w_cg.
      selection_mode = "signal", "jacobian" or "none" (ref_covariance_full).
      cache_dir = folder of the cache files, or None for no cache.
      counts = numpy array (cluster z bins, richness bins) of N_iA, or None
        for the reference counts.

    Returns:
      dict: cov (ndata, ndata) covariance in the joint layout; counts (the
      N_iA used) and counts_ref (the reference's), each (z bins, richness
      bins); S, the sample-variance factors of the counts; ref_dv (ndata,)
      the reference model vector; B (z bins, ntheta) and T (ntheta,
      ntheta) of the reference; edges, the theta bin edges in radians;
      sigma8 of the reference CAMB run.

    Side effects:
      runs CAMB and the reference model unless the cache holds the
      result; writes the cache file when cache_dir is given.
    """
    from reference_cluster import ClusterReference, COSMO_KEYS, FIDUCIAL
    from ref_cosmology import Cosmology
    from ref_covariance_full import full_covariance
    nmassive = camb_num_massive_neutrinos()
    # dict(settings, name=value, ...) is a copy of settings with three
    # entries added; the counts enter rounded to 8 decimals, as a list
    key = source_hash(dict(settings, cg_lens_bins=list(cg_lens_bins), num_massive_nu=nmassive,
                           counts=None if counts is None else
                           np.asarray(counts, dtype=float).round(8).tolist()),
                      selection_mode)
    cache = os.path.join(cache_dir, f"synthetic_cov_{key}.npz") if cache_dir else None
    if cache and os.path.exists(cache):
        log(f"covariance from cache {cache}")
        z = np.load(cache, allow_pickle=True)
        # dict comprehension: every array of the archive, by its name
        return {k: z[k] for k in z.files}
    t0 = time.time()
    from reference_cluster import DEFAULT_SETTINGS
    # the reference defaults, overridden by the given settings (the **
    # form passes the settings dict as keyword arguments)
    s = dict(DEFAULT_SETTINGS, **settings)
    # the cosmological subset of the reference's Table I values
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
    ClusterReference.data_vector).

    For ss, gs and gg the Limber spectra are interpolated onto every
    integer multipole (cl_on_integers) and projected with the bin-averaged
    full-sky kernels K (xi+, xi-, gamma_t, w); ss and gs are then
    multiplied by the shear calibrations (1 + m) of their source legs. The
    cluster blocks come in the reference's order
    N, cs, cc, cg and are copied block by block into the joint layout.
    The reference pairs cluster bin i with lens bin i in w_cg, which is
    the dataset's cg_lens_bins = 0, 1, 2; the size check below catches a
    different number of entries, not a different pairing.

    Arguments:
      ref = a reference_cluster.ClusterReference at the fiducial point.
      lay = the joint layout dict of ref_covariance_full.joint_layout.

    Returns:
      numpy array (ndata,), the reference data vector.

    Raises:
      RuntimeError when a reference cluster block has a size other than
      that of the joint layout.
    """
    from ref_covariance_full import kernels
    from ref_projection import cl_on_integers
    sp = ref.spectra()
    s = ref.settings
    K = kernels(ref.edges, s["lmax"])
    m = np.asarray(ref.params["shear_m"])
    dv = np.zeros(lay["ndata"])
    # one entry of lay["obs"] per two-point row: (block, field legs, kernel
    # kind, bin labels); lay["index"][o] holds the joint positions of row o,
    # and indexing dv with it writes the ntheta values there. K[kind] @ cl
    # is the matrix-vector product (ntheta, lmax) x (lmax,). The labels
    # unpack into bin indices: (kind, i, j) for ss ("_" discards the kind),
    # (k, j) for gs and the one-element tuple (k,) for gg.
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
        # the block that follows b in the reference vector, whose start
        # ends b
        end = {"N": "cs", "cs": "cc", "cc": "cg", "cg": "end"}[b]
        seg = cdv[st[b]:st[end]]
        if seg.size != lay["sizes"][b]:
            raise RuntimeError(f"reference block {b}: {seg.size} != {lay['sizes'][b]}")
        dv[lay["starts"][b]:lay["starts"][b] + seg.size] = seg
    return dv


def write_datavector(path, dv):
    """Write the data vector as "index value" lines.

    %.16e prints 17 significant digits, enough to read every double back
    exactly, so the likelihood sees the computed values bit for bit.

    Arguments:
      path = output file; an existing file is overwritten.
      dv = numpy array (ndata,).

    Returns:
      nothing.
    """
    np.savetxt(path, np.column_stack([np.arange(dv.size), dv]), fmt="%d %.16e")


def write_covariance(path, cov):
    """The packed upper triangle (row by row, diagonal included) as a 1-D
    little-endian float64 .npy array: the binary layout cosmolike's
    IPCluster::set_inv_cov reads (read_npy_packed_upper_cov).

    np.triu_indices(n) gives the (row, column) index arrays of the upper
    triangle with the diagonal, row by row (row i holds columns i .. n-1),
    so cov[iu] is the packed 1-D array of n(n + 1)/2 entries. "<f8" is
    little-endian float64, the dtype the C reader requires.

    Arguments:
      path = output file (.npy); an existing file is overwritten.
      cov = numpy array (n, n), a symmetric covariance.

    Returns:
      nothing.
    """
    iu = np.triu_indices(cov.shape[0])
    np.save(path, np.ascontiguousarray(cov[iu], dtype="<f8"))


def read_covariance(path, n):
    """The full n x n covariance from the packed .npy file of
    write_covariance, or from a text table (i, j, cov), the three-column
    text format IPCluster::set_inv_cov also reads (parsed with numpy, which
    rounds every value correctly, as the C reader does).

    The format is recognized by the first six bytes of the file, the .npy
    magic string, as in the C reader. Every stored entry is mirrored, so
    the result is symmetric.

    Arguments:
      path = the covariance file.
      n = the size of the joint vector.

    Returns:
      numpy array (n, n).

    Raises:
      ValueError when an .npy file does not hold n(n + 1)/2 entries.
    """
    cov = np.zeros((n, n))
    # "rb" opens the file as bytes; b"..." is a bytes literal
    with open(path, "rb") as f:
        is_npy = f.read(6) == b"\x93NUMPY"
    if is_npy:
        packed = np.load(path)
        iu = np.triu_indices(n)
        if packed.shape != (iu[0].size,):
            raise ValueError(f"{path}: {packed.shape} entries, the packed upper "
                             f"triangle of {n} x {n} has {iu[0].size}")
        cov[iu] = packed
        # cov.T is a view of the same memory (no copy): writing the upper
        # triangle of the transpose fills the lower triangle of cov
        cov.T[iu] = packed
        return cov
    t = np.loadtxt(path)
    i, j = t[:, 0].astype(int), t[:, 1].astype(int)
    cov[i, j] = t[:, 2]
    cov[j, i] = t[:, 2]
    return cov


# ============================================================================
# (c) checks
# ============================================================================
def effective_mask(mask, lay):
    """The mask as IPCluster applies it: the last theta bin of every cs row
    is always dropped (the last row of T vanishes).

    That entry is Y(R_max) = Sigma(R_max) - Sigma(R_max) = 0 in the model,
    the data and the covariance: a zero-variance entry by construction,
    which IPCluster::set_mask removes whatever the mask file says.

    Arguments:
      mask = the mask (ndata,), 0/1 or booleans.
      lay = the joint layout dict: lay["obs"] names the block of each
        two-point row and lay["index"] (rows, ntheta) its joint positions.

    Returns:
      numpy bool array (ndata,), a new array; mask is not modified.
    """
    # asarray can return the input itself when it is already a bool array;
    # .copy() keeps the caller's mask unchanged
    m = np.asarray(mask, dtype=bool).copy()
    nt = lay["index"].shape[1]
    for o, ob in enumerate(lay["obs"]):
        if ob[0] == "cs":
            m[lay["index"][o][nt - 1]] = False
    return m


def mask_report(cov, dv, mask, lay, label):
    """PD, condition numbers and S/N per block under one mask.

    The covariance C restricted to the kept entries is positive definite
    (PD) when its Cholesky factorization C = L L^T exists and the smallest
    eigenvalue of the correlation matrix C_ij/sqrt(C_ii C_jj) is positive.
    The signal-to-noise S/N = sqrt(d^T C^-1 d) of the model vector d is
    given in total and per block, each block with its own sub-covariance
    (so the correlations with the other blocks do not enter).

    Arguments:
      cov = numpy array (ndata, ndata), the covariance.
      dv = numpy array (ndata,), the model data vector.
      mask = numpy bool array (ndata,), True for a kept entry.
      lay = the joint layout dict (block starts and sizes).
      label = the name of the mask, stored in the report.

    Returns:
      dict: label, npoints, pd, cholesky, corr_eig_min and corr_eig_max,
      cond_corr (largest over smallest correlation eigenvalue, inf when
      the smallest is not positive), raw_eig_min, raw_eig_max and
      raw_eig_nonpositive (of C itself, round-off included),
      variance_range, SN_total, and blocks {block: {npoints, SN}} for the
      blocks with kept entries.
    """
    # np.where(mask)[0]: the positions of the kept entries
    idx = np.where(mask)[0]
    # np.ix_ makes the two index lists an open mesh, so the indexing takes
    # the (kept rows) x (kept columns) submatrix, not single entries
    C = cov[np.ix_(idx, idx)]
    d = np.sqrt(np.diag(C))
    # np.outer(d, d)[i, j] = d_i d_j: corr is the correlation matrix
    corr = C / np.outer(d, d)
    evc = np.linalg.eigvalsh(corr)
    # np.linalg.cholesky raises LinAlgError when C is not positive definite
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
        # sel marks the entries of block b; sel & mask its kept entries
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
    """(d1 - d2)^T C^-1 (d1 - d2) per block and in total under a mask.

    Each block uses its own sub-covariance; "total" uses every kept entry
    and the full covariance between them. np.linalg.solve solves
    C x = d1 - d2 instead of forming C^-1.

    Arguments:
      cov = numpy array (ndata, ndata).
      d1, d2 = numpy arrays (ndata,), the two data vectors.
      mask = numpy bool array (ndata,), True for a kept entry.
      lay = the joint layout dict (block starts and sizes).

    Returns:
      dict {block name or "total": delta chi2}; blocks without kept
      entries are left out.
    """
    out = {}
    delta = d1 - d2
    # tuple concatenation: the seven blocks, then "total"
    for b in BLOCKS + ("total",):
        if b == "total":
            ib = np.where(mask)[0]
        else:
            sel = np.zeros(mask.size, dtype=bool)
            sel[lay["starts"][b]:lay["starts"][b] + lay["sizes"][b]] = True
            ib = np.where(sel & mask)[0]
        # a size of 0 counts as false: blocks without kept entries are skipped
        if ib.size:
            out[b] = float(delta[ib] @ np.linalg.solve(cov[np.ix_(ib, ib)], delta[ib]))
    return out


# ============================================================================
def main(argv=None):
    """Run the steps (a) to (d), or one worker task, and return the report.

    With --worker the process is a worker started by run_worker: it runs
    worker_datavector or worker_chi2 and returns. Otherwise it reads the
    dataset and the 6x2pt + N likelihood defaults, builds the joint layout
    of the reference, and runs (a) to (d) of the module docstring with a
    temporary folder for the worker outputs. The checks of (c) use the
    covariance as read back from the file, the matrix the likelihood
    reads.

    Arguments:
      argv = list of command-line options, or None for sys.argv (the
        call from the command line).

    Returns:
      the report dict (also written as JSON with --json); None in a
      worker.

    Raises:
      SystemExit when ROOTDIR is not set (start_cocoa.sh not sourced);
      RuntimeError when the C vector or the C layout disagrees with the
      reference layout, or a worker fails; AssertionError when the
      covariance is not positive definite under a combination mask, or
      when chi2 at the fiducial is not below CHI2_TOL.

    Side effects:
      writes data/des_cluster_y6.datavector (unless --skip-datavector)
      and data/des_cluster_y6_cov.npy (unless --skip-covariance), the
      cache file with --cache-dir and the JSON report with --json.
    """
    # the first line of the module docstring is the --help summary
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--threads", type=int, default=DEFAULT_THREADS,
                    help="OpenMP and BLAS threads of the cobaya workers")
    ap.add_argument("--cache-dir", default=None, help="cache of the reference covariance")
    ap.add_argument("--selection-mode", default="signal",
                    choices=("signal", "jacobian", "none"))
    ap.add_argument("--skip-datavector", action="store_true",
                    help="reuse data/des_cluster_y6.datavector")
    ap.add_argument("--skip-covariance", action="store_true",
                    help="do not rewrite data/des_cluster_y6_cov.npy (the checks read the file)")
    ap.add_argument("--skip-chi2", action="store_true")
    ap.add_argument("--reference-counts", action="store_true",
                    help="counts of the Python reference in the covariance (default: "
                         "the counts of the synthetic data vector)")
    ap.add_argument("--json", default=None, help="write the report here")
    # options of the worker processes only; argparse.SUPPRESS hides them
    # from --help
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

    # every check result goes into this dict, the report
    report = {}
    ds = load_dataset()
    lik = load_yaml(os.path.join(LIKDIR, "combo_6x2pt_N.yaml"))
    from ref_covariance_full import joint_layout
    lay = joint_layout(ds["cluster_ntomo"], len(ds["richness_edges"]) - 1,
                       ds["source_ntomo"], ds["lens_ntomo"], ds["n_theta"], ds["cg_lens_bins"])
    report["layout"] = dict(sizes=lay["sizes"], starts=lay["starts"], ndata=lay["ndata"])
    # a new temporary folder for the worker outputs
    work = tempfile.mkdtemp(prefix="des_cluster_synth_main_")
    dv_path = os.path.join(DATA, DATAVECTOR_FILE)
    cov_path = os.path.join(DATA, COV_FILE)
    # the finally block at the end runs whether the steps succeed or
    # raise: it removes the folder and writes the report collected so far
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
            # {block: size} and {block: start} of the C code, the lists
            # paired with the block names by zip
            sizes_c = dict(zip(BLOCKS, z["sizes"].tolist()))
            starts_c = dict(zip(BLOCKS, z["starts"].tolist()))
            report["layout"]["compiled_sizes_match"] = sizes_c == lay["sizes"]
            report["layout"]["compiled_starts_match"] = starts_c == lay["starts"]
            if not (sizes_c == lay["sizes"] and starts_c == lay["starts"]):
                raise RuntimeError(f"layout mismatch: C {sizes_c} {starts_c}, "
                                   f"reference {lay['sizes']} {lay['starts']}")
            report["mask_ones_unmasked"] = int(z["mask"].sum())
            # the archive holds the JSON text as a numpy string; str() gives
            # the text back and json.loads the dict
            report["derived"] = json.loads(str(z["derived"]))
            T_c, B_c = z["T"], z["B"]
        else:
            dv = np.loadtxt(dv_path)[:, 1]
            T_c = B_c = None

        # --- (b) covariance ---
        settings = reference_settings(ds, lik)
        # the positions of the counts block N, as a slice object; reshaped
        # to (cluster z bins, richness bins), the block's own order
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
        # round trip: the largest |file - computed| relative to
        # sqrt(C_ii C_jj), entries with a zero scale skipped. The file holds
        # the upper triangle exactly, so a file written in this run gives 0
        # when cov is exactly symmetric (cov_symmetric_max below).
        scale = np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
        ok = scale > 0
        report["cov_file_roundtrip_max_rel"] = float(
            np.max(np.abs(cov_file - cov)[ok] / scale[ok]))
        report["cov_file_MiB"] = os.path.getsize(cov_path) / 2**20
        report["cov_symmetric_max"] = float(np.max(np.abs(cov - cov.T)))
        # every check below uses the covariance as stored in the file
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
        # column 4 of the Y3 table holds N, in the order of the N block
        y3 = np.loadtxt(os.path.join(DATA, "y3_redmapper_counts.txt"))[:, 4]
        report["counts_C_over_Y3"] = (dv[iN] / y3).tolist()
        # {label: bool array}: no cut, then the two combination masks
        # (column 1 of each file; > 0.5 turns 0.0 and 1.0 into booleans)
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
            # dict comprehension with a condition: the combinations whose
            # chi2 is not below CHI2_TOL ("not v < CHI2_TOL" also catches
            # a NaN, for which every comparison is false)
            bad = {k: v for k, v in report["chi2"].items() if not v < CHI2_TOL}
            if bad:
                raise AssertionError(f"chi2 at the fiducial above {CHI2_TOL}: {bad}")
    finally:
        shutil.rmtree(work, ignore_errors=True)
        if a.json:
            # default=float converts the numpy numbers JSON cannot write
            with open(a.json, "w") as f:
                json.dump(report, f, indent=1, default=float)
    return report


def print_report(r):
    """Print the report of main in readable form.

    The layout, the scalar checks, the derived parameters, and per mask
    the positive-definiteness diagnostics, the S/N in total and per block
    (for the 4x2pt + N mask next to the DES Y3 numbers), and the C model
    versus reference delta chi2 per block.

    Arguments:
      r = the report dict of main.

    Returns:
      nothing.
    """
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
            # = (S/N)^2 + n_points, so sqrt(S/N^2 - n) is its noise-free analog.
            # The conditional expression adds the Y3 note only for the
            # 4x2pt + N blocks that Y3 lists.
            y3 = (f" (Y3: {Y3_NPOINTS[b]} points, S/N {Y3_SN[b]}, noise-debiased "
                  f"{np.sqrt(Y3_SN[b]**2 - Y3_NPOINTS[b]):.1f})"
                  if label == "4x2pt_N" and b in Y3_SN else "")
            log(f"   {b:>2}: {v['npoints']:4d} points, S/N {v['SN']:7.1f}{y3}")
        # generator inside join: "block value" pieces separated by commas
        log(f"   C model vs reference, delta^T C^-1 delta: "
            + ", ".join(f"{b} {v:.3g}" for b, v in r["C_vs_reference_chi2"][label].items()))


if __name__ == "__main__":
    main()
