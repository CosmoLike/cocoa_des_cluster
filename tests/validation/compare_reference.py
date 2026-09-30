#!/usr/bin/env python
"""Validation of the C cluster port (cosmolike_core *_cluster files and the
des_cluster bindings) against the independent Python reference
(tests/reference), milestones M0/M1 of PORT_PLAN.md section 5.

Run from Cocoa/ in the cocoa environment (start_cocoa.sh sourced):

    python projects/des_cluster/tests/validation/compare_reference.py \
        [--cache-dir DIR] [--recompute-reference] [--threads 4] \
        [--hmf-alpha-mode {0,1}] [--skip-production] [--skip-diagnostics] \
        [--skip-determinism] [--skip-lighthouse] [--json OUT.json]

What it does
------------
1. REFERENCE (cached in --cache-dir, keyed on the configuration and the
   reference sources): the Python reference at the Table I fiducial of
   2503.13631 (reference_cluster.FIDUCIAL; CAMB with Omega_nu h^2 =
   0.00083, three degenerate massive neutrinos), its SETTINGS (not its
   physics) aligned to the likelihood:
     - cluster kernel <phi_i|z> = numpy.interp of data/des_y6_cluster.nz
       (photoz = "table"; every table node a quadrature break),
     - lens/source n(z) = the project files of the dataset,
     - LMAX = the lmax of likelihood/combo_4x2pt_N.yaml,
     - binning, richness and z edges, area, cg lens bins = the dataset,
     - Tinker amplitude mode (hmf_alpha_mode) = cluster_hmf_alpha_mode of
       the yaml (0 = alpha 0.368, DES; 1 = halo.c's alpha(z)), or the
       --hmf-alpha-mode override; the C side gets the same mode.
   It exports every quantity compared below, its Gaussian covariance
   (N, Sigma = T gamma_t, w_cc, w_cg) and the cosmology tables in the
   interface units of the likelihood (log10 k in h/Mpc, ln P in
   (Mpc/h)^3, chi in Mpc/h, G = D (1+z) on the dense 1D z grid z_G).
   Three variants:
     matched     THE VALIDATION. The C side gets the reference's own CAMB
                 z nodes and a dense uniform log10 k grid, and the
                 reference reads ln P_NL linearly in z between those
                 nodes, as cosmolike's p_nonlin does: both sides see the
                 same P(k, z) (CAMB Halofit finds k_NL by bisection to
                 |sigma - 1| <= 1e-3, so ln P_NL has ~1e-3 node-to-node
                 noise in z that any two z samplings resolve differently).
     production  The C side gets the tables on the likelihood grids
                 (z_interp_2D, log10k_interp_2D, 140 x 1500 nodes at
                 accuracyboost 1), the reference a cubic spline in z: the
                 whole-pipeline number (inputs included).
     diagnostic  = matched, with the source n(z) of the reference's
                 lensing efficiency cut at the file's last edge (z = 3.0),
                 as the core's g_tomo does (redshift_spline.c integrates
                 from a = 1/(1 + zmax_all), ignoring positive photo-z
                 shifts): isolates the cluster code in C_cs.
2. C PIPELINE through cosmolike_des_cluster_interface WITHOUT cobaya: the
   init chain of likelihood/_cosmolike_prototype_base.py (probe 4x2pt_N),
   then ci.set_cosmology with the tables of step 1 and the nuisance
   parameters at the same fiducial.
3. COMPARISON TABLE (max relative difference, max |diff|/max|signal| per
   row, and where): phi_cluster, nz_cluster, g_cluster; ncl_richness,
   bcl_richness at the a nodes of the cluster tables (and between them);
   pcm_1h_richness for k in [1e-3, 200] h/Mpc; N_cluster_tomo; the Limber
   C_cs, C_cc, C_cg (array wrappers, and the cached tables for l >=
   LMIN_tab) at l = 2 .. 5e4; gamma_t, Sigma = T gamma_t, w_cc, w_cg
   (Limber) at every theta bin; the selection factor and T; and the
   cluster blocks of the joint data vector, scored with delta^T C^-1 delta
   per block under the Y6 scale cuts AND with no cuts (the most
   aggressive positive-definite mask when the uncut covariance is not;
   cosmolike-dev SKILL.md "Accuracy tests must see the small scales").
4. DETERMINISM (fresh subprocesses): the uncut cluster data vector at
   OMP_NUM_THREADS = 1 and 8 bitwise identical, and bitwise identical
   again after moving to another point and back.
5. LIGHTHOUSE (fresh subprocess): the C code fed lighthouse's own
   background, growth and linear P(k) (EH98 + sigma_8, mnu = 0) with a
   TOP-HAT kernel table: n_A(z), b_A(z) and the counts against
   tests/lighthouse_reference (exact-P values). Lighthouse uses the fixed
   Tinker alpha = 0.368, as the C code's default mode 0 does, so those
   rows compare the raw numbers; in mode 1 (halo.c's alpha(z)) extra rows
   rescale the lighthouse densities by alpha(z)/0.368 first.

Targets: tables 1e-4 (max |diff|/max|signal| per row), delta^T C^-1 delta
< 0.01 per block (whole-code budget 0.2).
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

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PRJ = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(PRJ, "data")
REFDIR = os.path.join(PRJ, "tests", "reference")
LHDIR = os.path.join(PRJ, "tests", "lighthouse_reference")
LIKDIR = os.path.join(PRJ, "likelihood")
if REFDIR not in sys.path:
    sys.path.insert(0, REFDIR)

COVERH0 = 2997.92458                  # c/H0 in Mpc/h (the library's unit)
YAML = os.path.join(LIKDIR, "combo_4x2pt_N.yaml")
DATASET = os.path.join(DATA, "des_cluster_y6_4x2ptN.dataset")
PROBE = "4x2pt_N"
ELL_TEST = np.array([2.0, 10.0, 100.0, 1000.0, 1.0e4, 5.0e4])
LMIN_TAB = 20                         # limits.LMIN_tab: first l of the cached C_l tables
BLOCKS = ("ss", "gs", "gg", "cg", "N", "cc", "cs")
CLUSTER_BLOCKS = ("N", "cs", "cc", "cg")
TABLE_TOL = 1e-4
CHI2_TOL = 0.01
CHI2_BUDGET = 0.2
MIN_CORR_EIG = 1e-4
DENSE_K_FACTOR = 4                    # matched variant: log10 k nodes x the likelihood's
VARIANTS = ("matched", "production", "diagnostic", "nuisance")
# the "nuisance" variant: the diagnostic settings at a point with NLA and
# shear calibration switched on (both vanish at the Table I fiducial)
NUISANCE_POINT = dict(IA_A1=0.7, IA_eta1=-1.2, shear_m=[0.012, -0.021, 0.015, -0.006])
REFERENCE_EXPORT_VERSION = 1          # bump when build_reference changes (cache key)


# ============================================================================
# configuration: the likelihood yaml and the dataset
# ============================================================================
def load_yaml(path):
    """The likelihood yaml, ignoring cobaya tags (!defaults)."""
    import yaml

    class Loader(yaml.SafeLoader):
        pass

    Loader.add_multi_constructor("!", lambda loader, suffix, node: None)
    with open(path) as f:
        return yaml.load(f, Loader=Loader)


def load_config(hmf_alpha_mode=None):
    """The dataset and the likelihood keys; hmf_alpha_mode (0/1) overrides
    the yaml's cluster_hmf_alpha_mode on both sides."""
    from getdist import IniFile
    y = load_yaml(YAML)
    ini = IniFile(DATASET)

    def lst(key, tp):
        return [tp(x) for x in ini.string(key).replace(",", " ").split()]

    ds = dict(
        nz_lens_file=os.path.join(DATA, ini.string("nz_lens_file")),
        nz_source_file=os.path.join(DATA, ini.string("nz_source_file")),
        nz_cluster_file=os.path.join(DATA, ini.string("nz_cluster_file")),
        mask_file=os.path.join(DATA, ini.string("mask_file")),
        lens_ntomo=ini.int("lens_ntomo"), source_ntomo=ini.int("source_ntomo"),
        n_theta=ini.int("n_theta"), theta_min_arcmin=ini.float("theta_min_arcmin"),
        theta_max_arcmin=ini.float("theta_max_arcmin"),
        cluster_ntomo=ini.int("cluster_ntomo"),
        cluster_zbin_edges=lst("cluster_zbin_edges", float),
        richness_edges=lst("richness_edges", float),
        survey_area_deg2=ini.float("survey_area_deg2"),
        cg_lens_bins=lst("cg_lens_bins", int),
    )
    keys = ("accuracyboost", "internal_accuracyboost", "integration_accuracy",
            "photoz_interpolation_type", "photoz_zmid_convention", "adopt_limber_gs",
            "adopt_limber_gg", "include_HOD_GX", "include_halo_IA", "lmax", "IA_redshift_evolution",
            "IA_model", "IA_code", "bias_model", "cluster_kernel_mode", "cluster_selection_model",
            "cluster_ytransform", "cluster_include_ia", "cluster_magnification",
            "cluster_adopt_limber_cc", "cluster_adopt_limber_cg", "cluster_hmf_alpha_mode")
    lik = {k: y[k] for k in keys}
    if hmf_alpha_mode is not None:
        lik["cluster_hmf_alpha_mode"] = int(hmf_alpha_mode)
    return dict(dataset=ds, likelihood=lik)


def source_file_zmax(path):
    """zmax_all of an n(z) file as the core loader sets it (the last z plus
    one step: the right edge of the last Z_LOW cell)."""
    z = np.loadtxt(path)[:, 0]
    return float(z[-1] + (z[-1] - z[0]) / (z.size - 1.0))


def reference_settings(cfg, variant="matched"):
    ds, lk = cfg["dataset"], cfg["likelihood"]
    s = dict(
        photoz="table", phi_table_file=ds["nz_cluster_file"],
        source_nz=ds["nz_source_file"], lens_nz=ds["nz_lens_file"],
        lmax=int(lk["lmax"]), area_deg2=ds["survey_area_deg2"],
        zc_edges=ds["cluster_zbin_edges"], lambda_edges=ds["richness_edges"],
        ntheta=ds["n_theta"], tmin_arcmin=ds["theta_min_arcmin"],
        tmax_arcmin=ds["theta_max_arcmin"], lens_bins=ds["cg_lens_bins"],
        kernel_mode=int(lk["cluster_kernel_mode"]), C_c=float(lk["cluster_magnification"]),
        hmf_matter="tot", hmf_alpha_mode=int(lk["cluster_hmf_alpha_mode"]),
        pk_nl_z_order=(3 if variant == "production" else 1))
    if variant in ("diagnostic", "nuisance"):
        s["source_g_zmax"] = source_file_zmax(ds["nz_source_file"])
    return s


def likelihood_grids(accuracyboost):
    """z_interp_1D, z_interp_2D, log10k_interp_2D (1/Mpc) of the likelihood."""
    tmp = int(1000 + 250 * accuracyboost)
    z1 = np.concatenate((np.linspace(0.0, 3.0, max(100, int(0.80 * tmp)), endpoint=False),
                         np.linspace(3.0, 50.1, max(100, int(0.40 * tmp)), endpoint=False),
                         np.linspace(1070, 1100, max(50, int(0.10 * tmp)))), axis=0)
    m = int(min(2 ** np.ceil(np.log2(max(1.0, accuracyboost))), 16))
    z2 = np.concatenate((np.linspace(0, 3.0, 105 * m, endpoint=False),
                         np.linspace(3.0, 49.99, 34 * m + 1)), axis=0)
    lk = np.linspace(-4.99, 2.0, int(1250 + 250 * accuracyboost))
    return z1, z2, lk


# ============================================================================
# step 1: the Python reference (cached)
# ============================================================================
def config_hash(cfg, variant):
    s = json.dumps(cfg, sort_keys=True) + variant + str(REFERENCE_EXPORT_VERSION)
    for f in (cfg["dataset"]["nz_cluster_file"], cfg["dataset"]["nz_source_file"],
              cfg["dataset"]["nz_lens_file"]):
        with open(f, "rb") as fh:
            s += hashlib.sha256(fh.read()).hexdigest()
    for f in sorted(os.listdir(REFDIR)):
        if f.endswith(".py") and not f.startswith("test_"):
            with open(os.path.join(REFDIR, f), "rb") as fh:
                s += hashlib.sha256(fh.read()).hexdigest()
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def cluster_support(nz_cluster_file):
    """zmin/zmax per bin: the zero nodes bracketing the nonzero values (the
    interface convention, generic_interface_cluster.cpp set_cluster_zdist)."""
    tab = np.loadtxt(nz_cluster_file)
    z = tab[:, 0]
    out = []
    for i in range(tab.shape[1] - 1):
        nz = np.nonzero(tab[:, i + 1] > 0)[0]
        out.append((z[max(nz[0] - 1, 0)], z[min(nz[-1] + 1, z.size - 1)]))
    return np.array(out)


def cluster_a_nodes(support, n_a):
    """The a nodes of halo_cluster.c's tables (Ntable.halo_na_lens nodes
    uniform in a over every bin's support)."""
    a_lo = 1.0 / (1.0 + support[:, 1].max())
    a_hi = 1.0 / (1.0 + support[:, 0].min())
    return np.linspace(a_lo, a_hi, n_a)


def cosmology_inputs(cosmo, cfg, variant):
    """The ci.set_cosmology arrays in the likelihood's units (h/Mpc,
    (Mpc/h)^3, Mpc/h), on the grids of the variant."""
    lk = cfg["likelihood"]
    z1, z2, lk10 = likelihood_grids(lk["accuracyboost"])
    h = cosmo.h
    if variant == "production":
        log10k_h = lk10 - np.log10(h)
        z_2D = z2
    else:
        # the CAMB z nodes (<= 3 uniform segments), dense uniform log10 k
        log10k_h = np.linspace(lk10[0], lk10[-1], DENSE_K_FACTOR * lk10.size) - np.log10(h)
        z_2D = np.array(cosmo.z_pk)
    k_h = 10.0 ** log10k_h
    ZMAX_TAB = cosmo.z_pk[-1]

    def lnP_grid(kind):
        zz, kk = np.meshgrid(z_2D, k_h, indexing="ij")       # (nz, nk)
        lnp = cosmo.lnP(kk, np.minimum(zz, ZMAX_TAB), kind)
        # beyond the CAMB table (z > 4, never inside a cluster kernel):
        # matter-dominated growth, D proportional to a
        hi = zz > ZMAX_TAB
        lnp[hi] += 2.0 * np.log((1.0 + ZMAX_TAB) / (1.0 + zz[hi]))
        return lnp.flatten(order="F")                        # io(i*nz + j) = P(k_i, z_j)

    # G on the dense 1D grid, as the likelihood (clipped to the 2D z range)
    z_growth = z1[z1 <= z2[-1]]
    zc = np.minimum(z_growth, ZMAX_TAB)
    G = cosmo.growth(zc) * (1.0 + zc)                         # G = D (1+z), const for z > 4
    G = G / G[-1]
    return dict(cin_log10k_2D=log10k_h, cin_z_2D=z_2D, cin_lnPL=lnP_grid("lin"),
                cin_lnPNL=lnP_grid("nl"), cin_z_G=z_growth, cin_G=G, cin_z_1D=z1,
                cin_chi=cosmo.chi(z1), cin_omegam=cosmo.Omega_m, cin_omegab=cosmo.Omega_b,
                cin_H0=100.0 * h, cin_mnu=cosmo.mnu, cin_omnuh2=cosmo.omnuh2)


def build_reference(cfg, variant, path, verbose=True):
    from reference_cluster import ClusterReference
    from ref_limber import LimberModel

    t0 = time.time()
    settings = reference_settings(cfg, variant)
    # Table I fiducial (reference_cluster.FIDUCIAL), updated by the variant's point
    params = dict(NUISANCE_POINT) if variant == "nuisance" else None
    ref = ClusterReference(params, settings, verbose=verbose)
    cl, cosmo = ref.cluster, ref.cosmo
    out = ref.results(with_cov=(variant in ("matched", "production")))
    lk = cfg["likelihood"]

    # --- kernels on a z grid that is NOT the table grid (tests the reads)
    zk = np.arange(0.1502, 0.7200, 0.00037)
    kz = ref.kernels(zk)
    out.update(val_zk=zk, val_phi=kz["phi"], val_q=kz["q"])
    zg = np.concatenate([np.linspace(0.0005, 0.15, 60), zk])
    out.update(val_zg=zg, val_g=ref.kernels(zg)["g_c"])

    # --- n_A, b_A at the a nodes of the C tables (and the midpoints)
    support = cluster_support(settings["phi_table_file"])
    n_a = int(np.ceil(51 * lk["accuracyboost"]))      # Ntable.halo_na_lens
    a_nodes = cluster_a_nodes(support, n_a)
    a_mid = 0.5 * (a_nodes[1:] + a_nodes[:-1])
    for tag, a in (("nodes", a_nodes), ("mid", a_mid)):
        n, b = ref.nA_bA(1.0 / a - 1.0)
        out.update({"val_a_" + tag: a, "val_nA_" + tag: n, "val_bA_" + tag: b})

    # --- one-halo P_cm at k in [1e-3, 200] h/Mpc on a few z
    kp = np.logspace(-3, np.log10(200.0), 61)
    zp = np.array([1.0 / a_nodes[len(a_nodes) // 2] - 1.0, 0.3, 0.475, 0.6])
    out.update(val_k_p1h=kp, val_z_p1h=zp,
               val_p1h=np.array([ref.p1h(kp, z) for z in zp]))      # (z, A, k)

    # --- Limber spectra at the test multipoles
    s = ref.settings
    nuis = {k: ref.params[k] for k in ("lens_b1", "lens_bmag", "lens_dz", "lens_stretch",
                                       "source_dz", "IA_A1", "IA_eta1")}
    lm = LimberModel(cosmo, cl, ref.nz_src, ref.nz_lens, nuis, kernel_mode=s["kernel_mode"],
                     ells=ELL_TEST, lens_bins=s["lens_bins"], z_panel=s["z_panel"],
                     z_panel_far=s["z_panel_far"], order=s["z_order"],
                     mag_ell_prefactor=s["mag_ell_prefactor"],
                     spin2_prefactor=s["spin2_prefactor"], C_c=s["C_c"],
                     include_1h=s["include_1h"], source_g_zmax=s["source_g_zmax"])
    sp = lm.compute(with_cov_spectra=False)
    out.update(val_ells=ELL_TEST, val_C_cs=sp["C_cs"], val_C_cc=sp["C_cc"],
               val_C_cg=sp["C_cg"], val_C_cs_1h=sp["C_cs_1h"])

    out.update(cosmology_inputs(cosmo, cfg, variant))
    out["variant"] = np.array(variant)
    out["settings_json"] = np.array(json.dumps({k: (v if not isinstance(v, np.ndarray)
                                                    else v.tolist())
                                                for k, v in ref.settings.items()}, default=float))
    np.savez(path, **out)
    if verbose:
        print(f"[reference] {variant}: {path} written in {time.time() - t0:.0f} s", flush=True)


def load_reference(args, cfg, variant):
    path = os.path.join(args.cache_dir, f"reference_{variant}_{config_hash(cfg, variant)}.npz")
    if args.recompute_reference or not os.path.exists(path):
        build_reference(cfg, variant, path)
    return dict(np.load(path)), path


# ============================================================================
# step 2: the C pipeline (the likelihood's init chain, no cobaya)
# ============================================================================
def write_dummy_data(workdir, ndata):
    """Zero data vector and unit diagonal covariance (the interface needs
    files; only the theory vector is used here)."""
    os.makedirs(workdir, exist_ok=True)
    dvf = os.path.join(workdir, "zero.datavector")
    covf = os.path.join(workdir, "diag.cov")
    np.savetxt(dvf, np.column_stack([np.arange(ndata), np.zeros(ndata)]), fmt="%d %.1f")
    np.savetxt(covf, np.column_stack([np.arange(ndata), np.arange(ndata), np.ones(ndata)]),
               fmt="%d %d %.1f")
    return dvf, covf


def write_mask(workdir, name, mask):
    f = os.path.join(workdir, name)
    np.savetxt(f, np.column_stack([np.arange(mask.size), mask]), fmt="%d %d")
    return f


def init_c(cfg, workdir, threads, nz_cluster_file=None):
    """The init chain of _cosmolike_prototype_base.initialize (probe 4x2pt_N)."""
    import cosmolike_des_cluster_interface as ci
    ds, lk = cfg["dataset"], cfg["likelihood"]
    ci.set_omp_threads(threads)
    ci.initial_setup()
    ci.reset_cluster()
    ci.init_probes_cluster(possible_probes=PROBE)
    ci.init_binning(int(ds["n_theta"]), ds["theta_min_arcmin"], ds["theta_max_arcmin"])
    ci.set_log_level_info()
    ci.init_photoz_conventions(interpolation_type=int(lk["photoz_interpolation_type"]),
                               zmid_convention=int(lk["photoz_zmid_convention"]))
    ci.init_fpt_internal_boost(internal_boost=float(lk["internal_accuracyboost"]))
    ci.init_adopt_limber_gs(adopt_limber_gs=int(lk["adopt_limber_gs"]))
    ci.init_adopt_limber_gg(adopt_limber_gg=int(lk["adopt_limber_gg"]))
    ci.init_include_HOD_GX(include_HOD_GX=int(lk["include_HOD_GX"]))
    ci.init_include_halo_IA(include_halo_IA=int(lk["include_halo_IA"]))
    ci.init_ntable_lmax(lmax=int(lk["lmax"]))
    ci.init_accuracy_boost(accuracy_boost=float(lk["accuracyboost"]),
                           integration_accuracy=int(lk["integration_accuracy"]))
    ci.init_cosmo_runmode(is_linear=False)
    ci.init_redshift_distributions_from_files(
        lens_multihisto_file=ds["nz_lens_file"], lens_ntomo=int(ds["lens_ntomo"]),
        source_multihisto_file=ds["nz_source_file"], source_ntomo=int(ds["source_ntomo"]))
    # init_cluster_related
    ci.init_survey_parameters(surveyname="DES", area=ds["survey_area_deg2"], sigma_e=0.0)
    ci.init_cluster_model(mor_model=0, kernel_mode=int(lk["cluster_kernel_mode"]),
                          selection_model=int(lk["cluster_selection_model"]),
                          ytransform=int(lk["cluster_ytransform"]),
                          include_ia=int(lk["cluster_include_ia"]),
                          magnification=float(lk["cluster_magnification"]))
    ci.init_cluster_hmf_alpha_mode(hmf_alpha_mode=int(lk["cluster_hmf_alpha_mode"]))
    ci.init_cluster_adopt_limber(adopt_limber_cc=int(lk["cluster_adopt_limber_cc"]),
                                 adopt_limber_cg=int(lk["cluster_adopt_limber_cg"]))
    edges = np.array(ds["richness_edges"])
    ci.init_cluster_richness_bins(lambda_min=edges[:-1].copy(), lambda_max=edges[1:].copy())
    nzf = nz_cluster_file or ds["nz_cluster_file"]
    zed = np.array(ds["cluster_zbin_edges"])
    ci.set_cluster_zdist(nofz=np.loadtxt(nzf), zbin_min=zed[:-1].copy(), zbin_max=zed[1:].copy())
    ci.init_cluster_pairs(cg_lens_bin=list(ds["cg_lens_bins"]))
    ndata = int(np.array(ci.compute_data_vector_cluster_sizes()).sum())
    dvf, covf = write_dummy_data(workdir, ndata)
    ci.init_data_cluster(covf, ds["mask_file"], dvf)
    ci.init_IA(ia_model=int(lk["IA_model"]), ia_redshift_evolution=int(lk["IA_redshift_evolution"]),
               ia_code=int(lk["IA_code"]))
    ci.init_bias(bias_model=np.array(lk["bias_model"], dtype=float))
    return ci, dict(dvf=dvf, covf=covf, ndata=ndata)


def set_cosmology_c(ci, R, lnP_shift=0.0):
    """ci.set_cosmology with the reference tables (interface units)."""
    ci.set_cosmology(omegam=float(R["cin_omegam"]), omegab=float(R["cin_omegab"]),
                     H0=float(R["cin_H0"]), log10k_2D=R["cin_log10k_2D"], z_2D=R["cin_z_2D"],
                     lnP_linear=R["cin_lnPL"] + lnP_shift, lnP_nonlinear=R["cin_lnPNL"] + lnP_shift,
                     G=R["cin_G"], z_G=R["cin_z_G"], z_1D=R["cin_z_1D"], chi=R["cin_chi"])


def set_nuisance_c(ci, cfg, P, mor_shift=0.0):
    """Nuisance parameters at the reference point P (reference names)."""
    ds = cfg["dataset"]
    nl, ns = int(ds["lens_ntomo"]), int(ds["source_ntomo"])
    ci.set_point_mass(PMV=np.zeros(nl))
    ci.set_nuisance_bias(B1=np.array(P["lens_b1"], float), B2=np.zeros(nl),
                         B_MAG=np.array(P["lens_bmag"], float), B3nl=np.zeros(nl), BK=np.zeros(nl))
    ci.set_nuisance_clustering_photoz(bias=np.array(P["lens_dz"], float),
                                      stretch=np.array(P["lens_stretch"], float))
    ci.set_nuisance_shear_calib(M=np.array(P["shear_m"], float))
    ci.set_nuisance_shear_photoz(bias=np.array(P["source_dz"], float))
    A1 = np.zeros(ns)
    A1[0], A1[1] = P["IA_A1"], P["IA_eta1"]       # IA_REDSHIFT_EVOLUTION: a1, eta1
    ci.set_nuisance_ia(A1=A1, A2=np.zeros(ns), B_TA=np.zeros(ns))
    # cluster.mor order: ln lambda_0, A_lambda, sigma_int, B_lambda
    ci.set_nuisance_cluster_mor(MOR=np.array([P["lnlambda0"] + mor_shift, P["A"],
                                              P["sigma_int"], P["B"]]))
    ci.set_nuisance_cluster_selection(SEL=np.array([P["sel_s0"], P["sel_s1"], P["sel_s2"],
                                                    P["sel_s3"]]))


def c_block_slices(ci):
    sizes = np.array(ci.compute_data_vector_cluster_sizes())
    starts = np.array(ci.compute_data_vector_cluster_starts())
    return {b: slice(int(starts[n]), int(starts[n] + sizes[n])) for n, b in enumerate(BLOCKS)}


def cluster_uncut_mask(ci, cfg):
    """The production mask with every cluster entry switched on."""
    m = np.loadtxt(cfg["dataset"]["mask_file"])[:, 1].astype(int)
    sl = c_block_slices(ci)
    for b in CLUSTER_BLOCKS:
        m[sl[b]] = 1
    return m


def c_evaluate(ci, cfg, R, info, workdir, full=True):
    """Every C output of the comparison, at the reference point, from the
    cosmology tables of R."""
    ds = cfg["dataset"]
    nzc = int(ds["cluster_ntomo"])
    nA = len(ds["richness_edges"]) - 1
    C = {}
    ci.init_data_cluster(info["covf"], ds["mask_file"], info["dvf"])
    set_cosmology_c(ci, R)
    ci.cluster_warmup()
    C["pairs_cs"] = [(ci.ZC_cs(n), ci.ZS_cs(n)) for n in range(nzc * int(ds["source_ntomo"]))]
    C["pairs_cc"] = [(ci.NL1_cc(n), ci.NL2_cc(n)) for n in range(nA * (nA + 1) // 2)]
    C["pairs_cg"] = [(ci.ZC_cg(n), ci.ZG_cg(n)) for n in range(nzc)]
    if full:
        zk, zg = R["val_zk"], R["val_zg"]
        C["phi"] = np.array([[ci.phi_cluster(z, i) for z in zk] for i in range(nzc)])
        C["nz"] = np.array([[ci.nz_cluster(z, i, 0) for z in zk] for i in range(nzc)])
        C["g"] = np.array([[ci.g_cluster(1.0 / (1.0 + z), i, 0) for z in zg] for i in range(nzc)])
        for tag in ("nodes", "mid"):
            a = R["val_a_" + tag]
            C["nA_" + tag] = np.array([[ci.ncl_richness(x, A) for x in a]
                                       for A in range(nA)]) / COVERH0**3
            C["bA_" + tag] = np.array([[ci.bcl_richness(x, A) for x in a] for A in range(nA)])
        kp, zp = R["val_k_p1h"], R["val_z_p1h"]
        C["p1h"] = np.array([[[ci.pcm_1h_richness(k * COVERH0, 1.0 / (1.0 + z), A) for k in kp]
                              for A in range(nA)] for z in zp]) * COVERH0**3
        C["N"] = np.array(ci.N_cluster_tomo())
        C["T"] = np.array(ci.get_cluster_ytransform_matrix())
        C["B"] = np.array(ci.get_cluster_selection_factor())
    ells = R["val_ells"]
    C["C_cs"] = np.array(ci.C_cs_tomo_limber(l=ells))            # (pair, nl, ell)
    C["C_cc"] = np.array(ci.C_cc_tomo_limber(l=ells))            # (ni, pair, ell)
    C["C_cg"] = np.array(ci.C_cg_tomo_limber(l=ells))            # (pair, nl, ell)
    if full:
        et = ells[ells >= LMIN_TAB]
        C["C_cs_tab"] = np.array([[[ci.C_cs_tomo_limber(l, A, i, s) for l in et]
                                   for A in range(nA)] for (i, s) in C["pairs_cs"]])
        C["C_cc_tab"] = np.array([[[ci.C_cc_tomo_limber(l, A, B, i) for l in et]
                                   for (A, B) in C["pairs_cc"]] for i in range(nzc)])
        C["C_cg_tab"] = np.array([[[ci.C_cg_tomo_limber(l, A, i, g) for l in et]
                                   for A in range(nA)] for (i, g) in C["pairs_cg"]])
    C["gt"] = np.array(ci.w_gammat_cluster_tomo())               # (pair, nl, theta)
    C["wcc"] = np.array(ci.w_cc_tomo(1))                         # (ni, pair, theta)
    C["wcg"] = np.array(ci.w_cg_tomo(1))                         # (pair, nl, theta)
    C["dv_cut"] = np.array(ci.compute_data_vector_cluster_masked())
    C["mask_cut"] = np.array(ci.get_mask_cluster())
    m_all = cluster_uncut_mask(ci, cfg)
    ci.init_data_cluster(info["covf"], write_mask(workdir, "uncut.mask", m_all), info["dvf"])
    C["dv_all"] = np.array(ci.compute_data_vector_cluster_masked())
    C["mask_all"] = np.array(ci.get_mask_cluster())
    C["slices"] = c_block_slices(ci)
    return C


# ============================================================================
# comparison helpers
# ============================================================================
def compare(rows, name, c, r, labels=None, rel_floor=1e-12, tol=TABLE_TOL, note=""):
    """max relative and max |diff|/max|signal| (per row = last axis)."""
    c = np.asarray(c, dtype=float)
    r = np.asarray(r, dtype=float)
    assert c.shape == r.shape, (name, c.shape, r.shape)
    d = np.abs(c - r)
    scale = np.max(np.abs(r))
    ok = np.abs(r) > rel_floor * scale
    rel = np.where(ok, d / np.where(ok, np.abs(r), 1.0), 0.0)
    irel = np.unravel_index(np.argmax(rel), rel.shape)
    r2 = r.reshape(-1, r.shape[-1]) if r.ndim > 1 else r[None, :]
    d2 = d.reshape(-1, d.shape[-1]) if d.ndim > 1 else d[None, :]
    rowmax = np.max(np.abs(r2), axis=1)
    rowmax = np.where(rowmax > 0, rowmax, 1.0)
    sc = np.max(d2, axis=1) / rowmax
    irow = int(np.argmax(sc))
    isc = np.unravel_index(irow * r2.shape[1] + int(np.argmax(d2[irow])), r.shape)

    def where(ix):
        if labels is None:
            return str(tuple(int(i) for i in ix))
        return labels(tuple(int(i) for i in ix))

    row = dict(name=name, n=int(r.size), max_rel=float(rel[irel]), where_rel=where(irel),
               max_scaled=float(sc[irow]), where_scaled=where(isc),
               status=("ok" if sc[irow] < tol else "MISS"), tol=tol, note=note)
    rows.append(row)
    return row


def print_table(rows, title):
    print("\n" + title)
    print("-" * len(title))
    print(f"{'quantity':36s} {'n':>5s} {'max rel':>9s} {'at':30s} {'max|d|/max|s|':>13s} "
          f"{'at':30s}")
    for r in rows:
        if "chi2" in r:
            print(f"{r['name']:36s} {r['n']:5d} {'chi2 =':>9s} {r['chi2']:<30.3e} "
                  f"{'':13s} {r.get('note', '')[:30]:30s} {r['status']}")
            continue
        print(f"{r['name']:36s} {r['n']:5d} {r['max_rel']:9.2e} {r['where_rel'][:30]:30s} "
              f"{r['max_scaled']:13.2e} {r['where_scaled'][:30]:30s} {r['status']}"
              + (f"  {r['note']}" if r.get("note") else ""))


# ============================================================================
# covariance scoring
# ============================================================================
def min_corr_eig(C):
    d = np.sqrt(np.diag(C))
    return float(np.linalg.eigvalsh(C / np.outer(d, d)).min())


def aggressive_pd_mask(C, base, candidates, threshold=MIN_CORR_EIG):
    """Re-admit candidate points (in the given order) while the correlation
    matrix of the kept set keeps its smallest eigenvalue >= threshold."""
    keep = list(np.nonzero(base)[0])
    added = []
    for p in candidates:
        trial = sorted(keep + [p])
        if min_corr_eig(C[np.ix_(trial, trial)]) >= threshold:
            keep = trial
            added.append(p)
    m = np.zeros_like(base)
    m[keep] = 1
    return m, added


def chi2(delta, C, sel):
    idx = np.nonzero(sel)[0]
    if idx.size == 0:
        return 0.0, 0
    d = delta[idx]
    L = np.linalg.cholesky(C[np.ix_(idx, idx)])
    y = np.linalg.solve(L, d)
    return float(y @ y), int(idx.size)


def reference_in_c_layout(C, R, cfg):
    """The reference data vector and covariance (N, cs, cc, cg) mapped to
    the joint C layout (ss, gs, gg, cg, N, cc, cs); -1 = not a cluster entry."""
    ds = cfg["dataset"]
    nzc, ns, nt = int(ds["cluster_ntomo"]), int(ds["source_ntomo"]), int(ds["n_theta"])
    nA = len(ds["richness_edges"]) - 1
    sl, n = C["slices"], C["dv_all"].size
    idx = R["dv_start"]
    start = dict(N=idx[0], cs=idx[1], cc=idx[2], cg=idx[3])
    perm = np.full(n, -1)
    for i in range(nzc):                                   # N [zc][lambda]
        for A in range(nA):
            perm[sl["N"].start + nA * i + A] = start["N"] + nA * i + A
    for p, (i, s) in enumerate(C["pairs_cs"]):             # cs [pair][lambda][theta]
        for A in range(nA):
            for t in range(nt):
                perm[sl["cs"].start + nt * (nA * p + A) + t] = \
                    start["cs"] + nt * (nA * (ns * i + s) + A) + t
    ref_pairs_cc = [(a, b) for a in range(nA) for b in range(a, nA)]
    npc = len(ref_pairs_cc)
    for i in range(nzc):                                   # cc [zc][pair][theta]
        for p, (A, B) in enumerate(C["pairs_cc"]):
            for t in range(nt):
                perm[sl["cc"].start + nt * (npc * i + p) + t] = \
                    start["cc"] + nt * (npc * i + ref_pairs_cc.index((A, B))) + t
    for p, (i, g) in enumerate(C["pairs_cg"]):             # cg [pair][lambda][theta]
        for A in range(nA):
            for t in range(nt):
                perm[sl["cg"].start + nt * (nA * p + A) + t] = start["cg"] + nt * (nA * i + A) + t
    inC = perm >= 0
    ref_full = np.zeros(n)
    ref_full[inC] = R["data_vector"][perm[inC]]
    cov_full = None
    if "cov" in R:
        cov_full = np.zeros((n, n))
        ii = np.nonzero(inC)[0]
        cov_full[np.ix_(ii, ii)] = R["cov"][np.ix_(perm[ii], perm[ii])]
    return ref_full, cov_full


def compare_all(C, R, cfg, full=True, blocks=CLUSTER_BLOCKS, cov_full=None, pd_cache=None,
                shear_m=None):
    """Comparison rows of C against the reference R (and chi2 rows).
    shear_m: the point's shear calibration; the reference's gamma_t carries
    (1 + m_s), w_gammat_cluster_tomo does not (the interface applies it on
    the data vector)."""
    rows = []
    ds = cfg["dataset"]
    nzc = int(ds["cluster_ntomo"])
    nA = len(ds["richness_edges"]) - 1
    nt = int(ds["n_theta"])
    lens_bins = list(ds["cg_lens_bins"])
    pairs_cs, pairs_cc, pairs_cg = C["pairs_cs"], C["pairs_cc"], C["pairs_cg"]
    ells = R["val_ells"]
    th = R["theta_sel"] * 180 * 60 / np.pi
    if full:
        zk, zg = R["val_zk"], R["val_zg"]
        compare(rows, "phi_cluster", C["phi"], R["val_phi"],
                labels=lambda ix: f"bin {ix[0]} z={zk[ix[1]]:.4f}")
        compare(rows, "nz_cluster (volume kernel)", C["nz"], R["val_q"][:, 0, :],
                labels=lambda ix: f"bin {ix[0]} z={zk[ix[1]]:.4f}")
        compare(rows, "g_cluster (magnification eff.)", C["g"], R["val_g"][:, 0, :],
                labels=lambda ix: f"bin {ix[0]} z={zg[ix[1]]:.4f}")
        for tag in ("nodes", "mid"):
            a = R["val_a_" + tag]
            lab = (lambda aa: (lambda ix: f"lambda {ix[0]} z={1 / aa[ix[1]] - 1:.4f}"))(a)
            compare(rows, f"ncl_richness (a {tag})", C["nA_" + tag], R["val_nA_" + tag], labels=lab)
            compare(rows, f"bcl_richness (a {tag})", C["bA_" + tag], R["val_bA_" + tag], labels=lab)
        kp, zp = R["val_k_p1h"], R["val_z_p1h"]
        lab = lambda ix: f"z={zp[ix[0]]:.3f} l{ix[1]} k={kp[ix[2]]:.3g}"
        compare(rows, "pcm_1h_richness (k <= 20 h/Mpc)", C["p1h"][..., kp <= 20],
                R["val_p1h"][..., kp <= 20], labels=lab)
        compare(rows, "pcm_1h_richness (k <= 200 h/Mpc)", C["p1h"], R["val_p1h"], labels=lab)
        compare(rows, "N_cluster_tomo", C["N"], R["N"],
                labels=lambda ix: f"zbin {ix[0]} lambda {ix[1]}")

    Ccs_ref = np.array([[R["val_C_cs"][i, A, s] for A in range(nA)] for (i, s) in pairs_cs])
    Ccc_ref = np.array([[R["val_C_cc"][i, A, i, B] for (A, B) in pairs_cc] for i in range(nzc)])
    Ccg_ref = np.array([[R["val_C_cg"][i, A, lens_bins.index(g)] for A in range(nA)]
                        for (i, g) in pairs_cg])
    lab_cs = lambda e: (lambda ix: f"zc{pairs_cs[ix[0]][0]} zs{pairs_cs[ix[0]][1]} l{ix[1]} ell={e[ix[2]]:g}")
    lab_cc = lambda e: (lambda ix: f"zc{ix[0]} l{pairs_cc[ix[1]]} ell={e[ix[2]]:g}")
    lab_cg = lambda e: (lambda ix: f"zc{pairs_cg[ix[0]][0]} l{ix[1]} ell={e[ix[2]]:g}")
    if "cs" in blocks:
        compare(rows, "C_cs_tomo_limber (exact l)", C["C_cs"], Ccs_ref, labels=lab_cs(ells))
    if "cc" in blocks:
        compare(rows, "C_cc_tomo_limber (exact l)", C["C_cc"], Ccc_ref, labels=lab_cc(ells))
    if "cg" in blocks:
        compare(rows, "C_cg_tomo_limber (exact l)", C["C_cg"], Ccg_ref, labels=lab_cg(ells))
    if full:
        tab = ells >= LMIN_TAB
        et = ells[tab]
        compare(rows, "C_cs_tomo_limber (table, l>=20)", C["C_cs_tab"], Ccs_ref[..., tab],
                labels=lab_cs(et))
        compare(rows, "C_cc_tomo_limber (table, l>=20)", C["C_cc_tab"], Ccc_ref[..., tab],
                labels=lab_cc(et))
        compare(rows, "C_cg_tomo_limber (table, l>=20)", C["C_cg_tab"], Ccg_ref[..., tab],
                labels=lab_cg(et))

    lab_t = lambda ix: f"zc{pairs_cs[ix[0]][0]} zs{pairs_cs[ix[0]][1]} l{ix[1]} th={th[ix[2]]:.3g}'"
    if "cs" in blocks:
        gt_ref = np.array([[R["gamma_t"][i, A, s] for A in range(nA)] for (i, s) in pairs_cs])
        m = np.zeros(int(ds["source_ntomo"])) if shear_m is None else np.asarray(shear_m)
        gt_c = C["gt"] * (1.0 + np.array([m[s] for (i, s) in pairs_cs]))[:, None, None]
        compare(rows, "w_gammat_cluster_tomo (gamma_t)", gt_c, gt_ref, labels=lab_t,
                note=("" if shear_m is None else "x (1 + m_s)"))
        T = R["T"]
        Sig = np.einsum("tu,pau->pat", T, gt_c)
        Sig_ref = np.einsum("tu,pau->pat", T, gt_ref)
        compare(rows, "Sigma = T gamma_t (theta < N-1)", Sig[..., :-1], Sig_ref[..., :-1],
                labels=lab_t)
        if full:
            compare(rows, "Y transform matrix T", C["T"], T, rel_floor=1e-9)
    if "cc" in blocks:
        wcc_ref = np.array([[R["w_cc"][i, A, B] for (A, B) in pairs_cc] for i in range(nzc)])
        compare(rows, "w_cc_tomo (Limber)", C["wcc"], wcc_ref,
                labels=lambda ix: f"zc{ix[0]} l{pairs_cc[ix[1]]} th={th[ix[2]]:.3g}'")
    if "cg" in blocks:
        wcg_ref = np.array([[R["w_cg"][i, A] for A in range(nA)] for (i, g) in pairs_cg])
        compare(rows, "w_cg_tomo (Limber)", C["wcg"], wcg_ref,
                labels=lambda ix: f"zc{pairs_cg[ix[0]][0]} l{ix[1]} th={th[ix[2]]:.3g}'")
    if full:
        compare(rows, "selection factor B(theta)", C["B"], R["b_sel"],
                labels=lambda ix: f"zc{ix[0]} th={th[ix[1]]:.3g}'")

    # --- the joint data vector
    ref_full, cov_R = reference_in_c_layout(C, R, cfg)
    cov_full = cov_R if cov_full is None else cov_full
    sl, mask_cut, mask_all = C["slices"], C["mask_cut"], C["mask_all"]
    dv_cut, dv_all = C["dv_cut"], C["dv_all"]
    for b in blocks:
        s_ = sl[b]
        mk = mask_all[s_] == 1
        compare(rows, f"data vector {b} (uncut)", dv_all[s_][mk][None, :], ref_full[s_][mk][None, :],
                note="one row = the block")
    chi = {}
    if cov_full is None:
        return rows, chi
    delta_cut = dv_cut - ref_full
    delta_all = dv_all - ref_full
    total_cut = np.zeros(dv_all.size, int)
    total_all = np.zeros(dv_all.size, int)
    pd_cache = {} if pd_cache is None else pd_cache
    for b in blocks:
        s_ = sl[b]
        sel_cut = np.zeros(dv_all.size, int)
        sel_cut[s_] = mask_cut[s_]
        sel_all = np.zeros(dv_all.size, int)
        sel_all[s_] = mask_all[s_]
        note = ""
        if b not in pd_cache:
            ii = np.nonzero(sel_all)[0]
            eig = min_corr_eig(cov_full[np.ix_(ii, ii)])
            if eig < MIN_CORR_EIG:
                cand = [p for p in ii if not sel_cut[p]]
                # re-admit the largest scales first (theta descending)
                cand = sorted(cand, key=lambda p: -((p - s_.start) % nt))
                sel_pd, added = aggressive_pd_mask(cov_full, sel_cut, cand)
                pd_cache[b] = (sel_pd, f"uncut corr min eig {eig:.1e}: PD mask adds "
                                       f"{len(added)}/{len(cand)} cut points")
            else:
                pd_cache[b] = (sel_all, "")
        sel_all, note = pd_cache[b]
        c_cut, n_cut = chi2(delta_cut, cov_full, sel_cut)
        c_all, n_all = chi2(delta_all, cov_full, sel_all)
        chi[b] = dict(cut=c_cut, n_cut=n_cut, all=c_all, n_all=n_all, note=note)
        total_cut |= sel_cut
        total_all |= sel_all
        rows.append(dict(name=f"chi2 {b} (Y6 cuts)", n=n_cut, chi2=c_cut,
                         status="ok" if c_cut < CHI2_TOL else "MISS", note=""))
        rows.append(dict(name=f"chi2 {b} (uncut or PD)", n=n_all, chi2=c_all,
                         status="ok" if c_all < CHI2_TOL else "MISS", note=note))
    if len(blocks) > 1:
        ct, nct = chi2(delta_cut, cov_full, total_cut)
        ii = np.nonzero(total_all)[0]
        if min_corr_eig(cov_full[np.ix_(ii, ii)]) > 0:
            ca, nca = chi2(delta_all, cov_full, total_all)
        else:
            ca, nca = np.nan, int(ii.size)
        rows.append(dict(name="chi2 all cluster blocks (Y6 cuts)", n=nct, chi2=ct,
                         status="ok" if ct < CHI2_BUDGET else "MISS", note="budget 0.2"))
        rows.append(dict(name="chi2 all cluster blocks (uncut/PD)", n=nca, chi2=ca,
                         status="ok" if ca < CHI2_BUDGET else "MISS", note="budget 0.2"))
        chi["total"] = dict(cut=ct, all=ca)
    return rows, chi


# ============================================================================
# step 4: determinism (fresh processes)
# ============================================================================
def emit_dv(cfg, R, workdir, threads, out):
    """Uncut cluster data vector at the fiducial; then a detour to another
    point (cosmology and MOR) and back, in the same process."""
    from reference_cluster import FIDUCIAL
    P = dict(FIDUCIAL)
    ci, info = init_c(cfg, workdir, threads)
    m_all = cluster_uncut_mask(ci, cfg)
    ci.init_data_cluster(info["covf"], write_mask(workdir, "uncut.mask", m_all), info["dvf"])
    set_nuisance_c(ci, cfg, P)
    set_cosmology_c(ci, R)
    dv0 = np.array(ci.compute_data_vector_cluster_masked())
    set_cosmology_c(ci, R, lnP_shift=0.01)
    set_nuisance_c(ci, cfg, P, mor_shift=0.02)
    dv1 = np.array(ci.compute_data_vector_cluster_masked())
    set_cosmology_c(ci, R)
    set_nuisance_c(ci, cfg, P)
    dv2 = np.array(ci.compute_data_vector_cluster_masked())
    np.savez(out, dv0=dv0, dv1=dv1, dv2=dv2)


def mode_args(args):
    """The --hmf-alpha-mode override, passed on to the child processes."""
    if args.hmf_alpha_mode is None:
        return []
    return ["--hmf-alpha-mode", str(args.hmf_alpha_mode)]


def run_determinism(args):
    outs = {}
    for n in (1, 8):
        f = os.path.join(args.cache_dir, f"dv_omp{n}.npz")
        env = dict(os.environ, OMP_NUM_THREADS=str(n))
        cmd = [sys.executable, os.path.abspath(__file__), "--cache-dir", args.cache_dir,
               "--emit-dv", f, "--threads", str(n)] + mode_args(args)
        t0 = time.time()
        subprocess.run(cmd, env=env, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        print(f"[determinism] OMP {n}: {time.time() - t0:.0f} s", flush=True)
        outs[n] = np.load(f)
    a, b = outs[1], outs[8]
    res = dict(omp1_vs_omp8=bool(np.array_equal(a["dv0"], b["dv0"])),
               omp1_back_to_point=bool(np.array_equal(a["dv0"], a["dv2"])),
               omp8_back_to_point=bool(np.array_equal(b["dv0"], b["dv2"])),
               detour_changes_dv=bool(not np.array_equal(a["dv0"], a["dv1"])),
               omp1_vs_omp8_detour=bool(np.array_equal(a["dv1"], b["dv1"])),
               max_abs_diff_omp1_omp8=float(np.max(np.abs(a["dv0"] - b["dv0"]))))
    print("\nDeterminism (uncut joint data vector, fresh processes)")
    for k, v in res.items():
        print(f"  {k:26s} {v}")
    return res


# ============================================================================
# step 5: lighthouse cross-check (fresh process)
# ============================================================================
def lighthouse_inputs(cfg, workdir):
    """Lighthouse's own cosmology on the likelihood grids and a top-hat
    kernel table; returns the C inputs, the kernel file, lighthouse's
    arrays and configuration."""
    from scipy.interpolate import CubicSpline
    from scipy.integrate import cumulative_trapezoid
    L = np.load(os.path.join(LHDIR, "outputs", "lighthouse_reference_main.npz"))
    with open(os.path.join(LHDIR, "outputs", "config.json")) as f:
        conf = json.load(f)
    Om, h = conf["omega_m"], conf["h0"]
    z1, z2, lk10 = likelihood_grids(cfg["likelihood"]["accuracyboost"])
    # background without radiation (lighthouse hoverh0), exact quadrature
    zq = np.concatenate([np.linspace(0, 5, 50001), np.linspace(5.0001, 1100, 200000)])
    E = np.sqrt(Om * (1 + zq) ** 3 + 1 - Om)
    chiq = cumulative_trapezoid(1.0 / E, zq, initial=0.0) * COVERH0
    chi = np.interp(z1, zq, chiq)
    # growth D/D0 (lighthouse table on z <= 3; matter domination beyond)
    Dz = CubicSpline(L["bg_z"], L["bg_growth_D_over_D0"])
    z_growth = z1[z1 <= z2[-1]]
    zc = np.minimum(z_growth, L["bg_z"][-1])
    G = np.where(z_growth < L["bg_z"][0], 1.0, Dz(np.maximum(zc, L["bg_z"][0]))) * (1 + zc)
    G = G / G[-1]
    # linear P: lighthouse's z = 0 table x D^2, in h units
    k_h = L["pk_k_hMpc"]
    lnP0 = np.log(L["pk_lin_code"][0] * COVERH0**3)
    log10k_h = lk10 - np.log10(h)
    spl = CubicSpline(np.log(k_h), lnP0)
    lk = np.log(10.0 ** log10k_h)
    lo, hi = np.log(k_h[0]), np.log(k_h[-1])
    s_lo = (lnP0[1] - lnP0[0]) / (np.log(k_h[1]) - np.log(k_h[0]))
    s_hi = (lnP0[-1] - lnP0[-2]) / (np.log(k_h[-1]) - np.log(k_h[-2]))
    lnP_k = np.where(lk < lo, lnP0[0] + s_lo * (lk - lo),
                     np.where(lk > hi, lnP0[-1] + s_hi * (lk - hi), spl(np.clip(lk, lo, hi))))
    zz = np.clip(z2, L["bg_z"][0], L["bg_z"][-1])
    lnD2 = 2 * np.log(Dz(zz) * np.where(z2 > zz, (1 + zz) / (1 + z2), 1.0))
    lnPL = (lnP_k[None, :] + lnD2[:, None]).flatten(order="F")
    # top-hat kernel table (make_cluster_zdist.py --tophat)
    os.makedirs(workdir, exist_ok=True)
    th = os.path.join(workdir, "des_y6_cluster_tophat.nz")
    subprocess.run([sys.executable, os.path.join(PRJ, "scripts", "make_cluster_zdist.py"),
                    "--tophat", "--out", th], check=True, stdout=subprocess.DEVNULL)
    R = dict(cin_log10k_2D=log10k_h, cin_z_2D=z2, cin_lnPL=lnPL, cin_lnPNL=lnPL,
             cin_z_G=z_growth, cin_G=G, cin_z_1D=z1, cin_chi=chi, cin_omegam=Om,
             cin_omegab=conf["omega_b"], cin_H0=100 * h)
    return R, th, L, conf


def run_lighthouse_child(cfg, workdir, threads, out):
    from ref_halo import tinker_amplitude, TINKER_ALPHA_FIXED
    from scipy.interpolate import CubicSpline
    mode = int(cfg["likelihood"]["cluster_hmf_alpha_mode"])
    R, th, L, conf = lighthouse_inputs(cfg, workdir)
    P = dict(lnlambda0=conf["MOR"][0], A=conf["MOR"][1], sigma_int=conf["MOR"][2],
             B=conf["MOR"][3], sel_s0=1.0, sel_s1=0.0, sel_s2=30.0, sel_s3=0.0,
             lens_b1=conf["b1"], lens_bmag=[0.0] * 6, lens_dz=[0.0] * 6, lens_stretch=[1.0] * 6,
             source_dz=[0.0] * 4, shear_m=[0.0] * 4, IA_A1=0.0, IA_eta1=0.0)
    ci, info = init_c(cfg, workdir, threads, nz_cluster_file=th)
    set_nuisance_c(ci, cfg, P)
    set_cosmology_c(ci, R)
    ci.cluster_warmup()
    nA = len(conf["cluster_lambda_bins"]) - 1
    zz = L["clz_z"]
    a = 1.0 / (1.0 + zz)
    ok = (zz >= 0.2) & (zz <= 0.65)
    n_c = np.array([[ci.ncl_richness(x, A) if o else 0.0 for x, o in zip(a, ok)]
                    for A in range(nA)])            # (c/H0)^-3, as lighthouse
    b_c = np.array([[ci.bcl_richness(x, A) if o else 0.0 for x, o in zip(a, ok)]
                    for A in range(nA)])
    N_c = np.array(ci.N_cluster_tomo())
    # at the z of lighthouse's tabulated P(lambda bin|M) (mor_z)
    a_m = 1.0 / (1.0 + L["mor_z"])
    n_c_m = np.array([[ci.ncl_richness(x, A) for x in a_m] for A in range(nA)])
    b_c_m = np.array([[ci.bcl_richness(x, A) for x in a_m] for A in range(nA)])
    # alpha of the C side (0.368 in mode 0, halo.c's alpha(z) in mode 1)
    # over lighthouse's fixed 0.368: the rescaling that puts lighthouse's
    # densities on the C side's amplitude (1 in mode 0)
    alpha_m = tinker_amplitude(np.maximum(a_m, 0.25), mode) / TINKER_ALPHA_FIXED
    alpha = tinker_amplitude(np.maximum(a, 0.25), mode) / TINKER_ALPHA_FIXED
    nL = L["n_A_exact"] * alpha[None, :]
    # counts from the exact-P lighthouse densities: Omega_s int dz chi^2/E n_A
    # (top hat in true z; the grid nodes fall on the bin edges)
    zb = conf["cluster_zbins"]
    Om = conf["omega_m"]
    E = np.sqrt(Om * (1 + zz) ** 3 + 1 - Om)
    chiL = CubicSpline(L["bg_z"], L["bg_chi"])(zz)
    Omega_s = conf["survey_area"] * (np.pi / 180) ** 2
    N_L_exact = np.zeros((3, nA))
    N_L_exact_alpha = np.zeros((3, nA))
    for i in range(3):
        s = (zz >= zb[i] - 1e-9) & (zz <= zb[i + 1] + 1e-9)
        for A in range(nA):
            N_L_exact[i, A] = Omega_s * np.trapz(chiL[s] ** 2 / E[s] * L["n_A_exact"][A, s], zz[s])
            N_L_exact_alpha[i, A] = Omega_s * np.trapz(chiL[s] ** 2 / E[s] * nL[A, s], zz[s])
    np.savez(out, z=zz, ok=ok, n_c=n_c, b_c=b_c, N_c=N_c, n_L_alpha=nL, n_L=L["n_A_exact"],
             n_L_tab=L["n_A_tab"], b_L=L["b_A_exact"], b_L_tab=L["b_A_tab"], N_L=L["N_counts"],
             N_L_exact=N_L_exact, N_L_exact_alpha=N_L_exact_alpha, alpha_ratio=alpha,
             n_c_m=n_c_m, b_c_m=b_c_m, alpha_ratio_m=alpha_m, hmf_alpha_mode=mode)


def run_lighthouse(args):
    f = os.path.join(args.cache_dir, "lighthouse_c.npz")
    cmd = [sys.executable, os.path.abspath(__file__), "--cache-dir", args.cache_dir,
           "--lighthouse-child", f, "--threads", str(args.threads)] + mode_args(args)
    t0 = time.time()
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                   env=dict(os.environ, OMP_NUM_THREADS=str(args.threads)))
    print(f"[lighthouse] C run: {time.time() - t0:.0f} s", flush=True)
    d = np.load(f)
    # interior z only: lighthouse's b_A table drops to 0.01 below z = 0.2
    inner = d["ok"] & (d["z"] > 0.205) & (d["z"] < 0.645)
    rows = []

    def row(name, c, r):
        rel = np.abs(c / r - 1)
        i = np.unravel_index(np.argmax(rel), rel.shape)
        rows.append((name, float(rel.max()), tuple(int(x) for x in i)))

    mode = int(d["hmf_alpha_mode"])
    # raw: lighthouse as it is (alpha = 0.368); the like-with-like rows of
    # mode 0. Mode 1 adds the rows with lighthouse rescaled to alpha(z).
    row("n_A(z): C / LH exact-P (raw)", d["n_c"][:, inner], d["n_L"][:, inner])
    if mode == 1:
        row("n_A(z): C / (LH exact-P x alpha(z)/0.368)", d["n_c"][:, inner],
            d["n_L_alpha"][:, inner])
    row("b_A(z): C / LH exact-P", d["b_c"][:, inner], d["b_L"][:, inner])
    row("b_A(z): C / LH P-table (LH model)", d["b_c"][:, inner], d["b_L_tab"][:, inner])
    row("N: C / LH counts (P table; raw)", d["N_c"], d["N_L"])
    row("N: C / Omega int dV n_A^LH-exact (raw)", d["N_c"], d["N_L_exact"])
    if mode == 1:
        row("N: C / Omega int dV n_A^LH-exact x alpha(z)/0.368", d["N_c"],
            d["N_L_exact_alpha"])
    # n_A, b_A re-integrated from lighthouse's OWN tabulated ingredients
    # (dn/dM on 79 lg M nodes, exact P(lambda bin|M), B1) on a dense lg M
    # grid: separates lighthouse's CQUAD 1e-2 mass integrals from the model
    from scipy.interpolate import CubicSpline
    L = np.load(os.path.join(LHDIR, "outputs", "lighthouse_reference_main.npz"))
    lgM = L["halo_lgM"]
    x = np.linspace(lgM[0], lgM[-1], 20001)
    nA = d["n_c_m"].shape[0]
    nb, bb = np.zeros((nA, L["mor_z"].size)), np.zeros((nA, L["mor_z"].size))
    for im, zm in enumerate(L["mor_z"]):
        iz = int(np.argmin(np.abs(L["halo_z"] - zm)))
        dn = L["halo_massfunc_dndM"][iz] * 10**lgM                       # dn/dlnM
        lnb = CubicSpline(lgM, np.log(L["halo_B1"][iz]))(x)
        for A in range(nA):
            f = np.exp(CubicSpline(lgM, np.log(np.maximum(dn * L["mor_P_bin_given_M_exact"][A, im],
                                                         1e-300)))(x))
            nb[A, im] = np.trapz(f, x) * np.log(10.0)
            bb[A, im] = np.trapz(f * np.exp(lnb), x) * np.log(10.0) / nb[A, im]
    row("n_A(0.3/0.475/0.6): C / LH ingredients" + (" x alpha(z)/0.368" if mode == 1 else ""),
        d["n_c_m"], nb * d["alpha_ratio_m"][None, :])
    row("b_A(0.3/0.475/0.6): C / LH ingredients", d["b_c_m"], bb)
    print("\nLighthouse cross-check (top-hat kernel, lighthouse EH98 + sigma_8 cosmology, mnu = 0; "
          f"C hmf_alpha_mode = {mode})")
    for name, v, i in rows:
        print(f"  {name:50s} max |ratio - 1| = {v:.2e} at {i}")
    return {name: v for name, v, i in rows}


# ============================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache-dir", default=os.environ.get(
        "DES_CLUSTER_VALIDATION_CACHE", os.path.join(tempfile.gettempdir(), "des_cluster_validation")))
    ap.add_argument("--recompute-reference", action="store_true")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", 4)))
    ap.add_argument("--hmf-alpha-mode", type=int, default=None, choices=(0, 1),
                    help="Tinker amplitude on both sides (default: the yaml's "
                         "cluster_hmf_alpha_mode): 0 = 0.368 (DES), 1 = halo.c's alpha(z)")
    ap.add_argument("--skip-production", action="store_true")
    ap.add_argument("--skip-diagnostics", action="store_true")
    ap.add_argument("--skip-determinism", action="store_true")
    ap.add_argument("--skip-lighthouse", action="store_true")
    ap.add_argument("--json", default=None, help="write the comparison rows to this file")
    ap.add_argument("--emit-dv", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--lighthouse-child", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    os.makedirs(args.cache_dir, exist_ok=True)
    cfg = load_config(args.hmf_alpha_mode)
    workdir = os.path.join(args.cache_dir, "work_" + str(os.getpid()))

    try:
        if args.lighthouse_child:
            run_lighthouse_child(cfg, workdir, args.threads, args.lighthouse_child)
            return
        R, path = load_reference(args, cfg, "matched")
        if args.emit_dv:
            emit_dv(cfg, R, workdir, args.threads, args.emit_dv)
            return
        run_validation(args, cfg, R, path, workdir)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)   # dummy data files of this process


def run_validation(args, cfg, R, path, workdir):
    """Steps 2-5 of the module docstring; prints the tables."""
    from reference_cluster import FIDUCIAL
    print(f"[validation] reference (matched) {path}")
    print(f"[validation] Tinker amplitude: hmf_alpha_mode = "
          f"{int(cfg['likelihood']['cluster_hmf_alpha_mode'])} on both sides "
          f"(0 = alpha 0.368, 1 = halo.c's alpha(z))")
    print(f"[validation] CAMB: mnu = {float(R['cin_mnu']):.5f} eV (3 degenerate), Omega_nu h^2 = "
          f"{float(R['cin_omnuh2']):.5f}; Omega_m = {float(R['cin_omegam'])} (total), "
          f"Omega_b = {float(R['cin_omegab'])}, H0 = {float(R['cin_H0'])}; the C side gets "
          f"these tables and Omega_m, Omega_b, H0")
    out = dict(hmf_alpha_mode=int(cfg["likelihood"]["cluster_hmf_alpha_mode"]))
    t0 = time.time()
    ci, info = init_c(cfg, workdir, args.threads)
    set_nuisance_c(ci, cfg, dict(FIDUCIAL))
    C = c_evaluate(ci, cfg, R, info, workdir, full=True)
    rows, chi = compare_all(C, R, cfg, full=True)
    print(f"[validation] C evaluation + comparison: {time.time() - t0:.0f} s")
    print_table(rows, "MATCHED INPUTS: C port vs Python reference (Table I fiducial, "
                      "likelihood settings)")
    out["matched"] = dict(rows=rows, chi2=chi)
    np.savez(os.path.join(args.cache_dir, "c_results_matched.npz"),
             **{k: v for k, v in C.items() if isinstance(v, np.ndarray)})

    if not args.skip_diagnostics:
        Rd, _ = load_reference(args, cfg, "diagnostic")
        _, cov_m = reference_in_c_layout(C, R, cfg)
        rows_d, chi_d = compare_all(C, Rd, cfg, full=False, blocks=("cs",), cov_full=cov_m)
        print_table(rows_d, "DIAGNOSTIC: matched inputs, reference W_kappa with the source n(z) "
                            "cut at z = 3.0 as the core's g_tomo")
        out["diagnostic"] = dict(rows=rows_d, chi2=chi_d)

        Rn, _ = load_reference(args, cfg, "nuisance")
        Pn = dict(FIDUCIAL)
        Pn.update(NUISANCE_POINT)
        set_nuisance_c(ci, cfg, Pn)
        Cn = c_evaluate(ci, cfg, R, info, workdir, full=False)
        rows_n, chi_n = compare_all(Cn, Rn, cfg, full=False, blocks=("cs",), cov_full=cov_m,
                                    shear_m=NUISANCE_POINT["shear_m"])
        print_table(rows_n, "DIAGNOSTIC + NLA and shear calibration on (A1 = 0.7, eta1 = -1.2, "
                            "m = 0.012, -0.021, 0.015, -0.006)")
        out["nuisance"] = dict(rows=rows_n, chi2=chi_n)
        set_nuisance_c(ci, cfg, dict(FIDUCIAL))

    if not args.skip_production:
        Rp, _ = load_reference(args, cfg, "production")
        Cp = c_evaluate(ci, cfg, Rp, info, workdir, full=False)
        rows_p, chi_p = compare_all(Cp, Rp, cfg, full=False)
        print_table(rows_p, "PRODUCTION INPUTS: C fed on the likelihood grids, reference "
                            "cubic in z")
        out["production"] = dict(rows=rows_p, chi2=chi_p)

    if not args.skip_determinism:
        out["determinism"] = run_determinism(args)
    if not args.skip_lighthouse:
        out["lighthouse"] = run_lighthouse(args)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1, default=float)
    nmiss = sum(r["status"] == "MISS" for r in out["matched"]["rows"])
    print(f"\n[validation] matched: {len(out['matched']['rows'])} rows, {nmiss} outside target")


if __name__ == "__main__":
    main()
