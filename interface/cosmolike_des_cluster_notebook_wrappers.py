"""Notebook wrappers for the des_cluster Cosmolike interface.

The EXAMPLE_EVALUATE notebooks drive the compiled interface
(cosmolike_des_cluster_interface) through the same steps as the
likelihoods combo_4x2pt_N and combo_6x2pt_N: run CAMB once
(cnu.get_camb_cosmology), push the power spectra, growth and distances
into the interface with set_cosmology, set the nuisance parameters, and
read off a cluster observable, a halo-model ingredient, or the masked
data vector and its chi2. This module holds those steps once, so every
notebook of this project imports the same wrappers:

    import cosmolike_des_cluster_notebook_wrappers as nw
    nw.init_cosmolike(CLprobe="6x2pt_N", with_data=True)
    N = nw.N_cluster()                       # (richness bin, z bin)
    theta, gammat = nw.gamma_t_cluster()     # (theta, richness, z, source)
    nw.get_chi2(omegam=0.31)

Three layers of state matter here:

- The compiled interface is a C library with global state: every
  init_* and set_* call replaces part of it, and the observables read
  whatever was set last. Each wrapper therefore sets everything it
  depends on (tables, accuracy, cosmology, nuisances) on every call,
  so no call depends on which wrapper ran before it. Resetting the
  accuracy draws a new table key, so cosmolike rebuilds its tables on
  every call (a fraction of a second); the CAMB run is the expensive
  part, and this module keeps the last one (_camb_cosmology), so
  repeated calls at one cosmology cost one CAMB run.
- The project fiducial point lives in this module as plain constants
  (DES_CL_LNLAMBDA0, ...): Table I of arXiv 2503.13631, the point of
  the synthetic data vector (scripts/make_synthetic_data.py). A
  notebook overrides any of them per call
  (nw.N_cluster(MOR=[4.3, 0.943, 0.15, 0.207])).
- The values that mirror the likelihood yaml (the lmax of the internal
  C_ell tables, the integration accuracy, the cluster model switches)
  live in _CONFIG and are set once per notebook with configure(),
  before init_cosmolike.

Array layouts (the cluster wrappers of cosmo2D_wrapper_cluster.cpp):
the first axis is the angular bin or the multipole, the others are the
bins in the order (richness bin, cluster z bin, source or lens bin);
w_cc carries (richness bin 1, richness bin 2, cluster z bin).
cluster_blocks and data_cluster_blocks return the cluster blocks of
a joint vector (a theory vector, the data, its errors) in these same
layouts, so a model and the data go to the plotting functions
(cosmolike_notebook_utils.plot_datavectors_cluster) side by side. The
3x2pt blocks of 6x2pt + N (xi, gamma_t, w_theta) come back in the
layouts of the galaxy plotting functions (cnu.plot_xi, ...).

Every wrapper accepts the same accuracy arguments and applies the
house folds: CLAccuracyBoost multiplies by AccuracyBoost, the
integration accuracy grows as |3 (CLAccuracyBoost - 1)|, and the C_ell
table reaches lmax + 20000 (CLAccuracyBoost - 1).
"""

import os
import sys

import numpy as np
from getdist import IniFile

# the shared notebook utilities live in cosmolike_core; the compiled
# interface is on the path already (each project's interface/
# directory is part of the Cocoa PYTHONPATH)
sys.path.insert(0, os.environ["ROOTDIR"] + "/external_modules/code/cosmolike_core")
import cosmolike_notebook_utils as cnu
import cosmolike_des_cluster_interface as ci


# ----------------------------------------------------------------------
# Project fiducial point: Table I of arXiv 2503.13631, the evaluate
# override of EXAMPLE_EVALUATE1/2.yaml and the point of the synthetic
# data (FIDUCIAL_* of scripts/make_synthetic_data.py)
# ----------------------------------------------------------------------
As_1e9 = 2.19
ns = 0.96859
H0 = 69.0
omegab = 0.048
omegam = 0.3
# Omega_nu h^2 = 0.00083, one massive eigenstate
mnu = 0.00083 * 94.0708 / (3.046 / 3.0) ** 0.75
w = -1.0
w0pwa = -1.0

# sources (4 bins): photo-z shift, shear calibration, NLA (a1, eta1)
DES_DZ_S1 = 0.034
DES_DZ_S2 = 0.028
DES_DZ_S3 = 0.011
DES_DZ_S4 = -0.010
DES_M1 = 0.0
DES_M2 = 0.0
DES_M3 = 0.0
DES_M4 = 0.0
DES_A1_1 = 0.0
DES_A1_2 = 0.0

# MagLim lenses (6 bins): photo-z shift and stretch, linear bias,
# magnification (fixed), point mass
DES_DZ_L1 = 0.005
DES_DZ_L2 = 0.003
DES_DZ_L3 = 0.001
DES_DZ_L4 = -0.002
DES_DZ_L5 = 0.001
DES_DZ_L6 = 0.008
DES_DZ2_L1 = 1.0
DES_DZ2_L2 = 1.0
DES_DZ2_L3 = 1.0
DES_DZ2_L4 = 1.0
DES_DZ2_L5 = 1.0
DES_DZ2_L6 = 1.0
DES_B1_1 = 1.42
DES_B1_2 = 1.66
DES_B1_3 = 1.70
DES_B1_4 = 1.62
DES_B1_5 = 1.78
DES_B1_6 = 1.75
DES_BMAG_1 = -1.57
DES_BMAG_2 = -1.70
DES_BMAG_3 = -0.25
DES_BMAG_4 = 1.50
DES_BMAG_5 = 2.22
DES_BMAG_6 = 2.80
DES_PM1 = 0.0
DES_PM2 = 0.0
DES_PM3 = 0.0
DES_PM4 = 0.0
DES_PM5 = 0.0
DES_PM6 = 0.0

# clusters: mass-observable relation (eqs 18-19) and selection bias
# (eq 23; DES_CL_BSZ is the power of (1 + zbar)/1.45, 0 in the paper)
DES_CL_LNLAMBDA0 = 4.26
DES_CL_A_LAMBDA = 0.943
DES_CL_SIGMA_INT = 0.15
DES_CL_B_LAMBDA = 0.207
DES_CL_BS1 = 1.1
DES_CL_BS2 = 0.2
DES_CL_R0 = 30.0
DES_CL_BSZ = 0.0

# default nuisance vectors built from the constants above; wrappers
# take None and fall back to these, so a call overrides one vector
# without retyping the rest. The entries a model does not use are the
# values the likelihood sends (params_source_y6.yaml,
# params_lens_maglim.yaml): NLA reads A1 only.
A1_FID = [DES_A1_1, DES_A1_2, 0, 0]
A2_FID = [0, 0, 0, 0]
BTA_FID = [1.0, 0, 0, 0]
SHEAR_PHOTOZ_FID = [DES_DZ_S1, DES_DZ_S2, DES_DZ_S3, DES_DZ_S4]
M_FID = [DES_M1, DES_M2, DES_M3, DES_M4]
LENS_PHOTOZ_FID = [DES_DZ_L1, DES_DZ_L2, DES_DZ_L3,
                   DES_DZ_L4, DES_DZ_L5, DES_DZ_L6]
LENS_STRETCH_FID = [DES_DZ2_L1, DES_DZ2_L2, DES_DZ2_L3,
                    DES_DZ2_L4, DES_DZ2_L5, DES_DZ2_L6]
B1_FID = [DES_B1_1, DES_B1_2, DES_B1_3, DES_B1_4, DES_B1_5, DES_B1_6]
BMAG_FID = [DES_BMAG_1, DES_BMAG_2, DES_BMAG_3,
            DES_BMAG_4, DES_BMAG_5, DES_BMAG_6]
ZEROS6 = [0, 0, 0, 0, 0, 0]
PM_FID = [DES_PM1, DES_PM2, DES_PM3, DES_PM4, DES_PM5, DES_PM6]
# cluster.mor order: ln lambda_0, A_lambda, sigma_int, B_lambda
MOR_FID = [DES_CL_LNLAMBDA0, DES_CL_A_LAMBDA,
           DES_CL_SIGMA_INT, DES_CL_B_LAMBDA]
# cluster.selection order (Y6 model): b_s1, b_s2, r_0 [Mpc/h], s3
SELECTION_FID = [DES_CL_BS1, DES_CL_BS2, DES_CL_R0, DES_CL_BSZ]

# c/H0 in Mpc/h: cosmolike works with distances in c/H0, so k [h/Mpc]
# = k [code] / COVERH0 and P [(Mpc/h)^3] = P [code] * COVERH0^3
COVERH0 = 2997.92458

# ----------------------------------------------------------------------
# Per-notebook configuration
# ----------------------------------------------------------------------
# The options of likelihood/combo_6x2pt_N.yaml (same defaults), plus
# where the dataset lives. Set them with configure() BEFORE
# init_cosmolike: the init sequence reads them.
_CONFIG = {
    "lmax": 75000,              # base of the internal C_ell tables
    "accuracyboost": 1.0,
    "integration_accuracy": 1,  # 1 for the Y6 inputs (see the yaml)
    "non_linear_emul": 2,       # 1 = EuclidEmulator2, 2 = halofit
    "path": "../../external_modules/data/des_cluster",
    "data_file": "des_cluster_y6_6x2ptN.dataset",
    "IA_model": 0,              # NLA (cluster lensing has no TATT)
    "IA_redshift_evolution": 3,
    "IA_code": 0,               # 0 = C FASTPT (NLA always uses 0)
    # [b1, b2, bs2, b3, bmag] evolution flags (see bias.c)
    "bias_model": [0, 0, 0, 1, 0],
    # n(z) photo-z conventions: interpolation 0 = cspline, 1 = linear,
    # 2+ = Steffen monotone; z column 0 = Z_LOW, 1 = Z_MID
    "photoz_interpolation_type": 0,
    "photoz_zmid_convention": 0,
    # C-FAST-PT internal (convolution) grid / output grid; 1.0 = equal
    "internal_accuracyboost": 1.0,
    "adopt_limber_gs": 1,
    "adopt_limber_gg": 0,
    "include_HOD_GX": 0,
    "include_halo_IA": 0,
    # cluster model (structs_cluster.h; the likelihood yaml documents
    # every switch)
    "cluster_kernel_mode": 0,       # 0 = volume, 1 = abundance weighted
    "cluster_selection_model": 2,   # 0 = none, 1 = Y1, 2 = Y6 (eq 23)
    "cluster_ytransform": 1,        # 1 = Sigma = Y gamma_t (eq 15)
    "cluster_include_ia": 1,
    "cluster_magnification": -2.0,  # C_c (eq 28); 0 switches it off
    "cluster_hmf_alpha_mode": 0,    # 0 = Tinker alpha fixed at 0.368
    "cluster_adopt_limber_cc": 1,   # w_cc: 1 = Limber (the only option)
    "cluster_adopt_limber_cg": 1,   # w_cg: 1 = Limber (the only option)
}

# filled by init_cosmolike from the .dataset file: the angular binning
# and the bin edges of the loaded data (get_datavector needs the
# binning back after a wrapper re-binned)
_DATASET = {}

# the last CAMB run: (arguments, tuple of cnu.get_camb_cosmology)
_CAMB_CACHE = {"key": None, "value": None}


def configure(**overrides):
    """Sets this notebook's yaml-mirroring values, once per notebook.

    Every keyword must already exist in _CONFIG; an unknown name is
    almost always a typo, so it raises instead of being stored
    silently. Call it before init_cosmolike (the init sequence reads
    the model switches).

    Arguments:
      overrides = keyword form of any _CONFIG entry, e.g.
                  configure(cluster_ytransform=0).

    Returns:
      nothing; later wrapper calls read the stored values.

    Raises:
      KeyError naming the unknown keyword and the valid names.
    """
    for name, value in overrides.items():
        if name not in _CONFIG:
            raise KeyError(
                f"configure() got unknown option '{name}'; valid options: "
                + ", ".join(sorted(_CONFIG)))
        _CONFIG[name] = value


def _ini_list(ini, key, tp):
    """Comma- (or space-) separated dataset entry as a list of tp."""
    return [tp(x) for x in ini.string(key).replace(",", " ").split()]


def _vector(x):
    """A fresh float64 copy of x for the compiled interface.

    carma borrows the buffer of a numpy array handed to an armadillo
    argument and refuses one it cannot own (a slice such as z[::2], a
    read-only array); a copy is always accepted.
    """
    return np.array(x, dtype=np.float64)


def init_cosmolike(CLprobe="6x2pt_N", with_data=False, lmax=None):
    """One-time interface setup, in the order the likelihood runs it.

    The init chain of likelihood/_cosmolike_prototype_base.py
    (initialize, then init_cluster_related) for the cluster combos:
    probes, binning, photo-z and Limber conventions, accuracy, the
    lens and source n(z), then the cluster part (survey area, model
    switches, richness bins, selection kernels <phi_i|z>, pairs), the
    IA model and the galaxy-bias model. The chi2 machinery (mask, data
    vector, covariance of the joint vector) only loads when asked: the
    covariance file is large, and the plotting-only notebooks never
    need it.

    Arguments:
      CLprobe   = a probe name of ci.init_probes_cluster: "6x2pt_N"
                  (every block), "4x2pt_N" (gg + cg + N + cc + cs),
                  "N", "N_cc", "N_cs", "cs", "cc", "cg", "3x2pt". It
                  fixes which blocks enter the masked data vector; the
                  observable wrappers work with any of them.
      with_data = True also loads covariance, mask, and data vector,
                  which get_datavector and get_chi2 need.
      lmax      = base lmax of the internal C_ell tables, or None
                  for the configure()d value.

    Returns:
      the parsed IniFile, for notebooks that read extra entries.

    Side effects:
      replaces the compiled interface's global state and records the
      dataset's binning and bin edges in this module (_DATASET).
    """
    if lmax is None:
        lmax = _CONFIG["lmax"]
    ini = IniFile(os.path.normpath(
        os.path.join(_CONFIG["path"], _CONFIG["data_file"])))

    _DATASET.clear()
    _DATASET.update(
        ntheta=int(ini.int("n_theta")),
        theta_min_arcmin=ini.float("theta_min_arcmin"),
        theta_max_arcmin=ini.float("theta_max_arcmin"),
        lens_ntomo=int(ini.int("lens_ntomo")),
        source_ntomo=int(ini.int("source_ntomo")),
        cluster_ntomo=int(ini.int("cluster_ntomo")),
        cluster_zbin_edges=np.array(_ini_list(ini, "cluster_zbin_edges", float)),
        richness_edges=np.array(_ini_list(ini, "richness_edges", float)),
        survey_area_deg2=ini.float("survey_area_deg2"),
        cg_lens_bins=_ini_list(ini, "cg_lens_bins", int))

    ci.initial_setup()
    ci.reset_cluster()  # cluster defaults + fresh cluster cache keys
    ci.init_probes_cluster(possible_probes=CLprobe)
    ci.init_binning(_DATASET["ntheta"],
                    _DATASET["theta_min_arcmin"],
                    _DATASET["theta_max_arcmin"])
    ci.set_log_level_info()
    ci.init_photoz_conventions(
        interpolation_type=int(_CONFIG["photoz_interpolation_type"]),
        zmid_convention=int(_CONFIG["photoz_zmid_convention"]))
    ci.init_fpt_internal_boost(
        internal_boost=float(_CONFIG["internal_accuracyboost"]))
    ci.init_adopt_limber_gs(adopt_limber_gs=int(_CONFIG["adopt_limber_gs"]))
    ci.init_adopt_limber_gg(adopt_limber_gg=int(_CONFIG["adopt_limber_gg"]))
    ci.init_include_HOD_GX(include_HOD_GX=int(_CONFIG["include_HOD_GX"]))
    ci.init_include_halo_IA(include_halo_IA=int(_CONFIG["include_halo_IA"]))

    ci.init_ntable_lmax(lmax=int(lmax))
    ci.init_accuracy_boost(
        accuracy_boost=float(_CONFIG["accuracyboost"]),
        integration_accuracy=int(_CONFIG["integration_accuracy"]))
    ci.init_cosmo_runmode(is_linear=False)
    ci.init_redshift_distributions_from_files(
        lens_multihisto_file=ini.relativeFileName('nz_lens_file'),
        lens_ntomo=_DATASET["lens_ntomo"],
        source_multihisto_file=ini.relativeFileName('nz_source_file'),
        source_ntomo=_DATASET["source_ntomo"])

    # cluster model, richness bins, selection kernels and pairs: they
    # fix the block sizes of the joint vector (init_cluster_related of
    # the likelihood)
    # survey area (deg^2) of the counts, eq (16); sigma_e is read only
    # by the covariance code
    ci.init_survey_parameters(surveyname="DES",
                              area=_DATASET["survey_area_deg2"],
                              sigma_e=ini.float("sigma_e", 0.0))
    ci.init_cluster_model(
        mor_model=0,  # lognormal, eqs (18)-(19)
        kernel_mode=int(_CONFIG["cluster_kernel_mode"]),
        selection_model=int(_CONFIG["cluster_selection_model"]),
        ytransform=int(_CONFIG["cluster_ytransform"]),
        include_ia=int(_CONFIG["cluster_include_ia"]),
        magnification=float(_CONFIG["cluster_magnification"]))
    ci.init_cluster_hmf_alpha_mode(
        hmf_alpha_mode=int(_CONFIG["cluster_hmf_alpha_mode"]))
    ci.init_cluster_adopt_limber(
        adopt_limber_cc=int(_CONFIG["cluster_adopt_limber_cc"]),
        adopt_limber_cg=int(_CONFIG["cluster_adopt_limber_cg"]))
    ci.init_cluster_richness_bins(
        lambda_min=_DATASET["richness_edges"][:-1].copy(),
        lambda_max=_DATASET["richness_edges"][1:].copy())
    # columns: z, then <phi_i|z> of each cluster redshift bin
    nz_cluster = np.loadtxt(ini.relativeFileName('nz_cluster_file'))
    ci.set_cluster_zdist(
        nofz=nz_cluster,
        zbin_min=_DATASET["cluster_zbin_edges"][:-1].copy(),
        zbin_max=_DATASET["cluster_zbin_edges"][1:].copy())
    ci.init_cluster_pairs(cg_lens_bin=_DATASET["cg_lens_bins"])

    if with_data:
        ci.init_data_cluster(ini.relativeFileName('cov_file'),
                             ini.relativeFileName('mask_file'),
                             ini.relativeFileName('data_file'))

    ci.init_IA(ia_model=int(_CONFIG["IA_model"]),
               ia_redshift_evolution=int(_CONFIG["IA_redshift_evolution"]),
               ia_code=int(_CONFIG["IA_code"]))
    ci.init_bias(bias_model=_CONFIG["bias_model"])
    return ini


def dataset_info():
    """Binning and bin edges of the dataset init_cosmolike loaded.

    Returns:
      dict (a copy) with ntheta, theta_min_arcmin, theta_max_arcmin,
      lens_ntomo, source_ntomo, cluster_ntomo, cluster_zbin_edges,
      richness_edges, survey_area_deg2 and cg_lens_bins (the lens bin
      paired with each cluster redshift bin in w_cg, -1 = none).
    """
    return dict(_DATASET)


def _camb_cosmology(**kwargs):
    """cnu.get_camb_cosmology with a one-entry memory.

    Every wrapper rebuilds the full interface state, so a notebook
    that reads ten observables at one point would otherwise run CAMB
    ten times.

    Arguments:
      kwargs = the arguments of cnu.get_camb_cosmology.

    Returns:
      its tuple (log10k_2D, z_2D, lnPL, lnPNL, G, z_G, z_1D, chi).
    """
    key = tuple(sorted(kwargs.items()))
    if _CAMB_CACHE["key"] != key:
        _CAMB_CACHE["value"] = cnu.get_camb_cosmology(**kwargs)
        _CAMB_CACHE["key"] = key
    return _CAMB_CACHE["value"]


def _set_state(omegam, omegab, H0, ns, As_1e9, w, w0pwa, mnu,
               AccuracyBoost, kmax, k_per_logint, CAMBAccuracyBoost,
               CLAccuracyBoost, CLIntegrationAccuracy, non_linear_emul,
               binning=None, M=None, shear_photoz_bias=None,
               A1=None, A2=None, BTA=None,
               lens_photoz_bias=None, lens_photoz_stretch=None,
               B1=None, B2=None, B_MAG=None, B3nl=None, BK=None, PM=None,
               MOR=None, SEL=None):
    """Runs CAMB and pushes one complete state into the interface.

    This is the body every wrapper shares: the accuracy folds and
    lookup tables, the angular binning, the cosmology (power spectra,
    growth, distances from one CAMB run), and the nuisance parameters
    in the order the likelihood sets them (lens, source, cluster).
    None vectors fall back to the module fiducials, so every call
    sets the complete state.

    The growth factor goes in on the dense 1D redshift grid, as the
    likelihood sends it (z_G of ci.set_cosmology): cosmolike reads G
    linearly in z, and the cluster abundance amplifies a growth error
    5-15x, so the coarse grid of the power spectra is not enough.
    cnu.get_camb_cosmology returns G on that dense grid, with its z
    nodes (z_growth).

    Arguments:
      omegam ... non_linear_emul = the cosmology and accuracy
                 arguments, forwarded to cnu.get_camb_cosmology
                 (kmax in 1/Mpc; see its docstring for the grids).
      binning  = (ntheta, theta_min_arcmin, theta_max_arcmin) of the
                 real-space statistics, or None for the dataset's.
      M, shear_photoz_bias, A1, A2, BTA = source nuisance vectors.
      lens_photoz_bias, lens_photoz_stretch, B1, B2, B_MAG, B3nl, BK,
                 PM = lens nuisance vectors.
      MOR      = mass-observable relation {ln lambda_0, A_lambda,
                 sigma_int, B_lambda}.
      SEL      = selection bias {b_s1, b_s2, r_0 [Mpc/h], s3}.

    Returns:
      nothing; the interface state is the result.
    """
    (log10k_interp_2D, z_interp_2D, lnPL, lnPNL,
     G_growth, z_growth, z_interp_1D, chi) = _camb_cosmology(
        omegam=omegam, omegab=omegab, H0=H0, ns=ns, As_1e9=As_1e9,
        w=w, w0pwa=w0pwa, mnu=mnu, AccuracyBoost=AccuracyBoost,
        kmax=kmax, k_per_logint=k_per_logint,
        CAMBAccuracyBoost=CAMBAccuracyBoost,
        CLAccuracyBoost=CLAccuracyBoost,
        non_linear_emul=non_linear_emul)
    # CAMB (and other libraries) may call omp_set_num_threads: restore
    # the thread count of cosmolike's parallel regions, as the
    # likelihood does before every interface call
    ci.set_omp_threads(int(os.environ.get("OMP_NUM_THREADS", 1)))

    # the house accuracy folds: the overall boost multiplies the
    # cosmolike boost, and the integration accuracy and the C_ell
    # table length grow with it
    CLAccuracyBoost = CLAccuracyBoost * AccuracyBoost
    CLIntegrationAccuracy = max(
        0, CLIntegrationAccuracy + abs(3*(CLAccuracyBoost - 1.0)))
    ci.init_ntable_lmax(int(_CONFIG["lmax"] + 20000*(CLAccuracyBoost - 1)))
    ci.init_accuracy_boost(CLAccuracyBoost, int(CLIntegrationAccuracy))
    ci.init_photoz_conventions(
        int(_CONFIG["photoz_interpolation_type"]),
        int(_CONFIG["photoz_zmid_convention"]))
    ci.init_fpt_internal_boost(
        float(_CONFIG["internal_accuracyboost"]))
    if binning is None:
        binning = (_DATASET["ntheta"], _DATASET["theta_min_arcmin"],
                   _DATASET["theta_max_arcmin"])
    # cosmolike caches the bin-averaged Legendre kernels by the number
    # of angular bins and the table key, not by the angular range: the
    # new key drawn by init_accuracy_boost above is what makes a new
    # range with the same number of bins take effect
    ci.init_binning(int(binning[0]), binning[1], binning[2])
    ci.init_bias(bias_model=_CONFIG["bias_model"])

    ci.set_cosmology(omegam=omegam,
                     omegab=omegab,
                     H0=H0,
                     log10k_2D=log10k_interp_2D,
                     z_2D=z_interp_2D,
                     lnP_linear=lnPL,
                     lnP_nonlinear=lnPNL,
                     G=G_growth,
                     z_G=z_growth,
                     z_1D=z_interp_1D,
                     chi=chi)

    # lens, source and cluster nuisances, in the likelihood's order
    ci.set_point_mass(PMV=PM_FID if PM is None else PM)
    ci.set_nuisance_bias(B1=B1_FID if B1 is None else B1,
                         B2=ZEROS6 if B2 is None else B2,
                         B_MAG=BMAG_FID if B_MAG is None else B_MAG,
                         B3nl=ZEROS6 if B3nl is None else B3nl,
                         BK=ZEROS6 if BK is None else BK)
    ci.set_nuisance_clustering_photoz(
        bias=LENS_PHOTOZ_FID if lens_photoz_bias is None else lens_photoz_bias,
        stretch=(LENS_STRETCH_FID if lens_photoz_stretch is None
                 else lens_photoz_stretch))
    ci.set_nuisance_shear_calib(M=M_FID if M is None else M)
    ci.set_nuisance_shear_photoz(
        bias=SHEAR_PHOTOZ_FID if shear_photoz_bias is None
        else shear_photoz_bias)
    ci.set_nuisance_ia(A1=A1_FID if A1 is None else A1,
                       A2=A2_FID if A2 is None else A2,
                       B_TA=BTA_FID if BTA is None else BTA)
    ci.set_nuisance_cluster_mor(MOR=MOR_FID if MOR is None else MOR)
    ci.set_nuisance_cluster_selection(
        SEL=SELECTION_FID if SEL is None else SEL)
    ci.reset_bary_struct()


def _state(ntheta=None, theta_min_arcmin=None, theta_max_arcmin=None,
           omegam=omegam, omegab=omegab, H0=H0, ns=ns, As_1e9=As_1e9,
           w=w, w0pwa=w0pwa, mnu=mnu,
           AccuracyBoost=1.0, kmax=10.0, k_per_logint=12,
           CAMBAccuracyBoost=1.0, CLAccuracyBoost=None,
           CLIntegrationAccuracy=None, non_linear_emul=None, **nuisance):
    """The keyword interface every public wrapper shares.

    Fills the defaults that live in _CONFIG or in the dataset and
    calls _set_state.

    Arguments:
      ntheta, theta_min_arcmin, theta_max_arcmin = angular binning of
          the real-space statistics; None for the dataset's. It can
          change between calls without restarting the kernel.
      omegam, omegab, H0, ns, As_1e9, w, w0pwa, mnu = cosmology
          (module fiducials by default).
      AccuracyBoost, kmax, k_per_logint, CAMBAccuracyBoost,
      CLAccuracyBoost, CLIntegrationAccuracy = accuracy knobs; kmax
          and k_per_logint default to the camb block of the example
          yamls, the cosmolike ones to the configure()d likelihood
          values.
      non_linear_emul = 1 EuclidEmulator2, 2 halofit; None for the
          configure()d value.
      nuisance = any nuisance vector of _set_state (M, A1, B1, MOR,
          SEL, ...).

    Returns:
      nothing; the interface state is the result.
    """
    if non_linear_emul is None:
        non_linear_emul = _CONFIG["non_linear_emul"]
    if CLAccuracyBoost is None:
        CLAccuracyBoost = _CONFIG["accuracyboost"]
    if CLIntegrationAccuracy is None:
        CLIntegrationAccuracy = _CONFIG["integration_accuracy"]
    binning = (_DATASET["ntheta"] if ntheta is None else ntheta,
               (_DATASET["theta_min_arcmin"] if theta_min_arcmin is None
                else theta_min_arcmin),
               (_DATASET["theta_max_arcmin"] if theta_max_arcmin is None
                else theta_max_arcmin))
    _set_state(omegam, omegab, H0, ns, As_1e9, w, w0pwa, mnu,
               AccuracyBoost, kmax, k_per_logint, CAMBAccuracyBoost,
               CLAccuracyBoost, CLIntegrationAccuracy, non_linear_emul,
               binning=binning, **nuisance)


# ----------------------------------------------------------------------
# Cluster observables (real space and counts)
# ----------------------------------------------------------------------
def N_cluster(**kwargs):
    """Expected cluster counts per richness and redshift bin (eq 16).

    Arguments:
      kwargs = cosmology, accuracy and nuisance arguments of _state.

    Returns:
      2D array (n_richness, n_cluster_z): the counts in the survey
      area of the dataset.
    """
    _state(**kwargs)
    return np.array(ci.N_cluster_tomo_bins())


def gamma_t_cluster(**kwargs):
    """Cluster tangential shear gamma_t on the theta grid.

    This is gamma_t BEFORE the Y transform, the selection bias and the
    shear calibration, which the likelihood applies on the data vector
    (sigma_cluster returns that form).

    Arguments:
      kwargs = binning, cosmology, accuracy and nuisance arguments of
               _state.

    Returns:
      (theta, gammat): theta in arcmin, gammat a 4D array
      (n_theta, n_richness, n_cluster_z, n_source).
    """
    _state(**kwargs)
    return (np.array(ci.get_binning_real_space()),
            np.array(ci.w_gammat_cluster_tomo_bins()))


def sigma_cluster(**kwargs):
    """Cluster lensing as the data vector holds it, without the mask.

    Sigma = T gamma_t (the Y transform of eq 15; gamma_t itself when
    cluster_ytransform = 0), times the selection bias of eq (23) and
    the shear calibration (1 + m). With the Y transform the last
    angular bin is identically zero (the likelihood masks it), and the
    transform needs at least five angular bins.

    Arguments:
      kwargs = binning, cosmology, accuracy and nuisance arguments of
               _state.

    Returns:
      (theta, sigma): theta in arcmin, sigma a 4D array
      (n_theta, n_richness, n_cluster_z, n_source).
    """
    _state(**kwargs)
    return (np.array(ci.get_binning_real_space()),
            np.array(ci.w_sigma_cluster_tomo_bins()))


def w_cc(selection_bias=False, **kwargs):
    """Cluster-cluster angular correlation w_cc on the theta grid.

    Arguments:
      selection_bias = True multiplies by B(theta)^2, the selection
               bias of eq (23) as the data vector carries it (one
               factor per cluster leg); False returns the model before
               it.
      kwargs = binning, cosmology, accuracy and nuisance arguments of
               _state.

    Returns:
      (theta, wcc): theta in arcmin, wcc a 4D array (n_theta,
      n_richness, n_richness, n_cluster_z), symmetric in the two
      richness bins.
    """
    _state(**kwargs)
    wcc = np.array(ci.w_cc_tomo_bins(
        limber=int(_CONFIG["cluster_adopt_limber_cc"])))
    if selection_bias:
        # B is (cluster z bin, theta): move theta to the front and
        # let it broadcast over the two richness axes
        B = np.array(ci.get_cluster_selection_factor())
        wcc = wcc * (B.T ** 2)[:, None, None, :]
    return (np.array(ci.get_binning_real_space()), wcc)


def w_cg(selection_bias=False, **kwargs):
    """Cluster x galaxy angular correlation w_cg on the theta grid.

    Arguments:
      selection_bias = True multiplies by B(theta), the selection bias
               of eq (23) as the data vector carries it; False returns
               the model before it.
      kwargs = binning, cosmology, accuracy and nuisance arguments of
               _state.

    Returns:
      (theta, wcg): theta in arcmin, wcg a 4D array (n_theta,
      n_richness, n_cluster_z, n_lens); only the pairs of the
      dataset's cg_lens_bins are filled (ci.get_cg_redshift_bins
      lists them), the others are zero.
    """
    _state(**kwargs)
    wcg = np.array(ci.w_cg_tomo_bins(
        limber=int(_CONFIG["cluster_adopt_limber_cg"])))
    if selection_bias:
        B = np.array(ci.get_cluster_selection_factor())
        wcg = wcg * B.T[:, None, :, None]
    return (np.array(ci.get_binning_real_space()), wcg)


# ----------------------------------------------------------------------
# Cluster observables (Fourier space, Limber)
# ----------------------------------------------------------------------
def C_cs_tomo_limber(ell, **kwargs):
    """Cluster-lensing spectra C_cs at multipoles ell (exact Limber).

    Arguments:
      ell    = 1D float array of multipoles.
      kwargs = cosmology, accuracy and nuisance arguments of _state.

    Returns:
      4D array (n_ell, n_richness, n_cluster_z, n_source).
    """
    _state(**kwargs)
    return np.array(ci.C_cs_tomo_limber_bins(l=_vector(ell)))


def C_cc_tomo_limber(ell, **kwargs):
    """Cluster-clustering spectra C_cc at multipoles ell (exact Limber).

    Arguments:
      ell    = 1D float array of multipoles.
      kwargs = cosmology, accuracy and nuisance arguments of _state.

    Returns:
      4D array (n_ell, n_richness, n_richness, n_cluster_z).
    """
    _state(**kwargs)
    return np.array(ci.C_cc_tomo_limber_bins(l=_vector(ell)))


def C_cg_tomo_limber(ell, **kwargs):
    """Cluster x galaxy spectra C_cg at multipoles ell (exact Limber).

    Arguments:
      ell    = 1D float array of multipoles.
      kwargs = cosmology, accuracy and nuisance arguments of _state.

    Returns:
      4D array (n_ell, n_richness, n_cluster_z, n_lens); pairs outside
      the dataset's cg_lens_bins are zero.
    """
    _state(**kwargs)
    return np.array(ci.C_cg_tomo_limber_bins(l=_vector(ell)))


# ----------------------------------------------------------------------
# Galaxy observables: the 3x2pt blocks of 6x2pt + N (real space)
# ----------------------------------------------------------------------
# The same state, read through the galaxy functions of the interface.
# The return layouts are those of the galaxy plotting functions
# (cnu.plot_xi, cnu.plot_gammat_tomo_limber, cnu.plot_wtheta_tomo).
def xi(**kwargs):
    """Real-space shear correlations xi_plus/minus on the theta grid.

    Arguments:
      kwargs = binning, cosmology, accuracy and nuisance arguments of
               _state.

    Returns:
      (theta, xi_plus, xi_minus): theta in arcmin, the xi 3D arrays
      (n_theta, n_source, n_source).
    """
    _state(**kwargs)
    (xip, xim) = ci.xi_pm_tomo()
    return (np.array(ci.get_binning_real_space()),
            np.array(xip), np.array(xim))


def gamma_t(**kwargs):
    """Galaxy-galaxy lensing gamma_t on the theta grid.

    Arguments:
      kwargs = binning, cosmology, accuracy and nuisance arguments of
               _state.

    Returns:
      (theta, gammat): theta in arcmin, gammat a 3D array
      (n_theta, n_lens, n_source).
    """
    _state(**kwargs)
    return (np.array(ci.get_binning_real_space()),
            np.array(ci.w_gammat_tomo()))


def w_theta(**kwargs):
    """Galaxy clustering w(theta) on the theta grid.

    Arguments:
      kwargs = binning, cosmology, accuracy and nuisance arguments of
               _state.

    Returns:
      (theta, wtheta): theta in arcmin, wtheta a 3D array
      (n_theta, n_lens, n_lens); the auto-correlations are on the
      diagonal.
    """
    _state(**kwargs)
    return (np.array(ci.get_binning_real_space()),
            np.array(ci.w_gg_tomo()))


# ----------------------------------------------------------------------
# Halo-model ingredients and radial kernels
# ----------------------------------------------------------------------
# These take redshifts, masses in Msun/h and wavenumbers in h/Mpc and
# return number densities in (h/Mpc)^3 and power spectra in (Mpc/h)^3;
# the compiled functions work with the scale factor and with c/H0 as
# the unit of length (COVERH0 converts).
def prob_richness_bin_given_m(M, z, **kwargs):
    """P(richness bin | M, z) of the mass-observable relation.

    Arguments:
      M      = 1D array of halo masses in Msun/h (M200m).
      z      = 1D array of redshifts.
      kwargs = cosmology, accuracy and nuisance arguments of _state
               (only MOR matters: the relation needs no cosmology).

    Returns:
      3D array (n_M, n_z, n_richness): the probability that a halo of
      mass M at redshift z has observed richness in each bin.
    """
    _state(**kwargs)
    return np.array(ci.prob_richness_bin_given_m(
        lnM=np.log(_vector(M)), z=_vector(z)))


def ncl_richness(z, **kwargs):
    """Comoving number density of the clusters of each richness bin.

    Arguments:
      z      = 1D array of redshifts.
      kwargs = cosmology, accuracy and nuisance arguments of _state.

    Returns:
      2D array (n_z, n_richness) in (h/Mpc)^3; 0 outside the redshift
      range of the cluster bins (the tables cover their support).
    """
    _state(**kwargs)
    a = 1.0/(1.0 + _vector(z))
    return np.array(ci.ncl_richness(a=a))/COVERH0**3


def bcl_richness(z, **kwargs):
    """Richness-weighted linear bias of each richness bin (eq 21).

    Arguments:
      z      = 1D array of redshifts.
      kwargs = cosmology, accuracy and nuisance arguments of _state.

    Returns:
      2D array (n_z, n_richness); 0 outside the redshift range of the
      cluster bins.
    """
    _state(**kwargs)
    a = 1.0/(1.0 + _vector(z))
    return np.array(ci.bcl_richness(a=a))


def pcm_1h_richness(k, z, **kwargs):
    """One-halo cluster-matter power spectrum of each richness bin.

    Arguments:
      k      = 1D array of wavenumbers in h/Mpc.
      z      = 1D array of redshifts.
      kwargs = cosmology, accuracy and nuisance arguments of _state.

    Returns:
      3D array (n_k, n_z, n_richness) in (Mpc/h)^3; 0 outside the
      redshift range of the cluster bins.
    """
    _state(**kwargs)
    a = 1.0/(1.0 + _vector(z))
    k_code = _vector(k)*COVERH0
    return np.array(ci.pcm_1h_richness(k=k_code, a=a))*COVERH0**3


def phi_cluster(z):
    """Selection kernels <phi_i|z> of the cluster redshift bins.

    The probability that a cluster at true redshift z lands in each
    z_lambda bin: the table of the dataset's nz_cluster_file, no
    cosmology (only init_cosmolike is needed).

    Arguments:
      z = 1D array of true redshifts.

    Returns:
      2D array (n_z, n_cluster_z).
    """
    return np.array(ci.phi_cluster(z=_vector(z)))


def nz_cluster(z, **kwargs):
    """Normalized true-redshift distribution of the cluster bins.

    n(z) ~ dV/dz <phi_i|z> (times n_richness(z) for the
    abundance-weighted kernel), per unit z.

    Arguments:
      z      = 1D array of redshifts.
      kwargs = cosmology, accuracy and nuisance arguments of _state.

    Returns:
      3D array (n_z, n_cluster_z, n_richness); the same for every
      richness bin with the volume kernel (cluster_kernel_mode = 0).
    """
    _state(**kwargs)
    return np.array(ci.nz_cluster(z=_vector(z)))


def W_cluster(z, **kwargs):
    """Radial kernels of the cluster bins at redshifts z > 0.

    Arguments:
      z      = 1D array of redshifts, z > 0.
      kwargs = cosmology, accuracy and nuisance arguments of _state.

    Returns:
      (W, W_mag, g): three 3D arrays (n_z, n_cluster_z, n_richness).
      W is the density kernel n(z) H/H0 per unit comoving distance
      (in c/H0), W_mag the magnification kernel 1.5 Omega_m f_K/a g
      (without the coefficient C_c), g the lensing efficiency of the
      cluster distribution.
    """
    _state(**kwargs)
    a = 1.0/(1.0 + _vector(z))
    return (np.array(ci.W_cluster(a=a)),
            np.array(ci.W_mag_cluster(a=a)),
            np.array(ci.g_cluster(a=a)))


# ----------------------------------------------------------------------
# Data vector and chi2
# ----------------------------------------------------------------------
def get_datavector(**kwargs):
    """Masked joint theory vector (masked entries are zero).

    Requires init_cosmolike(with_data=True). The layout is ss, gs, gg,
    cg, N, cc, cs (ci.compute_data_vector_cluster_sizes and _starts
    give the blocks); the binning is always the dataset's, the one of
    the mask.

    Arguments:
      kwargs = cosmology, accuracy and nuisance arguments of _state
               (binning arguments are not accepted).

    Returns:
      1D numpy array, the full data-vector length.
    """
    for name in ("ntheta", "theta_min_arcmin", "theta_max_arcmin"):
        if name in kwargs:
            raise TypeError(
                f"get_datavector() does not take '{name}': the data "
                "vector uses the binning of the dataset")
    _state(**kwargs)
    return np.array(ci.compute_data_vector_cluster_masked())


def get_chi2(**kwargs):
    """chi2 of the masked theory vector against the loaded data.

    Requires init_cosmolike(with_data=True). Keyword arguments are
    those of get_datavector. The synthetic data vector is the
    likelihood's model at the module fiducial, so chi2 is small there
    (not zero: this module's CAMB run is not the cobaya one).

    Returns:
      float chi2.
    """
    return ci.compute_chi2_cluster(get_datavector(**kwargs))


def cluster_blocks(vector):
    """The cluster blocks of a joint vector as bin-indexed arrays.

    The joint vector is flat (ss, gs, gg, cg, N, cc, cs); this puts
    its four cluster blocks back into the array layouts of the
    observable wrappers, so a data vector, its errors or a masked
    theory vector can be compared with N_cluster, sigma_cluster, w_cc
    and w_cg bin by bin. Requires init_cosmolike (the pair lists and
    the block positions come from the interface).

    Arguments:
      vector = 1D array of the full data-vector length (the layout of
               get_datavector).

    Returns:
      dict of arrays:
        "N"  (n_richness, n_cluster_z),
        "cs" (n_theta, n_richness, n_cluster_z, n_source),
        "cc" (n_theta, n_richness, n_richness, n_cluster_z), filled
             symmetrically in the two richness bins,
        "cg" (n_theta, n_richness, n_cluster_z, n_lens).
      Bin combinations the data vector does not hold are NaN.
    """
    vector = np.asarray(vector, dtype=np.float64)
    sizes = np.array(ci.compute_data_vector_cluster_sizes()).astype(int)
    starts = np.array(ci.compute_data_vector_cluster_starts()).astype(int)
    # block order of the joint vector: ss, gs, gg, cg, N, cc, cs
    block = {name: vector[starts[i]:starts[i] + sizes[i]]
             for i, name in enumerate(("ss", "gs", "gg", "cg", "N", "cc", "cs"))}

    ntheta = _DATASET["ntheta"]
    nrichness = len(_DATASET["richness_edges"]) - 1
    ncluster = _DATASET["cluster_ntomo"]
    nsource = _DATASET["source_ntomo"]
    nlens = _DATASET["lens_ntomo"]
    # row n of each list is one pair of the block, in block order
    cs_pairs = np.array(ci.get_cs_redshift_bins()).astype(int)
    cc_pairs = np.array(ci.get_cc_richness_bins()).astype(int)
    cg_pairs = np.array(ci.get_cg_redshift_bins()).astype(int)

    # N block: [cluster z bin][richness bin]
    N = block["N"].reshape(ncluster, nrichness).T.copy()

    # cs block: [(cluster z, source) pair][richness bin][theta]
    cs = np.full((ntheta, nrichness, ncluster, nsource), np.nan)
    rows = block["cs"].reshape(len(cs_pairs), nrichness, ntheta)
    for n, (ni, ns_) in enumerate(cs_pairs):
        cs[:, :, ni, ns_] = rows[n].T

    # cc block: [cluster z bin][richness pair (nl1 <= nl2)][theta]
    cc = np.full((ntheta, nrichness, nrichness, ncluster), np.nan)
    rows = block["cc"].reshape(ncluster, len(cc_pairs), ntheta)
    for n, (nl1, nl2) in enumerate(cc_pairs):
        cc[:, nl1, nl2, :] = rows[:, n, :].T
        cc[:, nl2, nl1, :] = rows[:, n, :].T

    # cg block: [(cluster z, lens) pair][richness bin][theta]
    cg = np.full((ntheta, nrichness, ncluster, nlens), np.nan)
    rows = block["cg"].reshape(len(cg_pairs), nrichness, ntheta)
    for n, (ni, ng) in enumerate(cg_pairs):
        cg[:, :, ni, ng] = rows[n].T

    return {"N": N, "cs": cs, "cc": cc, "cg": cg}


def data_cluster_blocks():
    """The cluster data and their 1-sigma errors, bin by bin.

    Requires init_cosmolike(with_data=True). The errors are the square
    roots of the covariance diagonal. Entries the mask removes (the
    scale cuts) are NaN in both, which is how the plotting functions
    of plot_datavectors_cluster skip them.

    Returns:
      (data, error): two dicts in the layout of cluster_blocks.
    """
    mask = np.array(ci.get_mask_cluster()).astype(bool)
    data = np.array(ci.get_dv_masked_cluster(), dtype=np.float64)
    error = np.sqrt(np.diag(np.array(ci.get_cov_masked_cluster())))
    data[~mask] = np.nan
    error[~mask] = np.nan
    return cluster_blocks(data), cluster_blocks(error)
