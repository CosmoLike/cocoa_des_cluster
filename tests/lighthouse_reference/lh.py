"""ctypes bindings to the prebuilt lighthouse cluster library (read-only use).

Library: lighthouse/lib/like_cluster_Buzzard_y3_2_fast_v37_IA_test4.so
Build flags (lighthouse Makefile target like_cluster_fullsky_y3_IA_selectionB):
    -DCLASS_V29 -DIAC -DBuzzard -DFullsky -DDESY3 -DSELECTIONB
Driver logic mirrors lighthouse/python/cosmolike_libs_real_mpp_cluster.py
(init_all_nuisance_param + theory_wrapper), without its emcee/MPI imports.

Every library call must happen in a fresh process per data vector: the C
code keeps static caches keyed on cosmology/nuisance only.
"""
import ctypes
import os

import numpy as np

LIGHTHOUSE = "/Users/vivianmiranda/data/COCOA/september2026/test/lighthouse"
SO = os.path.join(LIGHTHOUSE, "lib", "like_cluster_Buzzard_y3_2_fast_v37_IA_test4.so")
HERE = os.path.dirname(os.path.abspath(__file__))

lib = ctypes.CDLL(SO)

c_d = ctypes.c_double
c_i = ctypes.c_int
c_p = ctypes.c_char_p
c_b = ctypes.c_bool
c_vp = ctypes.c_void_p
Double10 = c_d * 10
Double20 = c_d * 20
PD = ctypes.POINTER(c_d)
PI = ctypes.POINTER(c_i)


class InputCosmologyParams(ctypes.Structure):
    # input_cosmo_params_mpp (cosmolike_core/theory/structs.c:369)
    _fields_ = [("omega_m", c_d), ("sigma_8", c_d), ("A_s", c_d), ("n_s", c_d),
                ("w0", c_d), ("wa", c_d), ("omega_b", c_d), ("omega_nuh2", c_d),
                ("h0", c_d), ("MGSigma", c_d), ("MGmu", c_d), ("log10Tagn", c_d)]


class InputNuisanceParams(ctypes.Structure):
    # input_nuisance_params_mpp (cosmolike_core/theory/structs.c:396)
    _fields_ = [("bias", Double10), ("bias2", Double10), ("lens_z_bias", Double10),
                ("lens_z_stretch", Double10), ("source_z_bias", Double10),
                ("shear_m", Double10), ("A_z", Double10), ("lens_z_u", Double20),
                ("source_z_u", Double20), ("MOR", Double10), ("selection", Double10),
                ("b_mag", Double10), ("pm", Double10)]


def _f(name, restype, argtypes):
    f = getattr(lib, name)
    f.restype = restype
    f.argtypes = argtypes
    return f


# ---- init / driver API (like_real_mpp.c, init_des_real.c, init_cluster.c)
init_cosmo_runmode = _f("init_cosmo_runmode", None, [c_p])
init_source_sample_mpp = _f("init_source_sample_mpp", None, [c_p, c_i])
init_lens_sample_mpp = _f("init_lens_sample_mpp", None, [c_p, c_i, PD, PD, c_d])
init_binning_mpp = _f("init_binning_mpp", None, [c_i, c_d, c_d])
init_cluster_N200 = _f("init_cluster_N200", None, [c_i, PD])
init_tomo_cluster = _f("init_tomo_cluster", None, [c_i, PD])
set_cluster_cg_Npowerspectra = _f("set_cluster_cg_Npowerspectra", None, [c_i, PI, PI])
init_survey_mpp = _f("init_survey_mpp", None, [c_p, c_d, c_d, PD, PD, c_p])
init_cluster_area_interpolation = _f("init_cluster_area_interpolation", None, [c_p, c_b])
init_cluster_photo_z = _f("init_cluster_photo_z", None, [c_p, c_b])
set_nonlinear = _f("set_nonlinear", None, [c_i])
init_probes_4x2ptN = _f("init_probes_4x2ptN", None, [c_p])
set_Ystatistics = _f("set_Ystatistics", None, [c_i])
init_mask = _f("init_mask", None, [c_p])
getNdata = _f("getNdata", c_i, [])
get_N_ggl = _f("get_N_ggl", c_i, [])
get_N_tomo_shear = _f("get_N_tomo_shear", c_i, [])
get_N_tomo_clustering = _f("get_N_tomo_clustering", c_i, [])
get_tomo_clustering_zmax = _f("get_tomo_clustering_zmax", c_d, [c_i])
get_tomo_cluster_zmax = _f("get_tomo_cluster_zmax", c_d, [c_i])
theory_wrapper = _f("theory_wrapper", None, [InputCosmologyParams, InputNuisanceParams, PD])
write_datavector_wrapper = _f("write_datavector_wrapper", None, [c_p, InputCosmologyParams, InputNuisanceParams])
set_all_parameters = _f("set_all_parameters", None, [InputCosmologyParams, InputNuisanceParams])
apply_Ttransform_and_applymask = _f("apply_Ttransform_and_applymask", None, [PD])
T_Ytransform = _f("T_Ytransform", c_d, [c_i, c_i, c_i, c_d])
T_Ytransform_full = _f("T_Ytransform_full", c_d, [c_i, c_i])
ZL = _f("ZL", c_i, [c_i])
ZS = _f("ZS", c_i, [c_i])
ZC = _f("ZC", c_i, [c_i])
ZSC = _f("ZSC", c_i, [c_i])
N_cgl = _f("N_cgl", c_i, [c_i, c_i])

# ---- background / power spectrum (cosmo3D.c; chi in c/H0, k in H0/c)
chi = _f("chi", c_d, [c_d])
f_K = _f("f_K", c_d, [c_d])
a_chi = _f("a_chi", c_d, [c_d])
dchi_da = _f("dchi_da", c_d, [c_d])
growfac = _f("growfac", c_d, [c_d])
f_growth = _f("f_growth", c_d, [c_d])
p_lin = _f("p_lin", c_d, [c_d, c_d])
Pdelta = _f("Pdelta", c_d, [c_d, c_d])
sigma_r_sqr = _f("sigma_r_sqr", c_d, [])

# ---- redshift distributions (redshift_spline.c)
pf_photoz = _f("pf_photoz", c_d, [c_d, c_i])
zdistr_photoz = _f("zdistr_photoz", c_d, [c_d, c_i])
zmean = _f("zmean", c_d, [c_i])
zmean_source = _f("zmean_source", c_d, [c_i])
g_tomo = _f("g_tomo", c_d, [c_d, c_i])
W_kappa = _f("W_kappa", c_d, [c_d, c_d, c_d])
W_gal = _f("W_gal", c_d, [c_d, c_d])
b1_per_bin = _f("b1_per_bin", c_d, [c_d, c_i])

# ---- halo model (halo.c; M in Msun/h, a scale factor)
massfunc = _f("massfunc", c_d, [c_d, c_d])
B1 = _f("B1", c_d, [c_d, c_d])
conc = _f("conc", c_d, [c_d, c_d])
u_nfw_c = _f("u_nfw_c", c_d, [c_d, c_d, c_d, c_d])
nu = _f("nu", c_d, [c_d, c_d])
sigma2 = _f("sigma2", c_d, [c_d])
r_Delta = _f("r_Delta", c_d, [c_d, c_d])

# ---- cluster model (cluster_util.c, clusters_DES_nonlimber.c)
int_probability_observed_richness_given_mass = _f("int_probability_observed_richness_given_mass", c_d, [c_d, c_vp])
probability_observed_richness_given_mass = _f("probability_observed_richness_given_mass", c_d, [c_i, c_d, c_d])
probability_observed_richness_given_mass_tab = _f("probability_observed_richness_given_mass_tab", c_d, [c_i, c_d, c_d])
int_dndlogM_P_lambda_obs_z = _f("int_dndlogM_P_lambda_obs_z", c_d, [c_d, c_vp])
n_lambda_obs_z = _f("n_lambda_obs_z", c_d, [c_i, c_d])
n_lambda_obs_z_tab = _f("n_lambda_obs_z_tab", c_d, [c_i, c_d])
weighted_bias = _f("weighted_bias", c_d, [c_i, c_d])
weighted_bias_exact = _f("weighted_bias_exact", c_d, [c_i, c_d])
mass_mean = _f("mass_mean", c_d, [c_i, c_d])
mean_mass_Delta_lambda_obs = _f("mean_mass_Delta_lambda_obs", c_d, [c_i, c_i])
N_Delta_lambda_obs = _f("N_Delta_lambda_obs", c_d, [c_i, c_i])
zdistr_cluster = _f("zdistr_cluster", c_d, [c_d, c_i, c_i])
norm_z_cluster = _f("norm_z_cluster", c_d, [c_i, c_i])
W_cluster = _f("W_cluster", c_d, [c_d, c_i, c_i])
W_cl = _f("W_cl", c_d, [c_d, c_i, c_i])
W_mag_cluster = _f("W_mag_cluster", c_d, [c_d, c_d, c_i])
g_lens_cluster = _f("g_lens_cluster", c_d, [c_d, c_i, c_i])
C_cc = _f("C_clusterxclusterclustering_tomo", c_d, [c_d, c_i, c_i, c_i, c_i, c_d])
C_cc_tab = _f("C_clusterxclusterclustering_tomo_tab", c_d, [c_d, c_i, c_i, c_i, c_i])
C_cg = _f("C_clusterxgalaxyclustering_tomo", c_d, [c_d, c_i, c_i, c_i, c_i, c_d])
C_cg_tab = _f("C_clusterxgalaxyclustering_tomo_tab", c_d, [c_d, c_i, c_i, c_i, c_i])
C_cs = _f("C_clusterlensing_tomo", c_d, [c_d, c_i, c_i, c_i, c_i])
C_cs_tab = _f("C_clusterlensing_tomo_tab", c_d, [c_d, c_i, c_i, c_i, c_i])
C_cc_mix_tab = _f("C_clusterxclusterclustering_mix_tab", None, [c_i, c_i, c_i, c_i, c_i, c_i, PD, c_d, c_d])
C_cg_mix_tab = _f("C_clusterxgalaxyclustering_mix_tab", None, [c_i, c_i, c_i, c_i, c_i, PD, c_d, c_d])
P_cm_1h = _f("P_cluster_mass_given_Dlambda_obs_1halo", c_d, [c_d, c_d, c_i, c_i, c_i])
P_cm_1h_tab = _f("P_cluster_mass_given_Dlambda_obs_1halo_tab", c_d, [c_d, c_d, c_i, c_i, c_i])
P_cc_2h = _f("P_cluster_x_cluster_clustering_mass_given_Dlambda_obs", c_d, [c_d, c_d, c_i, c_i, c_d])
P_cg_2h = _f("P_cluster_x_galaxy_clustering_mass_given_Dlambda_obs", c_d, [c_d, c_d, c_i, c_i, c_i, c_d])
cluster_gamma_t = _f("cluster_gamma_t", c_d, [c_d, c_i, c_i, c_i])
w_cc_nonlimber = _f("w_clusterxcluster_nonlimber", c_d, [c_i, c_i, c_i, c_i, c_i])
w_cg_nonlimber = _f("w_clusterxgalaxy_nonlimber", c_d, [c_i, c_i, c_i, c_i])
C_cl_tomo = _f("C_cl_tomo", c_d, [c_d, c_i, c_i])
w_tomo_exact = _f("w_tomo_exact", c_d, [c_i, c_i, c_i])

ARCMIN = 2.90888208665721580e-4  # constants.arcmin in basics.c
COVERH0 = 2997.92458

# ------------------------------------------------------------------ config
CONFIG = dict(
    runmode="Halofit",
    source_nz=os.path.join(HERE, "inputs", "source_y6.nz"),
    lens_nz=os.path.join(HERE, "inputs", "lens_y6.nz"),
    ntomo_source=4,
    ntomo_lens=6,
    ggl_overlap_cut=1e-5,
    ntheta=20, theta_min_arcmin=2.5, theta_max_arcmin=250.0,
    cluster_zbins=[0.2, 0.4, 0.55, 0.65],
    cluster_lambda_bins=[20.0, 30.0, 45.0, 60.0, 500.0],
    cluster_tomo_cg=[[0, 0], [1, 1], [2, 2]],
    survey_area=4143.0,
    sigma_e=0.384666,
    lens_n_gal=[0.1380, 0.1016, 0.1071, 0.1381, 0.1054, 0.1045],
    source_n_gal=[2.1402, 2.14455, 2.1518, 2.11845],
    nonlinear=0,
    probes=["xip", "xim", "gammat", "wtheta", "w_cg", "gamma_c", "cluster_N", "w_cc"],
    Ystatistics=1,
    # cosmology (A_s = 2.19e-9 -> sigma_8 via CAMB 1.6.5, massless nu, Neff 3.046:
    # the Halofit runmode normalizes the EH P_lin with sigma_8 only)
    omega_m=0.3, A_s_target=2.19e-9, sigma_8=0.8425192400940533, n_s=0.96859,
    omega_b=0.048, h0=0.69, omega_nuh2=0.0, w0=-1.0, wa=0.0,
    # nuisance
    b1=[1.42, 1.66, 1.70, 1.62, 1.78, 1.75],  # analysis/yamlfiles/dataY6.yaml (= 2503.13631 Table I)
    MOR=[4.26, 0.943, 0.15, 0.207],           # ln lambda0, A, sigma_int, B  (Mpiv = 5e14 Msun/h hard-coded)
    selection=[1.0, 0.0, 30.0, 0.0],          # b_s1, b_s2, r0 [Mpc/h], z-slope -> b_sel = 1
)


def cosmo_struct(cfg=CONFIG):
    c = InputCosmologyParams()
    c.omega_m = cfg["omega_m"]
    c.sigma_8 = cfg["sigma_8"]
    c.A_s = 0.0  # theory_wrapper: NORM = A_s if A_s != 0 else sigma_8
    c.n_s = cfg["n_s"]
    c.w0 = cfg["w0"]
    c.wa = cfg["wa"]
    c.omega_b = cfg["omega_b"]
    c.omega_nuh2 = cfg["omega_nuh2"]
    c.h0 = cfg["h0"]
    c.MGSigma = 0.0
    c.MGmu = 0.0
    c.log10Tagn = 0.0
    return c


def nuisance_struct(cfg=CONFIG):
    n = InputNuisanceParams()
    n.bias[:] = list(cfg["b1"]) + [2.0] * (10 - len(cfg["b1"]))
    n.bias2[:] = [0.0] * 10
    n.lens_z_bias[:] = [0.0] * 10
    n.lens_z_stretch[:] = [1.0] * 10  # must be 1: pf_photoz divides by it
    n.source_z_bias[:] = [0.0] * 10
    n.shear_m[:] = [0.0] * 10
    n.A_z[:] = [0.0] * 10  # like.IA = 0 (init_IA_mpp never called): no IA anywhere
    n.lens_z_u[:] = [0.0] * 20
    n.source_z_u[:] = [0.0] * 20
    n.MOR[:] = list(cfg["MOR"]) + [0.0] * (10 - len(cfg["MOR"]))
    n.selection[:] = list(cfg["selection"]) + [0.0] * (10 - len(cfg["selection"]))
    n.b_mag[:] = [0.0] * 10
    n.pm[:] = [0.0] * 10
    return n


def init_all(cfg=CONFIG, mask_path=None, Ystatistics=None):
    """Replicates init_all_nuisance_param() of cosmolike_libs_real_mpp_cluster.py."""
    if Ystatistics is None:
        Ystatistics = cfg["Ystatistics"]
    init_cosmo_runmode(cfg["runmode"].encode())
    init_source_sample_mpp(cfg["source_nz"].encode(), cfg["ntomo_source"])
    init_lens_sample_mpp(cfg["lens_nz"].encode(), cfg["ntomo_lens"], Double10(), Double10(),
                         cfg["ggl_overlap_cut"])
    init_binning_mpp(cfg["ntheta"], cfg["theta_min_arcmin"], cfg["theta_max_arcmin"])
    nl = len(cfg["cluster_lambda_bins"]) - 1
    init_cluster_N200(nl, (c_d * (nl + 1))(*cfg["cluster_lambda_bins"]))
    nzc = len(cfg["cluster_zbins"]) - 1
    init_tomo_cluster(nzc, (c_d * (nzc + 1))(*cfg["cluster_zbins"]))
    ncg = len(cfg["cluster_tomo_cg"])
    zc = (c_i * (ncg + 1))(*[p[0] for p in cfg["cluster_tomo_cg"]])
    zg = (c_i * (ncg + 1))(*[p[1] for p in cfg["cluster_tomo_cg"]])
    set_cluster_cg_Npowerspectra(ncg, zc, zg)
    # params['nonlinear'] == 0 -> init_cluster_b2/set_cluster_b2 are not called
    lens_n = (c_d * (cfg["ntomo_lens"] + 1))(*cfg["lens_n_gal"])
    src_n = (c_d * (cfg["ntomo_source"] + 1))(*cfg["source_n_gal"])
    init_survey_mpp(b"buzzard", cfg["survey_area"], cfg["sigma_e"], lens_n, src_n, b"None")
    init_cluster_area_interpolation(b"", False)
    init_cluster_photo_z(b"", False)
    set_nonlinear(cfg["nonlinear"])
    init_probes_4x2ptN("".join(cfg["probes"]).encode())
    set_Ystatistics(int(Ystatistics))
    ndata = getNdata()
    if mask_path is None:
        mask_path = os.path.join(HERE, "inputs", "ones_%d.mask" % ndata)
    if not os.path.isfile(mask_path):
        with open(mask_path, "w") as f:
            for i in range(ndata):
                f.write("%d 1.0\n" % i)
    init_mask(mask_path.encode())
    return ndata


def layout(cfg=CONFIG):
    """Block layout of the data vector (like_real_mpp.c:1041-1076)."""
    nt = cfg["ntheta"]
    ns = cfg["ntomo_source"]
    nlens = cfg["ntomo_lens"]
    nshear = ns * (ns + 1) // 2
    nggl = get_N_ggl()
    ncg = len(cfg["cluster_tomo_cg"])
    nl = len(cfg["cluster_lambda_bins"]) - 1
    nzc = len(cfg["cluster_zbins"]) - 1
    ncgl = nzc * ns  # init_tomo_cluster "all pairs" hack
    blocks = []
    start = 0
    for name, size in [("xip", nt * nshear), ("xim", nt * nshear), ("gammat", nt * nggl),
                       ("wtheta", nt * nlens), ("w_cg", nt * ncg * nl), ("N", nzc * nl),
                       ("w_cc", nt * (nl * (nl - 1) // 2 + nl) * nzc), ("gamma_c", nt * ncgl * nl)]:
        blocks.append((name, start, start + size))
        start += size
    return blocks, start


def theta_bins(cfg=CONFIG):
    """theta edges and the area-weighted bin centre used by the C code (radian)."""
    nt = cfg["ntheta"]
    vtmin = cfg["theta_min_arcmin"] * ARCMIN
    vtmax = cfg["theta_max_arcmin"] * ARCMIN
    logdt = (np.log(vtmax) - np.log(vtmin)) / nt
    tmin = np.exp(np.log(vtmin) + np.arange(nt) * logdt)
    tmax = np.exp(np.log(vtmin) + (np.arange(nt) + 1.0) * logdt)
    th = 2.0 / 3.0 * (tmax ** 3 - tmin ** 3) / (tmax ** 2 - tmin ** 2)
    return tmin, tmax, th, logdt


def legendre_bin_weights(cfg=CONFIG, LMAX=100000):
    """Pl[i][l] of w_clusterx*_nonlimber: 1/(4pi) (P_{l+1}-P_{l-1})|_{xmax}^{xmin} / (xmin-xmax)."""
    tmin, tmax, _, _ = theta_bins(cfg)
    xmin = np.cos(tmin)
    xmax = np.cos(tmax)
    x = np.concatenate([xmin, xmax])
    P = np.empty((LMAX + 2, x.size))
    P[0] = 1.0
    P[1] = x
    for l in range(1, LMAX + 1):
        P[l + 1] = ((2 * l + 1) * x * P[l] - l * P[l - 1]) / (l + 1)
    nt = tmin.size
    Pmin = P[:, :nt]
    Pmax = P[:, nt:]
    W = np.zeros((nt, LMAX))
    W[:, 0] = 1.0
    l = np.arange(1, LMAX)
    W[:, 1:] = (1.0 / (4.0 * np.pi) * (Pmin[l + 1] - Pmax[l + 1] - Pmin[l - 1] + Pmax[l - 1])
                / (xmin - xmax)[None, :]).T
    return W


def hoverh0(a, cfg=CONFIG):
    # static inline in cosmo3D.c (not exported): no radiation, flat LCDM here
    # set_cosmology_params sets Omega_v = 1 - Omega_m; w0=-1, wa=0 here
    Om = cfg["omega_m"]
    return np.sqrt(Om / a ** 3 + (1.0 - Om))
