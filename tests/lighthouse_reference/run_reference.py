"""Produce lighthouse reference outputs. One variant per (fresh) process:

    python run_reference.py main      # Ystatistics=1: data vectors + intermediates
    python run_reference.py Yoff      # Ystatistics=0: data vector (no final T)
    python run_reference.py limber    # Ystatistics=1: w_gg, w_cg, w_cc replaced by Limber

Outputs go to ./outputs/. See README.md.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lh  # noqa: E402

OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
LMAX = 100000
cfg = lh.CONFIG
NL = len(cfg["cluster_lambda_bins"]) - 1
NZC = len(cfg["cluster_zbins"]) - 1
NT = cfg["ntheta"]


def savetxt_dv(path, v, header):
    with open(path, "w") as f:
        f.write("# " + header.replace("\n", "\n# ") + "\n")
        for i, x in enumerate(v):
            f.write("%d %.17e\n" % (i, x))


def block(v, name):
    for n, a, b in lh.layout()[0]:
        if n == name:
            return v[a:b]
    raise KeyError(name)


def theory(ndata):
    out = np.zeros(ndata)
    t0 = time.time()
    lh.theory_wrapper(lh.cosmo_struct(), lh.nuisance_struct(), out.ctypes.data_as(lh.PD))
    print("theory_wrapper: %.1f s" % (time.time() - t0), flush=True)
    return out


def apply_T(v):
    w = np.ascontiguousarray(v.copy())
    lh.apply_Ttransform_and_applymask(w.ctypes.data_as(lh.PD))
    return w


def cc_pairs():
    """(nz, l1, l2) in data-vector order of set_data_cc."""
    out = []
    for l1 in range(NL):
        for l2 in range(l1, NL):
            for nz in range(NZC):
                out.append((nz, l1, l2))
    idx = []
    for (nz, l1, l2) in out:
        start = NT * ((NL * (NL - 1)) // 2 + NL) * nz + l1 * NT * (2 * NL - (l1 - 1)) // 2 + NT * (l2 - l1)
        idx.append(start)
    return out, idx


def header_common():
    c = dict(cfg)
    return json.dumps(c, sort_keys=True)


def gamma_c_raw_and_T():
    """Raw flat-sky gamma_c(theta_i) and the T / T^+ matrices used by set_data_clusterWL."""
    tmin, tmax, th, logdt = lh.theta_bins(cfg)
    ncgl = lh.get_N_tomo_shear() * NZC
    raw = np.zeros(ncgl * NL * NT)
    pairs = []
    for n in range(ncgl):
        zc, zs = lh.ZC(n), lh.ZSC(n)
        pairs.append((zc, zs))
        for il in range(NL):
            for i in range(NT):
                raw[NT * NL * n + NT * il + i] = lh.cluster_gamma_t(th[i], il, zc, zs)
    T = np.array([[lh.T_Ytransform(i, j, NT, logdt) for j in range(NT)] for i in range(NT)])
    return raw, np.array(pairs), T


def main_variant():
    ndata = lh.init_all(Ystatistics=1)
    blocks, _ = lh.layout()
    dv_pre = theory(ndata)
    dv_Y = apply_T(dv_pre)
    txt_path = os.path.join(OUT, "dv_main_compute_data_vector.txt")
    t0 = time.time()
    lh.write_datavector_wrapper(txt_path.encode(), lh.cosmo_struct(), lh.nuisance_struct())
    print("write_datavector_wrapper: %.1f s" % (time.time() - t0), flush=True)
    hdr = ("lighthouse like_cluster_Buzzard_y3_2_fast_v37_IA_test4.so, Ystatistics=%d\n"
           "blocks (name, start, end): %s\nconfig: %s")
    savetxt_dv(os.path.join(OUT, "dv_main_theory_wrapper.txt"), dv_pre,
               hdr % (1, blocks, header_common()) + "\nNOTE: theory_wrapper output = before the final Y transform "
               "(gamma_c block = T^+ diag(b_sel) T gamma_c)")
    savetxt_dv(os.path.join(OUT, "dv_main_Ytransformed.txt"), dv_Y,
               hdr % (1, blocks, header_common()) + "\nNOTE: T applied to the gamma_c block (what enters chi2 and "
               "compute_data_vector when Ystatistics=1)")

    res = {}
    res["dv_theory_wrapper"] = dv_pre
    res["dv_Ytransformed"] = dv_Y
    res["block_names"] = np.array([b[0] for b in blocks])
    res["block_start"] = np.array([b[1] for b in blocks])
    res["block_end"] = np.array([b[2] for b in blocks])
    tmin, tmax, th, logdt = lh.theta_bins(cfg)
    res["theta_min_arcmin"] = tmin / lh.ARCMIN
    res["theta_max_arcmin"] = tmax / lh.ARCMIN
    res["theta_arcmin"] = th / lh.ARCMIN
    res["logdt"] = logdt

    # ---------------- gamma_c: raw, T, T^+ (numpy pinv), check vs data vector
    t0 = time.time()
    raw, pairs, T = gamma_c_raw_and_T()
    print("gamma_c raw: %.1f s" % (time.time() - t0), flush=True)
    Tp = np.linalg.pinv(T, rcond=1e-15)
    res["gamma_c_raw"] = raw
    res["gamma_c_pairs_zc_zs"] = pairs
    res["T_Ytransform"] = T
    res["T_pinv_numpy"] = Tp
    gc = block(dv_pre, "gamma_c").reshape(-1, NT)
    rawr = raw.reshape(-1, NT)
    res["gamma_c_TpT_python"] = (rawr @ (Tp @ T).T).ravel()
    res["Tfull_cs_block_check"] = np.array([[lh.T_Ytransform_full(lh.layout()[0][-1][1] + i, lh.layout()[0][-1][1] + j)
                                             for j in range(NT)] for i in range(NT)])

    # ---------------- background (chi in c/H0; distances in Mpc/h = chi * 2997.92458)
    z = np.linspace(0.0, 3.0, 301)
    z[0] = 1e-4
    a = 1.0 / (1.0 + z)
    res["bg_z"] = z
    res["bg_chi"] = np.array([lh.chi(x) for x in a])
    res["bg_fK"] = np.array([lh.f_K(lh.chi(x)) for x in a])
    res["bg_dchi_da"] = np.array([lh.dchi_da(x) for x in a])
    res["bg_hoverh0"] = 1.0 / (a * a * res["bg_dchi_da"])
    g1 = lh.growfac(1.0)
    res["bg_growth_D_over_D0"] = np.array([lh.growfac(x) / g1 for x in a])
    res["bg_f_growth"] = np.array([lh.f_growth(x) for x in z])

    # ---------------- power spectra (code units: k in H0/c, P in (c/H0)^3)
    kh = np.logspace(-4, 2, 301)  # h/Mpc
    zP = np.array([0.0, 0.3, 0.475, 0.6, 1.0])
    res["pk_k_hMpc"] = kh
    res["pk_z"] = zP
    res["pk_lin_code"] = np.array([[lh.p_lin(k * lh.COVERH0, 1.0 / (1 + zz)) for k in kh] for zz in zP])
    res["pk_nl_code"] = np.array([[lh.Pdelta(k * lh.COVERH0, 1.0 / (1 + zz)) for k in kh] for zz in zP])

    # ---------------- halo model (M in Msun/h; massfunc = dn/dM in (H0/c)^3 / (Msun/h))
    lgM = np.linspace(12.0, 15.9, 79)
    M = 10 ** lgM
    zH = np.array([0.0, 0.2, 0.3, 0.4, 0.475, 0.55, 0.6, 0.65])
    res["halo_lgM"] = lgM
    res["halo_z"] = zH
    res["halo_massfunc_dndM"] = np.array([[lh.massfunc(m, 1 / (1 + zz)) for m in M] for zz in zH])
    res["halo_B1"] = np.array([[lh.B1(m, 1 / (1 + zz)) for m in M] for zz in zH])
    res["halo_nu"] = np.array([[lh.nu(m, 1 / (1 + zz)) for m in M] for zz in zH])
    res["halo_conc"] = np.array([[lh.conc(m, 1 / (1 + zz)) for m in M] for zz in zH])
    res["halo_sigma2_z0"] = np.array([lh.sigma2(m) for m in M])
    res["halo_rDelta_code_z0"] = np.array([lh.r_Delta(m, 1.0) for m in M])

    # ---------------- mass-observable relation
    lam = np.logspace(np.log10(5.0), np.log10(500.0), 200)
    zM = np.array([0.3, 0.475, 0.6])
    dens = np.zeros((zM.size, lgM.size, lam.size))
    for iz, zz in enumerate(zM):
        for im, m in enumerate(M):
            p = (lh.c_d * 2)(m, zz)
            for il, ll in enumerate(lam):
                dens[iz, im, il] = lh.int_probability_observed_richness_given_mass(ll, ctypes_cast(p))
    res["mor_lambda"] = lam
    res["mor_z"] = zM
    res["mor_P_lambda_given_M_density"] = dens  # d P / d lambda_obs  [lnormal, 1/lambda]
    res["mor_P_bin_given_M_exact"] = np.array([[[lh.probability_observed_richness_given_mass(il, m, zz)
                                                  for m in M] for zz in zM] for il in range(NL)])
    res["mor_P_bin_given_M_tab"] = np.array([[[lh.probability_observed_richness_given_mass_tab(il, m, zz)
                                                for m in M] for zz in zM] for il in range(NL)])

    # ---------------- n_A(z), b_A(z), <M>, counts
    zc = np.linspace(0.15, 0.70, 111)
    res["clz_z"] = zc
    res["n_A_exact"] = np.array([[lh.n_lambda_obs_z(il, zz) for zz in zc] for il in range(NL)])
    res["n_A_tab"] = np.array([[lh.n_lambda_obs_z_tab(il, zz) for zz in zc] for il in range(NL)])
    res["b_A_tab"] = np.array([[lh.weighted_bias(il, zz) for zz in zc] for il in range(NL)])
    res["b_A_exact"] = np.array([[lh.weighted_bias_exact(il, zz) for zz in zc] for il in range(NL)])
    res["mass_mean_A"] = np.array([[lh.mass_mean(il, zz) for zz in zc] for il in range(NL)])
    zcen = 0.5 * (np.array(cfg["cluster_zbins"][:-1]) + np.array(cfg["cluster_zbins"][1:]))
    res["zc_centers"] = zcen
    res["b_A_at_centers_tab"] = np.array([[lh.weighted_bias(il, zz) for zz in zcen] for il in range(NL)])
    res["b_A_at_centers_exact"] = np.array([[lh.weighted_bias_exact(il, zz) for zz in zcen] for il in range(NL)])
    res["N_counts"] = np.array([[lh.N_Delta_lambda_obs(il, iz) for il in range(NL)] for iz in range(NZC)])
    res["mean_mass_bin"] = np.array([[lh.mean_mass_Delta_lambda_obs(il, iz) for il in range(NL)] for iz in range(NZC)])

    # ---------------- cluster n(z) kernel, weights
    zf = np.linspace(0.15, 0.70, 551)
    af = 1 / (1 + zf)
    res["kern_z"] = zf
    res["zdistr_cluster"] = np.array([[lh.zdistr_cluster(zz, iz, 0) for zz in zf] for iz in range(NZC)])
    res["norm_z_cluster"] = np.array([lh.norm_z_cluster(iz, 0) for iz in range(NZC)])
    res["W_cluster"] = np.array([[[lh.W_cluster(x, iz, il) for x in af] for il in range(NL)] for iz in range(NZC)])
    res["W_cl"] = np.array([[[lh.W_cl(x, iz, il) for x in af] for il in range(NL)] for iz in range(NZC)])
    zw = np.linspace(0.01, 0.70, 139)
    aw = 1 / (1 + zw)
    res["gl_z"] = zw
    res["g_lens_cluster"] = np.array([[lh.g_lens_cluster(x, iz, -1) for x in aw] for iz in range(NZC)])
    res["W_mag_cluster"] = np.array([[lh.W_mag_cluster(x, lh.f_K(lh.chi(x)), iz) for x in aw] for iz in range(NZC)])

    # ---------------- galaxy samples
    zg = np.linspace(0.0, 3.0, 601)
    res["nz_z"] = zg
    res["pf_photoz_lens"] = np.array([[lh.pf_photoz(zz, j) for zz in zg] for j in range(cfg["ntomo_lens"])])
    res["zdistr_photoz_source"] = np.array([[lh.zdistr_photoz(zz, j) for zz in zg] for j in range(cfg["ntomo_source"])])
    res["zmean_lens"] = np.array([lh.zmean(j) for j in range(cfg["ntomo_lens"])])
    res["zmean_source"] = np.array([lh.zmean_source(j) for j in range(cfg["ntomo_source"])])
    res["clustering_zmax"] = np.array([lh.get_tomo_clustering_zmax(j) for j in range(cfg["ntomo_lens"])])
    res["g_tomo_source"] = np.array([[lh.g_tomo(x, j) for x in aw] for j in range(cfg["ntomo_source"])])

    # ---------------- Limber C_ell (and linear), 1-halo, 2-halo P
    ells = np.array([2., 5., 10., 20., 50., 100., 200., 500., 1000., 2000., 5000., 1e4, 3e4, 9e4])
    res["cl_ell"] = ells
    pairs_cc, idx_cc = cc_pairs()
    res["cc_pairs_nz_l1_l2"] = np.array(pairs_cc)
    res["C_cc_limber_nl_exact"] = np.array([[lh.C_cc(l, l1, l2, nz, nz, 0.0) for l in ells] for (nz, l1, l2) in pairs_cc])
    res["C_cc_limber_lin_exact"] = np.array([[lh.C_cc(l, l1, l2, nz, nz, 1.0) for l in ells] for (nz, l1, l2) in pairs_cc])
    res["C_cc_limber_nl_tab"] = np.array([[lh.C_cc_tab(l, l1, l2, nz, nz) for l in ells] for (nz, l1, l2) in pairs_cc])
    cg = cfg["cluster_tomo_cg"]
    res["cg_pairs_zc_zg"] = np.array(cg)
    res["C_cg_limber_nl_exact"] = np.array([[[lh.C_cg(l, il, -1, zc_, zg_, 0.0) for l in ells] for il in range(NL)] for (zc_, zg_) in cg])
    res["C_cg_limber_lin_exact"] = np.array([[[lh.C_cg(l, il, -1, zc_, zg_, 1.0) for l in ells] for il in range(NL)] for (zc_, zg_) in cg])
    res["C_cg_limber_nl_tab"] = np.array([[[lh.C_cg_tab(l, il, -1, zc_, zg_) for l in ells] for il in range(NL)] for (zc_, zg_) in cg])
    t0 = time.time()
    res["C_cs_limber_exact"] = np.array([[[lh.C_cs(l, il, -1, zc_, zs_) for l in ells] for il in range(NL)] for (zc_, zs_) in pairs])
    res["C_cs_limber_tab"] = np.array([[[lh.C_cs_tab(l, il, -1, zc_, zs_) for l in ells] for il in range(NL)] for (zc_, zs_) in pairs])
    print("C_cs: %.1f s" % (time.time() - t0), flush=True)
    kk = np.array([0.01, 0.1, 0.3, 1.0, 3.0, 10.0])  # h/Mpc
    zz3 = zcen
    res["p1h_k_hMpc"] = kk
    res["p1h_z"] = zz3
    # P_cm^1h(k,a) [(c/H0)^3]: (lambda, zc bin, k, z at zc centre); zs index 0 (does not enter)
    res["P_cm_1h_exact"] = np.array([[[lh.P_cm_1h(k * lh.COVERH0, 1 / (1 + zz3[iz]), il, iz, 0) for k in kk]
                                      for iz in range(NZC)] for il in range(NL)])
    res["P_cc_2h_nl"] = np.array([[[lh.P_cc_2h(k * lh.COVERH0, 1 / (1 + zz3[iz]), il, il, 0.0) for k in kk]
                                   for iz in range(NZC)] for il in range(NL)])

    # ---------------- non-Limber C_ell of w_cc / w_cg exactly as the data vector builds them
    W = lh.legendre_bin_weights(cfg, LMAX)
    Cl = np.zeros(LMAX)
    cl_cc_mix = np.zeros((len(pairs_cc), 2001))
    cl_cc_lim = np.zeros((len(pairs_cc), 2001))
    w_cc_proj_mix = np.zeros((len(pairs_cc), NT))
    w_cc_proj_lim = np.zeros((len(pairs_cc), NT))
    order = sorted(range(len(pairs_cc)), key=lambda i: pairs_cc[i])  # nz outer, l1, l2 (as w_clusterxcluster_nonlimber)
    lgrid = np.arange(LMAX, dtype=float)
    t0 = time.time()
    for i in order:
        nz, l1, l2 = pairs_cc[i]
        Cl[:] = 0.0
        lh.C_cc_mix_tab(0, LMAX, nz, nz, l1, l2, Cl.ctypes.data_as(lh.PD), 0.1, 0.01)
        cl_cc_mix[i] = Cl[:2001]
        w_cc_proj_mix[i] = W[:, 1:] @ Cl[1:]
        Clim = np.array([lh.C_cc_tab(l, l1, l2, nz, nz) for l in lgrid[1:]])
        cl_cc_lim[i, 1:] = Clim[:2000]
        w_cc_proj_lim[i] = W[:, 1:] @ Clim
    print("cc mix/limber: %.1f s" % (time.time() - t0), flush=True)
    res["Cl_cc_nonlimber_mix_l0_2000"] = cl_cc_mix
    res["Cl_cc_limber_tab_l0_2000"] = cl_cc_lim
    res["w_cc_projected_from_mix"] = w_cc_proj_mix
    res["w_cc_limber"] = w_cc_proj_lim
    cl_cg_mix = np.zeros((len(cg), NL, 2001))
    cl_cg_lim = np.zeros((len(cg), NL, 2001))
    w_cg_proj_mix = np.zeros((len(cg), NL, NT))
    w_cg_proj_lim = np.zeros((len(cg), NL, NT))
    t0 = time.time()
    for n, (zc_, zg_) in enumerate(cg):
        for il in range(NL):
            Cl[:] = 0.0
            lh.C_cg_mix_tab(0, LMAX, zc_, zg_, il, Cl.ctypes.data_as(lh.PD), 0.1, 0.01)
            cl_cg_mix[n, il] = Cl[:2001]
            w_cg_proj_mix[n, il] = W[:, 1:] @ Cl[1:]
            Clim = np.array([lh.C_cg_tab(l, il, -1, zc_, zg_) for l in lgrid[1:]])
            cl_cg_lim[n, il, 1:] = Clim[:2000]
            w_cg_proj_lim[n, il] = W[:, 1:] @ Clim
    print("cg mix/limber: %.1f s" % (time.time() - t0), flush=True)
    res["Cl_cg_nonlimber_mix_l0_2000"] = cl_cg_mix
    res["Cl_cg_limber_tab_l0_2000"] = cl_cg_lim
    res["w_cg_projected_from_mix"] = w_cg_proj_mix
    res["w_cg_limber"] = w_cg_proj_lim
    res["cc_block_offsets"] = np.array(idx_cc)

    np.savez_compressed(os.path.join(OUT, "lighthouse_reference_main.npz"), **res)
    with open(os.path.join(OUT, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1, sort_keys=True)
    print("saved main", flush=True)


def ctypes_cast(p):
    return lh.ctypes.cast(p, lh.c_vp)


def yoff_variant():
    ndata = lh.init_all(Ystatistics=0)
    blocks, _ = lh.layout()
    dv = theory(ndata)
    txt_path = os.path.join(OUT, "dv_Yoff_compute_data_vector.txt")
    lh.write_datavector_wrapper(txt_path.encode(), lh.cosmo_struct(), lh.nuisance_struct())
    savetxt_dv(os.path.join(OUT, "dv_Yoff_theory_wrapper.txt"), dv,
               "Ystatistics=0\nblocks: %s\nconfig: %s" % (blocks, header_common()))
    np.savez_compressed(os.path.join(OUT, "lighthouse_reference_Yoff.npz"), dv_theory_wrapper=dv)
    print("saved Yoff", flush=True)


def limber_variant():
    ndata = lh.init_all(Ystatistics=1)
    blocks, _ = lh.layout()
    dv = theory(ndata)  # fills all caches exactly as in main
    W = lh.legendre_bin_weights(cfg, LMAX)
    lgrid = np.arange(1, LMAX, dtype=float)
    dvL = dv.copy()
    # w_gg: Limber full-sky Legendre (w_tomo_exact, cosmo2D_real.c: same bin-averaged P_l, C_cl_tomo)
    a, b = [(x[1], x[2]) for x in blocks if x[0] == "wtheta"][0]
    for j in range(cfg["ntomo_lens"]):
        for i in range(NT):
            dvL[a + NT * j + i] = lh.w_tomo_exact(i, j, j)
    # w_cg (index: start + NT*NL*nz + NT*nlambda + i)
    a, b = [(x[1], x[2]) for x in blocks if x[0] == "w_cg"][0]
    for n, (zc_, zg_) in enumerate(cfg["cluster_tomo_cg"]):
        for il in range(NL):
            Clim = np.array([lh.C_cg_tab(l, il, -1, zc_, zg_) for l in lgrid])
            dvL[a + NT * NL * n + NT * il: a + NT * NL * n + NT * il + NT] = W[:, 1:] @ Clim
    # w_cc
    a, b = [(x[1], x[2]) for x in blocks if x[0] == "w_cc"][0]
    pairs_cc, idx_cc = cc_pairs()
    for (nz, l1, l2), off in zip(pairs_cc, idx_cc):
        Clim = np.array([lh.C_cc_tab(l, l1, l2, nz, nz) for l in lgrid])
        dvL[a + off: a + off + NT] = W[:, 1:] @ Clim
    dvLY = apply_T(dvL)
    savetxt_dv(os.path.join(OUT, "dv_limber_theory_wrapper.txt"), dvL,
               "Limber-only w_gg (w_tomo_exact), w_cg and w_cc (C_*_tomo_tab, bin-averaged Legendre); "
               "other blocks = theory_wrapper\nblocks: %s\nconfig: %s" % (blocks, header_common()))
    savetxt_dv(os.path.join(OUT, "dv_limber_Ytransformed.txt"), dvLY,
               "as dv_limber_theory_wrapper.txt with T applied to gamma_c\nblocks: %s" % (blocks,))
    np.savez_compressed(os.path.join(OUT, "lighthouse_reference_limber.npz"), dv_theory_wrapper=dv,
                        dv_limber=dvL, dv_limber_Ytransformed=dvLY)
    print("saved limber", flush=True)


if __name__ == "__main__":
    v = sys.argv[1]
    {"main": main_variant, "Yoff": yoff_variant, "limber": limber_variant}[v]()
