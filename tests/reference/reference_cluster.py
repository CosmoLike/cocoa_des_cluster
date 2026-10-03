#!/usr/bin/env python
"""Independent Python reference of the DES cluster observables
(2503.13631, "Y6 methods" model) used as ground truth for the C port of
the cluster code into cosmolike_core (des_cluster project).

Modules (same folder):
  ref_cosmology  CAMB background, growth, P_lin/P_NL; n(z) files
  ref_halo       Tinker 2010 f(nu) (alpha = 0.368, or halo.c's alpha(z))
                 and b(nu), sigma(M), Bhattacharya c(M), NFW u(k|M)
  ref_cluster    MOR, P(lambda bin|M,z), n_A, b_A, P^1h_A, <phi_i|z>,
                 counts, radial kernels, cluster lensing efficiency
  ref_limber     Limber C_l: cs (2h + 1h + mag + NLA), cc, cg (+ gg, gs, ss)
  ref_projection full-sky bin-averaged Legendre w / gamma_t, Y transform,
                 selection factor
  ref_nonlimber  exact j_l linear spectra (Limber diagnostic, SSC)
  ref_covariance Gaussian covariance (+ counts Poisson + SSC)

Usage (python):

    from reference_cluster import ClusterReference, FIDUCIAL, DEFAULT_SETTINGS
    ref = ClusterReference(FIDUCIAL, dict(kernel_mode=0))
    N = ref.counts()                      # (3, 4)
    n, b = ref.nA_bA(z)                   # (4, nz) each
    P1h = ref.p1h(k, z)                   # (4, nk) at one z
    sp = ref.spectra()                    # dict of C_l on sp["ells"]
    rs = ref.real_space()                 # w_cc, w_cg, gamma_t, Sigma, ...
    dv = ref.data_vector()                # lighthouse layout, see below
    out = ref.results()                   # everything, flat dict for .npz

CLI:

    python reference_cluster.py --out ref_fiducial.npz [--kernel-mode 1]
        [--hmf-matter cb] [--hmf-alpha-mode 1] [--cov] [--nonlimber-check]
        [--set key=value ...]

Data-vector layout (ref.data_vector(), lighthouse inner ordering):
  N  [zc][lambda]
  cs [(zc, zs) pair, zc-major][lambda][theta]   Sigma = b_sel * (T gamma_t)
  cc [zc][lambda1 <= lambda2][theta]            w_cc * b_sel^2
  cg [(zc, zg = zc) pair][lambda][theta]        w_cg * b_sel
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from ref_cosmology import Cosmology, NzBins                       # noqa: E402
from ref_halo import HaloModel                                    # noqa: E402
from ref_cluster import ClusterModel                              # noqa: E402
from ref_limber import LimberModel, default_ells                  # noqa: E402
from ref_projection import (theta_edges, theta_area_weighted,     # noqa: E402
                            theta_log_centre, project, y_transform,
                            selection_factor)
from ref_covariance import cc_pairs                               # noqa: E402

LIGHTHOUSE_NZ = ("/Users/vivianmiranda/data/COCOA/september2026/test/lighthouse/"
                 "analysis/des_y6_code_comparison")

# Table I of 2503.13631 (fiducial = prior mean where the prior is Gaussian).
FIDUCIAL = dict(
    Omega_m=0.3, A_s=2.19e-9, n_s=0.96859, Omega_b=0.048, h=0.69,
    Omega_nu_h2=0.00083, w0=-1.0, wa=0.0,
    # mass-observable relation (Eqs. 18-19), M_piv = 5e14 Msun/h
    lnlambda0=4.26, A=0.943, B=0.207, sigma_int=0.15,
    # selection bias (Eq. 23): b_s1, b_s2, r0 [Mpc/h], z-slope (lighthouse s3)
    sel_s0=1.1, sel_s1=0.2, sel_s2=30.0, sel_s3=0.0,
    # lenses (MagLim, 6 bins; clusters use bins 1-3)
    lens_b1=[1.42, 1.66, 1.70, 1.62, 1.78, 1.75],
    lens_bmag=[-1.57, -1.70, -0.25, 1.50, 2.22, 2.80],
    lens_dz=[0.005, 0.003, 0.001, -0.002, 0.001, 0.008],
    lens_stretch=[1.0] * 6,
    # sources (4 bins)
    source_dz=[0.034, 0.028, 0.011, -0.010],
    shear_m=[0.0, 0.0, 0.0, 0.0],
    # NLA (TATT a1, eta1; A2 = 0 in the cluster lensing)
    IA_A1=0.0, IA_eta1=0.0,
)

DEFAULT_SETTINGS = dict(
    lambda_edges=[20.0, 30.0, 45.0, 60.0, 500.0],
    zc_edges=[0.2, 0.4, 0.55, 0.65],
    area_deg2=4143.0,
    photoz="gaussian", sigma_z0=0.006,
    phi_table_file=None,         # photoz="table": <phi_i|z> file (data/des_y6_cluster.nz format)
    M_piv=5e14, z_piv=1.45,
    lnM_min=np.log(1e12), lnM_max=np.log(1e16),
    mass_panel_width=0.25, mass_order=8,
    kernel_mode=0,               # 0 volume-only (default), 1 abundance-weighted
    hmf_matter="cb",             # production cb; "tot" only for diagnostics
    hmf_alpha_mode=0,            # Tinker alpha: 0 = 0.368 at every z (DES / lighthouse;
                                 # cluster.hmf_alpha_mode default), 1 = halo.c's alpha(z)
    ntheta=20, tmin_arcmin=2.5, tmax_arcmin=250.0, lmax=75000,
    l_exact=30, n_per_decade=60,
    z_panel=0.01, z_panel_far=0.02, z_order=8,
    mag_ell_prefactor=True, spin2_prefactor=True, include_1h=True, C_c=-2.0,
    theta_sel="area",            # "area" (2/3 (t1^3-t0^3)/(t1^2-t0^2)) or "log"
    source_nz=os.path.join(LIGHTHOUSE_NZ, "source.nz"),
    lens_nz=os.path.join(LIGHTHOUSE_NZ, "lens.nz"),
    lens_bins=[0, 1, 2],
    kmax=100.0, halofit="takahashi", k_per_logint=None,
    pk_nl_z_order=3,             # 3 = cubic, 1 = linear in z between CAMB nodes (ref_cosmology)
    source_g_zmax=None,          # DIAGNOSTIC: cut the source n(z) of W_kappa above this z
    # noise for the covariance (lighthouse dataY6.yaml): arcmin^-2, per-component sigma_e
    n_lens_arcmin2=[0.1380, 0.1016, 0.1071, 0.1381, 0.1054, 0.1045],
    n_src_arcmin2=[2.1402, 2.14455, 2.1518, 2.11845],
    sigma_e=0.384666 / np.sqrt(2.0),
)

COSMO_KEYS = ("Omega_m", "A_s", "n_s", "Omega_b", "h", "Omega_nu_h2", "w0", "wa")
MOR_KEYS = ("lnlambda0", "A", "B", "sigma_int")


class ClusterReference:
    """Reference model at one parameter point. Every intermediate is exposed.

    `cosmo` can be passed to reuse a CAMB run (same cosmological parameters).
    """

    def __init__(self, params=None, settings=None, cosmo=None, verbose=False):
        self.params = dict(FIDUCIAL)
        if params:
            self.params.update(params)
        self.settings = dict(DEFAULT_SETTINGS)
        if settings:
            self.settings.update(settings)
        s, p = self.settings, self.params
        self.verbose = verbose
        t0 = time.time()
        self.cosmo = cosmo if cosmo is not None else Cosmology(
            {k: p[k] for k in COSMO_KEYS}, kmax=s["kmax"], halofit=s["halofit"],
            k_per_logint=s["k_per_logint"], nl_z_order=s["pk_nl_z_order"])
        self._log("CAMB", t0)
        t0 = time.time()
        phi_table = None
        if s["photoz"] == "table":
            tab = np.loadtxt(s["phi_table_file"])
            phi_table = (tab[:, 0], tab[:, 1:])
        self.halo = HaloModel(self.cosmo, hmf_matter=s["hmf_matter"],
                              hmf_alpha_mode=int(s["hmf_alpha_mode"]))
        self.cluster = ClusterModel(
            self.cosmo, self.halo, {k: p[k] for k in MOR_KEYS},
            lambda_edges=s["lambda_edges"], zc_edges=s["zc_edges"],
            area_deg2=s["area_deg2"], photoz=s["photoz"], sigma_z0=s["sigma_z0"],
            M_piv=s["M_piv"], z_piv=s["z_piv"], lnM_min=s["lnM_min"], lnM_max=s["lnM_max"],
            mass_panel_width=s["mass_panel_width"], mass_order=s["mass_order"],
            phi_table=phi_table)
        self._log("halo model", t0)
        self.nz_src = NzBins(s["source_nz"])
        self.nz_lens = NzBins(s["lens_nz"])
        self.edges = theta_edges(s["ntheta"], s["tmin_arcmin"], s["tmax_arcmin"])
        self.T = y_transform(self.edges)
        self._spectra = None
        self._counts = None
        self._real = None

    def _log(self, what, t0):
        if self.verbose:
            print(f"[reference_cluster] {what}: {time.time() - t0:.1f} s", flush=True)

    # ------------------------------------------------------------------
    # one-point pieces
    # ------------------------------------------------------------------
    def counts(self):
        if self._counts is None:
            self._counts = self.cluster.counts()
        return self._counts

    def nA_bA(self, z):
        return self.cluster.n_b(np.atleast_1d(z))

    def p1h(self, k, z):
        return self.cluster.p1h(k, z)

    def kernels(self, z):
        """<phi_i|z>, normalized q_iA(z), lensing efficiency g_c,iA(z)."""
        cl = self.cluster
        z = np.asarray(z, dtype=float)
        km = self.settings["kernel_mode"]
        norms = cl.kernel_norms(km)
        phi = np.array([cl.phi(z, i) for i in range(cl.nzc)])
        q = np.array([cl.q(z, i, km, norms) for i in range(cl.nzc)])
        g = np.array([cl.magnification_efficiency(z, i, km, norms) for i in range(cl.nzc)])
        return dict(phi=phi, q=q, g_c=g, norms=norms)

    # ------------------------------------------------------------------
    # spectra and real space
    # ------------------------------------------------------------------
    def limber_model(self):
        s = self.settings
        nuis = {k: self.params[k] for k in ("lens_b1", "lens_bmag", "lens_dz", "lens_stretch",
                                            "source_dz", "IA_A1", "IA_eta1")}
        ells = default_ells(s["lmax"], s["l_exact"], s["n_per_decade"])
        return LimberModel(self.cosmo, self.cluster, self.nz_src, self.nz_lens, nuis,
                           kernel_mode=s["kernel_mode"], ells=ells, lens_bins=s["lens_bins"],
                           z_panel=s["z_panel"], z_panel_far=s["z_panel_far"], order=s["z_order"],
                           mag_ell_prefactor=s["mag_ell_prefactor"],
                           spin2_prefactor=s["spin2_prefactor"], C_c=s["C_c"],
                           include_1h=s["include_1h"], source_g_zmax=s["source_g_zmax"])

    def spectra(self, with_cov_spectra=True):
        if self._spectra is None:
            t0 = time.time()
            self.limber = self.limber_model()
            self._spectra = self.limber.compute(with_cov_spectra=with_cov_spectra)
            self._log("Limber spectra", t0)
        return self._spectra

    def theta_sel(self):
        return (theta_area_weighted(self.edges) if self.settings["theta_sel"] == "area"
                else theta_log_centre(self.edges))

    def selection(self):
        """b_sel(theta) per cluster z bin, shape (nzc, ntheta)."""
        p, cl = self.params, self.cluster
        th = self.theta_sel()
        zmid = 0.5 * (cl.zc_edges[:-1] + cl.zc_edges[1:])
        DM = self.cosmo.chi(zmid)
        return np.array([selection_factor(th, DM[i], p["sel_s0"], p["sel_s1"], p["sel_s2"],
                                          p["sel_s3"], z_mid=zmid[i])
                         for i in range(cl.nzc)])

    def real_space(self):
        if self._real is not None:
            return self._real
        sp = self.spectra()
        s = self.settings
        cl = self.cluster
        ells, lmax, e = sp["ells"], s["lmax"], self.edges
        nzc, nA = cl.nzc, cl.nA
        t0 = time.time()
        m = np.asarray(self.params["shear_m"])
        gt = project(ells, sp["C_cs"], e, lmax, "gt") * (1.0 + m)[None, None, :, None]
        gt_1h = project(ells, sp["C_cs_1h"], e, lmax, "gt") * (1.0 + m)[None, None, :, None]
        Ccc_same = np.array([sp["C_cc"][i, :, i, :, :] for i in range(nzc)])   # (i, A, B, l)
        wcc = project(ells, Ccc_same, e, lmax, "w")
        lens_bins = s["lens_bins"]
        Ccg_diag = np.array([sp["C_cg"][i, :, lens_bins.index(i), :] for i in range(nzc)])
        wcg = project(ells, Ccg_diag, e, lmax, "w")
        Sigma = np.einsum("tu,iasu->iast", self.T, gt)
        B = self.selection()
        self._log("projections", t0)
        self._real = dict(theta_edges=e, theta_sel=self.theta_sel(), gamma_t=gt, gamma_t_1h=gt_1h,
                    Sigma=Sigma, Sigma_sel=Sigma * B[:, None, None, :],
                    w_cc=wcc, w_cc_sel=wcc * (B**2)[:, None, None, :],
                    w_cg=wcg, w_cg_sel=wcg * B[:, None, :], b_sel=B, T=self.T)
        return self._real

    def data_vector(self, selected=True):
        """(vector, index dict) in the lighthouse layout (module docstring)."""
        rs = self.real_space()
        N = self.counts()
        nzc, nA, ns = N.shape[0], N.shape[1], self.nz_src.nbin
        cs = rs["Sigma_sel"] if selected else rs["gamma_t"]
        cc = rs["w_cc_sel"] if selected else rs["w_cc"]
        cg = rs["w_cg_sel"] if selected else rs["w_cg"]
        parts = [N.ravel()]
        parts.append(np.concatenate([cs[i, A, j] for i in range(nzc) for j in range(ns)
                                     for A in range(nA)]))
        parts.append(np.concatenate([cc[i, A, B] for i in range(nzc)
                                     for (A, B) in cc_pairs(nA)]))
        parts.append(np.concatenate([cg[i, A] for i in range(nzc) for A in range(nA)]))
        start = np.cumsum([0] + [len(x) for x in parts])
        return np.concatenate(parts), dict(N=start[0], cs=start[1], cc=start[2], cg=start[3],
                                           end=start[4])

    # ------------------------------------------------------------------
    # covariance and diagnostics
    # ------------------------------------------------------------------
    def covariance(self):
        """Full covariance (N, cs as Sigma, cc, cg); no selection factors."""
        from ref_covariance import GaussianCovariance, counts_covariance
        s = self.settings
        t0 = time.time()
        gc = GaussianCovariance(self.spectra(), self.counts(), self.cluster.Omega_s,
                                self.edges, self.T, s["lmax"],
                                n_lens_arcmin2=[s["n_lens_arcmin2"][g] for g in s["lens_bins"]],
                                n_src_arcmin2=s["n_src_arcmin2"], sigma_e=s["sigma_e"],
                                lens_bins=list(range(len(s["lens_bins"]))))
        c2, _ = gc.twopoint(y_transform=True)
        cN, info = counts_covariance(self)
        n1 = cN.shape[0]
        cov = np.zeros((n1 + c2.shape[0],) * 2)
        cov[:n1, :n1] = cN
        cov[n1:, n1:] = c2
        self._log("covariance", t0)
        return cov, info

    def wcc_nonlimber_check(self, i=0, A=0, B=0, ells=(2, 5, 10, 20, 30, 50), dchi=0.5):
        """Exact j_l vs Limber for the density leg of C_cc (bin i, lambda A x B)."""
        from ref_nonlimber import cl_exact_linear, cl_limber_linear
        cl, cosmo = self.cluster, self.cosmo
        km = self.settings["kernel_mode"]
        za, zb = cl.support(i)
        chi = np.arange(cosmo.chi(za), cosmo.chi(zb), dchi)
        z = cosmo.z_of_chi(chi)
        n, b = cl.n_b(z)
        norms = cl.kernel_norms(km)
        q = cl.q(z, i, km, norms, nb=(n, b))                        # dN/dz
        W = q / cosmo.dchi_dz(z)[None, :]                           # dN/dchi
        D = cosmo.growth(z)
        ker = np.array([W[A] * b[A] * D, W[B] * b[B] * D])
        ex = cl_exact_linear(cosmo, list(ells), chi, ker)[:, 0, 1]
        li = cl_limber_linear(cosmo, list(ells), chi, ker)[:, 0, 1]
        # Limber with P_NL (density leg only), same kernels
        ells_a = np.asarray(ells, dtype=float)
        kk = (ells_a[:, None] + 0.5) / chi[None, :]
        zz = np.broadcast_to(z, kk.shape)
        from scipy.integrate import simpson
        pnl = cosmo.P_nl(kk, zz) / D[None, :] ** 2
        nl = simpson(ker[0] * ker[1] * pnl / chi**2, x=chi, axis=-1)
        return dict(ells=ells_a, C_exact_lin=ex, C_limber_lin=li, C_limber_nl=nl,
                    C_fkem=nl + ex - li, ratio_exact_limber_lin=ex / li)

    # ------------------------------------------------------------------
    def results(self, with_cov=False, with_nonlimber=False, z_grid=None, k_grid=None):
        """Flat dict of every intermediate (for np.savez)."""
        cl = self.cluster
        out = {}
        z_grid = np.arange(0.10, 0.8001, 0.005) if z_grid is None else z_grid
        k_grid = np.logspace(-3, 2, 101) if k_grid is None else k_grid
        n, b = self.nA_bA(z_grid)
        out.update(z_grid=z_grid, n_A=n, b_A=b, mean_mass_A=cl.mean_mass(z_grid))
        zmid = 0.5 * (cl.zc_edges[:-1] + cl.zc_edges[1:])
        out.update(k_grid=k_grid, z_p1h=zmid,
                   P1h=np.array([self.p1h(k_grid, zz) for zz in zmid]))       # (z, A, k)
        kz = self.kernels(z_grid)
        out.update(phi=kz["phi"], q=kz["q"], g_c=kz["g_c"], kernel_norms=kz["norms"])
        out["N"] = self.counts()
        out["b_eff_counts"] = cl.counts_weighted_bias()
        sp = self.spectra()
        out.update(sp)
        for k, v in self.limber.nodes.items():
            out["nodes_" + k] = v
        rs = self.real_space()
        out.update(rs)
        dv, idx = self.data_vector(selected=True)
        dv0, _ = self.data_vector(selected=False)
        out.update(data_vector=dv, data_vector_noselection_noY=dv0,
                   dv_start=np.array([idx["N"], idx["cs"], idx["cc"], idx["cg"], idx["end"]]))
        out.update(self.cosmo.export_tables())
        out["lambda_edges"] = np.asarray(self.settings["lambda_edges"])
        out["zc_edges"] = np.asarray(self.settings["zc_edges"])
        out["lnM_range"] = np.asarray(cl.lnM_range)
        if with_cov:
            cov, info = self.covariance()
            out["cov"] = cov
            for k, v in info.items():
                out["cov_counts_" + k] = v
        if with_nonlimber:
            for k, v in self.wcc_nonlimber_check().items():
                out["nonlimber_" + k] = v
        return out


# ----------------------------------------------------------------------
def _parse_set(items):
    out = {}
    for it in items or []:
        key, val = it.split("=", 1)
        try:
            v = eval(val, {"np": np})
        except Exception:
            v = val
        out[key] = v
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="reference_cluster_fiducial.npz")
    ap.add_argument("--kernel-mode", type=int, default=0, choices=(0, 1))
    ap.add_argument("--hmf-matter", default="cb", choices=("tot", "cb"))
    ap.add_argument("--hmf-alpha-mode", type=int, default=0, choices=(0, 1),
                    help="Tinker alpha: 0 = 0.368 (DES), 1 = halo.c's alpha(z)")
    ap.add_argument("--cov", action="store_true", help="also compute the Gaussian covariance")
    ap.add_argument("--nonlimber-check", action="store_true")
    ap.add_argument("--set", nargs="*", help="parameter or setting overrides key=value")
    a = ap.parse_args(argv)
    over = _parse_set(a.set)
    params = {k: v for k, v in over.items() if k in FIDUCIAL}
    settings = {k: v for k, v in over.items() if k in DEFAULT_SETTINGS}
    unknown = set(over) - set(params) - set(settings)
    if unknown:
        raise SystemExit(f"unknown keys: {sorted(unknown)}")
    settings.update(kernel_mode=a.kernel_mode, hmf_matter=a.hmf_matter,
                    hmf_alpha_mode=a.hmf_alpha_mode)
    t0 = time.time()
    ref = ClusterReference(params, settings, verbose=True)
    res = ref.results(with_cov=a.cov, with_nonlimber=a.nonlimber_check)
    np.savez(a.out, **res)
    N = res["N"]
    print("N_iA =")
    print(np.array2string(N, precision=1, suppress_small=True))
    print(f"wrote {a.out} ({len(res)} arrays) in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
