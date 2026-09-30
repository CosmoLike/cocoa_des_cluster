"""Cluster ingredients of the DES cluster reference (2503.13631 model):
mass-observable relation, richness-bin mass integrals n_A(z), b_A(z),
P^1h_A(k, z), photo-z selection kernels, counts N_iA and the radial
kernels of the cluster two-point functions.

Model (equation numbers of 2503.13631):

  (19) <ln lambda|M,z> = ln lambda_0 + A ln(M/M_piv) + B ln((1+z)/1.45),
       M_piv = 5e14 M_sun/h (lighthouse cluster_util.c:613)
  (18) sigma^2_lnlambda = sigma_int^2 + (e^mu - 1)/e^(2 mu)   [only if mu > 0]
       P(lambda in [l1, l2) | M, z) = [erf((ln l2 - mu)/(sqrt2 s))
                                      - erf((ln l1 - mu)/(sqrt2 s))]/2
  n_A(z)    = int dlnM dn/dlnM P(A|M,z)                      [(h/Mpc)^3]
  (21) b_A(z)    = int dlnM dn/dlnM P(A|M,z) b_h(M,z) / n_A(z)
  (22) P^1h_A(k,z) = int dlnM dn/dlnM P(A|M,z) (M/rho_m) u(k|M,z) / n_A(z)
  <phi_i|z> = [erf((z_hi - z)/(sqrt2 s_z)) - erf((z_lo - z)/(sqrt2 s_z))]/2,
              s_z = sigma_z0 (1+z)       (or a top hat in z)
  (16) N_iA = Omega_s int dz chi^2 (c/H) <phi_i|z> n_A(z),  Omega_s in sr

Radial kernels of the 2-pt functions (dN/dz normalized to unit integral):
  kernel_mode 0 (default, "volume", Y1 eq. 15, lighthouse):
      q_i(z)  ~ chi^2/H <phi_i|z>
  kernel_mode 1 ("abundance", counts-consistent):
      q_iA(z) ~ chi^2/H <phi_i|z> n_A(z)

Quadratures are composite Gauss-Legendre ("gl_panels") with the
breakpoints of every non-smooth feature honoured.
"""

import numpy as np
from scipy.special import erf

from ref_cosmology import COVERH0
from ref_halo import u_nfw

DEG2_TO_SR = (np.pi / 180.0) ** 2


def gl_panels(breaks, max_width, order=8):
    """Composite Gauss-Legendre nodes and weights on [breaks[0], breaks[-1]].

    Every interval between consecutive breaks is split into equal panels
    no wider than max_width; each panel carries `order` GL nodes.
    """
    x0, w0 = np.polynomial.legendre.leggauss(order)
    breaks = np.unique(np.asarray(breaks, dtype=float))
    xs, ws = [], []
    for a, b in zip(breaks[:-1], breaks[1:]):
        n = max(1, int(np.ceil((b - a) / max_width - 1e-9)))
        e = np.linspace(a, b, n + 1)
        half = 0.5 * np.diff(e)
        mid = 0.5 * (e[1:] + e[:-1])
        xs.append((mid[:, None] + half[:, None] * x0[None, :]).ravel())
        ws.append((half[:, None] * w0[None, :]).ravel())
    return np.concatenate(xs), np.concatenate(ws)


def rev_cumtrapz(y, x):
    """R(x_i) = int_{x_i}^{x_end} y dx (trapezoid), same length as x."""
    seg = 0.5 * (y[1:] + y[:-1]) * np.diff(x)
    out = np.zeros_like(y)
    out[:-1] = np.cumsum(seg[::-1])[::-1]
    return out


def lensing_efficiency(z_eval, chi_eval, zf, nf, chif):
    """g(z) = int_z^inf dz' n(z') (1 - chi(z)/chi(z')) from fine samples.

    zf ascending, nf = n(zf) (dN/dz), chif = chi(zf). Linear interpolation
    of the two reverse cumulative integrals (they are smooth).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        n_over_chi = np.where(chif > 0, nf / chif, 0.0)
    Q0 = rev_cumtrapz(nf, zf)
    Q1 = rev_cumtrapz(n_over_chi, zf)
    q0 = np.interp(z_eval, zf, Q0)
    q1 = np.interp(z_eval, zf, Q1)
    g = q0 - chi_eval * q1
    return np.where(z_eval < zf[-1], np.maximum(g, 0.0), 0.0)


class ClusterModel:
    """Everything that depends on the cluster sample definition.

    mor = dict(lnlambda0, A, B, sigma_int)   (Table I of 2503.13631 names:
          ln lambda_0, A_lnlambda, B_lnlambda, sigma_int)
    """

    def __init__(self, cosmo, halo, mor, lambda_edges=(20, 30, 45, 60, 500),
                 zc_edges=(0.2, 0.4, 0.55, 0.65), area_deg2=4143.0,
                 photoz="gaussian", sigma_z0=0.006, M_piv=5e14, z_piv=1.45,
                 lnM_min=np.log(1e12), lnM_max=np.log(1e16),
                 mass_panel_width=0.25, mass_order=8, nsig_support=10.0):
        self.cosmo = cosmo
        self.halo = halo
        self.mor = dict(mor)
        self.lam_edges = np.asarray(lambda_edges, dtype=float)
        self.zc_edges = np.asarray(zc_edges, dtype=float)
        self.nA = len(self.lam_edges) - 1
        self.nzc = len(self.zc_edges) - 1
        self.area_deg2 = float(area_deg2)
        self.Omega_s = self.area_deg2 * DEG2_TO_SR
        self.photoz = photoz
        self.sigma_z0 = float(sigma_z0)
        self.M_piv = float(M_piv)
        self.z_piv = float(z_piv)
        self.lnM_range = (float(lnM_min), float(lnM_max))
        self.lnM, self.wM = gl_panels([lnM_min, lnM_max], mass_panel_width, mass_order)
        self.nsig = nsig_support

    # ------------------------------------------------------------------
    # mass-observable relation
    # ------------------------------------------------------------------
    def mor_mean_sigma(self, lnM, z):
        m = self.mor
        mu = (m["lnlambda0"] + m["A"] * (lnM - np.log(self.M_piv))
              + m["B"] * np.log((1.0 + z) / self.z_piv))
        var = m["sigma_int"] ** 2 + np.where(mu > 0, (np.exp(mu) - 1.0) / np.exp(2.0 * mu), 0.0)
        return mu, np.sqrt(var)

    def p_lambda_bins(self, lnM, z):
        """P(lambda in A | M, z), shape (nA,) + broadcast(lnM, z).shape."""
        mu, s = self.mor_mean_sigma(lnM, z)
        ll = np.log(self.lam_edges)
        cdf = 0.5 * (1.0 + erf((ll.reshape((-1,) + (1,) * np.ndim(mu)) - mu) / (np.sqrt(2.0) * s)))
        return cdf[1:] - cdf[:-1]

    # ------------------------------------------------------------------
    # mass integrals
    # ------------------------------------------------------------------
    def _mass_grid(self, z):
        z = np.atleast_1d(np.asarray(z, dtype=float))
        lnM = self.lnM[:, None]
        nu = self.halo.nu(lnM, z[None, :])
        dn = self.halo.dndlnM(lnM, z[None, :], nu=nu)              # (M, z)
        P = self.p_lambda_bins(lnM, z[None, :])                    # (A, M, z)
        return z, nu, dn, P

    def n_b(self, z):
        """n_A(z) [(h/Mpc)^3] and b_A(z), each shape (nA, nz)."""
        z, nu, dn, P = self._mass_grid(z)
        wdn = self.wM[:, None] * dn
        n = np.einsum("mz,amz->az", wdn, P)
        b = np.einsum("mz,amz->az", wdn * self.halo.bias(None, None, nu=nu), P) / n
        return n, b

    def mean_mass(self, z):
        z, nu, dn, P = self._mass_grid(z)
        wdn = self.wM[:, None] * dn
        n = np.einsum("mz,amz->az", wdn, P)
        return np.einsum("mz,amz->az", wdn * np.exp(self.lnM)[:, None], P) / n

    def p1h(self, k, z):
        """P^1h_A(k, z) at one redshift z for an array k (h/Mpc): shape (nA, nk).

        u(k|M) is the truncated NFW with the Bhattacharya concentration;
        no bias, no miscentering; normalized by n_A(z).
        """
        z = float(z)
        k = np.asarray(k, dtype=float)
        zz, nu, dn, P = self._mass_grid(np.array([z]))
        nu, dn, P = nu[:, 0], dn[:, 0], P[:, :, 0]
        c = self.halo.conc(self.lnM, z, nu=nu)
        rd = self.halo.r_delta(self.lnM)
        wdn = self.wM * dn
        n = P @ wdn
        wts = P * (wdn * np.exp(self.lnM) / self.halo.rho_m)[None, :] / n[:, None]   # (A, M)
        U = u_nfw(k.reshape(-1)[:, None], rd[None, :], c[None, :])              # (nk, M)
        return (wts @ U.T).reshape((self.nA,) + k.shape)

    # ------------------------------------------------------------------
    # selection in redshift
    # ------------------------------------------------------------------
    def phi(self, z, i):
        """<phi_i|z_true>: probability that z_lambda falls in bin i."""
        z = np.asarray(z, dtype=float)
        lo, hi = self.zc_edges[i], self.zc_edges[i + 1]
        if self.photoz == "tophat":
            return ((z >= lo) & (z < hi)).astype(float)
        s = self.sigma_z0 * (1.0 + z)
        return 0.5 * (erf((hi - z) / (np.sqrt(2.0) * s)) - erf((lo - z) / (np.sqrt(2.0) * s)))

    def support(self, i):
        lo, hi = self.zc_edges[i], self.zc_edges[i + 1]
        if self.photoz == "tophat":
            return lo, hi
        s = self.nsig * self.sigma_z0
        return max(1e-4, lo - s * (1.0 + lo)), (hi + s * (1.0 + hi)) / (1.0 - s)

    def breaks(self, i):
        a, b = self.support(i)
        return sorted({a, b, self.zc_edges[i], self.zc_edges[i + 1]})

    def dVdz(self, z):
        """chi^2 dchi/dz, comoving volume per unit z per sr [(Mpc/h)^3]."""
        return self.cosmo.chi(z) ** 2 * self.cosmo.dchi_dz(z)

    # ------------------------------------------------------------------
    # counts
    # ------------------------------------------------------------------
    def counts(self, z_panel=0.005, order=8):
        """N_iA, shape (nzc, nA)."""
        N = np.zeros((self.nzc, self.nA))
        for i in range(self.nzc):
            z, w = gl_panels(self.breaks(i), z_panel, order)
            n, _ = self.n_b(z)
            N[i] = self.Omega_s * n @ (w * self.dVdz(z) * self.phi(z, i))
        return N

    def counts_weighted_bias(self, z_panel=0.005, order=8):
        """Count-weighted effective bias b_iA = int dV phi n_A b_A / int dV phi n_A."""
        out = np.zeros((self.nzc, self.nA))
        for i in range(self.nzc):
            z, w = gl_panels(self.breaks(i), z_panel, order)
            n, b = self.n_b(z)
            ww = w * self.dVdz(z) * self.phi(z, i)
            out[i] = (n * b) @ ww / (n @ ww)
        return out

    # ------------------------------------------------------------------
    # radial kernels of the two-point functions
    # ------------------------------------------------------------------
    def kernel_norms(self, kernel_mode=0, z_panel=0.005, order=8):
        """int dz chi^2/H <phi_i|z> [n_A(z)], shape (nzc, nA)."""
        out = np.zeros((self.nzc, self.nA))
        for i in range(self.nzc):
            z, w = gl_panels(self.breaks(i), z_panel, order)
            base = w * self.dVdz(z) * self.phi(z, i)
            if kernel_mode == 0:
                out[i] = base.sum()
            else:
                n, _ = self.n_b(z)
                out[i] = n @ base
        return out

    def q(self, z, i, kernel_mode=0, norms=None, nb=None):
        """Normalized dN/dz of cluster bin i, shape (nA, nz); for kernel_mode 0
        all richness bins share it. `nb` = (n, b) at z (optional cache)."""
        z = np.asarray(z, dtype=float)
        if norms is None:
            norms = self.kernel_norms(kernel_mode)
        base = self.dVdz(z) * self.phi(z, i)
        if kernel_mode == 0:
            return np.broadcast_to(base / norms[i, 0], (self.nA,) + z.shape).copy()
        n = self.n_b(z)[0] if nb is None else nb[0]
        return n * base[None, :] / norms[i][:, None]

    def magnification_efficiency(self, z_eval, i, kernel_mode=0, norms=None, dz_fine=5e-5):
        """g_c,iA(z) = int dz' q_iA(z') (1 - chi(z)/chi(z')), shape (nA, nz).

        The integral runs over the whole cluster n(z): the magnification of
        the clusters by everything in front of them (FULL foreground).
        """
        a, b = self.support(i)
        zf = np.arange(a, b + dz_fine, dz_fine)
        chif = self.cosmo.chi(zf)
        qf = self.q(zf, i, kernel_mode, norms)
        chi_e = self.cosmo.chi(z_eval)
        return np.array([lensing_efficiency(z_eval, chi_e, zf, qf[A], chif)
                         for A in range(self.nA)])
