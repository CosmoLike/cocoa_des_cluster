"""Gaussian covariance of the DES cluster data vector (reference).

Two-point blocks (full sky, per multipole, Knox with f_sky):

  Cov(C^{AB}_l, C^{CD}_l') = delta_ll' [C~^{AC} C~^{BD} + C~^{AD} C~^{BC}] / ((2l+1) f_sky)
  C~^{XY} = C^{XY} + delta_XY N^X,
    N^{c_iA} = 1/n_cA, n_cA = N_iA/Omega_s          (clusters, per sr)
    N^{g_i}  = 1/n_g,i                              (lenses)
    N^{s_j}  = sigma_e^2/n_eff,j, sigma_e PER COMPONENT (E-mode shear power)
  projected with the SAME bin-averaged kernels as the signal
  (ref_projection: w -> Pw, gamma_t -> Pg), l = 1 .. LMAX-1:
  Cov(xi_i, xi'_j) = sum_l K_i(l) K'_j(l) [..] / ((2l+1) f_sky).
  The [..] is interpolated in ln l between the spectra nodes with the
  same cubic spline the signal uses (exact band weights, no binning).
  cs blocks are then Y-transformed: Cov_Sigma = R Cov R^T,
  R = blockdiag(T on each cs theta vector, 1 elsewhere). No selection
  factor enters the covariance (it multiplies the model only).

Counts:
  Cov(N_iA, N_jB) = delta_ij [delta_AB N_iA + N_iA N_iB S_i,AB]
  S_i,AB = sum_l (2l+1)/(4 pi) W_l^2 C_l^{iA,iB}
  WINDOW APPROXIMATION: the footprint is a spherical cap of the survey
  area (1 - cos theta_c = Omega_s/(2 pi)), W_l = (2 pi/Omega_s)
  [P_{l-1}(cos theta_c) - P_{l+1}(cos theta_c)]/(2l+1), W_0 = 1; the
  shell is the count-weighted radial distribution of bin (i, A) times
  b_A(z) D(z); linear P(k, 0); exact spherical-Bessel C_l for l <= 20,
  Limber above, l <= 300. No cross-z-bin (i != j) term (as instructed).
  S_i,AA / b_iA^2 is the "sigma_i^2" of the footprint x shell.
  N x 2pt = 0.

Data-vector layout (lighthouse inner ordering):
  N  [zc][lambda]                                       nzc*nA
  cs [(zc, zs) pair, zc-major][lambda][theta]           nzc*ns*nA*Nt
  cc [zc][lambda1 <= lambda2][theta]                    nzc*nA(nA+1)/2*Nt
  cg [(zc, zg = zc) pair][lambda][theta]                nzc*nA*Nt
"""

import numpy as np
from scipy.interpolate import CubicSpline

from ref_projection import projection_kernels, legendre_table
from ref_nonlimber import cl_exact_linear, cl_limber_linear, k_grid

ARCMIN2_TO_SR = (np.pi / 180.0 / 60.0) ** 2


def cc_pairs(nA):
    return [(a, b) for a in range(nA) for b in range(a, nA)]


def observables(nzc, nA, ns):
    """List of (block, legs, kernel) in data-vector order (2pt part only).

    legs are field labels ('c', i, A), ('s', j), ('g', i)."""
    obs = []
    for i in range(nzc):
        for s in range(ns):
            for A in range(nA):
                obs.append(("cs", (("c", i, A), ("s", s)), "gt", (i, s, A)))
    for i in range(nzc):
        for (A, B) in cc_pairs(nA):
            obs.append(("cc", (("c", i, A), ("c", i, B)), "w", (i, A, B)))
    for i in range(nzc):
        for A in range(nA):
            obs.append(("cg", (("c", i, A), ("g", i)), "w", (i, A)))
    return obs


def band_matrices(ells, edges, lmax):
    """M[(a,b)][i, j, node] = sum_l K^a_i K^b_j S_node(l)/(2l+1),
    S_node the cubic-spline basis in ln l on the spectra nodes."""
    Pw, Pg = projection_kernels(edges, lmax)
    K = {"w": Pw[:, 1:], "gt": Pg[:, 1:]}
    l = np.arange(1, lmax, dtype=float)
    basis = CubicSpline(np.log(ells), np.eye(ells.size), axis=0)(np.log(l))   # (L, nb)
    inv = 1.0 / (2.0 * l + 1.0)
    out = {}
    for a in ("w", "gt"):
        for b in ("w", "gt"):
            if (b, a) in out:
                out[(a, b)] = out[(b, a)].transpose(1, 0, 2)
                continue
            nt = K[a].shape[0]
            M = np.empty((nt, nt, ells.size))
            for i in range(nt):
                M[i] = (K[a][i][None, :] * inv[None, :] * K[b]) @ basis
            out[(a, b)] = M
    return out


class GaussianCovariance:
    """Assemble the Gaussian covariance from the reference spectra."""

    def __init__(self, spectra, counts, Omega_s, edges, T, lmax=75000,
                 n_lens_arcmin2=(0.1380, 0.1016, 0.1071), n_src_arcmin2=(2.1402, 2.14455, 2.1518, 2.11845),
                 sigma_e=0.384666 / np.sqrt(2.0), lens_bins=(0, 1, 2)):
        self.sp = spectra
        self.N = np.asarray(counts)
        self.nzc, self.nA = self.N.shape
        self.ns = spectra["C_ss"].shape[0]
        self.lens_bins = list(lens_bins)
        self.Omega_s = Omega_s
        self.fsky = Omega_s / (4.0 * np.pi)
        self.edges = edges
        self.T = T
        self.lmax = lmax
        self.n_lens = np.asarray(n_lens_arcmin2) / ARCMIN2_TO_SR
        self.n_src = np.asarray(n_src_arcmin2) / ARCMIN2_TO_SR
        self.sigma_e = sigma_e

    # field bookkeeping --------------------------------------------------
    def _fields(self):
        f = [("c", i, A) for i in range(self.nzc) for A in range(self.nA)]
        f += [("s", s) for s in range(self.ns)]
        f += [("g", g) for g in range(len(self.lens_bins))]
        return f, {x: n for n, x in enumerate(f)}

    def field_spectra(self, with_noise=True):
        """C~[X, Y, node] for every pair of fields."""
        sp = self.sp
        fields, idx = self._fields()
        nb = sp["ells"].size
        C = np.zeros((len(fields), len(fields), nb))
        nc = self.nzc * self.nA
        C[:nc, :nc] = sp["C_cc"].reshape(nc, nc, nb)
        cs = sp["C_cs"].reshape(nc, self.ns, nb)
        C[:nc, nc:nc + self.ns] = cs
        C[nc:nc + self.ns, :nc] = cs.transpose(1, 0, 2)
        cg = sp["C_cg"].reshape(nc, -1, nb)
        g0 = nc + self.ns
        C[:nc, g0:] = cg
        C[g0:, :nc] = cg.transpose(1, 0, 2)
        C[nc:g0, nc:g0] = sp["C_ss"]
        C[g0:, nc:g0] = sp["C_gs"]
        C[nc:g0, g0:] = sp["C_gs"].transpose(1, 0, 2)
        C[g0:, g0:] = sp["C_gg"]
        if with_noise:
            nden = (self.N / self.Omega_s).ravel()
            for n in range(nc):
                C[n, n] += 1.0 / nden[n]
            for s in range(self.ns):
                C[nc + s, nc + s] += self.sigma_e**2 / self.n_src[s]
            for g, gb in enumerate(self.lens_bins):
                C[g0 + g, g0 + g] += 1.0 / self.n_lens[gb]
        return C, idx

    def twopoint(self, y_transform=True):
        obs = observables(self.nzc, self.nA, self.ns)
        C, idx = self.field_spectra()
        M = band_matrices(self.sp["ells"], self.edges, self.lmax)
        nt = self.edges.size - 1
        a = np.array([idx[o[1][0]] for o in obs])
        b = np.array([idx[o[1][1]] for o in obs])
        # G[o1, o2, node] = C~[a1,a2] C~[b1,b2] + C~[a1,b2] C~[b1,a2]
        G = (C[a[:, None], a[None, :]] * C[b[:, None], b[None, :]]
             + C[a[:, None], b[None, :]] * C[b[:, None], a[None, :]])
        kinds = np.array([o[2] for o in obs])
        no = len(obs)
        cov = np.zeros((no, nt, no, nt))
        for ka in ("w", "gt"):
            ia = np.where(kinds == ka)[0]
            for kb in ("w", "gt"):
                ib = np.where(kinds == kb)[0]
                cov[np.ix_(ia, np.arange(nt), ib, np.arange(nt))] = np.einsum(
                    "ijn,pqn->piqj", M[(ka, kb)], G[np.ix_(ia, ib)], optimize=True)
        cov = cov.reshape(no * nt, no * nt) / self.fsky
        if y_transform:
            R = np.eye(no * nt)
            for o, ob in enumerate(obs):
                if ob[0] == "cs":
                    sl = slice(o * nt, (o + 1) * nt)
                    R[sl, sl] = self.T
            cov = R @ cov @ R.T
        return cov, obs

    # counts --------------------------------------------------------------
    @staticmethod
    def cap_window(Omega_s, lmax):
        c = 1.0 - Omega_s / (2.0 * np.pi)
        P = legendre_table(np.array([c]), lmax + 1)[0]
        l = np.arange(lmax + 1)
        W = np.empty(lmax + 1)
        W[0] = 1.0
        W[1:] = 2.0 * np.pi / Omega_s * (P[l[1:] - 1] - P[l[1:] + 1]) / (2 * l[1:] + 1)
        return W


def counts_covariance(ref, l_exact=20, l_max=300, dchi=0.5):
    """delta_ij [delta_AB N_iA + N_iA N_iB S_i,AB] and diagnostics.

    `ref` is a reference_cluster.ClusterReference."""
    cl = ref.cluster
    cosmo = ref.cosmo
    N = ref.counts()
    W = GaussianCovariance.cap_window(cl.Omega_s, l_max)
    cov = np.zeros((cl.nzc * cl.nA, cl.nzc * cl.nA))
    S_all = np.zeros((cl.nzc, cl.nA, cl.nA))
    ells_ex = np.arange(0, l_exact + 1)
    ells_lim = np.arange(l_exact + 1, l_max + 1)
    for i in range(cl.nzc):
        za, zb = cl.support(i)
        chi = np.arange(cosmo.chi(za), cosmo.chi(zb), dchi)
        z = cosmo.z_of_chi(chi)
        n, b = cl.n_b(z)
        dNdchi = cl.Omega_s * chi**2 * cl.phi(z, i)[None, :] * n          # (A, chi)
        D = cosmo.growth(z)
        h = dNdchi / N[i][:, None] * b * D[None, :]
        Cex = cl_exact_linear(cosmo, ells_ex, chi, h, k=k_grid(kmax=0.25))
        Clim = cl_limber_linear(cosmo, ells_lim, chi, h)
        Cl = np.concatenate([Cex, Clim], axis=0)                          # (l, A, B)
        l = np.arange(0, l_max + 1)
        S = np.einsum("l,lab->ab", (2 * l + 1) / (4.0 * np.pi) * W**2, Cl)
        S_all[i] = S
        sl = slice(i * cl.nA, (i + 1) * cl.nA)
        cov[sl, sl] = np.diag(N[i]) + np.outer(N[i], N[i]) * S
    beff = cl.counts_weighted_bias()
    sigma2 = np.array([[S_all[i, A, A] / beff[i, A] ** 2 for A in range(cl.nA)]
                       for i in range(cl.nzc)])
    return cov, dict(S=S_all, b_eff=beff, sigma2_eff=sigma2, W_l=W)
