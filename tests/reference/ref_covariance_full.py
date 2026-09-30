"""Gaussian covariance of the FULL joint DES cluster data vector
(reference; ss, gs, gg, cg, N, cc, cs of the likelihood's IPCluster).

Joint layout (generic_interface_cluster.hpp; Nt theta bins, NL richness
bins, NRP = NL(NL+1)/2 richness pairs; blocks in this order):

  ss  xi+ then xi-  [source pair i <= j, i outer][theta]      2 Nt ns(ns+1)/2
  gs  gamma_t       [(lens, source) pair, lens-major][theta]  Nt nl ns
  gg  w_gg          [lens bin][theta]                         Nt nl
  cg  w_cg          [(zc, zg = cg_lens_bins[zc])][lambda][theta]  Nt NL ncg
  N   counts        [zc][lambda]                              nzc NL
  cc  w_cc          [zc][lambda1 <= lambda2][theta]           Nt NRP nzc
  cs  Sigma         [(zc, zs) pair, cluster-major][lambda][theta]  Nt NL nzc ns

Two-point blocks (full sky, per multipole, Knox with f_sky), for fields
A, B, C, D in {cluster c_iA, lens g_k, source shear s_j}:

  Cov(C^{AB}_l, C^{CD}_l') = delta_ll' [C~^{AC} C~^{BD} + C~^{AD} C~^{BC}] / ((2l+1) f_sky)
  C~^{XY} = C^{XY} + delta_XY N^X,
    N^{c_iA} = 1/n_cA, n_cA = N_iA/Omega_s          (clusters, per sr)
    N^{g_k}  = 1/n_g,k                              (lenses)
    N^{s_j}  = sigma_e^2/n_eff,j, sigma_e PER COMPONENT (E and B modes)

  projected with the bin-averaged full-sky kernels of the model:
    w       : Pw_i(l) = (2l+1)/(4 pi) <P_l>_bin          (ref_projection)
    gamma_t : Pg_i(l) = (2l+1)/(4 pi l(l+1)) <P_l^2>_bin (ref_projection)
    xi+/-   : G+/-_i(l), cosmo2D.c xi_pm_tomo (xi_pm_kernels below)
  Cov(O_i, O'_j) = sum_{l=1}^{LMAX-1} K_i(l) K'_j(l) [..]_l / ((2l+1) f_sky),
  with [..] interpolated in ln l between the spectra nodes by the cubic
  spline the signal uses (band_matrices: exact band weights, no binning),
  exactly as ref_covariance.py does for its cluster-only blocks.

  Shear: xi+ = sum G+ (C^EE + C^BB), xi- = sum G- (C^EE - C^BB), and the
  B modes carry shape noise only, so every xi x xi pair also gets
  +/- [N^B_ac N^B_bd + N^B_ad N^B_bc] (+ for ++ and --, - for +-). gamma_t
  sees the E modes only.

  Conventions of the spectra: C^{ss}, C^{gs}, C^{cs} are the model's
  (cosmo2D.c: (l-1)l(l+1)(l+2)/(l+1/2)^4 per shear pair, the square root
  per shear leg); the kernels are the model's too, so the covariance is
  that of the data vector. The shape noise is added to that C^{ss} with no
  l-dependent factor (the ref_covariance.py convention; the exact field
  rescaling would divide it by (l-1)(l+2)/(l(l+1)), a < 2% change for
  l >= 10).

Data-vector-level factors (the model multiplies after the projection):
  cs: Sigma = B_i(theta) (T gamma_t) (1 + m_j);  cg: B_i w_cg;  cc: B_i^2 w_cc;
  ss: (1+m_i)(1+m_j) xi;  gs: (1 + m_j) gamma_t.
  selection_mode = "signal" (default): the observed cluster field is
    B(theta) x (the unbiased cluster SIGNAL) + (unscaled shot noise), the
    picture of the selection bias as an extra large-scale bias (eq 23).
    Each Wick term splits into signal x signal, signal x noise and
    noise x noise pieces; a row leg in a signal piece carries its factor
    (B for clusters, 1 + m for shear, 1 for lenses), a leg in a noise
    piece carries 1 (clusters) or 1 + m (shear). This is the exact
    Gaussian covariance of that random vector: positive semi-definite by
    construction, and equal to the selection-free one when B = 1.
  selection_mode = "jacobian": R C R^T with R = diag(the whole model
    factor): shot noise scaled by B as well (an upper bound).
  selection_mode = "none": no data-vector factor at all (ref_covariance.py).
  The Y transform acts on the gamma_t space of every cs row BEFORE the
  selection factor, as in the model: Cov_Sigma = T Cov_gamma_t T^T per cs
  block (eq 31 of 2503.13631), T of ref_projection (the same matrix for
  every cs block).

Counts: Cov(N_iA, N_jB) = delta_ij [delta_AB N_iA + N_iA N_iB S_i,AB]
  (ref_covariance.counts_covariance: Poisson + sample variance, spherical
  cap window); N x 2pt = 0. Stated omissions: super-sample covariance of
  the 2pt blocks, connected trispectrum, N x 2pt SSC.
"""

import numpy as np
from scipy.interpolate import CubicSpline

from ref_covariance import ARCMIN2_TO_SR, cc_pairs, counts_covariance
from ref_projection import legendre_table, projection_kernels

BLOCKS = ("ss", "gs", "gg", "cg", "N", "cc", "cs")
KINDS = ("xip", "xim", "gt", "w")
SELECTION_MODES = ("signal", "jacobian", "none")


# ----------------------------------------------------------------------
# xi+/- kernels (cosmo2D.c xi_pm_tomo)
# ----------------------------------------------------------------------
_XI_CACHE = {}


def legendre_and_derivative(x, lmax):
    """P_l(x) and P_l'(x) for l = 0..lmax (inclusive), shape (len(x), lmax+1).

    P_l from the three-term recurrence (ref_projection.legendre_table);
    P_l' from the exact identity P'_{l+1} = P'_{l-1} + (2l+1) P_l
    (no division by 1 - x^2, which is ~5e-7 at the 2.5' edge)."""
    P = legendre_table(x, lmax)
    dP = np.zeros_like(P)
    if lmax >= 1:
        dP[:, 1] = 1.0
    for l in range(1, lmax):
        dP[:, l + 1] = dP[:, l - 1] + (2 * l + 1) * P[:, l]
    return P, dP


def xi_pm_kernels(edges, lmax):
    """(G+, G-), each (Ntheta, lmax): column l multiplies C_l; l = 0 is 0.

    Bin average over x = cos(theta) in [x_max, x_min] (x_min = cos of the
    lower theta edge) of the un-integrated kernel

      (2l+1)/(2 pi l^2 (l+1)^2) G_l^{+/-}(x),
      G_l^{+/-}(x) = (1/2) (l+2)!/(l-2)! d^l_{2,+/-2}(x)
                   = l^2(l^2-1)/2 P_l - l(l-1) x P_l' + (4-l) P_l''
                     + (l+2) x P_{l-1}'' +/- 2[(l-1) x P_l'' - (l+2) P_{l-1}''],

    integrated term by term with  int P_l = (P_{l+1} - P_{l-1})/(2l+1),
    int x P_l' = x P_l - int P_l,  int P_l'' = P_l',
    int x P_l'' = x P_l' - P_l,  and x P_l = [(l+1) P_{l+1} + l P_{l-1}]/(2l+1)
    to regroup the first two terms. The sum over l runs from 1 to lmax-1
    (lmax = the model's Ntable.LMAX). Checked against direct quadrature of
    Wigner d^l_{2,+/-2} (Jacobi polynomials) in test_covariance_full.py."""
    key = (tuple(np.round(edges, 15)), int(lmax))
    if key in _XI_CACHE:
        return _XI_CACHE[key]
    xe = np.cos(edges)
    P, dP = legendre_and_derivative(xe, lmax + 1)
    Pmin, Pmax, dPmin, dPmax = P[:-1], P[1:], dP[:-1], dP[1:]
    xmin, xmax = xe[:-1][:, None], xe[1:][:, None]
    l = np.arange(1, lmax, dtype=float)
    li = np.arange(1, lmax)
    even = (-l * (l - 1.0) / 2.0 * (l + 2.0 / (2.0 * l + 1.0))
            * (Pmin[:, li - 1] - Pmax[:, li - 1])
            - l * (l - 1.0) * (2.0 - l) / 2.0
            * (xmin * Pmin[:, li] - xmax * Pmax[:, li])
            + l * (l - 1.0) / (2.0 * l + 1.0) * (Pmin[:, li + 1] - Pmax[:, li + 1])
            + (4.0 - l) * (dPmin[:, li] - dPmax[:, li])
            + (l + 2.0) * (xmin * dPmin[:, li - 1] - xmax * dPmax[:, li - 1]
                           - Pmin[:, li - 1] + Pmax[:, li - 1]))
    odd = (2.0 * (l - 1.0) * (xmin * dPmin[:, li] - xmax * dPmax[:, li]
                              - Pmin[:, li] + Pmax[:, li])
           - 2.0 * (l + 2.0) * (dPmin[:, li - 1] - dPmax[:, li - 1]))
    pref = (2.0 * l + 1.0) / (2.0 * np.pi * l * l * (l + 1.0) ** 2) / (xmin - xmax)
    Gp = np.zeros((edges.size - 1, lmax))
    Gm = np.zeros_like(Gp)
    Gp[:, 1:] = pref * (even + odd)
    Gm[:, 1:] = pref * (even - odd)
    _XI_CACHE[key] = (Gp, Gm)
    return Gp, Gm


def kernels(edges, lmax):
    """{kind: K (Ntheta, lmax)} for every kind of the joint vector."""
    Pw, Pg = projection_kernels(edges, lmax)
    Gp, Gm = xi_pm_kernels(edges, lmax)
    return {"w": Pw, "gt": Pg, "xip": Gp, "xim": Gm}


def band_matrices(ells, edges, lmax, kinds=KINDS):
    """M[(a,b)][i, j, node] = sum_{l=1}^{lmax-1} K^a_i K^b_j S_node(l)/(2l+1),
    S_node the cubic-spline basis in ln l on the spectra nodes (the same
    arithmetic as ref_covariance.band_matrices, extended to xi+/-)."""
    Kall = kernels(edges, lmax)
    K = {k: Kall[k][:, 1:] for k in kinds}
    l = np.arange(1, lmax, dtype=float)
    basis = CubicSpline(np.log(ells), np.eye(ells.size), axis=0)(np.log(l))   # (L, nb)
    inv = 1.0 / (2.0 * l + 1.0)
    out = {}
    for a in kinds:
        for b in kinds:
            if (b, a) in out:
                out[(a, b)] = out[(b, a)].transpose(1, 0, 2)
                continue
            nt = K[a].shape[0]
            M = np.empty((nt, nt, ells.size))
            for i in range(nt):
                M[i] = (K[a][i][None, :] * inv[None, :] * K[b]) @ basis
            out[(a, b)] = M
    return out


# ----------------------------------------------------------------------
# layout
# ----------------------------------------------------------------------
def joint_layout(nzc, nA, ns, nl, nt, cg_lens_bins):
    """Two-point observables of the joint vector and the block geometry.

    Returns dict(obs=[(block, (leg1, leg2), kind, label)], sizes, starts,
    ndata, index (nobs, nt) joint index of every 2pt entry, N_index
    (nzc, nA) joint index of the counts). Legs are field labels
    ('c', i, A), ('s', j), ('g', k)."""
    obs = []
    shear_pairs = [(i, j) for i in range(ns) for j in range(i, ns)]
    for kind in ("xip", "xim"):
        for (i, j) in shear_pairs:
            obs.append(("ss", (("s", i), ("s", j)), kind, (kind, i, j)))
    for k in range(nl):
        for j in range(ns):
            obs.append(("gs", (("g", k), ("s", j)), "gt", (k, j)))
    for k in range(nl):
        obs.append(("gg", (("g", k), ("g", k)), "w", (k,)))
    cg_pairs = [(i, g) for i, g in enumerate(cg_lens_bins) if 0 <= g < nl]
    for (i, g) in cg_pairs:
        for A in range(nA):
            obs.append(("cg", (("c", i, A), ("g", g)), "w", (i, g, A)))
    for i in range(nzc):
        for (A, B) in cc_pairs(nA):
            obs.append(("cc", (("c", i, A), ("c", i, B)), "w", (i, A, B)))
    for i in range(nzc):
        for j in range(ns):
            for A in range(nA):
                obs.append(("cs", (("c", i, A), ("s", j)), "gt", (i, j, A)))
    sizes = dict(ss=2 * len(shear_pairs) * nt, gs=nl * ns * nt, gg=nl * nt,
                 cg=len(cg_pairs) * nA * nt, N=nzc * nA,
                 cc=nzc * len(cc_pairs(nA)) * nt, cs=nzc * ns * nA * nt)
    starts, s = {}, 0
    for b in BLOCKS:
        starts[b] = s
        s += sizes[b]
    index = np.zeros((len(obs), nt), dtype=int)
    first = {}
    for o, ob in enumerate(obs):
        first.setdefault(ob[0], o)
        index[o] = starts[ob[0]] + (o - first[ob[0]]) * nt + np.arange(nt)
    N_index = starts["N"] + np.arange(nzc * nA).reshape(nzc, nA)
    return dict(obs=obs, sizes=sizes, starts=starts, ndata=s, index=index,
                N_index=N_index, shear_pairs=shear_pairs, cg_pairs=cg_pairs)


# ----------------------------------------------------------------------
# covariance
# ----------------------------------------------------------------------
class FullGaussianCovariance:
    """Gaussian covariance of the joint vector from Limber spectra.

    spectra: dict with ells and C_cc (i,A,j,B,l), C_cs (i,A,s,l),
      C_cg (i,A,g,l), C_gg (g,g',l), C_gs (g,s,l), C_ss (s,s',l), the lens
      index g running over ALL lens bins (len(n_lens_arcmin2) of them).
    counts: N_iA (nzc, nA) for the cluster shot noise (and the counts
      block when counts_cov is None: diag(N), no sample variance).
    selection: B_i(theta) (nzc, Ntheta), None = ones.
    shear_m: (1 + m_j) calibration of the model, None = zeros.
    """

    def __init__(self, spectra, counts, Omega_s, edges, T, lmax=75000,
                 selection=None, shear_m=None,
                 n_lens_arcmin2=(0.1380, 0.1016, 0.1071, 0.1381, 0.1054, 0.1045),
                 n_src_arcmin2=(2.1402, 2.14455, 2.1518, 2.11845),
                 sigma_e=0.384666 / np.sqrt(2.0), cg_lens_bins=(0, 1, 2),
                 selection_mode="signal", counts_cov=None):
        if selection_mode not in SELECTION_MODES:
            raise ValueError(f"selection_mode {selection_mode!r} not in {SELECTION_MODES}")
        self.sp = spectra
        self.N = np.asarray(counts, dtype=float)
        self.nzc, self.nA = self.N.shape
        self.ns = spectra["C_ss"].shape[0]
        self.nl = len(n_lens_arcmin2)
        if spectra["C_gg"].shape[0] != self.nl or spectra["C_cg"].shape[2] != self.nl:
            raise ValueError("the spectra must cover every lens bin of n_lens_arcmin2")
        self.Omega_s = float(Omega_s)
        self.fsky = self.Omega_s / (4.0 * np.pi)
        self.edges = np.asarray(edges, dtype=float)
        self.nt = self.edges.size - 1
        self.T = np.asarray(T, dtype=float)
        self.lmax = int(lmax)
        self.B = (np.ones((self.nzc, self.nt)) if selection is None
                  else np.asarray(selection, dtype=float))
        self.m = np.zeros(self.ns) if shear_m is None else np.asarray(shear_m, dtype=float)
        self.n_lens = np.asarray(n_lens_arcmin2) / ARCMIN2_TO_SR
        self.n_src = np.asarray(n_src_arcmin2) / ARCMIN2_TO_SR
        self.sigma_e = float(sigma_e)
        self.cg_lens_bins = list(cg_lens_bins)
        self.selection_mode = selection_mode
        self.counts_cov = (np.diag(self.N.ravel()) if counts_cov is None
                           else np.asarray(counts_cov, dtype=float))
        self.layout = joint_layout(self.nzc, self.nA, self.ns, self.nl, self.nt,
                                   self.cg_lens_bins)

    # fields -----------------------------------------------------------
    def fields(self):
        f = [("c", i, A) for i in range(self.nzc) for A in range(self.nA)]
        f += [("s", s) for s in range(self.ns)]
        f += [("g", g) for g in range(self.nl)]
        return f, {x: n for n, x in enumerate(f)}

    def field_spectra(self):
        """Signal S[X, Y, node] of every field pair and the auto noise N[X]
        (E-mode noise for shear; the B-mode noise equals it)."""
        sp = self.sp
        fields, idx = self.fields()
        nb = sp["ells"].size
        nc, ns = self.nzc * self.nA, self.ns
        g0 = nc + ns
        S = np.zeros((len(fields), len(fields), nb))
        S[:nc, :nc] = sp["C_cc"].reshape(nc, nc, nb)
        cs = sp["C_cs"].reshape(nc, ns, nb)
        S[:nc, nc:g0] = cs
        S[nc:g0, :nc] = cs.transpose(1, 0, 2)
        cg = sp["C_cg"].reshape(nc, self.nl, nb)
        S[:nc, g0:] = cg
        S[g0:, :nc] = cg.transpose(1, 0, 2)
        S[nc:g0, nc:g0] = sp["C_ss"]
        S[g0:, nc:g0] = sp["C_gs"]
        S[nc:g0, g0:] = sp["C_gs"].transpose(1, 0, 2)
        S[g0:, g0:] = sp["C_gg"]
        noise = np.zeros(len(fields))
        noise[:nc] = self.Omega_s / self.N.ravel()            # 1/n_cA
        noise[nc:g0] = self.sigma_e**2 / self.n_src
        noise[g0:] = 1.0 / self.n_lens
        is_shear = np.zeros(len(fields), dtype=bool)
        is_shear[nc:g0] = True
        return S, noise, is_shear, idx

    def leg_factors(self):
        """sig[X, theta], noi[X, theta]: the data-vector factor a leg of
        field X carries in a signal piece and in a noise piece."""
        fields, _ = self.fields()
        sig = np.ones((len(fields), self.nt))
        noi = np.ones((len(fields), self.nt))
        if self.selection_mode == "none":
            return sig, noi
        for n, f in enumerate(fields):
            if f[0] == "c":
                sig[n] = self.B[f[1]]
                if self.selection_mode == "jacobian":
                    noi[n] = self.B[f[1]]
            elif f[0] == "s":
                sig[n] = 1.0 + self.m[f[1]]
                noi[n] = 1.0 + self.m[f[1]]
        return sig, noi

    # two-point part ---------------------------------------------------
    def _terms(self):
        """The Gaussian covariance as a sum of pieces
        G[o1, o2, node] x row(o1, theta) x col(o2, theta')."""
        obs = self.layout["obs"]
        S, noise, is_shear, idx = self.field_spectra()
        sig, noi = self.leg_factors()
        a = np.array([idx[o[1][0]] for o in obs])
        b = np.array([idx[o[1][1]] for o in obs])
        A_, B_ = a[:, None], b[:, None]          # rows
        C_, D_ = a[None, :], b[None, :]          # columns
        Nd = np.diag(noise)
        nb = S.shape[2]
        SS = S[A_, C_] * S[B_, D_] + S[A_, D_] * S[B_, C_]
        NN = (Nd[A_, C_] * Nd[B_, D_] + Nd[A_, D_] * Nd[B_, C_])[:, :, None] * np.ones(nb)
        # B-mode shape noise of xi+/- pairs (xi+ = EE + BB, xi- = EE - BB)
        kinds = np.array([o[2] for o in obs])
        sgn = np.zeros(len(obs))
        sgn[kinds == "xip"] = 1.0
        sgn[kinds == "xim"] = -1.0
        NB = np.diag(np.where(is_shear, noise, 0.0))
        BB = ((NB[A_, C_] * NB[B_, D_] + NB[A_, D_] * NB[B_, C_])
              * sgn[:, None] * sgn[None, :])
        NN = NN + BB[:, :, None]
        terms = [
            # (G, row factor per leg a, b; col factor per leg c, d)
            (SS, sig[a] * sig[b], sig[a] * sig[b]),
            (NN, noi[a] * noi[b], noi[a] * noi[b]),
            # Wick 1 (a-c, b-d): signal ac x noise bd, noise ac x signal bd
            ((S[A_, C_] * Nd[B_, D_][:, :, None]), sig[a] * noi[b], sig[a] * noi[b]),
            ((Nd[A_, C_][:, :, None] * S[B_, D_]), noi[a] * sig[b], noi[a] * sig[b]),
            # Wick 2 (a-d, b-c): signal ad x noise bc, noise ad x signal bc
            ((S[A_, D_] * Nd[B_, C_][:, :, None]), sig[a] * noi[b], noi[a] * sig[b]),
            ((Nd[A_, D_][:, :, None] * S[B_, C_]), noi[a] * sig[b], sig[a] * noi[b]),
        ]
        return terms

    def _project(self, G, M):
        """(nobs, nt, nobs, nt) = sum_node M[(k1,k2)][i,j,node] G[o1,o2,node]."""
        obs = self.layout["obs"]
        kinds = np.array([o[2] for o in obs])
        no, nt, nb = len(obs), self.nt, G.shape[2]
        out = np.zeros((no, nt, no, nt))
        for ka in KINDS:
            ia = np.where(kinds == ka)[0]
            if ia.size == 0:
                continue
            for kb in KINDS:
                ib = np.where(kinds == kb)[0]
                if ib.size == 0:
                    continue
                g = G[np.ix_(ia, ib)].reshape(ia.size * ib.size, nb)
                blk = (g @ M[(ka, kb)].reshape(nt * nt, nb).T).reshape(ia.size, ib.size, nt, nt)
                out[np.ix_(ia, np.arange(nt), ib, np.arange(nt))] = blk.transpose(0, 2, 1, 3)
        return out

    def _ytransform(self, C):
        """T on the theta index of every cs observable (rows and columns)."""
        cs = np.array([o[0] == "cs" for o in self.layout["obs"]])
        C[cs] = np.einsum("tu,oujv->otjv", self.T, C[cs])
        C[:, :, cs] = np.einsum("vu,itou->itov", self.T, C[:, :, cs])
        return C

    def twopoint(self, M=None, return_terms=False, symmetrize=True):
        """Two-point covariance in observable order, shape (nobs nt, nobs nt).

        symmetrize = False returns the assembly as it is (the tests check
        that it is symmetric: a factor on the wrong leg would break it)."""
        if M is None:
            M = band_matrices(self.sp["ells"], self.edges, self.lmax)
        no, nt = len(self.layout["obs"]), self.nt
        cov = np.zeros((no, nt, no, nt))
        pieces = []
        for G, row, col in self._terms():
            if not np.any(G):
                continue
            C = self._ytransform(self._project(G, M))
            C *= row[:, :, None, None] * col[None, None, :, :]
            cov += C
            if return_terms:
                pieces.append(C)
        cov = cov.reshape(no * nt, no * nt) / self.fsky
        if symmetrize:
            cov = 0.5 * (cov + cov.T)
        return (cov, pieces) if return_terms else cov

    def full(self, M=None):
        """The joint covariance (ndata, ndata) in the likelihood layout."""
        lay = self.layout
        c2 = self.twopoint(M)
        idx2 = lay["index"].ravel()
        cov = np.zeros((lay["ndata"], lay["ndata"]))
        cov[np.ix_(idx2, idx2)] = c2
        iN = lay["N_index"].ravel()
        cov[np.ix_(iN, iN)] = self.counts_cov
        return cov


# ----------------------------------------------------------------------
# convenience: from a ClusterReference
# ----------------------------------------------------------------------
def full_covariance(ref, selection_mode="signal", counts=None, cg_lens_bins=(0, 1, 2),
                    return_object=False, M=None):
    """Joint covariance at the point of a reference_cluster.ClusterReference
    whose settings["lens_bins"] cover every lens bin (0 .. nl-1).

    counts = None: the reference counts N_iA (Poisson, shot noise and the
    sample-variance weights of ref_covariance.counts_covariance); an array
    overrides N_iA in all three (the sample-variance S_i,AB is a property
    of the shell and does not depend on the normalization).
    M: band_matrices of the reference's ells and binning (None: computed)."""
    s, p = ref.settings, ref.params
    if list(s["lens_bins"]) != list(range(len(s["n_lens_arcmin2"]))):
        raise ValueError("full_covariance needs settings lens_bins = every lens bin")
    cN_ref, info = counts_covariance(ref)
    if counts is None:
        N = ref.counts()
        cN = cN_ref
    else:
        N = np.asarray(counts, dtype=float)
        cN = np.zeros_like(cN_ref)
        for i in range(N.shape[0]):
            sl = slice(i * N.shape[1], (i + 1) * N.shape[1])
            cN[sl, sl] = np.diag(N[i]) + np.outer(N[i], N[i]) * info["S"][i]
    fc = FullGaussianCovariance(
        ref.spectra(), N, ref.cluster.Omega_s, ref.edges, ref.T, s["lmax"],
        selection=ref.selection(), shear_m=p["shear_m"],
        n_lens_arcmin2=s["n_lens_arcmin2"], n_src_arcmin2=s["n_src_arcmin2"],
        sigma_e=s["sigma_e"], cg_lens_bins=cg_lens_bins,
        selection_mode=selection_mode, counts_cov=cN)
    cov = fc.full(M)
    info = dict(info, counts=N, counts_cov=cN, layout=fc.layout)
    return (cov, info, fc) if return_object else (cov, info)
