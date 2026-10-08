"""Gaussian covariance of the full joint DES cluster data vector
(reference; ss, gs, gg, cg, N, cc, cs of the likelihood's IPCluster).

scripts/make_synthetic_data.py writes this covariance (selection_mode
"signal", its default) to data/des_cluster_y6_cov.npy: the Gaussian
covariance of all seven blocks of the joint data vector that both
cluster likelihoods read (2812 entries for the project's binning). It
extends ref_covariance.py (cluster blocks only) to the cosmic shear
xi+/-, galaxy-galaxy lensing and galaxy clustering blocks, the shear
calibration and the selection factor; test_covariance_full.py checks
the pieces.

Joint layout (generic_interface_cluster.hpp; Nt theta bins, NL richness
bins, NRP = NL(NL+1)/2 richness pairs, ns source bins, nl lens bins,
nzc cluster z bins, ncg = number of (cluster z bin, lens bin) pairs of
w_cg; blocks in this order):

  ss  xi+ then xi-  [source pair i <= j, i outer][theta]      2 Nt ns(ns+1)/2
  gs  gamma_t       [(lens, source) pair, lens-major][theta]  Nt nl ns
  gg  w_gg          [lens bin][theta]                         Nt nl
  cg  w_cg          [(zc, zg = cg_lens_bins[zc])][lambda][theta]  Nt NL ncg
  N   counts        [zc][lambda]                              nzc NL
  cc  w_cc          [zc][lambda1 <= lambda2][theta]           Nt NRP nzc
  cs  Sigma         [(zc, zs) pair, cluster-major][lambda][theta]  Nt NL nzc ns

Two-point blocks (full sky, per multipole, Knox with f_sky), for fields
A, B, C, D in {cluster c_iA, lens g_k, source shear s_j}:

  Cov(C^{AB}_l, C^{CD}_l')
      = delta_ll' [C~^{AC} C~^{BD} + C~^{AD} C~^{BC}] / ((2l+1) f_sky)
  C~^{XY} = C^{XY} + delta_XY N^X,
    N^{c_iA} = 1/n_cA, n_cA = N_iA/Omega_s          (clusters, per sr)
    N^{g_k}  = 1/n_g,k                              (lenses)
    N^{s_j}  = sigma_e^2/n_eff,j, sigma_e per component (E and B modes)

  projected with the bin-averaged full-sky kernels of the model:
    w       : Pw_i(l) = (2l+1)/(4 pi) <P_l>_bin          (ref_projection)
    gamma_t : Pg_i(l) = (2l+1)/(4 pi l(l+1)) <P_l^2>_bin (ref_projection)
    xi+/-   : G+/-_i(l), cosmo2D.c xi_pm_tomo (xi_pm_kernels below)
  Cov(O_i, O'_j) = sum_{l=1}^{LMAX-1} K_i(l) K'_j(l) [..]_l / ((2l+1) f_sky),
  with [..] interpolated in ln l between the spectra nodes by the cubic
  spline the signal uses (band_matrices: exact band weights, no binning),
  exactly as ref_covariance.py does for its cluster-only blocks. The pure
  noise x noise piece (flat in l) is summed to l = infinity instead
  (noise = "exact", default): delta_ij [..]/(8 pi^2 Delta x_i f_sky), the
  real-space pair-count variance (FullGaussianCovariance._noise_exact);
  noise = "lsum" keeps the LMAX-truncated sum of ref_covariance.py.

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
    B(theta) x (the unbiased cluster signal) + (unscaled shot noise), the
    picture of the selection bias as an extra large-scale bias (eq 23).
    Each Wick term splits into signal x signal, signal x noise and
    noise x noise pieces; a row leg in a signal piece carries its factor
    (B for clusters, 1 + m for shear, 1 for lenses), a leg in a noise
    piece carries 1 (clusters) or 1 + m (shear). This is the exact
    Gaussian covariance of that random vector: positive semi-definite by
    construction, and equal to the selection-free one when B = 1.
  selection_mode = "jacobian": R C R^T with R = diag(the whole model
    factor): shot noise scaled by B as well (an upper bound:
    test_selection_modes checks that every diagonal entry of the "signal"
    mode lies between those of "none" and "jacobian").
  selection_mode = "none": no data-vector factor at all (ref_covariance.py).
  The Y transform acts on the gamma_t space of every cs row before the
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

# The block order of the joint data vector (README table), the kernel
# kinds (xi+, xi-, gamma_t, w), and the accepted values of the
# selection_mode and noise arguments of FullGaussianCovariance (module
# docstring).
BLOCKS = ("ss", "gs", "gg", "cg", "N", "cc", "cs")
KINDS = ("xip", "xim", "gt", "w")
SELECTION_MODES = ("signal", "jacobian", "none")
NOISE_MODES = ("exact", "lsum")


# ----------------------------------------------------------------------
# xi+/- kernels (cosmo2D.c xi_pm_tomo)
# ----------------------------------------------------------------------
# Module-global cache of xi_pm_kernels, keyed like ref_projection's
# _KERNEL_CACHE: (rounded edges, lmax) -> (G+, G-), computed once per
# Python process.
_XI_CACHE = {}


def legendre_and_derivative(x, lmax):
    """P_l(x) and P_l'(x) for l = 0..lmax (inclusive), shape (len(x), lmax+1).

    P_l from the three-term recurrence (ref_projection.legendre_table);
    P_l' from the exact identity P'_{l+1} = P'_{l-1} + (2l+1) P_l
    (no division by 1 - x^2, which is ~5e-7 at the 2.5' edge).

    Arguments:
      x = points in [-1, 1], 1-D array.
      lmax = highest degree, >= 0.

    Returns:
      (P, dP): two numpy arrays of shape (x.size, lmax + 1), P_0' = 0 and
      P_1' = 1 starting the recurrence.
    """
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
    Wigner d^l_{2,+/-2} (Jacobi polynomials) in test_covariance_full.py.

    Arguments:
      edges = angular bin edges in radians, (Ntheta + 1,).
      lmax = number of columns, l = 0 .. lmax - 1. Columns 0 and 1 are
             0: a spin-2 field has no l < 2 multipole (at l = 1 every
             term vanishes, through l - 1 or because P_0 = 1 and
             P_1' = 1 are constant).

    Returns:
      (G+, G-): numpy arrays of shape (Ntheta, lmax), the cached arrays
      themselves (a caller must not modify them).
    """
    # dict key: the edges rounded to 15 decimals as a tuple (hashable)
    key = (tuple(np.round(edges, 15)), int(lmax))
    if key in _XI_CACHE:
        return _XI_CACHE[key]
    xe = np.cos(edges)
    P, dP = legendre_and_derivative(xe, lmax + 1)
    # rows of the lower (x_min) and upper (x_max) bin edges; xmin, xmax as
    # columns (Ntheta, 1) that broadcast against the row of l values, li
    # the integer column indices
    Pmin, Pmax, dPmin, dPmax = P[:-1], P[1:], dP[:-1], dP[1:]
    xmin, xmax = xe[:-1][:, None], xe[1:][:, None]
    l = np.arange(1, lmax, dtype=float)
    li = np.arange(1, lmax)
    # even: the bin integral of the terms of G_l without the +/- sign;
    # odd: that of the bracket after +/-. G+ = pref (even + odd) and
    # G- = pref (even - odd), pref = (2l+1)/(2 pi l^2 (l+1)^2)/(x_min - x_max)
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
    """{kind: K (Ntheta, lmax)} for every kind of the joint vector.

    Arguments:
      edges = angular bin edges in radians.
      lmax = LMAX (columns l = 0 .. lmax - 1).

    Returns:
      dict with "w" and "gt" (ref_projection.projection_kernels) and
      "xip", "xim" (xi_pm_kernels); the arrays are the cached ones.
    """
    Pw, Pg = projection_kernels(edges, lmax)
    Gp, Gm = xi_pm_kernels(edges, lmax)
    return {"w": Pw, "gt": Pg, "xip": Gp, "xim": Gm}


def band_matrices(ells, edges, lmax, kinds=KINDS):
    """M[(a,b)][i, j, node] = sum_{l=1}^{lmax-1} K^a_i K^b_j S_node(l)/(2l+1),
    S_node the cubic-spline basis in ln l on the spectra nodes (the same
    arithmetic as ref_covariance.band_matrices, extended to xi+/-).

    Arguments:
      ells = spectra node multipoles, increasing, (nb,).
      edges = angular bin edges in radians.
      lmax = LMAX of the sums (l = 1 .. lmax - 1).
      kinds = kernel kinds to include (default KINDS, all four).

    Returns:
      dict keyed by the kernel pair (a, b); each value an array
      (Ntheta, Ntheta, nb).
    """
    Kall = kernels(edges, lmax)
    # the kernels without the column l = 0 (a dict comprehension over the
    # kinds)
    K = {k: Kall[k][:, 1:] for k in kinds}
    l = np.arange(1, lmax, dtype=float)
    # basis[l - 1, node] = S_node(l), the spline through the unit vector
    # of each node (ref_covariance.band_matrices)
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
    ('c', i, A), ('s', j), ('g', k).

    The dict also holds shear_pairs (the source pairs i <= j of the ss
    block) and cg_pairs (the (cluster z bin, lens bin) pairs of w_cg);
    sizes and starts are keyed by the block names of BLOCKS, and ndata is
    the length of the joint vector. test_layout_sizes_starts compares the
    layout with the compiled code.

    Arguments:
      nzc = number of cluster z bins.
      nA = number of richness bins.
      ns = number of source bins.
      nl = number of lens bins.
      nt = number of theta bins.
      cg_lens_bins = lens bin paired with each cluster z bin in w_cg; an
                     entry outside 0 .. nl - 1 (the C code uses -1)
                     means no w_cg pair for that cluster bin.
    """
    obs = []
    # source pairs (i, j) with i <= j, i outer (a list comprehension with
    # two for clauses)
    shear_pairs = [(i, j) for i in range(ns) for j in range(i, ns)]
    for kind in ("xip", "xim"):
        for (i, j) in shear_pairs:
            obs.append(("ss", (("s", i), ("s", j)), kind, (kind, i, j)))
    for k in range(nl):
        for j in range(ns):
            obs.append(("gs", (("g", k), ("s", j)), "gt", (k, j)))
    for k in range(nl):
        obs.append(("gg", (("g", k), ("g", k)), "w", (k,)))
    # (cluster z bin i, its lens bin g): enumerate yields the (i, g) pairs
    # and the `if` clause of the comprehension skips a missing lens bin
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
    # start of every block: the running sum of the sizes in BLOCKS order
    starts, s = {}, 0
    for b in BLOCKS:
        starts[b] = s
        s += sizes[b]
    # index[o, t] = joint position of observable o at theta bin t: the
    # block start plus nt times the rank of o inside its block.
    # first.setdefault(block, o) stores o only when the block has no entry
    # yet, so first[block] is the first observable of the block.
    index = np.zeros((len(obs), nt), dtype=int)
    first = {}
    for o, ob in enumerate(obs):
        first.setdefault(ob[0], o)
        index[o] = starts[ob[0]] + (o - first[ob[0]]) * nt + np.arange(nt)
    # the counts block, (nzc, nA) with A fastest
    N_index = starts["N"] + np.arange(nzc * nA).reshape(nzc, nA)
    return dict(obs=obs, sizes=sizes, starts=starts, ndata=s, index=index,
                N_index=N_index, shear_pairs=shear_pairs, cg_pairs=cg_pairs)


# ----------------------------------------------------------------------
# covariance
# ----------------------------------------------------------------------
class FullGaussianCovariance:
    """Gaussian covariance of the joint vector from Limber spectra.

    The constructor arguments are documented in __init__. full() returns
    the joint matrix and twopoint() its 2pt part. Every Wick term is
    split into signal and noise pieces (_terms), projected on the angular
    bins (_project, or _noise_exact for the pure noise piece),
    Y-transformed on the cs rows (_ytransform) and multiplied by the
    data-vector factors of its legs (leg_factors).
    """

    def __init__(self, spectra, counts, Omega_s, edges, T, lmax=75000,
                 selection=None, shear_m=None,
                 n_lens_arcmin2=(0.1380, 0.1016, 0.1071, 0.1381, 0.1054, 0.1045),
                 n_src_arcmin2=(2.1402, 2.14455, 2.1518, 2.11845),
                 sigma_e=0.384666 / np.sqrt(2.0), cg_lens_bins=(0, 1, 2),
                 selection_mode="signal", counts_cov=None, noise="exact"):
        """Check and store the inputs and build the joint layout.

        Arguments:
          spectra = dict with "ells" and C_cc (i,A,j,B,l), C_cs (i,A,s,l),
                    C_cg (i,A,g,l), C_gg (g,g',l), C_gs (g,s,l),
                    C_ss (s,s',l), the lens index g running over all lens
                    bins (len(n_lens_arcmin2) of them).
          counts = N_iA (nzc, nA) for the cluster shot noise (and the
                   counts block when counts_cov is None: diag(N), no
                   sample variance).
          Omega_s = survey solid angle in sr.
          edges = angular bin edges in radians, (Ntheta + 1,).
          T = Y-transform matrix (Ntheta, Ntheta) (ref_projection).
          lmax = LMAX of the multipole sums (default 75000).
          selection = B_i(theta) (nzc, Ntheta), the selection factor of
                      each cluster z bin; None = ones.
          shear_m = multiplicative shear calibration m_j of each source
                    bin, (ns,); a shear leg carries 1 + m_j. None = zeros.
          n_lens_arcmin2 = lens densities per arcmin^2, one per lens bin
                           (default: the six MagLim bins of lens_n_gal in
                           the lighthouse DES Y6 file dataY6.yaml).
          n_src_arcmin2 = source densities per arcmin^2 (default: the four
                          bins of source_n_gal in dataY6.yaml).
          sigma_e = shape-noise dispersion per component (default
                    0.384666/sqrt(2): the sigma_e of dataY6.yaml counts
                    both components).
          cg_lens_bins = lens bin paired with each cluster z bin in w_cg
                         (default (0, 1, 2); joint_layout).
          selection_mode = "signal" (default), "jacobian" or "none"
                           (module docstring).
          counts_cov = the counts block (nzc nA, nzc nA), e.g. from
                       ref_covariance.counts_covariance (None: diag(N),
                       Poisson only).
          noise = "exact" (default: the noise x noise piece summed over
                  all multipoles) or "lsum" (summed to LMAX, the
                  ref_covariance.py convention).

        Raises:
          ValueError for an unknown selection_mode or noise, or when C_gg
          or C_cg does not cover every lens bin of n_lens_arcmin2.
        """
        if selection_mode not in SELECTION_MODES:
            raise ValueError(f"selection_mode {selection_mode!r} not in {SELECTION_MODES}")
        if noise not in NOISE_MODES:
            raise ValueError(f"noise {noise!r} not in {NOISE_MODES}")
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
        self.noise = noise
        self.counts_cov = (np.diag(self.N.ravel()) if counts_cov is None
                           else np.asarray(counts_cov, dtype=float))
        self.layout = joint_layout(self.nzc, self.nA, self.ns, self.nl, self.nt,
                                   self.cg_lens_bins)

    # fields -----------------------------------------------------------
    def fields(self):
        """Return the field labels and the label-to-row lookup.

        Order: the clusters ("c", i, A) with i outer, then the sources
        ("s", s), then every lens bin ("g", g).

        Returns:
          (f, idx): the list of labels and a dict {label: row index},
          built by a dict comprehension over enumerate(f), which yields
          the (index, label) pairs.
        """
        f = [("c", i, A) for i in range(self.nzc) for A in range(self.nA)]
        f += [("s", s) for s in range(self.ns)]
        f += [("g", g) for g in range(self.nl)]
        return f, {x: n for n, x in enumerate(f)}

    def field_spectra(self):
        """Signal S[X, Y, node] of every field pair and the auto noise N[X]
        (E-mode noise for shear; the B-mode noise equals it).

        Returns:
          (S, noise, is_shear, idx): S of shape (nf, nf, nb) (nf = fields,
          nb = spectra nodes); noise (nf,) per sr: Omega_s/N_iA for the
          clusters, sigma_e^2/n_s for the shear, 1/n_g for the lenses;
          is_shear (nf,) True on the source fields; idx the label-to-row
          dict of fields().
        """
        sp = self.sp
        fields, idx = self.fields()
        nb = sp["ells"].size
        nc, ns = self.nzc * self.nA, self.ns
        g0 = nc + ns
        S = np.zeros((len(fields), len(fields), nb))
        # (i, A, j, B, l) -> (i nA + A, j nA + B, l): reshape merges (i, A)
        # into one row index with A fastest; transpose(1, 0, 2) below
        # fills each mirror block by swapping the two field axes
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
        field X carries in a signal piece and in a noise piece.

        Clusters: B_i(theta) in a signal piece, and in a noise piece 1
        ("signal" mode) or B_i(theta) ("jacobian"). Shear: 1 + m_j in
        both. Lenses: 1. selection_mode "none": every factor is 1.

        Returns:
          (sig, noi): two numpy arrays of shape (nf, nt).
        """
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
        G[o1, o2, node] x row(o1, theta) x col(o2, theta').

        A field is X = sig s + noi n (its signal s times the leg factor
        sig, its noise n times the leg factor noi; leg_factors), so each
        Wick product <X_a X_c><X_b X_d> of the row observable (legs a, b)
        and the column observable (legs c, d) splits into signal x signal,
        signal x noise and noise x noise pieces, each carrying the leg
        factors of its own legs (the list below; module docstring).

        Returns:
          list of six (G, row, col) triples: G (no, no, nb) the node array
          of the piece, row (no, nt) the factor of the row observable's
          legs, col (no, nt) that of the column observable's. The entry of
          index 1 (the second) is the pure noise x noise piece (B-mode
          shape noise of the xi+/- pairs included), which twopoint can sum
          over all multipoles.
        """
        obs = self.layout["obs"]
        S, noise, is_shear, idx = self.field_spectra()
        sig, noi = self.leg_factors()
        # field row of the first and of the second leg of every observable
        a = np.array([idx[o[1][0]] for o in obs])
        b = np.array([idx[o[1][1]] for o in obs])
        # the legs of the row observable as columns (no, 1) and those of
        # the column observable as rows (1, no): S[A_, C_] is the
        # (no, no, nb) array S[a(o1), a(o2), node]
        A_, B_ = a[:, None], b[:, None]          # rows
        C_, D_ = a[None, :], b[None, :]          # columns
        # noise as a diagonal matrix: Nd[X, Y] = noise[X] if X = Y, else 0
        Nd = np.diag(noise)
        nb = S.shape[2]
        SS = S[A_, C_] * S[B_, D_] + S[A_, D_] * S[B_, C_]
        # noise x noise is constant in l: [:, :, None] * np.ones(nb)
        # copies it to every node
        NN = (Nd[A_, C_] * Nd[B_, D_] + Nd[A_, D_] * Nd[B_, C_])[:, :, None] * np.ones(nb)
        # B-mode shape noise of xi+/- pairs (xi+ = EE + BB, xi- = EE - BB):
        # sign +1 for xi+, -1 for xi-, 0 for the other kinds; the product
        # of the two signs adds the BB piece to ++ and --, subtracts it
        # from +-
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
            (NN, noi[a] * noi[b], noi[a] * noi[b]),     # second entry: noise x noise
            # Wick 1 (a-c, b-d): signal ac x noise bd, noise ac x signal bd
            ((S[A_, C_] * Nd[B_, D_][:, :, None]), sig[a] * noi[b], sig[a] * noi[b]),
            ((Nd[A_, C_][:, :, None] * S[B_, D_]), noi[a] * sig[b], noi[a] * sig[b]),
            # Wick 2 (a-d, b-c): signal ad x noise bc, noise ad x signal bc
            ((S[A_, D_] * Nd[B_, C_][:, :, None]), sig[a] * noi[b], noi[a] * sig[b]),
            ((Nd[A_, D_][:, :, None] * S[B_, C_]), noi[a] * sig[b], sig[a] * noi[b]),
        ]
        return terms

    def _project(self, G, M):
        """(nobs, nt, nobs, nt) = sum_node M[(k1,k2)][i,j,node] G[o1,o2,node].

        Arguments:
          G = node array of one piece, (no, no, nb).
          M = band matrices, dict keyed by the kernel pair (band_matrices).

        Returns:
          numpy array of shape (no, nt, no, nt), before the 1/f_sky.
        """
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
                # the sub-block of the observables ia x ib as a matrix
                # (pairs, nb) times M as (nt nt, nb) transposed: the node
                # sum for every pair at once, reshaped to
                # (ia, ib, nt, nt) and reordered to (ia, nt, ib, nt);
                # np.ix_ selects the rows ia and the columns ib
                g = G[np.ix_(ia, ib)].reshape(ia.size * ib.size, nb)
                blk = (g @ M[(ka, kb)].reshape(nt * nt, nb).T).reshape(ia.size, ib.size, nt, nt)
                out[np.ix_(ia, np.arange(nt), ib, np.arange(nt))] = blk.transpose(0, 2, 1, 3)
        return out

    def _noise_exact(self, G):
        """The noise x noise piece summed over ALL multipoles.

        Its angular power is flat, so the l sum is a completeness relation:
        sum_{l} (2l+1)/2 <K_l>_i <K_l>_j -> delta_ij / Delta x_i for the
        Legendre (w), associated Legendre (gamma_t) and Wigner d (xi+/-)
        bin averages, i.e.
          sum_l K_i(l) K_j(l)/(2l+1) = delta_ij / (8 pi^2 Delta x_i),
          Delta x_i = cos(theta_i) - cos(theta_{i+1}),
        and zero between different kinds (xi+ x xi-: E and B noise cancel).
        This is the real-space pair-count result (e.g. gamma_t:
        sigma_e^2 / (n_l n_s Omega_s 2 pi theta dtheta)); the l sum cut at
        LMAX instead is 4.8% low in the first theta bin and puts a 1.6%
        correlation between neighboring bins (noise = "lsum", the
        ref_covariance.py convention).

        The identity is exact for w. The gamma_t and xi+/- kernels carry
        the extra factors (l-1)(l+2)/(l(l+1)) and its square, which differ
        from 1 by terms of order 1/l^2, so for them
        delta_ij/(8 pi^2 Delta x_i) is the pair-count variance rather than
        the exact infinite sum over those kernels;
        test_pure_noise_is_pair_count_variance checks every block against
        the pair-count variances.

        Arguments:
          G = node array of the noise x noise piece, (no, no, nb); only
              G[:, :, 0] is read (the piece is constant in l).

        Returns:
          numpy array of shape (no, nt, no, nt), diagonal in theta, before
          the 1/f_sky.
        """
        obs = self.layout["obs"]
        kinds = np.array([o[2] for o in obs])
        xe = np.cos(self.edges)
        diag = 1.0 / (8.0 * np.pi**2 * (xe[:-1] - xe[1:]))        # (nt,)
        # 1 for two observables of the same kind, else 0 (a boolean outer
        # comparison converted to floats)
        same = (kinds[:, None] == kinds[None, :]).astype(float)
        g = G[:, :, 0] * same                                      # l independent
        no, nt = len(obs), self.nt
        out = np.zeros((no, nt, no, nt))
        # only the theta-diagonal entries t = t' are nonzero
        for t in range(nt):
            out[:, t, :, t] = g * diag[t]
        return out

    def _ytransform(self, C):
        """T on the theta index of every cs observable (rows and columns).

        Arguments:
          C = covariance piece of shape (no, nt, no, nt); modified in place.

        Returns:
          the same array C, after C[o, t, :, :] -> sum_u T[t, u] C[o, u, :, :]
          for every cs row observable o and the same on the column theta
          index for every cs column observable.
        """
        # boolean mask over the observables, True for the cs ones. einsum
        # "tu,oujv->otjv" contracts T with the row theta index u of the cs
        # rows; "vu,itou->itov" contracts it with the column theta index u
        # of the cs columns.
        cs = np.array([o[0] == "cs" for o in self.layout["obs"]])
        C[cs] = np.einsum("tu,oujv->otjv", self.T, C[cs])
        C[:, :, cs] = np.einsum("vu,itou->itov", self.T, C[:, :, cs])
        return C

    def twopoint(self, M=None, return_terms=False, symmetrize=True):
        """Two-point covariance in observable order, shape (nobs nt, nobs nt).

        symmetrize = False returns the assembly as it is (the tests check
        that it is symmetric: a factor on the wrong leg would break it).

        Each piece of _terms is projected (_project, or _noise_exact for
        the noise x noise piece when noise = "exact"), Y-transformed on the
        cs rows and columns, multiplied by its row and column leg factors
        (so the Y transform acts before the selection factor, as in the
        model) and summed; the sum is divided by f_sky.

        Arguments:
          M = band matrices (band_matrices(ells, edges, lmax); None:
              computed here, the expensive step).
          return_terms = True also returns the list of the pieces, each
                         (no, nt, no, nt) before the 1/f_sky.
          symmetrize = True (default) returns (C + C^T)/2, which removes
                       the rounding asymmetry.

        Returns:
          cov of shape (no nt, no nt) in observable-then-theta order, or
          (cov, pieces) when return_terms is True.
        """
        if M is None:
            M = band_matrices(self.sp["ells"], self.edges, self.lmax)
        no, nt = len(self.layout["obs"]), self.nt
        cov = np.zeros((no, nt, no, nt))
        pieces = []
        for n, (G, row, col) in enumerate(self._terms()):
            # a piece that is zero everywhere is skipped
            if not np.any(G):
                continue
            if n == 1 and self.noise == "exact":
                C = self._ytransform(self._noise_exact(G))
            else:
                C = self._ytransform(self._project(G, M))
            # the outer product of the row factor (no, nt) and the column
            # factor (no, nt), shape (no, nt, no, nt)
            C *= row[:, :, None, None] * col[None, None, :, :]
            cov += C
            if return_terms:
                pieces.append(C)
        cov = cov.reshape(no * nt, no * nt) / self.fsky
        if symmetrize:
            cov = 0.5 * (cov + cov.T)
        return (cov, pieces) if return_terms else cov

    def full(self, M=None):
        """The joint covariance (ndata, ndata) in the likelihood layout.

        The 2pt part (twopoint) goes to the joint positions of
        layout["index"], the counts block (counts_cov) to
        layout["N_index"]; the counts x 2pt blocks stay zero.

        Arguments:
          M = band matrices (None: computed by twopoint).

        Returns:
          numpy array of shape (ndata, ndata) (2812 x 2812 for the
          project's binning).
        """
        lay = self.layout
        c2 = self.twopoint(M)
        idx2 = lay["index"].ravel()
        cov = np.zeros((lay["ndata"], lay["ndata"]))
        # np.ix_(idx2, idx2) selects the rows and the columns idx2: the 2pt
        # entries scattered into their joint positions
        cov[np.ix_(idx2, idx2)] = c2
        iN = lay["N_index"].ravel()
        cov[np.ix_(iN, iN)] = self.counts_cov
        return cov


# ----------------------------------------------------------------------
# convenience: from a ClusterReference
# ----------------------------------------------------------------------
def full_covariance(ref, selection_mode="signal", counts=None, cg_lens_bins=(0, 1, 2),
                    return_object=False, M=None, noise="exact"):
    """Joint covariance at the point of a reference_cluster.ClusterReference
    whose settings["lens_bins"] cover every lens bin (0 .. nl-1).

    counts = None: the reference counts N_iA (Poisson, shot noise and the
    sample-variance weights of ref_covariance.counts_covariance); an array
    overrides N_iA in all three (the sample-variance S_i,AB is a property
    of the shell and does not depend on the normalization).
    M: band_matrices of the reference's ells and binning (None: computed).

    Arguments:
      ref = reference_cluster.ClusterReference; settings["lens_bins"]
            must list every lens bin, 0 .. nl - 1.
      selection_mode = "signal" (default), "jacobian" or "none".
      counts = None or an array (nzc, nA) of N_iA (above).
      cg_lens_bins = lens bin paired with each cluster z bin in w_cg
                     (default (0, 1, 2)).
      return_object = True also returns the FullGaussianCovariance.
      M = band matrices (above).
      noise = "exact" (default) or "lsum" (FullGaussianCovariance).

    Returns:
      (cov, info), or (cov, info, fc) when return_object is True: cov of
      shape (ndata, ndata); info = the dict of counts_covariance plus
      counts (the N_iA used), counts_cov (the counts block) and layout
      (joint_layout).

    Raises:
      ValueError when settings["lens_bins"] is not every lens bin.
    """
    s, p = ref.settings, ref.params
    if list(s["lens_bins"]) != list(range(len(s["n_lens_arcmin2"]))):
        raise ValueError("full_covariance needs settings lens_bins = every lens bin")
    # the counts block and S_i,AB at the reference counts
    cN_ref, info = counts_covariance(ref)
    if counts is None:
        N = ref.counts()
        cN = cN_ref
    else:
        N = np.asarray(counts, dtype=float)
        # the counts block rebuilt from the given N and the reference
        # S_i,AB, one z bin at a time
        cN = np.zeros_like(cN_ref)
        for i in range(N.shape[0]):
            sl = slice(i * N.shape[1], (i + 1) * N.shape[1])
            cN[sl, sl] = np.diag(N[i]) + np.outer(N[i], N[i]) * info["S"][i]
    fc = FullGaussianCovariance(
        ref.spectra(), N, ref.cluster.Omega_s, ref.edges, ref.T, s["lmax"],
        selection=ref.selection(), shear_m=p["shear_m"],
        n_lens_arcmin2=s["n_lens_arcmin2"], n_src_arcmin2=s["n_src_arcmin2"],
        sigma_e=s["sigma_e"], cg_lens_bins=cg_lens_bins,
        selection_mode=selection_mode, counts_cov=cN, noise=noise)
    cov = fc.full(M)
    # dict(info, key=value, ...) copies info and adds the keys
    info = dict(info, counts=N, counts_cov=cN, layout=fc.layout)
    return (cov, info, fc) if return_object else (cov, info)
