"""Gaussian covariance of the DES cluster data vector (reference).

The covariance C of a data vector holds the covariances between its
entries; chi2 = (d - m)^T C^-1 (d - m) weighs the data-model difference
d - m with its inverse. This module computes the Gaussian covariance of
the cluster blocks (counts N, cluster lensing cs, w_cc, w_cg), as
reference_cluster.ClusterReference.covariance assembles it; the joint
covariance of all seven blocks is ref_covariance_full, which reuses
cc_pairs, counts_covariance and the arithmetic of band_matrices. A
field is one projected tracer: the clusters of one (z bin i, richness
bin A), the shear of one source bin j, or the galaxies of one lens bin;
an observable is a pair of fields (its two legs) and a projection kernel.

Two-point blocks (full sky, per multipole, Knox with f_sky):

  Cov(C^{AB}_l, C^{CD}_l')
      = delta_ll' [C~^{AC} C~^{BD} + C~^{AD} C~^{BC}] / ((2l+1) f_sky)
  C~^{XY} = C^{XY} + delta_XY N^X,
    N^{c_iA} = 1/n_cA, n_cA = N_iA/Omega_s          (clusters, per sr)
    N^{g_i}  = 1/n_g,i                              (lenses)
    N^{s_j}  = sigma_e^2/n_eff,j, sigma_e per component (E-mode shear power)
  projected with the same bin-averaged kernels as the signal
  (ref_projection: w -> Pw, gamma_t -> Pg), l = 1 .. LMAX-1:
  Cov(xi_i, xi'_j) = sum_l K_i(l) K'_j(l) [..] / ((2l+1) f_sky).
  The [..] is interpolated in ln l between the spectra nodes with the
  same cubic spline the signal uses (exact band weights, no binning).
  cs blocks are then Y-transformed: Cov_Sigma = R Cov R^T,
  R = blockdiag(T on each cs theta vector, 1 elsewhere). No selection
  factor enters the covariance (it multiplies the model only).
  (Knox formula: the variance of the power-spectrum estimate of Gaussian
  fields observed on a fraction f_sky = Omega_s/(4 pi) of the sky. N^X
  is the shot or shape noise power of field X, which only an auto pair
  X = Y carries.)

Counts:
  Cov(N_iA, N_jB) = delta_ij [delta_AB N_iA + N_iA N_iB S_i,AB]
  S_i,AB = sum_l (2l+1)/(4 pi) W_l^2 C_l^{iA,iB}
  Window approximation: the footprint is a spherical cap of the survey
  area (1 - cos theta_c = Omega_s/(2 pi)), W_l = (2 pi/Omega_s)
  [P_{l-1}(cos theta_c) - P_{l+1}(cos theta_c)]/(2l+1), W_0 = 1; the
  shell is the count-weighted radial distribution of bin (i, A) times
  b_A(z) D(z); linear P(k, 0); exact spherical-Bessel C_l for l <= 20,
  Limber above, l <= 300. The blocks between different cluster z bins
  (i != j) are set to zero. This is an approximation: neighbouring
  shells share large-scale density modes and their true-redshift
  kernels overlap through the photo-z scatter, so their sample variances
  are correlated.
  S_i,AA / b_iA^2 is the "sigma_i^2" of the footprint x shell.
  N x 2pt = 0.
  (The first term is the Poisson variance of a count, equal to its mean;
  each cluster lands in exactly one observed bin, so it is diagonal. The
  second is the sample variance: all richness bins of one z bin trace
  the same large-scale density over the footprint.)

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

# one square arcminute in steradians: a density per arcmin^2 divided by
# it is a density per sr
ARCMIN2_TO_SR = (np.pi / 180.0 / 60.0) ** 2


def cc_pairs(nA):
    """Return the richness pairs (A, B) with A <= B in data-vector order.

    A list comprehension with two for clauses: A is the outer loop and B
    the inner one, from A to nA - 1, so nA = 4 gives the 10 pairs
    (0, 0), (0, 1), ..., (3, 3) of the w_cc rows.

    Arguments:
      nA = number of richness bins.

    Returns:
      list of nA (nA + 1)/2 tuples (A, B).
    """
    return [(a, b) for a in range(nA) for b in range(a, nA)]


def observables(nzc, nA, ns):
    """List of (block, legs, kernel, label) in data-vector order (2pt part only).

    legs are field labels ('c', i, A), ('s', j), ('g', i). kernel is "gt"
    (cluster lensing, projected with Pg) or "w" (w_cc and w_cg, with Pw);
    label holds the bin indices: (i, s, A) for cs, (i, A, B) for cc,
    (i, A) for cg. The lens leg of w_cg is ('g', i): position i of the
    covariance's lens fields, the lens bin paired with cluster bin i.

    Arguments:
      nzc = number of cluster z bins.
      nA = number of richness bins.
      ns = number of source bins.

    Returns:
      list of tuples: the cs rows, then cc, then cg (the module
      docstring's layout).
    """
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
    S_node the cubic-spline basis in ln l on the spectra nodes.

    The cubic spline through node values y is linear in y, so its value
    at l is sum_node y_node S_node(l), with S_node the spline through the
    unit vector of that node. Contracting M with a node array G (a sum
    over the node axis) then gives the multipole sum
    sum_l K^a_i K^b_j G(l)/(2l+1) of the splined G without a loop over
    l; test_band_matrices_equal_direct_sum checks this to 1e-10.

    Arguments:
      ells = spectra node multipoles, increasing, (nb,).
      edges = angular bin edges in radians.
      lmax = LMAX of the sums (l = 1 .. lmax - 1).

    Returns:
      dict keyed by the kernel pair (a, b), a and b in ("w", "gt"); each
      value an array (Ntheta, Ntheta, nb).
    """
    Pw, Pg = projection_kernels(edges, lmax)
    # the kernels without the column l = 0: column index = l - 1
    K = {"w": Pw[:, 1:], "gt": Pg[:, 1:]}
    l = np.arange(1, lmax, dtype=float)
    # basis[l - 1, node] = S_node(l): the splines through the columns of
    # the identity matrix (one unit vector per node) at every integer l
    basis = CubicSpline(np.log(ells), np.eye(ells.size), axis=0)(np.log(l))   # (L, nb)
    inv = 1.0 / (2.0 * l + 1.0)
    out = {}
    for a in ("w", "gt"):
        for b in ("w", "gt"):
            # M[(a, b)][i, j] = M[(b, a)][j, i]: reuse the transposed pair
            if (b, a) in out:
                out[(a, b)] = out[(b, a)].transpose(1, 0, 2)
                continue
            nt = K[a].shape[0]
            M = np.empty((nt, nt, ells.size))
            # row i of kernel a times every row of kernel b, weighted by
            # 1/(2l+1), then summed over l against the basis:
            # (Ntheta, L) @ (L, nb)
            for i in range(nt):
                M[i] = (K[a][i][None, :] * inv[None, :] * K[b]) @ basis
            out[(a, b)] = M
    return out


class GaussianCovariance:
    """Assemble the Gaussian covariance from the reference spectra.

    twopoint() gives the 2pt part (cs as Sigma, cc, cg) in the order of
    observables(); cap_window() gives the window of the counts footprint
    (counts_covariance).
    """

    def __init__(self, spectra, counts, Omega_s, edges, T, lmax=75000,
                 n_lens_arcmin2=(0.1380, 0.1016, 0.1071), n_src_arcmin2=(2.1402, 2.14455, 2.1518, 2.11845),
                 sigma_e=0.384666 / np.sqrt(2.0), lens_bins=(0, 1, 2)):
        """Store the spectra, the counts and the noise densities.

        Arguments:
          spectra = dict of ref_limber.LimberModel.compute with the
                    covariance spectra: "ells", "C_cc" (i,A,j,B,l),
                    "C_cs" (i,A,s,l), "C_cg" (i,A,g,l), "C_ss" (s,s',l),
                    "C_gs" (g,s,l), "C_gg" (g,g',l); g runs over the lens
                    fields.
          counts = N_iA, (nzc, nA): the cluster shot noise is
                   1/n_cA = Omega_s/N_iA.
          Omega_s = survey solid angle in sr.
          edges = angular bin edges in radians.
          T = Y-transform matrix (Ntheta, Ntheta) (ref_projection).
          lmax = LMAX of the multipole sums (default 75000).
          n_lens_arcmin2 = lens densities per arcmin^2 (default: MagLim
                           bins 1-3 of lens_n_gal in the lighthouse DES Y6
                           code-comparison file dataY6.yaml).
          n_src_arcmin2 = source densities per arcmin^2 (default: the
                          four bins of source_n_gal in dataY6.yaml).
          sigma_e = shape-noise dispersion per component (default
                    0.384666/sqrt(2): the sigma_e of dataY6.yaml counts
                    both components, as the lighthouse noise
                    sigma_e^2/(2 n) shows).
          lens_bins = for lens field g, the index into n_lens_arcmin2 of
                      its density (default (0, 1, 2)).
        """
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
        """Return the field labels and the label-to-row lookup.

        Order: the clusters ("c", i, A) with i outer, then the sources
        ("s", s), then the lens fields ("g", g).

        Returns:
          (f, idx): the list of labels and a dict {label: row index},
          built by a dict comprehension over enumerate(f), which yields
          the (index, label) pairs.
        """
        f = [("c", i, A) for i in range(self.nzc) for A in range(self.nA)]
        f += [("s", s) for s in range(self.ns)]
        f += [("g", g) for g in range(len(self.lens_bins))]
        return f, {x: n for n, x in enumerate(f)}

    def field_spectra(self, with_noise=True):
        """C~[X, Y, node] for every pair of fields.

        The signal spectra of every field pair, plus the noise power on
        the diagonal: 1/n_cA for the clusters (n_cA = N_iA/Omega_s per
        sr), sigma_e^2/n_s for the shear, 1/n_g for the lenses.

        Arguments:
          with_noise = True (default) adds the noise; False returns the
                       signal alone.

        Returns:
          (C, idx): C of shape (nf, nf, nb), nf = nzc nA + ns + the lens
          fields and nb = the number of spectra nodes; idx = the
          label-to-row dict of _fields.
        """
        sp = self.sp
        fields, idx = self._fields()
        nb = sp["ells"].size
        C = np.zeros((len(fields), len(fields), nb))
        nc = self.nzc * self.nA
        # (i, A, j, B, l) -> (i nA + A, j nA + B, l): reshape merges (i, A)
        # into one row index with A running fastest (C order)
        C[:nc, :nc] = sp["C_cc"].reshape(nc, nc, nb)
        cs = sp["C_cs"].reshape(nc, self.ns, nb)
        C[:nc, nc:nc + self.ns] = cs
        # the mirror block: transpose(1, 0, 2) swaps the two field axes
        C[nc:nc + self.ns, :nc] = cs.transpose(1, 0, 2)
        # -1 lets reshape infer the number of lens fields
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
        """Return the 2pt covariance (cs, cc, cg) in the order of observables().

        For every pair of observables o1 = (a1, b1) and o2 = (a2, b2) the
        node array G = C~[a1,a2] C~[b1,b2] + C~[a1,b2] C~[b1,a2] is
        contracted with the band matrices of their two kernels and divided
        by f_sky (module docstring).

        Shape flow (legend: no = observables, nt = theta bins, nb =
        spectra nodes):

          G (no, no, nb) and M[(ka, kb)] (nt, nt, nb)
            -> cov[p, i, q, j] = sum_n M[i, j, n] G[p, q, n]   (no, nt, no, nt)
            -> reshaped to (no nt, no nt), divided by f_sky, R cov R^T

        Arguments:
          y_transform = True (default): T on every cs theta vector (rows
                        and columns), the Sigma space of the data vector;
                        False: the gamma_t space.

        Returns:
          (cov, obs): cov of shape (no nt, no nt), rows ordered by
          observable, then theta bin; obs = observables(nzc, nA, ns).
        """
        obs = observables(self.nzc, self.nA, self.ns)
        C, idx = self.field_spectra()
        M = band_matrices(self.sp["ells"], self.edges, self.lmax)
        nt = self.edges.size - 1
        # row index (in C~) of the first and of the second leg of every
        # observable
        a = np.array([idx[o[1][0]] for o in obs])
        b = np.array([idx[o[1][1]] for o in obs])
        # G[o1, o2, node] = C~[a1,a2] C~[b1,b2] + C~[a1,b2] C~[b1,a2]
        # (C~ indexed by the column a[:, None] and the row a[None, :] is an
        # (no, no, nb) array)
        G = (C[a[:, None], a[None, :]] * C[b[:, None], b[None, :]]
             + C[a[:, None], b[None, :]] * C[b[:, None], a[None, :]])
        kinds = np.array([o[2] for o in obs])
        no = len(obs)
        cov = np.zeros((no, nt, no, nt))
        # One sub-block per kernel pair. np.ix_ builds an open mesh, so
        # cov[np.ix_(ia, all theta, ib, all theta)] is the block of the
        # rows ia and the columns ib; np.einsum sums over the index that
        # its output string omits: "ijn,pqn->piqj" is
        # cov[p, i, q, j] = sum_n M[i, j, n] G[p, q, n].
        for ka in ("w", "gt"):
            ia = np.where(kinds == ka)[0]
            for kb in ("w", "gt"):
                ib = np.where(kinds == kb)[0]
                cov[np.ix_(ia, np.arange(nt), ib, np.arange(nt))] = np.einsum(
                    "ijn,pqn->piqj", M[(ka, kb)], G[np.ix_(ia, ib)], optimize=True)
        cov = cov.reshape(no * nt, no * nt) / self.fsky
        if y_transform:
            # R = the identity with T on the diagonal block of every cs
            # observable
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
        """Return the window multipoles W_l of a spherical-cap footprint.

        A staticmethod is a function stored in the class that takes no
        self; it is called as GaussianCovariance.cap_window(...). The cap
        of solid angle Omega_s has 1 - cos theta_c = Omega_s/(2 pi), and
        W_l = (2 pi/Omega_s) [P_{l-1}(x_c) - P_{l+1}(x_c)]/(2l+1) with
        x_c = cos theta_c, W_0 = 1 (module docstring).

        Arguments:
          Omega_s = survey solid angle in sr.
          lmax = highest multipole.

        Returns:
          numpy array (lmax + 1,) of W_l, l = 0 .. lmax.
        """
        c = 1.0 - Omega_s / (2.0 * np.pi)
        P = legendre_table(np.array([c]), lmax + 1)[0]
        l = np.arange(lmax + 1)
        W = np.empty(lmax + 1)
        W[0] = 1.0
        W[1:] = 2.0 * np.pi / Omega_s * (P[l[1:] - 1] - P[l[1:] + 1]) / (2 * l[1:] + 1)
        return W


def counts_covariance(ref, l_exact=20, l_max=300, dchi=0.5):
    """delta_ij [delta_AB N_iA + N_iA N_iB S_i,AB] and diagnostics.

    `ref` is a reference_cluster.ClusterReference.

    For each cluster z bin i the radial kernels
    h_A(chi) = (dN_iA/dchi)/N_iA b_A(z) D(z), with
    dN_iA/dchi = Omega_s chi^2 <phi_i|z> n_A(z), give the shell spectra
    C_l^{iA,iB}: exact (ref_nonlimber.cl_exact_linear on a k grid cut at
    0.25 h/Mpc) for l <= l_exact, Limber above; the cap window turns
    them into S_i,AB (module docstring).

    Arguments:
      ref = reference_cluster.ClusterReference (its cluster model,
            cosmology and counts are read).
      l_exact = last multipole of the exact spherical-Bessel spectra
                (default 20).
      l_max = last multipole of the window sum (default 300).
      dchi = step of the uniform chi grid in Mpc/h (default 0.5).

    Returns:
      (cov, info): cov of shape (nzc nA, nzc nA), rows (i, A) with A
      fastest; info = dict(S (nzc, nA, nA), b_eff (nzc, nA) the
      count-weighted bias, sigma2_eff (nzc, nA) = S_i,AA/b_iA^2,
      W_l (l_max + 1,)).
    """
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
        # uniform chi grid over the support of bin i (np.arange stops
        # before the upper end) and the redshifts on it
        chi = np.arange(cosmo.chi(za), cosmo.chi(zb), dchi)
        z = cosmo.z_of_chi(chi)
        n, b = cl.n_b(z)
        # dN_iA/dchi = Omega_s chi^2 <phi_i|z> n_A(z), the expected counts
        # per unit chi; divided by N_iA it integrates to 1, and b_A D turns
        # it into the linear density kernel h_A of the shell
        dNdchi = cl.Omega_s * chi**2 * cl.phi(z, i)[None, :] * n          # (A, chi)
        D = cosmo.growth(z)
        h = dNdchi / N[i][:, None] * b * D[None, :]
        Cex = cl_exact_linear(cosmo, ells_ex, chi, h, k=k_grid(kmax=0.25))
        Clim = cl_limber_linear(cosmo, ells_lim, chi, h)
        Cl = np.concatenate([Cex, Clim], axis=0)                          # (l, A, B)
        l = np.arange(0, l_max + 1)
        # einsum "l,lab->ab": S[a, b] = sum_l (2l+1)/(4 pi) W_l^2 Cl[l, a, b]
        S = np.einsum("l,lab->ab", (2 * l + 1) / (4.0 * np.pi) * W**2, Cl)
        S_all[i] = S
        sl = slice(i * cl.nA, (i + 1) * cl.nA)
        # block of z bin i: the Poisson diag(N_i) plus N_iA N_iB S_i,AB
        # (np.outer builds the matrix N_iA N_iB)
        cov[sl, sl] = np.diag(N[i]) + np.outer(N[i], N[i]) * S
    beff = cl.counts_weighted_bias()
    # S_i,AA/b_iA^2 for every (i, A): a nested list comprehension, i outer
    sigma2 = np.array([[S_all[i, A, A] / beff[i, A] ** 2 for A in range(cl.nA)]
                       for i in range(cl.nzc)])
    return cov, dict(S=S_all, b_eff=beff, sigma2_eff=sigma2, W_l=W)
