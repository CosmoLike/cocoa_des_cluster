"""Cluster ingredients of the DES cluster reference (2503.13631 model):
mass-observable relation, richness-bin mass integrals n_A(z), b_A(z),
P^1h_A(k, z), photo-z selection kernels, counts N_iA and the radial
kernels of the cluster two-point functions.

A cluster is a dark-matter halo of mass M (M_200m, ref_halo) that the
redMaPPer cluster finder detects with an observed richness lambda (the
number of red-sequence member galaxies), the mass proxy. Clusters are
counted in observed bins: cluster redshift bin i (edges in the
photometric cluster redshift z_lambda, default [0.2, 0.4, 0.55, 0.65])
times richness bin A (default lambda edges [20, 30, 45, 60, 500]). Each
cluster lands in exactly one observed bin, while the true masses and
redshifts behind different bins overlap through the richness scatter of
the mass-observable relation (MOR) and through the photo-z scatter.
Units: M in M_sun/h, k in h/Mpc, chi in Mpc/h, number densities in
(h/Mpc)^3, survey area in deg^2 (converted to sr).

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
              photoz="table": the piecewise-linear interpolant (numpy.interp)
              of a tabulated <phi_i|z> (data/des_y6_cluster.nz format),
              zero outside the zero nodes that bracket the nonzero values of
              each column: the kernel the C code reads (structs_cluster.h).
              Every table node is a quadrature break (the kinks of the
              interpolant), so the z integrals are exact for it.
  (16) N_iA = Omega_s int dz chi^2 (c/H) <phi_i|z> n_A(z),  Omega_s in sr

  (mu = <ln lambda|M,z>, s = sigma_lnlambda; <phi_i|z> is the probability
  that a cluster at true redshift z has its z_lambda inside bin i, whose
  edges are z_lo and z_hi.)

Radial kernels of the 2-pt functions (dN/dz normalized to unit integral):
  kernel_mode 0 (default, "volume", DES Y1 arXiv:2008.10757 eq. 15,
  lighthouse):
      q_i(z)  ~ chi^2/H <phi_i|z>
  kernel_mode 1 ("abundance", counts-consistent):
      q_iA(z) ~ chi^2/H <phi_i|z> n_A(z)

Quadratures are composite Gauss-Legendre ("gl_panels") with the
breakpoints of every non-smooth feature honoured: an n-node
Gauss-Legendre rule integrates polynomials of degree up to 2n - 1
exactly on each panel, and no panel straddles a break (a bin edge or a
kink of the kernel), where the integrand is not smooth.
"""

import numpy as np
from scipy.special import erf

from ref_cosmology import COVERH0
from ref_halo import u_nfw

# one square degree in steradians, (pi/180)^2
DEG2_TO_SR = (np.pi / 180.0) ** 2


def gl_panels(breaks, max_width, order=8):
    """Composite Gauss-Legendre nodes and weights on [breaks[0], breaks[-1]].

    Every interval between consecutive breaks is split into equal panels
    no wider than max_width; each panel carries `order` GL nodes.

    Arguments:
      breaks = break points, in any order and with repeats allowed
               (np.unique sorts them and drops the repeats); the
               integrand may be non-smooth at them.
      max_width = largest panel width, in the unit of breaks.
      order = Gauss-Legendre nodes per panel (default 8).

    Returns:
      (x, w): 1-D numpy arrays of the nodes and the weights, ordered from
      breaks[0] to breaks[-1]; sum(w f(x)) approximates the integral of f.
    """
    # nodes x0 in (-1, 1) and weights w0 of the order-point rule
    x0, w0 = np.polynomial.legendre.leggauss(order)
    breaks = np.unique(np.asarray(breaks, dtype=float))
    xs, ws = [], []
    # zip pairs every break with the next one: (a, b) runs over the
    # consecutive intervals
    for a, b in zip(breaks[:-1], breaks[1:]):
        # panels per interval: ceil(width/max_width); the 1e-9 keeps a
        # width that is a multiple of max_width (up to rounding) from
        # gaining one more panel
        n = max(1, int(np.ceil((b - a) / max_width - 1e-9)))
        e = np.linspace(a, b, n + 1)
        half = 0.5 * np.diff(e)
        mid = 0.5 * (e[1:] + e[:-1])
        # x0 mapped to every panel, x = mid + half x0 and w = half w0: the
        # (n panels, order) arrays built from the column mid[:, None] and
        # the row x0[None, :] are flattened panel by panel (ravel)
        xs.append((mid[:, None] + half[:, None] * x0[None, :]).ravel())
        ws.append((half[:, None] * w0[None, :]).ravel())
    return np.concatenate(xs), np.concatenate(ws)


def rev_cumtrapz(y, x):
    """R(x_i) = int_{x_i}^{x_end} y dx (trapezoid), same length as x.

    Arguments:
      y = integrand samples, 1-D array.
      x = increasing sample points, 1-D array of the length of y.

    Returns:
      numpy array R of the length of x, with R[-1] = 0.
    """
    # trapezoid area of every segment [x_j, x_j+1]
    seg = 0.5 * (y[1:] + y[:-1]) * np.diff(x)
    out = np.zeros_like(y)
    # seg[::-1] reverses the segments; the cumulative sum of the reversed
    # array, reversed back, is out[i] = sum over j >= i of seg[j]
    out[:-1] = np.cumsum(seg[::-1])[::-1]
    return out


def lensing_efficiency(z_eval, chi_eval, zf, nf, chif):
    """g(z) = int_z^inf dz' n(z') (1 - chi(z)/chi(z')) from fine samples.

    zf ascending, nf = n(zf) (dN/dz), chif = chi(zf). Linear interpolation
    of the two reverse cumulative integrals (they are smooth).

    g = Q0(z) - chi(z) Q1(z), with Q0(z) = int_z n dz' and
    Q1(z) = int_z (n/chi) dz' as trapezoid sums on the fine grid. Small
    negative values from rounding are set to 0, and g = 0 at and beyond
    the last fine node.

    Arguments:
      z_eval = redshifts where g is wanted, 1-D array.
      chi_eval = chi(z_eval) in Mpc/h.
      zf = fine redshift grid, ascending, covering the support of n.
      nf = n(zf) per unit redshift (the callers pass normalized
           distributions, for which g tends to 1 as z tends to 0).
      chif = chi(zf) in Mpc/h (0 at z = 0).

    Returns:
      numpy array g of the shape of z_eval, dimensionless.
    """
    # n/chi, with 0 where chi = 0 (z = 0): np.errstate silences numpy's
    # division warnings inside the `with` block, np.where puts the 0
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

    The object holds the MOR, the binning, the survey area and the
    photo-z model, and computes every one-point cluster quantity:
    P(lambda bin|M, z), n_A, b_A, P^1h_A, <phi_i|z>, the counts N_iA and
    the radial kernels. The halo ingredients come from a ref_halo
    HaloModel, the distances from a ref_cosmology Cosmology.
    """

    def __init__(self, cosmo, halo, mor, lambda_edges=(20, 30, 45, 60, 500),
                 zc_edges=(0.2, 0.4, 0.55, 0.65), area_deg2=4143.0,
                 photoz="gaussian", sigma_z0=0.006, M_piv=5e14, z_piv=1.45,
                 lnM_min=np.log(1e12), lnM_max=np.log(1e16),
                 mass_panel_width=0.25, mass_order=8, nsig_support=10.0,
                 phi_table=None):
        """Store the sample definition and build the mass quadrature.

        Arguments:
          cosmo = ref_cosmology.Cosmology (distances, H(z)).
          halo = ref_halo.HaloModel (mass function, bias, concentration).
          mor = dict with "lnlambda0" (ln lambda_0), "A" (slope in ln M),
                "B" (slope in ln(1+z)) and "sigma_int"; copied.
          lambda_edges = observed-richness bin edges (default the paper's
                         [20, 30, 45, 60, 500]).
          zc_edges = cluster redshift (z_lambda) bin edges (default
                     [0.2, 0.4, 0.55, 0.65]).
          area_deg2 = survey area in deg^2 (default 4143, the DES Y3
                      cluster footprint of arXiv:2503.13632).
          photoz = selection-kernel model: "gaussian" (default: the erf
                   kernel of a Gaussian photo-z of width sigma_z0 (1+z)),
                   "tophat" (1 inside the bin, 0 outside) or "table"
                   (phi_table). Any other string selects the Gaussian
                   kernel: phi, support and breaks test only for "table"
                   and "tophat".
          sigma_z0 = Gaussian photo-z width per unit (1+z) (default 0.006;
                     the DES Y3 redMaPPer catalog has median
                     sigma_z/(1+z) = 0.0060, 0.0063, 0.0066 in the three
                     bins, PORT_PLAN.md section 1).
          M_piv = MOR pivot mass in M_sun/h (default 5e14).
          z_piv = MOR pivot of (1 + z), default 1.45: the name says z_piv
                  but the value is 1 + z_piv (mor_mean_sigma divides 1 + z
                  by it).
          lnM_min, lnM_max = ln of the mass range of the integrals, M in
                             M_sun/h (default 1e12 to 1e16, the range of
                             the C code's cluster.m).
          mass_panel_width = width in ln M of one Gauss-Legendre panel
                             (default 0.25).
          mass_order = Gauss-Legendre nodes per mass panel (default 8;
                       test_mass_integral_convergence finds n_A and b_A
                       within 1e-8 of 0.1-wide panels of order 12, and
                       P^1h within 5e-5).
          nsig_support = reach of the Gaussian kernel's support beyond
                         each bin edge, in photo-z widths sigma_z0 (1+z)
                         (default 10; support()).
          phi_table = (z, columns) when photoz = "table": z of shape (nz,)
                      and columns of shape (nz, nzc) holding <phi_i|z>;
                      not read otherwise.

        Raises:
          ValueError when photoz = "table" comes without phi_table, or
          with columns whose shape is not (nz, nzc).
        """
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
        # Mass quadrature: composite Gauss-Legendre nodes in ln M (296
        # nodes for the defaults, 37 panels of 8) and their weights; every
        # mass integral of this class is a sum over m of wM[m] f(lnM[m]).
        self.lnM, self.wM = gl_panels([lnM_min, lnM_max], mass_panel_width, mass_order)
        self.nsig = nsig_support
        if photoz == "table":
            if phi_table is None:
                raise ValueError("photoz='table' needs phi_table = (z, columns)")
            # unpack the (z, columns) pair, each member converted to a
            # float array (a generator expression feeding a two-name
            # assignment)
            ztab, tab = (np.asarray(x, dtype=float) for x in phi_table)
            if tab.ndim != 2 or tab.shape != (ztab.size, self.nzc):
                raise ValueError(f"phi_table: expected ({ztab.size}, {self.nzc}), "
                                 f"got {tab.shape}")
            self.phi_z, self.phi_tab = ztab, tab
            # support of bin i: the zero nodes bracketing its nonzero values.
            # np.nonzero(...)[0] lists the rows where the column is
            # positive; the support runs from the row before the first of
            # them to the row after the last (clipped to the table), as
            # structs_cluster.h describes for the C table.
            self.phi_support = []
            for i in range(self.nzc):
                nz = np.nonzero(tab[:, i] > 0)[0]
                lo, hi = max(nz[0] - 1, 0), min(nz[-1] + 1, ztab.size - 1)
                self.phi_support.append((ztab[lo], ztab[hi]))

    # ------------------------------------------------------------------
    # mass-observable relation
    # ------------------------------------------------------------------
    def mor_mean_sigma(self, lnM, z):
        """Return the MOR mean mu = <ln lambda|M, z> and scatter sigma_lnlambda.

        Eqs. 18-19 (module docstring); the Poisson term (e^mu - 1)/e^(2 mu)
        enters only where mu > 0, as in lighthouse cluster_util.c and
        halo_cluster.c.

        Arguments:
          lnM = ln(M/(M_sun/h)), scalar or array.
          z = true redshift, broadcast against lnM.

        Returns:
          (mu, s): numpy arrays of the broadcast shape.
        """
        m = self.mor
        mu = (m["lnlambda0"] + m["A"] * (lnM - np.log(self.M_piv))
              + m["B"] * np.log((1.0 + z) / self.z_piv))
        # np.where keeps the Poisson term where mu > 0 and 0 elsewhere
        var = m["sigma_int"] ** 2 + np.where(mu > 0, (np.exp(mu) - 1.0) / np.exp(2.0 * mu), 0.0)
        return mu, np.sqrt(var)

    def p_lambda_bins(self, lnM, z):
        """P(lambda in A | M, z), shape (nA,) + broadcast(lnM, z).shape.

        The lognormal MOR integrated over each richness bin in closed
        form: the difference of the normal cumulative distribution
        0.5 (1 + erf((ln lambda_edge - mu)/(sqrt2 s))) at the two edges.

        Arguments:
          lnM = ln(M/(M_sun/h)), scalar or array.
          z = true redshift, broadcast against lnM.

        Returns:
          numpy array of probabilities, axis 0 = richness bin A.
        """
        mu, s = self.mor_mean_sigma(lnM, z)
        ll = np.log(self.lam_edges)
        # ll.reshape((-1,) + (1,) * np.ndim(mu)) gives the nA + 1 edges one
        # trailing length-1 axis per axis of mu, so they broadcast against
        # mu: cdf has shape (nA + 1,) + mu.shape
        cdf = 0.5 * (1.0 + erf((ll.reshape((-1,) + (1,) * np.ndim(mu)) - mu) / (np.sqrt(2.0) * s)))
        # upper-edge minus lower-edge value of every bin
        return cdf[1:] - cdf[:-1]

    # ------------------------------------------------------------------
    # mass integrals
    # ------------------------------------------------------------------
    def _mass_grid(self, z):
        """Return the mass-node quantities that every mass integral needs.

        Arguments:
          z = redshifts, 1-D array (a scalar becomes an array of length 1).

        Returns:
          (z, nu, dn, P): z as an array (nz,), the peak heights nu and
          dn/dlnM in (h/Mpc)^3 at the mass nodes, both (nM, nz), and
          P(A|M, z) of shape (nA, nM, nz).
        """
        z = np.atleast_1d(np.asarray(z, dtype=float))
        # the column of the nM mass nodes against the row z[None, :]:
        # broadcasting gives every quantity the shape (nM, nz)
        lnM = self.lnM[:, None]
        nu = self.halo.nu(lnM, z[None, :])
        dn = self.halo.dndlnM(lnM, z[None, :], nu=nu)              # (M, z)
        P = self.p_lambda_bins(lnM, z[None, :])                    # (A, M, z)
        return z, nu, dn, P

    def n_b(self, z):
        """n_A(z) [(h/Mpc)^3] and b_A(z), each shape (nA, nz).

        n_A(z) = sum over the mass nodes m of wM[m] dn/dlnM P(A|M_m, z);
        b_A(z) is the same sum weighted by the halo bias, divided by n_A
        (eq. 21, module docstring).

        Arguments:
          z = redshifts, 1-D array (a scalar becomes length 1).

        Returns:
          (n, b): two numpy arrays of shape (nA, nz).
        """
        z, nu, dn, P = self._mass_grid(z)
        wdn = self.wM[:, None] * dn
        # np.einsum("mz,amz->az", wdn, P) multiplies wdn[m, z] by
        # P[a, m, z] and sums over m, the index missing from the output
        # string "az": n[a, z] = sum_m wdn[m, z] P[a, m, z]
        n = np.einsum("mz,amz->az", wdn, P)
        # bias(None, None, nu=nu): lnM and z are not read when the peak
        # heights are given
        b = np.einsum("mz,amz->az", wdn * self.halo.bias(None, None, nu=nu), P) / n
        return n, b

    def mean_mass(self, z):
        """Return the mean halo mass <M>_A(z) of each richness bin, in M_sun/h.

        <M>_A = int dlnM dn/dlnM P(A|M, z) M / n_A(z), with the einsum
        sums of n_b.

        Arguments:
          z = redshifts, 1-D array.

        Returns:
          numpy array of shape (nA, nz).
        """
        z, nu, dn, P = self._mass_grid(z)
        wdn = self.wM[:, None] * dn
        n = np.einsum("mz,amz->az", wdn, P)
        return np.einsum("mz,amz->az", wdn * np.exp(self.lnM)[:, None], P) / n

    def p1h(self, k, z):
        """P^1h_A(k, z) at one redshift z for an array k (h/Mpc): shape (nA, nk).

        u(k|M) is the truncated NFW with the Bhattacharya concentration;
        no bias, no miscentering; normalized by n_A(z).

        Shape flow (legend: nA = richness bins, nM = mass nodes,
        nk = wavenumbers):

          P (nA, nM) and the weights wM dn/dlnM (nM,)
            -> wts = P wM (dn/dlnM) (M/rho_m)/n_A          (nA, nM)
            -> U = u(k|M) on the (k, mass) grid            (nk, nM)
            -> wts @ U.T, the sum over the mass nodes      (nA, nk)

        Arguments:
          k = comoving wavenumber in h/Mpc, scalar or array (the result
              has the shape (nA,) + k.shape).
          z = one redshift (a float).

        Returns:
          numpy array of P^1h_A in (Mpc/h)^3.
        """
        z = float(z)
        k = np.asarray(k, dtype=float)
        zz, nu, dn, P = self._mass_grid(np.array([z]))
        # keep the single redshift column: nu and dn (nM,), P (nA, nM)
        nu, dn, P = nu[:, 0], dn[:, 0], P[:, :, 0]
        c = self.halo.conc(self.lnM, z, nu=nu)
        rd = self.halo.r_delta(self.lnM)
        wdn = self.wM * dn
        # n_A: the sum over the mass nodes as a matrix-vector product
        n = P @ wdn
        wts = P * (wdn * np.exp(self.lnM) / self.halo.rho_m)[None, :] / n[:, None]   # (A, M)
        U = u_nfw(k.reshape(-1)[:, None], rd[None, :], c[None, :])              # (nk, M)
        return (wts @ U.T).reshape((self.nA,) + k.shape)

    # ------------------------------------------------------------------
    # selection in redshift
    # ------------------------------------------------------------------
    def phi(self, z, i):
        """<phi_i|z_true>: probability that z_lambda falls in bin i.

        Three models (photoz): "table", the linear interpolant of the
        tabulated column inside its support (0 outside); "tophat", 1 for
        lo <= z < hi; any other value, the Gaussian photo-z
        [erf((hi - z)/(sqrt2 s)) - erf((lo - z)/(sqrt2 s))]/2 with
        s = sigma_z0 (1 + z) (lo, hi = the z_lambda edges of bin i).

        Arguments:
          z = true redshift, scalar or array.
          i = cluster redshift bin, 0 <= i < nzc.

        Returns:
          numpy array of probabilities in [0, 1], the shape of z.
        """
        z = np.asarray(z, dtype=float)
        lo, hi = self.zc_edges[i], self.zc_edges[i + 1]
        if self.photoz == "table":
            a, b = self.phi_support[i]
            out = np.interp(z, self.phi_z, self.phi_tab[:, i])
            return np.where((z >= a) & (z <= b), out, 0.0)
        if self.photoz == "tophat":
            return ((z >= lo) & (z < hi)).astype(float)
        s = self.sigma_z0 * (1.0 + z)
        return 0.5 * (erf((hi - z) / (np.sqrt(2.0) * s)) - erf((lo - z) / (np.sqrt(2.0) * s)))

    def support(self, i):
        """Return the true-redshift interval (z_a, z_b) outside which
        <phi_i|z> is treated as zero.

        "table": the bracketing zero nodes of the column; "tophat": the
        bin edges; Gaussian: (max(1e-4, lo - s (1 + lo)),
        (hi + s (1 + hi))/(1 - s)) with s = nsig_support sigma_z0. Each
        Gaussian end lies at least nsig_support photo-z widths
        sigma_z0 (1 + z), taken at that end, beyond the nearest bin edge
        (the closed forms are a little wider than needed), unless the
        floor at z = 1e-4, which keeps chi > 0, applies.

        Arguments:
          i = cluster redshift bin, 0 <= i < nzc.

        Returns:
          tuple (z_a, z_b) of floats.
        """
        lo, hi = self.zc_edges[i], self.zc_edges[i + 1]
        if self.photoz == "table":
            return self.phi_support[i]
        if self.photoz == "tophat":
            return lo, hi
        s = self.nsig * self.sigma_z0
        return max(1e-4, lo - s * (1.0 + lo)), (hi + s * (1.0 + hi)) / (1.0 - s)

    def breaks(self, i):
        """Return the sorted quadrature break points of bin i in z.

        "table": the support ends and every table node inside the support;
        otherwise the support ends and the two bin edges (where the top
        hat jumps and the erf kernel changes fastest).

        Arguments:
          i = cluster redshift bin, 0 <= i < nzc.

        Returns:
          sorted list of floats.
        """
        a, b = self.support(i)
        if self.photoz == "table":
            # every node of the piecewise-linear interpolant is a kink
            zt = self.phi_z
            # {a, b} | set(...) is a set union, which drops repeated
            # values; sorted() returns a list
            return sorted({a, b} | set(zt[(zt >= a) & (zt <= b)].tolist()))
        return sorted({a, b, self.zc_edges[i], self.zc_edges[i + 1]})

    def dVdz(self, z):
        """chi^2 dchi/dz, comoving volume per unit z per sr [(Mpc/h)^3].

        Arguments:
          z = redshift, scalar or array.

        Returns:
          numpy array of the shape of z.
        """
        return self.cosmo.chi(z) ** 2 * self.cosmo.dchi_dz(z)

    # ------------------------------------------------------------------
    # counts
    # ------------------------------------------------------------------
    def counts(self, z_panel=0.005, order=8):
        """N_iA, shape (nzc, nA).

        Eq. 16: N_iA = Omega_s int dz chi^2 (dchi/dz) <phi_i|z> n_A(z), the
        expected number of clusters of each observed bin over the survey
        area (an absolute number, not a density), integrated with a
        composite Gauss-Legendre rule in z between the breaks of bin i.

        Arguments:
          z_panel = largest z panel width (default 0.005).
          order = Gauss-Legendre nodes per panel (default 8;
                  test_counts_and_kernel_quadrature finds the counts
                  within 1e-9 of 0.002-wide panels of order 12).

        Returns:
          numpy array of shape (nzc, nA), dimensionless counts.
        """
        N = np.zeros((self.nzc, self.nA))
        for i in range(self.nzc):
            z, w = gl_panels(self.breaks(i), z_panel, order)
            n, _ = self.n_b(z)
            # n @ (...): (nA, nz) @ (nz,) sums over the z nodes, one count
            # per richness bin
            N[i] = self.Omega_s * n @ (w * self.dVdz(z) * self.phi(z, i))
        return N

    def counts_weighted_bias(self, z_panel=0.005, order=8):
        """Count-weighted effective bias b_iA = int dV phi n_A b_A / int dV phi n_A.

        Arguments:
          z_panel = largest z panel width (default 0.005).
          order = Gauss-Legendre nodes per panel (default 8).

        Returns:
          numpy array of shape (nzc, nA), dimensionless.
        """
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
        """int dz chi^2 (c/H) <phi_i|z> [n_A(z)], shape (nzc, nA).

        The normalization of the radial kernels q: int dz dV/dz <phi_i|z>
        (dV/dz = chi^2 c/H per sr, in (Mpc/h)^3) for kernel_mode 0, the
        same integral weighted by n_A(z) (dimensionless) for any other
        kernel_mode. In mode 0 all nA columns of a row hold one number.

        Arguments:
          kernel_mode = 0 (volume, default) or 1 (abundance); any nonzero
                        value acts as 1.
          z_panel = largest z panel width (default 0.005).
          order = Gauss-Legendre nodes per panel (default 8).

        Returns:
          numpy array of shape (nzc, nA).
        """
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
        all richness bins share it. `nb` = (n, b) at z (optional cache).

        q_iA(z) = dV/dz <phi_i|z> [n_A(z)]/norm per unit redshift, so that
        int q dz = 1 over the bin's support (kernel_norms).

        Arguments:
          z = redshifts, 1-D array.
          i = cluster redshift bin, 0 <= i < nzc.
          kernel_mode = 0 (volume, default) or nonzero (abundance).
          norms = output of kernel_norms(kernel_mode) (None: computed
                  here, a z quadrature of every bin).
          nb = (n, b) of n_b(z) at the same z, optional; read only for a
               nonzero kernel_mode, where it saves the mass integrals.

        Returns:
          numpy array of shape (nA, nz), per unit redshift.
        """
        z = np.asarray(z, dtype=float)
        if norms is None:
            norms = self.kernel_norms(kernel_mode)
        base = self.dVdz(z) * self.phi(z, i)
        if kernel_mode == 0:
            # np.broadcast_to repeats the kernel along a new first axis of
            # length nA as a read-only view; .copy() makes it an ordinary
            # writable array
            return np.broadcast_to(base / norms[i, 0], (self.nA,) + z.shape).copy()
        n = self.n_b(z)[0] if nb is None else nb[0]
        # n (nA, nz) times the row base[None, :], divided by the column
        # norms[i][:, None] (one norm per richness bin)
        return n * base[None, :] / norms[i][:, None]

    def magnification_efficiency(self, z_eval, i, kernel_mode=0, norms=None, dz_fine=5e-5):
        """g_c,iA(z) = int dz' q_iA(z') (1 - chi(z)/chi(z')), shape (nA, nz).

        The integral runs over the whole cluster n(z) behind z: matter at
        any redshift in front of a cluster magnifies it, so g_c is nonzero
        over the full foreground of the bin and not only inside it (the
        lighthouse code keeps the magnification inside the bin,
        PORT_PLAN.md section 3). lensing_efficiency evaluates it with
        trapezoid sums on a uniform fine grid over the bin's support.

        Arguments:
          z_eval = redshifts where g_c is wanted, 1-D array.
          i = cluster redshift bin, 0 <= i < nzc.
          kernel_mode = 0 (volume, default) or nonzero (abundance).
          norms = kernel_norms(kernel_mode) (None: computed in q).
          dz_fine = step of the fine grid (default 5e-5).

        Returns:
          numpy array of shape (nA, nz), dimensionless; it tends to 1 as
          z tends to 0 (test_counts_and_kernel_quadrature checks it to
          1e-3 at z = 1e-4).
        """
        a, b = self.support(i)
        zf = np.arange(a, b + dz_fine, dz_fine)
        chif = self.cosmo.chi(zf)
        qf = self.q(zf, i, kernel_mode, norms)
        chi_e = self.cosmo.chi(z_eval)
        # one row per richness bin A (a list comprehension), stacked into
        # an (nA, nz) array
        return np.array([lensing_efficiency(z_eval, chi_e, zf, qf[A], chif)
                         for A in range(self.nA)])
