"""Limber angular power spectra of the DES cluster reference.

The Limber approximation replaces the exact line-of-sight integrals over
spherical Bessel functions by their value at k = (l + 1/2)/chi; it holds
when the radial kernels are broad compared with 1/k. Every spectrum below
is a sum over Gauss-Legendre z nodes (NodeSet) of leg x leg x P_NL x the
Limber measure dchi/chi^2. reference_cluster.py projects the spectra to
real space (ref_projection), and ref_covariance and ref_covariance_full
use them in the Gaussian covariance.

With k = (l + 1/2)/chi and every kernel in 1/(Mpc/h),

  C^{XY}_l = int dchi L_X(chi, l) L_Y(chi, l) P_NL(k, z)/chi^2   (+ 1-halo for cs)

Legs (radial kernels), following cosmolike's cosmo2D.c conventions:

  cluster  c_iA : L = b_A(z) W_c,iA(chi) + e1(l) C_c W_mag,c,iA(chi),  C_c = -2
                  W_c = q_iA(z) H(z)/c,  W_mag,c = (3/2) Om (H0/c)^2 chi (1+z) g_c
                  (g_c: the cluster lensing efficiency over the full foreground)
  galaxy   g_i  : L = b_i W_g,i + e1(l) bmag_i W_mag,g,i
                  W_g = n_i(z) H/c, bmag_i = C_l of 2503.13631 Eq. 27 (cosmo2D.c gbmag)
  source   s_j  : L = W_kappa,j - W_src,j C1(z)        (NLA, IA.c conventions)
                  W_kappa = (3/2) Om (H0/c)^2 chi (1+z) g_j,  W_src = n_j(z) H/c
                  C1(z) = A1 ((1+z)/1.62)^eta1 Om c1rhocrit/D(z), c1rhocrit = 0.01389

  cluster lensing (2503.13631 Eqs. 17, 20, 22):
     C^{cs}_l = e2(l) int dchi/chi^2 { L_s L_c P_NL + W_kappa W_c P^1h_A(k, z) }
     (the 1-halo term has no bias, no magnification and no IA)

  e1(l) = l(l+1)/(l+1/2)^2 on every magnification leg (cosmo2D.c
  ell_prefactor), e2(l) = sqrt((l-1)l(l+1)(l+2))/(l+1/2)^2 on every
  spin-2 (shear) leg (cosmo2D.c ell_prefactor2); both can be switched off.

  (Om = Omega_m, total matter; q_iA, g_c: ref_cluster; n(z):
  ref_cosmology.NzBins; D(z): ref_cosmology growth; A1, eta1 = NLA
  amplitude and redshift power; b_i, bmag_i = lens bias and lens
  magnification coefficient.)

Cluster spectra are integrated on z nodes up to the top of the cluster
photo-z support: the cluster leg vanishes above it (W_c is zero there,
and so is the lensing efficiency g_c of the clusters), and with it the
integrand of every cluster spectrum. gg, gs and ss (needed by the
covariance only) are integrated on nodes up to the top of the shifted
lens and source n(z) (LimberModel.__init__).
"""

import numpy as np

from ref_cosmology import COVERH0
from ref_cluster import gl_panels, lensing_efficiency

# C1RHOCRIT is C1 rho_crit of the NLA amplitude (2503.13631 eq. 25 writes
# it C1-bar rho_crit), cosmolike's nuisance.c1rhocrit_ia (structs.c).
# EDGE_EPS: _fine places a node pair this close to every n(z) support
# end; it must be far below the fine step dz_fine (2e-4) and far above
# the rounding of z (about 1e-16), so the two nodes stay distinct and
# bracket the jump of n(z). ONEPLUSZ0_IA = 1 + z_0, the pivot of the NLA
# power law in (1 + z) (eq. 25), which the C interface also sets to 1.62.
C1RHOCRIT = 0.01389      # nuisance.c1rhocrit_ia (cosmolike)
EDGE_EPS = 1e-9          # half-gap of the node pair at an n(z) support end
ONEPLUSZ0_IA = 1.62      # nuisance.oneplusz0_ia (IA_REDSHIFT_EVOLUTION)


def default_ells(lmax=75000, l_exact=30, n_per_decade=40):
    """Integer l = 1..l_exact-1, then log-spaced nodes up to lmax.

    The spectra are computed on these nodes and splined in ln l onto
    every integer l (ref_projection.cl_on_integers); every integer below
    l_exact is a node, so the spline reproduces those multipoles exactly.

    Arguments:
      lmax = last node (default 75000, the LMAX of the projections).
      l_exact = first log-spaced node (default 30).
      n_per_decade = log-spaced nodes per decade of l (default 40;
                     reference_cluster.DEFAULT_SETTINGS uses 60).

    Returns:
      numpy array of increasing multipoles (floats): l_exact - 1
      integers, then ceil(n_per_decade log10(lmax/l_exact)) + 1
      log-spaced nodes from l_exact to lmax (234 nodes in all at the
      DEFAULT_SETTINGS).
    """
    n = int(np.ceil(n_per_decade * np.log10(lmax / l_exact))) + 1
    return np.concatenate([np.arange(1, l_exact, dtype=float),
                           np.logspace(np.log10(l_exact), np.log10(lmax), n)])


def ell_prefactors(ells):
    """Return the curved-sky multipole prefactors (e1, e2) of the legs.

    e1 = l(l+1)/(l+1/2)^2 (magnification legs) and
    e2 = sqrt((l-1) l (l+1) (l+2))/(l+1/2)^2 (shear legs), set to 0 where
    the product under the root is not positive (l = 1).

    Arguments:
      ells = multipoles, numpy array.

    Returns:
      (e1, e2): two numpy arrays of the shape of ells.
    """
    e1 = ells * (ells + 1.0) / (ells + 0.5) ** 2
    t = (ells - 1.0) * ells * (ells + 1.0) * (ells + 2.0)
    e2 = np.where(t > 0, np.sqrt(np.maximum(t, 0.0)) / (ells + 0.5) ** 2, 0.0)
    return e1, e2


class NodeSet:
    """Limber z nodes with the background and P_NL(k_l, z) on them.

    Attributes (nn = number of nodes, nl = number of multipoles):
      z, w = Gauss-Legendre nodes and weights in z, (nn,)
      chi = chi(z) in Mpc/h; E = H/H0; dchidz = c/H in Mpc/h; D = growth
      ells = the multipoles (nl,)
      k = (l + 1/2)/chi in h/Mpc, (nl, nn)
      P = Halofit P_NL(k, z) in (Mpc/h)^3, (nl, nn)
      F = P w dchidz/chi^2, (nl, nn): the Limber measure times P_NL, so a
          spectrum is the sum over the nodes of L_X L_Y F (dimensionless).
    """

    def __init__(self, cosmo, breaks, max_width, order, ells):
        """Build the nodes and read the background and P_NL on them.

        Arguments:
          cosmo = ref_cosmology.Cosmology.
          breaks = z break points of the quadrature (ref_cluster.gl_panels).
          max_width = largest z panel width.
          order = Gauss-Legendre nodes per panel.
          ells = multipoles, numpy array (nl,).
        """
        self.z, self.w = gl_panels(breaks, max_width, order)
        self.chi = cosmo.chi(self.z)
        self.E = cosmo.E(self.z)
        self.dchidz = COVERH0 / self.E
        self.D = cosmo.growth(self.z)
        self.ells = ells
        # Limber wavenumber on the (l, node) grid: the column ells[:, None]
        # against the row chi[None, :]
        self.k = (ells[:, None] + 0.5) / self.chi[None, :]            # (l, n)
        # np.broadcast_to repeats the node redshifts along the l axis (a
        # read-only view, no copy)
        self.P = cosmo.P_nl(self.k, np.broadcast_to(self.z, self.k.shape))
        self.F = self.P * (self.w * self.dchidz / self.chi**2)[None, :]  # measure x P


class LimberModel:
    """All Limber spectra of the cluster data vector (and those its
    Gaussian covariance needs).

    The legs of the module docstring are built on two node sets (near: the
    cluster spectra; far: gg, gs and ss) and summed in compute().
    """

    def __init__(self, cosmo, cluster, nz_src, nz_lens, nuis, kernel_mode=0,
                 ells=None, lens_bins=None, z_panel=0.01, z_panel_far=0.02, order=8,
                 mag_ell_prefactor=True, spin2_prefactor=True, C_c=-2.0,
                 include_1h=True, dz_fine=2e-4, source_g_zmax=None):
        """Store the inputs and build the two node sets.

        Arguments:
          cosmo = ref_cosmology.Cosmology.
          cluster = ref_cluster.ClusterModel.
          nz_src, nz_lens = ref_cosmology.NzBins of the sources and the
                            lenses.
          nuis = dict of nuisance parameters: "source_dz" (per source
                 bin), "lens_dz", "lens_b1", "lens_bmag" (per lens bin of
                 the n(z) file), optional "lens_stretch" (default ones),
                 "IA_A1" and "IA_eta1" (default 0).
          kernel_mode = cluster radial kernel: 0 volume (default), 1
                        abundance (ref_cluster module docstring).
          ells = node multipoles (None: default_ells()).
          lens_bins = lens bins used, indices into the lens n(z) file
                      (None: range(nzc), lens bin i for cluster bin i).
          z_panel = largest z panel of the near node set (default 0.01).
          z_panel_far = largest z panel of the far node set (default 0.02).
          order = Gauss-Legendre nodes per panel (default 8).
          mag_ell_prefactor = True: e1(l) on the magnification legs;
                              False: 1.
          spin2_prefactor = True: e2(l) on the shear legs; False: 1.
          C_c = cluster magnification coefficient (default -2, eq. 28;
                0 switches it off).
          include_1h = True: the one-halo term enters C_cs.
          dz_fine = step of the fine z grid of the source and lens
                    lensing efficiencies (default 2e-4).
          source_g_zmax = diagnostic only: None (default), or a z above
                          which the source n(z) is dropped from W_kappa.
        """
        self.cosmo = cosmo
        self.cl = cluster
        self.nz_src = nz_src
        self.nz_lens = nz_lens
        self.nuis = nuis
        self.kernel_mode = kernel_mode
        self.ells = default_ells() if ells is None else np.asarray(ells, dtype=float)
        self.lens_bins = list(range(cluster.nzc)) if lens_bins is None else list(lens_bins)
        self.C_c = C_c
        self.include_1h = include_1h
        e1, e2 = ell_prefactors(self.ells)
        self.e1 = e1 if mag_ell_prefactor else np.ones_like(e1)
        self.e2 = e2 if spin2_prefactor else np.ones_like(e2)
        self.dz_fine = dz_fine
        self.source_g_zmax = source_g_zmax

        # --- node sets -------------------------------------------------
        # near: z nodes from 1e-4 to the top of the highest cluster bin's
        # support, with panel ends at every break of every cluster kernel;
        # all cluster spectra (cc, cs, cg) are sums over these nodes. The
        # generator inside max() gives the largest upper support end.
        zc_top = max(cluster.support(i)[1] for i in range(cluster.nzc))
        cl_breaks = [1e-4, zc_top]
        for i in range(cluster.nzc):
            cl_breaks += cluster.breaks(i)
        self.near = NodeSet(cosmo, cl_breaks, z_panel, order, self.ells)
        # far: wider panels up to z_top for gg, gs and ss (and the top of
        # the fine grids of _fine). A positive photo-z shift moves an n(z)
        # up by dz; the extra 0.1 on the lens side extends the range above
        # the shifted lens support (the reason for the value is not
        # documented).
        z_top = max(nz_src.z_nodes[-1] + max(0.0, max(nuis["source_dz"])),
                    nz_lens.z_nodes[-1] + max(0.0, max(nuis["lens_dz"])) + 0.1)
        self.far = NodeSet(cosmo, [1e-4, zc_top, z_top], z_panel_far, order, self.ells)
        self.z_top = z_top

    # ------------------------------------------------------------------
    # kernels
    # ------------------------------------------------------------------
    def _fine(self, edges=()):
        """Fine z grid of the lensing-efficiency trapezoids. `edges` are the
        ends of an n(z) support: n(z) jumps to 0 there (NzBins), and a
        trapezoid cell straddling the jump is wrong by O(dz n_edge), a
        relative error of 1e-4 - 1e-3 on g at redshifts behind most of the
        bin's sources (where g is small). A pair of nodes EDGE_EPS inside
        and outside each end makes the trapezoid exact up to O(dz^2).

        Arguments:
          edges = support ends (z) of one n(z), e.g. NzBins.support(...);
                  empty (default) adds no node pairs.

        Returns:
          (zf, chif): the grid 0, dz_fine, 2 dz_fine, ... up to z_top plus
          the node pairs, sorted, and chi(zf) in Mpc/h.
        """
        zf = np.arange(0.0, self.z_top + self.dz_fine, self.dz_fine)
        if len(edges):
            # one node pair straddling every support end (a list of pairs,
            # flattened by np.concatenate); pairs at z <= 0 are dropped,
            # and np.unique merges the nodes into the grid and sorts it
            pts = np.concatenate([[e - EDGE_EPS, e + EDGE_EPS] for e in edges])
            zf = np.unique(np.concatenate([zf, pts[pts > 0]]))
        return zf, self.cosmo.chi(zf)

    def source_kernels(self, ns):
        """W_kappa (s, n) and W_src (s, n) and C1 (n) on a node set.

        W_kappa,j = (3/2) Omega_m (H0/c)^2 chi (1+z) g_j(z), with the
        lensing efficiency g_j of the shifted source n(z) computed on the
        fine grid of _fine; W_src,j = n_j(z) H/c; C1(z) = A1
        ((1+z)/1.62)^eta1 Omega_m c1rhocrit/D(z), the NLA amplitude
        (module docstring).

        Arguments:
          ns = the NodeSet of the integrals.

        Returns:
          (Wk, Ws, C1): Wk and Ws of shape (source bins, nn) in 1/(Mpc/h),
          C1 of shape (nn,), dimensionless.
        """
        # nu here is the nuisance-parameter dict (not a peak height)
        cosmo, nu = self.cosmo, self.nuis
        Om = cosmo.Omega_m
        # (3/2) Omega_m (H0/c)^2 chi (1+z) at the nodes, in 1/(Mpc/h)
        pref = 1.5 * Om / COVERH0**2 * ns.chi * (1.0 + ns.z)
        Wk, Ws = [], []
        for s in range(self.nz_src.nbin):
            dz = nu["source_dz"][s]
            zf, chif = self._fine(self.nz_src.support(s, shift=dz))
            nf = self.nz_src(zf, s, shift=dz)
            if self.source_g_zmax is not None:
                # Diagnostic only: the lensing efficiency ignores the
                # sources above source_g_zmax. The "diagnostic" variant of
                # tests/validation/compare_reference.py sets it to the
                # upper edge of the unshifted source n(z) file, to measure
                # an integral that stops at that edge.
                nf = np.where(zf <= self.source_g_zmax, nf, 0.0)
            g = lensing_efficiency(ns.z, ns.chi, zf, nf, chif)
            Wk.append(pref * g)
            Ws.append(self.nz_src(ns.z, s, shift=dz) * ns.E / COVERH0)
        # dict.get(key, default): 0 when nuis carries no IA entries
        A1 = nu.get("IA_A1", 0.0)
        eta = nu.get("IA_eta1", 0.0)
        C1 = A1 * ((1.0 + ns.z) / ONEPLUSZ0_IA) ** eta * Om * C1RHOCRIT / ns.D
        return np.array(Wk), np.array(Ws), C1

    def lens_kernels(self, ns):
        """W_g (g, n) and W_mag,g (g, n) for the lens bins in use.

        W_g = n_g(z) H/c of the shifted and stretched lens n(z), and
        W_mag,g = (3/2) Omega_m (H0/c)^2 chi (1+z) times the lensing
        efficiency of that n(z): the magnification leg before bmag and e1.

        Arguments:
          ns = the NodeSet of the integrals.

        Returns:
          (Wg, Wm): numpy arrays of shape (len(lens_bins), nn), in
          1/(Mpc/h).
        """
        # nu here is the nuisance-parameter dict (not a peak height)
        cosmo, nu = self.cosmo, self.nuis
        pref = 1.5 * cosmo.Omega_m / COVERH0**2 * ns.chi * (1.0 + ns.z)
        Wg, Wm = [], []
        for g in self.lens_bins:
            dz = nu["lens_dz"][g]
            # stretch of lens bin g (1 when nuis has no "lens_stretch")
            st = nu.get("lens_stretch", [1.0] * self.nz_lens.nbin)[g]
            zf, chif = self._fine(self.nz_lens.support(g, shift=dz, stretch=st))
            nf = self.nz_lens(zf, g, shift=dz, stretch=st)
            Wm.append(pref * lensing_efficiency(ns.z, ns.chi, zf, nf, chif))
            Wg.append(self.nz_lens(ns.z, g, shift=dz, stretch=st) * ns.E / COVERH0)
        return np.array(Wg), np.array(Wm)

    def cluster_kernels(self, ns):
        """W_c (i, A, n), W_mag,c (i, A, n), b_A (A, n), n_A (A, n).

        W_c = q_iA(z) H/c (ref_cluster ClusterModel.q) and
        W_mag,c = (3/2) Omega_m (H0/c)^2 chi (1+z) g_c,iA(z)
        (ClusterModel.magnification_efficiency).

        Arguments:
          ns = the NodeSet of the integrals.

        Returns:
          (Wc, Wm, b, n): Wc and Wm of shape (nzc, nA, nn) in 1/(Mpc/h),
          b (nA, nn) dimensionless, n (nA, nn) in (h/Mpc)^3.

        Side effects: stores the kernel normalizations in self.norms.
        """
        cl, cosmo = self.cl, self.cosmo
        norms = cl.kernel_norms(self.kernel_mode)
        n, b = cl.n_b(ns.z)
        pref = 1.5 * cosmo.Omega_m / COVERH0**2 * ns.chi * (1.0 + ns.z)
        Wc = np.zeros((cl.nzc, cl.nA, ns.z.size))
        Wm = np.zeros_like(Wc)
        for i in range(cl.nzc):
            q = cl.q(ns.z, i, self.kernel_mode, norms, nb=(n, b))
            Wc[i] = q * (ns.E / COVERH0)[None, :]
            Wm[i] = pref[None, :] * cl.magnification_efficiency(ns.z, i, self.kernel_mode, norms)
        self.norms = norms
        return Wc, Wm, b, n

    def p1h_on_nodes(self, ns, Wc):
        """P^1h_A(k_ln, z_n) (A, l, n); zero where no cluster kernel lives.

        The one-halo term enters C_cs multiplied by W_c, so it is computed
        only at the nodes where some cluster kernel is positive.

        Arguments:
          ns = the NodeSet (its k grid gives k_ln = (l + 1/2)/chi_n).
          Wc = cluster kernels of cluster_kernels, (nzc, nA, nn).

        Returns:
          numpy array of shape (nA, nl, nn) in (Mpc/h)^3.
        """
        P1 = np.zeros((self.cl.nA, self.ells.size, ns.z.size))
        # indices of the nodes where any W_c[i, A] > 0 (np.any over the
        # axes i and A)
        live = np.where(np.any(Wc > 0, axis=(0, 1)))[0]
        for n in live:
            P1[:, :, n] = self.cl.p1h(ns.k[:, n], ns.z[n])
        return P1

    # ------------------------------------------------------------------
    # spectra
    # ------------------------------------------------------------------
    def compute(self, with_cov_spectra=True):
        """Return a dict of spectra on self.ells:

          C_cc (i, A, j, B, l)   all z-bin pairs (data vector: i = j, A <= B)
          C_cs (i, A, s, l)      2-halo + 1-halo (+ magnification, IA)
          C_cs_1h (i, A, s, l)   the 1-halo part alone
          C_cg (i, A, g, l)      g over lens_bins (data vector: g = i)
          C_gg (g, g', l), C_gs (g, s, l), C_ss (s, s', l)   (with_cov_spectra)

        and "ells". Legend: i, j = cluster z bins; A, B = richness bins;
        s, s' = source bins; g, g' = lens bins of lens_bins; l = multipole
        nodes. Every C_l is dimensionless. The cluster spectra are sums
        over the near node set, gg, gs and ss over the far one.

        Arguments:
          with_cov_spectra = True (default) also computes C_gg, C_gs and
                             C_ss, which only the covariance needs.

        Side effects: sets self.nodes (the node-level kernels, kept for
        diagnostics) and self.norms (cluster_kernels).
        """
        ns = self.near
        # e1 as a column (nl, 1), to multiply legs of shape (.., nl, nn)
        e1 = self.e1[:, None]
        Wc, Wmc, b, nA = self.cluster_kernels(ns)
        Wk, Wsrc, C1 = self.source_kernels(ns)
        Wg, Wmg = self.lens_kernels(ns)
        bg = np.array([self.nuis["lens_b1"][g] for g in self.lens_bins])
        bm = np.array([self.nuis["lens_bmag"][g] for g in self.lens_bins])

        # legs on (.., l, n). Indexing with None inserts a length-1 axis:
        # b (A, n) as b[None, :, None, :] and Wc (i, A, n) as
        # Wc[:, :, None, :] broadcast to (i, A, l, n); the magnification
        # terms carry e1(l). Lenses: bias W_g + bmag e1 W_mag,g, (g, l, n).
        Lc = (b[None, :, None, :] * Wc[:, :, None, :]
              + self.C_c * e1[None, None] * Wmc[:, :, None, :])          # (i, A, l, n)
        Ls = Wk - Wsrc * C1[None, :]                                      # (s, n)
        Lg = bg[:, None, None] * Wg[:, None, :] + bm[:, None, None] * e1[None] * Wmg[:, None, :]

        F = ns.F
        out = {"ells": self.ells}
        # np.einsum sums the product of its operands over every index that
        # the output string omits: "ialn,jbln,ln->iajbl" is
        # C_cc[i,a,j,b,l] = sum_n Lc[i,a,l,n] Lc[j,b,l,n] F[l,n], the sum
        # over the z nodes n. optimize=True lets numpy choose the order of
        # the pairwise contractions.
        out["C_cc"] = np.einsum("ialn,jbln,ln->iajbl", Lc, Lc, F, optimize=True)
        # two-halo: C_cs[i,a,s,l] = sum_n Lc[i,a,l,n] Ls[s,n] F[l,n]
        cs2h = np.einsum("ialn,sn,ln->iasl", Lc, Ls, F, optimize=True)
        if self.include_1h:
            P1 = self.p1h_on_nodes(ns, Wc)
            # one-halo: W_c W_kappa P^1h with the Limber measure
            # w dchi/dz/chi^2 (no P_NL); no bias, magnification or IA
            meas = ns.w * ns.dchidz / ns.chi**2
            cs1h = np.einsum("ian,sn,aln,n->iasl", Wc, Wk, P1, meas, optimize=True)
        else:
            P1 = None
            cs1h = np.zeros_like(cs2h)
        # the shear prefactor e2(l) of the source leg, broadcast over
        # (i, A, s)
        out["C_cs"] = (cs2h + cs1h) * self.e2[None, None, None, :]
        out["C_cs_1h"] = cs1h * self.e2[None, None, None, :]
        # C_cg[i,a,g,l] = sum_n Lc[i,a,l,n] Lg[g,l,n] F[l,n]
        out["C_cg"] = np.einsum("ialn,gln,ln->iagl", Lc, Lg, F, optimize=True)

        # node-level diagnostics
        self.nodes = dict(z=ns.z, Wc=Wc, Wmag_c=Wmc, b_A=b, n_A=nA, Wkappa=Wk, Wsrc=Wsrc,
                          Wg=Wg, Wmag_g=Wmg, C1=C1)
        # gg, gs and ss on the far node set; C_gs carries e2 once (one
        # shear leg), C_ss e2^2 (two)
        if with_cov_spectra:
            fs = self.far
            Wk_f, Ws_f, C1_f = self.source_kernels(fs)
            Wg_f, Wmg_f = self.lens_kernels(fs)
            Ls_f = Wk_f - Ws_f * C1_f[None, :]
            Lg_f = (bg[:, None, None] * Wg_f[:, None, :]
                    + bm[:, None, None] * e1[None] * Wmg_f[:, None, :])
            out["C_gg"] = np.einsum("gln,hln,ln->ghl", Lg_f, Lg_f, fs.F, optimize=True)
            out["C_gs"] = (np.einsum("gln,sn,ln->gsl", Lg_f, Ls_f, fs.F, optimize=True)
                           * self.e2[None, None, :])
            out["C_ss"] = (np.einsum("sn,tn,ln->stl", Ls_f, Ls_f, fs.F, optimize=True)
                           * (self.e2**2)[None, None, :])
        return out
