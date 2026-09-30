"""Limber angular power spectra of the DES cluster reference.

With k = (l + 1/2)/chi and every kernel in 1/(Mpc/h),

  C^{XY}_l = int dchi L_X(chi, l) L_Y(chi, l) P_NL(k, z)/chi^2   (+ 1-halo for cs)

Legs (radial kernels), following cosmolike's cosmo2D.c conventions:

  cluster  c_iA : L = b_A(z) W_c,iA(chi) + e1(l) C_c W_mag,c,iA(chi),  C_c = -2
                  W_c = q_iA(z) H(z)/c,  W_mag,c = (3/2) Om (H0/c)^2 chi (1+z) g_c
                  (g_c: the cluster lensing efficiency over the FULL foreground)
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

Cluster spectra are integrated on z nodes up to the top of the cluster
photo-z support (both legs of each cluster spectrum vanish above it);
gg, gs and ss (needed by the covariance only) on nodes up to the top of
the n(z) files.
"""

import numpy as np

from ref_cosmology import COVERH0
from ref_cluster import gl_panels, lensing_efficiency

C1RHOCRIT = 0.01389      # nuisance.c1rhocrit_ia (cosmolike)
ONEPLUSZ0_IA = 1.62      # nuisance.oneplusz0_ia (IA_REDSHIFT_EVOLUTION)


def default_ells(lmax=75000, l_exact=30, n_per_decade=40):
    """Integer l = 1..l_exact-1, then log-spaced nodes up to lmax."""
    n = int(np.ceil(n_per_decade * np.log10(lmax / l_exact))) + 1
    return np.concatenate([np.arange(1, l_exact, dtype=float),
                           np.logspace(np.log10(l_exact), np.log10(lmax), n)])


def ell_prefactors(ells):
    e1 = ells * (ells + 1.0) / (ells + 0.5) ** 2
    t = (ells - 1.0) * ells * (ells + 1.0) * (ells + 2.0)
    e2 = np.where(t > 0, np.sqrt(np.maximum(t, 0.0)) / (ells + 0.5) ** 2, 0.0)
    return e1, e2


class NodeSet:
    """Limber z nodes with the background and P_NL(k_l, z) on them."""

    def __init__(self, cosmo, breaks, max_width, order, ells):
        self.z, self.w = gl_panels(breaks, max_width, order)
        self.chi = cosmo.chi(self.z)
        self.E = cosmo.E(self.z)
        self.dchidz = COVERH0 / self.E
        self.D = cosmo.growth(self.z)
        self.ells = ells
        self.k = (ells[:, None] + 0.5) / self.chi[None, :]            # (l, n)
        self.P = cosmo.P_nl(self.k, np.broadcast_to(self.z, self.k.shape))
        self.F = self.P * (self.w * self.dchidz / self.chi**2)[None, :]  # measure x P


class LimberModel:
    """All Limber spectra of the cluster data vector (and those its
    Gaussian covariance needs)."""

    def __init__(self, cosmo, cluster, nz_src, nz_lens, nuis, kernel_mode=0,
                 ells=None, lens_bins=None, z_panel=0.01, z_panel_far=0.02, order=8,
                 mag_ell_prefactor=True, spin2_prefactor=True, C_c=-2.0,
                 include_1h=True, dz_fine=2e-4):
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

        # --- node sets -------------------------------------------------
        zc_top = max(cluster.support(i)[1] for i in range(cluster.nzc))
        cl_breaks = [1e-4, zc_top]
        for i in range(cluster.nzc):
            cl_breaks += cluster.breaks(i)
        self.near = NodeSet(cosmo, cl_breaks, z_panel, order, self.ells)
        z_top = max(nz_src.z_nodes[-1] + max(0.0, max(nuis["source_dz"])),
                    nz_lens.z_nodes[-1] + max(0.0, max(nuis["lens_dz"])) + 0.1)
        self.far = NodeSet(cosmo, [1e-4, zc_top, z_top], z_panel_far, order, self.ells)
        self.z_top = z_top

    # ------------------------------------------------------------------
    # kernels
    # ------------------------------------------------------------------
    def _fine(self):
        zf = np.arange(0.0, self.z_top + self.dz_fine, self.dz_fine)
        return zf, self.cosmo.chi(zf)

    def source_kernels(self, ns):
        """W_kappa (s, n) and W_src (s, n) and C1 (n) on a node set."""
        cosmo, nu = self.cosmo, self.nuis
        zf, chif = self._fine()
        Om = cosmo.Omega_m
        pref = 1.5 * Om / COVERH0**2 * ns.chi * (1.0 + ns.z)
        Wk, Ws = [], []
        for s in range(self.nz_src.nbin):
            dz = nu["source_dz"][s]
            nf = self.nz_src(zf, s, shift=dz)
            g = lensing_efficiency(ns.z, ns.chi, zf, nf, chif)
            Wk.append(pref * g)
            Ws.append(self.nz_src(ns.z, s, shift=dz) * ns.E / COVERH0)
        A1 = nu.get("IA_A1", 0.0)
        eta = nu.get("IA_eta1", 0.0)
        C1 = A1 * ((1.0 + ns.z) / ONEPLUSZ0_IA) ** eta * Om * C1RHOCRIT / ns.D
        return np.array(Wk), np.array(Ws), C1

    def lens_kernels(self, ns):
        """W_g (g, n) and W_mag,g (g, n) for the lens bins in use."""
        cosmo, nu = self.cosmo, self.nuis
        zf, chif = self._fine()
        pref = 1.5 * cosmo.Omega_m / COVERH0**2 * ns.chi * (1.0 + ns.z)
        Wg, Wm = [], []
        for g in self.lens_bins:
            dz = nu["lens_dz"][g]
            st = nu.get("lens_stretch", [1.0] * self.nz_lens.nbin)[g]
            nf = self.nz_lens(zf, g, shift=dz, stretch=st)
            Wm.append(pref * lensing_efficiency(ns.z, ns.chi, zf, nf, chif))
            Wg.append(self.nz_lens(ns.z, g, shift=dz, stretch=st) * ns.E / COVERH0)
        return np.array(Wg), np.array(Wm)

    def cluster_kernels(self, ns):
        """W_c (i, A, n), W_mag,c (i, A, n), b_A (A, n), n_A (A, n)."""
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
        """P^1h_A(k_ln, z_n) (A, l, n); zero where no cluster kernel lives."""
        P1 = np.zeros((self.cl.nA, self.ells.size, ns.z.size))
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
        """
        ns = self.near
        e1 = self.e1[:, None]
        Wc, Wmc, b, nA = self.cluster_kernels(ns)
        Wk, Wsrc, C1 = self.source_kernels(ns)
        Wg, Wmg = self.lens_kernels(ns)
        bg = np.array([self.nuis["lens_b1"][g] for g in self.lens_bins])
        bm = np.array([self.nuis["lens_bmag"][g] for g in self.lens_bins])

        # legs on (.., l, n)
        Lc = (b[None, :, None, :] * Wc[:, :, None, :]
              + self.C_c * e1[None, None] * Wmc[:, :, None, :])          # (i, A, l, n)
        Ls = Wk - Wsrc * C1[None, :]                                      # (s, n)
        Lg = bg[:, None, None] * Wg[:, None, :] + bm[:, None, None] * e1[None] * Wmg[:, None, :]

        F = ns.F
        out = {"ells": self.ells}
        out["C_cc"] = np.einsum("ialn,jbln,ln->iajbl", Lc, Lc, F, optimize=True)
        cs2h = np.einsum("ialn,sn,ln->iasl", Lc, Ls, F, optimize=True)
        if self.include_1h:
            P1 = self.p1h_on_nodes(ns, Wc)
            meas = ns.w * ns.dchidz / ns.chi**2
            cs1h = np.einsum("ian,sn,aln,n->iasl", Wc, Wk, P1, meas, optimize=True)
        else:
            P1 = None
            cs1h = np.zeros_like(cs2h)
        out["C_cs"] = (cs2h + cs1h) * self.e2[None, None, None, :]
        out["C_cs_1h"] = cs1h * self.e2[None, None, None, :]
        out["C_cg"] = np.einsum("ialn,gln,ln->iagl", Lc, Lg, F, optimize=True)

        # node-level diagnostics
        self.nodes = dict(z=ns.z, Wc=Wc, Wmag_c=Wmc, b_A=b, n_A=nA, Wkappa=Wk, Wsrc=Wsrc,
                          Wg=Wg, Wmag_g=Wmg, C1=C1)
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
