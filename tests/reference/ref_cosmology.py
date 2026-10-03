"""Background, growth, linear/nonlinear P(k) and tomographic n(z) for the
DES cluster Python reference.

Everything is in "h units":

  k    in h/Mpc              chi, D_M  in Mpc/h
  P(k) in (Mpc/h)^3          M         in M_sun/h
  rho  in (M_sun/h)/(Mpc/h)^3

The power spectra come from python CAMB (linear, and Halofit for the
nonlinear one), called directly: no cobaya, no cosmolike tables.

Conventions mirrored from cocoa/cosmolike (so that the C port can be fed
the same inputs and compared number by number):

  * Omega_m is the TOTAL matter density, Omega_c + Omega_b + Omega_nu, as
    the cocoa likelihoods pass it (omegam). omch2 = Omega_m h^2 - ombh2 -
    omnuh2.
  * rho_crit = 7.4775e21 (M_sun/h)/(c/H0)^3 (cosmolike structs.c), i.e.
    2.77520e11 (M_sun/h)/(Mpc/h)^3 (CODATA gives 2.77537e11; 6e-5 apart).
  * The non-halo growth factor retains the external DES reference
    convention: D(z) = sqrt(P_lin(k0, z)/P_lin(k0, 0))
    with k0 = 0.0005 1/Mpc on the total-matter linear spectrum. It is
    scale independent by construction and used for non-halo terms. Halo
    statistics integrate P_cb(k,z) at each redshift directly.
  * n(z) files are read as in redshift_spline.c: the z column holds the
    LEFT edges of the histogram cells (Z_LOW convention), the values sit
    at the cell centres, each bin is normalized by the rectangle sum
    sum_k n_k dz (not by the integral of the spline), and n(z) is a
    natural cubic spline through the centres, zero outside
    [first centre, last centre]. Sources: n(z - dz_s). Lenses:
    n((z - dz_l - zbar)/s + zbar)/s with zbar the mean of the unshifted
    bin.
"""

import numpy as np
import camb
from scipy.interpolate import CubicSpline, RectBivariateSpline

C_KMS = 299792.458                  # speed of light in km/s
COVERH0 = 2997.92458                # c/H0 in Mpc/h
RHO_CRIT = 7.4775e21 / COVERH0**3   # cosmolike rho_crit in (M_sun/h)/(Mpc/h)^3
K0_GROWTH_INV_MPC = 0.0005          # k (1/Mpc) of the external DES convention


def default_pk_redshifts(z_max=4.0):
    """z nodes of the P(k, z) tables: dense where the clusters live."""
    z1 = np.linspace(0.0, 1.0, 81)                  # dz = 0.0125
    z2 = np.linspace(1.0, 2.0, 21)[1:]              # dz = 0.05
    z3 = np.linspace(2.0, z_max, 21)[1:]            # dz = 0.1
    return np.concatenate([z1, z2, z3])


class Cosmology:
    """CAMB background and power spectra, interpolated in (z, ln k).

    Parameters (dict `params`): Omega_m, Omega_b, h, A_s, n_s,
    Omega_nu_h2 (optional w0, wa, tau). Settings (keyword):

      kmax           largest k (h/Mpc) CAMB computes; beyond it ln P is
                     extrapolated linearly in ln k from the last two nodes
      halofit        CAMB halofit_version for P_NL ("takahashi")
      num_massive_nu number of degenerate massive species (3)
      z_nodes        z nodes of the P tables (default_pk_redshifts)
      nl_z_order     spline order in z of the ln P_NL table (3 = cubic,
                     1 = linear between the CAMB z nodes). CAMB's Halofit
                     finds its nonlinear scale by bisection to |sigma - 1|
                     <= 1e-3 (halofit.f90), so ln P_NL carries ~1e-3 node-
                     to-node noise in z at k > k_NL; a cubic spline through
                     it and a linear read of the same nodes (cosmolike's
                     p_nonlin) then differ by that much between nodes. A C
                     comparison uses 1 and hands the C side these nodes.
    """

    def __init__(self, params, kmax=100.0, halofit="takahashi",
                 num_massive_nu=3, z_nodes=None, accuracy_boost=1.0,
                 k_per_logint=None, nl_z_order=3):
        p = dict(params)
        self.params = p
        self.h = h = float(p["h"])
        self.Omega_m = float(p["Omega_m"])
        self.Omega_b = float(p["Omega_b"])
        self.omnuh2 = float(p.get("Omega_nu_h2", 0.0))
        self.Omega_nu = self.omnuh2 / h**2
        self.Omega_cb = self.Omega_m - self.Omega_nu
        self.A_s = float(p["A_s"])
        self.n_s = float(p["n_s"])
        self.w0 = float(p.get("w0", -1.0))
        self.wa = float(p.get("wa", 0.0))
        self.halofit = halofit
        self.kmax = float(kmax)
        z_nodes = default_pk_redshifts() if z_nodes is None else np.asarray(z_nodes)
        self.z_pk = np.sort(z_nodes)

        ombh2 = self.Omega_b * h**2
        omch2 = self.Omega_m * h**2 - ombh2 - self.omnuh2

        pars = camb.CAMBparams()
        mnu = 0.06
        pars.set_cosmology(H0=100.0 * h, ombh2=ombh2, omch2=omch2, omk=0.0,
                           mnu=mnu, num_massive_neutrinos=num_massive_nu,
                           neutrino_hierarchy="degenerate",
                           tau=float(p.get("tau", 0.0697186)))
        if self.omnuh2 > 0:
            # omnuh2 is linear in mnu: rescale to hit the target exactly
            mnu = mnu * self.omnuh2 / pars.omnuh2
            pars.set_cosmology(H0=100.0 * h, ombh2=ombh2, omch2=omch2, omk=0.0,
                               mnu=mnu, num_massive_neutrinos=num_massive_nu,
                               neutrino_hierarchy="degenerate",
                               tau=float(p.get("tau", 0.0697186)))
            assert abs(pars.omnuh2 / self.omnuh2 - 1) < 1e-10
        else:
            pars.set_cosmology(H0=100.0 * h, ombh2=ombh2, omch2=omch2, omk=0.0,
                               mnu=0.0, num_massive_neutrinos=0,
                               tau=float(p.get("tau", 0.0697186)))
        self.mnu = mnu
        pars.InitPower.set_params(As=self.A_s, ns=self.n_s)
        if self.w0 != -1.0 or self.wa != 0.0:
            pars.set_dark_energy(w=self.w0, wa=self.wa, dark_energy_model="ppf")
        pars.set_accuracy(AccuracyBoost=accuracy_boost)
        pars.set_matter_power(redshifts=list(self.z_pk[::-1]), kmax=self.kmax * h,
                              nonlinear=True, k_per_logint=k_per_logint,
                              accurate_massive_neutrino_transfers=True,
                              silent=True)
        pars.NonLinear = camb.model.NonLinear_both
        pars.NonLinearModel.set_params(halofit_version=halofit)
        pars.WantCls = False
        self.camb_pars = pars
        self.results = res = camb.get_results(pars)

        def table(var, nonlinear):
            if nonlinear:
                k, z, pk = res.get_nonlinear_matter_power_spectrum(
                    var1=var, var2=var, hubble_units=True, k_hunit=True)
            else:
                k, z, pk = res.get_linear_matter_power_spectrum(
                    var1=var, var2=var, hubble_units=True, k_hunit=True)
            z = np.asarray(z)
            order = np.argsort(z)
            return np.asarray(k), z[order], np.asarray(pk)[order]

        k, z, plin = table("delta_tot", False)
        _, _, pnl = table("delta_tot", True)
        _, _, pcb = table("delta_nonu", False)
        assert np.allclose(z, self.z_pk, atol=1e-8)
        self.k_pk = k
        self.lnk_pk = np.log(k)
        self._tabs = {"lin": np.log(plin), "nl": np.log(pnl), "cb": np.log(pcb)}
        self._spl = {key: RectBivariateSpline(self.z_pk, self.lnk_pk, tab,
                                              kx=(nl_z_order if key == "nl" else 3), ky=3)
                     for key, tab in self._tabs.items()}
        # end slopes d ln P / d ln k (per z node) for the extrapolation
        self._slope_lo = {key: (tab[:, 1] - tab[:, 0]) / (self.lnk_pk[1] - self.lnk_pk[0])
                          for key, tab in self._tabs.items()}
        self._slope_hi = {key: (tab[:, -1] - tab[:, -2]) / (self.lnk_pk[-1] - self.lnk_pk[-2])
                          for key, tab in self._tabs.items()}

        # Non-halo growth (external DES reference convention).
        k0 = K0_GROWTH_INV_MPC / h
        self._lnP0_k0 = float(self.lnP(np.array([k0]), np.array([0.0]), "lin")[0])
        self._k0 = k0

        # dense background tables for inversions z(chi)
        zz = np.linspace(0.0, 5.0, 20001)
        self._chi_tab = self.chi(zz)
        self._z_of_chi = CubicSpline(self._chi_tab, zz)

    # ------------------------------------------------------------------
    # background
    # ------------------------------------------------------------------
    def chi(self, z):
        """Comoving distance in Mpc/h (flat: also the transverse distance)."""
        z = np.asarray(z, dtype=float)
        out = np.asarray(self.results.comoving_radial_distance(np.atleast_1d(z))) * self.h
        return out.reshape(z.shape)

    def E(self, z):
        """H(z)/H0."""
        z = np.asarray(z, dtype=float)
        out = np.asarray(self.results.hubble_parameter(np.atleast_1d(z))) / (100.0 * self.h)
        return out.reshape(z.shape)

    def dchi_dz(self, z):
        """c/H(z) in Mpc/h."""
        return COVERH0 / self.E(z)

    def z_of_chi(self, chi):
        return self._z_of_chi(chi)

    def growth(self, z):
        """D(z), D(0) = 1: sqrt(P_lin(k0, z)/P_lin(k0, 0)), k0 = 0.0005/Mpc."""
        z = np.atleast_1d(np.asarray(z, dtype=float))
        lnp = self.lnP(np.full(z.shape, self._k0), z, "lin")
        return np.exp(0.5 * (lnp - self._lnP0_k0))

    # ------------------------------------------------------------------
    # power spectra
    # ------------------------------------------------------------------
    def lnP(self, k, z, kind="nl"):
        """ln P(k, z) at matching arrays k (h/Mpc), z (broadcast)."""
        k, z = np.broadcast_arrays(np.asarray(k, dtype=float), np.asarray(z, dtype=float))
        shape = k.shape
        lnk = np.log(k).ravel()
        zr = z.ravel()
        lo, hi = self.lnk_pk[0], self.lnk_pk[-1]
        lnk_c = np.clip(lnk, lo, hi)
        out = self._spl[kind].ev(zr, lnk_c)
        below = lnk < lo
        above = lnk > hi
        if below.any() or above.any():
            # end slopes interpolated linearly in z
            if below.any():
                s = np.interp(zr[below], self.z_pk, self._slope_lo[kind])
                out[below] += s * (lnk[below] - lo)
            if above.any():
                s = np.interp(zr[above], self.z_pk, self._slope_hi[kind])
                out[above] += s * (lnk[above] - hi)
        return out.reshape(shape)

    def P_lin(self, k, z, kind="lin"):
        return np.exp(self.lnP(k, z, kind))

    def P_nl(self, k, z):
        return np.exp(self.lnP(k, z, "nl"))

    def export_tables(self):
        """The inputs a C comparison should be fed: (z, k, ln P) and background."""
        zb = np.linspace(0.0, 4.0, 4001)
        return {
            "cosmo_z_pk": self.z_pk, "cosmo_k_pk": self.k_pk,
            "cosmo_lnP_lin": self._tabs["lin"], "cosmo_lnP_nl": self._tabs["nl"],
            "cosmo_lnP_cb": self._tabs["cb"],
            "cosmo_z_bg": zb, "cosmo_chi": self.chi(zb), "cosmo_E": self.E(zb),
            "cosmo_D": self.growth(zb),
        }


class NzBins:
    """Tomographic n(z) of a cosmolike-format text file (z, n_1, ..., n_N).

    Mirrors redshift_spline.c (see the module docstring): Z_LOW convention,
    rectangle-sum normalization, natural cubic spline through the cell
    centres. `norm="spline"` normalizes the spline itself to unit integral
    instead (not the C convention).
    """

    def __init__(self, path, convention="z_low", norm="rect"):
        data = np.loadtxt(path)
        zf = data[:, 0]
        raw = data[:, 1:]
        n = len(zf)
        zmin_all = max(zf[0], 1e-5)
        zmax_all = zf[-1] + (zf[-1] - zf[0]) / (n - 1.0)
        dz = (zmax_all - zmin_all) / n
        off = 0.5 if convention == "z_low" else 0.0
        self.z_nodes = zmin_all + (np.arange(n) + off) * dz
        self.dz = dz
        self.nbin = raw.shape[1]
        norm_rect = raw.sum(axis=0) * dz
        self.norm_rect = norm_rect
        self.splines = []
        for i in range(self.nbin):
            spl = CubicSpline(self.z_nodes, raw[:, i] / norm_rect[i], bc_type="natural")
            if norm == "spline":
                tot = spl.integrate(self.z_nodes[0], self.z_nodes[-1])
                spl = CubicSpline(self.z_nodes, raw[:, i] / norm_rect[i] / tot,
                                  bc_type="natural")
            self.splines.append(spl)
        # mean z of each unshifted bin (for the lens stretch)
        zz = np.linspace(self.z_nodes[0], self.z_nodes[-1], 30001)
        self.zmean = np.array([np.trapz(zz * s(zz), zz) / np.trapz(s(zz), zz)
                               for s in self.splines])

    def support(self, i, shift=0.0, stretch=1.0):
        """Ends (z) of the open interval where n_i(z; shift, stretch) != 0."""
        zb = self.zmean[i]
        return tuple(zb + stretch * (self.z_nodes[j] - zb) + shift for j in (0, -1))

    def __call__(self, z, i, shift=0.0, stretch=1.0):
        z = np.asarray(z, dtype=float)
        zz = (z - shift - self.zmean[i]) / stretch + self.zmean[i]
        out = self.splines[i](zz) / stretch
        inside = (zz > self.z_nodes[0]) & (zz < self.z_nodes[-1])
        return np.where(inside, out, 0.0)
