"""Halo-model ingredients of the DES cluster reference, with the
CONVENTIONS of cosmolike's halo.c (NEW library) but an independent
implementation (no C code structure is ported).

Conventions mirrored from halo.c (checked by reading the source):

  * Halo definition M_200m: Delta = 200 times the MEAN matter density,
    rho_m = rho_crit * Omega_m with Omega_m the TOTAL matter density
    (halo.c: `rho_m = cosmology.rho_crit * cosmology.Omega_m`).
  * Peak height nu = delta_c/sigma_cb(M,z), delta_c = 1.686. Integrate
    P_cb(k,z) directly with Lagrangian radius (3M/(4 pi rho_cb))^(1/3).
    The independent oracle retains hmf_matter="tot" for diagnostic
    comparisons; production halos always use cb.
  * Mass function dn/dlnM = (rho_cb/M) nu f(nu) dln nu/dln M with the
    Tinker et al. 2010 (1001.3162) multiplicity at Delta = 200m, Eqs. 8-12,
    Table 4: beta = 0.589 aa^-0.2, gamma = 0.864 aa^0.01,
    phi = -0.729 aa^0.08, eta = -0.243 aa^-0.27, aa = max(a, 0.25).
  * AMPLITUDE alpha (the C switch cluster.hmf_alpha_mode, HaloModel's
    `hmf_alpha_mode`):
      0 = HMF_ALPHA_FIXED (default): alpha = 0.368, the Table-4 value at
          Delta = 200m, at every z: the convention of the DES cluster
          analyses (lighthouse halo.c f_tinker) and of the C cluster code
          by default (halo_cluster.c, private copy of halo.c's shape);
      1 = HMF_ALPHA_NORMALIZED: halo.c's own fnu (`tinker_alpha`), alpha
          re-derived at every aa from the peak-background consistency
          relation (1001.3162 Eq. 7)
              int_0^inf b(nu) f(nu; aa) dnu = 1,
          with b the Tinker bias below and the shape at the same aa
          (alpha = 0.3684 at z = 0, falling with z: alpha/0.368 = 0.967,
          0.951, 0.936, 0.923, 0.909 at z = 0.2 .. 0.6; 0.2520 for z >= 3).
    The shape is the same in both modes, so n_A and the counts scale with
    alpha while b_A and P^1h_A (ratios over the mass function) do not.
  * Bias: Tinker 2010 Eq. 6 with y = log10(200); A = 1 + 0.24 y exp(-(4/y)^4),
    a = 0.44 y - 0.88, B = 0.183, b = 1.5, C = 0.019 + 0.107 y +
    0.19 exp(-(4/y)^4), c = 2.4, delta_c = 1.686. No z dependence.
  * dln nu/dln M = -(1/2) dln sigma_cb^2(M,z)/dln M. The C FFTLog
    differentiates the Mellin kernel; here differentiate the window
    inside the independent Simpson integral.
  * Concentration: Bhattacharya et al. 2013 (1112.5479) full sample, 200m:
    c = 9.0 nu^-0.29 D_cb(M,z)^1.15,
    D_cb(M,z) = sigma_cb(M,z)/sigma_cb(M,0).
  * NFW transform truncated at r_200m (comoving,
    r_Delta = (3M/(4 pi 200 rho_m))^(1/3), rho_m TOTAL in both HMF modes),
    u(k|M) = [sin x (Si(xu)-Si(x)) - sin(c x)/xu + cos x (Ci(xu)-Ci(x))]/m(c),
    x = k r_Delta/c, xu = (1+c) x, m(c) = ln(1+c) - c/(1+c)
    (astro-ph/0206508 Eq. 81), evaluated with scipy's Si/Ci (halo.c uses
    an A&S f,g table instead).
"""

import numpy as np
from scipy.integrate import quad, simpson
from scipy.interpolate import CubicSpline, RectBivariateSpline
from scipy.special import sici

from ref_cosmology import RHO_CRIT

DELTA_C = 1.686
DELTA_HALO = 200.0
# amplitude modes of the Tinker f(nu) (module docstring; structs_cluster.h)
HMF_ALPHA_FIXED = 0
HMF_ALPHA_NORMALIZED = 1
TINKER_ALPHA_FIXED = 0.368          # 1001.3162 Table 4, Delta = 200m
_Y = np.log10(DELTA_HALO)
_EXPY = np.exp(-(4.0 / _Y) ** 4)
TINKER_BIAS = dict(A=1.0 + 0.24 * _Y * _EXPY, a=0.44 * _Y - 0.88, B=0.183, b=1.5,
                   C=0.019 + 0.107 * _Y + 0.19 * _EXPY, c=2.4)


def tinker_bias(nu):
    """Tinker et al. 2010 Eq. 6 linear halo bias at Delta = 200m."""
    p = TINKER_BIAS
    nu = np.asarray(nu, dtype=float)
    na = nu ** p["a"]
    return (1.0 - p["A"] * na / (na + DELTA_C ** p["a"])
            + p["B"] * nu ** p["b"] + p["C"] * nu ** p["c"])


def tinker_shape_params(aa):
    aa = np.asarray(aa, dtype=float)
    return (0.589 * aa ** -0.2, 0.864 * aa ** 0.01,
            -0.729 * aa ** 0.08, -0.243 * aa ** -0.27)


def tinker_f_shape(nu, aa):
    """f(nu)/alpha of Tinker 2010 Eq. 8 (alpha = 1)."""
    beta, gamma, phi, eta = tinker_shape_params(aa)
    return (1.0 + (beta * nu) ** (-2.0 * phi)) * nu ** (2.0 * eta) * np.exp(-0.5 * gamma * nu * nu)


def tinker_alpha_exact(aa):
    """alpha(aa) = 1/int_0^inf b(nu) f_shape(nu; aa) dnu (quad in ln nu)."""
    def integrand(s):
        nu = np.exp(s)
        return tinker_bias(nu) * tinker_f_shape(nu, aa) * nu
    # the integrand ~ nu^(1+2eta-2phi) at small nu and Gaussian at large nu
    pts = [-60.0, -20.0, -5.0, -1.0, 0.0, 1.0, 1.5, 2.0, 3.0, 5.0]
    tot = 0.0
    for lo, hi in zip(pts[:-1], pts[1:]):
        val, _ = quad(integrand, lo, hi, epsabs=0.0, epsrel=1e-13, limit=200)
        tot += val
    return 1.0 / tot


class TinkerAlpha:
    """alpha(aa) on aa in [0.25, 1]: a dense table of exact quads + cubic spline
    (interpolation error < 1e-10; see the tests)."""

    def __init__(self, n=301):
        self.aa = np.linspace(0.25, 1.0, n)
        self.alpha = np.array([tinker_alpha_exact(a) for a in self.aa])
        self._spl = CubicSpline(self.aa, self.alpha)

    def __call__(self, aa):
        return self._spl(np.clip(aa, 0.25, 1.0))


_TINKER_ALPHA = None


def tinker_alpha(aa):
    global _TINKER_ALPHA
    if _TINKER_ALPHA is None:
        _TINKER_ALPHA = TinkerAlpha()
    return _TINKER_ALPHA(aa)


def tinker_amplitude(aa, alpha_mode=HMF_ALPHA_FIXED):
    """alpha of Tinker Eq. 8 at aa (already floored at 0.25): 0.368
    (HMF_ALPHA_FIXED) or halo.c's alpha(aa) (HMF_ALPHA_NORMALIZED)."""
    aa = np.asarray(aa, dtype=float)
    if alpha_mode == HMF_ALPHA_FIXED:
        return np.full_like(aa, TINKER_ALPHA_FIXED)
    if alpha_mode == HMF_ALPHA_NORMALIZED:
        return tinker_alpha(aa)
    raise ValueError(f"hmf_alpha_mode = {alpha_mode} not supported")


def tinker_f(nu, z, alpha_mode=HMF_ALPHA_FIXED):
    """Tinker f(nu) at redshift z (aa = max(1/(1+z), 0.25)); amplitude per
    alpha_mode (module docstring)."""
    aa = np.maximum(1.0 / (1.0 + np.asarray(z, dtype=float)), 0.25)
    return tinker_amplitude(aa, alpha_mode) * tinker_f_shape(nu, aa)


def tophat_W_and_dW(x):
    """W(x) = 3 j1(x)/x and dW/dx, with small-x series."""
    x = np.asarray(x, dtype=float)
    small = x < 1e-2
    xs = np.where(small, 1.0, x)
    s, c = np.sin(xs), np.cos(xs)
    W = 3.0 * (s - xs * c) / xs**3
    dW = 3.0 * ((xs * xs - 3.0) * s + 3.0 * xs * c) / xs**4
    x2 = x * x
    W = np.where(small, 1.0 - x2 / 10.0 + x2 * x2 / 280.0, W)
    dW = np.where(small, -x / 5.0 + x * x2 / 70.0, dW)
    return W, dW


class Sigma2:
    """Independent Simpson integrals of sigma^2(M,z) and its mass slope.

    Integrate each supplied redshift row directly; massive neutrinos do
    not admit a single growth factor multiplying every halo mass.
    """

    def __init__(self, cosmo, rho, kind="lin", lnM_min=np.log(1e9), lnM_max=np.log(1e17),
                 n_M=1201, lnk_min=np.log(1e-6), lnk_max=np.log(1e4), n_k=20001):
        if n_k % 2 != 1:
            raise ValueError("Simpson variance integral needs an odd number of k nodes")
        self.rho = rho
        lnk = np.linspace(lnk_min, lnk_max, n_k)
        k = np.exp(lnk)
        redshifts = cosmo.z_pk
        pk = cosmo.P_lin(k[None, :], redshifts[:, None], kind)

        # The uniform Simpson weights integrate dlnk. Matrix products
        # apply the same top-hat window to every redshift's spectrum.
        weights = np.ones(n_k)
        weights[1:-1:2] = 4.0
        weights[2:-1:2] = 2.0
        weights *= (lnk[1]-lnk[0])/3.0
        weighted_power = pk*(k**3*weights/(2.0*np.pi**2))
        lnM = np.linspace(lnM_min, lnM_max, n_M)
        s2 = np.empty((redshifts.size, n_M))
        ds2 = np.empty_like(s2)
        for start in range(0, n_M, 50):
            mass_slice = slice(start, min(start+50, n_M))
            radius = (3.0*np.exp(lnM[mass_slice])/(4.0*np.pi*rho))**(1.0/3.0)
            x = radius[:, None]*k[None, :]
            window, derivative = tophat_W_and_dW(x)
            s2[:, mass_slice] = weighted_power @ (window*window).T
            ds2[:, mass_slice] = weighted_power @ (2.0*window*derivative*x).T
        self.lnM = lnM
        self.ln_s2 = np.log(s2)
        self.dlns2_dlnM = ds2/s2/3.0
        self._spl = RectBivariateSpline(redshifts, lnM, self.ln_s2)
        self._dspl = RectBivariateSpline(redshifts, lnM, self.dlns2_dlnM)

    def sigma(self, lnM, z=0.0):
        lnM, z = np.broadcast_arrays(lnM, z)
        return np.exp(0.5*self._spl.ev(z.ravel(), lnM.ravel())).reshape(lnM.shape)

    def dlns2(self, lnM, z=0.0):
        lnM, z = np.broadcast_arrays(lnM, z)
        return self._dspl.ev(z.ravel(), lnM.ravel()).reshape(lnM.shape)


class HaloModel:
    """Tinker HMF + bias, Bhattacharya c(M), NFW u(k|M) in halo.c conventions.

    hmf_matter: "cb" (production default) or "tot" (diagnostic comparison).
    hmf_alpha_mode: HMF_ALPHA_FIXED (0.368, DES / lighthouse / the C cluster
    code's default) or HMF_ALPHA_NORMALIZED (halo.c's alpha(z), Eq. 7).
    """

    def __init__(self, cosmo, hmf_matter="cb", hmf_alpha_mode=HMF_ALPHA_FIXED):
        if hmf_alpha_mode not in (HMF_ALPHA_FIXED, HMF_ALPHA_NORMALIZED):
            raise ValueError(f"hmf_alpha_mode = {hmf_alpha_mode} not supported")
        self.cosmo = cosmo
        self.hmf_matter = hmf_matter
        self.hmf_alpha_mode = hmf_alpha_mode
        self.rho_m = RHO_CRIT * cosmo.Omega_m                  # NFW, 1-halo (always total)
        if hmf_matter == "tot":
            self.rho_hmf = self.rho_m
            kind = "lin"
        elif hmf_matter == "cb":
            self.rho_hmf = RHO_CRIT * cosmo.Omega_cb
            kind = "cb"
        else:
            raise ValueError(hmf_matter)
        self.sig = Sigma2(cosmo, self.rho_hmf, kind=kind)

    # all functions broadcast lnM against z
    def nu(self, lnM, z):
        return DELTA_C / self.sig.sigma(lnM, z)

    def dndlnM(self, lnM, z, nu=None):
        """dn/dlnM in (h/Mpc)^3."""
        if nu is None:
            nu = self.nu(lnM, z)
        dlnnu = -0.5 * self.sig.dlns2(lnM, z)
        return (self.rho_hmf / np.exp(lnM)) * nu * tinker_f(nu, z, self.hmf_alpha_mode) * dlnnu

    def bias(self, lnM, z, nu=None):
        if nu is None:
            nu = self.nu(lnM, z)
        return tinker_bias(nu)

    def conc(self, lnM, z, nu=None):
        D = self.sig.sigma(lnM, z)/self.sig.sigma(lnM, 0.0)
        if nu is None:
            nu = self.nu(lnM, z)
        return 9.0 * nu ** -0.29 * D ** 1.15

    def r_delta(self, lnM):
        return (3.0 * np.exp(lnM) / (4.0 * np.pi * DELTA_HALO * self.rho_m)) ** (1.0 / 3.0)


def u_nfw(k, r_delta, c):
    """Truncated-NFW normalized Fourier transform (broadcasting arrays)."""
    x = k * r_delta / c
    xu = (1.0 + c) * x
    si_x, ci_x = sici(x)
    si_u, ci_u = sici(xu)
    mc = np.log1p(c) - c / (1.0 + c)
    return (np.sin(x) * (si_u - si_x) - np.sin(c * x) / xu
            + np.cos(x) * (ci_u - ci_x)) / mc
