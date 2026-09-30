"""Halo-model ingredients of the DES cluster reference, with the
CONVENTIONS of cosmolike's halo.c (NEW library) but an independent
implementation (no C code structure is ported).

Conventions mirrored from halo.c (checked by reading the source):

  * Halo definition M_200m: Delta = 200 times the MEAN matter density,
    rho_m = rho_crit * Omega_m with Omega_m the TOTAL matter density
    (halo.c: `rho_m = cosmology.rho_crit * cosmology.Omega_m`).
  * Peak height nu = delta_c/(sigma(M) D(z)), delta_c = 1.686, sigma(M) the
    z = 0 rms of the linear density field in a top hat of Lagrangian
    radius R = (3M/(4 pi rho_m))^(1/3), from `p_lin`, which the cocoa
    interface fills with CAMB's ("delta_tot","delta_tot") spectrum.
    => by default the HMF uses the TOTAL-matter P_lin and rho_m (TOTAL),
    NOT the CDM+baryon convention of the DES Y1 PRL / lighthouse. The
    `hmf_matter="cb"` switch gives the lighthouse/Y1 convention
    (P_cb from CAMB "delta_nonu" and rho_cb = rho_crit (Omega_m -
    Omega_nu) in R(M) and in dn/dlnM).
  * D(z) scale independent: the cocoa growth (see ref_cosmology).
  * Mass function dn/dlnM = (rho_m/M) nu f(nu) dln nu/dln M with the
    Tinker et al. 2010 (1001.3162) multiplicity at Delta = 200m, Eqs. 8-12,
    Table 4: beta = 0.589 aa^-0.2, gamma = 0.864 aa^0.01,
    phi = -0.729 aa^0.08, eta = -0.243 aa^-0.27, aa = max(a, 0.25).
  * NORMALIZATION (halo.c `tinker_alpha`): alpha is NOT the fixed Table-4
    value 0.368 used by lighthouse; it is re-derived at every aa from the
    peak-background consistency relation (1001.3162 Eq. 7)
        int_0^inf b(nu) f(nu; aa) dnu = 1,
    with b the Tinker bias below and the shape evaluated at the same aa.
    (alpha = 0.3684 at z = 0, falling with z; 0.2520 for z >= 3.)
  * Bias: Tinker 2010 Eq. 6 with y = log10(200); A = 1 + 0.24 y exp(-(4/y)^4),
    a = 0.44 y - 0.88, B = 0.183, b = 1.5, C = 0.019 + 0.107 y +
    0.19 exp(-(4/y)^4), c = 2.4, delta_c = 1.686. No z dependence.
  * dln nu/dln M = -(1/2) dln sigma^2/dln M. halo.c takes a symmetric finite
    difference (+-0.05 in ln M); here it is the exact derivative of the
    sigma^2 integral.
  * Concentration: Bhattacharya et al. 2013 (1112.5479) full sample, 200m:
    c = 9.0 nu^-0.29 D(z)^1.15 with nu = 1.686/(sigma(M) D(z)).
  * NFW transform truncated at r_200m (comoving,
    r_Delta = (3M/(4 pi 200 rho_m))^(1/3), rho_m TOTAL in both HMF modes),
    u(k|M) = [sin x (Si(xu)-Si(x)) - sin(c x)/xu + cos x (Ci(xu)-Ci(x))]/m(c),
    x = k r_Delta/c, xu = (1+c) x, m(c) = ln(1+c) - c/(1+c)
    (astro-ph/0206508 Eq. 81), evaluated with scipy's Si/Ci (halo.c uses
    an A&S f,g table instead).
"""

import numpy as np
from scipy.integrate import quad, simpson
from scipy.interpolate import CubicSpline
from scipy.special import sici

from ref_cosmology import RHO_CRIT

DELTA_C = 1.686
DELTA_HALO = 200.0
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


def tinker_f(nu, z):
    """Tinker f(nu) at redshift z (aa = max(1/(1+z), 0.25)), halo.c normalization."""
    aa = np.maximum(1.0 / (1.0 + np.asarray(z, dtype=float)), 0.25)
    return tinker_alpha(aa) * tinker_f_shape(nu, aa)


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
    """sigma^2(M) at z = 0 and dln sigma^2/dln M, exact integrals on a dense
    ln M grid + cubic spline (relative error < 1e-8)."""

    def __init__(self, cosmo, rho, kind="lin", lnM_min=np.log(1e9), lnM_max=np.log(1e17),
                 n_M=1201, lnk_min=np.log(1e-6), lnk_max=np.log(1e4), n_k=20001):
        self.rho = rho
        lnk = np.linspace(lnk_min, lnk_max, n_k)
        k = np.exp(lnk)
        pk = cosmo.P_lin(k, np.zeros_like(k), kind)
        k3p = k**3 * pk / (2.0 * np.pi**2)
        lnM = np.linspace(lnM_min, lnM_max, n_M)
        s2 = np.empty(n_M)
        ds2 = np.empty(n_M)
        for sl in np.array_split(np.arange(n_M), max(1, n_M // 50)):
            R = (3.0 * np.exp(lnM[sl]) / (4.0 * np.pi * rho)) ** (1.0 / 3.0)
            x = np.outer(R, k)
            W, dW = tophat_W_and_dW(x)
            s2[sl] = simpson(k3p * W * W, x=lnk, axis=1)
            ds2[sl] = simpson(k3p * 2.0 * W * dW * x, x=lnk, axis=1)   # d sigma^2/d ln R
        self.lnM = lnM
        self.ln_s2 = np.log(s2)
        self.dlns2_dlnM = ds2 / s2 / 3.0
        self._spl = CubicSpline(lnM, self.ln_s2)
        self._dspl = CubicSpline(lnM, self.dlns2_dlnM)

    def sigma(self, lnM):
        return np.exp(0.5 * self._spl(lnM))

    def dlns2(self, lnM):
        return self._dspl(lnM)


class HaloModel:
    """Tinker HMF + bias, Bhattacharya c(M), NFW u(k|M) in halo.c conventions.

    hmf_matter: "tot" (halo.c, default) or "cb" (lighthouse / Y1 PRL).
    """

    def __init__(self, cosmo, hmf_matter="tot"):
        self.cosmo = cosmo
        self.hmf_matter = hmf_matter
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
        return DELTA_C / (self.sig.sigma(lnM) * self.cosmo.growth(z))

    def dndlnM(self, lnM, z, nu=None):
        """dn/dlnM in (h/Mpc)^3."""
        if nu is None:
            nu = self.nu(lnM, z)
        dlnnu = -0.5 * self.sig.dlns2(lnM)
        return (self.rho_hmf / np.exp(lnM)) * nu * tinker_f(nu, z) * dlnnu

    def bias(self, lnM, z, nu=None):
        if nu is None:
            nu = self.nu(lnM, z)
        return tinker_bias(nu)

    def conc(self, lnM, z, nu=None):
        D = self.cosmo.growth(z)
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
