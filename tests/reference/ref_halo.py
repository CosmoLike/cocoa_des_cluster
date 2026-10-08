"""Halo-model ingredients of the DES cluster reference: the halo mass
function, the linear halo bias, the variance sigma(M, z), the NFW
concentration and the NFW profile, with the conventions of cosmolike's
halo.c (cosmolike_core/cosmolike/halo.c) but an independent
implementation (no C code structure is copied).

ref_cluster weights these per-halo quantities with the probability that
a halo of mass M lands in a richness bin, which gives the number
density n_A, the bias b_A and the one-halo spectrum P^1h_A of each bin.
Units: M in M_sun/h (M_200m), k in h/Mpc, radii in comoving Mpc/h,
densities in (M_sun/h)/(Mpc/h)^3, dn/dlnM in (h/Mpc)^3.

Conventions mirrored from halo.c (checked by reading the source):

  * Halo definition M_200m: Delta = 200 times the mean matter density,
    rho_m = rho_crit * Omega_m with Omega_m the total matter density
    (halo.c: `rho_m = cosmology.rho_crit * cosmology.Omega_m`).
  * Peak height nu = delta_c/sigma_cb(M,z), delta_c = 1.686 (the linear
    collapse threshold of spherical collapse). sigma_cb^2(M, z) is the
    variance of the linear density of cold dark matter plus baryons
    (P_cb(k,z), integrated directly at each z) in a top-hat sphere of
    Lagrangian radius (3M/(4 pi rho_cb))^(1/3). HaloModel(hmf_matter=
    "tot") swaps in the total-matter spectrum and rho_m for diagnostic
    comparisons; the C code always uses cb.
  * Mass function dn/dlnM = (rho_cb/M) nu f(nu) dln nu/dln M with the
    Tinker et al. 2010 (1001.3162) multiplicity at Delta = 200m, Eqs. 8-12,
    Table 4: beta = 0.589 aa^-0.2, gamma = 0.864 aa^0.01,
    phi = -0.729 aa^0.08, eta = -0.243 aa^-0.27, aa = max(a, 0.25)
    (a = 1/(1+z); the fit's z evolution stops at z = 3).
  * Amplitude alpha (the C switch cluster.hmf_alpha_mode, HaloModel's
    `hmf_alpha_mode`):
      0 = HMF_ALPHA_FIXED (default): alpha = 0.368, the Table-4 value at
          Delta = 200m, at every z: the convention of the DES cluster
          analyses (f_tinker in theory/halo.c of the cosmolike_core
          branch cluster_chto, which the lighthouse code uses) and of the
          C cluster code by default (halo_cluster.c, private copy of
          halo.c's shape);
      1 = HMF_ALPHA_NORMALIZED: halo.c's own fnu (`tinker_alpha`), alpha
          re-derived at every aa from the peak-background consistency
          relation (1001.3162 Eq. 7)
              int_0^inf b(nu) f(nu; aa) dnu = 1,
          with b the Tinker bias below and the shape at the same aa
          (alpha = 0.3684 at z = 0, falling with z: alpha/0.368 = 0.967,
          0.951, 0.936, 0.923, 0.909 at z = 0.2, 0.3, 0.4, 0.5, 0.6;
          0.2520 for z >= 3; test_reference_cluster.py checks these
          values).
    The shape is the same in both modes, so n_A and the counts scale with
    alpha while b_A and P^1h_A (ratios over the mass function) do not.
  * Bias: Tinker 2010 Eq. 6 with y = log10(200); A = 1 + 0.24 y exp(-(4/y)^4),
    a = 0.44 y - 0.88, B = 0.183, b = 1.5, C = 0.019 + 0.107 y +
    0.19 exp(-(4/y)^4), c = 2.4, delta_c = 1.686. No z dependence.
  * dln nu/dln M = -(1/2) dln sigma_cb^2(M,z)/dln M. The C FFTLog
    differentiates the Mellin kernel; here the derivative of the top-hat
    window enters the same Simpson integral as sigma^2 (Sigma2).
  * Concentration: Bhattacharya et al. 2013 (1112.5479) full sample, 200m:
    c = 9.0 nu^-0.29 D_cb(M,z)^1.15,
    D_cb(M,z) = sigma_cb(M,z)/sigma_cb(M,0).
  * NFW transform truncated at r_200m (comoving,
    r_Delta = (3M/(4 pi 200 rho_m))^(1/3), rho_m total in both HMF modes),
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

# delta_c = (3/20)(12 pi)^(2/3) = 1.686: the linearly extrapolated
# density contrast at which a spherical perturbation collapses, the
# value the Tinker fits assume and halo.c uses
DELTA_C = 1.686
# halo overdensity relative to the mean matter density (M_200m)
DELTA_HALO = 200.0
# amplitude modes of the Tinker f(nu) (module docstring; structs_cluster.h)
HMF_ALPHA_FIXED = 0
HMF_ALPHA_NORMALIZED = 1
TINKER_ALPHA_FIXED = 0.368          # 1001.3162 Table 4, Delta = 200m
# Coefficients of the Tinker 2010 bias, Eq. 6 (Table 2 of 1001.3162):
# y = log10(Delta) and exp(-(4/y)^4) enter A and C (module docstring)
_Y = np.log10(DELTA_HALO)
_EXPY = np.exp(-(4.0 / _Y) ** 4)
TINKER_BIAS = dict(A=1.0 + 0.24 * _Y * _EXPY, a=0.44 * _Y - 0.88, B=0.183, b=1.5,
                   C=0.019 + 0.107 * _Y + 0.19 * _EXPY, c=2.4)


def tinker_bias(nu):
    """Return the Tinker et al. 2010 Eq. 6 linear halo bias at Delta = 200m.

    b(nu) = 1 - A nu^a/(nu^a + delta_c^a) + B nu^b + C nu^c, with the
    coefficients of TINKER_BIAS; the fit has no explicit z dependence (z
    enters through nu).

    Arguments:
      nu = peak height delta_c/sigma(M, z), >= 0, scalar or array.

    Returns:
      numpy array of b, dimensionless, the shape of nu.
    """
    p = TINKER_BIAS
    nu = np.asarray(nu, dtype=float)
    na = nu ** p["a"]
    return (1.0 - p["A"] * na / (na + DELTA_C ** p["a"])
            + p["B"] * nu ** p["b"] + p["C"] * nu ** p["c"])


def tinker_shape_params(aa):
    """Return the Tinker 2010 shape parameters (beta, gamma, phi, eta).

    Eqs. 9-12 of 1001.3162 with the Table 4 values at Delta = 200m:
    beta = 0.589 aa^-0.2, gamma = 0.864 aa^0.01, phi = -0.729 aa^0.08,
    eta = -0.243 aa^-0.27.

    Arguments:
      aa = scale factor already floored at 0.25 (aa = max(1/(1+z), 0.25)),
           scalar or array.

    Returns:
      tuple of four numpy arrays (beta, gamma, phi, eta), each the shape
      of aa.
    """
    aa = np.asarray(aa, dtype=float)
    return (0.589 * aa ** -0.2, 0.864 * aa ** 0.01,
            -0.729 * aa ** 0.08, -0.243 * aa ** -0.27)


def tinker_f_shape(nu, aa):
    """Return f(nu)/alpha of Tinker 2010 Eq. 8 (the multiplicity at alpha = 1).

    f(nu)/alpha = [1 + (beta nu)^(-2 phi)] nu^(2 eta) exp(-gamma nu^2/2),
    with the shape parameters of tinker_shape_params at aa.

    Arguments:
      nu = peak height, > 0, scalar or array.
      aa = scale factor floored at 0.25, broadcast against nu.

    Returns:
      numpy array of the broadcast shape of (nu, aa), dimensionless.
    """
    beta, gamma, phi, eta = tinker_shape_params(aa)
    return (1.0 + (beta * nu) ** (-2.0 * phi)) * nu ** (2.0 * eta) * np.exp(-0.5 * gamma * nu * nu)


def tinker_alpha_exact(aa):
    """alpha(aa) = 1/int_0^inf b(nu) f_shape(nu; aa) dnu (quad in ln nu).

    The consistency relation int b f dnu = 1 (1001.3162 Eq. 7) fixes the
    amplitude of f because alpha factors out of it. The integral runs in
    s = ln nu (dnu = nu ds) with scipy's adaptive quad on consecutive
    intervals between break points, each to a relative tolerance of
    1e-13.

    Arguments:
      aa = one scale factor, floored at 0.25 by the caller (a float: quad
           integrates one function at a time).

    Returns:
      alpha as a float (0.3684 at aa = 1).
    """
    def integrand(s):
        """Return b(nu) f_shape(nu; aa) nu at nu = e^s, the integrand of
        int ds after the change of variable (aa is read from the
        enclosing function).

        Arguments:
          s = ln nu, a float (quad passes one point at a time).

        Returns:
          the integrand as a float.
        """
        nu = np.exp(s)
        return tinker_bias(nu) * tinker_f_shape(nu, aa) * nu
    # The integrand ~ nu^(1+2eta-2phi) at small nu and Gaussian at large
    # nu. The break points span s = -60 .. 5 (nu = 9e-27 .. 148); outside
    # that range the integrand is negligible (nu^1.6 or smaller at the
    # low end, exp(-gamma nu^2/2) at the high end).
    pts = [-60.0, -20.0, -5.0, -1.0, 0.0, 1.0, 1.5, 2.0, 3.0, 5.0]
    tot = 0.0
    # zip pairs every break point with the next one: (lo, hi) runs over
    # the consecutive intervals
    for lo, hi in zip(pts[:-1], pts[1:]):
        val, _ = quad(integrand, lo, hi, epsabs=0.0, epsrel=1e-13, limit=200)
        tot += val
    return 1.0 / tot


class TinkerAlpha:
    """alpha(aa) on aa in [0.25, 1]: a dense table of exact quads + cubic spline.

    test_tinker_alpha_normalization (test_reference_cluster.py) checks the
    spline against tinker_alpha_exact between the nodes to a relative
    error of 1e-9. Calling an instance t as t(aa) reads the spline
    (__call__).
    """

    def __init__(self, n=301):
        """Tabulate alpha with tinker_alpha_exact on n nodes and spline it.

        Arguments:
          n = number of nodes uniform in aa on [0.25, 1] (default 301, a
              step of 0.0025; one quad integration per node).
        """
        self.aa = np.linspace(0.25, 1.0, n)
        # one exact alpha per node (a list comprehension over the nodes)
        self.alpha = np.array([tinker_alpha_exact(a) for a in self.aa])
        self._spl = CubicSpline(self.aa, self.alpha)

    def __call__(self, aa):
        """Return the tabulated alpha at aa.

        Arguments:
          aa = scale factor, scalar or array; clipped to [0.25, 1] first
               (z >= 3 reads the aa = 0.25 value).

        Returns:
          numpy array of alpha, the shape of aa.
        """
        return self._spl(np.clip(aa, 0.25, 1.0))


# Module-global cache of the TinkerAlpha table: None until the first call
# of tinker_alpha builds it (301 quad integrations), then reused by every
# later call in the same Python process.
_TINKER_ALPHA = None


def tinker_alpha(aa):
    """Return halo.c's alpha(aa) of HMF_ALPHA_NORMALIZED from the cached table.

    Arguments:
      aa = scale factor, scalar or array (clipped to [0.25, 1] by
           TinkerAlpha).

    Returns:
      numpy array of alpha, the shape of aa.

    Side effects: the first call builds the module-global table
    _TINKER_ALPHA.
    """
    # `global` lets this function rebind the module-level name instead of
    # creating a local variable of the same name
    global _TINKER_ALPHA
    if _TINKER_ALPHA is None:
        _TINKER_ALPHA = TinkerAlpha()
    return _TINKER_ALPHA(aa)


def tinker_amplitude(aa, alpha_mode=HMF_ALPHA_FIXED):
    """alpha of Tinker Eq. 8 at aa (already floored at 0.25): 0.368
    (HMF_ALPHA_FIXED) or halo.c's alpha(aa) (HMF_ALPHA_NORMALIZED).

    Arguments:
      aa = scale factor floored at 0.25, scalar or array.
      alpha_mode = HMF_ALPHA_FIXED (0, default) or HMF_ALPHA_NORMALIZED
                   (1).

    Returns:
      numpy array of alpha, the shape of aa (np.full_like fills an array
      of aa's shape with 0.368 in the fixed mode).

    Raises:
      ValueError for any other alpha_mode.
    """
    aa = np.asarray(aa, dtype=float)
    if alpha_mode == HMF_ALPHA_FIXED:
        return np.full_like(aa, TINKER_ALPHA_FIXED)
    if alpha_mode == HMF_ALPHA_NORMALIZED:
        return tinker_alpha(aa)
    raise ValueError(f"hmf_alpha_mode = {alpha_mode} not supported")


def tinker_f(nu, z, alpha_mode=HMF_ALPHA_FIXED):
    """Tinker f(nu) at redshift z (aa = max(1/(1+z), 0.25)); amplitude per
    alpha_mode (module docstring).

    Arguments:
      nu = peak height, > 0, scalar or array.
      z = redshift, broadcast against nu.
      alpha_mode = HMF_ALPHA_FIXED (0, default) or HMF_ALPHA_NORMALIZED
                   (1).

    Returns:
      numpy array of f(nu) = alpha f_shape(nu), dimensionless, the
      broadcast shape of (nu, z).
    """
    aa = np.maximum(1.0 / (1.0 + np.asarray(z, dtype=float)), 0.25)
    return tinker_amplitude(aa, alpha_mode) * tinker_f_shape(nu, aa)


def tophat_W_and_dW(x):
    """W(x) = 3 j1(x)/x and dW/dx, with small-x series.

    W(x) = 3 (sin x - x cos x)/x^3 is the Fourier transform of a top-hat
    sphere at x = k R, and dW/dx = 3 [(x^2 - 3) sin x + 3 x cos x]/x^4.
    Below x = 1e-2 the closed forms lose digits to cancellation, so the
    Taylor series W = 1 - x^2/10 + x^4/280, dW/dx = -x/5 + x^3/70 replace
    them.

    Arguments:
      x = k R, >= 0, scalar or array.

    Returns:
      (W, dW), two numpy arrays of the shape of x.
    """
    x = np.asarray(x, dtype=float)
    small = x < 1e-2
    # xs puts 1 in place of the small x, so the closed forms never divide
    # by a number near 0; np.where then picks the series at those points
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

    sigma^2(M, z) = int dln k k^3 P(k, z)/(2 pi^2) W^2(k R),
    R = (3M/(4 pi rho))^(1/3), with W the top-hat window (tophat_W_and_dW).
    The table rows are the z nodes of the Cosmology (z_pk), the columns
    n_M masses uniform in ln M; ln sigma^2 and dln sigma^2/dln M are read
    through 2-D cubic splines in (z, ln M), which clamp a point outside
    the table to its edge.
    """

    def __init__(self, cosmo, rho, kind="lin", lnM_min=np.log(1e9), lnM_max=np.log(1e17),
                 n_M=1201, lnk_min=np.log(1e-6), lnk_max=np.log(1e4), n_k=20001):
        """Integrate sigma^2 and its slope on the (z_pk, ln M) grid.

        Simpson's rule in ln k on n_k uniform nodes; the k range reaches
        beyond CAMB's, where Cosmology.lnP continues P as a power law.
        dsigma^2/dln R = int dln k k^3 P/(2 pi^2) 2 W (dW/dx) x, and
        dln R = dln M/3 turns it into the mass slope.

        Arguments:
          cosmo = ref_cosmology.Cosmology; its z_pk are the table rows.
          rho = comoving mean density that converts M to the Lagrangian
                radius, in (M_sun/h)/(Mpc/h)^3 (HaloModel passes rho_cb
                or rho_m).
          kind = spectrum of Cosmology.P_lin: "lin" (linear total matter,
                 default) or "cb" (linear cold dark matter plus baryons).
          lnM_min, lnM_max = ln of the mass range in M_sun/h (default
                             1e9 to 1e17; the cluster integrals use
                             1e12 to 1e16).
          n_M = number of mass nodes uniform in ln M (default 1201:
                dln M = 0.0154).
          lnk_min, lnk_max = ln of the k range in h/Mpc (default 1e-6 to
                             1e4).
          n_k = number of k nodes uniform in ln k (default 20001); must
                be odd, because Simpson's rule pairs the intervals.

        Raises:
          ValueError when n_k is even.
        """
        if n_k % 2 != 1:
            raise ValueError("Simpson variance integral needs an odd number of k nodes")
        self.rho = rho
        lnk = np.linspace(lnk_min, lnk_max, n_k)
        k = np.exp(lnk)
        redshifts = cosmo.z_pk
        # P on the (z node, k node) grid, shape (n_z, n_k): k[None, :] is
        # a (1, n_k) row and redshifts[:, None] an (n_z, 1) column, which
        # numpy broadcasting expands to the full grid
        pk = cosmo.P_lin(k[None, :], redshifts[:, None], kind)

        # The uniform Simpson weights integrate dlnk. Matrix products
        # apply the same top-hat window to every redshift's spectrum.
        # Weights: 1, 4, 2, 4, ..., 2, 4, 1 times dln k/3 (weights[1:-1:2]
        # is the start:stop:step slice of the odd interior nodes,
        # weights[2:-1:2] that of the even ones); weighted_power[z, k] =
        # P k^3 w_k/(2 pi^2) has shape (n_z, n_k).
        weights = np.ones(n_k)
        weights[1:-1:2] = 4.0
        weights[2:-1:2] = 2.0
        weights *= (lnk[1]-lnk[0])/3.0
        weighted_power = pk*(k**3*weights/(2.0*np.pi**2))
        lnM = np.linspace(lnM_min, lnM_max, n_M)
        s2 = np.empty((redshifts.size, n_M))
        ds2 = np.empty_like(s2)
        # Masses in chunks of 50, which bounds the (50, n_k) window arrays
        # (8 MB each at n_k = 20001). x = R k has shape (50, n_k), and
        # weighted_power @ (W^2).T sums over k for every z node and the 50
        # masses at once, an (n_z, 50) block.
        for start in range(0, n_M, 50):
            mass_slice = slice(start, min(start+50, n_M))
            radius = (3.0*np.exp(lnM[mass_slice])/(4.0*np.pi*rho))**(1.0/3.0)
            x = radius[:, None]*k[None, :]
            window, derivative = tophat_W_and_dW(x)
            s2[:, mass_slice] = weighted_power @ (window*window).T
            ds2[:, mass_slice] = weighted_power @ (2.0*window*derivative*x).T
        self.lnM = lnM
        self.ln_s2 = np.log(s2)
        # ds2 holds dsigma^2/dln R, and R is proportional to M^(1/3), so
        # dln R = dln M/3
        self.dlns2_dlnM = ds2/s2/3.0
        self._spl = RectBivariateSpline(redshifts, lnM, self.ln_s2)
        self._dspl = RectBivariateSpline(redshifts, lnM, self.dlns2_dlnM)

    def sigma(self, lnM, z=0.0):
        """Return sigma(M, z), the rms of the linear density contrast
        smoothed over a top-hat sphere of mass M.

        Arguments:
          lnM = ln(M/(M_sun/h)), scalar or array.
          z = redshift (default 0), broadcast against lnM.

        Returns:
          numpy array of sigma, dimensionless, the broadcast shape.
        """
        lnM, z = np.broadcast_arrays(lnM, z)
        return np.exp(0.5*self._spl.ev(z.ravel(), lnM.ravel())).reshape(lnM.shape)

    def dlns2(self, lnM, z=0.0):
        """Return the mass slope dln sigma^2/dln M.

        Arguments:
          lnM = ln(M/(M_sun/h)), scalar or array.
          z = redshift (default 0), broadcast against lnM.

        Returns:
          numpy array of the slope (negative), the broadcast shape.
        """
        lnM, z = np.broadcast_arrays(lnM, z)
        return self._dspl.ev(z.ravel(), lnM.ravel()).reshape(lnM.shape)


class HaloModel:
    """Tinker HMF + bias, Bhattacharya c(M), NFW u(k|M) in halo.c conventions.

    hmf_matter: "cb" (production default) or "tot" (diagnostic comparison).
    hmf_alpha_mode: HMF_ALPHA_FIXED (0.368, DES / lighthouse / the C cluster
    code's default) or HMF_ALPHA_NORMALIZED (halo.c's alpha(z), Eq. 7).
    Units: module docstring.
    """

    def __init__(self, cosmo, hmf_matter="cb", hmf_alpha_mode=HMF_ALPHA_FIXED):
        """Set the densities and build the variance table (Sigma2).

        Arguments:
          cosmo = ref_cosmology.Cosmology.
          hmf_matter = "cb" (default: the spectrum of cold dark matter
                       plus baryons and rho_cb set nu and the mass
                       function, as in the C code) or "tot" (total-matter
                       spectrum and rho_m, a diagnostic comparison).
          hmf_alpha_mode = HMF_ALPHA_FIXED (0, default) or
                           HMF_ALPHA_NORMALIZED (1): the Tinker amplitude
                           (module docstring).

        Raises:
          ValueError for any other hmf_matter or hmf_alpha_mode.
        """
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

    # Every method below takes lnM = ln(M/(M_sun/h)) and z as arrays that
    # numpy broadcasts against each other: lnM of shape (nM, 1) with z of
    # shape (1, nz) gives results of shape (nM, nz).
    def nu(self, lnM, z):
        """Return the peak height nu = delta_c/sigma(M, z).

        Arguments:
          lnM = ln(M/(M_sun/h)).
          z = redshift, broadcast against lnM.

        Returns:
          numpy array of nu, the broadcast shape.
        """
        return DELTA_C / self.sig.sigma(lnM, z)

    def dndlnM(self, lnM, z, nu=None):
        """dn/dlnM in (h/Mpc)^3.

        The comoving halo mass function per unit ln M,
        dn/dlnM = (rho_hmf/M) nu f(nu) dln nu/dln M with
        dln nu/dln M = -(1/2) dln sigma^2/dln M and the Tinker f of
        hmf_alpha_mode.

        Arguments:
          lnM = ln(M/(M_sun/h)).
          z = redshift, broadcast against lnM.
          nu = peak heights at (lnM, z), optional (None: computed here;
               ref_cluster passes them to save a spline read).

        Returns:
          numpy array of dn/dlnM, the broadcast shape.
        """
        if nu is None:
            nu = self.nu(lnM, z)
        dlnnu = -0.5 * self.sig.dlns2(lnM, z)
        return (self.rho_hmf / np.exp(lnM)) * nu * tinker_f(nu, z, self.hmf_alpha_mode) * dlnnu

    def bias(self, lnM, z, nu=None):
        """Return the Tinker linear halo bias b(nu) of halos of mass M at z.

        Arguments:
          lnM = ln(M/(M_sun/h)); not read when nu is given.
          z = redshift; not read when nu is given.
          nu = peak heights, optional (None: computed from lnM and z).

        Returns:
          numpy array of b, dimensionless.
        """
        if nu is None:
            nu = self.nu(lnM, z)
        return tinker_bias(nu)

    def conc(self, lnM, z, nu=None):
        """Return the Bhattacharya et al. 2013 concentration c(M, z).

        c = 9.0 nu^-0.29 D^1.15 with D = sigma(M, z)/sigma(M, 0), the
        growth factor of sigma at fixed mass (module docstring).

        Arguments:
          lnM = ln(M/(M_sun/h)).
          z = redshift, broadcast against lnM.
          nu = peak heights at (lnM, z), optional (None: computed here).

        Returns:
          numpy array of c = r_200m/r_s, dimensionless.
        """
        D = self.sig.sigma(lnM, z)/self.sig.sigma(lnM, 0.0)
        if nu is None:
            nu = self.nu(lnM, z)
        return 9.0 * nu ** -0.29 * D ** 1.15

    def r_delta(self, lnM):
        """Return the comoving halo radius r_200m in Mpc/h.

        M = (4 pi/3) r^3 200 rho_m, with the total-matter rho_m in both
        hmf_matter modes.

        Arguments:
          lnM = ln(M/(M_sun/h)), scalar or array.

        Returns:
          numpy array of r_200m, the shape of lnM.
        """
        return (3.0 * np.exp(lnM) / (4.0 * np.pi * DELTA_HALO * self.rho_m)) ** (1.0 / 3.0)


def u_nfw(k, r_delta, c):
    """Truncated-NFW normalized Fourier transform (broadcasting arrays).

    u(k|M) of the module docstring (astro-ph/0206508 Eq. 81), with the
    sine and cosine integrals Si, Ci of scipy.special.sici; u tends to 1
    as k tends to 0.

    Arguments:
      k = comoving wavenumber in h/Mpc.
      r_delta = comoving truncation radius r_200m in Mpc/h.
      c = concentration r_200m/r_s, > 0.
      (The three arrays are broadcast against each other.)

    Returns:
      numpy array of u, dimensionless, the broadcast shape.
    """
    x = k * r_delta / c
    xu = (1.0 + c) * x
    si_x, ci_x = sici(x)
    si_u, ci_u = sici(xu)
    mc = np.log1p(c) - c / (1.0 + c)
    return (np.sin(x) * (si_u - si_x) - np.sin(c * x) / xu
            + np.cos(x) * (ci_u - ci_x)) / mc
