"""Self-tests of the DES cluster Python reference (reference_cluster.py).

Run from Cocoa/ (cocoa environment, start_cocoa.sh sourced):

    python -m pytest ./projects/des_cluster/tests/reference -s

One CAMB run is shared by the whole module (about 1-2 minutes in total).
Convergence tests compare the default quadratures with finer ones; the
counts are compared with the DES Y3 catalog totals as a printout only.

The reference itself is checked against independent evaluations: scipy
quad integrals, scipy and mpmath Legendre functions, closed forms of
power laws, and finer quadratures. pytest collects every function whose
name starts with test_; the -s option shows their printouts.
"""

import os
import sys

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import erf, eval_legendre, lpmv

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from reference_cluster import ClusterReference, FIDUCIAL           # noqa: E402
from ref_halo import (tinker_alpha, tinker_alpha_exact, tinker_bias,  # noqa: E402
                      tinker_f_shape, HMF_ALPHA_FIXED, HMF_ALPHA_NORMALIZED,
                      TINKER_ALPHA_FIXED)
from ref_projection import (legendre_table, projection_kernels,   # noqa: E402
                            theta_edges, y_transform_matrices, selection_factor,
                            cl_on_integers)
from ref_covariance import band_matrices                          # noqa: E402


@pytest.fixture(scope="module")
def ref():
    """The reference at the Table I fiducial with its real-space statistics.

    A pytest fixture: a test that names ref as an argument receives this
    object, and scope="module" builds it once for all the tests of this
    file (one CAMB run).

    Returns:
      a ClusterReference with real_space() already computed.
    """
    r = ClusterReference()
    r.real_space()
    return r


def scaled_diff(a, b, axis=-1):
    """max |a - b| / max |b| along `axis` (per spectrum), maximized over the rest.

    Arguments:
      a, b = arrays of one shape; b is the reference.
      axis = the axis along which one spectrum or statistic runs.

    Returns:
      a float.
    """
    return float(np.max(np.max(np.abs(a - b), axis=axis) / np.max(np.abs(b), axis=axis)))


# ----------------------------------------------------------------------
# halo model conventions
# ----------------------------------------------------------------------
def test_tinker_alpha_normalization():
    """The normalized Tinker amplitude alpha(a) of halo.c and its consistency relation.

    alpha(a) is fixed by int b(nu) f(nu) dnu = 1 (matter is unbiased with
    respect to itself). Checked: the values 0.3684 at z = 0 and 0.2520 at
    z >= 3 (a = 0.25) quoted in halo.c, the relation itself with an
    independent scipy quad over panels in nu at four scale factors (to
    1e-8), and the tabulated alpha against its exact solution (to 1e-9).
    """
    # the HMF_ALPHA_NORMALIZED amplitude (halo.c's fnu):
    # halo.c header: alpha = 0.3684 at z = 0, 0.2520 for z >= 3
    assert abs(tinker_alpha(1.0) - 0.3684) < 1e-4
    assert abs(tinker_alpha(0.25) - 0.2520) < 1e-4
    # consistency relation int b f dnu = 1, with an independent quad in nu
    for aa in (1.0, 1 / 1.3, 1 / 1.6, 0.4):
        pts = [0.0, 1e-3, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 30.0]
        val = sum(quad(lambda nu: tinker_bias(nu) * tinker_alpha(aa) * tinker_f_shape(nu, aa),
                       a, b, limit=200, epsabs=1e-14, epsrel=1e-10)[0]
                  for a, b in zip(pts[:-1], pts[1:]))
        assert abs(val - 1.0) < 1e-8, (aa, val)
    aa = np.linspace(0.26, 0.99, 7) + 0.003
    exact = np.array([tinker_alpha_exact(a) for a in aa])
    assert np.max(np.abs(tinker_alpha(aa) / exact - 1)) < 1e-9


def test_hmf_alpha_modes(ref):
    """Both Tinker amplitude modes share the mass-function shape.

    Switching the mode changes dn/dlnM by alpha(a)/0.368 at every mass
    (a capped at 0.25, z = 3), and that ratio matches the values quoted in
    structs_cluster.h at z = 0.2 ... 0.6 to 5e-4. The test switches the
    mode on the shared fixture object and restores it in the finally block,
    which runs even when an assertion fails.

    Arguments:
      ref = the module fixture.
    """
    # default = HMF_ALPHA_FIXED (alpha = 0.368 at every z, DES / lighthouse)
    assert ref.settings["hmf_alpha_mode"] == HMF_ALPHA_FIXED
    assert ref.halo.hmf_alpha_mode == HMF_ALPHA_FIXED
    # the two modes share the shape: dn/dlnM(mode 1)/dn/dlnM(mode 0) =
    # alpha(aa)/0.368 at every mass, with aa = max(a, 0.25)
    lnM = np.log(np.array([1e12, 1e13, 1e14, 1e15, 5e15]))[:, None]
    z = np.array([0.0, 0.2, 0.45, 0.65, 3.5])[None, :]
    dn0 = ref.halo.dndlnM(lnM, z)
    ref.halo.hmf_alpha_mode = HMF_ALPHA_NORMALIZED
    try:
        dn1 = ref.halo.dndlnM(lnM, z)
    finally:
        ref.halo.hmf_alpha_mode = HMF_ALPHA_FIXED
    aa = np.maximum(1.0 / (1.0 + z), 0.25)
    expect = np.broadcast_to(tinker_alpha(aa) / TINKER_ALPHA_FIXED, dn0.shape)
    assert np.max(np.abs(dn1 / dn0 / expect - 1)) < 1e-13
    # the ratios quoted in structs_cluster.h (and the fall with z)
    zq = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
    ratio = tinker_alpha(1.0 / (1.0 + zq)) / TINKER_ALPHA_FIXED
    assert np.max(np.abs(ratio - np.array([0.967, 0.951, 0.936, 0.923, 0.909]))) < 5e-4
    print("\n  alpha(z)/0.368 at z = 0.2 .. 0.6:", np.array2string(ratio, precision=4))


def test_sigma2_table_and_slope(ref):
    """The sigma^2(M) table and its slope against direct integration.

    sigma^2(M) = int dlnk k^3 P_cb(k)/(2 pi^2) W(kR)^2 at z = 0, with the
    top-hat window W(x) = 3 (sin x - x cos x)/x^3 and R = (3M/(4 pi rho))^(1/3)
    for the cold dark matter + baryon spectrum, integrated with scipy quad
    over 40 panels in ln k between 1e-6 and 1e4 h/Mpc (to 2e-6), and
    dln sigma^2/dlnM against a central finite difference (to 1e-5).

    Arguments:
      ref = the module fixture.
    """
    sig = ref.halo.sig
    cosmo = ref.cosmo
    for M in (3.3e12, 2.2e14, 7.0e15):
        R = (3 * M / (4 * np.pi * sig.rho)) ** (1 / 3)

        def f(lnk):
            """Integrand of sigma^2 in ln k: k^3 P_cb(k)/(2 pi^2) W(kR)^2.

            Arguments:
              lnk = ln of the wavenumber in h/Mpc, a float.

            Returns:
              a float.
            """
            k = np.exp(lnk)
            x = k * R
            W = 3 * (np.sin(x) - x * np.cos(x)) / x**3
            return k**3 * cosmo.P_lin(np.array([k]), np.array([0.0]), kind="cb")[0] * W * W / (2 * np.pi**2)
        s2 = sum(quad(f, a, b, limit=400, epsrel=1e-10)[0]
                 for a, b in zip(np.linspace(np.log(1e-6), np.log(1e4), 41)[:-1],
                                 np.linspace(np.log(1e-6), np.log(1e4), 41)[1:]))
        assert abs(sig.sigma(np.log(M)) ** 2 / s2 - 1) < 2e-6, M
    lnM = np.log(np.array([1e12, 1e13, 1e14, 1e15, 1e16]))
    h = 1e-3
    fd = (np.log(sig.sigma(lnM + h) ** 2) - np.log(sig.sigma(lnM - h) ** 2)) / (2 * h)
    assert np.max(np.abs(sig.dlns2(lnM) / fd - 1)) < 1e-5


# ----------------------------------------------------------------------
# MOR and mass integrals
# ----------------------------------------------------------------------
def test_p_lambda_erf_vs_quad(ref):
    """P(richness bin | M, z) from the error function against direct quadrature.

    The lognormal richness density integrated over each richness bin with
    scipy quad must equal the closed erf form to 1e-10 at 12 random (M, z)
    (fixed seed). The MOR variance adds its Poisson term only when the
    mean <ln lambda> is positive: at M = 1e11 M_sun/h (mean below zero) the
    scatter equals sigma_int exactly.

    Arguments:
      ref = the module fixture.
    """
    cl = ref.cluster
    rng = np.random.default_rng(1)
    for _ in range(12):
        lnM = np.log(10 ** rng.uniform(13.0, 15.5))
        z = rng.uniform(0.15, 0.7)
        mu, s = cl.mor_mean_sigma(lnM, z)
        P = cl.p_lambda_bins(np.array([lnM]), np.array([z]))[:, 0]
        for A in range(cl.nA):
            l1, l2 = cl.lam_edges[A], cl.lam_edges[A + 1]
            num, _ = quad(lambda lam: np.exp(-0.5 * ((np.log(lam) - mu) / s) ** 2)
                          / (np.sqrt(2 * np.pi) * s * lam), l1, l2, epsabs=1e-14, epsrel=1e-12,
                          limit=200)
            assert abs(P[A] - num) < 1e-10
    # sigma: Poisson term only for mu > 0
    mu, s = cl.mor_mean_sigma(np.log(1e11), 0.3)
    assert mu < 0 and abs(s - FIDUCIAL["sigma_int"]) < 1e-15


def test_mass_integral_convergence(ref):
    """Finer mass quadrature leaves n_A, b_A and P^1h unchanged.

    Panels of 0.1 in ln M with 12 nodes against the default 0.25 with 8:
    n_A and b_A agree to 1e-8, the one-halo power to 5e-5.

    Arguments:
      ref = the module fixture (its CAMB run is reused).
    """
    z = np.linspace(0.12, 0.75, 43)
    fine = ClusterReference(settings=dict(mass_panel_width=0.1, mass_order=12), cosmo=ref.cosmo)
    n0, b0 = ref.nA_bA(z)
    n1, b1 = fine.nA_bA(z)
    dn, db = np.max(np.abs(n1 / n0 - 1)), np.max(np.abs(b1 / b0 - 1))
    k = np.logspace(-3, 2, 60)
    dp = max(np.max(np.abs(fine.p1h(k, zz) / ref.p1h(k, zz) - 1)) for zz in (0.3, 0.6))
    print(f"\n  mass quadrature (0.25 -> 0.1 panels): n_A {dn:.1e}, b_A {db:.1e}, P1h {dp:.1e}")
    assert dn < 1e-8 and db < 1e-8 and dp < 5e-5


def test_mass_range(ref):
    """The default mass range [1e12, 1e16] M_sun/h holds all the counts.

    A wider and a narrower range change the counts by less than 1e-6
    (n_A and b_A changes are printed).

    Arguments:
      ref = the module fixture (its CAMB run is reused).
    """
    N0 = ref.counts()
    z = np.linspace(0.12, 0.75, 22)
    n0, b0 = ref.nA_bA(z)
    for lo, hi in ((1e11, 10**16.5), (1e12, 10**15.9)):
        r = ClusterReference(settings=dict(lnM_min=np.log(lo), lnM_max=np.log(hi)), cosmo=ref.cosmo)
        n1, b1 = r.nA_bA(z)
        dN = np.max(np.abs(r.counts() / N0 - 1))
        print(f"\n  mass range [{lo:.1e}, {hi:.1e}] vs [1e12, 1e16]: N {dN:.1e}, "
              f"n_A {np.max(np.abs(n1 / n0 - 1)):.1e}, b_A {np.max(np.abs(b1 / b0 - 1)):.1e}")
        assert dN < 1e-6


def test_counts_and_kernel_quadrature(ref):
    """Count and kernel quadratures are converged and normalized.

    Finer redshift panels change the counts by less than 1e-9; in both
    kernel modes every normalized kernel q integrates to 1 over its support
    (trapezoid rule on 40001 points, to 1e-6), and the magnification
    efficiency g of each kernel tends to 1 next to the observer (z = 1e-4,
    to 1e-3), as a lensing efficiency of a normalized distribution must.

    Arguments:
      ref = the module fixture.
    """
    cl = ref.cluster
    N0 = cl.counts()
    N1 = cl.counts(z_panel=0.002, order=12)
    assert np.max(np.abs(N1 / N0 - 1)) < 1e-9
    for mode in (0, 1):
        norms = cl.kernel_norms(mode)
        for i in range(cl.nzc):
            a, b = cl.support(i)
            zz = np.linspace(a, b, 40001)
            q = cl.q(zz, i, mode, norms)
            assert np.max(np.abs(np.trapz(q, zz, axis=-1) - 1)) < 1e-6
            g = cl.magnification_efficiency(np.array([1e-4]), i, mode, norms)
            assert np.max(np.abs(g - 1)) < 1e-3


def test_counts_vs_y3_printout(ref):
    """Print the counts next to the DES Y3 redMaPPer totals (no assertion).

    The catalog totals 5632, 6308 and 4551 per redshift bin are those of
    the 4143 deg^2 footprint, scaled to the reference area.

    Arguments:
      ref = the module fixture.
    """
    N = ref.counts()
    y3 = np.array([5632.0, 6308.0, 4551.0]) * ref.cluster.area_deg2 / 4143.0
    print("\n  N_iA (rows z_lambda [0.2,0.4) [0.4,0.55) [0.55,0.65); columns lambda [20,30,45,60,500)):")
    for i in range(N.shape[0]):
        print("   ", np.array2string(N[i], precision=1), f" total {N[i].sum():8.1f}"
              f"  Y3 catalog {y3[i]:7.1f}  ratio {N[i].sum() / y3[i]:.3f}")


# ----------------------------------------------------------------------
# Limber spectra and projections
# ----------------------------------------------------------------------
def test_limber_convergence(ref):
    """Finer Limber quadrature and denser multipoles leave the projections unchanged.

    Smaller redshift panels with more nodes change gamma_t, w_cc and w_cg
    by less than 1e-6 (max |diff|/max|signal|); a denser multipole
    sampling, exact up to l = 60, by less than 1e-5.

    Arguments:
      ref = the module fixture (its CAMB run is reused).
    """
    rs0 = ref.real_space()
    keys = ("gamma_t", "w_cc", "w_cg")
    for settings, tol in ((dict(z_panel=0.005, z_order=10), 1e-6),
                          (dict(n_per_decade=120, l_exact=60), 1e-5)):
        rs = ClusterReference(settings=settings, cosmo=ref.cosmo).real_space()
        d = {k: scaled_diff(rs[k], rs0[k]) for k in keys}
        print(f"\n  {settings}: " + ", ".join(f"{k} {v:.1e}" for k, v in d.items()))
        assert max(d.values()) < tol


def test_legendre_polynomials():
    """The Legendre recursion table stays accurate up to l = 75000.

    P_l(cos theta) at four angles between 2.5 and 250 arcmin against
    scipy's eval_legendre (l <= 2000, to 1e-12) and against mpmath with 40
    significant digits at l = 30000 and 75000 (to 1e-10), where an
    unstable recursion would have lost all digits.
    """
    x = np.cos(np.array([2.5, 11.0, 70.0, 250.0]) * np.pi / 180 / 60)
    P = legendre_table(x, 75001)
    for l in (0, 1, 2, 7, 150, 2000):
        assert np.max(np.abs(P[:, l] - eval_legendre(l, x))) < 1e-12
    import mpmath
    mpmath.mp.dps = 40
    for l in (30000, 75000):
        for j, xx in enumerate(x):
            exact = float(mpmath.legendre(l, mpmath.mpf(float(xx))))
            assert abs(P[j, l] - exact) < 1e-10, (l, xx)


def test_projection_vs_bruteforce():
    """Power-law C_l: analytic bin-averaged kernels vs direct averaging over the bin.

    For C_l = 1e-5 (l + 10)^-1.5, w and gamma_t of each of 20 bins from
    the analytic kernels must equal the average over the bin, in
    x = cos(theta), of the Legendre sums (P_l for w, the associated P_l^2
    with 1/(l(l+1)) for gamma_t) taken with 96 Gauss-Legendre points: to
    1e-9 for w and 1e-8 for gamma_t.
    """
    lmax = 3000
    edges = theta_edges(20, 2.5, 250.0)
    Pw, Pg = projection_kernels(edges, lmax)
    l = np.arange(lmax, dtype=float)
    C = np.zeros(lmax)
    C[1:] = 1e-5 * (l[1:] + 10.0) ** -1.5
    w_an, g_an = Pw @ C, Pg @ C
    xg, wg = np.polynomial.legendre.leggauss(96)
    ll = np.arange(1, lmax)
    for i in range(edges.size - 1):
        xmin, xmax = np.cos(edges[i]), np.cos(edges[i + 1])
        xs = 0.5 * (xmin - xmax) * xg + 0.5 * (xmin + xmax)
        wts = 0.5 * wg
        Pl = legendre_table(xs, lmax)[:, 1:lmax]
        w_bf = np.sum(wts * (Pl @ ((2 * ll + 1) / (4 * np.pi) * C[1:])))
        P2 = np.array([lpmv(2, ll, xx) for xx in xs])
        g_bf = np.sum(wts * (P2 @ ((2 * ll + 1) / (4 * np.pi * ll * (ll + 1)) * C[1:])))
        assert abs(w_an[i] / w_bf - 1) < 1e-9, i
        assert abs(g_an[i] / g_bf - 1) < 1e-8, i


def test_cl_spline_reproduces_integer_nodes(ref):
    """Below l_exact every integer l is a node, so the spline returns it exactly.

    Arguments:
      ref = the module fixture.
    """
    sp = ref.spectra()
    ells = sp["ells"]
    ci = cl_on_integers(ells, sp["C_cc"][0, 0, 0, 0], 30)
    assert np.allclose(ci[1:30], sp["C_cc"][0, 0, 0, 0, :29], rtol=1e-12, atol=0)


# ----------------------------------------------------------------------
# Y transform and selection
# ----------------------------------------------------------------------
def test_y_transform():
    """The Y transform matrix T: null last row, stencils and power-law accuracy.

    T maps gamma_t to Y(theta) = int from ln theta to ln theta_max of
    [2 gamma + d gamma/d ln theta]. Its last row is zero (Y(theta_max) = 0
    by definition) and its rank is ntheta - 1 = 19. The derivative D uses
    a one-sided 5-point stencil at the first node and a centered 9-point
    stencil inside. For gamma = theta^-alpha the closed form is
    Y = (2 - alpha)/alpha (theta^-alpha - theta_N^-alpha); T reproduces it
    to 2e-2 with 20 bins and 3e-4 with 200 (discretization error).
    """
    S, D, T = y_transform_matrices(20, np.log(100.0) / 20)
    assert np.all(T[-1] == 0.0) and np.linalg.matrix_rank(T) == 19
    assert np.allclose(D[0, :5] * np.log(100.0) / 20, [-25 / 12, 4, -3, 4 / 3, -1 / 4])
    assert np.allclose(D[10, 6:15] * np.log(100.0) / 20,
                       [1 / 280, -4 / 105, 1 / 5, -4 / 5, 0, 4 / 5, -1 / 5, 4 / 105, -1 / 280])
    # gamma ~ theta^-alpha: Y(theta) = (2 - alpha)/alpha (theta^-a - theta_N^-a)
    for n, tol in ((20, 2e-2), (200, 3e-4)):
        e = theta_edges(n, 2.5, 250.0)
        th = np.sqrt(e[1:] * e[:-1])
        T = y_transform_matrices(n, np.log(e[-1] / e[0]) / n)[2]
        for alpha in (0.5, 1.0, 1.5):
            g = th ** -alpha
            Y = (2 - alpha) / alpha * (th ** -alpha - th[-1] ** -alpha)
            assert np.max(np.abs((T @ g)[:-1] / Y[:-1] - 1)) < tol, (n, alpha)


def test_selection_factor(ref):
    """The selection factor of Eq. 23 against its formula.

    At the fiducial (redshift power 0) B = s0 + s1 exp(-theta chi/s2) with
    chi the comoving distance to the bin midpoint (to 1e-14). At theta = 0
    with z_mid = 0.45 the redshift factor ((1 + z_mid)/1.45)^s3 is 1, so
    B = s0 + s1 = 1.3 for any s3.

    Arguments:
      ref = the module fixture.
    """
    B = ref.selection()
    th = ref.theta_sel()
    zmid = 0.5 * (ref.cluster.zc_edges[:-1] + ref.cluster.zc_edges[1:])
    chi = ref.cosmo.chi(zmid)
    p = FIDUCIAL
    manual = (p["sel_s0"] + p["sel_s1"] * np.exp(-th[None, :] * chi[:, None] / p["sel_s2"]))
    assert np.allclose(B, manual, rtol=1e-14)
    assert np.isclose(selection_factor(0.0, 1000.0, 1.1, 0.2, 30.0, 0.5, z_mid=0.45), 1.3)


# ----------------------------------------------------------------------
# covariance plumbing and the Limber diagnostic
# ----------------------------------------------------------------------
def test_band_matrices_equal_direct_sum(ref):
    """The covariance band matrices equal the direct sum over multipoles.

    For a smooth positive G(l) given on the multipole nodes, the
    w x gamma_t band matrix applied to G must equal
    sum_l Pw_i(l) Pg_j(l) G(l)/(2l + 1) with G splined onto every integer
    l < 5000 (to 1e-10).

    Arguments:
      ref = the module fixture.
    """
    sp = ref.spectra()
    ells = sp["ells"][sp["ells"] <= 5000]
    lmax = 5000
    edges = ref.edges
    M = band_matrices(ells, edges, lmax)
    G = sp["C_cc"][0, 0, 0, 0, :ells.size] ** 2 + 1e-9          # any smooth positive function
    from scipy.interpolate import CubicSpline
    l = np.arange(1, lmax, dtype=float)
    Gl = CubicSpline(np.log(ells), G)(np.log(l))
    Pw, Pg = projection_kernels(edges, lmax)
    direct = (Pw[:, 1:] * Gl / (2 * l + 1)) @ Pg[:, 1:].T
    assert np.allclose(M[("w", "gt")] @ G, direct, rtol=1e-10, atol=1e-10 * np.abs(direct).max())


def test_wcc_limber_diagnostic(ref):
    """At l = 50 the Limber C_cc agrees with the exact one to 3%.

    The exact (spherical Bessel) and Limber linear spectra of the density
    leg of w_cc (cluster bin 0, richness bins 0 x 0) are compared at l = 5
    (printed) and l = 50 (asserted).

    Arguments:
      ref = the module fixture.
    """
    d = ref.wcc_nonlimber_check(ells=(5, 50))
    print("\n  C_cc exact/Limber (linear, density leg, bin 0, lambda 0x0): "
          + ", ".join(f"l={int(l)}: {r:.4f}" for l, r in zip(d["ells"], d["ratio_exact_limber_lin"])))
    assert abs(d["ratio_exact_limber_lin"][-1] - 1) < 0.03
