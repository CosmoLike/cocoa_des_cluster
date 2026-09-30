"""Full-sky bin-averaged Legendre projections, the Park+2021 Y transform
and the SELECTIONB selection factor of the DES cluster reference.

Projections (conventions of cosmolike cosmo2D.c w_gg_tomo /
w_gammat_tomo and basics.c set_bin_average):

  theta edges  : t_i = exp(ln tmin + i dlnt), dlnt = ln(tmax/tmin)/Ntheta
  x_min = cos(t_i), x_max = cos(t_{i+1})  (so x_min > x_max)

  w(theta_i)       = sum_{l=1}^{LMAX-1} Pw_i(l) C_l
  Pw_i(l)          = [P_{l+1}(x_min) - P_{l+1}(x_max) - P_{l-1}(x_min) + P_{l-1}(x_max)]
                     / (4 pi (x_min - x_max))
                   = (2l+1)/(4 pi) * <P_l(x)>_bin

  gamma_t(theta_i) = sum_{l=1}^{LMAX-1} Pg_i(l) C_l
  Pg_i(l)          = (2l+1)/(4 pi l(l+1)) <P_l^2(x)>_bin
                   = (2l+1)/(4 pi l(l+1) (x_min - x_max))
                     * [ (l + 2/(2l+1)) (P_{l-1}(x_min) - P_{l-1}(x_max))
                         + (2 - l) (x_min P_l(x_min) - x_max P_l(x_max))
                         - 2/(2l+1) (P_{l+1}(x_min) - P_{l+1}(x_max)) ]

  (<.>_bin = average over x = cos(theta) uniform in [x_max, x_min],
   P_l^2 = (1 - x^2) P_l'' with no Condon-Shortley sign.)

The P_l(x) are built with the three-term recurrence in numpy; the tests
check them against scipy / mpmath and the bin averages against direct
numerical averaging.

Y transform (Park, Rozo & Krause 2021, 2004.07504 Eqs. 9-10, as in
lighthouse cluster_util.c:252-361): on the Ntheta log-spaced bins,
Delta = ln(tmax/tmin)/Ntheta,

  S[i,i] = Delta/2, S[i,j] = Delta for i < j < N-1, S[i,N-1] = Delta/2,
  row N-1 of S = 0 (trapezoid integral from theta_i to theta_max);
  D = d/dln(theta): central stencils of half-width min(i, N-1-i, 4)
  (orders 2, 4, 6, 8) and the 5-point one-sided stencil in rows 0, N-1,
  divided by Delta;
  T = 2 S + S D.

Selection factor (SELECTIONB, 2503.13631 Eq. 23 with the lighthouse z
term):  b_sel(theta) = [s0 + s1 exp(-theta D_M(z_mid)/s2)] ((1+z_mid)/1.45)^s3,
theta in radians at the area-weighted bin centre
(2/3)(tmax^3 - tmin^3)/(tmax^2 - tmin^2), D_M and s2 in comoving Mpc/h.
Model: Sigma = b_sel * (T gamma_t), w_cg * b_sel, w_cc * b_sel^2.
"""

import numpy as np
from scipy.interpolate import CubicSpline

ARCMIN = np.pi / 180.0 / 60.0


def theta_edges(ntheta=20, tmin_arcmin=2.5, tmax_arcmin=250.0):
    """Bin edges in radians (cosmolike: exp of uniform steps in ln theta)."""
    lmin, lmax = np.log(tmin_arcmin * ARCMIN), np.log(tmax_arcmin * ARCMIN)
    return np.exp(lmin + np.arange(ntheta + 1) * (lmax - lmin) / ntheta)


def theta_area_weighted(edges):
    t0, t1 = edges[:-1], edges[1:]
    return 2.0 / 3.0 * (t1**3 - t0**3) / (t1**2 - t0**2)


def theta_log_centre(edges):
    return np.sqrt(edges[:-1] * edges[1:])


def legendre_table(x, lmax):
    """P_l(x) for l = 0..lmax (inclusive), shape (len(x), lmax+1)."""
    x = np.asarray(x, dtype=float)
    P = np.empty((x.size, lmax + 1))
    P[:, 0] = 1.0
    if lmax >= 1:
        P[:, 1] = x
    for l in range(1, lmax):
        P[:, l + 1] = ((2 * l + 1) * x * P[:, l] - l * P[:, l - 1]) / (l + 1)
    return P


_KERNEL_CACHE = {}


def projection_kernels(edges, lmax):
    """(Pw, Pg), each shape (Ntheta, lmax): column l multiplies C_l; l = 0 is 0.

    lmax plays the role of cosmolike's Ntable.LMAX: the sums run over
    l = 1 .. lmax-1.
    """
    key = (tuple(np.round(edges, 15)), int(lmax))
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]
    xe = np.cos(edges)
    P = legendre_table(xe, lmax + 1)                  # l = 0..lmax+1
    xmin, xmax = xe[:-1], xe[1:]
    Pmin, Pmax = P[:-1], P[1:]
    l = np.arange(1, lmax, dtype=float)
    li = np.arange(1, lmax)
    dx = (xmin - xmax)[:, None]
    Pw = np.zeros((edges.size - 1, lmax))
    Pw[:, 1:] = ((Pmin[:, li + 1] - Pmax[:, li + 1] - Pmin[:, li - 1] + Pmax[:, li - 1])
                 / (4.0 * np.pi * dx))
    Pg = np.zeros_like(Pw)
    Pg[:, 1:] = ((2 * l + 1) / (4.0 * np.pi * l * (l + 1)) / dx
                 * ((l + 2.0 / (2 * l + 1)) * (Pmin[:, li - 1] - Pmax[:, li - 1])
                    + (2 - l) * (xmin[:, None] * Pmin[:, li] - xmax[:, None] * Pmax[:, li])
                    - 2.0 / (2 * l + 1) * (Pmin[:, li + 1] - Pmax[:, li + 1])))
    _KERNEL_CACHE[key] = (Pw, Pg)
    return Pw, Pg


def cl_on_integers(ells, cl, lmax):
    """Cubic spline in ln l of C_l (last axis) evaluated at l = 0..lmax-1
    (l = 0 set to 0). Integer nodes are reproduced exactly."""
    l = np.arange(1, lmax, dtype=float)
    spl = CubicSpline(np.log(ells), cl, axis=-1)
    out = np.zeros(cl.shape[:-1] + (lmax,))
    out[..., 1:] = spl(np.log(l))
    return out


def project(ells, cl, edges, lmax=75000, kind="w"):
    """Real-space projection of spectra given on the node multipoles `ells`
    (last axis of cl). kind = "w" (spin 0 x 0) or "gt" (spin 2 x 0)."""
    Pw, Pg = projection_kernels(edges, lmax)
    K = Pw if kind == "w" else Pg
    cli = cl_on_integers(ells, cl, lmax)
    return cli @ K.T


# ----------------------------------------------------------------------
# Y transform and selection bias
# ----------------------------------------------------------------------
_D_CENTRAL = {
    1: [0.5],
    2: [2.0 / 3.0, -1.0 / 12.0],
    3: [3.0 / 4.0, -3.0 / 20.0, 1.0 / 60.0],
    4: [4.0 / 5.0, -1.0 / 5.0, 4.0 / 105.0, -1.0 / 280.0],
}
_D_FORWARD = [-25.0 / 12.0, 4.0, -3.0, 4.0 / 3.0, -1.0 / 4.0]


def y_transform_matrices(ntheta, delta):
    """S, D and T = 2S + S D of Park+2021 (see module docstring)."""
    N = ntheta
    if N < 5:
        raise ValueError("the Y transform needs at least 5 bins")
    S = np.zeros((N, N))
    for i in range(N - 1):
        S[i, i] = 0.5 * delta
        S[i, i + 1:N - 1] = delta
        S[i, N - 1] = 0.5 * delta
    D = np.zeros((N, N))
    D[0, :5] = _D_FORWARD
    D[N - 1, N - 5:] = [-c for c in _D_FORWARD[::-1]]
    for i in range(1, N - 1):
        w = min(i, N - 1 - i, 4)
        for m, c in enumerate(_D_CENTRAL[w], start=1):
            D[i, i + m] = c
            D[i, i - m] = -c
    D /= delta
    T = 2.0 * S + S @ D
    return S, D, T


def y_transform(edges):
    ntheta = edges.size - 1
    delta = np.log(edges[-1] / edges[0]) / ntheta
    return y_transform_matrices(ntheta, delta)[2]


def selection_factor(theta, D_M, s0, s1, s2, s3=0.0, z_mid=None, z_piv=1.45):
    """b_sel(theta) = [s0 + s1 exp(-theta D_M/s2)] ((1+z_mid)/z_piv)^s3."""
    zf = 1.0 if (z_mid is None or s3 == 0.0) else ((1.0 + z_mid) / z_piv) ** s3
    return (s0 + s1 * np.exp(-np.asarray(theta) * D_M / s2)) * zf
