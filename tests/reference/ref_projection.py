"""Full-sky bin-averaged Legendre projections, the Park+2021 Y transform
and the SELECTIONB selection factor of the DES cluster reference.

A projection turns an angular power spectrum C_l into a real-space
correlation function averaged over an angular bin: the full-sky Legendre
series (no flat-sky approximation), with the kernel of every multipole
averaged analytically over the bin, as cosmolike does.
reference_cluster.py applies it to the Limber spectra of ref_limber, and
ref_covariance and ref_covariance_full reuse the kernels
(projection_kernels) in their covariance sums.

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

Why Y: gamma_t at angle theta responds to all the mass inside theta.
T gamma_t, the bin-grid version of
  2 int_theta^theta_N-1 gamma_t dln theta' + gamma_t(theta_N-1) - gamma_t(theta)
(theta_N-1 = the last angular bin), is instead the convergence
difference kappa(theta) - kappa(theta_N-1): for one lens and one source
plane [Sigma(R) - Sigma(R_max)]/Sigma_crit (Park+2021 Eq. 9), which
depends only on the profile between R and R_max.
The data vector stores T gamma_t and names it Sigma (2503.13631 Eq. 15);
no Sigma_crit is computed. Row N-1 of T is zero, so the last angular bin
of every cluster-lensing row is zero by construction and always masked.

Selection factor (SELECTIONB, the name of this model in the lighthouse
code: 2503.13631 Eq. 23 with the lighthouse z term):
  b_sel(theta) = [s0 + s1 exp(-theta D_M(z_mid)/s2)] ((1+z_mid)/1.45)^s3,
with (s0, s1, s2, s3) = (b_s1, b_s2, r_0, the power of the z term, 0 in
the paper; reference_cluster.FIDUCIAL sel_s0 .. sel_s3), theta in
radians at the area-weighted bin centre
(2/3)(tmax^3 - tmin^3)/(tmax^2 - tmin^2), z_mid the midpoint of the
cluster bin's z_lambda edges, D_M = chi(z_mid) and s2 in comoving Mpc/h
(reference_cluster.ClusterReference.selection makes these choices).
Model: Sigma = b_sel * (T gamma_t), w_cg * b_sel, w_cc * b_sel^2.
"""

import numpy as np
from scipy.interpolate import CubicSpline

# one arcminute in radians
ARCMIN = np.pi / 180.0 / 60.0


def theta_edges(ntheta=20, tmin_arcmin=2.5, tmax_arcmin=250.0):
    """Bin edges in radians (cosmolike: exp of uniform steps in ln theta).

    t_i = exp(ln tmin + i dlnt), dlnt = ln(tmax/tmin)/ntheta.

    Arguments:
      ntheta = number of angular bins (default 20).
      tmin_arcmin = lower edge of the first bin in arcmin (default 2.5).
      tmax_arcmin = upper edge of the last bin in arcmin (default 250).

    Returns:
      numpy array of ntheta + 1 increasing edges in radians.
    """
    lmin, lmax = np.log(tmin_arcmin * ARCMIN), np.log(tmax_arcmin * ARCMIN)
    return np.exp(lmin + np.arange(ntheta + 1) * (lmax - lmin) / ntheta)


def theta_area_weighted(edges):
    """Return the area-weighted centre of every angular bin, in radians.

    (2/3)(t1^3 - t0^3)/(t1^2 - t0^2) is the mean of theta over the bin
    with the flat-sky area weight theta dtheta; the C code
    (compute_binning_real_space, generic_interface.cpp) uses these
    centres.

    Arguments:
      edges = bin edges in radians, (Ntheta + 1,).

    Returns:
      numpy array (Ntheta,) of the centres.
    """
    t0, t1 = edges[:-1], edges[1:]
    return 2.0 / 3.0 * (t1**3 - t0**3) / (t1**2 - t0**2)


def theta_log_centre(edges):
    """Return the logarithmic centre sqrt(t0 t1) of every bin, in radians.

    reference_cluster.ClusterReference.theta_sel returns these when the
    setting theta_sel is "log" (the default is "area").

    Arguments:
      edges = bin edges in radians, (Ntheta + 1,).

    Returns:
      numpy array (Ntheta,) of the centres.
    """
    return np.sqrt(edges[:-1] * edges[1:])


def legendre_table(x, lmax):
    """P_l(x) for l = 0..lmax (inclusive), shape (len(x), lmax+1).

    Bonnet's three-term recurrence
    (l + 1) P_{l+1} = (2l + 1) x P_l - l P_{l-1}, from P_0 = 1 and
    P_1 = x, one column per degree; the upward recurrence is stable for
    |x| <= 1 (test_legendre_polynomials checks it against scipy and
    mpmath up to l = 75000).

    Arguments:
      x = points in [-1, 1], scalar or 1-D array.
      lmax = highest degree, >= 0.

    Returns:
      numpy array of shape (x.size, lmax + 1).
    """
    x = np.asarray(x, dtype=float)
    P = np.empty((x.size, lmax + 1))
    P[:, 0] = 1.0
    if lmax >= 1:
        P[:, 1] = x
    for l in range(1, lmax):
        P[:, l + 1] = ((2 * l + 1) * x * P[:, l] - l * P[:, l - 1]) / (l + 1)
    return P


# Module-global cache of projection_kernels: key (rounded edges, lmax),
# value (Pw, Pg). Each pair is computed once per Python process; at the
# defaults (20 bins, lmax = 75000) Pw and Pg hold 1.5 million doubles
# (12 MB) each.
_KERNEL_CACHE = {}


def projection_kernels(edges, lmax):
    """(Pw, Pg), each shape (Ntheta, lmax): column l multiplies C_l; l = 0 is 0.

    lmax plays the role of cosmolike's Ntable.LMAX: the sums run over
    l = 1 .. lmax-1.

    The closed forms of the module docstring, from Legendre polynomials
    at the bin edges (no quadrature), stored in _KERNEL_CACHE.

    Arguments:
      edges = bin edges in radians, (Ntheta + 1,) increasing.
      lmax = number of columns, l = 0 .. lmax - 1; the sums use
             l = 1 .. lmax - 1 (column 0 is 0, and Pg is 0 up to
             rounding at l = 1, where P_1^2 = 0).

    Returns:
      (Pw, Pg): numpy arrays of shape (Ntheta, lmax); Pw for spin 0 x 0
      (w), Pg for spin 2 x 0 (gamma_t). They are the cached arrays
      themselves (no copy), so a caller must not modify them.
    """
    # dict key: the edges rounded to 15 decimals (radians) as a tuple,
    # which is hashable unlike an array, and lmax
    key = (tuple(np.round(edges, 15)), int(lmax))
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]
    # x = cos(theta) at the edges: x decreases as theta grows
    xe = np.cos(edges)
    P = legendre_table(xe, lmax + 1)                  # l = 0..lmax+1
    # x_min = cos(lower edge) > x_max = cos(upper edge) for every bin;
    # Pmin and Pmax are the matching rows of P
    xmin, xmax = xe[:-1], xe[1:]
    Pmin, Pmax = P[:-1], P[1:]
    # l as floats for the arithmetic, li as integers for the column
    # lookups li - 1, li and li + 1; column 0 (l = 0) of Pw, Pg stays 0
    l = np.arange(1, lmax, dtype=float)
    li = np.arange(1, lmax)
    # a column (Ntheta, 1), broadcast against the row of l values
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
    (l = 0 set to 0). Integer nodes are reproduced exactly.

    The spline passes through its nodes, so at node multipoles that are
    integers (l < l_exact of ref_limber.default_ells) the value is the
    node value up to rounding (test_cl_spline_reproduces_integer_nodes).

    Arguments:
      ells = node multipoles, increasing, (nb,).
      cl = spectra on the nodes, shape (..., nb) (last axis = ells).
      lmax = length of the last axis of the result (l = 0 .. lmax - 1).

    Returns:
      numpy array of shape cl.shape[:-1] + (lmax,).
    """
    l = np.arange(1, lmax, dtype=float)
    spl = CubicSpline(np.log(ells), cl, axis=-1)
    out = np.zeros(cl.shape[:-1] + (lmax,))
    # `...` (Ellipsis) stands for all leading axes
    out[..., 1:] = spl(np.log(l))
    return out


def project(ells, cl, edges, lmax=75000, kind="w"):
    """Real-space projection of spectra given on the node multipoles `ells`
    (last axis of cl). kind = "w" (spin 0 x 0) or "gt" (spin 2 x 0).

    C_l is splined onto every integer l = 1 .. lmax - 1 (cl_on_integers)
    and summed against the bin-averaged kernels: w_i = sum_l Pw_i(l) C_l
    or gamma_t,i = sum_l Pg_i(l) C_l.

    Arguments:
      ells = node multipoles of the spectra, (nb,).
      cl = spectra, shape (..., nb).
      edges = angular bin edges in radians, (Ntheta + 1,).
      lmax = LMAX of the sums (default 75000).
      kind = "w" for the Legendre kernels; any other value selects the
             gamma_t kernels (no check).

    Returns:
      numpy array of shape cl.shape[:-1] + (Ntheta,), dimensionless.
    """
    Pw, Pg = projection_kernels(edges, lmax)
    K = Pw if kind == "w" else Pg
    cli = cl_on_integers(ells, cl, lmax)
    # (..., lmax) @ (lmax, Ntheta): the sum over l for every leading index
    return cli @ K.T


# ----------------------------------------------------------------------
# Y transform and selection bias
# ----------------------------------------------------------------------
# Central first-derivative stencils: _D_CENTRAL[w] holds the coefficients
# of f(i+1) .. f(i+w) of the stencil of half-width w (accuracy order 2w);
# f(i-m) gets the opposite sign. _D_FORWARD is the one-sided 5-point
# stencil (order 4), the coefficients of f(0) .. f(4). Both are those of
# lighthouse cluster_util.c T_Ytransform (module docstring).
_D_CENTRAL = {
    1: [0.5],
    2: [2.0 / 3.0, -1.0 / 12.0],
    3: [3.0 / 4.0, -3.0 / 20.0, 1.0 / 60.0],
    4: [4.0 / 5.0, -1.0 / 5.0, 4.0 / 105.0, -1.0 / 280.0],
}
_D_FORWARD = [-25.0 / 12.0, 4.0, -3.0, 4.0 / 3.0, -1.0 / 4.0]


def y_transform_matrices(ntheta, delta):
    """S, D and T = 2S + S D of Park+2021 (see module docstring).

    Arguments:
      ntheta = number of angular bins N, >= 5 (the one-sided stencils
               need five bins).
      delta = ln(theta_max/theta_min)/N, the bin width in ln theta.

    Returns:
      (S, D, T): three numpy arrays of shape (N, N); T applied to a
      gamma_t vector gives Sigma (module docstring).

    Raises:
      ValueError when ntheta < 5.
    """
    N = ntheta
    if N < 5:
        raise ValueError("the Y transform needs at least 5 bins")
    S = np.zeros((N, N))
    # row i: trapezoid weights from bin i to bin N-1 (the slice
    # i + 1:N - 1 is columns i+1 .. N-2); row N-1 stays zero
    for i in range(N - 1):
        S[i, i] = 0.5 * delta
        S[i, i + 1:N - 1] = delta
        S[i, N - 1] = 0.5 * delta
    D = np.zeros((N, N))
    D[0, :5] = _D_FORWARD
    # the last row mirrors the forward stencil: reversed order ([::-1])
    # and opposite sign (a list comprehension)
    D[N - 1, N - 5:] = [-c for c in _D_FORWARD[::-1]]
    for i in range(1, N - 1):
        w = min(i, N - 1 - i, 4)
        # enumerate(..., start=1) yields (1, c_1), (2, c_2), ...: m is the
        # offset of coefficient c from the centre i
        for m, c in enumerate(_D_CENTRAL[w], start=1):
            D[i, i + m] = c
            D[i, i - m] = -c
    D /= delta
    # @ is the matrix product
    T = 2.0 * S + S @ D
    return S, D, T


def y_transform(edges):
    """Return the Y-transform matrix T (Ntheta, Ntheta) for log-spaced edges.

    Arguments:
      edges = angular bin edges in radians, uniform in ln theta.

    Returns:
      numpy array T of y_transform_matrices with
      delta = ln(edges[-1]/edges[0])/Ntheta.
    """
    ntheta = edges.size - 1
    delta = np.log(edges[-1] / edges[0]) / ntheta
    return y_transform_matrices(ntheta, delta)[2]


def selection_factor(theta, D_M, s0, s1, s2, s3=0.0, z_mid=None, z_piv=1.45):
    """b_sel(theta) = [s0 + s1 exp(-theta D_M/s2)] ((1+z_mid)/z_piv)^s3.

    Arguments:
      theta = angle in radians (the bin centres), scalar or array.
      D_M = comoving transverse distance to z_mid, in Mpc/h.
      s0 = b_s1 of eq. 23, the large-scale value.
      s1 = b_s2 of eq. 23; (s0 + s1) times the redshift factor is the
           value at theta -> 0.
      s2 = r_0 of eq. 23, the transition scale in comoving Mpc/h.
      s3 = power of the redshift factor (default 0, the paper's model).
      z_mid = cluster bin midpoint; None (or s3 = 0) drops the redshift
              factor.
      z_piv = pivot of (1 + z_mid), default 1.45 (the value is 1 + z_piv,
              as in the MOR).

    Returns:
      numpy array of b_sel, dimensionless, the shape of theta.
    """
    zf = 1.0 if (z_mid is None or s3 == 0.0) else ((1.0 + z_mid) / z_piv) ** s3
    return (s0 + s1 * np.exp(-np.asarray(theta) * D_M / s2)) * zf
