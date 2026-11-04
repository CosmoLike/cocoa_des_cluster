"""Exact (spherical-Bessel) linear-theory angular spectra, used for two
diagnostics of the DES cluster reference and for its non-Limber spectra
(item 3):

  1. the Limber check of w_cc at l <= 50 (one richness pair, density leg
     only, linear P with the scale-independent growth D(z);
     reference_cluster.ClusterReference.wcc_nonlimber_check):

       C_l^exact = (2/pi) int dk k^2 P_lin(k, 0) Delta_1(k) Delta_2(k),
       Delta_X(k) = int dchi W_X(chi) b_X(z) D(z) j_l(k chi)

     against the Limber value of the same integrand,
       C_l^Limber = int dchi W_1 W_2 b_1 b_2 D^2 P_lin((l+1/2)/chi, 0)/chi^2,
     and the FKEM-style corrected spectrum
       C_l = C_l^Limber[P_NL] + (C_l^exact - C_l^Limber)[P_lin];

  2. the super-sample (footprint x shell) variance of the counts
     (ref_covariance);

  3. the non-Limber corrections with redshift-space distortions and
     magnification of every clustering spectrum (C_cc, C_cg, C_gg) below
     the likelihood's limit l = 150 (fkem_corrections), which the
     reference adds to its Limber spectra when settings["nonlimber_lmax"]
     > 0, as the synthetic covariance does.

j_l is the spherical Bessel function; W_X is the radial kernel per unit
chi (normalized dN/dchi), b_X the bias, D(z) the growth factor of
ref_cosmology and P_lin(k, 0) the linear total-matter spectrum at z = 0;
k in h/Mpc, chi in Mpc/h. The Limber approximation (ref_limber) replaces
j_l by its value at k = (l + 1/2)/chi and fails at low l, where the
exact integral is needed. FKEM is the non-Limber method of Fang, Krause,
Eifler and MacCrann (arXiv:1911.11947): the exact-minus-Limber
difference, computed in linear theory, is added to the nonlinear Limber
spectrum. The super-sample variance of item 2 is the sample variance of
the counts: the fluctuation of the matter density averaged over the
survey footprint times the redshift shell of a cluster bin.

Radial integrals: Simpson on a uniform chi grid that the caller passes
(0.5 Mpc/h in the two diagnostics; fkem_corrections builds its own,
1 Mpc/h by default); k integrals: Simpson in ln k on a grid
uniform in ln k below k_lin and uniform in k above it. Delta(k)
oscillates in k with the period 2 pi/chi of j_l(k chi) (3.2e-3 h/Mpc at
chi = 1950 Mpc/h, z = 0.8, the top of the default Gaussian support of
the last cluster bin), under an envelope set by the width of the
kernel; the uniform step dk = 2.5e-4 h/Mpc of k_grid samples that period
about 13 times.
"""

import numpy as np
from scipy.integrate import simpson
from scipy.special import spherical_jn


def k_grid(kmin=1e-5, k_lin=5e-3, kmax=0.5, n_log=400, dk=2.5e-4):
    """Return the k grid of the exact integrals, in h/Mpc.

    n_log nodes uniform in ln k on [kmin, k_lin], then a uniform step dk
    from k_lin + dk up to kmax (inclusive): 2380 nodes at the defaults.

    Arguments:
      kmin = first node (default 1e-5).
      k_lin = end of the logarithmic part (default 5e-3).
      kmax = last node (default 0.5; ref_covariance.counts_covariance
             passes 0.25).
      n_log = number of logarithmic nodes (default 400).
      dk = step of the uniform part (default 2.5e-4; module docstring).

    Returns:
      numpy array of increasing wavenumbers.
    """
    lo = np.exp(np.linspace(np.log(kmin), np.log(k_lin), n_log))
    hi = np.arange(k_lin + dk, kmax + 0.5 * dk, dk)
    return np.concatenate([lo, hi])


def delta_l(ell, chi, kernels, k):
    """Delta[f, k] = int dchi kernels[f](chi) j_l(k chi), Simpson in chi.

    Arguments:
      ell = multipole l (converted to int).
      chi = uniform comoving-distance grid in Mpc/h, (nchi,).
      kernels = radial kernels on chi, (nf, nchi), with every
                z-dependent factor included.
      k = wavenumbers in h/Mpc, (nk,).

    Returns:
      numpy array of shape (nf, nk).
    """
    out = np.empty((kernels.shape[0], k.size))
    # k in chunks of about 200 values (np.array_split cuts the index range
    # into k.size // 200 pieces), so the Bessel table J (chunk, nchi)
    # stays small; kernels[:, None, :] * J[None] broadcasts to
    # (nf, chunk, nchi), and simpson integrates the last axis (chi)
    for sl in np.array_split(np.arange(k.size), max(1, k.size // 200)):
        J = spherical_jn(int(ell), np.outer(k[sl], chi))          # (k, chi)
        out[:, sl] = simpson(kernels[:, None, :] * J[None], x=chi, axis=-1)
    return out


def cl_exact_linear(cosmo, ells, chi, kernels, k=None):
    """(2/pi) int dk k^2 P_lin(k,0) Delta_f Delta_f', shape (nl, nf, nf).

    kernels (nf, nchi) must already contain every z-dependent factor
    (W(chi) b(z) D(z)); P_lin is the z = 0 total-matter spectrum.

    The k integral runs in ln k (dk k^2 = dln k k^3), with Simpson's rule
    on the nonuniform ln k nodes of the grid.

    Arguments:
      cosmo = ref_cosmology.Cosmology.
      ells = multipoles (one Bessel table per multipole).
      chi = uniform comoving-distance grid in Mpc/h, (nchi,).
      kernels = (nf, nchi) radial kernels in 1/(Mpc/h).
      k = wavenumbers in h/Mpc (None: k_grid()).

    Returns:
      numpy array of shape (len(ells), nf, nf): the spectrum of every
      kernel pair.
    """
    k = k_grid() if k is None else k
    pk = cosmo.P_lin(k, np.zeros_like(k))
    out = np.empty((len(ells), kernels.shape[0], kernels.shape[0]))
    lnk = np.log(k)
    for a, ell in enumerate(ells):
        d = delta_l(ell, chi, kernels, k)
        # d[:, None, :] * d[None, :, :] is the outer product over the
        # kernel index, (nf, nf, nk), holding Delta_f Delta_f'
        integrand = (k**3 * pk)[None, None, :] * d[:, None, :] * d[None, :, :]
        out[a] = 2.0 / np.pi * simpson(integrand, x=lnk, axis=-1)
    return out


def cl_limber_linear(cosmo, ells, chi, kernels):
    """Limber counterpart of cl_exact_linear on the same kernels.

    C_l = int dchi K_f K_f' P_lin((l + 1/2)/chi, 0)/chi^2, Simpson in chi.

    Arguments:
      cosmo = ref_cosmology.Cosmology.
      ells = multipoles.
      chi = uniform comoving-distance grid in Mpc/h, (nchi,).
      kernels = (nf, nchi) radial kernels, as for cl_exact_linear.

    Returns:
      numpy array of shape (len(ells), nf, nf).
    """
    ells = np.asarray(ells, dtype=float)
    kk = (ells[:, None] + 0.5) / chi[None, :]
    pk = cosmo.P_lin(kk, np.zeros_like(kk))
    # integrand (nl, nf, nf, nchi): kernels[None, :, None, :] times
    # kernels[None, None, :, :] is the pair product, and
    # (pk/chi^2)[:, None, None, :] adds the multipole axis
    integrand = (kernels[None, :, None, :] * kernels[None, None, :, :]
                 * (pk / chi[None, :] ** 2)[:, None, None, :])
    return simpson(integrand, x=chi, axis=-1)


def simpson_weights(n, dx):
    """Simpson weights of n (odd) uniform nodes of step dx."""
    if n % 2 == 0 or n < 3:
        raise ValueError(f"Simpson needs an odd number of nodes >= 3, got {n}")
    w = np.full(n, 2.0)
    w[1::2] = 4.0
    w[0] = w[-1] = 1.0
    return w * dx / 3.0


def fkem_corrections(lim, cosmo, ells, dchi=1.0, k=None, rsd=True, magnification=True):
    """Non-Limber corrections of every clustering spectrum (FKEM, with RSD).

    The non-Limber spectrum of the likelihood (cosmolike_core
    cosmo2D_cluster.c, section NON-LIMBER; cosmo2D.c C_cl_tomo for w_gg)
    is, below its multipole limit,
      C_l = C_l^Limber[P_NL] + Delta C_l,
      Delta C_l = C_l^exact[P_lin] - C_l^Limber[P_lin],
    the second term in linear theory with separable growth. This function
    returns Delta C for every pair of clustering fields: the cluster fields
    c_iA (cluster bin i, richness bin A) and the lens fields g of
    lim.lens_bins.

    Each field X has three radial pieces on a uniform chi grid, built from
    the kernels of ref_limber (the same W_c, W_mag,c, b_A, W_g, W_mag,g as
    the Limber spectra), times the growth D(z) (D(0) = 1):
      density        K_d = b W D           (cluster: b_A, lens: b1)
      RSD            K_v = -f W D          (f = dln D/dln a; the Kaiser term)
      magnification  K_m = C W_mag D       (cluster: C_c, lens: bmag)
    and its exact transfer at multipole l is
      Delta_X(l, k) = int dchi [K_d j_l(x) + K_v j_l''(x)
                                + l(l+1) K_m j_l(x)/x^2],   x = k chi,
      C^exact_XY = (2/pi) int dk k^2 P_lin(k, 0) Delta_X Delta_Y.
    Its Limber counterpart keeps density and magnification, with the
    magnification prefactor e1(l) = l(l+1)/(l+1/2)^2 that the Limber limit
    of the exact magnification piece gives:
      C^Limber_XY = int dchi L_X L_Y P_lin((l+1/2)/chi, 0)/chi^2,
      L = K_d + e1(l) K_m.
    RSD enters only through the exact term, as in the C model, whose Limber
    cluster spectra carry no RSD either; Delta C -> 0 at high l because the
    density and magnification pieces cancel and the RSD piece decays.

    Quadrature: Simpson in chi (step dchi, Mpc/h) from z = 1e-4 to the last
    node where some kernel is nonzero (at most the top of the far Limber
    node set, lim.z_top), Simpson in ln k on k_grid(). j_l'' from the
    spherical Bessel equation,
      j_l'' = -(2/x) j_l' - (1 - l(l+1)/x^2) j_l,  j_l' = j_(l-1) - (l+1) j_l/x.
    The cost is about one Bessel table of k_grid() x chi per multipole
    (seconds each): reference_cluster evaluates it on the nodes of
    correction_nodes and splines the result (fkem_on_integers).

    Arguments:
      lim = the ref_limber.LimberModel of the reference (its kernel
            methods, lens bins, nuisance values and C_c).
      cosmo = ref_cosmology.Cosmology.
      ells = integer multipoles of the corrections.
      dchi = chi step in Mpc/h (default 1.0: about 12 nodes per period of
             j_l at the largest k of k_grid, 0.5 h/Mpc).
      k = wavenumbers in h/Mpc (None: k_grid()).
      rsd, magnification = False drops that radial piece (K_v or K_m)
             from both terms; the self-tests use it to compare the density
             piece alone with cl_exact_linear - cl_limber_linear.

    Returns:
      dict with "ells" and the corrections dC_cc (i, A, j, B, l),
      dC_cg (i, A, g, l) and dC_gg (g, g', l) in the shapes of the
      ref_limber spectra.
    """
    from types import SimpleNamespace
    k = k_grid() if k is None else k
    ells = np.asarray(ells, dtype=int)

    # uniform chi grid with an odd number of nodes (Simpson)
    chi0, chi1 = cosmo.chi(1e-4), cosmo.chi(lim.z_top)
    n = int(np.ceil((chi1 - chi0) / dchi)) + 1
    n += (n + 1) % 2
    chi = np.linspace(chi0, chi1, n)
    wchi = simpson_weights(n, chi[1] - chi[0])
    z = cosmo.z_of_chi(chi)
    D = cosmo.growth(z)
    # f = dln D/dln a = -(1 + z) dln D/dz, by central differences
    h = 1e-4
    f = -(1.0 + z) * (np.log(cosmo.growth(z + h)) - np.log(cosmo.growth(np.maximum(z - h, 0.0)))) \
        / (z + h - np.maximum(z - h, 0.0))
    ns = SimpleNamespace(z=z, chi=chi, E=cosmo.E(z))

    # the kernels of ref_limber on the chi grid. The cluster kernels and
    # their lensing efficiency vanish above the top of the cluster support,
    # so they are evaluated below it only (inside the cluster mass tables)
    zc_top = max(lim.cl.support(i)[1] for i in range(lim.cl.nzc))
    near = z <= zc_top
    Wc_n, Wmc_n, b_n, _ = lim.cluster_kernels(
        SimpleNamespace(z=z[near], chi=chi[near], E=ns.E[near]))
    Wc = np.zeros(Wc_n.shape[:2] + (n,))
    Wmc = np.zeros_like(Wc)
    b = np.zeros((b_n.shape[0], n))
    Wc[..., near], Wmc[..., near], b[:, near] = Wc_n, Wmc_n, b_n
    Wg, Wmg = lim.lens_kernels(ns)                   # (g, n)
    bg = np.array([lim.nuis["lens_b1"][g] for g in lim.lens_bins])
    bm = np.array([lim.nuis["lens_bmag"][g] for g in lim.lens_bins])
    nzc, nA = Wc.shape[0], Wc.shape[1]
    ng = Wg.shape[0]

    # field order: cluster (i, A) row-major, then the lens bins
    Kd = np.concatenate([(b[None] * Wc).reshape(nzc * nA, n), bg[:, None] * Wg]) * D
    Kv = np.concatenate([Wc.reshape(nzc * nA, n), Wg]) * (-f * D) * float(rsd)
    Km = np.concatenate([lim.C_c * Wmc.reshape(nzc * nA, n), bm[:, None] * Wmg]) * D \
        * float(magnification)
    nf = Kd.shape[0]

    # Beyond the last node where some kernel is nonzero (the top of the lens
    # n(z) tables) every integrand vanishes: drop those nodes, keeping an
    # odd count for Simpson. This changes no sum.
    used = np.flatnonzero(np.any((Kd != 0) | (Kv != 0) | (Km != 0), axis=0))
    m = int(used[-1]) + 1 if used.size else n       # n is odd, so m + 1 <= n
    m += (m + 1) % 2
    chi, Kd, Kv, Km = chi[:m], Kd[:, :m], Kv[:, :m], Km[:, :m]
    wchi = simpson_weights(m, chi[1] - chi[0])

    lnk = np.log(k)
    pk = cosmo.P_lin(k, np.zeros_like(k))
    dC = np.empty((ells.size, nf, nf))
    for a, ell in enumerate(ells):
        lp = ell * (ell + 1.0)
        delta = np.empty((nf, k.size))
        # k in chunks: the Bessel tables (chunk, n) stay a few MB
        for sl in np.array_split(np.arange(k.size), max(1, k.size // 100)):
            x = np.outer(k[sl], chi)
            j = spherical_jn(int(ell), x)
            # j_l' = j_(l-1) - (l+1) j_l/x (j_0' = -j_1): one Bessel call
            # instead of the two of spherical_jn(derivative=True)
            if ell > 0:
                jp = spherical_jn(int(ell) - 1, x) - (ell + 1.0) * j / x
            else:
                jp = -spherical_jn(1, x)
            j2 = -(2.0 / x) * jp - (1.0 - lp / x**2) * j
            delta[:, sl] = ((Kd * wchi) @ j.T + (Kv * wchi) @ j2.T
                            + lp * ((Km * wchi) @ (j / x**2).T))
        exact = 2.0 / np.pi * simpson((k**3 * pk)[None, None, :]
                                      * delta[:, None, :] * delta[None, :, :], x=lnk, axis=-1)
        e1 = lp / (ell + 0.5) ** 2
        L = Kd + e1 * Km
        kl = (ell + 0.5) / chi
        pl = cosmo.P_lin(kl, np.zeros_like(kl)) / chi**2
        limber = (L * (wchi * pl)) @ L.T
        dC[a] = exact - limber

    nc = nzc * nA
    # move the multipole axis last, as in the ref_limber spectra
    cc = dC[:, :nc, :nc].reshape(ells.size, nzc, nA, nzc, nA)
    cg = dC[:, :nc, nc:].reshape(ells.size, nzc, nA, ng)
    gg = dC[:, nc:, nc:]
    return dict(ells=ells.astype(float),
                dC_cc=np.moveaxis(cc, 0, -1), dC_cg=np.moveaxis(cg, 0, -1),
                dC_gg=np.moveaxis(gg, 0, -1))


def correction_nodes(ells, steps=((20, 2), (40, 4))):
    """Multipoles at which to evaluate fkem_corrections for the targets ells.

    Every integer target below the first entry of steps, then every
    step-th integer from each start (20 and 40 by default: every 2nd from
    20, every 4th from 40), always ending at the largest target. The
    correction varies smoothly with l, but the cross-bin cluster pairs
    vary on Delta l ~ 10 at l ~ 100, so the nodes stay a few multipoles
    apart there (log-spaced nodes 10 apart missed it by 5% of the
    correction); the spline error falls as the fourth power of the step.

    Arguments:
      ells = increasing integer target multipoles.
      steps = (start, step) pairs, increasing starts.

    Returns:
      increasing integer array.
    """
    ells = np.asarray(ells, dtype=int)
    nodes = [ells[ells < steps[0][0]]]
    bounds = [s for s, _ in steps] + [ells[-1] + 1]
    for (start, step), stop in zip(steps, bounds[1:]):
        nodes.append(np.arange(max(start, ells[0]), min(stop, ells[-1] + 1), step))
    nodes.append(ells[-1:])
    nodes = np.union1d(np.concatenate(nodes), [])
    return nodes[(nodes >= ells[0]) & (nodes <= ells[-1])].astype(int)


def fkem_on_integers(lim, cosmo, ells, **kwargs):
    """fkem_corrections on correction_nodes(ells), splined onto ells.

    A cubic spline in ln l of each correction (not-a-knot, scipy's
    CubicSpline), exact at the nodes; below 20 every target is a node.

    Arguments:
      lim, cosmo = as for fkem_corrections.
      ells = increasing integer target multipoles.
      kwargs = passed to fkem_corrections.

    Returns:
      the dict of fkem_corrections, on ells.
    """
    from scipy.interpolate import CubicSpline
    ells = np.asarray(ells, dtype=int)
    nodes = correction_nodes(ells)
    corr = fkem_corrections(lim, cosmo, nodes, **kwargs)
    out = dict(ells=ells.astype(float))
    for key in ("dC_cc", "dC_cg", "dC_gg"):
        out[key] = CubicSpline(np.log(nodes), corr[key], axis=-1)(np.log(ells))
    return out
