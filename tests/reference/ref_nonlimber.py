"""Exact (spherical-Bessel) linear-theory angular spectra, used for two
diagnostics of the DES cluster reference:

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
     (ref_covariance).

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
(0.5 Mpc/h in both callers); k integrals: Simpson in ln k on a grid
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
