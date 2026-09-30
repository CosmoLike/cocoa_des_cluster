"""Exact (spherical-Bessel) linear-theory angular spectra, used for two
diagnostics of the DES cluster reference:

  1. the Limber check of w_cc at l <= 50 (one richness pair, density leg
     only, linear P with the scale-independent growth D(z)):

       C_l^exact = (2/pi) int dk k^2 P_lin(k, 0) Delta_1(k) Delta_2(k),
       Delta_X(k) = int dchi W_X(chi) b_X(z) D(z) j_l(k chi)

     against the Limber value of the same integrand,
       C_l^Limber = int dchi W_1 W_2 b_1 b_2 D^2 P_lin((l+1/2)/chi, 0)/chi^2,
     and the FKEM-style corrected spectrum
       C_l = C_l^Limber[P_NL] + (C_l^exact - C_l^Limber)[P_lin];

  2. the super-sample (footprint x shell) variance of the counts
     (ref_covariance).

Radial integrals: Simpson on a uniform chi grid (0.5 Mpc/h default),
k integrals: Simpson on a grid uniform in ln k below k_lin and uniform in
k above it (the Delta's oscillate with period ~ 2 pi/width of the kernel).
"""

import numpy as np
from scipy.integrate import simpson
from scipy.special import spherical_jn


def k_grid(kmin=1e-5, k_lin=5e-3, kmax=0.5, n_log=400, dk=2.5e-4):
    lo = np.exp(np.linspace(np.log(kmin), np.log(k_lin), n_log))
    hi = np.arange(k_lin + dk, kmax + 0.5 * dk, dk)
    return np.concatenate([lo, hi])


def delta_l(ell, chi, kernels, k):
    """Delta[f, k] = int dchi kernels[f](chi) j_l(k chi), Simpson in chi."""
    out = np.empty((kernels.shape[0], k.size))
    for sl in np.array_split(np.arange(k.size), max(1, k.size // 200)):
        J = spherical_jn(int(ell), np.outer(k[sl], chi))          # (k, chi)
        out[:, sl] = simpson(kernels[:, None, :] * J[None], x=chi, axis=-1)
    return out


def cl_exact_linear(cosmo, ells, chi, kernels, k=None):
    """(2/pi) int dk k^2 P_lin(k,0) Delta_f Delta_f', shape (nl, nf, nf).

    kernels (nf, nchi) must already contain every z-dependent factor
    (W(chi) b(z) D(z)); P_lin is the z = 0 total-matter spectrum.
    """
    k = k_grid() if k is None else k
    pk = cosmo.P_lin(k, np.zeros_like(k))
    out = np.empty((len(ells), kernels.shape[0], kernels.shape[0]))
    lnk = np.log(k)
    for a, ell in enumerate(ells):
        d = delta_l(ell, chi, kernels, k)
        integrand = (k**3 * pk)[None, None, :] * d[:, None, :] * d[None, :, :]
        out[a] = 2.0 / np.pi * simpson(integrand, x=lnk, axis=-1)
    return out


def cl_limber_linear(cosmo, ells, chi, kernels):
    """Limber counterpart of cl_exact_linear on the same kernels."""
    ells = np.asarray(ells, dtype=float)
    kk = (ells[:, None] + 0.5) / chi[None, :]
    pk = cosmo.P_lin(kk, np.zeros_like(kk))
    integrand = (kernels[None, :, None, :] * kernels[None, None, :, :]
                 * (pk / chi[None, :] ** 2)[:, None, None, :])
    return simpson(integrand, x=chi, axis=-1)
