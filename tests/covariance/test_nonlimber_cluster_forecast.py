"""Check the non-Limber Gaussian spectra of the joint cluster forecast.

Below nonlimber_lmax the joint forecast (des_cluster_joint_covariance.py,
default gaussian = {"nonlimber": True, "rsd": True, "ia": "none"}) adds to
the Limber spectra of every pair with a galaxy or cluster leg the
correction of Fang et al. (arXiv:1911.11947),

  Delta C_AB(l) = C_AB^exact[P_lin] - C_AB^Limber[P_lin],

on the separable linear field D(a) delta(k,1). Each density field's exact
transfer is

  F_A(l,k) = integral dchi W_A [b_A D j_l(k chi) - f D j_l''(k chi)],

with the second (Kaiser, redshift-space) term only when rsd is on; the
compiled code computes it with FFTLog (covariance_cluster_nonlimber,
cosmolike_core covariances/nonlimber_cluster_cov.c). The checks:

  1. A cluster auto-spectrum correction, with and without RSD, against
     an independent sum: spherical Bessel functions from scipy on the
     Gauss-Legendre nodes of the Limber snapshot, Simpson in ln k, the
     cluster window built from the catalog readers (phi_cluster,
     ncl_richness, bcl_richness) as survey_cluster.selected_windows does.
  2. The galaxy and source rows without RSD equal the correction of the
     galaxy forecast path (covariance_limber_spectra with nonlimber_lmax,
     apply_nonlimber_cov) to round-off: inserting the clusters leaves the
     galaxy fields unchanged.
  3. In the joint forecast the Gaussian model changes G only: SSC, cNG and
     the counts block are bitwise those of the Limber forecast, and the
     non-Limber total is positive definite.

Run from the cocoa/Cocoa folder: python -m pytest
projects/des_cluster/tests/covariance/test_nonlimber_cluster_forecast.py
"""

from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.integrate import simpson
from scipy.special import spherical_jn

project = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project/'interface'))
sys.path.insert(0, str(project/'covariance'))
sys.path.insert(0, str(project.parents[1]/'external_modules/code/cosmolike_core'))
import cosmolike_des_cluster_interface as ci
import des_cluster_joint_covariance as survey

# check 1: multipoles, and the largest |FFTLog - direct| relative to the
# direct exact spectrum (measured 2026-11-04: 2.6e-6 without RSD, 1.5e-5
# with it)
DIRECT_ELLS = (2, 8, 30)
DIRECT_RTOL = 1.0e-4
# the direct sum's radial rule and largest wavenumber (h/Mpc)
DIRECT_NQUAD = 512
DIRECT_KMAX = 0.3
# check 2: round-off bound, relative to the largest |correction| of a row
# (measured 7.9e-16)
GALAXY_ATOL = 1.0e-12
# small grids shared by the checks (as test_joint_forecast.py)
SMALL = {
    'theta_edges_arcmin': np.geomspace(2.5, 250.0, 6),
    'ell_max': 128,
    'mask_ell_max': 128,
    'ng_ell': np.geomspace(2.5, 128.5, 4)-0.5,
    'angle_nquad': 64,
    'halo_mass_nquad': 64,
    'nwindow': 1025,
}


@pytest.fixture(scope='module')
def settings():
    """The joint forecast state on small grids (CAMB runs once).

    Returns:
      the resolved settings of des_cluster_joint_covariance.configuration
      with the SMALL grids, after survey.initialize.

    Side effects:
      replaces the cosmology, catalogs and thread count of the library.
    """
    resolved = survey.configuration(accuracy_boost=1, ytransform=False)
    resolved.update(SMALL)
    survey.initialize(interface=ci, settings=resolved)
    ci.set_omp_threads(n=4)
    return resolved


def correction(settings, lmax, rsd):
    """covariance_cluster_nonlimber on the settings' radial rule.

    Arguments:
      settings = the module fixture.
      lmax = last corrected multipole.
      rsd = include the Kaiser term.

    Returns:
      the binding's dict (ell, correction [nell,nfield,nfield]).
    """
    return ci.covariance_cluster_nonlimber(
        a_edges=np.asarray(settings['a_edges'], dtype=float),
        nquad=settings['radial_nquad'], nwindow=settings['nwindow'],
        include_rsd=rsd, nonlimber_lmax=lmax,
        nonlimber_nchi=settings['nonlimber_nchi'],
    )


def direct_cluster_correction(settings, ells, rsd):
    """Exact minus matched linear spectrum of cluster category 0, directly.

    The window q = f_K^2 phi_0 n_0/nbar times b_0 (and q f for the Kaiser
    term) on the Gauss-Legendre nodes of a Limber snapshot with
    DIRECT_NQUAD nodes per panel (finer than the forecast's), whose dchi
    weights also give nbar. D(a) and P_lin(k,1) are the library's readers
    (growfac, p_lin at a = 1); f = dln D/dln a by central differences.
    The k integral stops at DIRECT_KMAX: the selection kernels are smooth
    on the photo-z scale (about 20 Mpc/h), so the transfers vanish beyond
    it, and the 512-node rule still resolves j_l(k chi) there.

    Arguments:
      settings = the module fixture.
      ells = integer multipoles.
      rsd = include the Kaiser term.

    Returns:
      numpy array [len(ells)] in the core harmonic convention.
    """
    snapshot = ci.covariance_limber_spectra(
        ell=np.array([2.0]), a_edges=np.asarray(settings['a_edges'], dtype=float),
        nquad=DIRECT_NQUAD, nwindow=settings['nwindow'],
        include_ia=False, include_rsd=False, linear=False,
    )
    a, unused, fk, dchi = np.asarray(snapshot['geometry'])
    nrichness = len(settings['cluster_richness_edges'])-1
    density = (np.asarray(ci.covariance.phi_cluster(z=np.ascontiguousarray(1.0/a-1.0)))[:, 0]
               *np.asarray(ci.covariance.ncl_richness(a=np.ascontiguousarray(a)))[:, 0])
    bias = np.asarray(ci.covariance.bcl_richness(a=np.ascontiguousarray(a)))[:, 0]
    assert nrichness >= 1
    q = fk**2*density/np.sum(fk**2*density*dchi)
    keep = q > 0.0
    a, fk, dchi, q, bias = a[keep], fk[keep], dchi[keep], q[keep], bias[keep]
    growth = np.array([ci.growfac(a=x) for x in a])
    step = 1.0e-4
    rate = np.array([(np.log(ci.growfac(a=x*(1+step)))-np.log(ci.growfac(a=x*(1-step))))
                     /(np.log1p(step)-np.log1p(-step)) for x in a])
    # k in c/H0 units (c/H0 = 2997.92458 Mpc/h)
    k = np.geomspace(1.0e-4, DIRECT_KMAX, 3001)*2997.92458
    power = np.array([ci.p_lin(k=x, a=1.0) for x in k])
    result = []
    for ell in ells:
        x = np.outer(k, fk)
        j = spherical_jn(ell, x)
        transfer = (j*(dchi*q*bias*growth)).sum(axis=1)
        if rsd:
            jp = spherical_jn(ell-1, x)-(ell+1.0)*j/x
            j2 = -(2.0/x)*jp-(1.0-ell*(ell+1.0)/x**2)*j
            transfer -= (j2*(dchi*q*rate*growth)).sum(axis=1)
        exact = 2.0/np.pi*simpson(k**3*power*transfer**2, x=np.log(k))
        limber_k = (ell+0.5)/fk
        limber_power = np.array([ci.p_lin(k=x, a=1.0) for x in limber_k])
        matched = np.sum(dchi*(q*bias*growth)**2*limber_power/fk**2)
        result.append((exact-matched, exact))
    return np.array(result)


def test_cluster_correction_against_direct_bessel(settings):
    """FFTLog cluster correction = direct Bessel sum, with and without RSD.

    Arguments:
      settings = the module fixture.
    """
    nlens = len(settings['lens_density_arcmin2'])
    for rsd in (False, True):
        fftlog = correction(settings, max(DIRECT_ELLS), rsd)['correction']
        direct = direct_cluster_correction(settings, DIRECT_ELLS, rsd)
        for (ell, (delta, exact)) in zip(DIRECT_ELLS, direct):
            value = fftlog[ell-2, nlens, nlens]
            print(f"\n  rsd={rsd} l={ell}: FFTLog {value:.6e}, direct {delta:.6e}, "
                  f"exact {exact:.6e}, |difference|/exact {abs(value-delta)/exact:.2e}")
            assert abs(value-delta) < DIRECT_RTOL*abs(exact)


def test_galaxy_rows_match_galaxy_path(settings):
    """Galaxy and source rows equal the galaxy forecast's correction.

    Arguments:
      settings = the module fixture.
    """
    lmax = 40
    ell = np.arange(2, lmax+1, dtype=float)
    common = dict(ell=ell, a_edges=np.asarray(settings['a_edges'], dtype=float),
                  nquad=settings['radial_nquad'], nwindow=settings['nwindow'],
                  include_ia=False, include_rsd=False, linear=False)
    limber = np.asarray(ci.covariance_limber_spectra(**common)['spectra'])
    hybrid = np.asarray(ci.covariance_limber_spectra(
        **common, nonlimber_lmax=lmax, nonlimber_nchi=settings['nonlimber_nchi'],
    )['spectra'])
    galaxy = hybrid-limber
    joint = np.asarray(correction(settings, lmax, False)['correction'])
    nlens = len(settings['lens_density_arcmin2'])
    ncount = joint.shape[1]-galaxy.shape[1]
    keep = np.concatenate((np.arange(nlens), np.arange(nlens+ncount, joint.shape[1])))
    joint = joint[:, keep[:, None], keep]
    scale = np.max(np.abs(galaxy), axis=0)+1e-300
    error = float(np.max(np.abs(joint-galaxy)/scale))
    print(f"\n  galaxy rows: max |joint - galaxy|/max|correction| = {error:.2e}")
    assert error < GALAXY_ATOL
    nsource = len(settings['source_density_arcmin2'])
    assert np.all(joint[:, -nsource:, -nsource:] == 0.0)


def test_gaussian_model_changes_only_gaussian(settings):
    """SSC, cNG and counts are those of the Limber forecast; G changes.

    Arguments:
      settings = the module fixture.
    """
    nonlimber = survey.compute(interface=ci, settings=settings)
    limber_settings = dict(settings)
    limber_settings['gaussian'] = dict(settings['gaussian'], nonlimber=False, rsd=False)
    limber = survey.compute(interface=ci, settings=limber_settings)
    for name in ('ssc', 'cng', 'mean_counts'):
        np.testing.assert_array_equal(nonlimber[name], limber[name])
    counts = nonlimber['count_positions']
    np.testing.assert_array_equal(nonlimber['gaussian'][np.ix_(counts, counts)],
                                  limber['gaussian'][np.ix_(counts, counts)])
    assert np.any(nonlimber['gaussian'] != limber['gaussian'])
    assert 'all-pairs non-Limber corrections' not in nonlimber['settings']['omitted_terms']
    assert 'all-pairs non-Limber corrections' in limber['settings']['omitted_terms']
    total = nonlimber['total']
    scale = np.sqrt(np.diag(total))
    np.linalg.cholesky((total/scale[:, None])/scale[None, :])
