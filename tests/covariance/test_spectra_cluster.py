"""Independent all-pairs cluster projection checks.

The compiled function covariance_cluster_spectra returns the Limber angular
spectra the Gaussian covariance needs for every pair of fields, from
tables supplied on radial nodes chi (transverse distance f_K):

  cluster x galaxy or source X:
    C_cX(l) = integral dchi/f_K^2 q_c W_X [b_c P(k) + P_1h,c(k)]
  cluster x cluster:
    C_cc'(l) = integral dchi/f_K^2 q_c b_c q_c' b_c' P(k)

with k = (l + 1/2)/f_K, q_c the normalized radial window of cluster
category c, b_c its selected bias, W_X a galaxy or lensing window, P the
nonlinear matter power and P_1h,c the one-halo cluster-matter power of the
selected halos. The one-halo term enters only for the lensing (source)
fields, and source spectra carry the spin factor of spin_factor below.

The supplied-table boundary projects cc/cg from biased matter power and
adds the selected halo profile only to cluster lensing. Polynomial examples
check those distinct terms with closed integrals; general supplied tables
check radial contraction, thread repeatability, units and output ownership.
These tests do not calibrate the survey's physical cluster model.

Run from the cocoa/Cocoa folder: python -m pytest
projects/des_cluster/tests/covariance/test_spectra_cluster.py
"""

from pathlib import Path
import sys

import numpy as np
from abort_check import assert_aborts
import pytest
from scipy.special import roots_legendre

project = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project/'interface'))
import cosmolike_des_cluster_interface as ci


def inputs(nnode=9):
    """Supply two cluster populations, two galaxies and one source field.

    Gauss-Legendre nodes on chi in [0.2, 0.8]; multipoles 2, 31.5 and 500;
    constant base windows 0.2 and 0.5 (the two galaxy fields, nlens = 2)
    and 0.8 (the source field); cluster windows q = 1/0.6, normalized over
    the interval width 0.6; biases 1.3 and 2.1; zero power and profile,
    which the tests fill. richness = [1, 0]: cluster 0 uses profile 1 and
    cluster 1 uses profile 0, so a swapped index would show.

    Arguments:
      nnode = number of radial nodes, a positive integer.

    Returns:
      a dict of the keyword arguments of covariance_cluster_spectra.
    """
    nodes, weights = roots_legendre(n=nnode)
    distance = 0.5+0.3*nodes
    ell = np.array([2.0, 31.5, 500.0])
    ncluster = 2
    return {
        'ell': ell,
        'distance': distance,
        'dchi': 0.3*weights,
        'base': np.repeat(np.array([[0.2], [0.5], [0.8]]), nnode, axis=1),
        'window': np.full((ncluster, nnode), 1.0/0.6),
        'bias': np.repeat(np.array([[1.3], [2.1]]), nnode, axis=1),
        'power': np.zeros((len(ell), nnode)),
        'profile': np.zeros((2, len(ell), nnode)),
        'richness': np.array([1, 0], dtype=np.int32),
        'nlens': 2,
    }


def spin_factor(ell):
    """Return the core harmonic source convention, separate from noise.

    sqrt((l - 1) l (l + 1) (l + 2))/(l + 1/2)^2 converts the Limber
    spectrum of a lensing convergence into that of the shear (spin-2)
    field, as the core's source spectra do; it tends to 1 at large l.

    Arguments:
      ell = multipoles, float array.

    Returns:
      the factor at each multipole, same shape as ell.
    """
    return np.sqrt((ell-1)*ell*(ell+1)*(ell+2))/(ell+0.5)**2


def test_one_halo_only_affects_cluster_lensing():
    """With P=0, a profile proportional to chi^2 integrates in closed form.

    P_1h = A chi^2 cancels 1/f_K^2, and q integrates to 1, so the cluster x
    source spectrum is 0.8 A times the spin factor; galaxy and
    cluster-cluster spectra stay zero. Multiplying the biases by 7 must
    change nothing: the bias multiplies the matter field around a halo, not
    the halo's own mass profile.

    Arguments:
      none.

    Returns:
      nothing; a failed assertion fails the test.
    """
    values = inputs()
    amplitude = np.array([[0.03, 0.06, 0.02], [0.08, 0.09, 0.07]])
    values['profile'] = amplitude[:, :, None]*values['distance']**2
    actual = ci.covariance_cluster_spectra(**values)

    # q integrates to one and P1h/chi^2 is constant. Only the source
    # window survives; bias must not multiply the halo's own mass profile.
    expected = np.zeros((3, 2, 3))
    for cluster, richness in enumerate(values['richness']):
        expected[:, cluster, 2] = (
            0.8*amplitude[richness]*spin_factor(ell=values['ell'])
        )
    np.testing.assert_allclose(actual['cluster_base'], expected, rtol=2.e-14)
    np.testing.assert_array_equal(actual['cluster_cluster'], np.zeros((3, 2, 2)))
    values['bias'] *= 7.0
    changed = ci.covariance_cluster_spectra(**values)
    np.testing.assert_array_equal(changed['cluster_base'], actual['cluster_base'])


def test_biased_power_closed_integral():
    """Without a profile, all cross pairs share the same biased matter field.

    P = A chi^2 cancels 1/f_K^2; the cross spectra become A b_c W_X (times
    the spin factor for the source field) and the cluster auto spectra
    A b_c b_c' / 0.6. rtol = 2e-14 allows rounding only (the quadrature is
    exact for these polynomials).

    Arguments:
      none.

    Returns:
      nothing; a failed assertion fails the test.
    """
    values = inputs()
    amplitude = np.array([0.03, 0.06, 0.02])
    values['power'] = amplitude[:, None]*values['distance']**2
    actual = ci.covariance_cluster_spectra(**values)
    bias = np.array([1.3, 2.1])
    base = np.array([0.2, 0.5, 0.8])
    expected_cross = amplitude[:, None, None]*bias[None, :, None]*base
    expected_cross[:, :, 2] *= spin_factor(ell=values['ell'])[:, None]

    # Both q factors are 1/0.6 and the interval width is 0.6. This
    # leaves 1/0.6 in cc; cg has only one normalized cluster window.
    expected_auto = amplitude[:, None, None]*np.outer(bias, bias)/0.6
    np.testing.assert_allclose(actual['cluster_base'], expected_cross, rtol=2.e-14)
    np.testing.assert_allclose(actual['cluster_cluster'], expected_auto, rtol=2.e-14)


@pytest.mark.parametrize('nnode', [1, 6, 9])
def test_units_threads_and_ownership(nnode):
    """General tables agree with NumPy, including odd pair and node tails.

    Random positive tables and signed profiles (fixed seed) are compared
    with matrix products written in NumPy (@ is the matrix product), and
    every output must repeat bit for bit at 1, 2, 4 and 8 OpenMP threads
    (.view(np.uint64) compares raw 64-bit patterns). A length rescaling by
    3000 (close to c/H0 in Mpc/h) must leave the dimensionless spectra
    unchanged, and a later call must not change an earlier result.

    Arguments:
      nnode = number of radial nodes, set by @pytest.mark.parametrize,
              which runs the test once per listed value.

    Returns:
      nothing; a failed assertion fails the test.

    Side effects:
      sets the OpenMP thread count of the compiled library.
    """
    values = inputs(nnode=nnode)
    rng = np.random.default_rng(seed=710)
    values['power'] = rng.uniform(0.01, 0.03, size=(3, nnode))
    values['profile'] = rng.uniform(-0.001, 0.004, size=(2, 3, nnode))
    values['base'] *= rng.uniform(0.5, 1.5, size=(3, nnode))
    values['bias'] *= rng.uniform(0.5, 1.5, size=(2, nnode))
    measure = values['dchi']/values['distance']**2
    qb = values['window']*values['bias']
    expected_cross = np.empty((3, 2, 3))
    expected_auto = np.empty((3, 2, 2))
    for index in range(3):
        expected_auto[index] = (qb*measure*values['power'][index]) @ qb.T
        expected_cross[index] = (qb*measure*values['power'][index]) @ values['base'].T
        for cluster, richness in enumerate(values['richness']):
            own = values['window'][cluster]*values['profile'][richness, index]
            expected_cross[index, cluster, 2] += np.sum(
                own*values['base'][2]*measure
            )
        expected_cross[index, :, 2] *= spin_factor(ell=values['ell'][index])

    baseline = None
    for threads in (1, 2, 4, 8):
        ci.set_omp_threads(n=threads)
        actual = ci.covariance_cluster_spectra(**values)
        np.testing.assert_allclose(actual['cluster_base'], expected_cross, rtol=2.e-14)
        np.testing.assert_allclose(actual['cluster_cluster'], expected_auto, rtol=2.e-14)
        if baseline is None:
            baseline = actual
        for key in actual:
            np.testing.assert_array_equal(
                actual[key].view(np.uint64), baseline[key].view(np.uint64)
            )

    # Changing the numerical length unit must cancel between dchi/f_K^2,
    # two windows and the volume dimension of each power spectrum.
    converted = dict(values)
    factor = 3000.0
    for key in ('distance', 'dchi'):
        converted[key] = values[key]*factor
    for key in ('base', 'window'):
        converted[key] = values[key]/factor
    for key in ('power', 'profile'):
        converted[key] = values[key]*factor**3
    actual = ci.covariance_cluster_spectra(**converted)
    for key in actual:
        np.testing.assert_allclose(actual[key], baseline[key], rtol=2.e-14)
    saved = baseline['cluster_base'].copy()
    values['profile'] *= 2.0
    ci.covariance_cluster_spectra(**values)
    np.testing.assert_array_equal(baseline['cluster_base'], saved)


def test_invalid_inputs_raise_before_c():
    """Malformed array shapes, domains and profile indices raise Python errors.

    Each case replaces one input of a copy of the valid inputs: a multipole
    below 2, a zero distance, negative radial weights, a power table with 2
    instead of 3 multipoles, a profile with 8 instead of 9 nodes, a NaN
    bias, a richness index past the two profiles, and nlens = 4 for 3 base
    fields. Each must be refused: assert_aborts runs the call in a
    forked child and requires a nonzero exit status
    (abort_check.py explains both refusal paths).

    Arguments:
      none.

    Returns:
      nothing; a failed assertion fails the test.
    """
    values = inputs()
    for key, bad in (
        ('ell', np.array([1.0, 2.0, 3.0])),
        ('distance', np.zeros(9)),
        ('dchi', -values['dchi']),
        ('power', np.zeros((2, 9))),
        ('profile', np.zeros((2, 3, 8))),
        ('bias', np.full((2, 9), np.nan)),
        ('richness', np.array([0, 2], dtype=np.int32)),
        ('nlens', 4),
    ):
        malformed = dict(values)
        malformed[key] = bad
        def attempt():
            ci.covariance_cluster_spectra(**malformed)
        assert_aborts(attempt)
