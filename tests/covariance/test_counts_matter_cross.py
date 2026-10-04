"""Non-SSC count-matter projection: area cancellation, units and pair factors."""

from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.special import roots_legendre
from scipy.stats import poisson

project = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project/'interface'))
sys.path.insert(0, str(project.parents[1]/'external_modules/code/cosmolike_core'))
import cosmolike_des_cluster_interface as ci
from cosmolike_notebook_utils.covariance.counts_cluster import count_matter_cross


def inputs():
    """Supply polynomial shell moments over chi in [1,2]."""
    nodes, weights = roots_legendre(n=8)
    distance = 1.5+0.5*nodes
    nk = 3
    selected = np.array([0.2, 0.7])
    first, second = np.triu_indices(n=nk)
    single = np.zeros(shape=(2, 8, 2, nk))
    pair = np.zeros(shape=(3, 8, 2, len(first)))
    single[1] = distance[:, None, None]**2*selected[None, :, None]
    # A selected covariance kernel proportional to chi^2 cancels the
    # projection's chi^-2, leaving a constant integrand with known area.
    diagonal = np.flatnonzero(first == second)
    for mode, index in enumerate(diagonal):
        pair[0, :, :, index] = distance[:, None]**2*selected*(mode+1.0)
    return {
        'interface': ci,
        'distance': distance,
        'dchi': 0.5*weights,
        'pair_window': np.repeat(np.array([[0.4], [1.1]]), 8, axis=1),
        'transfer': np.array([[1., 1., 1.], [0.5, 0.9, 0.99]]),
        'linear_power': np.repeat(np.array([[0.03, 0.02, 0.01]]), 8, axis=0),
        'i11': np.full(shape=(8, nk), fill_value=0.7),
        'moments': {'single': single, 'pair': pair},
    }


def test_closed_projection_and_thread_repeatability():
    """Both halo terms include their own factors and shared field transfers."""
    values = inputs()
    selected = np.array([0.2, 0.7])
    window = np.array([0.4, 1.1])
    common = selected[:, None, None]*window[None, :, None]*values['transfer']
    expected_one = common*np.array([1., 2., 3.])
    expected_two = common*1.4*np.array([0.03, 0.02, 0.01])
    baseline = None
    for threads in (1, 2, 4, 8):
        ci.set_omp_threads(n=threads)
        actual = count_matter_cross(**values)
        np.testing.assert_allclose(actual['one_halo'], expected_one, rtol=2.e-14)
        np.testing.assert_allclose(actual['two_halo'], expected_two, rtol=2.e-14)
        np.testing.assert_allclose(actual['total'], expected_one+expected_two,
                                   rtol=2.e-14)
        if baseline is None:
            baseline = actual
        for key in actual:
            np.testing.assert_array_equal(actual[key].view(np.uint64),
                                          baseline[key].view(np.uint64))


def test_length_units_cancel():
    """The projected count-spectrum covariance is independent of length units."""
    values = inputs()
    baseline = count_matter_cross(**values)
    factor = 3000.0
    converted = dict(values)
    converted['distance'] = values['distance']*factor
    converted['dchi'] = values['dchi']*factor
    converted['pair_window'] = values['pair_window']/factor**2
    converted['linear_power'] = values['linear_power']*factor**3
    pair = values['moments']['pair'].copy()
    pair[0] *= factor**3
    pair[1:] *= factor**6
    converted['moments'] = {'single': values['moments']['single'], 'pair': pair}
    changed = count_matter_cross(**converted)
    for key in changed:
        np.testing.assert_allclose(changed[key], baseline[key], rtol=2.e-14)

def test_exact_poisson_counts_and_different_halo_pairs():
    """Enumerate a toy halo population, without assuming the analytic kernel."""
    density = np.array([1.3, 2.1])
    selection = np.array([0.3, 0.0])
    profile = np.array([1.4, 0.4])
    bias = np.array([2.1, 1.2])
    biased_profile = bias*profile
    power = 0.02
    moments = ci.covariance_cluster_moments(
        weight=np.ascontiguousarray((density*selection)[None, None, :]),
        bias=np.ascontiguousarray(bias[None, :]),
        profile=np.ascontiguousarray(profile[None, None, :]),
    )
    actual = count_matter_cross(
        interface=ci, distance=np.ones(shape=1), dchi=np.ones(shape=1),
        pair_window=np.ones(shape=(1, 1)), transfer=np.ones(shape=(1, 1)),
        linear_power=np.array([[power]]),
        i11=np.array([[np.dot(density, biased_profile)]]), moments=moments,
    )

    # Poisson thinning makes three independent counts: detected halos in
    # mass bin 1, missed halos in that bin, and all halos in mass bin 2.
    # Enumerate their probabilities, rather than drawing a noisy simulation.
    count = np.arange(45)
    detected = count[:, None, None]
    missed = count[None, :, None]
    other = count[None, None, :]
    first = detected+missed
    for volume in (0.7, 2.3):
        probabilities = (
            poisson.pmf(k=detected, mu=volume*density[0]*selection[0])
            *poisson.pmf(k=missed, mu=volume*density[0]*(1-selection[0]))
            *poisson.pmf(k=other, mu=volume*density[1])
        )
        np.testing.assert_allclose(probabilities.sum(), 1.0, rtol=2.e-14)
        mean_count = np.sum(probabilities*detected)
        one = (first*profile[0]**2+other*profile[1]**2)/volume

        # Distinct-halo pairs use N(N-1) for identical mass bins, and N1*N2
        # for different bins. This excludes pairing an object with itself.
        pairs = (first*(first-1)*biased_profile[0]**2
                 +2*first*other*biased_profile[0]*biased_profile[1]
                 +other*(other-1)*biased_profile[1]**2)
        two = power*pairs/volume**2
        for name, estimator in (('one_halo', one), ('two_halo', two)):
            mean_power = np.sum(probabilities*estimator)
            cross = np.sum(probabilities*detected*estimator)-mean_count*mean_power
            np.testing.assert_allclose(actual[name][0, 0, 0], cross,
                                       rtol=3.e-13)


def test_bad_projection_inputs():
    """Reject malformed shell grids and moment axes before integration."""
    values = inputs()
    for key, bad in (
        ('distance', np.zeros(shape=8)),
        ('dchi', np.ones(shape=7)),
        ('i11', np.ones(shape=(8, 2))),
        ('linear_power', -np.ones(shape=(8, 3))),
        ('transfer', np.ones(shape=(2, 2))),
        ('pair_window', np.full(shape=(2, 8), fill_value=np.nan)),
        ('moments', {'single': np.ones(shape=(2, 8, 2, 3)),
                     'pair': np.ones(shape=(3, 8, 2, 5))}),
    ):
        malformed = dict(values)
        malformed[key] = bad
        with pytest.raises(ValueError):
            count_matter_cross(**malformed)
