"""Selected halo integrals, independent of a chosen mass-function fit.

Closed polynomial integrals check profile powers, bias and selection
normalization. Independent NumPy contractions exercise general signed
profiles. These are algebra and units tests, not survey calibration.
"""

from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.special import roots_legendre

project = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project/'interface'))
import cosmolike_des_cluster_interface as ci


def reference(weight, bias, profile):
    """Evaluate each mass integral by independent NumPy tensor sums."""
    na, nselection, nmass = weight.shape
    nk = profile.shape[1]
    first, second = np.triu_indices(n=nk)
    # a=state, s=selection, m=mass, k=profile and p=triangular pair.
    density = np.stack((weight.sum(axis=2),
                        np.einsum('asm,am->as', weight, bias)))
    single = np.stack((np.einsum('asm,akm->ask', weight, profile),
                       np.einsum('asm,am,akm->ask', weight, bias, profile)))
    product = profile[:, first]*profile[:, second]
    pair = np.stack((
        np.einsum('asm,apm->asp', weight, product),
        np.einsum('asm,apm,apm->asp', weight, product, profile[:, first]),
        np.einsum('asm,apm,apm->asp', weight, product, profile[:, second]),
    ))
    return {
        'density': density[0],
        'biased_density': density[1],
        'J01': single[0],
        'J11': single[1],
        'J02': pair[0],
        'J03_KKQ': pair[1],
        'J03_KQQ': pair[2],
    }


def test_closed_polynomial_integrals_and_selection_partition():
    """A probabilistic bin enters once, not once per same-halo leg."""
    nodes, measure = roots_legendre(n=8)
    mass = 1.5+0.5*nodes
    measure = 0.5*measure
    fraction = np.array([0.25, 0.75])
    amplitude = np.array([1.0, 2.0, -0.5])
    values = {
        'weight': np.ascontiguousarray(fraction[None, :, None]*measure),
        'bias': np.ascontiguousarray(mass[None, :]),
        'profile': np.ascontiguousarray(amplitude[None, :, None]*mass),
    }
    actual = ci.covariance_cluster_moments(**values)
    # Integrals over x in [1,2]: int 1=1, int x=3/2, int x^2=7/3,
    # int x^3=15/4. Here b=x and p_k=amplitude_k*x.
    expected_density = np.stack((fraction, 1.5*fraction))[:, None, :]
    expected_single = np.stack((
        1.5*fraction[:, None]*amplitude,
        (7.0/3.0)*fraction[:, None]*amplitude,
    ))[:, None, :, :]
    first, second = np.triu_indices(n=3)
    product = amplitude[first]*amplitude[second]
    expected_pair = np.stack((
        (7.0/3.0)*fraction[:, None]*product,
        (15.0/4.0)*fraction[:, None]*product*amplitude[first],
        (15.0/4.0)*fraction[:, None]*product*amplitude[second],
    ))[:, None, :, :]
    for key, expected in (
        ('density', expected_density[0]),
        ('biased_density', expected_density[1]),
        ('J01', expected_single[0]),
        ('J11', expected_single[1]),
        ('J02', expected_pair[0]),
        ('J03_KKQ', expected_pair[1]),
        ('J03_KQQ', expected_pair[2]),
    ):
        np.testing.assert_allclose(actual[key], expected, rtol=2.e-14)

    # A partition of observed categories recovers the unselected integral.
    # The true-mass ranges overlap entirely; that does not square S_i.
    united = dict(values)
    united['weight'] = np.ascontiguousarray(measure[None, None, :])
    union_result = ci.covariance_cluster_moments(**united)
    for key in actual:
        np.testing.assert_allclose(
            actual[key].sum(axis=1, keepdims=True), union_result[key],
            rtol=2.e-14,
        )


@pytest.mark.parametrize('nk,nmass', [(1, 1), (2, 6), (3, 9), (5, 257)])
def test_signed_profiles_threads_units_and_ownership(nk, nmass):
    """Odd/even tails preserve each ordered integral at 1/2/4/8 workers."""
    rng = np.random.default_rng(seed=1017)
    values = {
        'weight': rng.uniform(0.01, 0.1, size=(2, 3, nmass)),
        'bias': rng.uniform(0.3, 2.5, size=(2, nmass)),
        'profile': rng.uniform(-0.5, 2.0, size=(2, nk, nmass)),
    }
    expected = reference(**values)
    baseline = None
    for threads in (1, 2, 4, 8):
        ci.set_omp_threads(n=threads)
        actual = ci.covariance_cluster_moments(**values)
        for key in actual:
            np.testing.assert_allclose(
                actual[key], expected[key], rtol=2.e-14, atol=1.e-14,
            )
        if baseline is None:
            baseline = actual
        for key in actual:
            np.testing.assert_array_equal(
                actual[key].view(np.uint64), baseline[key].view(np.uint64),
            )

    # A length conversion sends dn -> dn/f^3 and M/rho -> f^3 M/rho.
    # Abundances, one-, two- and three-profile moments scale differently.
    factor = 3000.0
    converted = dict(values)
    converted['weight'] = values['weight']/factor**3
    converted['profile'] = values['profile']*factor**3
    changed = ci.covariance_cluster_moments(**converted)
    for key in ('density', 'biased_density'):
        np.testing.assert_allclose(changed[key]*factor**3, baseline[key],
                                   rtol=2.e-14)
    for key in ('J01', 'J11'):
        np.testing.assert_allclose(changed[key], baseline[key],
                                   rtol=2.e-14, atol=1.e-14)
    for key, power in (('J02', 3), ('J03_KKQ', 6), ('J03_KQQ', 6)):
        np.testing.assert_allclose(changed[key]/factor**power, baseline[key],
                                   rtol=2.e-14, atol=1.e-14)
    saved = baseline['J02'].copy()
    values['profile'] *= 2.0
    ci.covariance_cluster_moments(**values)
    np.testing.assert_array_equal(baseline['J02'], saved)



def test_empty_selection_and_input_guards():
    """An empty population is zero; malformed inputs fail before C starts."""
    values = {
        'weight': np.zeros(shape=(1, 2, 3)),
        'bias': np.ones(shape=(1, 3)),
        'profile': np.ones(shape=(1, 2, 3)),
    }
    for value in ci.covariance_cluster_moments(**values).values():
        np.testing.assert_array_equal(value, np.zeros_like(value))
    for key, bad in (
        ('weight', -np.ones(shape=(1, 2, 3))),
        ('weight', np.zeros(shape=(1, 0, 3))),
        ('weight', np.zeros(shape=(2, 3))),
        ('bias', np.ones(shape=(1, 2))),
        ('bias', np.full(shape=(1, 3), fill_value=np.nan)),
        ('profile', np.ones(shape=(2, 2, 3))),
        ('profile', np.ones(shape=(1, 2, 4))),
        ('profile', np.full(shape=(1, 2, 3), fill_value=np.inf)),
    ):
        malformed = dict(values)
        malformed[key] = bad
        with pytest.raises(ValueError):
            ci.covariance_cluster_moments(**malformed)


@pytest.mark.parametrize('layout', ['fortran', 'sliced', 'readonly'])
def test_named_armadillo_moments_preserve_notebook_inputs(layout):
    """Named matrices/cubes retain physical axes under notebook conversions."""
    rng = np.random.default_rng(seed=762)
    values = {
        'weight': rng.uniform(0.1, 0.4, size=(2, 3, 5)),
        'bias': rng.uniform(0.5, 2.0, size=(2, 5)),
        'profile': rng.normal(size=(2, 4, 5)),
    }
    expected = reference(**values)
    for key, array in values.items():
        if layout == 'fortran':
            values[key] = np.array(array, order='F', copy=True)
        elif layout == 'sliced':
            shape = tuple(2*size for size in array.shape)
            parent = np.zeros(shape=shape)
            selection = tuple(slice(None, None, 2) for size in array.shape)
            view = parent[selection]
            view[...] = array
            values[key] = view
        else:
            array.setflags(write=False)
    saved = {}
    for key, array in values.items():
        saved[key] = array.copy()
    actual = ci.covariance_cluster_moments(**values)
    for key, result in actual.items():
        np.testing.assert_allclose(result, expected[key], rtol=2.e-14)
        assert result.ndim <= 3
    for key, array in values.items():
        np.testing.assert_array_equal(array, saved[key])
