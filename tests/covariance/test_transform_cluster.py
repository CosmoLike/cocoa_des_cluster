"""Cluster-lensing localization on complete joint covariance blocks.

Analytic angular polynomials check the mean transformation. A dense joint
matrix supplies an independent propagation reference, including count cross
blocks, repeated observable rows and the known final-bin null modes.
"""

from pathlib import Path
import sys

import numpy as np
import pytest

project = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project/'interface'))
sys.path.insert(0, str(project.parents[1]/'external_modules/code/cosmolike_core'))
import cosmolike_des_cluster_interface as ci
from cosmolike_notebook_utils.covariance.transform_cluster import localize_covariance


def angular_operator(ntheta):
    """Read the mean model's T on logarithmic bins spanning a factor 100."""
    ci.init_binning(ntheta_bins=ntheta, theta_min_arcmin=2.5,
                    theta_max_arcmin=250.0)
    return np.asarray(a=ci.get_cluster_ytransform_matrix())


@pytest.mark.parametrize('ntheta', [5, 7, 20])
def test_localization_of_angular_polynomials(ntheta):
    """For gamma=A+B*x, Y integrates 2*gamma+d(gamma)/dx exactly."""
    transform = angular_operator(ntheta=ntheta)
    step = np.log(100.0)/ntheta
    coordinate = np.arange(ntheta)*step
    outer = coordinate[-1]
    gamma = 0.3-0.2*coordinate
    expected = (0.6-0.2)*(outer-coordinate)-0.2*(outer**2-coordinate**2)
    np.testing.assert_allclose(transform @ gamma, expected,
                               rtol=2.e-14, atol=2.e-15)
    np.testing.assert_array_equal(transform[-1], np.zeros(shape=ntheta))


@pytest.mark.parametrize('ntheta', [5, 20])
def test_all_cross_blocks_against_dense_transformation(ntheta):
    """Every transformed row sees the count and other-lensing fluctuations."""
    transform = angular_operator(ntheta=ntheta)
    ndata = 2*ntheta+3
    # Three untransformed entries separate two cluster-lensing rows. They
    # can be counts or other probes; neither is assumed to lie at the end.
    positions = np.array([np.arange(ntheta), np.arange(ntheta+3, ndata)])
    untouched = np.arange(ntheta, ntheta+3)
    random = np.random.default_rng(seed=701)
    samples = random.normal(size=(ndata, ndata+5))
    covariance = samples @ samples.T+np.eye(ndata)
    original = covariance.copy()
    joint = np.eye(ndata)
    for row in positions:
        joint[np.ix_(row, row)] = transform
    expected = joint @ covariance @ joint.T
    baseline = None

    for threads in (1, 2, 4, 8):
        ci.set_omp_threads(n=threads)
        actual = localize_covariance(interface=ci, covariance=covariance,
                                     indices=positions, operator=transform)
        np.testing.assert_allclose(actual, expected, rtol=5.e-13, atol=5.e-13)
        np.testing.assert_array_equal(actual[np.ix_(untouched, untouched)],
                                      covariance[np.ix_(untouched, untouched)])
        np.testing.assert_array_equal(actual[positions[:, -1]], 0.0)
        np.testing.assert_array_equal(actual[:, positions[:, -1]], 0.0)
        if baseline is None:
            baseline = actual
        np.testing.assert_array_equal(actual.view(np.uint64),
                                      baseline.view(np.uint64))

    np.testing.assert_array_equal(covariance, original)
    # Removing the two deterministic zero rows leaves an invertible
    # covariance. No eigenvalue has been clipped to obtain this result.
    retained = np.delete(arr=np.arange(ndata), obj=positions[:, -1])
    np.linalg.cholesky(actual[np.ix_(retained, retained)])


def test_signed_components_and_invalid_inputs():
    """Linear propagation preserves signed components and rejects bad maps."""
    positions = np.array([[0, 1], [3, 4]])
    operator = np.array([[1.0, -2.0], [0.0, 0.0]])
    covariance = np.arange(25, dtype=float).reshape(5, 5)-12.0
    covariance += covariance.T.copy()
    values = {
        'interface': ci,
        'covariance': covariance,
        'indices': positions,
        'operator': operator,
    }
    result = localize_covariance(**values)
    opposite = dict(values)
    opposite['covariance'] = -covariance
    np.testing.assert_array_equal(localize_covariance(**opposite), -result)

    for bad_indices in (positions.astype(float), [[0, 1], [1, 2]],
                        [[0, 1], [3, 5]], [[-1, 1], [3, 4]], [], [0, 1]):
        bad = dict(values)
        bad['indices'] = bad_indices
        with pytest.raises(ValueError):
            localize_covariance(**bad)
    for key, replacement in (('covariance', np.ones(shape=(5, 4))),
                             ('covariance', np.full(shape=(5, 5), fill_value=np.nan)),
                             ('operator', np.ones(shape=(3, 3))),
                             ('operator', np.full(shape=(2, 2), fill_value=np.inf))):
        bad = dict(values)
        bad[key] = replacement
        with pytest.raises(ValueError):
            localize_covariance(**bad)
