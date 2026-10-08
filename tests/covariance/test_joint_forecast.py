"""Check the full cluster forecast layout on inexpensive physical grids.

The joint 6x2pt + N forecast (covariance/des_cluster_joint_covariance.py)
returns the Gaussian (G), super-sample (SSC) and connected non-Gaussian
(cNG) parts of the covariance of the joint vector ss, gs, gg, cg, N, cc,
cs, and their total. With 5 angular bins instead of 20 the vector has
140 two-point rows x 5 bins + 12 counts = 712 entries.

These grids test assembly, normalization, archive contents and deterministic
threading. They do not establish numerical convergence for DES inference.
The independent full-resolution component comparison is separate.

Run from the cocoa/Cocoa folder: python -m pytest
projects/des_cluster/tests/covariance/test_joint_forecast.py
"""

import json
from pathlib import Path
import sys

import numpy as np
import pytest

project = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project/'interface'))
sys.path.insert(0, str(project/'covariance'))
sys.path.insert(0, str(project.parents[1]/'external_modules/code/cosmolike_core'))
import cosmolike_des_cluster_interface as ci
import des_cluster_joint_covariance as survey
from cosmolike_notebook_utils.covariance.forecast_cluster import save_forecast
from cosmolike_notebook_utils.covariance.transform_cluster import localize_covariance


def test_joint_model_rejects_partial_gaussian_extensions():
    """Galaxy-only physics must not silently label a joint cluster forecast.

    The joint forecast supports only Limber spectra without intrinsic
    alignment; asking for non-Limber spectra or NLA must raise ValueError
    with a message naming the joint selected-cluster forecast.

    Arguments:
      none.

    Returns:
      nothing; a failed assertion fails the test.
    """
    for gaussian in ({"nonlimber": True, "ia": "none"},
                     {"nonlimber": False, "ia": "NLA", "A1": 0.6}):
        with pytest.raises(ValueError, match="joint selected-cluster"):
            survey.configuration(gaussian=gaussian)


def test_joint_forecast_threads_localization_and_archive(tmp_path):
    """All components repeat and Y transforms every count and two-point cross.

    Steps, each a set of assertions:
      1. Without the Y localization, the notebook route and the
         command-line route (backend=ci.covariance) agree exactly, and
         every component repeats bit for bit at 1 and 8 OpenMP threads.
      2. The counts have Gaussian (Poisson) variance equal to their mean,
         zero Gaussian covariance with the two-point entries, all-zero cNG
         rows, and a nonzero SSC covariance with the two-point entries; the
         forecast's mean counts match the likelihood's N_cluster_tomo to
         2e-6 (computed along a different integration route).
      3. With the Y localization, every component equals
         localize_covariance applied to the untransformed one, bit for
         bit; the 48 last-bin rows are zero, leaving 712 - 48 = 664 valid
         entries; the mean vector transforms as T gamma_t; and the total
         restricted to the valid entries is positive definite (Cholesky of
         its correlation matrix).
      4. The saved .npz archive reads back without pickle (no executable
         Python objects) and records the settings and the omitted terms.

    Arguments:
      tmp_path = pytest's built-in fixture: a fresh temporary directory
                 (a pathlib.Path) for the archive.

    Returns:
      nothing; a failed assertion fails the test.

    Side effects:
      runs CAMB and replaces the state and OpenMP thread count of the
      compiled library.
    """
    settings = survey.configuration(accuracy_boost=1, ytransform=False)
    settings.update({
        'theta_edges_arcmin': np.geomspace(2.5, 250.0, 6),
        'ell_max': 128,
        'mask_ell_max': 128,
        'ng_ell': np.geomspace(2.5, 128.5, 4)-0.5,
        'angle_nquad': 64,
        'halo_mass_nquad': 64,
        'nwindow': 1025,
    })
    survey.initialize(interface=ci, settings=settings)
    baseline = None
    for threads in (1, 8):
        ci.set_omp_threads(n=threads)
        result = survey.compute(interface=ci, settings=settings)
        production = survey.compute(
            interface=ci, settings=settings, backend=ci.covariance,
        )
        for name in ('gaussian', 'ssc', 'cng', 'total', 'joint_signal', 'mean_counts'):
            np.testing.assert_array_equal(production[name], result[name])
        assert result['total'].shape == (712, 712)
        assert result['rows'].shape == (140, 3)
        assert len(result['count_positions']) == 12
        assert len(result['valid_indices']) == 712
        for name in ('gaussian', 'ssc', 'cng', 'total', 'joint_signal', 'mean_counts'):
            assert np.all(np.isfinite(result[name]))
            if baseline is not None:
                np.testing.assert_array_equal(result[name].view(np.uint64),
                                              baseline[name].view(np.uint64))
        baseline = result

    # The twelve counts have Poisson variance equal to their mean. All
    # count-two-point correlation is SSC under this forecast's stated model.
    counts = result['count_positions']
    spectra = result['two_point_positions']
    np.testing.assert_array_equal(result['gaussian'][np.ix_(counts, counts)],
                                  np.diag(result['mean_counts']))
    np.testing.assert_array_equal(result['gaussian'][np.ix_(counts, spectra)], 0.0)
    np.testing.assert_array_equal(result['cng'][counts], 0.0)
    assert np.any(result['ssc'][np.ix_(counts, spectra)] != 0.0)
    # N_cluster_tomo returns [richness, cluster z]; .T.ravel() lists it in
    # the vector order, cluster z bin then richness bin (richness fastest)
    np.testing.assert_allclose(result['mean_counts'],
                               np.asarray(ci.N_cluster_tomo()).T.ravel(), rtol=2.e-6)

    # Y is a change of statistic, not a mask or a covariance correction.
    # Compute it through the complete driver and independently on the raw
    # components, including count-lensing crosses and the mean vector.
    settings['cluster_ytransform'] = True
    localized = survey.compute(interface=ci, settings=settings)
    production = survey.compute(
        interface=ci, settings=settings, backend=ci.covariance,
    )
    for name in ('gaussian', 'ssc', 'cng', 'total', 'joint_signal'):
        np.testing.assert_array_equal(production[name], localized[name])
    positions = result['cluster_lensing_positions']
    operator = ci.get_cluster_ytransform_matrix()
    for name in ('gaussian', 'ssc', 'cng'):
        expected = localize_covariance(interface=ci, covariance=result[name],
                                        indices=positions, operator=operator)
        # rebuild the matrix from its upper triangle (diagonal included) and
        # the mirror of the strict upper triangle, so it is exactly symmetric
        expected = np.triu(expected)+np.triu(expected, k=1).T
        np.testing.assert_array_equal(localized[name].view(np.uint64),
                                      expected.view(np.uint64))
        np.testing.assert_array_equal(localized[name][positions[:, -1]], 0.0)
    expected_mean = result['joint_signal'][positions] @ operator.T
    np.testing.assert_allclose(localized['joint_signal'][positions], expected_mean,
                               rtol=1.e-13, atol=1.e-16)
    assert len(localized['valid_indices']) == 664
    for name in ('gaussian', 'ssc', 'cng', 'total'):
        np.testing.assert_array_equal(localized[name], localized[name].T)
    valid = localized['valid_indices']
    total = localized['total'][np.ix_(valid, valid)]
    scale = np.sqrt(np.diag(total))
    np.linalg.cholesky((total/scale[:, None])/scale[None, :])

    # The saved row positions and omitted physics must travel with the
    # matrix. Reading it should never require a pickle or today's defaults.
    filename = tmp_path/'joint_forecast.npz'
    save_forecast(result=localized, filename=filename)
    with np.load(filename, allow_pickle=False) as archive:
        for key, values in localized.items():
            if isinstance(values, np.ndarray):
                np.testing.assert_array_equal(archive[key], values)
        saved_settings = json.loads(str(archive['settings_json']))
        np.testing.assert_array_equal(saved_settings['ng_ell'], settings['ng_ell'])
        assert saved_settings['count_cross_model'] == 'SSC only'
        missing = saved_settings['omitted_terms']
        assert 'non-SSC count-spectrum cross covariance' in missing
        assert saved_settings['cluster_ytransform'] is True
