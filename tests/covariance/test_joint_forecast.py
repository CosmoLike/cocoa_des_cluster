"""Check the full cluster forecast layout on inexpensive physical grids.

These grids test assembly, normalization, archive contents and deterministic
threading. They do not establish numerical convergence for DES inference.
The independent full-resolution component comparison is separate.
"""

import json
from pathlib import Path
import sys

import numpy as np

project = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project/'interface'))
sys.path.insert(0, str(project/'covariance'))
sys.path.insert(0, str(project.parents[1]/'external_modules/code/cosmolike_core'))
import cosmolike_des_cluster_interface as ci
import des_cluster_joint_covariance as survey
from cosmolike_notebook_utils.covariance.forecast_cluster import save_forecast
from cosmolike_notebook_utils.covariance.transform_cluster import localize_covariance


def test_joint_forecast_threads_localization_and_archive(tmp_path):
    """All components repeat and Y transforms every count and two-point cross."""
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
    np.testing.assert_allclose(result['mean_counts'],
                               np.asarray(ci.N_cluster_tomo()).T.ravel(), rtol=2.e-6)

    # Y is a change of statistic, not a mask or a covariance correction.
    # Compute it through the complete driver and independently on the raw
    # components, including count-lensing crosses and the mean vector.
    settings['cluster_ytransform'] = True
    localized = survey.compute(interface=ci, settings=settings)
    positions = result['cluster_lensing_positions']
    operator = ci.get_cluster_ytransform_matrix()
    for name in ('gaussian', 'ssc', 'cng'):
        expected = localize_covariance(interface=ci, covariance=result[name],
                                        indices=positions, operator=operator)
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
