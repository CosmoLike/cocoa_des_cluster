"""Test joint row ordering, count normalization and streamed cross spectra.

The helpers under test (cosmolike_notebook_utils/covariance/survey_cluster.py)
prepare the joint 6x2pt + N forecast: observable_layout numbers every
measured row of the joint vector (ss, gs, gg, cg, N, cc, cs) and finds
where the counts sit; selected_windows turns the cluster selection and
abundance into absolute densities (for counts) and normalized radial
windows (for cluster density contrasts); all_pairs_spectra adds every
cluster spectrum to the galaxy and shear spectra.

Analytic supplied catalogs isolate geometry and indexing from halo fits.
The production C integrators are used, but their inputs come from explicit
functions below. An independent NumPy projection checks every field pair.

Run from the cocoa/Cocoa folder: python -m pytest
projects/des_cluster/tests/covariance/test_survey_cluster.py
"""

from pathlib import Path
import sys

import numpy as np
import pytest

project = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project/'interface'))
sys.path.insert(0, str(project.parents[1]/'external_modules/code/cosmolike_core'))
import cosmolike_des_cluster_interface as ci

# The analytic catalog binds C functions at class definition time, before
# session fixtures run. A data-vector-only build omits those functions.
if not ci.has_covariance:
    pytest.skip("Covariance generation is disabled", allow_module_level=True)

from cosmolike_notebook_utils.covariance.survey_cluster import (
    all_pairs_spectra,
    observable_layout,
    selected_windows,
)


class AnalyticCatalog:
    """Supply two redshift bins and two richness bins without mutable state.

    An object of this class is passed as the interface argument of the
    helpers in place of the compiled library: the helpers call the
    functions below by the library's names, so they read these simple
    formulas instead of the halo model, while the two integrators are the
    library's own. staticmethod stores a plain function on the class, one
    that receives no self argument.
    """

    covariance_project = staticmethod(ci.covariance_project)
    covariance_cluster_spectra = staticmethod(ci.covariance_cluster_spectra)

    @staticmethod
    def phi_cluster(z):
        """Return overlapping radial selection probabilities [z,bin].

        Arguments:
          z = true redshifts, float array.

        Returns:
          float array [len(z), 2]: 0.2 + 0.1 z and 0.6 - 0.1 z.
        """
        return np.column_stack((0.2+0.1*z, 0.6-0.1*z))

    @staticmethod
    def ncl_richness(a):
        """Return selected abundances [a,richness] in inverse length cubed.

        Arguments:
          a = scale factors, float array.

        Returns:
          float array [len(a), 2]: 2 a and 3 a.
        """
        return np.column_stack((2.0*a, 3.0*a))

    @staticmethod
    def bcl_richness(a):
        """Return distinct dimensionless halo biases [a,richness].

        Arguments:
          a = scale factors, float array.

        Returns:
          float array [len(a), 2]: 1 + a and 2 + a.
        """
        return np.column_stack((1.0+a, 2.0+a))

    @staticmethod
    def covariance_power(a, k, linear):
        """Supply a smooth nonlinear power with units of volume.

        Arguments:
          a      = scale factors, float array.
          k      = wavenumbers, float array of the same shape.
          linear = must be False: the cluster spectra use nonlinear power.

        Returns:
          a/(1 + k), elementwise.
        """
        assert not linear
        return a/(1.0+k)

    @staticmethod
    def pcm_1h_richness(k, a):
        """Supply the public [k,a,richness] profile-power convention.

        Arguments:
          k = wavenumbers, float array [nk].
          a = scale factors, float array [na].

        Returns:
          float array [nk, na, 2]: (k + 2) a times 0.3 and 0.7 for the two
          richness bins.
        """
        return (k[:, None, None]+2.0)*a[None, :, None]*np.array([0.3, 0.7])


def geometry():
    """Keep all distances away from zero; weights need not be uniform.

    Returns:
      float array [4, 5], the rows (a, chi, f_K, dchi) of five radial
      nodes, the geometry argument of selected_windows; f_K equals chi
      here (no spatial curvature).
    """
    return np.array([
        [0.5, 0.6, 0.7, 0.8, 0.9],
        [0.8, 0.6, 0.4, 0.2, 0.1],
        [0.8, 0.6, 0.4, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.2, 0.1],
    ])


def test_des_joint_order():
    """Counts sit between cg and cc; richness is fastest within cs pairs.

    The DES layout (6 lens, 4 source, 3 cluster redshift and 4 richness
    bins, 20 angular bins) has 140 two-point rows: 10 xi_+, 10 xi_-, 24
    gamma_t, 6 w_gg, 12 w_cg, 30 w_cc (10 richness pairs per redshift bin)
    and 48 cluster-lensing rows. ss + gs + gg + cg hold 400 + 480 + 120 +
    240 = 1240 entries, so the 12 counts occupy positions 1240-1251, and
    the counts plus the two-point positions cover 0-2811 exactly once.
    A row is (probe, field A, field B) with probe 2 = tangential shear and
    3 = clustering; field IDs run over the 6 lens bins, then the 12 cluster
    categories (6-17, richness fastest), then the 4 source bins (18-21).
    The first cluster-lensing row starts at 1240 + 12 + 600 = 1852.

    Arguments:
      none.

    Returns:
      nothing; a failed assertion fails the test.
    """
    layout = observable_layout(nlens=6, nsource=4, ncluster_z=3,
                                nrichness=4, cg_lens_bin=[0, 1, 2], nbin=20)
    rows = layout['rows']
    assert rows.shape == (140, 3)
    np.testing.assert_array_equal(layout['count_positions'], np.arange(1240, 1252))
    complete = np.concatenate((layout['count_positions'], layout['two_point_positions']))
    np.testing.assert_array_equal(np.sort(complete), np.arange(2812))
    np.testing.assert_array_equal(rows[50:54], [[3, 6, 0], [3, 7, 0],
                                              [3, 8, 0], [3, 9, 0]])
    np.testing.assert_array_equal(rows[62], [3, 6, 6])
    np.testing.assert_array_equal(rows[92:96], [[2, 6, 18], [2, 7, 18],
                                              [2, 8, 18], [2, 9, 18]])
    assert layout['cluster_lensing_positions'].shape == (48, 20)
    np.testing.assert_array_equal(layout['cluster_lensing_positions'][0],
                                  np.arange(1852, 1872))


def test_smaller_layout_and_measured_exclusion():
    """Removing one measured gs row preserves the full source field IDs.

    With 2 lens, 1 source, 2 cluster redshift and 2 richness bins and the
    gamma_t pair (lens 0, source 0) excluded, 19 rows remain, and the
    remaining gamma_t row still names the source by field ID 6 (after the
    2 lens bins and the 4 cluster categories): excluding a measured row
    must not renumber the fields of the internal spectra.

    Arguments:
      none.

    Returns:
      nothing; a failed assertion fails the test.
    """
    layout = observable_layout(nlens=2, nsource=1, ncluster_z=2,
                                nrichness=2, cg_lens_bin=[1, 0], nbin=3,
                                excluded_gammat=[(0, 0)])
    assert layout['rows'].shape == (19, 3)
    np.testing.assert_array_equal(layout['rows'][2], [2, 1, 6])
    np.testing.assert_array_equal(layout['count_positions'], np.arange(27, 31))
    assert layout['cluster_lensing_positions'].shape == (4, 3)


def test_selected_catalog_normalization_and_threads():
    """Counts integrate n chi^2 while each normalized q integrates to one.

    The expected arrays are built here from the AnalyticCatalog formulas:
    density n_i = phi_i(z) n_richness(a) for the 4 categories (redshift
    then richness), derivative n_i b_i, clusters per steradian
    sum n_i f_K^2 dchi, and windows q_i with sum q_i dchi = 1. The results
    must repeat bit for bit at 1, 2, 4 and 8 OpenMP threads
    (.view(np.uint64) compares raw 64-bit patterns).

    Arguments:
      none.

    Returns:
      nothing; a failed assertion fails the test.

    Side effects:
      sets the OpenMP thread count of the compiled library.
    """
    radial = geometry()
    a, unused, distance, dchi = radial
    selection = np.array([0.2+0.1*(1.0/a-1.0), 0.6-0.1*(1.0/a-1.0)])
    number = np.array([2.0*a, 3.0*a])
    density = (selection[:, None, :]*number[None]).reshape(4, 5)
    bias = np.tile(np.array([1.0+a, 2.0+a]), (2, 1))
    per_sr = np.sum(density*distance**2*dchi, axis=1)
    previous = None
    for threads in (1, 2, 4, 8):
        ci.set_omp_threads(n=threads)
        result = selected_windows(interface=AnalyticCatalog(), geometry=radial)
        np.testing.assert_array_equal(result['density'], density)
        np.testing.assert_array_equal(result['derivative'], density*bias)
        np.testing.assert_allclose(result['number_per_sr'], per_sr, rtol=3.e-15)
        np.testing.assert_allclose(np.sum(result['window']*dchi, axis=1), 1.0,
                                   rtol=3.e-15)
        if previous is not None:
            for key in ('density', 'derivative', 'window', 'bias', 'number_per_sr'):
                np.testing.assert_array_equal(result[key].view(np.uint64),
                                              previous[key].view(np.uint64))
        previous = result


@pytest.mark.parametrize('nell', [3, 1027])
def test_all_pairs_streaming_and_spin(nell):
    """Cross the 1024-mode block boundary and check every signed field pair.

    all_pairs_spectra processes 1024 multipoles at a time, so nell = 1027
    spans two blocks and nell = 3 one. The snapshot stands in for the
    galaxy/shear spectra: fields 0 and 1 are galaxies, field 2 a source,
    all their spectra 0.01. The expected 7x7 field matrix (galaxies 0-1,
    clusters 2-5, source 6) is built with NumPy from the AnalyticCatalog
    formulas, the cluster x source entries times the shear spin factor;
    the result must match to 3e-14, be exactly symmetric, and repeat bit
    for bit at 1 and 8 OpenMP threads.

    Arguments:
      nell = number of multipoles, set by @pytest.mark.parametrize, which
             runs the test once per listed value.

    Returns:
      nothing; a failed assertion fails the test.

    Side effects:
      sets the OpenMP thread count of the compiled library.
    """
    radial = geometry()
    catalogs = selected_windows(interface=AnalyticCatalog(), geometry=radial)
    ell = np.geomspace(2.0, 500.0, nell)
    base = np.zeros((2, 3, 5))
    base[0, :2] = np.array([[0.4], [0.7]])
    base[1, 2] = np.array([0.1, 0.3, 0.5, 0.6, 0.4])
    original = np.full((nell, 3, 3), 0.01)
    snapshot = {
        'geometry': radial,
        'windows': base,
        'nlens': 2,
        'spectra': original,
    }
    expected = np.empty((nell, 7, 7))
    base_positions = np.array([0, 1, 6])
    # base_positions[:, None] (a column) and base_positions (a row) broadcast
    # to the 3x3 grid of (row, column) field pairs
    expected[:, base_positions[:, None], base_positions] = original
    measure = radial[3]/radial[2]**2
    weighted = catalogs['window']*catalogs['bias']
    ordinary = np.concatenate((base[0, :2], base[1, 2:]))
    for index, mode in enumerate(ell):
        wave = (mode+0.5)/radial[2]
        power = radial[0]/(1.0+wave)
        cross = (weighted*measure*power) @ ordinary.T
        for cluster in range(4):
            # categories run redshift then richness, so cluster % 2 is the
            # richness bin of the one-halo amplitude
            amplitude = [0.3, 0.7][cluster % 2]
            profile = amplitude*(wave+2.0)*radial[0]
            cross[cluster, 2] += np.sum(catalogs['window'][cluster]*measure
                                         *profile*ordinary[2])
        spin = np.sqrt((mode-1)*mode*(mode+1)*(mode+2))/(mode+0.5)**2
        cross[:, 2] *= spin
        matrix = expected[index]
        cluster_positions = np.arange(2, 6)
        matrix[np.ix_(cluster_positions, base_positions)] = cross
        matrix[np.ix_(base_positions, cluster_positions)] = cross.T
        matrix[2:6, 2:6] = (weighted*measure*power) @ weighted.T
    baseline = None
    for threads in (1, 8):
        ci.set_omp_threads(n=threads)
        actual = all_pairs_spectra(interface=AnalyticCatalog(), ell=ell,
                                   snapshot=snapshot, catalogs=catalogs)
        np.testing.assert_allclose(actual, expected, rtol=3.e-14)
        np.testing.assert_array_equal(actual, actual.transpose(0, 2, 1))
        if baseline is not None:
            np.testing.assert_array_equal(actual.view(np.uint64), baseline.view(np.uint64))
        baseline = actual


def test_bad_layout_and_geometry():
    """Reject a missing matched bin or a singular radial measure directly.

    A cg_lens_bin list one entry short, zero angular bins, and a zero f_K
    node (row 2 of the geometry) must each raise ValueError whose message
    contains the given word (pytest.raises(..., match=...) checks it).

    Arguments:
      none.

    Returns:
      nothing; a failed assertion fails the test.
    """
    with pytest.raises(ValueError, match='cg_lens_bin'):
        observable_layout(nlens=2, nsource=1, ncluster_z=2, nrichness=2,
                          cg_lens_bin=[0], nbin=3)
    with pytest.raises(ValueError, match='nbin'):
        observable_layout(nlens=2, nsource=1, ncluster_z=2, nrichness=2,
                          cg_lens_bin=[0, 1], nbin=0)
    bad = geometry()
    bad[2, 0] = 0.0
    with pytest.raises(ValueError, match='geometry'):
        selected_windows(interface=AnalyticCatalog(), geometry=bad)
