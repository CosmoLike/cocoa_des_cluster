"""Non-SSC count-matter projection: area cancellation, units and pair factors.

The function under test, count_matter_cross of
cosmolike_notebook_utils/covariance/counts_cluster.py, computes the part of
the covariance between a cluster count N_i and a projected matter spectrum
C_AB that does not come from super-survey modes (Schaan, Takada & Spergel
2014, arXiv:1406.3330, Eq. 35):

  Cov(N_i, C_AB) = integral dchi W_A W_B/f_K^2
                   [J02_i(k,k) + 2 P_lin(k) I11(k) J11_i(k)],
  k = (l + 1/2)/f_K.

The one-halo term J02 puts both matter points of the spectrum in a counted
halo; the two-halo term pairs a counted halo with another halo, and either
leg of the spectrum may sit in the counted halo, hence the factor 2. J11
and J02 are mass integrals over the selected clusters (profile moments),
I11 the same one-profile moment over all halos. The survey area of the
absolute count cancels the inverse area of the spectrum's covariance.

The tests use polynomial inputs with closed-form integrals, a change of
length unit, an exact enumeration of a toy Poisson halo population, and
malformed inputs.

Run from the cocoa/Cocoa folder: python -m pytest
projects/des_cluster/tests/covariance/test_counts_matter_cross.py
"""

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
    """Supply polynomial shell moments over chi in [1,2].

    8 Gauss-Legendre nodes on [1, 2], two count bins with selected weights
    0.2 and 0.7, and nk = 3 wavenumber samples. J11 = chi^2 times the
    selected weight; J02 is chi^2 times the weight times 1, 2, 3 on its
    K = Q entries and zero elsewhere (J02 stores the nk(nk+1)/2 = 6
    pairs K <= Q of the triangle). Two observables with pair windows
    W_A W_B = 0.4 and 1.1, transfers [1, 1, 1] and [0.5, 0.9, 0.99],
    P_lin = 0.03, 0.02, 0.01 and I11 = 0.7 at every node.

    Arguments:
      none.

    Returns:
      a dict of the keyword arguments of count_matter_cross.
    """
    nodes, weights = roots_legendre(n=8)
    distance = 1.5+0.5*nodes
    nk = 3
    selected = np.array([0.2, 0.7])
    first, second = np.triu_indices(n=nk)
    single = np.zeros(shape=(2, 8, 2, nk))
    pair = np.zeros(shape=(3, 8, 2, len(first)))
    single[1] = distance[:, None, None]**2*selected[None, :, None]
    # A selected covariance kernel proportional to chi^2 cancels the
    # projection's chi^-2, leaving a constant integrand whose integral is
    # the interval length, 1. Only single[1] and pair[0] are used: the
    # moments are views that start inside larger arrays.
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
        'moments': {'J11': single[1], 'J02': pair[0]},
    }


def test_closed_projection_and_thread_repeatability():
    """Both halo terms include their own factors and shared field transfers.

    With the inputs above the integrals are exact (Gauss-Legendre on a
    constant integrand), so the expected arrays [count, observable, k]
    are: one-halo = weight x window x transfer x (1, 2, 3); two-halo =
    weight x window x transfer x 2 x 0.7 x P_lin. rtol = 2e-14 allows
    rounding only. The results must also be identical bit for bit
    (.view(np.uint64) compares raw 64-bit patterns) at 1, 2, 4 and 8
    OpenMP threads.

    Arguments:
      none.

    Returns:
      nothing; a failed assertion fails the test.

    Side effects:
      sets the OpenMP thread count of the compiled library.
    """
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
    """The projected count-spectrum covariance is independent of length units.

    Every length is multiplied by 3000 (close to c/H0 in Mpc/h, the length
    unit of cosmolike): distances and weights by the factor, W_A W_B (units
    L^-2) by its inverse square, P_lin and J02 (units L^3) by its cube; J11
    is dimensionless. The result, a covariance of a count with a
    dimensionless spectrum, must not change.

    Arguments:
      none.

    Returns:
      nothing; a failed assertion fails the test.
    """
    values = inputs()
    baseline = count_matter_cross(**values)
    factor = 3000.0
    converted = dict(values)
    converted['distance'] = values['distance']*factor
    converted['dchi'] = values['dchi']*factor
    converted['pair_window'] = values['pair_window']/factor**2
    converted['linear_power'] = values['linear_power']*factor**3
    converted['moments'] = {
        'J11': values['moments']['J11'],
        'J02': values['moments']['J02']*factor**3,
    }
    changed = count_matter_cross(**converted)
    for key in changed:
        np.testing.assert_allclose(changed[key], baseline[key], rtol=2.e-14)

def test_exact_poisson_counts_and_different_halo_pairs():
    """Enumerate a toy halo population, without assuming the analytic kernel.

    Two halo mass bins with number densities 1.3 and 2.1 (per unit
    volume), detection probabilities 0.3 and 0, profiles 1.4 and 0.4 and
    biases 2.1 and 1.2, in one radial shell with unit window. The count is
    the number of detected halos; the one-halo "spectrum" estimator sums
    profile^2 over halos, the two-halo one sums biased profile products
    over pairs of distinct halos. Their exact covariance follows from
    Poisson statistics, summed over every count from 0 to 44 (the
    probabilities must sum to 1 to rtol 2e-14, so the truncation is
    negligible). Two volumes, 0.7 and 2.3, must give the same answer: the
    area cancels. rtol = 3e-13 is looser than elsewhere because the
    covariance is a difference of two sums of 45^3 terms.

    Arguments:
      none.

    Returns:
      nothing; a failed assertion fails the test.
    """
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
    # the three None-padded views broadcast into a [45, 45, 45] grid of
    # (detected, missed, other) counts
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
    """Reject malformed shell grids and moment axes before integration.

    Each case replaces one input of a copy of the valid inputs: a zero
    distance, a quadrature weight array one node short, I11 with 2 instead
    of 3 wavenumbers, a negative linear power, a transfer with 2 instead of
    3 wavenumbers, a NaN window, and moments whose J02 triangle holds 5
    instead of 6 entries. Each must raise ValueError (pytest.raises fails
    the test when it does not).

    Arguments:
      none.

    Returns:
      nothing; a failed assertion fails the test.
    """
    values = inputs()
    for key, bad in (
        ('distance', np.zeros(shape=8)),
        ('dchi', np.ones(shape=7)),
        ('i11', np.ones(shape=(8, 2))),
        ('linear_power', -np.ones(shape=(8, 3))),
        ('transfer', np.ones(shape=(2, 2))),
        ('pair_window', np.full(shape=(2, 8), fill_value=np.nan)),
        ('moments', {'J11': np.ones(shape=(8, 2, 3)),
                     'J02': np.ones(shape=(8, 2, 5))}),
    ):
        malformed = dict(values)
        malformed[key] = bad
        with pytest.raises(ValueError):
            count_matter_cross(**malformed)
