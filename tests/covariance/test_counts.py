"""Independent shell-volume and joint-SSC checks for cluster counts.

The function under test, count_statistics of
cosmolike_notebook_utils/covariance/counts_cluster.py, integrates supplied
radial "shells" of a cluster sample. For count bin i at transverse
comoving distance f_K, with survey solid angle Omega:

  mean    N_i = integral dchi Omega f_K^2 n_i        (an absolute number)
  Poisson C_ij = N_i delta_ij
  SSC     C_ij = integral dchi sigma_b^2 Phi_i Phi_j,
          Phi_i = Omega f_K^2 dn_i/d(delta_b)

Counts are absolute numbers of clusters, so their Poisson (shot) noise is
the mean count. The observed bins are exclusive (each cluster lands in
one richness/redshift bin), so the Poisson part is diagonal, while the
super-sample covariance (SSC, the response of every bin to one density
mode delta_b larger than the survey, of variance sigma_b^2) correlates
all bins. The count x two-point SSC block uses the response Phi_AB of a
two-point function to the same mode.

Polynomial shell integrals have closed forms, so these tests need no
other covariance code as a reference: Gauss-Legendre quadrature with n
nodes integrates polynomials up to degree 2n - 1 exactly, so the only
error is floating-point rounding. They check the distance powers,
exclusive-bin Poisson noise, unit invariance, common-mode positivity,
supplied signed responses and one-to-eight-thread consistency. No
cosmological model or survey-accuracy prescription is calibrated here.

Run from the cocoa/Cocoa folder: python -m pytest
projects/des_cluster/tests/covariance/test_counts.py
"""

from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.special import roots_legendre

# project = projects/des_cluster; the shared core (cosmolike_notebook_utils)
# and interface/ (the compiled library) go first on the import path
project = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project.parents[1]/"external_modules/code/cosmolike_core"))
sys.path.insert(0, str(project/"interface"))

import cosmolike_des_cluster_interface as ci
from cosmolike_notebook_utils.covariance.counts_cluster import count_statistics


def shell_inputs(nnode=9):
    """Supply constant selected abundances on a radial interval [0.2,0.8].

    Three count bins with constant densities n = 0.4, 0.2, 0.1 (L^-3) and
    linear biases b = 1.5, 2.0, 2.5, so dn/d(delta_b) = b n; solid angle
    0.7 sr; background variance sigma_b^2 = 0.03 (L) at every node. The
    Gauss-Legendre nodes x in [-1, 1] map to distances 0.5 + 0.3 x, and
    the weights scale by the half-width 0.3. L is an arbitrary length unit.

    Arguments:
      nnode = number of Gauss-Legendre nodes, a positive integer.

    Returns:
      a dict of the keyword arguments of count_statistics: distance and
      dchi [nnode], density and derivative [3, nnode], area_sr (float),
      background_variance [nnode].
    """
    nodes, weights = roots_legendre(n=nnode)
    distance = 0.5+0.3*nodes
    dchi = 0.3*weights
    number = np.array([0.4, 0.2, 0.1])
    bias = np.array([1.5, 2.0, 2.5])
    # number[:, None] is a [3, 1] column; repeat copies it into nnode
    # columns, giving the [count bin, node] layout
    density = np.repeat(number[:, None], nnode, axis=1)
    return {
        "distance": distance,
        "dchi": dchi,
        "density": density,
        "derivative": density*bias[:, None],
        "area_sr": 0.7,
        "background_variance": np.full(nnode, 0.03),
    }


def two_point_shell(distance):
    """Construct Phi_AB=A/f_K^2 using the production two-point response.

    covariance_ssc_shell_response returns, per two-point row and node,
    Phi_AB = W_A W_B D((l+1/2)/f_K)/f_K^2 - (U_A + U_B) C_AB. With a unit
    pair window W_A W_B = 1, no mean subtraction (U = 0) and a constant
    power response D = A, it reduces to A/f_K^2. One amplitude is negative:
    a response may have either sign.

    Arguments:
      distance = transverse distances f_K of the shell nodes, float [nnode].

    Returns:
      float array [2, nnode]: Phi_AB of two two-point rows, amplitudes
      0.02 and -0.07.
    """
    amplitude = np.array([0.02, -0.07])
    shape = (2, len(distance))
    return ci.covariance_ssc_shell_response(
        distance=distance, signal=np.zeros(2),
        pair_window=np.ones(shape), mean_window=np.zeros(shape),
        power_response=np.repeat(amplitude[:, None], len(distance), axis=1),
    )


def test_counts_and_cross_ssc_against_closed_integrals():
    """Volume powers cancel only in the count-two-point cross integral.

    With constant densities the integrands are polynomials in chi (chi^2
    for the mean, chi^4 for the count SSC, chi^0 for the cross term), so
    9 Gauss-Legendre nodes give the closed forms below exactly; rtol =
    2e-14, about a hundred double-precision rounding units, leaves room
    for summation order only.

    Arguments:
      none.

    Returns:
      nothing; a failed assertion fails the test.
    """
    inputs = shell_inputs()
    other = two_point_shell(distance=inputs["distance"])
    actual = count_statistics(interface=ci, two_point_response=other, **inputs)
    number = np.array([0.4, 0.2, 0.1])
    biased_number = number*np.array([1.5, 2.0, 2.5])

    # Integral chi^2 dchi gives a volume; the count SSC contains chi^4.
    expected_mean = 0.7*number*(0.8**3-0.2**3)/3.0
    expected_ssc = (0.7**2*0.03*(0.8**5-0.2**5)/5.0
                    *np.outer(biased_number, biased_number))

    # Phi_N contains chi^2 and Phi_AB contains chi^-2. Their product is
    # constant here. A cross term computed without the 1/f_K^2 of Phi_AB
    # would grow as chi^2 and fail this check.
    expected_cross = (0.7*0.03*(0.8-0.2)
                      *np.outer(biased_number, np.array([0.02, -0.07])))
    np.testing.assert_allclose(actual["mean"], expected_mean, rtol=2.e-14)
    np.testing.assert_allclose(actual["poisson"], np.diag(expected_mean), rtol=2.e-14)
    np.testing.assert_allclose(actual["ssc"], expected_ssc, rtol=2.e-14)
    np.testing.assert_allclose(actual["cross_ssc"], expected_cross, rtol=2.e-14)
    assert np.linalg.eigvalsh(actual["total"])[0] > 0.0

    # The joint SSC comes from one background field. Negative cross entries
    # are allowed; every joint variance must remain nonnegative to roundoff.
    # covariance_project sums weight*left_i*right_j over the nodes, so
    # other_ssc is the two-point SSC block, and np.block assembles the
    # 2x2 arrangement of blocks into one [5, 5] matrix. Its smallest
    # eigenvalue may fall below zero only by rounding relative to the
    # largest one.
    other_ssc = ci.covariance_project(
        left=other, right=other,
        weight=inputs["dchi"]*inputs["background_variance"],
    )
    joint = np.block([
        [actual["ssc"], actual["cross_ssc"]],
        [actual["cross_ssc"].T, other_ssc],
    ])
    eigenvalues = np.linalg.eigvalsh(joint)
    assert eigenvalues[0] > -2.e-15*eigenvalues[-1]


def test_length_units_and_observable_transform():
    """Changing length units or linearly combining measurements is consistent.

    Every result is a number of clusters or a dimensionless covariance, so
    rescaling every length by the same factor must leave it unchanged.

    Arguments:
      none.

    Returns:
      nothing; a failed assertion fails the test.
    """
    inputs = shell_inputs()
    other = two_point_shell(distance=inputs["distance"])
    original = count_statistics(interface=ci, two_point_response=other, **inputs)

    # A larger numerical distance unit scales n and dn by its inverse cube,
    # Phi by its inverse, and the collapsed background variance by its length.
    # The factor 3000 is close to c/H0 in Mpc/h (2997.9), the length unit of
    # cosmolike. dict(inputs) copies the dict, so inputs keeps its values.
    factor = 3000.0
    converted = dict(inputs)
    converted["distance"] = inputs["distance"]*factor
    converted["dchi"] = inputs["dchi"]*factor
    converted["density"] = inputs["density"]/factor**3
    converted["derivative"] = inputs["derivative"]/factor**3
    converted["background_variance"] = inputs["background_variance"]*factor
    actual = count_statistics(
        interface=ci, two_point_response=other/factor, **converted,
    )
    for key in ("mean", "poisson", "ssc", "total", "cross_ssc"):
        np.testing.assert_allclose(actual[key], original[key], rtol=2.e-14)

    # Counts have one index; only the two-point side receives this transform.
    # A linear map T of the two-point rows (an angular-bin average and the Y
    # localization are such maps) turns the cross block X into X T^t; here T
    # is a [3, 2] matrix and @ is the matrix product.
    transform = np.array([[1.0, 0.4], [0.2, -0.3], [-0.5, 1.0]])
    combined = count_statistics(
        interface=ci, two_point_response=transform @ other, **inputs,
    )
    np.testing.assert_allclose(
        combined["cross_ssc"], original["cross_ssc"] @ transform.T,
        rtol=2.e-14,
    )


@pytest.mark.parametrize("nnode", [1, 6, 9])
def test_threads_tails_and_output_ownership(nnode):
    """Single, paired and odd shell arrays repeat bitwise with owned outputs.

    The C code processes two shell nodes at a time (SIMD lanes) and treats
    an odd last node separately, so nnode = 1, 6 and 9 test the single,
    paired and odd cases. Every output must be identical bit for bit at
    1, 2, 4 and 8 OpenMP threads: .view(np.uint64) reads each float64 as
    its raw 64-bit pattern, so even a last-digit difference fails. Without
    a two-point response the cross block has zero columns, shape (3, 0).
    The ownership check calls again with doubled densities and requires
    the first result to stay unchanged: each call returns arrays of its
    own, not views of a buffer that a later call overwrites.

    Arguments:
      nnode = number of shell nodes, set by @pytest.mark.parametrize,
              which runs the test once per listed value.

    Returns:
      nothing; a failed assertion fails the test.

    Side effects:
      sets the OpenMP thread count of the compiled library.
    """
    inputs = shell_inputs(nnode=nnode)
    baseline = None
    for threads in (1, 2, 4, 8):
        ci.set_omp_threads(n=threads)
        actual = count_statistics(interface=ci, **inputs)
        assert actual["cross_ssc"].shape == (3, 0)
        if baseline is None:
            baseline = actual
        for key in actual:
            np.testing.assert_array_equal(
                actual[key].view(np.uint64), baseline[key].view(np.uint64),
            )

    saved = baseline["shell_density"].copy()
    changed_inputs = dict(inputs)
    changed_inputs["density"] = inputs["density"]*2.0
    changed = count_statistics(interface=ci, **changed_inputs)
    np.testing.assert_array_equal(baseline["shell_density"], saved)
    np.testing.assert_array_equal(changed["shell_density"], 2.0*saved)


def test_zero_response_and_validation():
    """A nonresponding catalog has Poisson noise only; malformed inputs stop.

    With dn/d(delta_b) = 0 the SSC term vanishes and the total equals the
    Poisson diagonal. Each malformed case replaces one input of a copy of
    the valid inputs: zero or more than full-sky area (4 pi sr), a zero
    distance, a negative density, a density with one node missing, a NaN
    response, a zero quadrature weight, a negative background variance,
    and a two-point response with 8 nodes instead of 9. Each must raise
    ValueError (pytest.raises fails the test when it does not).

    Arguments:
      none.

    Returns:
      nothing; a failed assertion fails the test.
    """
    inputs = shell_inputs()
    inputs["derivative"] = np.zeros_like(inputs["derivative"])
    result = count_statistics(interface=ci, **inputs)
    np.testing.assert_array_equal(result["total"], result["poisson"])
    np.testing.assert_array_equal(result["ssc"], np.zeros((3, 3)))

    for key, bad in (
        ("area_sr", 0.0),
        ("area_sr", 4*np.pi+0.1),
        ("distance", np.zeros(9)),
        ("density", -inputs["density"]),
        ("density", inputs["density"][:, :-1]),
        ("derivative", np.full((3, 9), np.nan)),
        ("dchi", np.zeros(9)),
        ("background_variance", np.full(9, -1.0)),
        ("two_point_response", np.zeros((2, 8))),
    ):
        malformed = dict(inputs)
        malformed[key] = bad
        with pytest.raises(ValueError):
            count_statistics(interface=ci, **malformed)
