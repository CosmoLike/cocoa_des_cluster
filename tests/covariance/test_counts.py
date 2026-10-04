"""Independent shell-volume and joint-SSC checks for cluster counts.

Polynomial shell integrals have closed forms, so these tests do not use
the legacy cluster covariance as a reference. They check the distance
powers, exclusive-bin Poisson noise, unit invariance, common-mode
positivity, supplied signed responses and one-to-eight-thread consistency.
No cosmological model or survey-accuracy prescription is calibrated here.
"""

from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.special import roots_legendre

project = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project.parents[1]/"external_modules/code/cosmolike_core"))
sys.path.insert(0, str(project/"interface"))

import cosmolike_des_cluster_interface as ci
from cosmolike_notebook_utils.covariance.counts_cluster import count_statistics


def shell_inputs(nnode=9):
    """Supply constant selected abundances on a radial interval [0.2,0.8]."""
    nodes, weights = roots_legendre(n=nnode)
    distance = 0.5+0.3*nodes
    dchi = 0.3*weights
    number = np.array([0.4, 0.2, 0.1])
    bias = np.array([1.5, 2.0, 2.5])
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
    """Construct Phi_AB=A/f_K^2 using the production two-point response."""
    amplitude = np.array([0.02, -0.07])
    shape = (2, len(distance))
    return ci.covariance_ssc_shell_response(
        distance=distance, signal=np.zeros(2),
        pair_window=np.ones(shape), mean_window=np.zeros(shape),
        power_response=np.repeat(amplitude[:, None], len(distance), axis=1),
    )


def test_counts_and_cross_ssc_against_closed_integrals():
    """Volume powers cancel only in the count-two-point cross integral."""
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
    # constant here. The legacy missing-denominator error fails this check.
    expected_cross = (0.7*0.03*(0.8-0.2)
                      *np.outer(biased_number, np.array([0.02, -0.07])))
    np.testing.assert_allclose(actual["mean"], expected_mean, rtol=2.e-14)
    np.testing.assert_allclose(actual["poisson"], np.diag(expected_mean), rtol=2.e-14)
    np.testing.assert_allclose(actual["ssc"], expected_ssc, rtol=2.e-14)
    np.testing.assert_allclose(actual["cross_ssc"], expected_cross, rtol=2.e-14)
    assert np.linalg.eigvalsh(actual["total"])[0] > 0.0

    # The joint SSC comes from one background field. Negative cross entries
    # are allowed; every joint variance must remain nonnegative to roundoff.
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
    """Changing length units or linearly combining measurements is consistent."""
    inputs = shell_inputs()
    other = two_point_shell(distance=inputs["distance"])
    original = count_statistics(interface=ci, two_point_response=other, **inputs)

    # A larger numerical distance unit scales n and dn by its inverse cube,
    # Phi by its inverse, and the collapsed background variance by its length.
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
    # This is the algebra needed for angular bins and a later Y transform.
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
    """Single, paired and odd shell arrays repeat bitwise with owned outputs."""
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
    """A nonresponding catalog has Poisson noise only; malformed inputs stop."""
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
