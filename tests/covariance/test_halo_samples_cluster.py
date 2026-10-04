"""Selected halo samples against public readers and the Tinker formula.

These checks separate sampling and normalization from the moment integrator.
They share the physical sigma, bias and NFW readers with production, so
they are not an independent calibration of those halo fits.
"""

from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.special import roots_legendre

project = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project/'interface'))
sys.path.insert(0, str(project/'covariance'))
sys.path.insert(0, str(project.parents[1]/'external_modules/code/cosmolike_core'))
import cosmolike_des_cluster_interface as ci
import des_cluster_covariance as survey


@pytest.fixture(scope='module')
def initialized():
    """Install a massless forecast and four fixed richness selections."""
    settings = survey.configuration(accuracy_boost=1)
    survey.initialize(interface=ci, settings=settings)
    ci.reset_cluster()
    ci.init_cluster_model(mor_model=0, kernel_mode=1, selection_model=0,
                          ytransform=0, include_ia=0, magnification=0.0)
    edges = np.array([20., 30., 45., 60., 500.])
    ci.init_cluster_richness_bins(lambda_min=edges[:-1].copy(),
                                  lambda_max=edges[1:].copy())
    ci.set_nuisance_cluster_mor(MOR=[4.26, 0.943, 0.15, 0.207])
    return settings


def sample_inputs(nmass):
    """Use a common Gaussian mass rule and distinct k grids per redshift."""
    node, measure = roots_legendre(n=nmass)
    lower = np.log(1.e12)
    upper = np.log(1.e16)
    return {
        'a': np.array([0.2, 0.25, 0.63, 0.81]),
        'k': 2997.92458*np.array([[0.0, 0.01, 0.7], [0.02, 0.1, 2.0],
                                 [0.001, 1.0, 10.0], [0.0, 0.3, 30.0]]),
        'lnm': 0.5*(upper+lower)+0.5*(upper-lower)*node,
        'dlnm': 0.5*(upper-lower)*measure,
    }


@pytest.mark.parametrize('mode', [0, 1])
@pytest.mark.parametrize('nmass', [1, 2, 9, 257])
def test_samples_against_scalar_readers(initialized, mode, nmass):
    """Independent mass weights check both HMF amplitudes and the z=3 cap."""
    ci.init_cluster_hmf_alpha_mode(hmf_alpha_mode=mode)
    values = sample_inputs(nmass=nmass)
    mass = np.exp(values['lnm'])
    rho = 7.4775e21*initialized['cosmology']['omegam']
    selected = np.asarray(ci.prob_richness_bin_given_m(
        lnM=values['lnm'], z=1.0/values['a']-1.0,
    ))
    expected_weight = np.empty(shape=(4, 4, nmass))
    expected_bias = np.empty(shape=(4, nmass))
    expected_profile = np.empty(shape=(4, 3, nmass))

    for state, a in enumerate(values['a']):
        fit_a = max(a, 0.25)
        beta = 0.589*fit_a**-0.2
        gamma = 0.864*fit_a**0.01
        phi = -0.729*fit_a**0.08
        eta = -0.243*fit_a**-0.27
        for node, halo_mass in enumerate(mass):
            nu = 1.686/np.sqrt(ci.sigma2(M=halo_mass, a=a, field=1))
            if mode == 0:
                # Evaluate the full fixed-alpha formula at each nu.
                # Production instead rescales the public normalized fnu.
                multiplicity = (0.368*(1.0+(beta*nu)**(-2.0*phi))
                                *nu**(2.0*eta)*np.exp(-gamma*nu**2/2.0))
            else:
                multiplicity = ci.fnu(nu=nu, a=a)
            number = (values['dlnm'][node]*rho/halo_mass*multiplicity*nu
                      *ci.dlognudlogm(M=halo_mass, a=a))
            expected_weight[state, :, node] = number*selected[node, state]
            expected_bias[state, node] = ci.hb1nu(nu=nu, a=a)
            concentration = ci.conc(m=halo_mass, a=a)
            for mode_index, k in enumerate(values['k'][state]):
                # At zero wavenumber all mass elements add with unit
                # Fourier phase, so the normalized profile is exactly 1.
                value = 1.0
                if k > 0.0:
                    value = ci.u_nfw_c(c=concentration, k=k, m=halo_mass, a=a)
                expected_profile[state, mode_index, node] = halo_mass/rho*value

    baseline = None
    for threads in (1, 2, 4, 8):
        ci.set_omp_threads(n=threads)
        actual = ci.covariance_cluster_halo_samples(**values)
        for array in actual.values():
            assert np.all(np.isfinite(array))
        np.testing.assert_allclose(actual['weight'], expected_weight, rtol=5.e-14)
        np.testing.assert_array_equal(actual['bias'], expected_bias)
        np.testing.assert_allclose(actual['profile'], expected_profile, rtol=3.e-15)
        if baseline is None:
            baseline = actual
        for key in actual:
            np.testing.assert_array_equal(actual[key].view(np.uint64),
                                          baseline[key].view(np.uint64))

    # The reader returns independent storage, ready for the separate mass
    # integrator. A later evaluation must not overwrite an earlier sample.
    saved = actual['weight'].copy()
    ci.covariance_cluster_halo_samples(**sample_inputs(nmass=3))
    np.testing.assert_array_equal(actual['weight'], saved)
    moments = ci.covariance_cluster_moments(**actual)
    np.testing.assert_allclose(moments['density'][0], np.sum(expected_weight, axis=2),
                               rtol=3.e-15)


def test_selected_profile_response_uses_projected_catalog_mean(initialized):
    """Perturb one shell's halo abundance, then normalize the whole catalog."""
    ci.init_cluster_hmf_alpha_mode(hmf_alpha_mode=0)
    values = sample_inputs(nmass=96)
    values['a'] = values['a'][2:].copy()
    distance = np.array([0.6, 0.25])
    # A fixed angular mode samples k=(ell+1/2)/distance in each shell.
    modes = np.array([100.5, 1000.5, 10000.5])
    values['k'] = np.ascontiguousarray(modes[None, :]/distance[:, None])
    sampled = ci.covariance_cluster_halo_samples(**values)
    moments = ci.covariance_cluster_moments(**sampled)
    density = moments['density'][0, :, 0]
    biased_density = moments['density'][1, :, 0]
    own = moments['single'][0, :, 0]
    biased_own = moments['single'][1, :, 0]
    dchi = np.array([0.06, 0.03])
    source_window = np.array([0.4, 0.7])
    number_per_sr = np.sum(dchi*distance**2*density)
    signal = np.sum(dchi[:, None]*source_window[:, None]*own, axis=0)/number_per_sr
    window = distance**2*density/number_per_sr
    mean_window = distance**2*biased_density/number_per_sr
    actual = ci.covariance_ssc_shell_response(
        distance=distance, signal=signal,
        pair_window=np.repeat((window*source_window)[None, :], 3, axis=0),
        mean_window=np.repeat(mean_window[None, :], 3, axis=0),
        power_response=np.ascontiguousarray((biased_own/density[:, None]).T),
    )

    # Independently perturb dn by 1+b*delta at ONE shell, keeping each
    # halo's selection and profile fixed. Recompute the angular numerator
    # and total catalog denominator from mass samples, not from J11.
    step = 1.e-6
    finite_difference = np.empty_like(prototype=actual)
    for state in range(2):
        changed_signal = []
        for perturbation in (-step, step):
            weight = sampled['weight'][:, 0].copy()
            weight[state] *= 1.0+sampled['bias'][state]*perturbation
            changed_density = np.sum(weight, axis=1)
            changed_own = np.einsum('am,akm->ak', weight, sampled['profile'])
            denominator = np.sum(dchi*distance**2*changed_density)
            numerator = np.sum(dchi[:, None]*source_window[:, None]*changed_own,
                                axis=0)
            changed_signal.append(numerator/denominator)
        finite_difference[:, state] = ((changed_signal[1]-changed_signal[0])
                                       /(2.0*step*dchi[state]))
    np.testing.assert_allclose(actual, finite_difference, rtol=3.e-8, atol=1.e-16)


def test_sample_domain_guards(initialized):
    """Reject malformed numerical inputs before calling the C readers."""
    values = sample_inputs(nmass=3)
    for key, replacement in (('a', np.array([1.0]*4)),
                             ('a', np.array([np.nan]*4)),
                             ('k', -values['k']-1.0),
                             ('k', values['k'][:2]),
                             ('lnm', values['lnm']+100.0),
                             ('dlnm', np.zeros(shape=3)),
                             ('dlnm', np.ones(shape=4))):
        bad = dict(values)
        bad[key] = replacement
        with pytest.raises(ValueError):
            ci.covariance_cluster_halo_samples(**bad)
