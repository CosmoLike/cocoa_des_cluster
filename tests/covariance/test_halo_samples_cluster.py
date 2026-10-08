"""Selected halo samples against public readers and the Tinker formula.

The compiled function covariance_cluster_halo_samples evaluates, on the
redshift, wavenumber and mass nodes of a covariance calculation, the
three arrays that covariance_cluster_moments integrates:

  weight[state, richness, mass] = dlnM (dn/dlnM) P(richness bin | M, z),
      the selected number density, with the Tinker mass function
      dn/dlnM = (rho_m/M) nu f(nu) dln(nu)/dlnM and the lognormal
      mass-observable relation giving the richness-bin probability;
  bias[state, mass] = the linear halo bias b(nu);
  profile[state, k, mass] = (M/rho_m) u_NFW(k | M), the Fourier transform
      of the NFW mass profile normalized to 1 at k = 0, times M/rho_m.

nu = 1.686/sigma(M, z) is the peak height of a halo of mass M (sigma of
the cold dark matter + baryon field). Lengths are in units of c/H0 =
2997.92458 Mpc/h, so a wavenumber in h/Mpc is multiplied by 2997.92458,
and rho_m = 7.4775e21 Omega_m M_sun/h per (c/H0)^3.

These checks separate sampling and normalization from the moment integrator.
They share the physical sigma, bias and NFW readers with production, so
they are not an independent calibration of those halo fits.

Run from the cocoa/Cocoa folder: python -m pytest
projects/des_cluster/tests/covariance/test_halo_samples_cluster.py
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
    """Install a massless forecast and four fixed richness selections.

    A pytest fixture: a test that names it as an argument receives its
    return value, and scope='module' runs it once for all the tests of
    this file. It runs CAMB through the forecast adapter
    (covariance/des_cluster_covariance.py), then sets the cluster model:
    lognormal mass-observable relation with the Table I values (ln
    lambda_0, A_lambda, sigma_int, B_lambda), abundance-weighted kernels,
    no selection bias, no Y transform, no intrinsic alignment and no
    magnification, and the four DES richness bins 20-30-45-60-500.

    Returns:
      the forecast settings dictionary (its "cosmology" entry gives
      Omega_m).

    Side effects:
      replaces the cosmology and cluster state of the compiled library.
    """
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
    """Use a common Gaussian mass rule and distinct k grids per redshift.

    Four scale factors (z = 4, 3, 0.59, 0.23; the first lies beyond the
    z = 3 cap of the Tinker redshift evolution), three wavenumbers per
    redshift (h/Mpc, converted to 1/(c/H0); zero included, where u_NFW = 1)
    and nmass Gauss-Legendre nodes in ln M between 1e12 and 1e16 M_sun/h.

    Arguments:
      nmass = number of mass nodes, a positive integer.

    Returns:
      a dict of the keyword arguments of covariance_cluster_halo_samples:
      a [4], k [4, 3], lnm [nmass] and dlnm [nmass].
    """
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
    """Independent mass weights check both HMF amplitudes and the z=3 cap.

    The expected arrays are rebuilt node by node from scalar readers of
    the library (sigma2, fnu, dlognudlogm, hb1nu, conc, u_nfw_c) and, for
    hmf_alpha_mode 0, from the Tinker et al. (2010) multiplicity
    f(nu) = 0.368 [1 + (beta nu)^(-2 phi)] nu^(2 eta) exp(-gamma nu^2/2)
    with beta, gamma, phi, eta evolving as powers of a, frozen at z = 3
    (fit_a = max(a, 0.25)). hmf_alpha_mode 1 uses the normalized
    multiplicity of the library (fnu). The weights must agree to 1e-13
    (see the comment below), the biases exactly and the profiles to 3e-15;
    every array must repeat bit for bit at 1, 2, 4 and 8 OpenMP threads,
    stay unchanged by a later call, and integrate to the density of
    covariance_cluster_moments.

    Arguments:
      initialized = the fixture above (the forecast settings).
      mode        = hmf_alpha_mode, 0 or 1, set by
                    @pytest.mark.parametrize.
      nmass       = number of mass nodes, set by @pytest.mark.parametrize;
                    the two decorators run every (mode, nmass) combination.

    Returns:
      nothing; a failed assertion fails the test.

    Side effects:
      sets the HMF amplitude mode and the OpenMP thread count of the
      compiled library.
    """
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
        # The independent formula squares nu before multiplying by gamma;
        # the C reader multiplies left to right. In the exponential tail,
        # rounding the exponent near -241 changes weights by 6e-14.
        # Allow that arithmetic difference, retaining bitwise thread checks.
        np.testing.assert_allclose(actual['weight'], expected_weight, rtol=1.e-13)
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
    # Test the integrator against the weights it actually received. The
    # independent formula comparison above has its own rounding allowance.
    np.testing.assert_allclose(moments['density'], np.sum(actual['weight'], axis=2),
                               rtol=3.e-15)


def test_selected_profile_response_uses_projected_catalog_mean(initialized):
    """Perturb one shell's halo abundance, then normalize the whole catalog.

    A cluster-lensing signal normalized by the catalog mean,
    signal = sum_shells dchi W_source J01 / (clusters per steradian),
    responds to a background density mode delta_b in two ways: the halo
    abundance in a shell grows as 1 + b delta_b, and so does the catalog
    count that divides it. covariance_ssc_shell_response gives this
    response per unit radial distance analytically (pair window minus the
    catalog-mean window). The reference is a central finite difference:
    perturb the weights of one shell by 1 + b delta with delta = +-1e-6,
    recompute numerator and denominator from the mass samples, and divide
    by 2 delta dchi. rtol = 3e-8 covers the finite-difference error.

    Arguments:
      initialized = the module fixture (forecast and cluster model set).

    Returns:
      nothing; a failed assertion fails the test.

    Side effects:
      sets the HMF amplitude mode of the compiled library to 0.
    """
    ci.init_cluster_hmf_alpha_mode(hmf_alpha_mode=0)
    values = sample_inputs(nmass=96)
    values['a'] = values['a'][2:].copy()
    distance = np.array([0.6, 0.25])
    # A fixed angular mode samples k=(ell+1/2)/distance in each shell.
    modes = np.array([100.5, 1000.5, 10000.5])
    values['k'] = np.ascontiguousarray(modes[None, :]/distance[:, None])
    sampled = ci.covariance_cluster_halo_samples(**values)
    moments = ci.covariance_cluster_moments(**sampled)
    density = moments['density'][:, 0]
    biased_density = moments['biased_density'][:, 0]
    own = moments['J01'][:, 0]
    biased_own = moments['J11'][:, 0]
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
    """Reject malformed numerical inputs before calling the C readers.

    Each case replaces one input of a copy of the valid inputs: a = 1
    (outside a < 1), NaN scale factors, negative wavenumbers, a k table
    with 2 instead of 4 redshift rows, masses beyond the sigma table
    (ln M + 100), zero quadrature weights, and 4 weights for 3 masses.
    Each must raise ValueError (pytest.raises fails the test when it
    does not).

    Arguments:
      initialized = the module fixture (forecast and cluster model set).

    Returns:
      nothing; a failed assertion fails the test.
    """
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
