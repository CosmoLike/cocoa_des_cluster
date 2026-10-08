"""Survey settings of the des_cluster galaxy/shear covariance forecast.

The covariance matrix C of a data vector gives the expected scatter of each
entry and the correlation between entries; the likelihood's chi2 uses C^-1.
The shared code in cosmolike_notebook_utils.covariance computes C as the
sum of three parts, kept separate for inspection:

  G    Gaussian: products of two-point functions, signal and noise;
  SSC  super-sample covariance: the joint response of every observable to
       density modes larger than the survey;
  cNG  connected non-Gaussian: the halo-model four-point (trispectrum) term
       of modes inside the survey.

This module holds only the project choices for the galaxy and shear blocks
ss, gs and gg: the six MagLim lens and four source n(z) files, number
densities, shape noise, galaxy bias, angular bins, Fourier bands and a
forecast cosmology. des_cluster_joint_covariance.py extends these settings
to the joint 6x2pt + N vector, and compute_covariance.py and the notebook
EXAMPLE_EVALUATE_COVARIANCE.ipynb call them. The forecast uses massless
neutrinos, its own cosmology and explicit Gaussian non-Limber and
intrinsic-alignment choices, so it does not reproduce the covariance
shipped with the likelihood (data/des_cluster_y6_cov.npy).
covariance/README.md records the catalog assumptions and their sources.
"""

from pathlib import Path

import numpy as np

from cosmolike_notebook_utils.covariance.forecast import (
    initialize_forecast,
    gaussian_model,
    compute_forecast,
)
from cosmolike_notebook_utils import covariance as cov


def configuration(accuracy_boost=None, gaussian=None, **accuracy_overrides):
    """Return resolved survey, cosmology and YAML accuracy choices.

    The covariance README records the catalog assumptions and their sources.
    The n(z) files fix only the shape of each bin's redshift distribution;
    the number densities below fix how many galaxies a bin holds, which
    sets its shot and shape noise.

    Arguments:
        accuracy_boost = None uses default.yaml; 1, 2, 4 or 8 refines it.
        gaussian = optional nonlimber/ia/A1/A2/B_TA model mapping (see
            gaussian_model in cosmolike_notebook_utils.covariance.forecast).
        accuracy_overrides = named internal controls from default.yaml;
            **accuracy_overrides collects every other keyword argument of
            the call into this dictionary.
    Returns:
        Fully resolved settings, a dictionary: the cosmology, file names,
        survey and binning entries below, the accuracy entries of
        default.yaml (including the unboosted accuracy parameters) and the
        resolved "gaussian" model.
    Raises:
        TypeError for an unknown accuracy control and ValueError for an
        invalid gaussian mapping (both from the shared code).
    """
    numerical = cov.load_covariance_accuracy(
        filename=Path(__file__).with_name("default.yaml"),
        accuracy_boost=accuracy_boost, **accuracy_overrides,
    )

    # 15 Fourier bands with logarithmic edges from l = 30 to 4000. Band i
    # covers band_first[i] ... band_last[i] = band_first[i+1] - 1 inclusive,
    # so every integer multipole is counted once.
    band_edges = np.rint(np.geomspace(30, 4001, 16)).astype(np.int32)

    # The fiducial is shared by G, SSC and cNG; CAMB runs only once. It is a
    # forecast cosmology, not the fiducial point of the synthetic data:
    # massless neutrinos (mnu = 0) and the Takahashi halofit for the
    # nonlinear spectrum.
    settings = {
        "cosmology": {
            "omegam": 0.3,
            "omegab": 0.05,
            "H0": 70.0,
            "ns": 0.965,
            "As_1e9": 2.1,
            "w": -1.0,
            "w0pwa": -1.0,
            "mnu": 0.0,
            "AccuracyBoost": 1.0,
            "CLAccuracyBoost": 1.0,
            "CAMBAccuracyBoost": 1.0,
            "kmax": 20.0,
            "k_per_logint": 20,
            "non_linear_emul": 2,
            "lens_potential_accuracy": 1.0,
            "halofit_version": "takahashi",
        },

        # File columns describe radial shapes; these flags fix their z
        # convention as in the likelihood: photoz_zmid = 0 reads the z column
        # as left bin edges.
        "lens_file": "data/des_y6_maglim.nz",
        "source_file": "data/des_y6_source.nz",
        "photoz_interpolation": 0,
        "photoz_zmid": 0,
        # Unit width factors keep the supplied lens n(z) shapes unchanged.
        "lens_photoz_stretch": [1.0]*6,

        # Measured pair and band choices are fixed during accuracy refinement.
        # excluded_gammat lists (lens, source) pairs left out of gs (none);
        # lnm_edges are the ln(M/[M_sun/h]) panels of the halo-mass integrals.
        "excluded_gammat": [],
        "band_first": band_edges[:-1],
        "band_last": band_edges[1:]-1,
        "lnm_edges": cov.halo_mass_edges(),

        # Densities are per square arcminute. Shape noise is per component:
        # the dispersion 0.384666 of both ellipticity components together,
        # divided by sqrt(2). Area, densities and dispersion are those of
        # scripts/make_synthetic_data.py (4143 deg^2 is the DES Y3 cluster
        # footprint); the bias values are the Table I fiducials of the six
        # MagLim bins (arXiv:2503.13631).
        "area_deg2": 4143.0,
        "lens_density_arcmin2": [0.1380, 0.1016, 0.1071, 0.1381, 0.1054, 0.1045],
        "source_density_arcmin2": [2.1402, 2.14455, 2.1518, 2.11845],
        "sigma_e_component": [0.384666/np.sqrt(2.0)]*4,
        "bias": [1.42, 1.66, 1.70, 1.62, 1.78, 1.75],

        # 20 logarithmic angular bins, 2.5 to 250 arcmin, as in the data
        # vector. a_edges are the panels of the radial (line-of-sight)
        # Gauss-Legendre quadrature in scale factor a = 1/(1+z). Shell edges
        # increase in a, from the distant boundary (z = 3.1, beyond the n(z)
        # tails) to z = 1e-5 next to the observer.
        "theta_edges_arcmin": np.geomspace(start=2.5, stop=250.0, num=21),
        "a_edges": 1.0/(1.0+np.array([3.1, 2., 1.5, 1., .7, .4, .2, 1.e-5])),
    }
    settings.update(numerical)
    settings["gaussian"] = gaussian_model(
        gaussian=gaussian, nsource=len(settings["source_density_arcmin2"]),
    )
    return settings


def initialize(interface, settings):
    """Run CAMB once and install the complete forecast state without a covariance.

    "Install" means: pass the cosmology, n(z), densities and binning of
    settings to the compiled library, whose global state the covariance
    calls then read.

    Arguments:
        interface = imported cosmolike_des_cluster_interface module.
        settings = resolved mapping from configuration().
    Returns:
        CAMB input tables as a dict, suitable for saving beside results.
    Side effects:
        Replaces the interface's global cosmology and nuisance state. The
        likelihood covariance, data vector and mask are never loaded.
    """
    return initialize_forecast(
        interface=interface, settings=settings,
        project=Path(__file__).resolve().parents[1],
    )


def compute(interface, settings, space="real", rows=None, progress=None,
            backend=None):
    """Return the galaxy/shear forecast with G, SSC, connected and total matrices.

    The full real-space layout has 1000 entries: xi_+ and xi_- of 10 source
    pairs (400), gamma_t of 24 lens-source pairs (480) and w of 6 lens bins
    (120), 20 angular bins each. The Fourier layout has 600 entries: the
    same 10 + 24 + 6 spectra in 15 bands.

    Arguments:
        interface = the compiled project module, initialized by initialize().
        settings = the configuration() output.
        space = "real" or "fourier".
        rows = optional subset of measured rows; None keeps all.
        progress = optional callback receiving (stage, elapsed_seconds).
        backend = None for the notebook wrappers, interface.covariance for
            the command-line route (compute_covariance.py).
    Returns:
        the shared forecast dictionary, including resolved settings and
        coordinates.
    """
    return compute_forecast(
        interface=interface, settings=settings, space=space, rows=rows,
        progress=progress, backend=backend,
    )
