"""DES cluster 6x2pt + N settings for the shared covariance forecast.

The joint vector holds, in the likelihood order ss, gs, gg, cg, N, cc, cs,
the galaxy and shear blocks of des_cluster_covariance.py plus the cluster
blocks: cluster x galaxy clustering, counts, cluster clustering and
cluster lensing. The clusters are selected in three observed-redshift bins
and four richness bins (richness lambda, the number of red member galaxies,
is the mass proxy). With six lens and four source bins and twenty angular
bins this gives 2800 two-point entries plus 12 counts (2812).

The forecast is a limited model, documented by the shared generator
(cosmolike_notebook_utils.covariance.forecast_cluster): the cNG term treats
clusters as linearly biased tracers of matter, and the counts correlate
with the two-point blocks only through SSC. It therefore does not
reproduce the covariance shipped with the likelihood.
"""

from pathlib import Path

import numpy as np

import des_cluster_covariance as galaxy_survey
from cosmolike_notebook_utils.covariance.forecast_cluster import compute_forecast


def configuration(accuracy_boost=None, ytransform=True, gaussian=None,
                  **accuracy_overrides):
    """Return DES physical choices and one resolved covariance accuracy boost.

    Starts from the galaxy/shear settings of des_cluster_covariance.py and
    adds the cluster entries. The Y localization (Park, Rozo & Krause 2021)
    replaces gamma_t of each cluster-lensing row by Y(R) = Sigma(R) -
    Sigma(R_max), a fixed linear combination of the angular bins that
    removes the dependence on the mass inside the smallest radius; the
    likelihood's mean data vector uses it.

    Arguments:
        accuracy_boost = None uses the YAML; 1, 2, 4 or 8 refines tables/cutoffs.
        ytransform = whether to localize cluster lensing as the DES mean does.
        gaussian = optional Gaussian model mapping; None means
            {"nonlimber": False, "ia": "none"}, the only choice this joint
            forecast accepts.
        accuracy_overrides = internal refinement controls from default.yaml
            (every other keyword argument of the call).
    Returns:
        Settings mapping with the project YAML baseline and explicit model limits.
    Raises:
        ValueError when gaussian asks for non-Limber spectra or intrinsic
        alignment.
    """
    if gaussian is None:
        gaussian = {"nonlimber": False, "ia": "none"}
    settings = galaxy_survey.configuration(
        accuracy_boost=accuracy_boost, gaussian=gaussian, **accuracy_overrides,
    )
    model = settings["gaussian"]
    if model["nonlimber"] or model["ia"] != "none":
        raise ValueError("joint selected-cluster covariance still requires "
                         "gaussian.nonlimber=false and gaussian.ia=none; "
                         "use the galaxy/shear adapter for Gaussian NL/IA")
    settings.update({
        'cluster_file': 'data/des_y6_cluster.nz',
        'cluster_richness_edges': np.array([20., 30., 45., 60., 500.]),
        'cluster_redshift_edges': np.array([0.2, 0.4, 0.55, 0.65]),
        'cluster_mor': [4.26, 0.943, 0.15, 0.207],
        'cluster_hmf_alpha_mode': 0,
        'cluster_lnm_bounds': np.log(np.array([1.e12, 1.e16])),
        'cluster_ytransform': ytransform,
        'cg_lens_bin': [0, 1, 2],
        # cluster_mor = ln lambda_0, A_lambda, sigma_int, B_lambda (Table I
        # fiducials); cluster_lnm_bounds = ln of the cluster mass range
        # 1e12 to 1e16 M_sun/h; cg_lens_bin[i] = the lens bin (zero-based)
        # paired with cluster bin i in cg.
        # Include selection support and catalog-bin edges in the radial rule:
        # the radial quadrature panels end at the support of the selection
        # kernels (z = 0.1565 to 0.7095, the range of des_y6_cluster.nz) and
        # at the cluster bin edges 0.2, 0.4, 0.55, 0.65, so no kink of the
        # integrand falls inside a panel.
        'a_edges': 1.0/(1.0+np.array([3.1, 2., 1.5, 1., .7095, .65,
                                     .55, .4, .2, .1565, .1, 1.e-5])),
    })
    return settings


def initialize(interface, settings):
    """Initialize the forecast without reading a likelihood covariance or mask.

    The angular edges are checked first: the mean Y operator needs at least
    five logarithmic angular bins (equal steps in ln theta).

    Arguments:
        interface = imported cosmolike_des_cluster_interface.
        settings = configuration() output, optionally with explicit refinements.
    Returns:
        CAMB input tables for saving beside the forecast.
    Raises:
        FileNotFoundError when the cluster redshift file is missing;
        ValueError when the angular edges are not at least six finite,
        positive, increasing, logarithmically spaced values.
    Side effects:
        Replaces the core cosmology and galaxy/source/cluster state. Sets
        zero IA, magnification, RSD and environmental selection correction.
        Cluster windows are abundance-weighted (kernel_mode 1) and the MOR
        is lognormal.
    """
    project = Path(__file__).resolve().parents[1]
    cluster_file = project/settings['cluster_file']
    if not cluster_file.is_file():
        raise FileNotFoundError(f"cluster redshift file does not exist: {cluster_file}")
    angles = np.asarray(settings['theta_edges_arcmin'], dtype=float)
    if (angles.ndim != 1 or len(angles) < 6 or np.any(angles <= 0.0)
            or not np.all(np.isfinite(angles)) or np.any(np.diff(angles) <= 0.0)):
        raise ValueError(
            "cluster angular bins need at least five finite positive intervals"
        )
    steps = np.diff(np.log(angles))
    if not np.allclose(steps, steps[0], rtol=1.e-12, atol=0.0):
        raise ValueError("the cluster mean Y operator requires logarithmic angular bins")

    tables = galaxy_survey.initialize(interface=interface, settings=settings)
    interface.reset_cluster()
    interface.init_binning(ntheta_bins=len(angles)-1,
                            theta_min_arcmin=angles[0], theta_max_arcmin=angles[-1])
    interface.init_cluster_probes(N=1, cs=1, cc=1, cg=1)
    interface.init_survey_parameters(surveyname='DES cluster covariance forecast',
                                     area=settings['area_deg2'], sigma_e=0.0)
    interface.init_cluster_model(mor_model=0, kernel_mode=1, selection_model=0,
                                 ytransform=int(settings['cluster_ytransform']),
                                 include_ia=0, magnification=0.0)
    interface.init_cluster_hmf_alpha_mode(
        hmf_alpha_mode=settings['cluster_hmf_alpha_mode'],
    )
    richness = settings['cluster_richness_edges']
    redshift = settings['cluster_redshift_edges']
    interface.init_cluster_richness_bins(lambda_min=richness[:-1].copy(),
                                         lambda_max=richness[1:].copy())
    interface.set_cluster_zdist(nofz=np.loadtxt(cluster_file),
                                zbin_min=redshift[:-1].copy(),
                                zbin_max=redshift[1:].copy())
    interface.init_cluster_pairs(cg_lens_bin=settings['cg_lens_bin'])
    interface.set_nuisance_cluster_mor(MOR=settings['cluster_mor'])
    return tables


def compute(interface, settings, progress=None, backend=None):
    """Compute the shared angular forecast in the full DES 2812-entry order.

    Arguments:
        interface = project initialized with these settings.
        settings = configuration() output.
        progress = optional callback receiving (stage, elapsed_seconds).
        backend = None for notebook wrappers, interface.covariance for CLI.
    Returns:
        Separate G/SSC/cNG/total matrices, layout, mean signals and model limits.
        Apply valid_indices before a positivity check when Y is enabled:
        the last angular bin of each of the 48 cluster-lensing rows (3
        cluster z bins x 4 richness bins x 4 source bins) is Sigma(R_max) -
        Sigma(R_max) = 0 by construction, so those 48 entries have zero
        variance and stay in the full returned layout.
        Physical scale cuts and inference convergence are separate steps.
    """
    return compute_forecast(
        interface=interface, settings=settings, progress=progress, backend=backend,
    )
