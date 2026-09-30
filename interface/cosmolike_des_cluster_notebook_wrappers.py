"""Notebook wrappers for the des_cluster Cosmolike interface (barebones).

Only what a notebook accuracy check needs: set up the compiled
interface once, then evaluate the masked data vector or its chi2 at a
point with the numerical knobs of your choice, and compare against
the default knobs:

    import cosmolike_des_cluster_notebook_wrappers as nw
    nw.init_cosmolike(CLprobe="3x2pt", with_data=True)
    chi2_default = nw.get_chi2()
    chi2_boosted = nw.get_chi2(AccuracyBoost=2.0)

The model is NLA intrinsic alignments with no baryonic contamination,
the configuration of EXAMPLE_EVALUATE1/2.yaml.

The compiled interface is a C library with global state: every init_*
and set_* call replaces part of it. Each wrapper therefore rebuilds
everything it depends on (tables, accuracy, cosmology, nuisances) on
every call, so no call depends on which wrapper ran before it.
"""

import os
import sys

import numpy as np
from getdist import IniFile

# the shared notebook utilities live in cosmolike_core; the compiled
# interface is on the path already (each project's interface/
# directory is part of the Cocoa PYTHONPATH)
sys.path.insert(0, os.environ["ROOTDIR"] + "/external_modules/code/cosmolike_core")
import cosmolike_notebook_utils as cnu
import cosmolike_des_cluster_interface as ci


# ----------------------------------------------------------------------
# Project fiducial point (the evaluate override of the example yamls;
# the lens values are the params_lens_redmagic.yaml reference point)
# ----------------------------------------------------------------------
As_1e9 = 2.1
ns = 0.96605
H0 = 67.32
omegab = 0.04
omegam = 0.3
mnu = 0.06
w = -1.0
w0pwa = -1.0

A1_FID = [-0.7, -1.7, 0, 0]                  # NLA amplitude, z power law
SHEAR_PHOTOZ_FID = [0.0, 0.0, 0.0, 0.0]
M_FID = [-0.0063, -0.0198, -0.0241, -0.0369]
LENS_PHOTOZ_FID = [0.006, 0.001, 0.004, -0.002, -0.007]
LENS_STRETCH_FID = [1.0, 1.0, 1.0, 1.0, 1.23]  # redMaGiC photo-z stretch
B1_FID = [1.7, 1.7, 1.7, 2.0, 2.0]
BMAG_FID = [0.63, -3.04, -1.33, 2.50, 1.93]    # fixed magnification
PM_FID = [0.0, 0.0, 0.0, 0.0, 0.0]

# NLA has no A2 / b_TA terms and the bias model uses b1 and bmag only,
# but the interface setters take every vector
ZEROS4 = [0, 0, 0, 0]
ZEROS5 = [0, 0, 0, 0, 0]

# ----------------------------------------------------------------------
# Configuration mirrored from the example yamls
# ----------------------------------------------------------------------
_CONFIG = {
    "lmax": 75000,              # base of the internal C_ell tables
    "non_linear_emul": 2,       # 1 = EuclidEmulator2, 2 = halofit
    "path": "../../external_modules/data/des_cluster",
    "data_file": "des_y3_real.dataset",
    "IA_redshift_evolution": 3,
    # [b1, b2, bs2, b3, bmag] evolution flags (see bias.c)
    "bias_model": [0, 0, 0, 1, 0],
}


def init_cosmolike(CLprobe="3x2pt", with_data=True, lmax=None):
    """One-time interface setup, in the order the likelihood runs it.

    Arguments:
      CLprobe   = "xi", "2x2pt" or "3x2pt".
      with_data = True also loads covariance, mask, and data vector
                  (get_chi2 needs them).
      lmax      = base lmax of the internal C_ell tables, or None for
                  the _CONFIG value.

    Returns:
      the parsed IniFile of the .dataset file.
    """
    if lmax is None:
        lmax = _CONFIG["lmax"]
    ini = IniFile(os.path.normpath(
        os.path.join(_CONFIG["path"], _CONFIG["data_file"])))
    ci.initial_setup()
    ci.init_probes(possible_probes=CLprobe)
    ci.init_binning(int(ini.int("n_theta")),
                    ini.float("theta_min_arcmin"),
                    ini.float("theta_max_arcmin"))
    ci.init_cosmo_runmode(is_linear=False)
    ci.init_IA(ia_model=0,
               ia_redshift_evolution=int(_CONFIG["IA_redshift_evolution"]),
               ia_code=0)
    ci.init_redshift_distributions_from_files(
        lens_multihisto_file=ini.relativeFileName('nz_lens_file'),
        lens_ntomo=int(ini.int("lens_ntomo")),
        source_multihisto_file=ini.relativeFileName('nz_source_file'),
        source_ntomo=int(ini.int("source_ntomo")))
    if with_data:
        ci.init_data_real(ini.relativeFileName('cov_file'),
                          ini.relativeFileName('mask_file'),
                          ini.relativeFileName('data_file'))
    if CLprobe != "xi":
        ci.init_bias(bias_model=_CONFIG["bias_model"])
    ci.init_ntable_lmax(lmax=int(lmax))
    ci.init_accuracy_boost(1.0, int(1))
    return ini


def _set_state(omegam, omegab, H0, ns, As_1e9, w, w0pwa,
               AccuracyBoost, kmax, k_per_logint, CAMBAccuracyBoost,
               CLAccuracyBoost, CLIntegrationAccuracy):
    """Runs CAMB and pushes one complete state into the interface.

    The accuracy folds are the house convention: the overall
    AccuracyBoost multiplies the cosmolike boost, and the integration
    accuracy and the C_ell table length grow with it. The nuisance
    parameters are always the module fiducials.
    """
    (log10k_interp_2D, z_interp_2D, lnPL, lnPNL,
     G_growth, z_interp_1D, chi) = cnu.get_camb_cosmology(
        omegam=omegam, omegab=omegab, H0=H0, ns=ns, As_1e9=As_1e9,
        w=w, w0pwa=w0pwa, mnu=mnu, AccuracyBoost=AccuracyBoost,
        kmax=kmax, k_per_logint=k_per_logint,
        CAMBAccuracyBoost=CAMBAccuracyBoost,
        CLAccuracyBoost=CLAccuracyBoost,
        non_linear_emul=_CONFIG["non_linear_emul"])

    CLAccuracyBoost = CLAccuracyBoost * AccuracyBoost
    CLIntegrationAccuracy = max(
        0, CLIntegrationAccuracy + abs(3*(CLAccuracyBoost - 1.0)))
    ci.init_ntable_lmax(int(_CONFIG["lmax"] + 20000*(CLAccuracyBoost - 1)))
    ci.init_accuracy_boost(CLAccuracyBoost, int(CLIntegrationAccuracy))
    ci.init_bias(bias_model=_CONFIG["bias_model"])

    ci.set_cosmology(omegam=omegam,
                     H0=H0,
                     log10k_2D=log10k_interp_2D,
                     z_2D=z_interp_2D,
                     lnP_linear=lnPL,
                     lnP_nonlinear=lnPNL,
                     G=G_growth,
                     z_1D=z_interp_1D,
                     chi=chi)

    ci.set_nuisance_shear_calib(M=M_FID)
    ci.set_nuisance_shear_photoz(bias=SHEAR_PHOTOZ_FID)
    ci.set_nuisance_clustering_photoz(bias=LENS_PHOTOZ_FID,
                                      stretch=LENS_STRETCH_FID)
    ci.set_nuisance_bias(B1=B1_FID, B2=ZEROS5, B_MAG=BMAG_FID,
                         B3nl=ZEROS5, BK=ZEROS5)
    ci.set_nuisance_ia(A1=A1_FID, A2=ZEROS4, B_TA=ZEROS4)
    ci.set_point_mass(PMV=PM_FID)
    ci.reset_bary_struct()


def get_datavector(omegam=omegam, omegab=omegab, H0=H0, ns=ns,
                   As_1e9=As_1e9, w=w, w0pwa=w0pwa,
                   AccuracyBoost=1.0, kmax=7.5, k_per_logint=10,
                   CAMBAccuracyBoost=1.0, CLAccuracyBoost=1.0,
                   CLIntegrationAccuracy=0):
    """Masked theory data vector (masked entries are zero).

    Requires init_cosmolike first. The accuracy arguments are the
    knobs an accuracy check varies (see _set_state for the folds).

    Returns:
      1D numpy array, the full data-vector length.
    """
    _set_state(omegam, omegab, H0, ns, As_1e9, w, w0pwa,
               AccuracyBoost, kmax, k_per_logint, CAMBAccuracyBoost,
               CLAccuracyBoost, CLIntegrationAccuracy)
    return np.array(ci.compute_data_vector_masked())


def get_chi2(**kwargs):
    """chi2 of the masked theory vector against the loaded data.

    Requires init_cosmolike(with_data=True) first. Keyword arguments
    are those of get_datavector.

    Returns:
      float chi2.
    """
    return ci.compute_chi2(get_datavector(**kwargs))
