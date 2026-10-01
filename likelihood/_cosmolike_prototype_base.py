# Python 2/3 compatibility - must be first line
from __future__ import absolute_import, division, print_function
import os
import numpy as np
import scipy
from scipy.interpolate import interp1d
import sys
import time
import functools
from collections.abc import Mapping

# Local
from cobaya.likelihoods.base_classes import DataSetLikelihood
from cobaya.log import LoggedError
from getdist import IniFile

import euclidemu2 as ee2
import math

from contextlib import contextmanager
@contextmanager
def timer(label):
  t0 = time.perf_counter()
  yield
  print(f"{label}: {time.perf_counter() - t0:.4f}s")

import cosmolike_des_cluster_interface as ci

COSMOLIKE_OMP_THREADS = int(os.environ.get("OMP_NUM_THREADS", 1))

def with_omp_threads(fn):
    """
    WHY THIS EXISTS
    ---------------
    Cosmolike's hot loops are parallelized with OpenMP and rely on
    omp_get_max_threads() returning the value set by OMP_NUM_THREADS
    However, some Python libraries silently call omp_set_num_threads(1). 
    This globally drops the OpenMP thread count, forcing cosmolike's 
    parallel-for regions to run on a single core afterwards.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        ci.set_omp_threads(COSMOLIKE_OMP_THREADS)
        return fn(*args, **kwargs)
    return wrapper

survey = "DES"

# ----------------------------------------------------------------------------
# Clusters (4x2pt + N, arXiv 2503.13631)
# ----------------------------------------------------------------------------
# Probe names handled by ci.init_probes_cluster: the joint vector
# ss, gs, gg, cg, N, cc, cs (every other probe goes through ci.init_probes).
CLUSTER_PROBES = ("4x2pt_n", "6x2pt_n", "n", "n_cc", "n_cs", "cs", "cc", "cg")

# Mass-observable relation, eqs (18)-(19), in the order of cluster.mor:
# ln lambda_0, A_lambda (slope in ln M), sigma_int, B_lambda (slope in
# ln(1+z)). No neutral default exists: every name must be a parameter.
CLUSTER_MOR_PARAMS = [survey+"_CL_LNLAMBDA0", survey+"_CL_A_LAMBDA",
                      survey+"_CL_SIGMA_INT", survey+"_CL_B_LAMBDA"]

# Selection bias in the order of cluster.selection. Selection model 2 (Y6,
# eq 23): b_s1, b_s2, r_0 [comoving Mpc/h], s3 (power of (1+zbar)/1.45).
# Selection model 1 (Y1): the same slots hold b_s0, b_s1 (mass slope),
# b_s2 (power of (1+z)/1.45) and an unused entry. The defaults of a missing
# name switch the selection bias off (factor 1) in either model; under Y6,
# r_0 only divides b_s2's term, so its default is irrelevant at b_s2 = 0
# (it must still be positive).
CLUSTER_SELECTION_PARAMS = [survey+"_CL_BS1", survey+"_CL_BS2",
                            survey+"_CL_R0",  survey+"_CL_BSZ"]
CLUSTER_SELECTION_DEFAULTS = {2: [1.0, 0.0, 30.0, 0.0],  # Y6
                              1: [1.0, 0.0, 0.0, 0.0],   # Y1
                              0: [1.0, 0.0, 0.0, 0.0]}   # none

def _ini_list(ini, key, tp):
  """Comma- (or space-) separated dataset entry as a list of tp."""
  return [tp(x) for x in ini.string(key).replace(",", " ").split()]

class _cosmolike_prototype_base(DataSetLikelihood):

  @classmethod
  def get_modified_defaults(cls, defaults, input_options={}):
    """Apply the yaml option `fixed_params` to the default parameters.

    `params: !defaults [...]` cannot be extended in the same yaml, so a combo
    lists the parameters it fixes under `fixed_params` (e.g. CL+GC fixes the
    lens bins it does not use). Each entry replaces the parameter's default
    info with the cobaya merge rule: a value drops prior, ref and proposal.
    A user yaml can override `fixed_params` like any other option.
    """
    fixed = input_options.get("fixed_params", defaults.get("fixed_params"))
    params = defaults.get("params") or {}
    for p, info in (fixed or {}).items():
      old = params.get(p)
      new = {}
      if isinstance(old, Mapping):
        new = {k: v for k, v in old.items() if k not in ("prior", "ref", "proposal")}
      if isinstance(info, Mapping):
        new.update(info)
      else:
        new["value"] = info
      params[p] = new
    if params:
      defaults["params"] = params
    return defaults

  def initialize(self, probe):
    ini = IniFile(os.path.normpath(os.path.join(self.path, self.data_file)))
    self.probe = probe
    self.has_clusters = probe.strip().lower() in CLUSTER_PROBES
    self.data_vector_file = ini.relativeFileName('data_file')
    self.cov_file = ini.relativeFileName('cov_file')
    self.mask_file = ini.relativeFileName('mask_file')
    self.lens_file = ini.relativeFileName('nz_lens_file')
    self.source_file = ini.relativeFileName('nz_source_file')
    self.lens_ntomo = ini.int("lens_ntomo")
    self.source_ntomo = ini.int("source_ntomo")
    self.ntheta = ini.int("n_theta")
    self.theta_min_arcmin = ini.float("theta_min_arcmin")
    self.theta_max_arcmin = ini.float("theta_max_arcmin")

    # ------------------------------------------------------------------------   
    tmp=int(1000 + 250*self.accuracyboost)
    self.z_interp_1D = np.concatenate((np.linspace(0.0,3.0,max(100,int(0.80*tmp)),endpoint=False),
                                       np.linspace(3.0,50.1,max(100,int(0.40*tmp)),endpoint=False),
                                       np.linspace(1070,1100,max(50,int(0.10*tmp)))),axis=0)
    self.len_z_interp_1D = len(self.z_interp_1D)

    # The z nodes of the 2D power-spectrum tables handed to cosmolike,
    # which interpolates LINEARLY in z between exactly these nodes (its
    # piecewise-uniform direct indexing uses the handed grid; there is
    # no internal regridding). Linear interpolation leaves a sawtooth-
    # shaped O(dz^2) residual that vanishes at the nodes, so two grids
    # that do not share nodes disagree by the FULL residual amplitude.
    # The previous count, min(120 + 20*boost, 250), re-phased that
    # sawtooth at every boost value: measured in roman_kl, order-unity
    # chi2 jitter in its clustering vector, and smaller but equally
    # non-convergent re-phasing shifts in this project. The dyadic factor m = 2^ceil(log2(boost)) below
    # refines each uniform block by an integer factor with the same
    # endpoints, so (a) every block stays uniform (cosmolike keeps its
    # two-segment direct indexing, no search), (b) every coarser
    # grid's nodes are a subset of every finer grid's nodes, making a
    # boost increase a true refinement (error falls like 1/m^2, no
    # re-phasing), and (c) boost 1 reproduces the previous 140-node
    # grid exactly, so results at the default accuracy are unchanged.
    # The low block multiplies its node count (endpoint=False, spacing
    # 3/n); the high block multiplies its INTERVAL count
    # (endpoint=True: 35 nodes = 34 intervals -> 34*m + 1 nodes).
    # zmax of the hybrid emulator is 50
    #
    # pk_z_refinement multiplies m on top of the accuracy boost (which stays
    # global: it still sets m as before). A Fourier-space data vector reads
    # P(k, z) at fixed multipoles, where the linear z interpolation residual
    # does not average out as it does in real space: roman_fourier's 3x2pt
    # chi2 moves by 0.25, 0.030, 0.002 from m = 1 to 2, 4, 8 (measured
    # 2026-10-01), roman_real's and lsst_y1's by <= 0.004 from 1 to 2.
    zref = getattr(self, "pk_z_refinement", 1)
    if not (float(zref) == int(zref) and int(zref) >= 1):
      raise LoggedError(self.log, "pk_z_refinement = %s: must be a positive "
                        "integer", zref)
    m = int(min(2**np.ceil(np.log2(max(1.0, self.accuracyboost))), 16))
    m = m*int(zref)
    self.z_interp_2D = np.concatenate((np.linspace(0,3.0,105*m,endpoint=False), 
                                       np.linspace(3.0,49.99,34*m + 1)),axis=0)
    self.len_z_interp_2D = len(self.z_interp_2D)
    # CAMB's transfer module caps the number of requested redshifts at
    # 256, so the list handed to CAMB through the Pk_interpolator
    # requirement stays at this boost-independent 140-node grid (the
    # m = 1 grid above). The denser nested nodes only re-evaluate the
    # smooth z-spline CAMB builds from these transfer redshifts when
    # the cosmolike tables are filled, so raising the boost refines
    # exactly the table resampling that produced the jitter, and the
    # CAMB side never exceeds its cap.
    self.z_interp_2D_camb = np.concatenate((np.linspace(0,3.0,105,endpoint=False), 
                                            np.linspace(3.0,49.99,35)),axis=0)
    
    self.log10k_interp_2D = np.linspace(-4.99,2.0,int(1250+250*self.accuracyboost))
    self.len_log10k_interp_2D = len(self.log10k_interp_2D)
    # ------------------------------------------------------------------------

    ci.initial_setup()
    ci.reset_cluster() # cluster defaults + fresh cluster cache keys
    if self.has_clusters:
      # like.* for ss/gs/gg and cluster.probe_* for cg/N/cc/cs
      ci.init_probes_cluster(possible_probes=self.probe)
    else:
      ci.init_probes(possible_probes=self.probe)
    ci.init_binning(int(self.ntheta), self.theta_min_arcmin, self.theta_max_arcmin)

    if self.debug:
      ci.set_log_level_debug()
    else:
      ci.set_log_level_info()

    ci.init_photoz_conventions(
        interpolation_type=int(getattr(self, "photoz_interpolation_type", 0)),
        zmid_convention=int(getattr(self, "photoz_zmid_convention", 0)))

    ci.init_fpt_internal_boost(
        internal_boost=float(getattr(self, "internal_accuracyboost", 1.0)))

    # the non-Limber FFTLog chi grid, refined on top of the accuracy boost
    # (narrow lens bins need it: see init_nonlimber_accuracy_boost)
    ci.init_nonlimber_accuracy_boost(
        nonlimber_boost=float(getattr(self, "nonlimber_accuracyboost", 1.0)))

    ci.init_adopt_limber_gs(
        adopt_limber_gs=int(getattr(self, "adopt_limber_gs", 1)))

    ci.init_adopt_limber_gg(
        adopt_limber_gg=int(getattr(self, "adopt_limber_gg", 0)))
    # 0 = perturbative galaxy bias, 1 = halo-model (HOD) galaxy power;
    # always set, so a model never inherits the previous model's value
    ci.init_include_HOD_GX(
        include_HOD_GX=int(getattr(self, "include_HOD_GX", 0)))
    # 0 = the init_IA model, 1 = halo-model IA (Fortuna et al. 2021)
    ci.init_include_halo_IA(
        include_halo_IA=int(getattr(self, "include_halo_IA", 0)))
    # density field of the halo model's peak height: 0 = total matter,
    # 1 = cold dark matter + baryons (sigma(M) from the linear P_cb, and
    # rho_crit (Omega_m - Omega_nu) in R(M) and in the rho/M of dn/dM; see
    # get_neutrino_inputs); always set, so a model never inherits the
    # previous model's value
    self.halo_matter_field = int(getattr(self, "halo_matter_field", 0))
    if self.halo_matter_field not in (0, 1):
      raise LoggedError(self.log, "halo_matter_field = %d: must be 0 (total "
                        "matter) or 1 (cold dark matter + baryons)",
                        self.halo_matter_field)
    ci.init_halo_matter_field(halo_matter_field=self.halo_matter_field)
    if (self.halo_matter_field == 1) and (self.use_emulator == 2):
      self.log.info("halo_matter_field = 1 with use_emulator = 2: the "
                    "emulators have no cold dark matter + baryon spectrum, "
                    "so P_cb = P_lin/(1 - f_nu)^2 (an approximation; see "
                    "get_neutrino_inputs)")

    if self.has_clusters and (self.use_emulator == 1):
      raise LoggedError(self.log, "probe %s: clusters have no emulator path",
                        self.probe)
    if self.has_clusters and (self.use_baryon_pca or self.create_baryon_pca):
      raise LoggedError(self.log, "probe %s: baryon PCAs are not supported "
                        "with clusters", self.probe)

    if self.use_emulator == 1:
      ci.init_redshift_distributions_from_files(
          lens_multihisto_file=self.lens_file,
          lens_ntomo=int(self.lens_ntomo), 
          source_multihisto_file=self.source_file,
          source_ntomo=int(self.source_ntomo))
      ci.init_data_real(self.cov_file, self.mask_file, self.data_vector_file)  
      ci.init_accuracy_boost(accuracy_boost=0.35, 
                             integration_accuracy=-1) # seems enough to compute PM
    else:
      ci.init_ntable_lmax(lmax=int(self.lmax))
      ci.init_accuracy_boost(accuracy_boost=self.accuracyboost, 
                             integration_accuracy=int(self.integration_accuracy))
      ci.init_cosmo_runmode(is_linear=False)

      if self.external_nz_modeling: 
        (self.lens_nz, self.source_nz) = ci.read_redshift_distributions(
            lens_multihisto_file = self.lens_file,
            lens_ntomo = int(self.lens_ntomo), 
            source_multihisto_file = self.source_file,
            source_ntomo = int(self.source_ntomo)
          ) 
        ci.init_lens_sample_size(int(self.lens_ntomo))
        ci.init_source_sample_size(int(self.source_ntomo))
        ci.init_ntomo_powerspectra() # must be called after set_source/lens_size  
      else:
        ci.init_redshift_distributions_from_files(
          lens_multihisto_file=self.lens_file,
          lens_ntomo=int(self.lens_ntomo), 
          source_multihisto_file=self.source_file,
          source_ntomo=int(self.source_ntomo))
      
      if self.has_clusters:
        # cluster model, richness bins, selection kernels and pairs: they
        # fix the block sizes of the joint vector read below
        self.init_cluster_related(ini)
        ci.init_data_cluster(self.cov_file, self.mask_file, self.data_vector_file)
      else:
        ci.init_data_real(self.cov_file, self.mask_file, self.data_vector_file)

      if (int(self.IA_model) == 0) and (int(self.IA_code) == 1):
   		# Fall back to C FASTPT under NLA
        self.IA_code = 0
      ci.init_IA(ia_model = int(self.IA_model), 
                ia_redshift_evolution = int(self.IA_redshift_evolution),
                ia_code = int(self.IA_code))

      if self.probe != "xi":
        # (b1, b2, bs2, b3, bmag). 0 = one amplitude per bin
        ci.init_bias(bias_model=self.bias_model)

      if self.non_linear_emul == 1:
        self.emulator = ee2.PyEuclidEmulator()

      # JVR NOTE: introducing the `external_baryon_suppression` variable to the likelihood
      # This option is excludent with using PCA and adding baryons to DV
      if self.external_baryon_suppression:
          self.use_baryon_pca = False
          self.add_baryons_on_dv = False

      if self.create_baryon_pca:
        self.external_baryon_suppression = False
        self.use_baryon_pca = False
        self.allsims = ini.relativeFileName('all_sims_hdf5_file')
      else:
        if self.add_baryons_on_dv:
          self.external_baryon_suppression = False
          sim = self.which_bsims_add_on_dv
          self.allsims = ini.relativeFileName('all_sims_hdf5_file')
          ci.init_baryons_contamination(sim = sim, allsims=self.allsims)

    if self.use_baryon_pca:
      baryon_pca_file = ini.relativeFileName('baryon_pca_file')
      self.npcs = 4
      ci.set_baryon_pcs(eigenvectors = np.loadtxt(baryon_pca_file))
      self.log.info('use_baryon_pca = True')
      self.log.info('baryon_pca_file = %s loaded', baryon_pca_file)
    else:
      self.log.info('use_baryon_pca = False')

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def init_cluster_related(self, ini):
    """Cluster part of the init chain (4x2pt + N, arXiv 2503.13631).

    Dataset keys: nz_cluster_file (z column, then <phi_i|z> of each cluster
    bin), cluster_ntomo, cluster_zbin_edges, richness_edges, survey_area_deg2,
    cg_lens_bins (one lens bin per cluster bin, -1 = no w_cg). Runs after the
    lens/source n(z) and init_ntomo_powerspectra, before init_data_cluster.
    """
    self.nz_cluster_file = ini.relativeFileName('nz_cluster_file')
    self.cluster_ntomo = ini.int("cluster_ntomo")
    self.cluster_zbin_edges = np.array(_ini_list(ini, "cluster_zbin_edges", float))
    self.richness_edges = np.array(_ini_list(ini, "richness_edges", float))
    self.survey_area_deg2 = ini.float("survey_area_deg2")
    self.cg_lens_bins = _ini_list(ini, "cg_lens_bins", int)

    if len(self.cluster_zbin_edges) != self.cluster_ntomo + 1:
      raise LoggedError(self.log, "cluster_zbin_edges: %d edges for %d bins",
                        len(self.cluster_zbin_edges), self.cluster_ntomo)
    if len(self.cg_lens_bins) != self.cluster_ntomo:
      raise LoggedError(self.log, "cg_lens_bins: %d entries for %d bins",
                        len(self.cg_lens_bins), self.cluster_ntomo)
    if len(self.richness_edges) < 2:
      raise LoggedError(self.log, "richness_edges: at least two edges needed")

    # survey area (deg^2) of the counts, eq (16); sigma_e is read only by
    # the covariance code
    ci.init_survey_parameters(surveyname=survey,
                              area=self.survey_area_deg2,
                              sigma_e=ini.float("sigma_e", 0.0))

    self.cluster_selection_model = int(getattr(self, "cluster_selection_model", 2))
    if self.cluster_selection_model not in CLUSTER_SELECTION_DEFAULTS:
      raise LoggedError(self.log, "cluster_selection_model = %d not supported",
                        self.cluster_selection_model)
    ci.init_cluster_model(
        mor_model=0, # lognormal, eqs (18)-(19)
        kernel_mode=int(getattr(self, "cluster_kernel_mode", 0)),
        selection_model=self.cluster_selection_model,
        ytransform=int(getattr(self, "cluster_ytransform", 1)),
        include_ia=int(getattr(self, "cluster_include_ia", 1)),
        magnification=float(getattr(self, "cluster_magnification", -2.0)))

    # amplitude of the Tinker 2010 cluster mass function: 0 = 0.368 at every
    # z (DES), 1 = alpha(z) from int b f dnu = 1 (halo.c)
    ci.init_cluster_hmf_alpha_mode(
        hmf_alpha_mode=int(getattr(self, "cluster_hmf_alpha_mode", 0)))

    ci.init_cluster_adopt_limber(
        adopt_limber_cc=int(getattr(self, "cluster_adopt_limber_cc", 1)),
        adopt_limber_cg=int(getattr(self, "cluster_adopt_limber_cg", 1)))

    ci.init_cluster_richness_bins(lambda_min=self.richness_edges[:-1].copy(),
                                  lambda_max=self.richness_edges[1:].copy())

    nz_cluster = np.loadtxt(self.nz_cluster_file)
    if nz_cluster.ndim != 2 or nz_cluster.shape[1] != self.cluster_ntomo + 1:
      raise LoggedError(self.log, "%s: expected %d columns (z + %d bins)",
                        self.nz_cluster_file, self.cluster_ntomo + 1,
                        self.cluster_ntomo)
    ci.set_cluster_zdist(nofz=nz_cluster,
                         zbin_min=self.cluster_zbin_edges[:-1].copy(),
                         zbin_max=self.cluster_zbin_edges[1:].copy())

    ci.init_cluster_pairs(cg_lens_bin=self.cg_lens_bins)

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def get_requirements(self):
    if self.use_emulator == 1:
      if self.probe == "xi":
        return {
          'cosmic_shear': None
        }
      elif self.probe == "3x2pt":
        return {
          "H0": None,
          'cosmic_shear': None,
          'ggl': None,
          'wtheta': None,
          'comoving_radial_distance': {
            "z": self.z_interp_1D 
          } # in Mpc
        }
      elif self.probe == "xi_gg":
        return {
          'cosmic_shear': None,
          'wtheta': None
        }
      elif self.probe == "xi_ggl":
        return {
          "H0": None,
          'cosmic_shear': None,
          'ggl': None,
          'comoving_radial_distance': {
            "z": self.z_interp_1D
          } # in Mpc
        }
      elif self.probe == "2x2pt":
        return {
          "H0": None,
          'ggl': None,
          'wtheta': None,
          'comoving_radial_distance': {
            "z": self.z_interp_1D 
          } # in Mpc
        }     
    elif self.use_emulator == 2:
      _requirements_ = {
        "As": None,
        "H0": None,
        "omegam": None,
        "omegab": None,
        "Pk_interpolator": {
          "z": self.z_interp_2D_camb,
          "k_max": self.kmax_boltzmann * self.accuracyboost,
          "nonlinear": (True,False),
          "vars_pairs": ([("delta_tot", "delta_tot")])
        },
        "comoving_radial_distance": {
          "z": self.z_interp_1D
        }, # in Mpc
      }
      # Also need Python FAST-PT if IA_code == 1
      if (self.IA_code == 1):
        _requirements_["IA_PS"] = None
        _requirements_["bias_PS"] = None
      if self.non_linear_emul == 1:
        _requirements_["omegab"] = None
        _requirements_["mnu"] = None
        _requirements_["w"] = None
        _requirements_["wa"] = None
      # mnu gives Omega_nu h^2 on this path (get_neutrino_inputs)
      _requirements_["mnu"] = None
      return _requirements_
    else:
      _requirements_ = {
        "As": None,
        "H0": None,
        "omegam": None,
        "omegab": None,
        "Pk_interpolator": {
          "z": self.z_interp_2D_camb,
          "k_max": self.kmax_boltzmann * self.accuracyboost,
          "nonlinear": (True,False),
          "vars_pairs": ([("delta_tot", "delta_tot")])
        },
        "comoving_radial_distance": {
          "z": self.z_interp_1D
        }, # in Mpc
        "Cl": { # DONT REMOVE THIS - SOME WEIRD BEHAVIOR IN CAMB WITHOUT WANTS_CL
          'tt': 0
        }
      }
      # JVR NOTE: our likelihood must communicate with the baryons theory 
      #           which (k,z) values to compute the baryon suppression factor
      # NOTE: log10k_interp_2D is in 1/Mpc, the baryons theory must 
      #       do the conversion if necessary
      if self.external_baryon_suppression:
          _requirements_["baryon_suppression"] = {
              "z": self.z_interp_2D,
              "k": np.power(
                  10.0, self.log10k_interp_2D
              ),
          }
      # Also need Python FAST-PT if IA_code == 1
      if (self.IA_code == 1):
        _requirements_["IA_PS"] = None
        _requirements_["bias_PS"] = None
      if self.non_linear_emul == 1:
        _requirements_["omegab"] = None
        _requirements_["mnu"] = None
        _requirements_["w"] = None
        _requirements_["wa"] = None
      # Omega_nu h^2 of the massive neutrinos (CAMB's omnuh2) and, for
      # the cold dark matter + baryon halo field, the linear P_cb
      # (get_neutrino_inputs)
      _requirements_["omnuh2"] = None
      if self.halo_matter_field == 1:
        _requirements_["Pk_interpolator"]["vars_pairs"] = [
          ("delta_tot", "delta_tot"),
          ("delta_nonu", "delta_nonu")]
      return _requirements_

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  @with_omp_threads
  def set_cosmo_related(self):
    h = self.provider.get_param("H0")/100.0
    if not (self.use_emulator == 1):
      PKL  = self.provider.get_Pk_interpolator(("delta_tot", "delta_tot"), 
                                               nonlinear=False, 
                                               extrap_kmin=1e-6,
                                               extrap_kmax=2.5e2*self.accuracyboost)
      lnPL = PKL.logP(self.z_interp_2D,
                      np.power(10.0,self.log10k_interp_2D)).flatten(order='F')+np.log(h**3)

      if self.non_linear_emul == 1:
        params = {
          'Omm'  : self.provider.get_param("omegam"),
          'As'   : self.provider.get_param("As"),
          'Omb'  : self.provider.get_param("omegab"),
          'ns'   : self.provider.get_param("ns"),
          'h'    : h,
          'mnu'  : self.provider.get_param("mnu"), 
          'w'    : self.provider.get_param("w"),
          'wa'   : self.provider.get_param("wa"),
        }
        # Euclid Emulator only works on z<10.0
        kbt, tmp_bt = ee2.get_boost2(params, 
                                     self.z_interp_2D[self.z_interp_2D < 10.0], 
                                     self.emulator, 
                                     10**np.linspace(-2.0589,0.973,self.len_log10k_interp_2D))
        bt = np.array(tmp_bt, dtype='float64')
        tmp = interp1d(np.log10(kbt), 
                        np.log(bt), 
                        axis=1,
                        kind='linear', 
                        fill_value='extrapolate', 
                        assume_sorted=True)(self.log10k_interp_2D-np.log10(h)) #h/Mpc
        tmp[:,10**(self.log10k_interp_2D-np.log10(h)) < 8.73e-3] = 0.0
        lnbt = np.zeros((self.len_z_interp_2D, self.len_log10k_interp_2D))
        lnbt[self.z_interp_2D < 10.0, :] = tmp
        # Use Halofit first that works on all redshifts
        lnPNL = self.provider.get_Pk_interpolator(("delta_tot", "delta_tot"),
          nonlinear=True, 
          extrap_kmin=1e-6,
          extrap_kmax =2.5e2*self.accuracyboost).logP(self.z_interp_2D,
          np.power(10.0,self.log10k_interp_2D)).flatten(order='F')+np.log(h**3) 
        # on z < 10.0, replace it with EE2
        lnPNL = np.where((self.z_interp_2D<10)[:,None], 
          lnPL.reshape(self.len_z_interp_2D,self.len_log10k_interp_2D,order='F')+lnbt, 
          lnPNL.reshape(self.len_z_interp_2D,self.len_log10k_interp_2D,order='F')).ravel(order='F')
      elif self.non_linear_emul == 2:
        lnPNL = self.provider.get_Pk_interpolator(("delta_tot", "delta_tot"),
          nonlinear=True, 
          extrap_kmin=1e-6,
          extrap_kmax=2.5e2*self.accuracyboost).logP(self.z_interp_2D,
          np.power(10.0,self.log10k_interp_2D)).flatten(order='F')+np.log(h**3)   
      else:
        raise LoggedError(self.log, "non_linear_emul = %d is an invalid option", non_linear_emul)

      # G on the dense 1D z grid (clipped to the P(k) interpolator range):
      # cosmolike reads G linearly in z, and on the coarse 2D grid
      # (dz ~ 0.03) the linear read misses D by up to 7e-5, which the
      # cluster abundance of rare massive halos amplifies 5-15x.
      z_growth = self.z_interp_1D[self.z_interp_1D <= self.z_interp_2D[-1]]
      G_growth = np.sqrt(PKL.P(z_growth,0.0005)/PKL.P(0,0.0005))*(1+z_growth)
      # historical normalization of the table (every project): divided by
      # G at the last z_2D node; cosmolike's growfac still normalizes to
      # G(z = 0) = 1 on its side
      z_norm = self.z_interp_2D[-1]
      G_growth /= np.sqrt(PKL.P(z_norm,0.0005)/PKL.P(0,0.0005))*(1+z_norm)
      # Apply baryon suppression factors from theory block (if enabled)
      # The baryon suppression theory block computes S(k,z) for each requested z
      # and applies calibration masking. Here we simply retrieve and apply those factors.
      if self.external_baryon_suppression:
        try:
          supp_dict = self.provider.get_result("baryon_suppression")
          self.log.info(
            "Applying baryon suppression: %d redshifts from theory block",
            len(supp_dict),
          )

          for i, z_val in enumerate(self.z_interp_2D):
            if z_val in supp_dict:
              sup_array = supp_dict[z_val]
              lnbt_baryon = np.log(sup_array)
              lnPNL[i :: self.len_z_interp_2D] += lnbt_baryon
              self.log.debug(
                  "Applied baryon suppression at z=%.3f: "
                  "min_sup=%.6f, max_sup=%.6f",
                  z_val,
                  sup_array.min(),
                  sup_array.max(),
              )
            else:
              self.log.warning(
                  "baryon_suppression dict does not contain z=%.3f; skipping",
                  z_val,
              )
        except Exception as e:
            self.log.error(
                "Failed to retrieve baryon suppression from theory block: %s; "
                "skipping baryon suppression",
                str(e),
            )

      # the massive neutrinos: Omega_nu h^2 and, for the cold dark matter
      # + baryon halo field, the linear P_cb (get_neutrino_inputs)
      (omegan2, lnPL_cb) = self.get_neutrino_inputs(lnPL=lnPL, h=h)

      ci.set_cosmology(
        omegam=self.provider.get_param("omegam"),
        omegab=self.provider.get_param("omegab"),
        omegan2=omegan2,
        H0=self.provider.get_param("H0"),
        log10k_2D=self.log10k_interp_2D-np.log10(h), #h/Mpc
        z_2D=self.z_interp_2D,
        lnP_linear=lnPL, 
        lnP_linear_cb=lnPL_cb,
        lnP_nonlinear=lnPNL, 
        G=G_growth,
        z_G=z_growth,
        z_1D=self.z_interp_1D,
        chi=self.provider.get_comoving_radial_distance(self.z_interp_1D)*h # convert to Mpc/h
      )
      
      # IA power spectra from FAST-PT 
      # Must be called after ci.set_cosmology b/c it resets random state cosmology.random
      if int(self.IA_code) == 1:
        FPTIA, FPTIA_kcut  = self.provider.get_IA_PS()
        FPTbias, sigma4    = self.provider.get_bias_PS()
        FPT_kmin, FPT_kmax = FPTIA[-2,0], FPTIA[-2,-1]
        
        ci.set_IA_PS(PS=FPTIA.flatten(order='C'), 
                     kmin=FPT_kmin, 
                     kmax=FPT_kmax, 
                     cutoff=FPTIA_kcut, 
                     N=len(FPTIA[0]))
        
        ci.set_bias_PS(PS=FPTbias.flatten(order='C'), 
                       kmin=FPT_kmin, 
                       kmax=FPT_kmax, 
                       cutoff=FPTIA_kcut, 
                       sigma4=sigma4, 
                       N=len(FPTIA[0]))
    else:
      ci.set_distances(
        z=self.z_interp_1D,
        chi=self.provider.get_comoving_radial_distance(self.z_interp_1D)*h
      )

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  def get_neutrino_inputs(self, lnPL, h):
    """Return the massive-neutrino inputs of ci.set_cosmology.

    omegan2 is Omega_nu h^2 of the massive neutrinos today, part of
    omegam. It always reaches cosmolike. The halo model reads it only
    when it counts halos of cold dark matter + baryons
    (halo_matter_field = 1): the neutrinos free-stream out of halos, so
    rho_crit (Omega_m - Omega_nu) replaces the total matter density in
    the Lagrangian radius of sigma(M) and in the rho/M of dn/dM.

    lnPL_cb is ln P_cb, the linear power spectrum of cold dark matter +
    baryons, which sigma^2(M) integrates under halo_matter_field = 1.
    It is an empty list under halo_matter_field = 0: nothing reads it,
    and cosmolike then drops any table of a previous call.

    The two theory paths:
      CAMB (use_emulator = 0): omegan2 is CAMB's omnuh2 and P_cb its
        ("delta_nonu", "delta_nonu") linear spectrum, read like P_lin
        (get_requirements asks for both).
      emulators (use_emulator = 2): the emulators take no neutrino
        parameter (they were trained at mnu = 0.06 eV) and have no cb
        spectrum. omegan2 = mnu (3.046/3)^0.75/94.0708, the neutrino
        density the yaml's omegach2 subtracts, and
        P_cb = P_lin/(1 - f_nu)^2 with f_nu = omegan2/(omegam h^2): the
        ratio of the two spectra far above the neutrino free-streaming
        scale, an approximation on cluster scales. Its measured size is
        in projects/des_cluster/README.md.

    Arguments:
      lnPL = ln P_lin [(Mpc/h)^3], flattened as set_cosmology's
             lnP_linear (Fortran order: k index slow, z index fast)
      h    = H0/100

    Returns:
      (omegan2, lnPL_cb): a float and a numpy array of lnPL's shape, or
      an empty list when halo_matter_field = 0.
    """
    if self.use_emulator == 2:
      mnu = self.provider.get_param("mnu")
      omegan2 = mnu*(3.046/3.0)**0.75/94.0708
    else:
      omegan2 = self.provider.get_param("omnuh2")

    if self.halo_matter_field == 0:
      return (omegan2, [])

    if self.use_emulator == 2:
      # P_cb/P_lin = 1/(1 - f_nu)^2 where the neutrinos no longer
      # cluster (delta_m = (1 - f_nu) delta_cb)
      f_nu = omegan2/(self.provider.get_param("omegam")*h*h)
      lnPL_cb = lnPL - 2.0*np.log(1.0 - f_nu)
    else:
      # the same k extrapolation, (z, k) grid, flattening and units as
      # lnPL in set_cosmo_related
      PKL_cb = self.provider.get_Pk_interpolator(("delta_nonu", "delta_nonu"),
                                                 nonlinear=False,
                                                 extrap_kmin=1e-6,
                                                 extrap_kmax=2.5e2*self.accuracyboost)
      k_grid = np.power(10.0, self.log10k_interp_2D)
      lnPL_cb = PKL_cb.logP(self.z_interp_2D, k_grid).flatten(order='F')
      lnPL_cb = lnPL_cb + np.log(h**3)
    return (omegan2, lnPL_cb)

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  @with_omp_threads
  def set_source_related(self, **params):
    ntomo = self.source_ntomo
    ci.set_nuisance_shear_calib(
      M=[params.get(p,0) for p in [survey+"_M"+str(i+1) for i in range(ntomo)]]
    )
    if not (self.use_emulator == 1):
      if self.external_nz_modeling: 
        # here we send n(z) at every point in the chain as the user may
        # modify it using an external function (example: adding outliers)
       
        # to modify it
        # (1) deep copy the numpy array (so we keep track of the fiducial
        # (2) modify the copy
        # (3) call set_source_sample
        source_nz_local = self.source_nz.copy()

        # insert mod function here <-
        #source_nz_local = f(source_nz_local, nuisance parameters)

        ci.set_source_sample(source_nz_local)

        # user may choose to still add photo-z bias or not (here we ad)
        ci.set_nuisance_shear_photoz(
          bias=[params.get(p,0) for p in [survey+"_DZ_S"+str(i+1) for i in range(ntomo)]]
        )
      else:
        ci.set_nuisance_shear_photoz(
          bias=[params.get(p,0) for p in [survey+"_DZ_S"+str(i+1) for i in range(ntomo)]]
        )
      ci.set_nuisance_ia(
        A1=[params.get(p,0) for p in [survey+"_A1_"+str(i+1) for i in range(ntomo)]],
        A2=[params.get(p,0) for p in [survey+"_A2_"+str(i+1) for i in range(ntomo)]],
        B_TA=[params.get(p,0) for p in [survey+"_BTA_"+str(i+1) for i in range(ntomo)]]
      )

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  @with_omp_threads
  def set_lens_related(self, **params):
    ntomo = self.lens_ntomo
    ci.set_point_mass(
      PMV = [params.get(p, 0) for p in [survey+"_PM"+str(i+1) for i in range(ntomo)]]
    )
    if not (self.use_emulator == 1):
      ci.set_nuisance_bias(
        B1=[params.get(p,1) for p in [survey+"_B1_"+str(i+1) for i in range(ntomo)]],
        B2=[params.get(p,0) for p in [survey+"_B2_"+str(i+1) for i in range(ntomo)]],
        B_MAG=[params.get(p,0) for p in [survey+"_BMAG_"+str(i+1) for i in range(ntomo)]],
        B3nl=[params.get(p,0) for p in [survey+"_B3NL_"+str(i+1) for i in range(ntomo)]],
        BK=[params.get(p,0) for p in [survey+"_BK_"+str(i+1) for i in range(ntomo)]]
      )
      if self.external_nz_modeling: 
        # here we send n(z) at every point in the chain as the user may
        # modify it using an external function (example: adding outliers)
       
        # to modify it
        # (1) deep copy the numpy array (so we keep track of the fiducial
        # (2) modify the copy
        # (3) call set_source_sample
        lens_nz_local = self.lens_nz.copy()

        # insert mod function here <-
        #lens_nz_local = f(lens_nz_local, nuisance parameters)

        ci.set_lens_sample(lens_nz_local)

        # user may choose to still add photo-z bias or not (here we ad)
        ci.set_nuisance_clustering_photoz(
          bias=[params.get(p,0) for p in [survey+"_DZ_L"+str(i+1) for i in range(ntomo)]],
          stretch=[params.get(p,0) for p in [survey+"_DZ2_L"+str(i+1) for i in range(ntomo)]]
        )
      else:
        ci.set_nuisance_clustering_photoz(
          bias=[params.get(p,0) for p in [survey+"_DZ_L"+str(i+1) for i in range(ntomo)]],
          stretch=[params.get(p,0) for p in [survey+"_DZ2_L"+str(i+1) for i in range(ntomo)]]
        )

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  @with_omp_threads
  def set_cluster_related(self, **params):
    """Cluster nuisance parameters: MOR (eqs 18-19) and selection bias."""
    missing = [p for p in CLUSTER_MOR_PARAMS if p not in params]
    if missing:
      raise LoggedError(self.log, "missing cluster MOR parameters %s", missing)
    ci.set_nuisance_cluster_mor(
      MOR=[params[p] for p in CLUSTER_MOR_PARAMS]
    )
    defaults = CLUSTER_SELECTION_DEFAULTS[self.cluster_selection_model]
    ci.set_nuisance_cluster_selection(
      SEL=[params.get(p, d) for p, d in zip(CLUSTER_SELECTION_PARAMS, defaults)]
    )

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def compute_logp(self, datavector):
    if self.has_clusters:
      return -0.5 * ci.compute_chi2_cluster(datavector)
    return -0.5 * ci.compute_chi2(datavector)

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def logp(self, **params):
    datavector = self.internal_get_datavector(**params)
    return self.compute_logp(datavector)

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  @with_omp_threads
  def get_datavector(self, **params):        
    if self.use_emulator == 1:
      #dv = self.internal_get_datavector_emulator(**params)
      dv = 0.0
    else:
      dv = self.internal_get_datavector(**params)
    return np.array(dv,dtype='float64')

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def internal_get_datavector(self, **params):
    self.set_cosmo_related()
    if self.probe != "xi":
        self.set_lens_related(**params)
    self.set_source_related(**params)
    
    if self.has_clusters:
      self.set_cluster_related(**params)
      datavector = ci.compute_data_vector_cluster_masked()
    elif self.create_baryon_pca:
      pcs = ci.compute_baryon_pcas(scenarios=self.baryon_pca_select_sims, allsims=self.allsims)
      np.savetxt(self.filename_baryon_pca, pcs)
      datavector = ci.compute_data_vector_masked()
    elif self.use_baryon_pca: 
      Q = [params.get(p,0) for p in [survey+"_BARYON_Q"+str(i+1) for i in range(self.npcs)]]     
      datavector = ci.compute_data_vector_masked_with_baryon_pcs(Q=Q)
    else:  
      datavector = ci.compute_data_vector_masked()

    if self.print_datavector:
      size = len(datavector)
      out = np.zeros(shape=(size, 2))
      out[:,0] = np.arange(0, size)
      out[:,1] = datavector
      fmt = '%d', '%1.8e'
      np.savetxt(self.print_datavector_file, out, fmt = fmt)
    return datavector
