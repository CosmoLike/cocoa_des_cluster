"""Shared Cobaya likelihood class of the des_cluster project.

Cobaya is the sampler framework that runs these likelihoods: it reads a
YAML file, builds a model from a theory code (CAMB, or trained emulators)
and the likelihoods, and asks each likelihood for ln L at every sampled
point. Each likelihood file of this folder (cosmic_shear.py,
combo_2x2pt.py, combo_3x2pt.py, combo_4x2pt_N.py, combo_6x2pt_N.py)
subclasses _cosmolike_prototype_base and only passes its probe name to
initialize; the defaults of its options live in the yaml file of the same
name (e.g. combo_4x2pt_N.yaml).

The computation happens in CosmoLike, the compiled C/C++ library imported
as ci (cosmolike_des_cluster_interface, built from interface/interface.cpp).
The library keeps one global state per Python process, and this class
fills it in two stages:

  1. initialize, once: read the .dataset file (a small text file of
     `key = value` lines naming the data vector, covariance, mask and n(z)
     files and giving the binning), choose the redshift and wavenumber
     grids, and pass the probes, binning, accuracy settings, n(z) and, for
     the cluster likelihoods, the cluster model to the library;
  2. logp, at every point: pass the cosmology (distances, growth, linear
     and nonlinear P(k, z)) and the nuisance parameters to the library,
     compute the masked theory data vector and return -chi2/2.

The joint cluster data vector stores the blocks ss, gs, gg, cg, N, cc, cs
(cosmic shear, galaxy-galaxy lensing, galaxy clustering, cluster x galaxy
clustering, cluster counts, cluster clustering, cluster lensing). The
project README describes the model (arXiv:2503.13631) and every option.

Units at the boundary: Cobaya's theory codes work in Mpc and 1/Mpc, while
CosmoLike expects comoving distances in Mpc/h, wavenumbers in h/Mpc and
P(k) in (Mpc/h)^3; set_cosmo_related converts.
"""
# These __future__ imports change nothing under Python 3. Python requires
# them before every other statement of the file (only the docstring and
# comments may precede them).
from __future__ import absolute_import, division, print_function
import os
import numpy as np
import scipy
from scipy.interpolate import interp1d
import sys
import time
import functools
from collections.abc import Mapping

# DataSetLikelihood: the Cobaya base class of likelihoods whose data are
# described by a .dataset file. LoggedError: the Cobaya exception that also
# writes its message to the run log. IniFile: getdist's reader of
# `key = value` text files such as the .dataset file.
from cobaya.likelihoods.base_classes import DataSetLikelihood
from cobaya.log import LoggedError
from getdist import IniFile

# EuclidEmulator2, the emulator of the nonlinear boost used when
# non_linear_emul = 1.
import euclidemu2 as ee2
import math

from contextlib import contextmanager
@contextmanager
def timer(label):
  """Print the wall-clock time spent inside a with-block.

  @contextmanager turns this generator function into an object for a
  `with` statement: the code before `yield` runs when the block starts,
  the print after it when the block ends. `with timer("dv"): ...` prints
  "dv: 0.1234s". No code of this project calls it.

  Arguments:
    label = text printed before the elapsed time, a string

  Returns:
    a context manager; the elapsed time is printed in seconds.
  """
  t0 = time.perf_counter()
  yield
  print(f"{label}: {time.perf_counter() - t0:.4f}s")

# ci = the compiled CosmoLike library of this project.
import cosmolike_des_cluster_interface as ci

# OpenMP threads of CosmoLike's parallel loops, read once at import from the
# environment variable OMP_NUM_THREADS (1 when it is unset).
COSMOLIKE_OMP_THREADS = int(os.environ.get("OMP_NUM_THREADS", 1))

def with_omp_threads(fn):
    """Wrap a method so CosmoLike's OpenMP thread count is reset before it runs.

    CosmoLike's hot loops are parallelized with OpenMP (threads sharing the
    process memory) and take their thread count from omp_get_max_threads(),
    which starts at OMP_NUM_THREADS. Some Python libraries call
    omp_set_num_threads(1) when they run. That setting is process-wide, so
    every later CosmoLike loop would run on one core. The wrapper calls
    ci.set_omp_threads(COSMOLIKE_OMP_THREADS) at the start of every call.

    It is used as a decorator: the line `@with_omp_threads` above a method
    definition replaces the method by wrapper. functools.wraps copies the
    method's name and docstring onto wrapper, and *args, **kwargs forward
    every positional and keyword argument unchanged.

    Arguments:
      fn = the method to wrap

    Returns:
      wrapper, a function taking fn's arguments and returning fn's result.

    Side effects:
      sets the OpenMP thread count of the process at every call.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        """Reset the OpenMP thread count, then call fn with the same arguments.

        functools.wraps replaces this docstring by fn's at decoration time.

        Arguments:
          args, kwargs = every positional and keyword argument of the call,
                         passed to fn unchanged.

        Returns:
          fn's return value.
        """
        ci.set_omp_threads(COSMOLIKE_OMP_THREADS)
        return fn(*args, **kwargs)
    return wrapper

# Prefix of every nuisance-parameter name of this project (DES_DZ_S1,
# DES_CL_LNLAMBDA0, ...) and the survey name passed to init_survey_parameters.
survey = "DES"

# ----------------------------------------------------------------------------
# Clusters (4x2pt + N, arXiv 2503.13631)
# ----------------------------------------------------------------------------
# Probe names (compared in lower case) handled by ci.init_probes_cluster,
# which lays out the joint vector ss, gs, gg, cg, N, cc, cs; every other
# probe goes through ci.init_probes.
CLUSTER_PROBES = ("4x2pt_n", "6x2pt_n", "n", "n_cc", "n_cs", "cs", "cc", "cg")

# Mass-observable relation (MOR), eqs (18)-(19). The richness lambda (the
# number of red member galaxies, the observed mass proxy) is lognormal at
# fixed halo mass M and redshift z, with mean
#   <ln lambda> = ln lambda_0 + A_lambda ln(M/M_piv) + B_lambda ln((1+z)/1.45)
# (M_piv = 5e14 M_sun/h) and variance sigma_int^2 plus a Poisson term. The
# names follow the slot order of cluster.mor: ln lambda_0, A_lambda,
# sigma_int, B_lambda. No value switches the relation off, so
# set_cluster_related requires all four parameters.
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
  """Read one list-valued .dataset entry, e.g. `richness_edges = 20, 30, 45`.

  Commas become spaces, the text is split at whitespace, and the list
  comprehension converts each piece with tp.

  Arguments:
    ini = the getdist IniFile holding the .dataset entries
    key = the entry name, a string
    tp  = the conversion applied to each piece (float or int)

  Returns:
    a Python list of tp values, in file order.

  Raises:
    ValueError when a piece is not a valid tp literal; getdist raises an
    error naming the key when the entry is missing.
  """
  return [tp(x) for x in ini.string(key).replace(",", " ").split()]

class _cosmolike_prototype_base(DataSetLikelihood):
  """Base class of every des_cluster likelihood: one probe, one .dataset file.

  Cobaya creates one instance per likelihood named in the YAML and sets
  every option of the likelihood's defaults yaml (path, data_file,
  accuracyboost, use_emulator, IA_model, the cluster_* keys, ...) as an
  attribute of the instance. The methods run in this order:

    get_modified_defaults  class level, while Cobaya assembles the defaults
    initialize(probe)      once; the subclass passes its probe name
    get_requirements       once; what the theory code must compute
    logp(**params)         at every sampled point; returns -chi2/2

  initialize sets, among others, probe, has_clusters (True for the joint
  cluster vector), the file names read from the .dataset file, and the
  grids z_interp_1D, z_interp_2D, z_interp_2D_camb and log10k_interp_2D.
  """

  @classmethod
  def get_modified_defaults(cls, defaults, input_options={}):
    """Apply the yaml option `fixed_params` to the default parameters.

    `params: !defaults [...]` cannot be extended in the same yaml, so a combo
    lists the parameters it fixes under `fixed_params` (e.g. CL+GC fixes the
    lens bins it does not use). Each entry replaces the parameter's default
    info with the cobaya merge rule: a value drops prior, ref and proposal.
    A user yaml can override `fixed_params` like any other option.

    Cobaya calls this class method (it receives the class, cls, not an
    instance) while it assembles the defaults, before any instance exists.
    An entry of fixed_params is either a mapping such as {value: 1.62},
    merged into the parameter's entry, or a bare number, which becomes the
    value.

    Arguments:
      defaults      = the defaults dictionary of the likelihood (its yaml
                      file, with the !defaults parameter files included)
      input_options = the options the user's YAML gives this likelihood;
                      its fixed_params, when present, replaces the default

    Returns:
      defaults, modified in place: each fixed parameter keeps its other keys
      (e.g. latex) and gains its value.
    """
    fixed = input_options.get("fixed_params", defaults.get("fixed_params"))
    params = defaults.get("params") or {}
    for p, info in (fixed or {}).items():
      old = params.get(p)
      new = {}
      if isinstance(old, Mapping):
        # copy every key of the default entry except prior, ref and proposal
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
    """Read the .dataset file and initialize CosmoLike for one probe.

    Runs once, when Cobaya builds the model. The .dataset file (the options
    path and data_file) names the data vector, covariance, mask and n(z)
    files and gives the angular binning; relativeFileName resolves each file
    name relative to the folder of the .dataset file. The method then fixes
    the redshift and wavenumber grids of the tables passed to CosmoLike at
    every evaluation, and passes the probes, angular binning, accuracy
    settings, n(z), intrinsic-alignment and bias models and, for cluster
    probes, the cluster model (init_cluster_related) to the library. The
    order of the ci.init_* calls matters: the probes and bin counts set the
    sizes of the arrays that the data, mask and covariance calls fill.

    Arguments:
      probe = the probe name passed by the subclass: "xi" (cosmic shear),
              "2x2pt", "3x2pt", "4x2pt_N" or "6x2pt_N"

    Raises:
      LoggedError when pk_z_refinement is not a positive integer, when a
      cluster probe asks for use_emulator 1 or for a baryon PCA, and (from
      init_cluster_related) when the cluster entries of the .dataset file
      are inconsistent.

    Side effects:
      sets the global state of the compiled library (one per process) and
      the attributes listed in the class docstring.
    """
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
    # z_interp_1D: the redshift nodes of the 1D tables (comoving distance and
    # growth). Three blocks: [0, 3) holds the survey kernels and gets most
    # nodes; [3, 50.1) covers the rest of the power-spectrum range (the 2D
    # grid below ends at z = 49.99); [1070, 1100] brackets recombination
    # (z near 1090), so the distance table reaches the last-scattering
    # surface. tmp = 1000 + 250*accuracyboost is the node budget: the blocks
    # get 80%, 40% and 10% of it, and never fewer than 100, 100 and 50 nodes.
    tmp=int(1000 + 250*self.accuracyboost)
    self.z_interp_1D = np.concatenate((np.linspace(0.0,3.0,max(100,int(0.80*tmp)),endpoint=False),
                                       np.linspace(3.0,50.1,max(100,int(0.40*tmp)),endpoint=False),
                                       np.linspace(1070,1100,max(50,int(0.10*tmp)))),axis=0)
    self.len_z_interp_1D = len(self.z_interp_1D)

    # z_interp_2D: the z nodes of the 2D power-spectrum tables passed to
    # cosmolike, which interpolates LINEARLY in z between exactly these
    # nodes (its piecewise-uniform direct indexing uses the grid it is
    # given; there is no internal regridding). Linear interpolation leaves
    # a sawtooth-shaped O(dz^2) residual that vanishes at the nodes, so two
    # grids that do not share nodes disagree by the FULL residual amplitude.
    # A node count that changes freely with the boost, such as
    # min(120 + 20*boost, 250), moves the nodes and re-phases that sawtooth
    # at every boost value (order-unity chi2 jitter in the clustering vector
    # of the roman_kl project, smaller shifts that do not converge here).
    # The dyadic factor m = 2^ceil(log2(boost)), at most 16, refines each
    # uniform block by an integer factor with the same endpoints, so
    # (a) every block stays uniform (cosmolike keeps its two-segment direct
    # indexing, no search), (b) every coarser grid's nodes are a subset of
    # every finer grid's nodes, making a boost increase a true refinement
    # (error falls like 1/m^2, no re-phasing), and (c) boost 1 gives the
    # 140-node grid (105 + 35 nodes) that z_interp_2D_camb below also uses.
    # The low block multiplies its node count (endpoint=False, spacing
    # 3/n); the high block multiplies its INTERVAL count
    # (endpoint=True: 35 nodes = 34 intervals -> 34*m + 1 nodes).
    # The grid ends at z = 49.99, below z = 50, the largest redshift of the
    # hybrid emulators.
    #
    # pk_z_refinement multiplies m on top of the factor the accuracy boost
    # sets. A Fourier-space data vector reads P(k, z) at fixed multipoles,
    # where the linear z interpolation residual does not average out as it
    # does in real-space bin averages: a Fourier-space 3x2pt chi2
    # (roman_fourier project) moves by 0.25, 0.030, 0.002 from m = 1 to
    # 2, 4, 8, real-space ones (roman_real, lsst_y1) by <= 0.004 from 1 to 2.
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

    # log10 of the wavenumbers of the 2D tables, k in 1/Mpc (set_cosmo_related
    # converts to h/Mpc): 1250 + 250*accuracyboost nodes from 10^-4.99 to
    # 10^2 /Mpc.
    self.log10k_interp_2D = np.linspace(-4.99,2.0,int(1250+250*self.accuracyboost))
    self.len_log10k_interp_2D = len(self.log10k_interp_2D)
    # ------------------------------------------------------------------------

    ci.initial_setup()
    ci.reset_cluster() # cluster defaults + fresh cluster cache keys
    if self.has_clusters:
      # init_probes_cluster records the galaxy blocks (ss, gs, gg) in the C
      # struct like and the cluster blocks (cg, N, cc, cs) in the C struct
      # cluster
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
    # Halo statistics use cold dark matter + baryons. The emulator path
    # has no separate cb spectrum and uses the documented small-scale ratio.
    if self.use_emulator == 2:
      self.log.info("Halo P_cb uses P_lin/(1 - f_nu)^2 because the "
                    "emulators have no cb spectrum (an approximation; "
                    "see get_neutrino_inputs)")

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
        # NLA (IA_model 0) runs with the C FAST-PT path (IA_code 0) even
        # when the yaml asks for the Python FAST-PT theory block (IA_code 1)
        self.IA_code = 0
      ci.init_IA(ia_model = int(self.IA_model), 
                ia_redshift_evolution = int(self.IA_redshift_evolution),
                ia_code = int(self.IA_code))

      if self.probe != "xi":
        # bias_model = the redshift-evolution model of each galaxy-bias term,
        # in the order (b1, b2, bs2, b3, bmag); 0 = one amplitude per lens bin
        ci.init_bias(bias_model=self.bias_model)

      if self.non_linear_emul == 1:
        self.emulator = ee2.PyEuclidEmulator()

      # external_baryon_suppression (a theory block supplies the baryonic
      # suppression of the nonlinear spectrum) excludes the baryon PCA and
      # the baryon contamination of the data vector: both are switched off.
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
      # the data vector gains four baryon principal components, with the
      # amplitudes DES_BARYON_Q1 ... DES_BARYON_Q4 (internal_get_datavector)
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

    <phi_i|z> is the probability that a cluster at true redshift z is
    observed in cluster redshift bin i: a selection kernel, not a
    normalized n(z). The richness edges define the observed richness bins
    (the richness lambda, the number of red member galaxies, is the mass
    proxy).

    Arguments:
      ini = the getdist IniFile of the .dataset file

    Raises:
      LoggedError when the numbers of redshift edges, cg lens bins or n(z)
      columns do not match cluster_ntomo, when fewer than two richness
      edges are given, or when cluster_selection_model is not 0, 1 or 2.

    Side effects:
      sets the survey area, cluster model, richness bins, selection kernels
      and (cluster bin, lens bin) pairs of the compiled library, and the
      attributes nz_cluster_file, cluster_ntomo, cluster_zbin_edges,
      richness_edges, survey_area_deg2, cg_lens_bins and
      cluster_selection_model.
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
    # z (DES), 1 = the alpha(z) of halo.c, fixed by int b f dnu = 1 (matter
    # is unbiased with respect to itself)
    ci.init_cluster_hmf_alpha_mode(
        hmf_alpha_mode=int(getattr(self, "cluster_hmf_alpha_mode", 0)))

    ci.init_cluster_adopt_limber(
        adopt_limber_cc=int(getattr(self, "cluster_adopt_limber_cc", 1)),
        adopt_limber_cg=int(getattr(self, "cluster_adopt_limber_cg", 1)))

    # [:-1] = every edge but the last (lower edges), [1:] = every edge but the
    # first (upper edges); .copy() passes owned, contiguous arrays instead of
    # views into richness_edges
    ci.init_cluster_richness_bins(lambda_min=self.richness_edges[:-1].copy(),
                                  lambda_max=self.richness_edges[1:].copy())

    # columns of the cluster n(z) file: true z, then <phi_i|z> of each bin
    nz_cluster = np.loadtxt(self.nz_cluster_file)
    if nz_cluster.ndim != 2 or nz_cluster.shape[1] != self.cluster_ntomo + 1:
      raise LoggedError(self.log, "%s: expected %d columns (z + %d bins)",
                        self.nz_cluster_file, self.cluster_ntomo + 1,
                        self.cluster_ntomo)
    ci.set_cluster_zdist(nofz=nz_cluster,
                         zbin_min=self.cluster_zbin_edges[:-1].copy(),
                         zbin_max=self.cluster_zbin_edges[1:].copy())

    # cg_lens_bins[i] = the lens bin correlated with cluster bin i in w_cg
    ci.init_cluster_pairs(cg_lens_bin=self.cg_lens_bins)

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def get_requirements(self):
    """Tell Cobaya which quantities the theory code must compute.

    Cobaya calls this once, after initialize, and makes the theory code
    (CAMB, or the emulators when use_emulator = 2) provide every listed
    quantity at each sampled point; set_cosmo_related reads them through
    self.provider. The returned dictionary maps a quantity to None (a
    parameter such as H0, or a quantity without options) or to its options:

      Pk_interpolator: P(k, z) at the redshifts z_interp_2D_camb, up to
        k_max = kmax_boltzmann*accuracyboost [1/Mpc], linear and nonlinear,
        for the field pairs in vars_pairs ("delta_tot" = total matter; with
        CAMB also "delta_nonu" = cold dark matter + baryons, the field of
        the halo statistics);
      comoving_radial_distance: chi(z) [Mpc] at z_interp_1D;
      IA_PS, bias_PS: the Python FAST-PT tables (IA_code = 1);
      baryon_suppression: the suppression factor at (z_interp_2D, k) when
        external_baryon_suppression is set;
      mnu (emulators) or omnuh2 (CAMB): the neutrino density.

    The CAMB path also asks for Cl {'tt': 0}; the comment at that line
    says CAMB misbehaves without a C_l request. use_emulator = 1 (a
    data-vector emulator, not shipped with this project) asks only for what
    that emulator needs, per probe.

    Returns:
      a dictionary {quantity name: None or options}; None for a probe that
      the use_emulator = 1 branch does not list.
    """
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
      # The baryon theory block computes the suppression factor at the
      # (z, k) nodes requested here. k is in 1/Mpc (log10k_interp_2D); a
      # block that works in h/Mpc must convert.
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
      # Keep both fields available to the likelihood and direct halo readers.
      # CAMB obtains them from the same transfer-function calculation.
      _requirements_["Pk_interpolator"]["vars_pairs"] = [
        ("delta_tot", "delta_tot"),
        ("delta_nonu", "delta_nonu")]
      return _requirements_

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  @with_omp_threads
  def set_cosmo_related(self):
    """Pass the cosmology of the current point to CosmoLike.

    Reads from the theory code (self.provider) the linear and nonlinear
    matter power spectra, the growth and the comoving distances, converts
    them to CosmoLike's units and calls ci.set_cosmology. It runs at every
    evaluation (a hot path), with the OpenMP thread count reset first
    (@with_omp_threads).

    Tables passed (n_z = len(z_interp_2D), n_k = len(log10k_interp_2D)):
      lnP_linear, lnP_linear_cb, lnP_nonlinear = ln P [(Mpc/h)^3], flat
        arrays of n_z*n_k values in Fortran order: flatten(order='F') of a
        [n_z, n_k] array, so the z index runs fastest;
      log10k_2D = log10 k [h/Mpc], i.e. log10 k[1/Mpc] - log10 h;
      G, z_G = G(z) = D(z)(1+z) at the z_interp_1D nodes up to the last
        z_interp_2D node, divided by its value at that last node;
      chi, z_1D = comoving distance [Mpc/h] at z_interp_1D.
    Adding ln h^3 converts ln P from Mpc^3 to (Mpc/h)^3.

    non_linear_emul selects the nonlinear spectrum: 2 = the theory code's
    own; 1 = the linear spectrum times the EuclidEmulator2 boost below
    z = 10 and the theory code's nonlinear spectrum above. With
    use_emulator = 1 only the distances are passed.

    Raises:
      an error for a non_linear_emul value other than 1 or 2.

    Side effects:
      sets the cosmology of the compiled library, which draws a new
      cosmology.random key: every cosmology-dependent table of the library
      is rebuilt at its next use.
    """
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
        # EuclidEmulator2 covers z < 10 and k from 10^-2.0589 = 8.73e-3 to
        # 10^0.973 = 9.4 h/Mpc; get_boost2 returns, on that k grid, the boost
        # B = P_nonlinear/P_linear at each of those redshifts
        kbt, tmp_bt = ee2.get_boost2(params, 
                                     self.z_interp_2D[self.z_interp_2D < 10.0], 
                                     self.emulator, 
                                     10**np.linspace(-2.0589,0.973,self.len_log10k_interp_2D))
        bt = np.array(tmp_bt, dtype='float64')
        # ln B interpolated linearly in log10 k onto the table's k nodes in
        # h/Mpc, and extrapolated beyond the emulator's k range
        tmp = interp1d(np.log10(kbt), 
                        np.log(bt), 
                        axis=1,
                        kind='linear', 
                        fill_value='extrapolate', 
                        assume_sorted=True)(self.log10k_interp_2D-np.log10(h)) #h/Mpc
        # below the emulator's smallest k (8.73e-3 h/Mpc) the boost is set to
        # 1 (ln B = 0): those scales are linear. lnbt then holds ln B on the
        # full [n_z, n_k] table, zero in the rows z >= 10 the emulator skips.
        tmp[:,10**(self.log10k_interp_2D-np.log10(h)) < 8.73e-3] = 0.0
        lnbt = np.zeros((self.len_z_interp_2D, self.len_log10k_interp_2D))
        lnbt[self.z_interp_2D < 10.0, :] = tmp
        # start from the theory code's nonlinear spectrum, available at
        # every redshift
        lnPNL = self.provider.get_Pk_interpolator(("delta_tot", "delta_tot"),
          nonlinear=True, 
          extrap_kmin=1e-6,
          extrap_kmax =2.5e2*self.accuracyboost).logP(self.z_interp_2D,
          np.power(10.0,self.log10k_interp_2D)).flatten(order='F')+np.log(h**3) 
        # for z < 10 use ln P_linear + ln B instead: np.where picks, row by
        # row, the first array where (z < 10)[:, None] is True ([:, None]
        # turns the z mask into a column that broadcasts over k)
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
        raise LoggedError(self.log, "non_linear_emul = %d is an invalid option", self.non_linear_emul)

      # G on the dense 1D z grid (clipped to the P(k) interpolator range):
      # cosmolike reads G linearly in z, and on the coarse 2D grid
      # (dz ~ 0.03) the linear read misses D by up to 7e-5, which the
      # cluster abundance of rare massive halos amplifies 5-15x.
      z_growth = self.z_interp_1D[self.z_interp_1D <= self.z_interp_2D[-1]]
      # G is sampled at growth_k (default 0.05/Mpc), a sub-horizon scale.
      # At k = 5e-4/Mpc (about 2 H0/c) CAMB's dark-energy perturbations
      # change the growth by 0.5-0.9% at w != -1 (z = 0.5 to 2), while every
      # reader of G (IA amplitudes, one-loop D^4, sigma(M, z), the growth
      # rate f) describes sub-horizon modes; with 0.06 eV neutrinos the
      # growth varies by 0.03% above 0.05/Mpc
      growth_k = float(getattr(self, "growth_k", 0.05))
      G_growth = np.sqrt(PKL.P(z_growth,growth_k)/PKL.P(0,growth_k))*(1+z_growth)
      # Every Cocoa project divides this table by G at the last z_2D node, a
      # shared convention; cosmolike's growfac divides by G(z = 0) on its
      # side, so the growth factor it uses has D(z = 0) = 1 either way
      z_norm = self.z_interp_2D[-1]
      G_growth /= np.sqrt(PKL.P(z_norm,growth_k)/PKL.P(0,growth_k))*(1+z_norm)
      # With external_baryon_suppression, a theory block computes the
      # suppression S(k, z) (with its own calibration masking) at each
      # requested z, and ln S is added to ln P_nonlinear here. A redshift
      # missing from its result, or a failure to read it, is logged and
      # that suppression is skipped.
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
              # Fortran order puts (z_i, k_j) at index i + j*n_z, so the
              # slice i::n_z (start i, step n_z) is every k at redshift i
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
      
      # IA power spectra from the Python FAST-PT theory block. These calls
      # must follow ci.set_cosmology, which resets the cache key
      # cosmology.random.
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

    omegan2 is Omega_nu h^2 of massive neutrinos today, part of omegam.
    Halo variances use the cold dark matter + baryon spectrum P_cb at
    each redshift. Their mass-radius relation and mass-function density
    use rho_crit (Omega_m - Omega_nu). Total matter remains available
    for lensing and for the separate total-matter variance.

    lnPL_cb is ln P_cb on the same (k,z) grid and in the same units as
    lnPL. Both spectra are provided so direct halo readers can be used
    even after a likelihood evaluation that did not count halos.

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
      (omegan2, lnPL_cb): a float and a numpy array of lnPL's shape.
    """
    if self.use_emulator == 2:
      mnu = self.provider.get_param("mnu")
      omegan2 = mnu*(3.046/3.0)**0.75/94.0708
    else:
      omegan2 = self.provider.get_param("omnuh2")

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
    """Pass the source-galaxy nuisance parameters of the current point to CosmoLike.

    Per source bin i = 1, ..., source_ntomo: the shear calibration DES_M<i>
    and the photo-z shift DES_DZ_S<i>. The intrinsic-alignment lists
    DES_A1_<i>, DES_A2_<i>, DES_BTA_<i> are parameter slots whose meaning
    depends on IA_redshift_evolution: with the power-law evolution of the
    cluster likelihoods, DES_A1_1 is the NLA amplitude and DES_A1_2 its
    redshift power (params_source_y6.yaml). A name absent from params
    counts as 0, so a likelihood may omit parameters its model does not
    use (e.g. the TATT terms under NLA).

    With external_nz_modeling, the source n(z) read at initialize is
    passed again at every point, so a user function of the nuisance
    parameters can modify it first (the template comment below marks
    where).

    Arguments:
      params = keyword arguments {parameter name: value} of the current
               point, as Cobaya passes them to logp

    Side effects:
      sets the source nuisance state of the compiled library.
    """
    ntomo = self.source_ntomo
    # Each list below holds params.get(name, 0) for the names DES_<X>1, ...,
    # DES_<X>n: the inner comprehension builds the names, the outer one
    # reads their values (0 when absent).
    ci.set_nuisance_shear_calib(
      M=[params.get(p,0) for p in [survey+"_M"+str(i+1) for i in range(ntomo)]]
    )
    if not (self.use_emulator == 1):
      if self.external_nz_modeling: 
        # The n(z) goes to the library at every point, so a user function of
        # the nuisance parameters (for example, an outlier population) can
        # modify it first:
        #   (1) copy the n(z) read at initialize, which stays unmodified,
        #   (2) modify the copy,
        #   (3) pass the copy with set_source_sample.
        source_nz_local = self.source_nz.copy()

        # The user's modification goes here, e.g.
        #   source_nz_local = f(source_nz_local, params)

        ci.set_source_sample(source_nz_local)

        # the photo-z shifts DES_DZ_S<i> still apply on top of this n(z)
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
    """Pass the lens-galaxy nuisance parameters of the current point to CosmoLike.

    Per lens bin i = 1, ..., lens_ntomo: the point mass DES_PM<i> of
    galaxy-galaxy lensing, the bias terms DES_B1_<i> (linear), DES_B2_<i>,
    DES_BMAG_<i> (magnification), DES_B3NL_<i>, DES_BK_<i>, the photo-z
    shift DES_DZ_L<i> and the photo-z stretch DES_DZ2_L<i>. A name absent
    from params counts as 0, except the linear bias, which counts as 1.
    The value 0 is not neutral for the stretch: the stretch multiplies the
    width of the n(z) about its mean (1 = no stretch), which is why the lens
    parameter files always declare DES_DZ2_L<i>.

    With external_nz_modeling, the lens n(z) read at initialize is passed
    again at every point, so a user function of the nuisance parameters can
    modify it first.

    Arguments:
      params = keyword arguments {parameter name: value} of the current
               point, as Cobaya passes them to logp

    Side effects:
      sets the lens nuisance state of the compiled library.
    """
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
        # The n(z) goes to the library at every point, so a user function of
        # the nuisance parameters (for example, an outlier population) can
        # modify it first:
        #   (1) copy the n(z) read at initialize, which stays unmodified,
        #   (2) modify the copy,
        #   (3) pass the copy with set_lens_sample.
        lens_nz_local = self.lens_nz.copy()

        # The user's modification goes here, e.g.
        #   lens_nz_local = f(lens_nz_local, params)

        ci.set_lens_sample(lens_nz_local)

        # the photo-z shifts and stretches still apply on top of this n(z)
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
    """Pass the cluster nuisance parameters: MOR (eqs 18-19) and selection bias.

    The four MOR parameters (CLUSTER_MOR_PARAMS) are required. A missing
    selection-bias parameter takes the value of CLUSTER_SELECTION_DEFAULTS
    for the active cluster_selection_model, which switches that part of
    the selection bias off.

    Arguments:
      params = keyword arguments {parameter name: value} of the current
               point, as Cobaya passes them to logp

    Raises:
      LoggedError naming the missing MOR parameters.

    Side effects:
      sets the cluster nuisance state of the compiled library.
    """
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
    """Return ln L = -chi2/2 of a theory data vector against the data.

    chi2 = (m - d)^T C^-1 (m - d) over the entries the mask keeps, with m
    the theory vector, d the data vector and C the covariance read at
    initialize restricted to the kept entries (the compiled library holds
    its inverse). The constant normalization of the Gaussian likelihood is
    dropped. Cluster probes use the joint ss, gs, gg, cg, N, cc, cs layout.

    Arguments:
      datavector = the theory vector in the full layout, masked entries
                   zero, as internal_get_datavector returns it

    Returns:
      a float, -chi2/2.
    """
    if self.has_clusters:
      return -0.5 * ci.compute_chi2_cluster(datavector)
    return -0.5 * ci.compute_chi2(datavector)

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def logp(self, **params):
    """Return ln L at one sampled point; Cobaya calls this at every point.

    Arguments:
      params = keyword arguments {parameter name: value} holding the
               sampled and fixed parameters of this likelihood

    Returns:
      a float, -chi2/2 (see compute_logp).
    """
    datavector = self.internal_get_datavector(**params)
    return self.compute_logp(datavector)

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  @with_omp_threads
  def get_datavector(self, **params):        
    """Return the theory data vector at one point as a numpy array.

    Used outside the sampler, e.g. by scripts/make_synthetic_data.py.

    Arguments:
      params = keyword arguments {parameter name: value}, as for logp

    Returns:
      a float64 numpy array in the full layout, masked entries zero.

    Raises:
      LoggedError under use_emulator = 1: this project ships no
      data-vector emulator.
    """
    if self.use_emulator == 1:
      # No data-vector emulator exists for this project: a placeholder
      # would feed a zero data vector to the likelihood, so refuse.
      raise LoggedError(self.log,
                        "use_emulator = 1 is not implemented in this project")
    else:
      dv = self.internal_get_datavector(**params)
    return np.array(dv,dtype='float64')

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def internal_get_datavector(self, **params):
    """Compute the masked theory data vector for the current point.

    Passes the cosmology (set_cosmo_related) and the nuisance parameters
    (lens, skipped for cosmic shear; source; cluster) to the library, then
    computes the vector: the joint cluster vector for cluster probes;
    otherwise the galaxy vector, with baryon principal components
    (use_baryon_pca, amplitudes DES_BARYON_Q<i>) or after writing them to
    a file (create_baryon_pca).

    Arguments:
      params = keyword arguments {parameter name: value}, as for logp

    Returns:
      the theory vector in the full layout, masked entries zero, as the
      list of floats the compiled library returns.

    Side effects:
      with create_baryon_pca, writes the principal components to
      filename_baryon_pca; with print_datavector, writes the vector to
      print_datavector_file in two columns (index, value).
    """
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
      # a tuple with one format per column: the integer index, then the value
      # with 9 significant digits
      fmt = '%d', '%1.8e'
      np.savetxt(self.print_datavector_file, out, fmt = fmt)
    return datavector
