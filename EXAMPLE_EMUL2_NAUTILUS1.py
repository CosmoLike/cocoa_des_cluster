import warnings
import os
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings(
    "ignore",
    message=".*column is deprecated.*",
    module=r"sacc\.sacc"
)
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r".*invalid value encountered*"
)
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r".*overflow encountered*"
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*Function not smooth or differentiabl*"
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*Hartlap correction*"
)
import argparse, random
import numpy as np
from cobaya.yaml import yaml_load
from cobaya.model import get_model
from nautilus import Prior, Sampler
from getdist import loadMCSamples
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser(prog='EXAMPLE_EMUL2_NAUTILUS1')
parser.add_argument("--root",
                    dest="root",
                    help="Name of the Output File",
                    nargs='?',
                    const=1,
                    default="./projects/des_cluster/")
parser.add_argument("--outroot",
                    dest="outroot",
                    help="Name of the Output File",
                    nargs='?',
                    const=1,
                    default="EXAMPLE_EMUL2_NAUTILUS1")
parser.add_argument("--nlive",
                    dest="nlive",
                    help="Number of live points ",
                    type=int,
                    nargs='?',
                    const=1,
                    default=1000)
parser.add_argument("--maxfeval",
                    dest="maxfeval",
                    help="Minimizer: maximum number of likelihood evaluations",
                    type=int,
                    nargs='?',
                    const=1,
                    default=100000)
parser.add_argument("--neff",
                    dest="neff",
                    help="Minimum effective sample size. ",
                    type=int,
                    nargs='?',
                    const=1,
                    default=10000)
parser.add_argument("--flive",
                    dest="flive",
                    help="Maximum fraction of the evidence contained in the live set before building the initial shells terminates",
                    type=float,
                    nargs='?',
                    const=1,
                    default=0.01)
parser.add_argument("--nnetworks",
                    dest="nnetworks",
                    help="Number of Neural Networks",
                    type=int,
                    nargs='?',
                    const=1,
                    default=4)
args, unknown = parser.parse_known_args()
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
yaml_string=r"""
likelihood:
  des_cluster.combo_4x2pt_N:
    use_emulator: 2
    path: ./external_modules/data/des_cluster
    data_file: des_cluster_y6_4x2ptN.dataset
    print_datavector: False
    print_datavector_file: "./projects/des_cluster/chains/theory_4x2pt_N.modelvector"
    # WARNING: keep accuracyboost <= 3 (above it the integration tables of the
    # donor project desy1xplanck broke down; not re-measured here).
    accuracyboost: 1.0
    # integer 0, 1 or 2; 1 for the Y6 inputs (the MagLim lens n(z) tails reach
    # z = 2.99: delta chi2 vs high accuracy 1.65 at 0, 0.048 at 1)
    integration_accuracy: 1
    lmax: 75000 # lmax computed for real space correlation function
    kmax_boltzmann: 10.0
    # 1 EE2, 2 Halofit
    non_linear_emul: 2
    IA_model: 0 # NLA (0); the cluster lensing code has no TATT
    IA_code: 0  # CFASTPT (0) or FASTPT (1) (only works if not NLA)
    IA_redshift_evolution: 3
    debug: false
    # radial kernel of clusters in the 2pt functions: 0 = volume only (what
    # DES ran), 1 = abundance weighted
    cluster_kernel_mode: 0
    # selection bias: 0 = none, 1 = Y1 (mass dependent), 2 = Y6 (scale
    # dependent, on the data vector; eq 23)
    cluster_selection_model: 2
    # cluster lensing: 1 = Sigma = Y gamma_t (eq 15), 0 = gamma_t (the data
    # and covariance files must be in the same space)
    cluster_ytransform: 1
    # 1 = intrinsic alignment of the sources in the 2-halo cluster-lensing term
    cluster_include_ia: 1
    # cluster magnification coefficient C_c (eq 28: -2); 0 switches it off
    cluster_magnification: -2.0
    # Tinker 2010 amplitude: 0 = 0.368 at every z (what DES ran), 1 = alpha(z)
    cluster_hmf_alpha_mode: 0
    # w_cc and w_cg: 1 = Limber at every multipole, 0 = non-Limber
    cluster_adopt_limber_cc: 1
    cluster_adopt_limber_cg: 1

prior:
  # These priors are meant to prevent the sampler to wander far off training
  g1: "lambda As_1e9: stats.norm.logpdf(As_1e9, loc=2.35, scale=1.6)"
  g2: "lambda ns: stats.norm.logpdf(ns, loc=0.96, scale=0.05)"
  g3: "lambda H0: stats.norm.logpdf(H0, loc=70, scale=10.0)"
  g4: "lambda omegab: stats.norm.logpdf(omegab, loc=0.045, scale=0.012)"
  g5: "lambda omegam: stats.norm.logpdf(omegam, loc=0.3 , scale=0.25)"
params:
  As_1e9:
    prior:
      min: 0.5
      max: 5
    ref:
      dist: norm
      loc: 2.1
      scale: 0.25
    proposal: 0.2
    latex: 10^9 A_\mathrm{s}
    renames: A
  ns:
    prior:
      min: 0.87
      max: 1.07
    ref:
      dist: norm
      loc: 0.96605
      scale: 0.01
    proposal: 0.01
    latex: n_\mathrm{s}
  H0:
    prior:
      min: 55
      max: 91
    ref:
      dist: norm
      loc: 67.32
      scale: 5
    proposal: 3
    latex: H_0
  omegab:
    prior:
      min: 0.03
      max: 0.07
    ref:
      dist: norm
      loc: 0.0495
      scale: 0.004
    proposal: 0.004
    latex: \Omega_\mathrm{b}
  omegam:
    prior:
      min: 0.1
      max: 0.9
    ref:
      dist: norm
      loc: 0.316
      scale: 0.01
    proposal: 0.01
    latex: \Omega_\mathrm{m}
  w0pwa:
    value: -1.0
    latex: w_{0,\mathrm{DE}}+w_{a,\mathrm{DE}}
  w:
    value: -1.0
    latex: w_{0,\mathrm{DE}}
  wa:
    value: 'lambda w0pwa, w: w0pwa - w'
    derived: false
    latex: 'w_{a,\mathrm{DE}}'
  mnu:
    value: 0.06
  omegabh2:
    value: 'lambda omegab, H0: omegab*(H0/100)**2'
    latex: \Omega_\mathrm{b} h^2
  omegach2:
    value: 'lambda omegam, omegab, mnu, H0: (omegam-omegab)*(H0/100)**2-(mnu*(3.046/3)**0.75)/94.0708'
    latex: \Omega_\mathrm{c} h^2
  As:
    value: 'lambda As_1e9: 1e-9 * As_1e9'
    latex: A_\mathrm{s}
  # ----------------------------------------------------------------------------
  # The nuisance parameters (sources, MagLim lenses, clusters) and their
  # priors come from the likelihood defaults: likelihood/params_source_y6.yaml,
  # params_lens_maglim.yaml and params_cluster.yaml. CL+GC fixes the bias and
  # photo-z shift of lens bins 4-6, and the point masses of galaxy-galaxy
  # lensing, which it does not contain (fixed_params of combo_4x2pt_N.yaml).
  # ----------------------------------------------------------------------------

theory:
  emulrdrag:
    path: ./cobaya/cobaya/theories/
    provides: ['rdrag']
    extra_args:
      file: ['external_modules/data/emultrf/BAO_SN_RES/emul_lcdm_rdrag_GP.joblib'] 
      extra: ['external_modules/data/emultrf/BAO_SN_RES/extra_lcdm_rdrag.npy'] 
      ord: [['omegabh2','omegach2']]
  emulbaosn:
    path: ./cobaya/cobaya/theories/
    stop_at_error: True
    provides: ['comoving_radial_distance', 'angular_diameter_distance', 'Hubble']
    extra_args:
      device: "cuda"
      file:  [None, 'external_modules/data/emultrf/BAO_SN_RES/w0wa/emul_w0wa_H.pt']
      extra: [None, 'external_modules/data/emultrf/BAO_SN_RES/w0wa/extra_w0wa_H.npy']    
      ord: [None, ['omegam','H0','w','wa']]
      extrapar: [{'MLA': 'INT', 'ZMIN' : 0.0001, 'ZMAX' : 3, 'NZ' : 600},
                 {'MLA': 'ResMLP', 'offset' : 0.0, 'INTDIM' : 4, 'NLAYER' : 6,
                  'TMAT': 'external_modules/data/emultrf/BAO_SN_RES/w0wa/PCA_w0wa_H.npy',
                  'ZLIN': 'external_modules/data/emultrf/BAO_SN_RES/w0wa/z_lin_w0wa.npy'}]
  emulmps:
    path: ./cobaya/cobaya/theories/
    stop_at_error: True
    extra_args:
      model_file:    "external_modules/data/emultrf/emulmps/npce_emul/emulator_npce.keras"
      metadata_file: "external_modules/data/emultrf/emulmps/npce_emul/metadata.joblib"
      nl_model_file:    "external_modules/data/emultrf/emulmps/w0wa_halofit/emulator_halofit.keras"
      nl_metadata_file: "external_modules/data/emultrf/emulmps/w0wa_halofit/metadata.joblib"
      use_syren: False
      param_order: ["As_1e9", "ns", "H0", "omegab", "omegam", 'w', 'wa']
"""
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
model = get_model(yaml_load(yaml_string))
def chi2(p):
    p = [float(v) for v in p.values()] if isinstance(p, dict) else p
    if np.any(np.isinf(p)) or  np.any(np.isnan(p)):
      raise ValueError(f"At least one parameter value was infinite (CoCoa) param = {p}")
    point = dict(zip(model.parameterization.sampled_params(), p))
    res1 = model.logprior(point,make_finite=False)
    if np.isinf(res1) or  np.any(np.isnan(res1)):
      return 1.e20
    # return_derived=True: emulbaosn needs rdrag, which cobaya passes
    # from emulrdrag as a derived parameter (off, that store is None
    # and the evaluation fails); [0] is the log-likelihood
    res2 = model.loglike(point,
                         make_finite=False,
                         cached=False,
                         return_derived=True)[0]
    if np.isinf(res2) or  np.any(np.isnan(res2)):
      return 1e20
    return -2.0*(res1+res2)

def likelihood(params):
  res = chi2(params)
  if (res > 1.e19 or np.isinf(res) or  np.isnan(res)):
    return -np.inf
  else:
    return -0.5*res
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
from mpi4py.futures import MPIPoolExecutor

if __name__ == '__main__':
    print(f"nlive={args.nlive}, output={args.root}chains/{args.outroot}")
    # Build Nautilus Prior from Cobaya
    NautilusPrior = Prior()                                       # Nautilus Call 
    dim    = model.prior.d()                                      # Cobaya call
    bounds = model.prior.bounds(confidence=0.999999)              # Cobaya call
    names  = list(model.parameterization.sampled_params().keys()) # Cobaya Call
    for b, name in zip(bounds, names):
      NautilusPrior.add_parameter(name, dist=(b[0], b[1]))
    
    sampler = Sampler(NautilusPrior, 
                      likelihood,  
                      filepath=f"{args.root}chains/{args.outroot}_checkpoint.hdf5", 
                      n_dim=dim,
                      pool=MPIPoolExecutor(),
                      n_live=args.nlive,
                      n_networks=args.nnetworks,
                      resume=True)
    sampler.run(f_live=args.flive,
                n_eff=args.neff,
                n_like_max=args.maxfeval,
                verbose=True,
                discard_exploration=True)
    points, log_w, log_l = sampler.posterior()
    
    # Save output file ---------------------------------------------------------
    os.makedirs(os.path.dirname(f"{args.root}chains/"),exist_ok=True)
    np.savetxt(f"{args.root}chains/{args.outroot}.1.txt",
               np.column_stack((np.exp(log_w), log_l, points, -2*log_l)),
               fmt="%.5e",
               header=f"nlive={args.nlive}, maxfeval={args.maxfeval}, log-Z ={sampler.log_z}\n"+' '.join(names),
               comments="# ")
    
    # Save a range files -------------------------------------------------------
    rows = [(str(n),float(l),float(h)) for n,l,h in zip(names,bounds[:,0],bounds[:,1])]
    with open(f"{args.root}chains/{args.outroot}.ranges", "w") as f: 
      f.writelines(f"{n} {l:.5e} {h:.5e}\n" for n, l, h in rows)

    # Save a paramname files ---------------------------------------------------
    param_info = model.info()['params']
    latex  = [param_info[x]['latex'] for x in names]
    names.append("chi2*")
    latex.append("\\chi^2")
    np.savetxt(f"{args.root}chains/{args.outroot}.paramnames", 
               np.column_stack((names,latex)),
               fmt="%s")

    # Save a cov matrix --------------------------------------------------------
    samples = loadMCSamples(f"{args.root}chains/{args.outroot}",
                            settings={'ignore_rows': u'0.0'})
    np.savetxt(f"{args.root}chains/{args.outroot}.covmat",
               np.array(samples.cov(), dtype='float64'),
               fmt="%.5e",
               header=' '.join(names),
               comments="# ")
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------