"""Cosmolike-only timing of the des_cluster data vector, per probe set.

The script builds the Cobaya model of EXAMPLE_EVALUATE2.yaml (the
6x2pt + N likelihood with CAMB) with Cobaya's timers on. For each probe
set it evaluates one warm-up point, then NEVAL points with EVERY sampled
parameter moved (cosmology, photo-z shifts, shear calibration, IA, galaxy
bias, MOR, selection). CosmoLike keeps its tables (power spectra, kernels,
halo-model integrals) in caches keyed by the parameters they depend on; a
point that repeated any of them would be served partly from a cache and
time too short. The timer of the likelihood component (cosmolike and its
Python glue) is read apart from the timer of the theory component (CAMB),
and the printed table gives the cosmolike time in seconds per evaluation
(mean and standard deviation) and the CAMB time beside it.

Run inside the cocoa environment (start_cocoa.sh sourced, so ROOTDIR is
set), from anywhere, on a quiet machine:

  OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python projects/des_cluster/tests/validation/time_cosmolike.py

OMP_NUM_THREADS is the number of cosmolike threads; the two other
variables keep the linear-algebra libraries on one thread each. NEVAL
(an environment variable, default 10) sets the number of timed
evaluations per probe set.
"""
import os
import numpy as np
import cosmolike_des_cluster_interface as ci
from cobaya.yaml import yaml_load_file
from cobaya.model import get_model

# ROOT = the cocoa/Cocoa folder (set by start_cocoa.sh). PROBES = the probe
# sets of the timing table of the README, from the full joint vector down
# to single cluster blocks.
ROOT = os.environ["ROOTDIR"]
NEVAL = int(os.environ.get("NEVAL", "10"))
PROBES = ["6x2pt_N", "4x2pt_N", "3x2pt", "N", "cc", "cg", "cs"]

info = yaml_load_file(ROOT + "/projects/des_cluster/EXAMPLE_EVALUATE2.yaml")
info["likelihood"] = {"des_cluster.combo_6x2pt_N": {
    "path": ROOT + "/external_modules/data/des_cluster"}}
# drop the baryon PC amplitudes when the YAML's own params block lists them:
# the cluster likelihoods use no baryon PCs (pop(p, None) ignores absent
# names)
for p in ["DES_BARYON_Q1", "DES_BARYON_Q2"]:
  info["params"].pop(p, None)
info["debug"] = False
info["timing"] = True
os.chdir(ROOT)
model = get_model(info)
lik = list(model.likelihood.values())[0]
theory = list(model.theory.values())[0]

# the starting point: a reference draw of every sampled parameter (fixed
# seed), then the fiducial cosmology and cluster parameters on top; zip
# pairs each name with its value and dict() builds {name: value}
names = list(model.parameterization.sampled_params())
point = dict(zip(names, model.prior.reference(random_state=1)))
point.update({"As_1e9": 2.19, "ns": 0.96859, "H0": 69.0, "omegab": 0.048,
              "omegam": 0.3, "w": -1.0, "w0pwa": -1.0, "mnu": 0.06})
fid_cl = {"DES_CL_LNLAMBDA0": 4.26, "DES_CL_A_LAMBDA": 0.943,
          "DES_CL_SIGMA_INT": 0.15, "DES_CL_B_LAMBDA": 0.207,
          "DES_CL_BS1": 1.1, "DES_CL_BS2": 0.2, "DES_CL_R0": 30.0}
for p in names:
  if p in fid_cl:
    point[p] = fid_cl[p]


def jittered(step):
  """Return the starting point with every sampled parameter moved.

  An MCMC-like step: every sampled parameter moves by a small, bounded
  amount (<= 0.8% over the whole run), so every point stays inside the
  priors. A nonzero value is multiplied by 1 + 1e-4 step; a zero value
  becomes 1e-5 step. Fixed entries of the point keep their value.

  Arguments:
    step = the index of the evaluation (a positive integer); the default
           run reaches step 80.

  Returns:
    a dict {parameter name: value} for model.logposterior.
  """
  q = {}
  for name, value in point.items():
    if name in names:
      q[name] = value*(1 + 1e-4*step) if value != 0 else 1e-5*step
    else:
      q[name] = value
  return q


def run(probe, offset):
  """Time NEVAL evaluations of one probe set after one warm-up.

  ci.init_probes_cluster switches the blocks the compiled library
  computes; the likelihood keeps the 6x2pt + N data, mask and covariance,
  so only the timing (not the chi2) of a reduced probe set is
  meaningful. cached=False makes Cobaya recompute every component even if
  it saw the point before. The timers accumulate time_sum (seconds) and n
  (calls), so each evaluation's time is the difference before and after.

  Arguments:
    probe  = a probe name of ci.init_probes_cluster, e.g. "6x2pt_N".
    offset = the step of the warm-up; the timed points use offset + 1 ...
             offset + NEVAL, so no two probe sets share a point.

  Returns:
    (t_like, t_camb): numpy arrays [NEVAL] of the cosmolike and CAMB time
    of each evaluation, in seconds.

  Raises:
    RuntimeError when an evaluation did not call the likelihood or gave a
    non-finite log posterior.
  """
  ci.init_probes_cluster(possible_probes=probe)
  model.logposterior(jittered(offset), cached=False)  # warm-up
  t_like, t_camb = [], []
  for i in range(1, NEVAL + 1):
    l0, c0, n0 = lik.timer.time_sum, theory.timer.time_sum, lik.timer.n
    lp = model.logposterior(jittered(offset + i), cached=False)
    if lik.timer.n != n0 + 1 or not np.isfinite(lp.logpost):
      raise RuntimeError("step %d of %s did not evaluate the likelihood"
                         % (i, probe))
    t_like.append(lik.timer.time_sum - l0)
    t_camb.append(theory.timer.time_sum - c0)
  return np.array(t_like), np.array(t_camb)


rows = []
for k, probe in enumerate(PROBES):
  t_like, t_camb = run(probe, 10*(k + 1))
  rows.append((probe, t_like.mean(), t_like.std(), t_camb.mean()))

print("OMP_NUM_THREADS =", os.environ.get("OMP_NUM_THREADS"),
      " NEVAL =", NEVAL,
      " block sizes", ci.compute_data_vector_cluster_sizes())
for probe, mean, std, camb in rows:
  print("%-8s cosmolike %7.3f +- %5.3f s   CAMB %7.3f s"
        % (probe, mean, std, camb))
