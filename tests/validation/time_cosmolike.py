# Cosmolike-only timing of the des_cluster data vector, per probe set.
#
# Recipe of the cosmolike-dev skill: cobaya timing on, one warm-up, then
# NEVAL evaluations with EVERY sampled parameter moved (cosmology, photo-z
# shifts, shear calibration, IA, galaxy bias, MOR, selection), so no
# cosmology- or nuisance-keyed table is served from a cache. The likelihood
# component's timer (cosmolike + its Python glue) is read apart from the
# theory (CAMB) timer.
#
# Run inside the cocoa environment, from anywhere, on a quiet machine:
#
#   OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
#     python projects/des_cluster/tests/validation/time_cosmolike.py
#
# NEVAL (default 10) sets the number of timed evaluations per probe set.
import os
import numpy as np
import cosmolike_des_cluster_interface as ci
from cobaya.yaml import yaml_load_file
from cobaya.model import get_model

ROOT = os.environ["ROOTDIR"]
NEVAL = int(os.environ.get("NEVAL", "10"))
PROBES = ["6x2pt_N", "4x2pt_N", "3x2pt", "N", "cc", "cg", "cs"]

info = yaml_load_file(ROOT + "/projects/des_cluster/EXAMPLE_EVALUATE2.yaml")
info["likelihood"] = {"des_cluster.combo_6x2pt_N": {
    "path": ROOT + "/external_modules/data/des_cluster"}}
for p in ["DES_BARYON_Q1", "DES_BARYON_Q2"]:
  info["params"].pop(p, None)
info["debug"] = False
info["timing"] = True
os.chdir(ROOT)
model = get_model(info)
lik = list(model.likelihood.values())[0]
theory = list(model.theory.values())[0]

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
  # an MCMC-like step: every sampled parameter moves by a small, bounded
  # amount (<= 0.8% over the whole run), so every point stays inside the
  # priors
  q = {}
  for name, value in point.items():
    if name in names:
      q[name] = value*(1 + 1e-4*step) if value != 0 else 1e-5*step
    else:
      q[name] = value
  return q


def run(probe, offset):
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
