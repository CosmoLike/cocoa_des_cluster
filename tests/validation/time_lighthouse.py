# Timing of the ORIGINAL cosmolike cluster code (lighthouse, prebuilt .so,
# single-threaded: it has no OpenMP) per probe set, one fresh process per
# set (static caches). Each evaluation changes sigma_8 so every table
# refills. The in-house EH + Halofit P(k) build (lighthouse's counterpart
# of CAMB, which the cocoa timings exclude) is timed separately as the
# first Pdelta call after a cosmology change, and subtracted.
#
#   NEVAL=3 python projects/des_cluster/tests/validation/time_lighthouse.py
#   NEVAL=3 python .../time_lighthouse.py cs      (one probe set)
import os, sys, time, subprocess
import numpy as np

HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "lighthouse_reference")
NEVAL = int(os.environ.get("NEVAL", "3"))

SETS = {
  "6x2pt_N": ["xip", "xim", "gammat", "wtheta", "w_cg", "gamma_c",
              "cluster_N", "w_cc"],
  "4x2pt_N": ["wtheta", "w_cg", "gamma_c", "cluster_N", "w_cc"],
  "3x2pt":   ["xip", "xim", "gammat", "wtheta"],
  "N":       ["cluster_N"],
  "cc":      ["w_cc"],
  "cg":      ["w_cg"],
  "cs":      ["gamma_c"],
}


def worker(name):
  sys.path.insert(0, HERE)
  import lh
  cfg = dict(lh.CONFIG)
  cfg["probes"] = SETS[name]
  ndata = lh.init_all(cfg)
  out = (lh.c_d*ndata)()
  nuis = lh.nuisance_struct(cfg)

  def cosmo(i):
    c = dict(cfg)
    c["sigma_8"] = cfg["sigma_8"]*(1 + 1e-3*i)
    return lh.cosmo_struct(c)

  lh.theory_wrapper(cosmo(0), nuis, out)  # warm-up
  tot, pk = [], []
  for i in range(1, NEVAL + 1):
    # P(k) build alone: set the new cosmology, force the Halofit table
    lh.set_all_parameters(cosmo(i), nuis)
    t0 = time.perf_counter()
    lh.Pdelta(0.1*2997.92458, 0.8)
    pk.append(time.perf_counter() - t0)
    # full theory at a NEW cosmology (everything refills, P(k) included)
    t0 = time.perf_counter()
    lh.theory_wrapper(cosmo(100 + i), nuis, out)
    tot.append(time.perf_counter() - t0)
  tot, pk = np.array(tot), np.array(pk)
  print("%-8s total %8.2f +- %5.2f s   EH+Halofit P(k) %6.2f s   "
        "cosmolike-only %8.2f s" % (name, tot.mean(), tot.std(), pk.mean(),
                                    tot.mean() - pk.mean()), flush=True)


if __name__ == "__main__":
  if len(sys.argv) > 1:
    worker(sys.argv[1])
  else:
    for name in SETS:
      subprocess.run([sys.executable, __file__, name], check=True)
