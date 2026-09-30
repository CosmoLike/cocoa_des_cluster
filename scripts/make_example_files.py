"""Maintainer tool: write the sampler examples of des_cluster.

The two evaluate examples (EXAMPLE_EVALUATE1.yaml = 4x2pt + N,
EXAMPLE_EVALUATE2.yaml = 6x2pt + N) are the hand-written source of truth:
their likelihood blocks, their nuisance-parameter note and their fiducial
point. This script derives every other example from them, so the
fourteen files cannot drift apart:

  EXAMPLE_MCMC{1,2}.yaml                 Metropolis-Hastings, CAMB
  EXAMPLE_EMUL2_EVALUATE{1,2}.yaml       evaluate, emulated Boltzmann inputs
  EXAMPLE_EMUL2_EVALUATE{3,4}.yaml       1000 evaluations at fresh draws of
                                         every parameter (the perf benchmark)
  EXAMPLE_EMUL2_MCMC{1,2}.yaml           Metropolis-Hastings
  EXAMPLE_EMUL2_POLY{1,2}.yaml           PolyChord
  EXAMPLE_EMUL2_MINIMIZE{1,2}.py         minimizer
  EXAMPLE_EMUL2_PROFILE{1,2}.py          profile likelihood
  EXAMPLE_EMUL2_NAUTILUS{1,2}.py         Nautilus

(1 = 4x2pt + N, 2 = 6x2pt + N; for EVALUATE, 3 and 4 likewise.) The EMUL2
cosmology and theory blocks and the sampler blocks are read from the donor
projects, which must be installed: projects/des_y3 (EXAMPLE_EMUL2_
EVALUATE2.yaml, EXAMPLE_EMUL2_MCMC2.yaml) and projects/desy1xplanck
(EXAMPLE_EMUL2_POLY1.yaml and the three .py scripts).

Usage (cocoa environment active, start_cocoa.sh sourced):

    python ./projects/des_cluster/scripts/make_example_files.py
"""
import os, re, sys
ROOT = os.environ["ROOTDIR"]
P = ROOT + "/projects/des_cluster"
D = ROOT + "/projects/des_y3"

def split(text):
  """header (before likelihood), likelihood, params, theory, sampler+output"""
  idx = {k: text.index("\n" + k + ":") + 1 for k in ["likelihood", "params", "theory", "sampler"]}
  return (text[:idx["likelihood"]], text[idx["likelihood"]:idx["params"]],
          text[idx["params"]:idx["theory"]], text[idx["theory"]:idx["sampler"]],
          text[idx["sampler"]:])

ev = {1: open(P + "/EXAMPLE_EVALUATE1.yaml").read(), 2: open(P + "/EXAMPLE_EVALUATE2.yaml").read()}
donor = open(D + "/EXAMPLE_EMUL2_EVALUATE2.yaml").read()
d_head, d_like, d_params, d_theory, d_samp = split(donor)

# EMUL2 cosmology block: everything of the donor params up to its nuisance section
emul_cosmo = d_params[:d_params.index("  # ----------------------------------------------------------------------------\n  # DES-Y3 3x2pt nuisance")]
# EMUL2 theory block without the commented fastpt note
emul_theory = d_theory[:d_theory.index("#  NOTE: under TATT")]

def nuisance_note(params_block):
  i = params_block.index("  # ----------------------------------------------------------------------------\n  # The nuisance parameters")
  return params_block[i:]

def evaluate_override(samp, emul):
  if not emul:
    return samp
  # the emulators are trained at mnu = 0.06 (fixed in the params block)
  out = []
  skip = False
  for line in samp.splitlines(keepends=True):
    if "Omega_nu h^2 = 0.00083" in line or re.match(r"\s+mnu:", line):
      continue
    out.append(line)
  return "".join(out)

EMUL_NOTE = """# Hybrid (EMUL2) version: cosmolike computes the data vector, but the
# Boltzmann inputs (distances, linear and nonlinear P(k)) come from the
# neural-network emulators of the theory block instead of CAMB, so nearly
# all the run time is cosmolike. The emulators are trained at mnu = 0.06
# eV, while the synthetic data vector was generated with CAMB at
# Omega_nu h^2 = 0.00083 (mnu = 0.077 eV): chi2 at the point below is
# small but not zero.
"""

def build(n, emul, sampler_text, name, extra_header=""):
  head, like, params, theory, samp = split(ev[n])
  if emul:
    like = like.replace("    path: ./external_modules/data/des_cluster\n",
                        "    use_emulator: 2\n    path: ./external_modules/data/des_cluster\n")
    params = emul_cosmo + nuisance_note(params)
    theory = emul_theory
    head = head.rstrip("\n") + "\n" + EMUL_NOTE
  if extra_header:
    # no evaluate point in the benchmark: drop the sentence about it
    head = "".join(l for l in head.splitlines(keepends=True)
                   if not l.startswith("# The data vector is the model at the point")
                   and not l.startswith("# (scripts/make_synthetic_data.py")
                   and not l.startswith("# small but not zero."))
    head = head.replace("Omega_nu h^2 = 0.00083 (mnu = 0.077 eV): chi2 at the point below is\n",
                        "Omega_nu h^2 = 0.00083 (mnu = 0.077 eV).\n")
    head = head.rstrip("\n") + "\n" + extra_header
  if sampler_text is None:
    body = evaluate_override(samp, emul)
    body = re.sub(r"output: .*\n?$", "", body).rstrip("\n") + "\n"
  else:
    body = sampler_text
  text = head + like + params + theory + body + "output: ./projects/des_cluster/chains/%s\n" % name
  open(P + "/" + name + ".yaml", "w").write(text)
  print("wrote", name + ".yaml", len(text.splitlines()), "lines")

if __name__ == "__main__":
  what = sys.argv[1] if len(sys.argv) > 1 else "all"
  if what == "all":
    sys.argv = [sys.argv[0], "all"]
  if what in ("evaluate", "all"):
    build(1, True, None, "EXAMPLE_EMUL2_EVALUATE1")
    build(2, True, None, "EXAMPLE_EMUL2_EVALUATE2")

# ---------------------------------------------------------------------------
# samplers
# ---------------------------------------------------------------------------
X = ROOT + "/projects/desy1xplanck"
SRC  = ["DES_DZ_S1", "DES_DZ_S2", "DES_DZ_S3", "DES_DZ_S4", "DES_A1_1", "DES_A1_2"]
SHEARCAL = ["DES_M1", "DES_M2", "DES_M3", "DES_M4"]
PM   = ["DES_PM%d" % i for i in range(1, 7)]
CL   = ["DES_CL_LNLAMBDA0", "DES_CL_A_LAMBDA", "DES_CL_SIGMA_INT", "DES_CL_B_LAMBDA",
        "DES_CL_BS1", "DES_CL_BS2", "DES_CL_R0"]
def lens(n):
  nb = 3 if n == 1 else 6
  return ["DES_DZ_L%d" % i for i in range(1, nb + 1)] + ["DES_B1_%d" % i for i in range(1, nb + 1)]

def fix_param(params, name, value, latex):
  """replace the block of a top-level parameter by a fixed value"""
  pat = re.compile(r"(?m)^  %s:\n(?:    .*\n)+" % re.escape(name))
  assert len(pat.findall(params)) == 1, name
  repl = "  %s:\n    value: %s\n    latex: %s\n" % (name, value, latex)
  return pat.sub(lambda m: repl, params)

def wrap(names, indent="            "):
  lines, cur = [], indent
  for nm in names:
    piece = nm + ", "
    if len(cur) + len(piece) > 78:
      lines.append(cur.rstrip()); cur = indent
    cur += piece
  lines.append(cur.rstrip().rstrip(","))
  return "\n".join(lines)

def mcmc_sampler(n, emul):
  donor_mcmc = open(D + "/EXAMPLE_EMUL2_MCMC2.yaml").read()
  s = donor_mcmc[donor_mcmc.index("\nsampler:") + 1:]
  s = s[:s.index("output:")]
  s = s.replace("    covmat: './projects/des_y3/EXAMPLE_MCMC2.covmat'\n",
                "    # (no covmat is shipped for the cluster combos: the first run learns it)\n"
                "    covmat: null\n")
  a = s.index("    blocking:"); b = s.index("    # ---------------------------------------------------------------------\n", a)
  cosmo = ["As_1e9", "ns", "H0", "omegab", "omegam"] + ([] if emul else ["mnu"])
  speeds = (1, 2, 25) if emul else (1, 6, 50)
  note = ("    # speeds are estimates (measure_speeds is off): with the emulated\n"
          "    # Boltzmann inputs a cosmology step costs one full cosmolike refill, a\n"
          "    # nuisance step a partial one; shear calibration only rescales blocks\n"
          if emul else
          "    # speeds are estimates (measure_speeds is off): a cosmology step pays\n"
          "    # CAMB (~6x the cosmolike time), a nuisance step only cosmolike; shear\n"
          "    # calibration only rescales blocks\n")
  blocking = (note + "    blocking:\n"
              "      - [%d,\n          [\n%s\n          ]\n        ]\n" % (speeds[0], wrap(cosmo)) +
              "      - [%d,\n          [\n%s\n          ]\n        ]\n" % (speeds[1], wrap(SRC + lens(n) + PM + CL)) +
              "      - [%d,\n          [\n%s\n          ]\n        ]\n" % (speeds[2], wrap(SHEARCAL)))
  return s[:a] + blocking + s[b:]

def poly_sampler():
  donor = open(X + "/EXAMPLE_EMUL2_POLY1.yaml").read()
  s = donor[donor.index("\nsampler:") + 1:]
  return s[:s.index("output:")]

PARTS = {}

LCDM_NOTE = """# Cosmology: LCDM as in the paper (w = -1, w0 + wa = -1 fixed). To sample
# w0-wa, give `w` and `w0pwa` a prior as in the EVALUATE examples.
"""

def build_sampling(n, emul, sampler_text, name):
  head, like, params, theory, samp = split(ev[n])
  if emul:
    like = like.replace("    path: ./external_modules/data/des_cluster\n",
                        "    use_emulator: 2\n    path: ./external_modules/data/des_cluster\n")
    params = emul_cosmo + nuisance_note(params)
    theory = emul_theory
    head = head.rstrip("\n") + "\n" + EMUL_NOTE
  head = head.replace("timing: True", "timing: False").replace("stop_at_error: True", "stop_at_error: False")
  head = head.rstrip("\n") + "\n" + LCDM_NOTE
  params = fix_param(params, "w", "-1.0", r"w_{0,\mathrm{DE}}")
  params = fix_param(params, "w0pwa", "-1.0", r"w_{0,\mathrm{DE}}+w_{a,\mathrm{DE}}")
  if not emul:
    # CAMB takes w and wa: w0pwa must not reach it
    params = params.replace("  w0pwa:\n    value: -1.0\n", "  w0pwa:\n    value: -1.0\n    drop: true\n")
  text = head + like + params + theory + sampler_text + "output: ./projects/des_cluster/chains/%s\n" % name
  open(P + "/" + name + ".yaml", "w").write(text)
  print("wrote", name + ".yaml", len(text.splitlines()), "lines")
  return like, params, theory

EMUL_PRIOR = '''prior:
  # These priors are meant to prevent the sampler to wander far off training
  g1: "lambda As_1e9: stats.norm.logpdf(As_1e9, loc=2.35, scale=1.6)"
  g2: "lambda ns: stats.norm.logpdf(ns, loc=0.96, scale=0.05)"
  g3: "lambda H0: stats.norm.logpdf(H0, loc=70, scale=10.0)"
  g4: "lambda omegab: stats.norm.logpdf(omegab, loc=0.045, scale=0.012)"
  g5: "lambda omegam: stats.norm.logpdf(omegam, loc=0.3 , scale=0.25)"
'''

def build_script(donor_file, name, like, params, theory, old_prog, old_root, old_outroot):
  t = open(donor_file).read()
  a = t.index('yaml_string=r"""') + len('yaml_string=r"""')
  b = t.index('\n"""', a)
  t = t[:a] + "\n" + like + EMUL_PRIOR + params + theory.rstrip("\n") + t[b:]
  for old, new in [(old_prog, name), (old_root, "./projects/des_cluster/"), (old_outroot, name.lower())]:
    assert old in t, (donor_file, old)
    t = t.replace(old, new)
  assert "desy1xplanck" not in t and "lsst_y1" not in t and "projects/example" not in t, \
      [l for l in t.splitlines() if "desy1xplanck" in l or "lsst_y1" in l or "projects/example" in l][:5]
  open(P + "/" + name + ".py", "w").write(t)
  print("wrote", name + ".py", len(t.splitlines()), "lines")

if __name__ == "__main__" and sys.argv[1] in ("samplers", "all"):
  for n in (1, 2):
    build_sampling(n, False, mcmc_sampler(n, False), "EXAMPLE_MCMC%d" % n)
    like, params, theory = build_sampling(n, True, mcmc_sampler(n, True), "EXAMPLE_EMUL2_MCMC%d" % n)
    build_sampling(n, True, poly_sampler(), "EXAMPLE_EMUL2_POLY%d" % n)
    PARTS[n] = {"like": like, "params": params, "theory": theory}

if __name__ == "__main__" and sys.argv[1] in ("scripts", "all"):
  donors = [
    ("EXAMPLE_EMUL2_MINIMIZE%d", X + "/EXAMPLE_EMUL2_MINIMIZE1.py",
     "prog='EXAMPLE_MINIMIZE1'", 'default="./projects/lsst_y1/"', 'default="example_min1"'),
    ("EXAMPLE_EMUL2_PROFILE%d", X + "/EXAMPLE_EMUL2_PROFILE1.py",
     "prog='EXAMPLE_EMUL_PROFILE1'", 'default="./projects/example/"', 'default="test.dat"'),
    ("EXAMPLE_EMUL2_NAUTILUS%d", X + "/EXAMPLE_EMUL2_NAUTILUS1.py",
     "prog='EXAMPLE_PROJECT_NAUTILUS1'", 'default="./projects/lsst_y1/"', 'default="example_nautilus1"'),
  ]
  for n in (1, 2):
    parts = PARTS[n]
    for pat, donor_file, old_prog, old_root, old_out in donors:
      name = pat % n
      t = open(donor_file).read()
      a = t.index('yaml_string=r"""') + len('yaml_string=r"""')
      b = t.index('\n"""', a)
      t = t[:a] + "\n" + parts["like"] + EMUL_PRIOR + parts["params"] + parts["theory"].rstrip("\n") + t[b:]
      for old, new in [(old_prog, "prog='%s'" % name),
                       (old_root, 'default="./projects/des_cluster/"'),
                       (old_out, 'default="%s"' % name)]:
        assert t.count(old) == 1, (donor_file, old, t.count(old))
        t = t.replace(old, new)
      # emulbaosn reads rdrag from emulrdrag, and cobaya hands it over as a
      # DERIVED parameter: with return_derived=False the derived store is
      # None and emulrdrag fails. Evaluate with derived parameters on and
      # keep the log-likelihood (element 0 of the returned pair).
      old_call = ("    res2 = model.loglike(point,\n"
                  "                         make_finite=False,\n"
                  "                         cached=False,\n"
                  "                         return_derived=False)\n")
      new_call = ("    # return_derived=True: emulbaosn needs rdrag, which cobaya passes\n"
                  "    # from emulrdrag as a derived parameter (off, that store is None\n"
                  "    # and the evaluation fails); [0] is the log-likelihood\n"
                  "    res2 = model.loglike(point,\n"
                  "                         make_finite=False,\n"
                  "                         cached=False,\n"
                  "                         return_derived=True)[0]\n")
      assert t.count(old_call) == 1, (donor_file, t.count(old_call))
      t = t.replace(old_call, new_call)
      left = [l for l in t.splitlines() if "desy1xplanck." in l or "/desy1xplanck" in l or "lsst_y1" in l or "projects/example" in l]
      assert not left, left[:5]
      open(P + "/" + name + ".py", "w").write(t)
      print("wrote", name + ".py", len(t.splitlines()), "lines")

BENCH_NOTE = """# Benchmark version: N evaluations, each at a fresh draw of EVERY sampled
# parameter from its `ref` distribution (no override), so no cosmolike
# table is served from a cache. With the emulated Boltzmann inputs nearly
# all the run time is cosmolike: the workload to put under `perf stat`.
# The seed makes the sequence of points reproducible.
"""
BENCH_SAMPLER = """sampler:
  evaluate:
    N: 1000
    seed: 1234
"""
if __name__ == "__main__" and sys.argv[1] in ("bench", "all"):
  build(1, True, BENCH_SAMPLER, "EXAMPLE_EMUL2_EVALUATE3", extra_header=BENCH_NOTE)
  build(2, True, BENCH_SAMPLER, "EXAMPLE_EMUL2_EVALUATE4", extra_header=BENCH_NOTE)
