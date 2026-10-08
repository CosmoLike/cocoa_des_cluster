"""Maintainer tool: write the sampler examples of des_cluster.

An example is the input of one cobaya run (cobaya is the sampler
framework that loads the likelihoods): a YAML file, the text format of
nested `key: value` blocks, or a Python script that carries one. The
two evaluate examples (EXAMPLE_EVALUATE1.yaml = 4x2pt + N,
EXAMPLE_EVALUATE2.yaml = 6x2pt + N) are the hand-written source of truth:
their likelihood blocks, their nuisance-parameter note and their fiducial
point. This script derives sixteen examples from them, so these files
cannot drift apart:

  EXAMPLE_MCMC{1,2}.yaml                 Metropolis-Hastings, CAMB
  EXAMPLE_EMUL2_EVALUATE{1,2}.yaml       evaluate, emulated Boltzmann inputs
  EXAMPLE_EMUL2_BENCHMARK{1,2}.yaml      1000 evaluations at fresh draws of
                                         every parameter (the workload for
                                         timing, e.g. under Linux perf)
  EXAMPLE_EMUL2_MCMC{1,2}.yaml           Metropolis-Hastings
  EXAMPLE_EMUL2_POLY{1,2}.yaml           PolyChord
  EXAMPLE_EMUL2_MINIMIZE{1,2}.py         minimizer
  EXAMPLE_EMUL2_PROFILE{1,2}.py          profile likelihood
  EXAMPLE_EMUL2_NAUTILUS{1,2}.py         Nautilus

(1 = 4x2pt + N, 2 = 6x2pt + N in every name.) The other examples of the
project (EXAMPLE_EVALUATE3.yaml, EXAMPLE_EVALUATE4.yaml and
EXAMPLE_EVALUATE_COVARIANCE.yaml) are hand-written and not touched. The
EMUL2 cosmology and theory blocks and the sampler blocks are read from the
donor projects, which must be installed: projects/des_y3 (EXAMPLE_EMUL2_
EVALUATE2.yaml, EXAMPLE_EMUL2_MCMC2.yaml) and projects/desy1xplanck
(EXAMPLE_EMUL2_POLY1.yaml and the three .py scripts).

STEPS. The optional command-line argument selects one step:
  evaluate  EXAMPLE_EMUL2_EVALUATE{1,2}.yaml
  samplers  EXAMPLE_MCMC, EXAMPLE_EMUL2_MCMC and EXAMPLE_EMUL2_POLY{1,2}
  scripts   EXAMPLE_EMUL2_{MINIMIZE,PROFILE,NAUTILUS}{1,2}.py
  bench     EXAMPLE_EMUL2_BENCHMARK{1,2}.yaml
  all       every step above, in this order (the default)
Python runs a file from top to bottom. The steps are the
`if __name__ == "__main__"` blocks placed between the definitions
(__name__ is "__main__" only when the file runs as a script, not when it
is imported), and each runs when execution reaches it, using the names
defined above it. The scripts step reads the blocks that the samplers
step stores in PARTS, so it works only within "all"; "scripts" alone
stops with a KeyError. A name that matches no step writes nothing.

LIMIT OF THE SCRIPTS STEP. It copies the configuration that each donor
.py script embeds in a yaml_string block. The desy1xplanck donors, like
the shipped EXAMPLE_EMUL2_*.py of des_cluster, are short wrappers of
cosmolike_core/cocoa_hybrid_sampling.py, which reads
EXAMPLE_EMUL2_EVALUATE<n>.yaml when it runs; they hold no yaml_string,
and on them the scripts step stops with a ValueError (after the samplers
step, before the bench step).

Usage (cocoa environment active, start_cocoa.sh sourced: the script
reads the cocoa/Cocoa folder from the environment variable ROOTDIR):

    python ./projects/des_cluster/scripts/make_example_files.py [STEP]
"""
import os, re, sys
# ROOTDIR is the cocoa/Cocoa folder, set by start_cocoa.sh; without it
# this line raises a KeyError
ROOT = os.environ["ROOTDIR"]
# this project (P) and the des_y3 donor project (D)
P = ROOT + "/projects/des_cluster"
D = ROOT + "/projects/des_y3"

def split(text):
  """header (before likelihood), likelihood, params, theory, sampler+output

  Cuts the text of a cobaya YAML file at its top-level keys likelihood:,
  params:, theory: and sampler:, which must each start a line and appear
  in this order. The five pieces are consecutive, so their concatenation
  is the original text, and an edited piece can be put back in place.

  Arguments:
    text = the whole YAML file as one string.

  Returns:
    tuple of five strings: (header, likelihood block, params block,
    theory block, sampler block up to the end of the file, output line
    included).

  Raises:
    ValueError (from str.index) when one of the four keys is missing.
  """
  # dict comprehension: for each key, the index of the first character of
  # its line (str.index finds the first "\n<key>:"; + 1 skips the newline)
  idx = {k: text.index("\n" + k + ":") + 1 for k in ["likelihood", "params", "theory", "sampler"]}
  return (text[:idx["likelihood"]], text[idx["likelihood"]:idx["params"]],
          text[idx["params"]:idx["theory"]], text[idx["theory"]:idx["sampler"]],
          text[idx["sampler"]:])

# The hand-written sources, read when the file runs: {1: the text of
# EXAMPLE_EVALUATE1.yaml, 2: the text of EXAMPLE_EVALUATE2.yaml}
ev = {1: open(P + "/EXAMPLE_EVALUATE1.yaml").read(), 2: open(P + "/EXAMPLE_EVALUATE2.yaml").read()}
# the des_y3 EMUL2 evaluate example, cut into its five sections (the
# tuple assignment names the five strings split returns)
donor = open(D + "/EXAMPLE_EMUL2_EVALUATE2.yaml").read()
d_head, d_like, d_params, d_theory, d_samp = split(donor)

# EMUL2 cosmology block: everything of the donor params up to its nuisance
# section (As_1e9, ns, H0, omegab, omegam, the dark-energy parameters,
# mnu fixed at 0.06 eV, and the derived inputs of the theory codes)
emul_cosmo = d_params[:d_params.index("  # ----------------------------------------------------------------------------\n  # DES-Y3 3x2pt nuisance")]
# EMUL2 theory block (the emulators of distances, r_drag and P(k)) without
# the commented fastpt note that follows it in the donor
emul_theory = d_theory[:d_theory.index("#  NOTE: under TATT")]

def nuisance_note(params_block):
  """Return the closing note of a des_cluster params block.

  The note starts at the dashed comment line above "# The nuisance
  parameters" and runs to the end of the block; it tells the reader that
  the nuisance parameters and their priors come from the likelihood
  defaults (likelihood/params_*.yaml), so the EMUL2 examples keep it
  below the donor's cosmology block.

  Arguments:
    params_block = the params section of an EXAMPLE_EVALUATE yaml, as
      returned by split.

  Returns:
    the note, a string ending where the params block ends.

  Raises:
    ValueError (from str.index) when the note is not found.
  """
  i = params_block.index("  # ----------------------------------------------------------------------------\n  # The nuisance parameters")
  return params_block[i:]

def evaluate_override(samp, emul):
  """Adapt the evaluate sampler block of an example to the EMUL2 version.

  The EMUL2 params block fixes mnu at 0.06 eV, the value the emulators
  are trained at, so the override point drops its mnu line and the
  comment line above it (the one that converts Omega_nu h^2 = 0.00083 to
  mnu). Without emulators the block is returned unchanged.

  Arguments:
    samp = the sampler section of an EXAMPLE_EVALUATE yaml (from split).
    emul = True for the EMUL2 version.

  Returns:
    the sampler text, edited when emul is True.
  """
  if not emul:
    return samp
  # the emulators are trained at mnu = 0.06 (fixed in the params block)
  out = []
  skip = False
  # splitlines(keepends=True) keeps each line's newline, so "".join puts
  # the kept lines back together unchanged
  for line in samp.splitlines(keepends=True):
    # re.match tests the start of the line: indentation, then "mnu:"
    if "Omega_nu h^2 = 0.00083" in line or re.match(r"\s+mnu:", line):
      continue
    out.append(line)
  return "".join(out)

# Header comment added to every EMUL2 example (a triple-quoted string
# keeps its line breaks)
EMUL_NOTE = """# Hybrid (EMUL2) version: cosmolike computes the data vector, but the
# Boltzmann inputs (distances, linear and nonlinear P(k)) come from the
# neural-network emulators of the theory block instead of CAMB, so nearly
# all the run time is cosmolike. The emulators are trained at mnu = 0.06
# eV, while the synthetic data vector was generated with CAMB at
# Omega_nu h^2 = 0.00083 (mnu = 0.077 eV): chi2 at the point below is
# small but not zero.
"""

def build(n, emul, sampler_text, name, extra_header=""):
  """Write one evaluate-type example (EMUL2 evaluate or benchmark).

  Starts from evaluate example n. With emul the likelihood block gains
  "use_emulator: 2" above its path line, the params block becomes the
  donor's EMUL2 cosmology block plus this project's nuisance note, the
  theory block becomes the emulator block, and EMUL_NOTE joins the
  header. The sampler block
  is the example's own (sampler_text None, after evaluate_override and
  without its output line) or sampler_text; the output line names
  chains/<name>. With extra_header (the benchmark) the header also loses
  its sentences about the evaluate point (the example's own and the chi2
  clause of EMUL_NOTE), and extra_header is appended.

  Arguments:
    n = 1 (4x2pt + N, EXAMPLE_EVALUATE1.yaml) or 2 (6x2pt + N).
    emul = True for the EMUL2 (emulated Boltzmann inputs) version.
    sampler_text = the sampler block to use, or None to keep the
      example's evaluate block.
    name = the example name, without .yaml.
    extra_header = text appended to the header ("" for none).

  Returns:
    nothing.

  Side effects:
    writes projects/des_cluster/<name>.yaml (overwritten) and prints a
    line.
  """
  head, like, params, theory, samp = split(ev[n])
  if emul:
    like = like.replace("    path: ./external_modules/data/des_cluster\n",
                        "    use_emulator: 2\n    path: ./external_modules/data/des_cluster\n")
    params = emul_cosmo + nuisance_note(params)
    theory = emul_theory
    head = head.rstrip("\n") + "\n" + EMUL_NOTE
  if extra_header:
    # no evaluate point in the benchmark: drop the sentence about it. The
    # generator inside join keeps the header lines (newlines included)
    # that start with none of the three prefixes.
    head = "".join(l for l in head.splitlines(keepends=True)
                   if not l.startswith("# The data vector is the model at the point")
                   and not l.startswith("# (scripts/make_synthetic_data.py")
                   and not l.startswith("# small but not zero."))
    head = head.replace("Omega_nu h^2 = 0.00083 (mnu = 0.077 eV): chi2 at the point below is\n",
                        "Omega_nu h^2 = 0.00083 (mnu = 0.077 eV).\n")
    head = head.rstrip("\n") + "\n" + extra_header
  if sampler_text is None:
    body = evaluate_override(samp, emul)
    # remove the final "output: ..." line ($ anchors at the end of the
    # text); the output line is written again below
    body = re.sub(r"output: .*\n?$", "", body).rstrip("\n") + "\n"
  else:
    body = sampler_text
  text = head + like + params + theory + body + "output: ./projects/des_cluster/chains/%s\n" % name
  open(P + "/" + name + ".yaml", "w").write(text)
  print("wrote", name + ".yaml", len(text.splitlines()), "lines")

# Step selection (module docstring, STEPS), then the evaluate step.
if __name__ == "__main__":
  what = sys.argv[1] if len(sys.argv) > 1 else "all"
  # the later step blocks test sys.argv[1]: without an argument it would
  # not exist, so "all" is written into the argument list
  if what == "all":
    sys.argv = [sys.argv[0], "all"]
  if what in ("evaluate", "all"):
    build(1, True, None, "EXAMPLE_EMUL2_EVALUATE1")
    build(2, True, None, "EXAMPLE_EMUL2_EVALUATE2")

# ---------------------------------------------------------------------------
# samplers
# ---------------------------------------------------------------------------
# the desy1xplanck donor project (PolyChord block, the three .py scripts)
X = ROOT + "/projects/desy1xplanck"
# Parameter names of the MCMC blocking (mcmc_sampler): the source photo-z
# shifts and NLA parameters, the shear calibrations, the six point masses
# of gamma_t ("%d" % i writes the integer i into the name), and the
# sampled cluster parameters (mass-observable relation, selection bias)
SRC  = ["DES_DZ_S1", "DES_DZ_S2", "DES_DZ_S3", "DES_DZ_S4", "DES_A1_1", "DES_A1_2"]
SHEARCAL = ["DES_M1", "DES_M2", "DES_M3", "DES_M4"]
PM   = ["DES_PM%d" % i for i in range(1, 7)]
CL   = ["DES_CL_LNLAMBDA0", "DES_CL_A_LAMBDA", "DES_CL_SIGMA_INT", "DES_CL_B_LAMBDA",
        "DES_CL_BS1", "DES_CL_BS2", "DES_CL_R0"]
def lens(n):
  """The lens nuisance parameters that combination n samples.

  4x2pt + N (n = 1) uses lens bins 1-3, 6x2pt + N (n = 2) all six: the
  photo-z shift DES_DZ_L<i> and the linear bias DES_B1_<i> of each.

  Arguments:
    n = 1 (4x2pt + N) or 2 (6x2pt + N).

  Returns:
    list of parameter names: the shifts, then the biases.
  """
  nb = 3 if n == 1 else 6
  return ["DES_DZ_L%d" % i for i in range(1, nb + 1)] + ["DES_B1_%d" % i for i in range(1, nb + 1)]

def point_mass(n):
  """The point masses of gamma_t that combination n samples: all six for
  6x2pt + N (n = 2), none for 4x2pt + N (n = 1), which has no gamma_t block
  and fixes them in combo_4x2pt_N.yaml. The MCMC blocking may only name
  sampled parameters.

  Arguments:
    n = 1 (4x2pt + N) or 2 (6x2pt + N).

  Returns:
    list of parameter names (PM or an empty list).
  """
  return PM if n == 2 else []

def fix_param(params, name, value, latex):
  """replace the block of a top-level parameter by a fixed value

  The block is the line "  <name>:" followed by every line indented by
  at least four spaces; it becomes "value:" and "latex:" lines only, so
  the parameter is fixed (no prior, no ref, no proposal).

  Arguments:
    params = the text of a params block.
    name = the parameter name, e.g. "w".
    value = the fixed value, as text (e.g. "-1.0").
    latex = the LaTeX label, as text.

  Returns:
    the params text with that one block replaced.

  Raises:
    AssertionError naming the parameter when its block is not found
    exactly once.
  """
  # (?m) is multiline mode: ^ matches at the start of every line; the
  # (?:    .*\n)+ group takes the one or more indented lines of the block,
  # and re.escape protects any regex character in the name
  pat = re.compile(r"(?m)^  %s:\n(?:    .*\n)+" % re.escape(name))
  assert len(pat.findall(params)) == 1, name
  repl = "  %s:\n    value: %s\n    latex: %s\n" % (name, value, latex)
  # the replacement is a function that returns repl unchanged: a plain
  # replacement string would have its backslashes read as escapes, and
  # the LaTeX of latex (e.g. \mathrm) would be mangled or rejected
  return pat.sub(lambda m: repl, params)

def wrap(names, indent="            "):
  """Join parameter names into indented lines of at most 78 characters.

  Used for the lists of the MCMC blocking. Each name is followed by
  ", "; a new line starts when the next name would pass 78 characters,
  and the comma after the last name is removed.

  Arguments:
    names = list of parameter names.
    indent = the prefix of every line (12 spaces by default, the depth of
      the blocking lists).

  Returns:
    one string, the lines joined with newlines.
  """
  # lines collects the finished lines; cur is the line being filled
  lines, cur = [], indent
  for nm in names:
    piece = nm + ", "
    # the line is full: store it without its trailing space and start a
    # new one (";" separates two statements on one line)
    if len(cur) + len(piece) > 78:
      lines.append(cur.rstrip()); cur = indent
    cur += piece
  lines.append(cur.rstrip().rstrip(","))
  return "\n".join(lines)

def mcmc_sampler(n, emul):
  """The Metropolis-Hastings sampler block of combination n.

  Starts from the sampler block of des_y3/EXAMPLE_EMUL2_MCMC2.yaml (from
  "sampler:" to "output:"), replaces its covmat (the proposal covariance
  file of des_y3) with null, and replaces its blocking list with three
  blocks of this project's sampled parameters: cosmology (mnu added
  under CAMB), the nuisance parameters (sources, lenses, point masses
  for n = 2, clusters), and the shear calibrations. Each block carries
  its oversampling factor (cobaya's manual blocking: how much more often
  the sampler proposes steps in that block than in the slowest one):
  (1, 6, 50) under CAMB, whose run takes about six times the cosmolike
  time, and (1, 2, 25) with the emulators. The factors are estimates of
  the relative speeds (measure_speeds is off in the donor block), as the
  comment written above the blocking says.

  Arguments:
    n = 1 (4x2pt + N) or 2 (6x2pt + N).
    emul = True for the EMUL2 version.

  Returns:
    the sampler block, a string ending before the output line.
  """
  donor_mcmc = open(D + "/EXAMPLE_EMUL2_MCMC2.yaml").read()
  s = donor_mcmc[donor_mcmc.index("\nsampler:") + 1:]
  s = s[:s.index("output:")]
  s = s.replace("    covmat: './projects/des_y3/EXAMPLE_MCMC2.covmat'\n",
                "    # (no covmat is shipped for the cluster combos: the first run learns it)\n"
                "    covmat: null\n")
  # the donor's blocking list runs from "    blocking:" (index a) to the
  # next dashed comment line after it (index b); a; b on one line are two
  # statements
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
  # each block is "- [factor, [names]]" in cobaya's blocking syntax; %
  # puts the factor and the wrapped names into the template
  blocking = (note + "    blocking:\n"
              "      - [%d,\n          [\n%s\n          ]\n        ]\n" % (speeds[0], wrap(cosmo)) +
              "      - [%d,\n          [\n%s\n          ]\n        ]\n" % (speeds[1], wrap(SRC + lens(n) + point_mass(n) + CL)) +
              "      - [%d,\n          [\n%s\n          ]\n        ]\n" % (speeds[2], wrap(SHEARCAL)))
  return s[:a] + blocking + s[b:]

def poly_sampler():
  """The PolyChord sampler block of desy1xplanck/EXAMPLE_EMUL2_POLY1.yaml.

  The local name donor hides the module-level donor inside this function
  only.

  Returns:
    the block from "sampler:" up to (not including) "output:".
  """
  donor = open(X + "/EXAMPLE_EMUL2_POLY1.yaml").read()
  s = donor[donor.index("\nsampler:") + 1:]
  return s[:s.index("output:")]

# {n: {"like", "params", "theory": text}}: the EMUL2 blocks of
# combination n, stored by the samplers step for the scripts step
PARTS = {}

# Header comment of the sampling examples (build_sampling fixes w and w0pwa)
LCDM_NOTE = """# Cosmology: LCDM as in the paper (w = -1, w0 + wa = -1 fixed). To sample
# w0-wa, give `w` and `w0pwa` a prior as in the EVALUATE examples.
"""

def build_sampling(n, emul, sampler_text, name):
  """Write one sampling example (MCMC or PolyChord) of combination n.

  Starts from evaluate example n and makes the same EMUL2 changes as
  build. The header switches timing and stop_at_error off (with
  stop_at_error: False cobaya gives a point whose evaluation raises an
  error zero likelihood and continues) and gains LCDM_NOTE; w and w0pwa
  are fixed at -1 (LCDM, as in the paper), and without emulators w0pwa
  is also dropped from the parameters passed on, because CAMB takes w
  and wa.

  Arguments:
    n = 1 (4x2pt + N) or 2 (6x2pt + N).
    emul = True for the EMUL2 version.
    sampler_text = the sampler block (mcmc_sampler or poly_sampler).
    name = the example name, without .yaml.

  Returns:
    (like, params, theory): the three edited blocks, which the scripts
    step reuses.

  Side effects:
    writes projects/des_cluster/<name>.yaml (overwritten) and prints a
    line.
  """
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

# The prior block placed in the YAML of the generated .py scripts:
# Gaussian priors on the five cosmological parameters that keep the
# sampler near the emulators' training region. cobaya evaluates each
# lambda string with scipy.stats available as stats. The centers and
# widths are those of the EMUL profile scripts of other Cocoa projects
# (e.g. projects/roman_real/EXAMPLE_EMUL_PROFILE1.py); how they were
# chosen is not recorded here.
EMUL_PRIOR = '''prior:
  # These priors are meant to prevent the sampler to wander far off training
  g1: "lambda As_1e9: stats.norm.logpdf(As_1e9, loc=2.35, scale=1.6)"
  g2: "lambda ns: stats.norm.logpdf(ns, loc=0.96, scale=0.05)"
  g3: "lambda H0: stats.norm.logpdf(H0, loc=70, scale=10.0)"
  g4: "lambda omegab: stats.norm.logpdf(omegab, loc=0.045, scale=0.012)"
  g5: "lambda omegam: stats.norm.logpdf(omegam, loc=0.3 , scale=0.25)"
'''

def build_script(donor_file, name, like, params, theory, old_prog, old_root, old_outroot):
  """Write one .py example from a donor script that embeds its YAML.

  Not called anywhere in this file: the scripts step below repeats these
  steps inline, with exact-count checks and an extra rewrite of the
  model.loglike call. The donor's YAML text, the raw triple-quoted string
  assigned to yaml_string, is replaced by like + EMUL_PRIOR + params +
  theory; then the three donor names are replaced and no line may still
  name a donor project.

  Arguments:
    donor_file = path of the donor .py script.
    name = the example name, without .py.
    like, params, theory = the EMUL2 blocks of build_sampling.
    old_prog = the donor's program name, replaced by name.
    old_root = the donor's default root folder, replaced by
      "./projects/des_cluster/".
    old_outroot = the donor's default output name, replaced by
      name.lower().

  Returns:
    nothing.

  Raises:
    ValueError when the donor has no yaml_string block; AssertionError
    when a donor name is missing or a donor project is still named.

  Side effects:
    writes projects/des_cluster/<name>.py (overwritten) and prints a
    line.
  """
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

# The samplers step (module docstring, STEPS): three sampling examples
# per combination, and the EMUL2 blocks kept in PARTS for the scripts step.
if __name__ == "__main__" and sys.argv[1] in ("samplers", "all"):
  for n in (1, 2):
    build_sampling(n, False, mcmc_sampler(n, False), "EXAMPLE_MCMC%d" % n)
    like, params, theory = build_sampling(n, True, mcmc_sampler(n, True), "EXAMPLE_EMUL2_MCMC%d" % n)
    build_sampling(n, True, poly_sampler(), "EXAMPLE_EMUL2_POLY%d" % n)
    PARTS[n] = {"like": like, "params": params, "theory": theory}

# The scripts step (module docstring, STEPS and LIMIT): PARTS[n] exists
# only when the samplers step ran in this same run.
if __name__ == "__main__" and sys.argv[1] in ("scripts", "all"):
  # one tuple per script: (output name pattern, donor script, then the
  # three donor strings to replace: its argparse program name, its
  # default root folder and its default output name)
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
      # % n writes the combination number into the name pattern
      name = pat % n
      t = open(donor_file).read()
      # the embedded YAML runs from just after 'yaml_string=r"""' (index
      # a) to the next line that closes the string (index b); str.index
      # raises ValueError when the donor has no such block
      a = t.index('yaml_string=r"""') + len('yaml_string=r"""')
      b = t.index('\n"""', a)
      t = t[:a] + "\n" + parts["like"] + EMUL_PRIOR + parts["params"] + parts["theory"].rstrip("\n") + t[b:]
      # each donor name must occur exactly once, so a replacement can
      # neither miss nor hit unrelated text
      for old, new in [(old_prog, "prog='%s'" % name),
                       (old_root, 'default="./projects/des_cluster/"'),
                       (old_out, 'default="%s"' % name)]:
        assert t.count(old) == 1, (donor_file, old, t.count(old))
        t = t.replace(old, new)
      # emulbaosn reads rdrag from emulrdrag, and cobaya hands it over as a
      # derived parameter: with return_derived=False the derived store is
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
      # a donor that already evaluates with return_derived=True carries
      # this other comment block above the call: map that form onto the
      # same text too
      fixed_call = ("    # return_derived=True: emulbaosn needs rdrag, which emulrdrag also writes\n"
                    "    # into the derived-parameter store; with return_derived=False that store\n"
                    "    # is None and the evaluation fails. [0] is the log-likelihood\n"
                    "    res2 = model.loglike(point,\n"
                    "                         make_finite=False,\n"
                    "                         cached=False,\n"
                    "                         return_derived=True)[0]\n")
      if t.count(old_call) == 1:
        t = t.replace(old_call, new_call)
      else:
        assert t.count(fixed_call) == 1, (donor_file, "loglike call not found")
        t = t.replace(fixed_call, new_call)
      # comprehension with a condition: the lines that still name a donor
      # project; any such line stops the script
      left = [l for l in t.splitlines() if "desy1xplanck." in l or "/desy1xplanck" in l or "lsst_y1" in l or "projects/example" in l]
      assert not left, left[:5]
      open(P + "/" + name + ".py", "w").write(t)
      print("wrote", name + ".py", len(t.splitlines()), "lines")

# Header comment of the benchmark examples
BENCH_NOTE = """# Benchmark version: N evaluations, each at a fresh draw of EVERY sampled
# parameter from its `ref` distribution (no override), so no cosmolike
# table is served from a cache. With the emulated Boltzmann inputs nearly
# all the run time is cosmolike: the workload to put under `perf stat`.
# The seed makes the sequence of points reproducible.
"""
# The benchmark's sampler block: N = 1000 evaluations per run, each at a
# point drawn from the ref distributions of the sampled parameters. Any
# fixed seed makes the sequence of points reproducible, so two runs time
# the same work; the choice of 1000 is not explained here.
BENCH_SAMPLER = """sampler:
  evaluate:
    N: 1000
    seed: 1234
"""
# The bench step (module docstring, STEPS).
if __name__ == "__main__" and sys.argv[1] in ("bench", "all"):
  build(1, True, BENCH_SAMPLER, "EXAMPLE_EMUL2_BENCHMARK1", extra_header=BENCH_NOTE)
  build(2, True, BENCH_SAMPLER, "EXAMPLE_EMUL2_BENCHMARK2", extra_header=BENCH_NOTE)
