# Unit tests for the likelihoods

These tests catch three kinds of silent breakage: a $\chi^2$ that
drifted because code or data changed by accident, a race condition (a
bug where evaluating several points in a row corrupts a later result
through leftover internal state or colliding OpenMP threads), and a
stale cache (a table that is not rebuilt when one of the parameters it
depends on moves).

They cover the two cluster combinations of arXiv 2503.13631 on the
synthetic DES Y6-like data set, and the three galaxy-only likelihoods
of this project on the DES Y3 placeholder data set:

| example | likelihood | data set | blocks of the data vector |
|---|---|---|---|
| `example1` | `des_cluster.combo_4x2pt_N` (4x2pt + N, the paper's CL+GC) | synthetic DES Y6-like | cluster counts N, cluster lensing, cluster clustering, cluster x galaxy, galaxy clustering (MagLim bins 1-3) |
| `example2` | `des_cluster.combo_6x2pt_N` (6x2pt + N, the paper's CL+3x2pt) | synthetic DES Y6-like | the blocks of `example1` plus cosmic shear, galaxy-galaxy lensing, and galaxy clustering of all six MagLim bins |
| `example3` | `des_cluster.cosmic_shear` | DES Y3 placeholder | cosmic shear |
| `example4` | `des_cluster.combo_3x2pt` | DES Y3 placeholder | cosmic shear, galaxy-galaxy lensing, galaxy clustering (redMaGiC bins 1-5) |
| `example4_2x2pt` | `des_cluster.combo_2x2pt` | DES Y3 placeholder | galaxy-galaxy lensing and galaxy clustering: `example4` with the likelihood renamed |

The intrinsic-alignment model of the cluster combinations is NLA: the
cluster lensing code has no TATT. The galaxy-only likelihoods run NLA
and TATT, and their tests cover both.

# Table of contents

1. [Running the tests](#run_tests)
2. [The tests](#the_tests)
    1. [The sector-ladder cache check](#cache_ladder)
    2. [The TATT sector ladder](#cache_ladder_tatt)
    3. [The non-Limber galaxy-galaxy lensing check](#nonlimber_ggl)
    4. [The non-Limber galaxy clustering check](#nonlimber_gg)
    5. [The photo-z convention checks](#photoz_conventions)
    6. [Accuracy checks](#accuracy_checks)
3. [Appendix](#appendix)
    1. [FAQ: Do the tests keep their own data?](#frozen_copy)
    2. [FAQ: Why is the reference $\chi^2$ zero?](#synthetic_vectors)
    3. [FAQ: Why do some tests start a process of their own?](#own_process)
    4. [FAQ: How can maintainers refresh the snapshot?](#refreeze)
    5. [FAQ: What are the other folders under tests?](#other_folders)

## Running the tests <a name="run_tests"></a>

We assume users are in the Conda cocoa environment from a previous
`conda activate cocoa` command, that the shell is bash, and that the
current folder is the cocoa main folder `cocoa/Cocoa`.

**Step :one:**: activate the private Python environment by sourcing
the script `start_cocoa.sh`

    source start_cocoa.sh

**Step :two:**: run the tests of this project

    python -m pytest ./projects/des_cluster/tests

Without pytest:

    python -m unittest discover -s ./projects/des_cluster/tests -v

The tests change no project files. Each test prints a progress line
per model build and per evaluation, then a report with the
computed $\chi^2$, the stored reference, the difference, and the pass
limit.

A full run performs about 355 likelihood evaluations, about 230 of
them on the cluster combinations. The test files force
`OMP_NUM_THREADS=4` internally, and every model build of the
reference, race, and accuracy tests runs in its own worker subprocess:
cosmolike aborts a process that initializes two configurations with
different data-set dimensions, and the cluster and the galaxy-only
examples use two data sets (2812 and 900 entries). The two
covariances of the frozen data (`frozen/data/des_cluster_y6.cov`,
124 MB, and `frozen/data/des_y3_cov_unblinded_final.txt`, 19 MB) are
stored with Git LFS: a clone without `git lfs pull` holds a pointer
file in place of each, and every test then stops at the manifest
check, naming the file.

## The tests <a name="the_tests"></a>

Each example gets a $\chi^2$ drift check and a race-condition check;
the galaxy-only examples get both in the NLA and in the TATT
intrinsic-alignment model. The TATT variants set

    IA_model: 1
    DES_A2_1: 0.05
    DES_BTA_1: 0.05
    DES_A2_2: -1.51541

The two checks and their pass limits:

| check | pass limit                                        | a failure means                    |
|-------|---------------------------------------------------|------------------------------------|
| $\Delta\chi^2$ | the recomputed $\chi^2$ must stay within 0.2 of the value stored in `frozen/reference_chi2.json` | code or data changed the numbers |
| race condition | the fiducial evaluated on its own vs evaluated again after nine other cosmologies; the two must agree within $10^{-4}$ | leftover state or an OpenMP race |

Everything the tests compare against lives under `frozen/`: one
snapshot of configurations, data, and reference values, captured
together when the references were generated and unchanged since. The
[Appendix](#appendix) explains how the snapshot is protected.

The test files and the configurations they cover:

| test | file | configuration | what it checks |
|---|---|---|---|
| 1 | `test_example1.py` | 4x2pt + N; IA modeling: NLA | $\Delta\chi^2$ against the stored reference at the fiducial point |
| 2 | `test_example1.py` | 4x2pt + N; IA modeling: NLA | race condition (OpenMP threading): fiducial alone vs after nine other cosmologies |
| 3 | `test_example2.py` | 6x2pt + N; IA modeling: NLA | $\Delta\chi^2$ against the stored reference at the fiducial point |
| 4 | `test_example2.py` | 6x2pt + N; IA modeling: NLA | race condition (OpenMP threading): fiducial alone vs after nine other cosmologies |
| 5 | `test_example3.py` | cosmic shear; IA modeling: NLA | $\Delta\chi^2$ against the stored reference at the fiducial point |
| 6 | `test_example3.py` | cosmic shear; IA modeling: NLA | race condition (OpenMP threading): fiducial alone vs after nine other cosmologies |
| 7 | `test_example3.py` | cosmic shear; IA modeling: TATT | $\Delta\chi^2$ against the stored reference at the fiducial point |
| 8 | `test_example3.py` | cosmic shear; IA modeling: TATT | race condition (OpenMP threading): fiducial alone vs after nine other cosmologies |
| 9 | `test_example4.py` | 3x2pt; IA modeling: NLA | $\Delta\chi^2$ against the stored reference at the fiducial point |
| 10 | `test_example4.py` | 3x2pt; IA modeling: NLA | race condition (OpenMP threading): fiducial alone vs after nine other cosmologies |
| 11 | `test_example4.py` | 3x2pt; IA modeling: TATT | $\Delta\chi^2$ against the stored reference at the fiducial point |
| 12 | `test_example4.py` | 3x2pt; IA modeling: TATT | race condition (OpenMP threading): fiducial alone vs after nine other cosmologies |
| 13 | `test_example4_2x2pt.py` | 2x2pt (`des_cluster.combo_2x2pt`); IA modeling: NLA | $\Delta\chi^2$ against the stored reference at the fiducial point |
| 14 | `test_example4_2x2pt.py` | 2x2pt (`des_cluster.combo_2x2pt`); IA modeling: NLA | race condition (OpenMP threading): fiducial alone vs after nine other cosmologies |
| 15 | `test_example4_2x2pt.py` | 2x2pt (`des_cluster.combo_2x2pt`); IA modeling: TATT | $\Delta\chi^2$ against the stored reference at the fiducial point |
| 16 | `test_example4_2x2pt.py` | 2x2pt (`des_cluster.combo_2x2pt`); IA modeling: TATT | race condition (OpenMP threading): fiducial alone vs after nine other cosmologies |
| - | `test_cache_consistency.py` | 4x2pt + N and 6x2pt + N; IA modeling: NLA | the sector ladder: every parameter group moves its own blocks of the data vector and no other, and the result never depends on the update history |
| - | `test_cache_consistency_tatt.py` | 3x2pt; IA modeling: TATT | the sector ladder with the FAST-PT tables of TATT: the result never depends on the update history |
| - | `test_nonlimber_ggl.py` | 3x2pt; IA modeling: NLA | Limber vs non-Limber galaxy-galaxy lensing (`adopt_limber_gs`): the flag reaches the code and moves $\gamma_t$ alone |
| - | `test_nonlimber_gg.py` | 3x2pt; IA modeling: NLA | Limber vs non-Limber galaxy clustering (`adopt_limber_gg`): the flag reaches the code and moves $w(\theta)$ alone |
| - | `test_photoz_conventions.py` | cosmic shear; IA modeling: NLA | the n(z) interpolation and z-column conventions: each flag reaches the code and the n(z) caches rebuild |
| A1-A6 | `test_accuracy.py` | 4x2pt + N, 6x2pt + N, cosmic shear, 2x2pt, 3x2pt with NLA; 3x2pt with TATT | advisory: $\Delta\chi^2$ of the default numerical settings against high-accuracy settings (no pass/fail) |

### The sector-ladder cache check (`test_cache_consistency.py`) <a name="cache_ladder"></a>

cosmolike caches every expensive stage behind its own key, and the
cluster tables (the richness-weighted mass integrals, the one-halo
lensing table, the cluster kernels, the Limber spectra of every
cluster pair) are keyed on combinations of those keys: the cosmology,
the mass-observable relation, the selection bias, the cluster redshift
kernels, the photo-z shifts, the intrinsic alignment, the galaxy bias.
A partial-invalidation bug - one sector's update path leaving a stale
static another sector consumes - produces wrong data vectors only in
MIXED update sequences, which the per-point checks never exercise.

The joint data vector has seven blocks: `ss` (cosmic shear), `gs`
(galaxy-galaxy lensing), `gg` (galaxy clustering), `cg` (cluster x
galaxy), `N` (cluster counts), `cc` (cluster clustering), `cs`
(cluster lensing). The test walks a deterministic ladder in one
process, three steps per sector, evaluating after every step:

| sector | parameters moved | blocks that must move |
|---|---|---|
| cosmology | `omegam`, `H0`, `As_1e9` | every block |
| intrinsic alignment | `DES_A1_1`, `DES_A1_2` | `ss`, `gs`, `cs` |
| source photo-z | every `DES_DZ_S` | `ss`, `gs`, `cs` |
| lens photo-z | every `DES_DZ_L` | `gs`, `gg`, `cg` |
| shear calibration | every `DES_M` | `ss`, `gs`, `cs` |
| galaxy bias | every `DES_B1` | `gs`, `gg`, `cg` |
| point mass | every `DES_PM` | `gs` |
| mass-observable relation | `DES_CL_LNLAMBDA0`, `DES_CL_A_LAMBDA`, `DES_CL_SIGMA_INT`, `DES_CL_B_LAMBDA` | `cg`, `N`, `cc`, `cs` |
| selection bias | `DES_CL_BS1`, `DES_CL_BS2`, `DES_CL_R0` | `cg`, `cc`, `cs` |

The cluster blocks read the galaxy-side sectors through their second
leg: cluster lensing carries the sources, cluster x galaxy carries
the lenses. 4x2pt + N has no `ss` and no `gs` block, so its point-mass
steps must leave the whole vector untouched. What must hold, for each
combination:

1. every step changes every unmasked entry of the blocks its sector
   enters, and leaves every other block bitwise unchanged;
2. the shear-calibration steps equal the analytic $(1+m_i)(1+m_j)$
   block rescale to $10^{-12}$ (cluster lensing scales by its source
   bin's factor);
3. the selection-bias steps equal the rescale by the selection factor
   $B(\theta)$ of eq. (23) to $10^{-12}$: cluster lensing and cluster x
   galaxy by one factor, cluster clustering by its square, the counts
   by nothing;
4. a no-op update changes nothing;
5. after a scramble (every sector moved at once), and after moving
   each sector alone, returning to the ladder's final point
   reproduces the recorded data vector and $\chi^2$ bit for bit;
6. a second instance walking the mirrored sector order lands on the
   same vector bit for bit;
7. a fresh process that evaluates the final point as its first and
   only evaluation computes the same vector and $\chi^2$ bit for bit.

Check 7 is the one that sees a table filled at the first point and
never rebuilt: every path inside one process shares that table, so
checks 5 and 6 agree with each other and only a process that starts
at the final point disagrees.

### The TATT sector ladder (`test_cache_consistency_tatt.py`) <a name="cache_ladder_tatt"></a>

The ladder above runs NLA, the only intrinsic-alignment model of the
cluster combinations, so it never reaches the FAST-PT tables TATT
adds, nor the code that rebuilds them.
This test walks the ladder of the des_y3 project on the galaxy-only
3x2pt likelihood with TATT: three cosmology steps, then three steps
of the intrinsic alignment (`DES_A1_1`, `DES_A1_2`, `DES_A2_1`,
`DES_A2_2`, `DES_BTA_1`), the source photo-z, the lens photo-z
(every `DES_DZ_L` and the stretch `DES_DZ2_L5`), and the shear
calibration, evaluating after every step. It then scrambles every
sector at once (galaxy bias and point mass included) and returns to
the ladder's final point. What must hold:

1. every step changes the data vector;
2. the shear-calibration steps equal the analytic $(1+m_i)(1+m_j)$
   block rescale to $10^{-12}$;
3. a no-op update changes nothing;
4. the return after the scramble reproduces the recorded data vector
   and $\chi^2$ bit for bit;
5. a second instance walking the mirrored sector order lands on the
   same vector bit for bit.

The NLA ladder of the galaxy-only likelihoods is not repeated: the
`ss`, `gs`, and `gg` blocks of 6x2pt + N already walk it.

### The non-Limber galaxy-galaxy lensing check (`test_nonlimber_ggl.py`) <a name="nonlimber_ggl"></a>

The likelihood yaml key `adopt_limber_gs` chooses how the
galaxy-galaxy lensing spectrum $C_\ell^{gs}$ is computed: `1` (the
default) uses the Limber approximation at every multipole; `0` takes
the multipoles below $\ell = 150$ from the exact projection, computed
by cosmolike's `C_gs_tomo` with the split of Fang, Krause, Eifler &
MacCrann (arXiv:1911.11947): an FFTLog integral of the linear power
spectrum plus, in Limber, what linear theory misses. Galaxy-galaxy
lensing defaults to Limber because its lensing kernel is broad; galaxy
clustering has its own key, `adopt_limber_gg` (next section). Every
likelihood of this project reads the two keys, the cluster
combinations included, and these two checks are the only tests that
flip them.

The test evaluates the frozen 3x2pt fiducial with Limber,
non-Limber, and Limber again in one process and reports
$\Delta\chi^2 = \delta^T C^{-1} \delta$, with $\delta$ the data-vector
difference and $C^{-1}$ the masked inverse covariance: the $\chi^2$ the
Limber model would score against a data set generated with non-Limber
galaxy-galaxy lensing. It also prints the contribution of each
lens-source pair. The assertions are a dead-flag floor
on $\Delta\chi^2$, that only galaxy-galaxy lensing entries change, a
bit-identical round trip back to Limber, and, last, the
frozen-reference check on the Limber evaluation.

> [!NOTE]
> The des_y3 version of this check also pins $\Delta\chi^2$ to a
> measured value within 5%. Here that assertion is armed by the
> constant `DCHI2_MEASURED` of the test file, which holds no value
> yet: the test prints the $\Delta\chi^2$ it measures, and recording
> that number in the constant switches the assertion on. The same
> holds for the galaxy clustering check below.

### The non-Limber galaxy clustering check (`test_nonlimber_gg.py`) <a name="nonlimber_gg"></a>

The likelihood yaml key `adopt_limber_gg` chooses how the galaxy
clustering spectrum $C_\ell^{gg}$ is computed: `0` takes the
multipoles below $\ell = 150$ from the exact projection (cosmolike's
`C_cl_tomo`, the same FFTLog split as the galaxy-galaxy lensing check
above), `1` uses the Limber approximation at every multipole; `0`
(non-Limber) is this project's default, in the cluster combinations
too. The lens galaxy redshift distributions are narrow, so the Limber
approximation fails at low $\ell$ for the clustering auto spectra.

The test evaluates the frozen 3x2pt fiducial with the default, the
other setting, and the default again in one process and reports
$\Delta\chi^2 = \delta^T C^{-1} \delta$, with $\delta$ the non-Limber
minus the Limber data vector, and the contribution of each lens bin.
The assertions are a dead-flag floor on $\Delta\chi^2$, that only
clustering entries change, a bit-identical round trip back to the
default, and, last, the frozen-reference check on the default
evaluation.

### The photo-z convention checks (`test_photoz_conventions.py`) <a name="photoz_conventions"></a>

The likelihood exposes two runtime knobs for how the n(z) table files
become the smooth distributions the Limber integrals consume, both
declared in the likelihood yamls and both defaulting to the
historical behavior: `photoz_interpolation_type` (0 = cubic spline,
1 = linear, 2+ = Steffen monotone, which cannot overshoot below zero
around a sharp feature in the table) and `photoz_zmid_convention`
(0 = the z column of the n(z) file holds Z_LOW left bin edges, so
the tabulated value belongs at the cell center z + dz/2; 1 = the
column holds Z_MID sample points). The two z-column readings differ
by a rigid dz/2 shift of every distribution. Every likelihood of
this project reads the two keys, and this check is the only test
that moves them.

The test evaluates the frozen cosmic-shear fiducial under five
settings in one process - the default, each alternative, and the
default again - and measures every alternative against the default:
$\Delta\chi^2 = \delta^T C^{-1} \delta$, with $\delta$ the
data-vector difference and $C^{-1}$ the masked inverse covariance.
The assertions are a dead-flag floor on each alternative (a stale
n(z) cache would give exactly zero), the frozen-reference check on
the default, and a bit-identical round trip back to the default (a
cache that fails to rebuild on the way back would fail loudly).

### Accuracy checks (`test_accuracy.py`, A1-A6) <a name="accuracy_checks"></a>

Checks A1 (4x2pt + N), A2 (6x2pt + N), A3 (cosmic shear), A4 (2x2pt),
and A5 (3x2pt), all with NLA, and A6 (3x2pt with TATT) re-evaluate
the frozen configuration at its frozen point with every numerical
setting pushed far beyond the defaults at once, and report the
$\Delta\chi^2$ between the high-accuracy evaluation and the stored
default-settings reference: the numerical error of the defaults.
There is no pass/fail. The settings:

| setting | raised to | what it controls |
|---------|-----------|------------------|
| `accuracyboost` (cosmolike) | 3 | sizes of cosmolike's internal lookup tables, including the dyadic z grid of the power-spectrum tables |
| `internal_accuracyboost` (cosmolike) | 2 | density of the C-FAST-PT convolution grid relative to the output table the likelihood interpolates |
| `integration_accuracy` (cosmolike) | 10 | order of cosmolike's fixed Gauss-Legendre quadratures and of the cluster mass integrals (the default is 1 in the cluster combinations and 0 in the galaxy-only likelihoods; 3 and above select the largest tables) |
| `lmax` (cosmolike) | 200000 | highest multipole of the internal harmonic-space $C_\ell$ tables that cosmolike transforms into the real-space correlation functions; arcminute scales need very high $\ell$ |
| `kmax_boltzmann` (cosmolike) | 40 | the k cutoff of the power spectrum the likelihood requests from CAMB |
| `AccuracyBoost` (CAMB) | 2 | CAMB's overall accuracy multiplier: denser sampling in every internal CAMB grid, the most expensive setting |
| `k_per_logint` (CAMB) | 50 | k samples CAMB computes per logarithmic interval of the transfer functions |
| `kmax` (CAMB) | 50 | highest k of CAMB's matter power spectrum; one physical cutoff with `kmax_boltzmann`, seen from the CAMB side |

`accuracyboost` stays at 3: the examples warn that the integration
tables of the donor project (desy1xplanck) broke down above it, and a
breakdown would read as a large numerical error of the defaults.

The numbers are those of the scale cuts in each example's mask, which
hide the smallest scales, where the numerical error is largest.
`validation/knob_sweep.py` measures the settings one at a time on the
cluster combinations with the small scales visible.

TATT gets one check, A6: the TATT tables enter cosmic shear and
galaxy-galaxy lensing, and the 3x2pt vector holds both blocks.

> [!NOTE]
> High-accuracy evaluations take minutes.

To run the accuracy checks on their own:

    python -m pytest ./projects/des_cluster/tests/test_accuracy.py

To run every other test while skipping these:

    python -m pytest ./projects/des_cluster/tests --ignore ./projects/des_cluster/tests/test_accuracy.py

# Appendix <a name="appendix"></a>

## :interrobang: FAQ: Do the tests keep their own data? <a name="frozen_copy"></a>

The tests read nothing from the live project: not `../data`, not the
`EXAMPLE_EVALUATE` yaml files, and not the likelihood default yaml
files. Instead, `frozen/` holds:

| `frozen/` entry | holds |
|---|---|
| `frozen_config_<example>.py` | the complete cobaya configuration as a yaml string, plus the exact evaluation point |
| `data/` | the tests' own copy of the two data sets. The synthetic DES Y6-like one: the dataset descriptors, the joint data vector, the covariance, the two masks, and the n(z) of the sources, the lenses, and the clusters. The DES Y3 placeholder: the dataset descriptor, the data vector, the covariance, the baseline mask, and the n(z) of the sources and the lenses. Plus the two data vectors generated for the galaxy-only examples, with their descriptors (next FAQ) |
| `reference_chi2.json` | the default-settings $\chi^2$ of each configuration |
| `EXAMPLE_EVALUATE{1,2,3,4}.yaml` | snapshots kept only so a human can diff how the live examples drifted since the freeze |

In the configuration modules every option and every parameter is
written out, including the ones that normally come from
`params_cluster.yaml` and the other default files, so editing those
files cannot change what the tests evaluate.

The files of `../data` that no example reads stay out of the copy
(the all-ones mask of the DES Y3 placeholder data set, the baryon PCA
inputs, and the table of observed redMaPPer counts; `DATA_IGNORE` in
`generate_frozen_reference.py` lists them).

`manifest_sha256.json` stores a SHA-256 hash (a fingerprint that
changes when any byte changes) of every file under `frozen/`. The
tests verify the manifest first and refuse to run when a file under
`frozen/` was edited, naming the file.

## :interrobang: FAQ: Why is the reference $\chi^2$ zero? <a name="synthetic_vectors"></a>

Every test evaluates against a data vector that is the model itself
at the test's own point, so every reference sits at the minimum of
the $\chi^2$, at zero.

That is what a $\chi^2$ comparison needs. Away from the minimum the
$\chi^2$ responds linearly to tiny numerical changes, and a harmless
rounding-level shift reads as an alarming difference; at the minimum
the response is quadratic, so the 0.2 limit of the drift checks and
the accuracy numbers measure the numerics.

The cluster combinations get such a vector from the project itself:
the shipped data vector is synthetic. `scripts/make_synthetic_data.py`
wrote the model, at the fiducial point of the examples (Table I of
arXiv 2503.13631), into `data/des_cluster_y6.datavector`, and the
frozen copy of that file serves `example1` and `example2`.

The galaxy-only likelihoods do not: the data vector of the DES Y3
placeholder is the real measurement, and the point of their examples
is not a fit to it. Their vectors are generated when the snapshot is
created, one per intrinsic-alignment model:

| data vector | generated from | serves |
|---|---|---|
| `frozen/data/synthetic_des_y3.dataset` | the 3x2pt model (`example4`) with NLA at the frozen point | the NLA tests of `example3`, `example4`, and `example4_2x2pt` |
| `frozen/data/tatt_des_y3.dataset` | the 3x2pt model (`example4`) with TATT at the TATT point | the TATT tests of the same three examples |

The full-length 3x2pt vector serves every galaxy-only probe (the
masks select their sections). The two examples therefore share their
numerical settings: `EXAMPLE_EVALUATE3.yaml` sets the
`kmax_boltzmann` of `EXAMPLE_EVALUATE4.yaml`, not the default of the
cosmic-shear likelihood.

## :interrobang: FAQ: Why do some tests start a process of their own? <a name="own_process"></a>

The cosmolike C layer keeps the dimensions of the data vector in
global variables and aborts a process that initializes a second
configuration with other dimensions. This project's tests use two
data sets (the 2812-entry joint vector of the cluster combinations
and the 900-entry vector of the DES Y3 placeholder), while pytest
runs every test file in one process.

The reference, race, and accuracy tests are covered by the worker
subprocess of each model build. The tests that must build their
models in process are not: the two ladders, the two non-Limber
checks, and the photo-z convention check read the full data vector
from the compiled interface and need several model builds to share
cosmolike's caches. The cluster ladder keeps pytest's process; each
galaxy-only test of that kind (`test_cache_consistency_tatt.py`,
`test_nonlimber_ggl.py`, `test_nonlimber_gg.py`,
`test_photoz_conventions.py`) restarts itself in a process of its
own, prints its report from there, and passes when that process
exits cleanly (the decorator `own_process` of `cocoa_test_utils.py`).
The commands above stay the same.

## :interrobang: FAQ: How can maintainers refresh the snapshot? <a name="refreeze"></a>

A deliberate change to the data vectors, n(z), covariance, examples,
or likelihood defaults requires a re-freeze.

We assume users are in the Conda cocoa environment from a previous
`conda activate cocoa` command, that the shell is bash, and that the
current folder is the cocoa main folder `cocoa/Cocoa`.

**Step :one:**: activate the private Python environment by sourcing
the script `start_cocoa.sh`

    source start_cocoa.sh

**Step :two:**: rebuild the snapshot

    python ./projects/des_cluster/tests/generate_frozen_reference.py --overwrite

It rebuilds `frozen/` from the current project, prints the eight new
reference $\chi^2$ values (one NLA value per example, plus one TATT
value per galaxy-only example), and rewrites the manifest. Review the
printed $\chi^2$ values before committing: they define what every
later test run compares against. A cluster reference far from zero
means the shipped data vector is no longer the model at the examples'
point (the model changed, or the examples did): regenerate the data
with `scripts/make_synthetic_data.py` first, or the tests would run
away from the minimum.

## :interrobang: FAQ: What are the other folders under tests? <a name="other_folders"></a>

pytest does not collect them (`pytest.ini`); each is run on its own:

| folder | holds |
|---|---|
| `reference/` | the independent Python reference model of the cluster observables and of the joint Gaussian covariance, with its own tests (`python -m pytest ./projects/des_cluster/tests/reference`) |
| `validation/` | the C code against the Python reference (`compare_reference.py`), the accuracy-setting sweep (`knob_sweep.py`), and the timing scripts |
| `lighthouse_reference/` | data vectors of the original cluster code (lighthouse), kept for comparison |
