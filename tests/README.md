# Unit tests for the likelihoods

These tests catch three kinds of silent breakage: a $\chi^2$ that
drifted because code or data changed by accident, a race condition (a
bug where evaluating several points in a row corrupts a later result
through leftover internal state or colliding OpenMP threads), and a
stale cache (a table that is not rebuilt when one of the parameters it
depends on moves).

They cover the two cluster combinations of arXiv 2503.13631 on the
synthetic DES Y6-like data set:

| example | likelihood | blocks of the joint data vector |
|---|---|---|
| `example1` | `des_cluster.combo_4x2pt_N` (4x2pt + N, the paper's CL+GC) | cluster counts N, cluster lensing, cluster clustering, cluster x galaxy, galaxy clustering (MagLim bins 1-3) |
| `example2` | `des_cluster.combo_6x2pt_N` (6x2pt + N, the paper's CL+3x2pt) | the blocks of `example1` plus cosmic shear, galaxy-galaxy lensing, and galaxy clustering of all six MagLim bins |

The intrinsic-alignment model is NLA everywhere: the cluster lensing
code has no TATT. The galaxy-only likelihoods of this project
(`cosmic_shear`, `combo_3x2pt`, `combo_2x2pt`) are not tested here.

# Table of contents

1. [Running the tests](#run_tests)
2. [The tests](#the_tests)
    1. [The sector-ladder cache check](#cache_ladder)
    2. [Accuracy checks](#accuracy_checks)
3. [Appendix](#appendix)
    1. [FAQ: Do the tests keep their own data?](#frozen_copy)
    2. [FAQ: Why is the reference $\chi^2$ zero?](#synthetic_vectors)
    3. [FAQ: How can maintainers refresh the snapshot?](#refreeze)
    4. [FAQ: What are the other folders under tests?](#other_folders)

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

A full run performs about 230 likelihood evaluations and takes
about six minutes. The test files force `OMP_NUM_THREADS=4`
internally, and every model build of the reference, race, and
accuracy tests runs in its own worker subprocess (cosmolike aborts a
process that initializes two configurations with different data-set
dimensions; the two examples share one data set, so here the
isolation is preventive). The covariance of the frozen data set
(`frozen/data/des_cluster_y6.cov`, 124 MB) is stored with Git LFS: a
clone without `git lfs pull` holds a pointer file in its place, and
every test then stops at the manifest check, naming that file.

## The tests <a name="the_tests"></a>

Each example gets a $\chi^2$ drift check and a race-condition check:

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
| - | `test_cache_consistency.py` | 4x2pt + N and 6x2pt + N; IA modeling: NLA | the sector ladder: every parameter group moves its own blocks of the data vector and no other, and the result never depends on the update history |
| A1-A2 | `test_accuracy.py` | 4x2pt + N and 6x2pt + N; IA modeling: NLA | advisory: $\Delta\chi^2$ of the default numerical settings against high-accuracy settings (no pass/fail) |

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

### Accuracy checks (`test_accuracy.py`, A1-A2) <a name="accuracy_checks"></a>

Checks A1 (4x2pt + N) and A2 (6x2pt + N) re-evaluate the frozen
configuration at its frozen point with every numerical setting pushed
far beyond the defaults at once, and report the $\Delta\chi^2$
between the high-accuracy evaluation and the stored default-settings
reference: the numerical error of the defaults. There is no
pass/fail. The settings:

| setting | raised to | what it controls |
|---------|-----------|------------------|
| `accuracyboost` (cosmolike) | 3 | sizes of cosmolike's internal lookup tables, including the dyadic z grid of the power-spectrum tables |
| `internal_accuracyboost` (cosmolike) | 2 | density of the C-FAST-PT convolution grid relative to the output table the likelihood interpolates |
| `integration_accuracy` (cosmolike) | 10 | order of cosmolike's fixed Gauss-Legendre quadratures and of the cluster mass integrals (the default is 1; 3 and above select the largest tables) |
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
`validation/knob_sweep.py` measures the settings one at a time with
the small scales visible.

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
| `data/` | the tests' own copy of the synthetic DES Y6-like data set: the dataset descriptors, the joint data vector, the covariance, the two masks, and the n(z) of the sources, the lenses, and the clusters |
| `reference_chi2.json` | the default-settings $\chi^2$ of each configuration |
| `EXAMPLE_EVALUATE{1,2}.yaml` | snapshots kept only so a human can diff how the live examples drifted since the freeze |

In the configuration modules every option and every parameter is
written out, including the ones that normally come from
`params_cluster.yaml` and the other default files, so editing those
files cannot change what the tests evaluate.

The files of `../data` that the cluster examples never read stay out
of the copy (the DES-Y3 3x2pt placeholder data set and the baryon PCA
inputs; `DATA_IGNORE` in `generate_frozen_reference.py` lists them).

`manifest_sha256.json` stores a SHA-256 hash (a fingerprint that
changes when any byte changes) of every file under `frozen/`. The
tests verify the manifest first and refuse to run when a file under
`frozen/` was edited, naming the file.

## :interrobang: FAQ: Why is the reference $\chi^2$ zero? <a name="synthetic_vectors"></a>

This project's shipped data vector is synthetic:
`scripts/make_synthetic_data.py` wrote the model itself, at the
fiducial point of the examples (Table I of arXiv 2503.13631), into
`data/des_cluster_y6.datavector`. The examples therefore sit at the
minimum of the $\chi^2$, and both references are zero.

That is what a $\chi^2$ comparison needs. Away from the minimum the
$\chi^2$ responds linearly to tiny numerical changes, and a harmless
rounding-level shift reads as an alarming difference; at the minimum
the response is quadratic, so the 0.2 limit of the drift checks and
the accuracy numbers measure the numerics. (A project whose shipped
data is a real measurement has to generate such a vector when its
snapshot is created; here the frozen copy of the shipped vector
already is one.)

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

It rebuilds `frozen/` from the current project, prints the new
reference $\chi^2$ values, and rewrites the manifest. Review the
printed $\chi^2$ values before committing: they define what every
later test run compares against. A reference far from zero means the
shipped data vector is no longer the model at the examples' point
(the model changed, or the examples did): regenerate the data with
`scripts/make_synthetic_data.py` first, or the tests would run away
from the minimum.

## :interrobang: FAQ: What are the other folders under tests? <a name="other_folders"></a>

pytest does not collect them (`pytest.ini`); each is run on its own:

| folder | holds |
|---|---|
| `reference/` | the independent Python reference model of the cluster observables and of the joint Gaussian covariance, with its own tests (`python -m pytest ./projects/des_cluster/tests/reference`) |
| `validation/` | the C code against the Python reference (`compare_reference.py`), the accuracy-setting sweep (`knob_sweep.py`), and the timing scripts |
| `lighthouse_reference/` | data vectors of the original cluster code (lighthouse), kept for comparison |
