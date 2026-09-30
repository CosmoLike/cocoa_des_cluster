# Unit tests for the likelihoods

This project carries a barebones suite: the accuracy advisory checks
of `test_accuracy.py`. They measure how much numerical error the
examples' default settings carry, on cosmic shear, 2x2pt, and 3x2pt,
with the NLA intrinsic-alignment model.

# Table of contents

1. [Running the tests](#run_tests)
2. [The accuracy checks](#accuracy_checks)
3. [Appendix](#appendix)
    1. [FAQ: Do the tests keep their own data?](#frozen_copy)
    2. [FAQ: Why do the tests use their own data vector?](#synthetic_vectors)
    3. [FAQ: How can maintainers refresh the snapshot?](#refreeze)

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

The tests change no project files. Each check prints a progress line
per model build and per evaluation, then a report with the computed
$\chi^2$, the stored reference, and their difference. A
high-accuracy evaluation takes minutes. The test file forces
`OMP_NUM_THREADS=4` internally, and every model build runs in its own
worker subprocess (cosmolike aborts a process that initializes two
configurations with different data-set dimensions).

## The accuracy checks (`test_accuracy.py`, A1-A3) <a name="accuracy_checks"></a>

Each check re-evaluates one frozen configuration at its frozen point
with every numerical setting pushed far beyond the defaults at once,
and reports the $\Delta\chi^2$ between the high-accuracy evaluation
and the stored default-settings reference: the numerical error of
the defaults. There is no pass/fail.

| check | configuration | likelihood |
|---|---|---|
| A1 | `example1`: cosmic shear, NLA | `des_cluster.cosmic_shear` |
| A2 | `example2_2x2pt`: the 3x2pt configuration reduced to galaxy clustering plus galaxy-galaxy lensing, NLA | `des_cluster.combo_2x2pt` |
| A3 | `example2`: 3x2pt, NLA | `des_cluster.combo_3x2pt` |

| setting | raised to | what it controls |
|---------|-----------|------------------|
| `accuracyboost` (cosmolike) | 3 | sizes of cosmolike's internal lookup tables, including the dyadic z grid of the power-spectrum tables |
| `internal_accuracyboost` (cosmolike) | 2 | density of the C-FAST-PT convolution grid relative to the output table the likelihood interpolates |
| `integration_accuracy` (cosmolike) | 10 | extra refinement passes of cosmolike's numerical integrals |
| `lmax` (cosmolike) | 200000 | highest multipole of the internal harmonic-space $C_\ell$ tables that cosmolike transforms into the real-space correlation functions; arcminute scales need very high $\ell$ |
| `kmax_boltzmann` (cosmolike) | 40 | the k cutoff of the power spectrum the likelihood requests from CAMB |
| `AccuracyBoost` (CAMB) | 2 | CAMB's overall accuracy multiplier: denser sampling in every internal CAMB grid, the most expensive setting |
| `k_per_logint` (CAMB) | 50 | k samples CAMB computes per logarithmic interval of the transfer functions |
| `kmax` (CAMB) | 50 | highest k of CAMB's matter power spectrum; one physical cutoff with `kmax_boltzmann`, seen from the CAMB side |

`accuracyboost` stays at 3: the examples warn that the integration
tables of the donor project (desy1xplanck) broke down above it, and a
breakdown would read as a large numerical error of the defaults.

# Appendix <a name="appendix"></a>

## :interrobang: FAQ: Do the tests keep their own data? <a name="frozen_copy"></a>

The tests read nothing from the live project: not `../data`, not the
`EXAMPLE_EVALUATE` yaml files, and not the likelihood default yaml
files. Instead, `frozen/` holds:

| `frozen/` entry | holds |
|---|---|
| `frozen_config_<example>.py` | the complete cobaya configuration as a yaml string, plus the exact evaluation point |
| `data/` | the tests' own copy of the data vectors, covariance, n(z), and masks, plus the synthetic data vector |
| `reference_chi2.json` | the default-settings $\chi^2$ of each configuration |
| `EXAMPLE_EVALUATE{1,2}.yaml` | snapshots kept only so a human can diff how the live examples drifted since the freeze |

In the configuration modules every option and every parameter is
written out, including the ones that normally come from
`params_source.yaml` and the other default files, so editing those
files cannot change what the tests evaluate.

`manifest_sha256.json` stores a SHA-256 hash (a fingerprint that
changes when any byte changes) of every file under `frozen/`. The
tests verify the manifest first and refuse to run when a file under
`frozen/` was edited, naming the file.

## :interrobang: FAQ: Why do the tests use their own data vector? <a name="synthetic_vectors"></a>

This project's shipped data vector is real data, and the example
cosmology is not its best fit, so the $\chi^2$ there sits far from the
minimum, where it responds linearly to tiny numerical changes.

Every check therefore evaluates against a data vector generated with
the NLA model at the fiducial point when the snapshot was created:
`frozen/data/synthetic_des_cluster.dataset`. It comes from the 3x2pt
model, whose full-length vector serves every probe (each
likelihood's mask selects its section); at its own minimum the
$\chi^2$ response is quadratic and the accuracy numbers stay
meaningful.

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
later test run compares against.
