"""Shared harness for the des_cluster unit tests: the project's data
bound to the shared Cocoa test machinery.

The machinery itself (frozen-state verification, the chi2 pipeline,
worker-subprocess isolation, the race check, and the terminal
reports) lives in
external_modules/code/cosmolike_core/cocoa_testing.py. This file
carries what is des_cluster's alone - the examples table (the two
cluster combinations of arXiv 2503.13631: example1 = 4x2pt + N,
example2 = 6x2pt + N, both on the synthetic DES Y6-like data set),
and the high-accuracy settings of the accuracy checks - and binds it
to ONE cocoa_testing.CocoaTestHarness instance whose methods are
re-exported under the historical names, so the test modules and
generate_frozen_reference.py import everything from this module.

The intrinsic-alignment model is NLA everywhere: the cluster lensing
code has no TATT, so this project carries no TATT point, no TATT
data vectors, and no FAST-PT comparison (the harness tables of those
checks are empty).

The frozen-state doctrine is unchanged: everything a test evaluates
lives under tests/frozen/, pinned byte for byte by
tests/manifest_sha256.json and verified before any model is built;
refreshing the frozen state stays a deliberate maintainer action
(generate_frozen_reference.py --overwrite).
"""

import os
import sys

# ---- tests/ paths -----------------------------------------------------------

# Everything the tests read or write lives relative to this folder, so
# the suite works no matter which directory pytest is launched from
# (__file__ is this module's own path; dirname strips the file name).
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
FROZEN_DIR = os.path.join(TESTS_DIR, "frozen")
MANIFEST_FILE = os.path.join(TESTS_DIR, "manifest_sha256.json")
REFERENCE_FILE = os.path.join(FROZEN_DIR, "reference_chi2.json")

# ---- the shared machinery ---------------------------------------------------

# The import is path-based (tests/ is three levels below Cocoa/, which
# holds external_modules/code/cosmolike_core) so it works before
# start_cocoa.sh's python-path setup runs.
_CORE_DIR = os.path.abspath(os.path.join(
    TESTS_DIR, "..", "..", "..", "external_modules", "code",
    "cosmolike_core"))
if _CORE_DIR not in sys.path:
    sys.path.insert(0, _CORE_DIR)
import cocoa_testing as _cct

# ---- the project data ------------------------------------------------------

# This project's shipped data_file is SYNTHETIC: the joint data vector
# data/des_cluster_y6.datavector is the model itself at the examples'
# fiducial point (scripts/make_synthetic_data.py, Table I of arXiv
# 2503.13631), so the fiducial sits AT the chi2 minimum (chi2 ~ 0),
# where the chi2 responds quadratically to tiny theory changes and a
# chi2 comparison measures the numerics. A project whose shipped data
# is real (des_y3) has to generate such a vector at freeze time; here
# the frozen copy of the shipped vector already is one, so every
# configuration evaluates against its own frozen data_file and the
# harness gets no synthetic dataset name (nla_dataset stays None).

# High-accuracy settings for the accuracy advisory checks
# (test_accuracy.py): the same physics evaluated with the numerical
# knobs pushed far beyond the defaults (the CAMB side,
# HIGH_ACCURACY_CAMB_EXTRA_ARGS, is shared by every project and lives
# in cocoa_testing).
HIGH_ACCURACY_LIKELIHOOD = {
    # boost 3, not higher: the examples warn that the integration
    # tables of the donor project (desy1xplanck) broke down above 3,
    # and a breakdown would read as a huge "numerical error" of the
    # defaults
    "accuracyboost": 3.0,       # default 1.0
    "internal_accuracyboost": 2.0, # default 1.0 (denser convolution grid)
    # the cluster quadratures and the halo-model mass integrals read
    # 0, 1, 2 and "3 or more" (their largest tables), so 10 selects
    # the top rung of every ladder
    "integration_accuracy": 10,  # default 1
    "lmax": 200000,             # default 75000
    "kmax_boltzmann": 40.0,     # default 10.0
}

# The frozen configurations. "likelihood" is the cobaya component
# name, needed to reach that block inside the loaded info dictionary;
# "provenance" names the human-readable snapshot (never loaded).
# Both examples read the ONE joint data vector and covariance of the
# synthetic DES Y6-like data set (ss, gs, gg, cg, N, cc, cs; 2812
# entries); each combination's mask selects its blocks.
EXAMPLES = {
    # CL+GC: cluster counts N, cluster lensing, cluster clustering,
    # cluster x galaxy, and galaxy clustering of MagLim bins 1-3
    "example1": {
        "frozen_module": "frozen_config_example1.py",
        "provenance": "EXAMPLE_EVALUATE1.yaml",
        "likelihood": "des_cluster.combo_4x2pt_N",
    },
    # CL+3x2pt: the blocks of example1 plus the galaxy 3x2pt (cosmic
    # shear, galaxy-galaxy lensing, galaxy clustering of all six bins)
    "example2": {
        "frozen_module": "frozen_config_example2.py",
        "provenance": "EXAMPLE_EVALUATE2.yaml",
        "likelihood": "des_cluster.combo_6x2pt_N",
    },
}

# ---- project-independent constants ------------------------------------------

# These are identical in every project and live in the core module.
REQUIRED_OMP_THREADS = _cct.REQUIRED_OMP_THREADS
CHI2_TOLERANCE = _cct.CHI2_TOLERANCE
RACE_TOLERANCE = _cct.RACE_TOLERANCE
RACE_PERTURBATIONS = _cct.RACE_PERTURBATIONS
HIGH_ACCURACY_CAMB_EXTRA_ARGS = _cct.HIGH_ACCURACY_CAMB_EXTRA_ARGS

# ---- the harness -----------------------------------------------------------

# ONE instance binds the shared machinery to this project's data;
# everything below re-exports its surface under the historical names.
# The constructor also takes the tables of checks this project does
# not run (TATT variants, the one-knob-at-a-time scan, the
# CFASTPT-vs-FASTPT comparison); they are empty here, so a later
# project test that needs one must fill its table first.
_H = _cct.CocoaTestHarness(
    worker_file=__file__,
    interface_module="cosmolike_des_cluster_interface",
    examples=EXAMPLES,
    tatt_point={},
    accuracy_knobs=[],
    high_accuracy_likelihood=HIGH_ACCURACY_LIKELIHOOD,
    fastpt_low_settings={},
    fastpt_high_settings={},
    fastpt_points=[],
)

# ---- module functions re-exported from the core (no project state) ----------
require_cocoa_environment = _cct.require_cocoa_environment
assert_omp_threads = _cct.assert_omp_threads
sha256_of = _cct.sha256_of
make_model = _cct.make_model
evaluate_chi2 = _cct.evaluate_chi2
report_chi2_test = _cct.report_chi2_test
report_race_test = _cct.report_race_test
report_accuracy = _cct.report_accuracy

# ---- bound methods of the harness (the machinery, project-bound) ------------
compute_manifest = _H.compute_manifest
verify_frozen = _H.verify_frozen
load_reference = _H.load_reference
_frozen_module = _H._frozen_module
load_frozen_info = _H.load_frozen_info
load_frozen_point = _H.load_frozen_point
build_point = _H.build_point
_single_model_chi2_impl = _H._single_model_chi2_impl
_ten_in_a_row_impl = _H._ten_in_a_row_impl
single_model_chi2 = _H.single_model_chi2
ten_in_a_row_chi2 = _H.ten_in_a_row_chi2
# the worker subprocess loads this file by path and calls _worker by
# name (cocoa_testing._WORKER_DRIVER), so this re-export is
# load-bearing even though no test module calls it
_worker = _H._worker
_run_isolated = _H._run_isolated
