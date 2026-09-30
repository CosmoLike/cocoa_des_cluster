"""Shared harness for the des_cluster unit tests: the project's data
bound to the shared Cocoa test machinery.

The machinery itself (frozen-state verification, the chi2 pipeline,
worker-subprocess isolation, and the terminal reports) lives in
external_modules/code/cosmolike_core/cocoa_testing.py. This file
carries what is des_cluster's alone - the examples table (cosmic
shear and the 3x2pt/2x2pt combinations), the synthetic NLA vector
every configuration evaluates (NLA_DATASET; the shipped data_file is
real data, far from the fiducial's minimum), and the high-accuracy
settings of the accuracy checks - and binds it to ONE
cocoa_testing.CocoaTestHarness instance whose methods are
re-exported under the historical names, so test_accuracy.py and
generate_frozen_reference.py import everything from this module.

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

# This project's shipped data_file is REAL data, and the example
# cosmology is not its best fit: the chi2 sits far from the minimum
# (hundreds to thousands), where it responds LINEARLY to tiny theory
# changes. A chi2 comparison evaluated there reports alarming
# shifts that say nothing about the numerics near a fit. Every
# configuration therefore evaluates against a SYNTHETIC data vector,
# generated with the default (NLA) model at the fiducial point during
# the freeze: at its own minimum the chi2 response is quadratic and
# stable.
NLA_DATASET = "synthetic_des_cluster.dataset"

# Every generated vector: {".dataset" filename: source example}. The
# examples share one data set, so one full-length vector generated
# from the example2 (3x2pt) model serves every configuration (the
# other probes' masks select their sections).
SYNTHETIC_VECTORS = {
    NLA_DATASET: "example2",
}

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
    "integration_accuracy": 10,  # default 0
    "lmax": 200000,             # default 75000
    "kmax_boltzmann": 40.0,     # default 7.5
}

# The frozen configurations. "likelihood" is the cobaya component
# name, needed to reach that block inside the loaded info dictionary;
# "provenance" names the human-readable snapshot (never loaded).
EXAMPLES = {
    "example1": {
        "frozen_module": "frozen_config_example1.py",
        "provenance": "EXAMPLE_EVALUATE1.yaml",
        "likelihood": "des_cluster.cosmic_shear",
    },
    "example2": {
        "frozen_module": "frozen_config_example2.py",
        "provenance": "EXAMPLE_EVALUATE2.yaml",
        "likelihood": "des_cluster.combo_3x2pt",
    },
    # example2 with the likelihood renamed: the same options and data,
    # with cosmolike selecting galaxy clustering plus galaxy-galaxy
    # lensing only
    "example2_2x2pt": {
        "frozen_module": "frozen_config_example2_2x2pt.py",
        "provenance": "EXAMPLE_EVALUATE2.yaml",
        "source_likelihood": "des_cluster.combo_3x2pt",
        "likelihood": "des_cluster.combo_2x2pt",
    },
}

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
    nla_dataset=NLA_DATASET,
)

# ---- module functions re-exported from the core (no project state) ----------
require_cocoa_environment = _cct.require_cocoa_environment
make_model = _cct.make_model
evaluate_chi2 = _cct.evaluate_chi2
report_accuracy = _cct.report_accuracy

# ---- bound methods of the harness (the machinery, project-bound) ------------
compute_manifest = _H.compute_manifest
verify_frozen = _H.verify_frozen
load_reference = _H.load_reference
_frozen_module = _H._frozen_module
load_frozen_info = _H.load_frozen_info
build_point = _H.build_point
single_model_chi2 = _H.single_model_chi2
# the worker subprocess loads this file by path and calls _worker by
# name (cocoa_testing._WORKER_DRIVER), so this re-export is
# load-bearing even though no test module calls it
_worker = _H._worker
