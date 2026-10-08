"""Shared harness for the des_cluster unit tests: the project's data
bound to the shared Cocoa test machinery.

The harness is the code every test uses to build a Cobaya model from a
stored configuration, evaluate chi2 and compare it with a stored
reference value. The machinery itself (frozen-state verification, the
chi2 pipeline, worker-subprocess isolation, the race check, and the
terminal reports) lives in
external_modules/code/cosmolike_core/cocoa_testing.py. A worker
subprocess is a separate Python process started for one evaluation, so
the C globals of one configuration never meet another; the race check
evaluates the fiducial point, nine other cosmologies and the fiducial
point again in one process, and a change of the last chi2 reveals state
leaking between evaluations or a race between OpenMP threads.

This file carries what is des_cluster's alone: the examples table
covering BOTH data sets of the project (examples 1-2: the two cluster
combinations of arXiv 2503.13631, example1 = 4x2pt + N and
example2 = 6x2pt + N, on the synthetic DES Y6-like data set;
examples 3-4: the galaxy-only likelihoods, example3 = cosmic shear
and example4 = 3x2pt with its 2x2pt reduction, on the DES Y3
placeholder data set), the TATT point and the generated data vectors
of the galaxy-only examples, the high-accuracy settings of the
accuracy checks, and the process isolation of the galaxy tests that
build their models in process. It binds all of it to ONE
cocoa_testing.CocoaTestHarness instance whose methods are re-exported
here under the names the test modules use, so the test modules and
generate_frozen_reference.py import everything from this module.

The intrinsic-alignment model of the cluster examples is NLA: the
cluster lensing code has no TATT, so examples 1-2 carry no TATT
dataset and no TATT test. The galaxy-only likelihoods run both
models, so examples 3-4 carry both variants. has_tatt() tells the
two kinds apart.

The frozen-state rule: everything a test evaluates (configurations,
data files, reference chi2 values) lives under tests/frozen/, a
snapshot taken by generate_frozen_reference.py, so a change of the live
project cannot change what a test compares. tests/manifest_sha256.json
pins every snapshot file byte for byte (a SHA-256 digest is a
fingerprint of a file's bytes), and the digests are verified before any
model is built; refreshing the frozen state is a deliberate maintainer
action (generate_frozen_reference.py --overwrite).
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

# The cluster examples' shipped data_file is SYNTHETIC: the joint data
# vector data/des_cluster_y6.datavector is the model at the examples'
# fiducial point (scripts/make_synthetic_data.py, Table I of
# arXiv 2503.13631) without the redshift-by-redshift cold-matter halo
# variance sigma_cb(M, z) of the present model (README, "Cluster
# options"); the present model differs from it by chi2 = 0.1507
# (example1) and 0.1518 (example2) (tests/frozen/reference_chi2.json).
# The fiducial therefore sits close to the chi2 minimum, where the chi2
# responds quadratically to tiny theory changes and a chi2 comparison
# measures the numerics. Examples 1-2 therefore evaluate against their own
# frozen data_file and carry no "nla_dataset" key (the harness then keeps
# the shipped data_file).

# The galaxy-only examples' shipped data_file is REAL data (the DES-Y3
# measurement, the placeholder data set copied from the project
# des_y3), and the example cosmology is not a best fit of it: the chi2
# sits far from the minimum, where it responds LINEARLY to tiny theory
# changes. A chi2 comparison evaluated there reports alarming shifts
# that say nothing about the numerics near a fit. The NLA variants of
# examples 3-4 therefore evaluate against a SYNTHETIC data vector,
# generated with the default (NLA) model at the fiducial point during
# the freeze, and the TATT variants against a vector generated WITH
# TATT at the TATT point, for the same reason (against an NLA-based
# vector the TATT chi2 would sit away from its minimum): at its own
# minimum the chi2 response is quadratic and stable. One full-length
# vector per IA model, generated from the 3x2pt example, serves every
# galaxy-only probe (the other probes' masks select their sections).
# Every generated vector: {".dataset" filename: (source example, TATT?)}.
SYNTHETIC_VECTORS = {
    "synthetic_des_y3.dataset": ("example4", False),
    "tatt_des_y3.dataset": ("example4", True),
}

# The TATT (Tidal Alignment and Tidal Torquing, an intrinsic-alignment
# model with tidal second-order terms) tests replace these values in
# the frozen point. In the NLA reference point A2 and BTA are zero, so
# the nonzero values here make the TATT reference genuinely exercise
# the second-order terms. Only the galaxy-only examples sample these
# parameters (likelihood/params_source.yaml); the cluster examples fix
# them (params_source_y6.yaml) and are never evaluated with TATT.
TATT_POINT = {
    "DES_A2_1": 0.05,
    "DES_BTA_1": 0.05,
    "DES_A2_2": -1.51541,
}

# High-accuracy settings for the accuracy advisory checks
# (test_accuracy.py): the same physics evaluated with the numerical
# knobs pushed far beyond the defaults (the CAMB side,
# HIGH_ACCURACY_CAMB_EXTRA_ARGS, is shared by every project and lives
# in cocoa_testing).
HIGH_ACCURACY_LIKELIHOOD = {
    # boost 3, not higher: the examples warn that the integration
    # tables of the project this one was created from (desy1xplanck)
    # broke down above 3, and a breakdown would read as a huge
    # "numerical error" of the defaults
    "accuracyboost": 3.0,       # default 1.0
    "internal_accuracyboost": 2.0, # default 1.0 (denser convolution grid)
    # the cluster quadratures and the halo-model mass integrals read
    # 0, 1, 2 and "3 or more" (their largest tables), so 10 selects
    # the top rung of every ladder
    "integration_accuracy": 10,  # default 1 (cluster), 0 (galaxy-only)
    "lmax": 200000,             # default 75000
    "kmax_boltzmann": 40.0,     # default 10.0 (cosmic shear: 7.5)
}

# The frozen configurations. "likelihood" is the cobaya component
# name, needed to reach that block inside the loaded info dictionary;
# "provenance" names the human-readable snapshot (never loaded).
# Examples 1-2 read the ONE joint data vector and covariance of the
# synthetic DES Y6-like data set (ss, gs, gg, cg, N, cc, cs; 2812
# entries); each combination's mask selects its blocks. Examples 3-4
# (and example4's 2x2pt reduction) read the DES-Y3 placeholder data
# set (ss, gs, gg; 900 entries) through the generated vectors above:
# "nla_dataset" and "tatt_dataset" name the descriptor each IA model
# evaluates against. An entry WITHOUT "tatt_dataset" has no TATT
# variant (has_tatt below).
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
    # cosmic shear alone (xi+ and xi-)
    "example3": {
        "frozen_module": "frozen_config_example3.py",
        "provenance": "EXAMPLE_EVALUATE3.yaml",
        "likelihood": "des_cluster.cosmic_shear",
        "tatt_dataset": "tatt_des_y3.dataset",
        "nla_dataset": "synthetic_des_y3.dataset",
    },
    # 3x2pt: cosmic shear, galaxy-galaxy lensing, galaxy clustering
    "example4": {
        "frozen_module": "frozen_config_example4.py",
        "provenance": "EXAMPLE_EVALUATE4.yaml",
        "likelihood": "des_cluster.combo_3x2pt",
        "tatt_dataset": "tatt_des_y3.dataset",
        "nla_dataset": "synthetic_des_y3.dataset",
    },
    # example4 with the likelihood renamed: the same options and data,
    # with cosmolike selecting galaxy clustering plus galaxy-galaxy
    # lensing only
    "example4_2x2pt": {
        "frozen_module": "frozen_config_example4_2x2pt.py",
        "provenance": "EXAMPLE_EVALUATE4.yaml",
        "source_likelihood": "des_cluster.combo_3x2pt",
        "likelihood": "des_cluster.combo_2x2pt",
        "tatt_dataset": "tatt_des_y3.dataset",
        "nla_dataset": "synthetic_des_y3.dataset",
    },
}


def has_tatt(example):
    """Tell whether one example carries a TATT variant.

    The harness reads the TATT data vector of a configuration from
    its "tatt_dataset" key, so that key IS the statement "this
    example runs TATT": the galaxy-only examples carry it, the
    cluster ones do not (the cluster lensing code has no TATT). The
    generator asks before it evaluates a TATT reference; the test
    modules of the cluster examples simply never request the variant.

    Arguments:
      example = a key of EXAMPLES.

    Returns:
      True when the example has a TATT variant.

    Raises:
      KeyError when example names no EXAMPLES entry.
    """
    return "tatt_dataset" in EXAMPLES[example]


# ---- project-independent constants ------------------------------------------

# These are identical in every project and live in the core module, where
# each carries the reason for its value: REQUIRED_OMP_THREADS = "4", the
# thread count of the race tests and workers (one thread could not show a
# race); CHI2_TOLERANCE = 0.2, the largest |chi2(now) - chi2(reference)| a
# reference test accepts; RACE_TOLERANCE = 1e-4, the float noise allowed
# between two evaluations of the same point; RACE_PERTURBATIONS, the nine
# cosmologies of the race test; HIGH_ACCURACY_CAMB_EXTRA_ARGS, the CAMB
# side of the accuracy checks.
REQUIRED_OMP_THREADS = _cct.REQUIRED_OMP_THREADS
CHI2_TOLERANCE = _cct.CHI2_TOLERANCE
RACE_TOLERANCE = _cct.RACE_TOLERANCE
RACE_PERTURBATIONS = _cct.RACE_PERTURBATIONS
HIGH_ACCURACY_CAMB_EXTRA_ARGS = _cct.HIGH_ACCURACY_CAMB_EXTRA_ARGS

# ---- the harness -----------------------------------------------------------

# ONE instance binds the shared machinery to this project's data;
# everything below re-exports its functions under the names the test
# modules import.
# The constructor also takes the tables of checks this project does
# not run (the one-knob-at-a-time scan, the CFASTPT-vs-FASTPT
# comparison); they are empty here, so a later project test that
# needs one must fill its table first. No project-wide nla_dataset:
# the galaxy-only entries name their own, and the cluster entries
# must keep their shipped data_file.
_H = _cct.CocoaTestHarness(
    worker_file=__file__,
    interface_module="cosmolike_des_cluster_interface",
    examples=EXAMPLES,
    tatt_point=TATT_POINT,
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
_load_datavector = _cct._load_datavector
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

# ---- one data set per process -----------------------------------------------

# The cosmolike C layer keeps the data-vector dimensions in C globals
# and aborts a process that initializes a second configuration with
# different dimensions (cocoa_testing, WORKER ISOLATION). This suite
# holds two data sets - the cluster examples (the 2812-entry joint
# vector, six lens bins) and the galaxy-only examples (900 entries,
# five lens bins) - and pytest imports every test module into ONE
# process. The tests that go through single_model_chi2 and
# ten_in_a_row_chi2 are covered by the harness's workers. The tests
# that must build their models IN PROCESS are not: they read the full
# data vector from the compiled interface, or need several model
# builds to share the C caches (the cache ladders, the flag sweeps).
# The cluster ladder (test_cache_consistency.py) keeps the pytest
# process; every galaxy-only in-process test gets a process of its
# own through the decorator below, so the two data sets never meet.
# The variable marks the process that runs such a test's body.
OWN_PROCESS_FLAG = "COCOA_TESTS_OWN_PROCESS"


def own_process(test_file):
    """Decorator: run one test method in a python process of its own.

    Used as @own_process(__file__) on a unittest test method. In the
    process pytest (or unittest discover) collected the test in, the
    decorated method does not run its body: it starts a fresh python
    on the test's own file, naming that one test on the command line
    (unittest.main runs exactly the test it is given), and passes
    when the child exits with code 0. The child carries
    OWN_PROCESS_FLAG, so there the decorated method runs the body in
    process, as written. The child's report streams to the same
    terminal, so a failure prints its assertion there; the parent
    only relays the verdict.

    Arguments:
      test_file = the path of the test module (__file__ in that
                  module), the file the child process runs.

    Returns:
      the decorator to apply to the test method.

    Raises:
      nothing at decoration time; the decorated method fails with an
      AssertionError naming the test when its process exits nonzero.
    """
    # functools.wraps copies the method's name and docstring onto the
    # wrapper, so unittest and pytest still collect and report the
    # test under its own name
    import functools

    def decorator(test_method):
        """Return the replacement of test_method that runs it in a child process.

        Arguments:
          test_method = the unittest test method being decorated.

        Returns:
          wrapper, which keeps test_method's name and docstring.
        """
        @functools.wraps(test_method)
        def wrapper(self):
            """Run the test body here in the child, else start the child and check it.

            functools.wraps replaces this docstring by the test method's.

            Arguments:
              self = the unittest.TestCase instance of the test.

            Returns:
              the body's result in the child process; nothing in the parent.

            Raises:
              AssertionError in the parent when the child exits nonzero.
            """
            # .get returns None when the variable is absent: only the
            # child process, which carries the flag, runs the body
            if os.environ.get(OWN_PROCESS_FLAG) == "1":
                return test_method(self)
            import subprocess

            # "Class.method", the form unittest.main accepts as a
            # command-line test name inside the file it runs
            test_id = f"{type(self).__name__}.{self._testMethodName}"
            # dict(os.environ) is a COPY of the environment: the
            # edits below reach only the child process
            environment = dict(os.environ)
            environment[OWN_PROCESS_FLAG] = "1"
            # OpenMP reads this at library load inside the fresh
            # process (the test file sets it at its first line too)
            environment["OMP_NUM_THREADS"] = REQUIRED_OMP_THREADS
            print(f"  running {test_id} in its own process ...",
                  flush=True)
            # subprocess.run starts the child (sys.executable = this
            # same python) and BLOCKS until it exits
            completed = subprocess.run(
                [sys.executable, os.path.abspath(test_file), test_id],
                env=environment)
            self.assertEqual(
                completed.returncode, 0,
                f"{test_id} failed in its own process (exit code "
                f"{completed.returncode}); its report is printed above")
        return wrapper
    return decorator
