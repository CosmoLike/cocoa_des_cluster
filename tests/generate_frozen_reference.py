"""Maintainer tool: (re)create the frozen state the unit tests run on.

Running this REDEFINES what the tests protect, so it refuses to run
without the explicit --overwrite flag. Only run it when a change to
the data vectors, n(z), covariance, examples, or likelihood defaults
is deliberate, and review the printed chi2 values before committing:
they become the new references the tests compare against.

What one run produces, all under tests/ (see cocoa_test_utils for how
the tests consume each piece):

  - frozen/data/: a copy of the files of the CURRENT ../data folder
    the examples read: the synthetic DES Y6-like data set of the
    cluster examples (dataset descriptors, joint data vector,
    covariance, masks, n(z)) and the DES-Y3 placeholder data set of
    the galaxy-only examples (descriptor, data vector, covariance,
    baseline mask, n(z)); DATA_IGNORE below lists what stays out.
    Plus the generated NLA and TATT data vectors of the galaxy-only
    examples and their dataset descriptors
    (cocoa_test_utils.SYNTHETIC_VECTORS).
  - frozen/EXAMPLE_EVALUATE{1,2,3,4}.yaml: snapshots of the current
    examples, kept for humans to diff (the tests never load them).
  - frozen/frozen_config_<example>.py: for each configuration of
    cocoa_test_utils.EXAMPLES, the model is built from the CURRENT
    example yaml, cobaya resolves it against the CURRENT likelihood
    defaults, and the complete resolved configuration is written back
    out as a yaml string, together with the exact evaluation point.
    Writing out every resolved option and parameter is what makes the
    tests independent of later edits to the live files.
  - frozen/reference_chi2.json: the reference chi2 values, computed
    FROM the frozen modules just written, exactly the way the tests
    will compute them: one NLA value per configuration (example1 =
    4x2pt + N, example2 = 6x2pt + N, example3 = cosmic shear,
    example4 = 3x2pt, example4_2x2pt = 2x2pt) and one TATT value per
    galaxy-only configuration (the cluster lensing code has no
    TATT). Every one sits near zero: the cluster examples' shipped
    data vector is the model at their fiducial point, and the
    galaxy-only examples evaluate against the vectors generated
    here.
  - manifest_sha256.json: the SHA-256 pin of every frozen file.

Usage (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python ./projects/des_cluster/tests/generate_frozen_reference.py --overwrite
"""

import json
import os
import shutil
import sys
import time

# The references must be produced under the same OpenMP thread count
# the tests enforce, and before any cobaya/cosmolike import.
os.environ["OMP_NUM_THREADS"] = "4"

# tests/ is not a package; insert(0, ...) puts it FIRST on the import
# search path so cocoa_test_utils resolves from anywhere
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cocoa_test_utils as u

PROJECT_DIR = os.path.dirname(u.TESTS_DIR)

# Stored inside the frozen configurations as the likelihood data path.
# ROOTDIR-relative on purpose: the frozen modules stay portable across
# machines, and the tests overwrite it with the absolute path anyway.
FROZEN_DATA_RELPATH = "./projects/des_cluster/tests/frozen/data"

# Files of ../data the freeze leaves out: no example reads them, so
# copying them would only grow the working tree and the manifest (the
# baryon simulation table alone is 10 MB). shutil.ignore_patterns
# matches each entry against the file NAME with shell wildcards. The
# list:
#   .DS_Store                macOS Finder metadata
#   README.md                the description of ../data, not data
#   ones.mask                the all-ones 900-entry mask of the DES-Y3
#                            placeholder data set: the galaxy-only
#                            examples run under 3x2pt_baseline.mask
#   pca.txt,                 the baryon PCA inputs: both descriptors
#   baryons_logPkR.h5        name them, but the likelihood resolves
#                            the two keys only with a baryon option
#                            switched on (it refuses them with
#                            clusters), and no example switches one on
#   y3_redmapper_counts.txt  the observed DES-Y3 counts, an input of
#                            the covariance script, not of the model
# A new file in ../data is frozen unless it is added here. The two
# covariances (des_cluster_y6_cov.npy, des_y3_cov_unblinded_final.txt) are
# stored with Git LFS; ../.gitattributes matches them by file name, so
# their frozen copies are LFS files too.
DATA_IGNORE = (
    ".DS_Store",
    "README.md",
    "ones.mask",
    "pca.txt",
    "baryons_logPkR.h5",
    "y3_redmapper_counts.txt",
)

# The skeleton of one generated frozen-configuration module. The yaml
# goes inside r""" so latex backslashes in parameter labels survive
# (r = raw string: backslashes are kept as typed). The backslash right
# after ''' eats the first newline, so the generated file starts at
# its docstring rather than a blank line. {point} and {yaml} are
# .format placeholders filled in freeze_example.
MODULE_TEMPLATE = '''\
"""AUTO-GENERATED by generate_frozen_reference.py --overwrite ({stamp}).

Frozen, fully expanded cobaya configuration for {example}
(source: {source} resolved against the likelihood defaults of that day).
Do not edit by hand: the SHA-256 manifest pins this file, and any edit
makes every test fail. Regenerate deliberately instead.
"""

# The exact sampled-parameter point the reference chi2 was evaluated at
point = {point}

# Every option and every parameter written out explicitly: changing the
# live EXAMPLE_EVALUATE yaml files or the likelihood default yaml files
# does not affect this configuration
yaml_string = r"""
{yaml}
"""
'''


def freeze_example(example, stamp):
    """Write one example's fully expanded frozen-configuration module.

    The expansion works by round trip through cobaya: build the model
    from the live example yaml (letting cobaya merge in the live
    likelihood defaults), then ask the model for its resolved
    configuration with model.info() and store THAT. The resolved form
    lists every option and every parameter explicitly, so the frozen
    module no longer depends on any live default.

    Arguments:
      example = a key of u.EXAMPLES.
      stamp   = the UTC time string written into the module header.

    Returns:
      nothing; frozen/frozen_config_<example>.py is written.

    Raises:
      ValueError when a sampled parameter is neither overridden by the
      example's evaluate block nor given a ref center to fall back on
      (the point would be undefined at that parameter).
    """
    from cobaya.yaml import yaml_load_file, yaml_dump

    cfg = u.EXAMPLES[example]
    live = yaml_load_file(os.path.join(PROJECT_DIR, cfg["provenance"]))
    # The evaluate sampler's override block is the example's fiducial
    # point; keep it before stripping the sampler from the info.
    # dict(...) COPIES the block, so the pop below cannot take it
    # along; pop(key, None) removes a key without erroring when it
    # is already absent.
    override = dict(live["sampler"]["evaluate"]["override"])
    live.pop("sampler", None)
    live.pop("output", None)
    live["debug"] = 30
    live["timing"] = False
    # the 2x2pt configuration reuses example4's yaml with the
    # likelihood renamed (combo_3x2pt -> combo_2x2pt: same options,
    # same data, different probe selection inside cosmolike)
    # .get falls back to its second argument when the key is
    # absent, so most entries rename nothing
    source_name = cfg.get("source_likelihood", cfg["likelihood"])
    # pop removes the entry AND hands back its value: the block
    # leaves under its old name and is reinserted under the new one
    likelihood_block = live["likelihood"].pop(source_name)
    live["likelihood"][cfg["likelihood"]] = likelihood_block
    likelihood_block["path"] = FROZEN_DATA_RELPATH
    # every configuration is frozen with NLA, so it cannot inherit
    # another value from an edited example; the TATT variants of the
    # galaxy-only examples switch the model when the tests load the
    # configuration (load_frozen_info), and the cluster lensing code
    # has no other model
    likelihood_block["IA_model"] = 0

    # make_model builds the evaluable cobaya Model; building it is
    # what forces cobaya to merge the live likelihood defaults into
    # the configuration
    model = u.make_model(live)
    # model.info() returns the configuration with every default
    # resolved: the complete likelihood option set and the complete
    # parameter list, exactly what must be frozen.
    updated = model.info()
    updated.pop("output", None)
    updated.pop("packages_path", None)
    updated["likelihood"][cfg["likelihood"]]["path"] = FROZEN_DATA_RELPATH

    # The frozen point: the example's override where given, the ref
    # center otherwise. Fallbacks are printed so the maintainer can
    # confirm each one is intended.
    point = {}
    fallback = []
    for name in model.parameterization.sampled_params():
        if name in override:
            point[name] = override[name]
        else:
            ref = updated["params"][name].get("ref")
            # a dict-valued ref describes a distribution; its center
            # "loc" is the fallback. The ternary picks loc for a
            # dict and the plain number otherwise.
            point[name] = ref["loc"] if isinstance(ref, dict) else ref
            fallback.append(name)
            if point[name] is None:
                raise ValueError(
                    f"parameter {name}: not in override and no ref")
    # a list is false when empty: the note prints only when at
    # least one parameter fell back to its ref center
    if fallback:
        print(f"{example}: point uses ref centers for {fallback}", flush=True)

    with open(os.path.join(u.FROZEN_DIR, cfg["frozen_module"]), "w") as f:
        # .format fills each {name} placeholder in the template with
        # the keyword argument of the same name. json.dumps writes
        # the point as JSON text, which doubles as a python literal
        # here; sort_keys makes regenerated modules diff cleanly.
        # yaml_dump returns the configuration as one yaml string and
        # rstrip trims its trailing newlines before the closing """.
        f.write(MODULE_TEMPLATE.format(
            stamp=stamp,
            example=example,
            source=cfg["provenance"],
            point=json.dumps(point, indent=4, sort_keys=True),
            yaml=yaml_dump(updated).rstrip("\n"),
        ))
    print(f"{example}: frozen config expanded to "
          f"{len(updated['params'])} parameters "
          f"({len(live['params'])} in the live example yaml)", flush=True)


def generate_datavector(dataset_name):
    """Write one generated data vector and its dataset descriptor.

    u.SYNTHETIC_VECTORS names the source example and the IA model:
    the NLA vector puts the fiducial point at the chi2 minimum for
    the NLA tests of the galaxy-only examples (their shipped
    data_file is real data), the TATT vector does the same for the
    TATT tests. The vector is evaluated with datavector printing
    enabled against the ORIGINAL frozen dataset (the descriptor
    written here does not exist yet; the printed theory vector does
    not depend on which data vector it is compared against). Runs
    inside a --vector-one worker subprocess: it builds a model, and
    two different-dimension builds in one process abort (see
    cocoa_testing).

    Arguments:
      dataset_name = a key of u.SYNTHETIC_VECTORS, which is also the
                     file name of the descriptor to write.

    Returns:
      nothing; frozen/data/ gains the .modelvector and .dataset files.

    Raises:
      RuntimeError when the generated vector's length differs from
      the original data vector (a masking or probe mismatch), or when
      the dataset descriptor does not contain exactly one data_file
      line to replace.
    """
    from cobaya.yaml import yaml_load

    # the table's value is an (example, TATT?) pair; the assignment
    # unpacks it into the two names
    example, use_tatt = u.SYNTHETIC_VECTORS[dataset_name]
    cfg = u.EXAMPLES[example]
    # the original dataset name comes from the frozen configuration
    # itself (load_frozen_info already points data_file at the
    # generated descriptor, which does not exist yet)
    frozen_info = yaml_load(u._frozen_module(example).yaml_string)
    original_dataset = frozen_info["likelihood"][cfg["likelihood"]]["data_file"]

    # str.replace swaps the extension, pairing the vector's file
    # name with the descriptor's
    vector_name = dataset_name.replace(".dataset", ".modelvector")
    info = u.load_frozen_info(example, tatt=use_tatt)
    likelihood_block = info["likelihood"][cfg["likelihood"]]
    likelihood_block["data_file"] = original_dataset
    likelihood_block["print_datavector"] = True
    likelihood_block["print_datavector_file"] = (
        FROZEN_DATA_RELPATH + "/" + vector_name)

    # ternary: "TATT" when use_tatt is True, "NLA" otherwise
    ia_label = "TATT" if use_tatt else "NLA"
    print(f"generating {vector_name} ({example}, {ia_label} point) ...",
          flush=True)
    model = u.make_model(info)
    point = u.build_point(model, example, tatt=use_tatt)
    u.evaluate_chi2(model, point)

    # sanity: the generated vector must have the same length as the
    # original one, or the masks would select the wrong entries
    data_dir = os.path.join(u.FROZEN_DIR, "data")
    with open(os.path.join(data_dir, vector_name)) as f:
        # sum(1 for _ in f) walks the file line by line and adds 1
        # per line: a line count that never loads the whole file
        # into memory (the with closes the file either way)
        generated_lines = sum(1 for _ in f)
    descriptor_path = os.path.join(data_dir, original_dataset)
    with open(descriptor_path) as f:
        # .read() with no size returns the whole file as one string
        descriptor = f.read()
    original_vector = None
    # splitlines() cuts the text into a list of lines, newlines
    # removed
    for line in descriptor.splitlines():
        if line.strip().startswith("data_file"):
            # split("=", 1) cuts at the FIRST "=" only; [1] is the
            # part after the cut, and strip() drops the blanks
            # around it
            original_vector = line.split("=", 1)[1].strip()
    with open(os.path.join(data_dir, original_vector)) as f:
        # the same load-nothing line count as above
        original_lines = sum(1 for _ in f)
    if generated_lines != original_lines:
        raise RuntimeError(
            f"generated data vector has {generated_lines} lines; the "
            f"original {original_vector} has {original_lines}")

    # the generated dataset descriptor: the original with only the
    # data_file line replaced
    replaced = 0
    out_lines = []
    # keepends=True leaves the newline on the end of every line, so
    # joining the pieces rebuilds the file byte for byte and only
    # the retyped line differs
    for line in descriptor.splitlines(keepends=True):
        if line.strip().startswith("data_file"):
            out_lines.append(f"data_file = {vector_name}\n")
            replaced += 1
        else:
            out_lines.append(line)
    if replaced != 1:
        raise RuntimeError(
            f"{original_dataset}: expected exactly one data_file "
            f"line, found {replaced}")
    with open(os.path.join(data_dir, dataset_name), "w") as f:
        # "".join(out_lines) glues the list into one string; each
        # piece still ends in its own newline
        f.write("".join(out_lines))
    print(f"generated data vector: {vector_name} ({generated_lines} lines); "
          f"descriptor: {dataset_name}", flush=True)


def main():
    """Rebuild tests/frozen/ and the manifest from the current project.

    Worker modes come first: --freeze-one X and --vector-one D each
    run a single model-building step and exit. The parent spawns one
    subprocess per step: a process that initializes configurations
    with different data-set dimensions aborts inside cosmolike (see
    cocoa_testing), and this project's examples do use two data sets
    (the cluster one and the DES-Y3 placeholder).

    The parent's steps, in order: refuse without --overwrite; delete
    and recreate frozen/; copy the data and the example snapshots;
    write the expanded frozen-configuration modules; generate the NLA
    and TATT data vectors of the galaxy-only examples; evaluate the
    reference chi2 values from those modules (the same code path the
    tests use); write the reference file; hash everything into the
    manifest. The manifest comes last so it covers every file the
    earlier steps produced. The cluster examples need no generated
    vector: their shipped one already is the model at the fiducial
    point (see cocoa_test_utils).

    Returns:
      0 on success, 1 when --overwrite was not given (the usage text
      and the refusal reason are printed).
    """
    # `in` scans the argument list for the flag; .index returns the
    # position of its first occurrence, so [index + 1] is the value
    # that follows the flag
    if "--freeze-one" in sys.argv:
        u.require_cocoa_environment()
        example = sys.argv[sys.argv.index("--freeze-one") + 1]
        stamp = sys.argv[sys.argv.index("--stamp") + 1]
        freeze_example(example, stamp)
        return 0
    if "--vector-one" in sys.argv:
        u.require_cocoa_environment()
        dataset_name = sys.argv[sys.argv.index("--vector-one") + 1]
        generate_datavector(dataset_name)
        return 0
    if "--overwrite" not in sys.argv:
        print(__doc__)
        print("Refusing to run without --overwrite (this redefines the "
              "frozen state every test compares against).")
        return 1
    u.require_cocoa_environment()
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    if os.path.isdir(u.FROZEN_DIR):
        shutil.rmtree(u.FROZEN_DIR)
    os.makedirs(u.FROZEN_DIR)

    print("freezing ../data ...", flush=True)
    # the data copy is what lets users change ../data later without
    # touching the tests; DATA_IGNORE names the files no example
    # reads. ignore_patterns builds the filter function copytree
    # calls in every folder; matching names are skipped (the *
    # spreads the tuple into separate arguments).
    shutil.copytree(os.path.join(PROJECT_DIR, "data"),
                    os.path.join(u.FROZEN_DIR, "data"),
                    ignore=shutil.ignore_patterns(*DATA_IGNORE))
    # example4 and its 2x2pt reduction share one provenance file; the
    # second copy rewrites the same bytes
    for cfg in u.EXAMPLES.values():
        shutil.copy2(os.path.join(PROJECT_DIR, cfg["provenance"]),
                     os.path.join(u.FROZEN_DIR, cfg["provenance"]))

    # one worker subprocess per model-building step (see the docstring)
    import subprocess
    # __file__ is this script's own path: each worker re-runs this
    # very file with a mode flag
    self_path = os.path.abspath(__file__)
    # iterating a dict yields its KEYS: each example name in turn.
    # subprocess.run starts the worker (sys.executable = this same
    # python) and waits for it to finish
    for example in u.EXAMPLES:
        completed = subprocess.run(
            [sys.executable, self_path, "--freeze-one", example,
             "--stamp", stamp])
        if completed.returncode != 0:
            raise RuntimeError(f"freeze worker for {example} failed")
    # the generated data vectors must exist before the reference loop
    # below: every galaxy-only reference evaluates against them
    for dataset_name in u.SYNTHETIC_VECTORS:
        completed = subprocess.run(
            [sys.executable, self_path, "--vector-one", dataset_name])
        if completed.returncode != 0:
            raise RuntimeError(f"vector worker for {dataset_name} failed")

    reference = {
        "_meta": {
            "generated_utc": stamp,
            "omp_num_threads": os.environ["OMP_NUM_THREADS"],
            "chi2_tolerance": u.CHI2_TOLERANCE,
        }
    }
    for example in u.EXAMPLES:
        for tatt in (False, True):
            # the cluster examples have no TATT variant (the cluster
            # lensing code has no TATT): they get the NLA reference
            # alone
            if tatt and not u.has_tatt(example):
                continue
            # the ternary inside the f-string picks the key suffix:
            # "tatt" when tatt is True, "nla" otherwise; the suffix
            # names the IA model the reference was evaluated with,
            # the key the test modules read
            key = f"{example}_{'tatt' if tatt else 'nla'}"
            t0 = time.time()
            chi2 = u.single_model_chi2(example, tatt)
            # :.6e = scientific notation with six decimals (the
            # references sit near zero, where fixed decimals would
            # print 0.000000); :.1f = one decimal for the elapsed
            # seconds
            print(f"{key}: chi2 = {chi2:.6e}  ({time.time() - t0:.1f}s)",
                  flush=True)
            reference[key] = chi2

    with open(u.REFERENCE_FILE, "w") as f:
        # json.dump writes the table as JSON text into the open
        # file; indent and sort_keys keep the result stable and
        # diffable, and the extra write adds the final newline text
        # files end with (load_reference round-trips it back)
        json.dump(reference, f, indent=2, sort_keys=True)
        f.write("\n")

    # compute_manifest returns {relative path: sha256} for every file
    # now under frozen/; writing it LAST means it covers every file
    # the steps above produced
    manifest = {
        "_comment": "SHA-256 of every file under tests/frozen/; verified by "
                    "every test before evaluating anything.",
        "files": u.compute_manifest(),
    }
    with open(u.MANIFEST_FILE, "w") as f:
        # the same stable JSON write as the reference file above
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"manifest: {len(manifest['files'])} files pinned")
    return 0


# __name__ is "__main__" only when this file runs directly as a
# script; sys.exit(main()) hands main's return value to the shell
# as the exit code (0 = success)
if __name__ == "__main__":
    sys.exit(main())
