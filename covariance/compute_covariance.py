"""Compute and save the des_cluster covariance through the production interface.

The production interface is the compiled module
cosmolike_des_cluster_interface called directly on numpy arrays, the
route meant for long and HPC runs; the notebook wrappers reach the same C
kernels with extra array copies. This script evaluates the joint
6x2pt + N forecast covariance (2812 entries, see
des_cluster_joint_covariance.py) at the cosmology of a Cobaya-style YAML
file and saves its Gaussian (G), super-sample (SSC), connected
non-Gaussian (cNG) and total matrices in one .npz archive.

From the cocoa/Cocoa folder of an activated Cocoa installation:
    python projects/des_cluster/covariance/compute_covariance.py \
        projects/des_cluster/EXAMPLE_EVALUATE_COVARIANCE.yaml

OMP_NUM_THREADS must be set: it is the number of CosmoLike threads. See --help and
covariance/README.md for space and accuracy options. The numerical model
and survey settings are shared with the notebook.
"""

import os
from pathlib import Path
import sys

# Set external numerical libraries to one worker before their first import.
# CosmoLike's own OpenMP team is controlled by OMP_NUM_THREADS in the environment.
# (OpenBLAS, MKL and Apple's vecLib are the linear-algebra libraries numpy may
# use; with one thread each they do not compete with the OpenMP threads.)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

# This runner evaluates one matrix in one process. Cobaya supplies the YAML
# reader; it does not launch MPI workers or a sampler for this calculation.
os.environ["COBAYA_NOMPI"] = "1"

# project = projects/des_cluster; core = the shared CosmoLike folder, which
# holds cosmolike_notebook_utils; interface/ holds the compiled library. The
# survey adapter des_cluster_joint_covariance.py sits next to this file, a
# folder Python searches first when it runs a script.
project = Path(__file__).resolve().parents[1]
core = project.parents[1]/"external_modules/code/cosmolike_core"
sys.path.insert(0, str(core))
sys.path.insert(0, str(project/"interface"))

import cosmolike_des_cluster_interface as ci
import des_cluster_joint_covariance as survey
from cosmolike_notebook_utils.covariance.command_line import run_covariance


# Python sets __name__ to "__main__" only when this file is run as a script.
if __name__ == "__main__":
    run_covariance(
        interface=ci, survey=survey, default_space="real", joint=True,
    )
