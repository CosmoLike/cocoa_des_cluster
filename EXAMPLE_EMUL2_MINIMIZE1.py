"""Annealed minimization of the hybrid 4x2pt + N example of des_cluster.

The hybrid example EXAMPLE_EMUL2_EVALUATE1.yaml runs the likelihood
des_cluster.combo_4x2pt_N with use_emulator: 2: trained neural-network
emulators replace the Boltzmann code (CAMB) for the background expansion
and the matter power spectra, while CosmoLike still computes the survey
projections. The likelihood data are synthetic.

This file is a thin launcher: it makes the shared CosmoLike core folder
importable and calls cocoa_hybrid_sampling.run with mode "minimize" and
example number 1, so the run reads EXAMPLE_EMUL2_EVALUATE1.yaml. That run
searches for the smallest -2 log posterior with an annealed emcee
ensemble: a set of points called walkers samples the posterior raised to
the power 1/T while the temperature T is lowered stage by stage, so the
walkers gather around the best point. The docstring of
cocoa_hybrid_sampling.py (printed by --help) lists every option.

Run from the cocoa/Cocoa folder after `source start_cocoa.sh`:

    python ./projects/des_cluster/EXAMPLE_EMUL2_MINIMIZE1.py --check
    mpirun -n 2 --bind-to none python \
        ./projects/des_cluster/EXAMPLE_EMUL2_MINIMIZE1.py \
        --nstw 200 --outroot hybrid_min1

--check evaluates the fiducial point of the YAML, prints the sampled-
parameter order and the scores, and stops. The emulators assume a neutrino
mass of 0.06 eV while the synthetic data were made with CAMB at 0.077 eV,
so the chi2 of the data at the fiducial point is not zero. The second
command starts one process per MPI rank (each builds its own Cobaya model)
and writes chains/hybrid_min1.json and chains/hybrid_min1.txt (the best
point and its score) in the project folder.
"""

from pathlib import Path
import sys

# This file sits in cocoa/Cocoa/projects/des_cluster: parents[1] of that
# folder is cocoa/Cocoa, and the / operator of pathlib.Path joins path
# pieces.
project = Path(__file__).resolve().parent
core = project.parents[1]/"external_modules/code/cosmolike_core"
# sys.path is the list of folders Python searches on import; putting the
# shared core first lets the next import find cocoa_hybrid_sampling.py.
sys.path.insert(0, str(core))

from cocoa_hybrid_sampling import run


# Python sets __name__ to "__main__" only in the file named on the command
# line (each MPI rank runs it that way), so importing this file starts
# no run.
if __name__ == "__main__":
    run(mode="minimize", project=project, example=1)
