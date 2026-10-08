"""Posterior profile of the hybrid 6x2pt + N example of des_cluster.

The hybrid example EXAMPLE_EMUL2_EVALUATE2.yaml runs the likelihood
des_cluster.combo_6x2pt_N with use_emulator: 2: trained neural-network
emulators replace the Boltzmann code (CAMB) for the background expansion
and the matter power spectra, while CosmoLike still computes the survey
projections. The likelihood data are synthetic.

This file is a thin launcher: it makes the shared CosmoLike core folder
importable and calls cocoa_hybrid_sampling.run with mode "profile" and
example number 2, so the run reads EXAMPLE_EMUL2_EVALUATE2.yaml. That run
fixes one sampled parameter at each value of a grid centered on a saved
minimization result and repeats the annealed minimization over the other
parameters; the minimized -2 log posterior against the fixed value is the
profile (priors included). The docstring of cocoa_hybrid_sampling.py
(printed by --help) lists every option.

Run from the cocoa/Cocoa folder after `source start_cocoa.sh`:

    python ./projects/des_cluster/EXAMPLE_EMUL2_PROFILE2.py --check
    mpirun -n 2 --bind-to none python \
        ./projects/des_cluster/EXAMPLE_EMUL2_PROFILE2.py \
        --profile 0 --nstw 200 --numpts 11 --factor 1 \
        --minfile ./projects/des_cluster/chains/hybrid_min2.json \
        --outroot hybrid_profile2

--check evaluates the fiducial point of the YAML, prints the sampled-
parameter order and the scores, and stops. The emulators assume a neutrino
mass of 0.06 eV while the synthetic data were made with CAMB at 0.077 eV,
so the chi2 of the data at the fiducial point is not zero. The second
command starts one process per MPI rank (each builds its own Cobaya model)
and needs the record hybrid_min2.json of a previous
EXAMPLE_EMUL2_MINIMIZE2.py run, and writes chains/hybrid_profile2.json
plus one row per grid value in chains/hybrid_profile2.<parameter
name>.txt.
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
    run(mode="profile", project=project, example=2)
