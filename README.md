# The des_cluster project <a name="des_cluster_overview"></a>

This project runs the DES Y6-style joint analysis of galaxy clusters, galaxy clustering, and weak lensing of [arXiv:2503.13631](https://arxiv.org/abs/2503.13631) in Cocoa. The cluster code is a port of the original CosmoLike cluster code ([arXiv:2008.10757](https://arxiv.org/abs/2008.10757)) into Cocoa's `cosmolike_core`, where it lives in the files ending in `_cluster`. The file [PORT_PLAN.md](PORT_PLAN.md) records the model, the design decisions, the validation, and the status of the port.

The project provides two likelihoods:

| likelihood | name in the paper | blocks |
|---|---|---|
| `des_cluster.combo_4x2pt_N` | CL+GC = 4x2pt + N | cluster counts, cluster lensing, cluster clustering, cluster x galaxy clustering, galaxy clustering (MagLim lens bins 1-3) |
| `des_cluster.combo_6x2pt_N` | CL+3x2pt = 6x2pt + N | the blocks above plus cosmic shear, galaxy-galaxy lensing, and galaxy clustering of all six MagLim lens bins |

Both likelihoods read one joint data vector of 2812 entries and one covariance; they differ only in the scale-cut mask. The blocks are stored in the order `ss`, `gs`, `gg`, `cg`, `N`, `cc`, `cs`:

| block | content | layout | entries | kept by the 4x2pt + N mask | kept by the 6x2pt + N mask |
|---|---|---|---|---|---|
| `ss` | cosmic shear, $\xi_+$ then $\xi_-$ | [source pair][$\theta$] | 400 | 0 | 190 |
| `gs` | galaxy-galaxy lensing $\gamma_t$ | [(lens, source) pair][$\theta$] | 480 | 0 | 312 |
| `gg` | galaxy clustering $w_{gg}$ | [lens bin][$\theta$] | 120 | 31 | 69 |
| `cg` | cluster x galaxy clustering $w_{cg}$ | [(cluster z bin, lens bin) pair][richness bin][$\theta$] | 240 | 128 | 128 |
| `N` | cluster counts | [cluster z bin][richness bin] | 12 | 12 | 12 |
| `cc` | cluster clustering $w_{cc}$ | [cluster z bin][richness pair][$\theta$] | 600 | 230 | 230 |
| `cs` | cluster lensing, $\Sigma = Y \gamma_t$ | [(cluster z bin, source bin) pair][richness bin][$\theta$] | 960 | 488 | 488 |
| total | | | 2812 | 889 | 1429 |

The binning is the one of the paper: three cluster redshift bins with $z_\lambda$ edges [0.2, 0.4, 0.55, 0.65], four richness bins with $\lambda$ edges [20, 30, 45, 60, 500], six MagLim lens bins, four source bins, and 20 logarithmic angular bins between 2.5 and 250 arcmin.

> [!Warning]
> **All data in this project are synthetic.** No public DES cluster data vectors or covariances exist (neither for Y1 nor for Y3), so the project ships a DES Y6-like stand-in: the data vector is the model itself, computed at the fiducial point of Table I of arXiv:2503.13631, and the covariance is an analytic Gaussian covariance. Nothing here is a DES measurement, and no chain run on these files is a DES result. The section [Synthetic data](#des_cluster_data) describes the files and how to regenerate them.

> [!Warning]
> The code and examples of this project are in alpha stage. Read [Status and known limitations](#des_cluster_status) before using the results.

## Running Cosmolike projects (Basic instructions) <a name="des_cluster_running_cosmolike_projects"></a>

From `Cocoa/Readme` instructions:

> [!Note]
> We provide several cosmolike projects that can be loaded and compiled using `setup_cocoa.sh` and `compile_cocoa.sh` scripts. Cocoa skips the project des_cluster by default. To activate it, comment the line `export IGNORE_COSMOLIKE_DES_CLUSTER_CODE=1` on `set_installation_options.sh`
>
>     [Adapted from Cocoa/set_installation_options.sh shell script]
>     (...)
>
>     # ------------------------------------------------------------------------------
>     # The keys below control which cosmolike projects will be installed and compiled
>     # ------------------------------------------------------------------------------
>     #export IGNORE_COSMOLIKE_LSST_Y1_CODE=1
>     #export IGNORE_COSMOLIKE_DES_Y3_CODE=1
>     (...)
>     # The two projects below are skipped by default: comment the key to
>     # download and compile the project.
>     #export IGNORE_COSMOLIKE_DES_CLUSTER_CODE=1
>
>     (...)
>
>     # ------------------------------------------------------------------------------
>     # Cosmolike projects below -------------------------------------------
>     # ------------------------------------------------------------------------------
>     (...)
>     export DES_CLUSTER_URL="https://github.com/CosmoLike/cocoa_des_cluster.git"
>     export DES_CLUSTER_NAME="des_cluster"
>     export DES_CLUSTER_GIT_BRANCH="main" # no tagged release yet

> [!NOTE]
> The covariance `data/des_cluster_y6_cov.npy` (32 MB) is stored with Git LFS. A clone without `git lfs pull` holds a small pointer file in its place, and the likelihoods cannot load it.

> [!NOTE]
> If users want to recompile cosmolike, there is no need to rerun the Cocoa general scripts. Instead, run the following three commands:
>
>      source start_cocoa.sh
>
> and
>
>      source ./installation_scripts/setup_cosmolike_projects.sh
>
> and
>
>       source ./installation_scripts/compile_all_projects.sh
>
> or (in case users just want to compile des_cluster project)
>
>       source ./projects/des_cluster/scripts/compile_des_cluster.sh

> [!TIP]
> Assuming Cocoa is installed on a local (not remote!) machine, type the command below after step 2 to run Jupyter Notebooks.
>
>     jupyter notebook --no-browser --port=8888
>
> The terminal will then show a message similar to the following template:
>
>     (...)
>     [... NotebookApp] Jupyter Notebook 6.1.1 is running at:
>     [... NotebookApp] http://f0a13949f6b5:8888/?token=XXX
>     [... NotebookApp] or http://127.0.0.1:8888/?token=XXX
>     [... NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
>
> Now go to the local internet browser and type `http://127.0.0.1:8888/?token=XXX`, where XXX is the previously saved token displayed on the line
>
>     [... NotebookApp] or http://127.0.0.1:8888/?token=XXX
>
> The project des_cluster contains two jupyter notebook examples located at `projects/des_cluster` (see [The notebooks and the notebook wrappers](#des_cluster_notebook)).

To run the example

 **Step :one:**: activate the cocoa Conda environment,  and the private Python environment

      conda activate cocoa

and

      source start_cocoa.sh

 **Step :two:**: Select the number of OpenMP cores (below, we set it to 8).

  - Linux

        export OMP_NUM_THREADS=8; export OMP_PROC_BIND=close; \
        export OMP_PLACES=cores; export OMP_DYNAMIC=FALSE; \
        export OPENBLAS_NUM_THREADS=1; export MKL_NUM_THREADS=1

  - macOS (arm)

        export OMP_NUM_THREADS=8; export OMP_PROC_BIND=disabled; \
        export OMP_PLACES=cores; export OMP_DYNAMIC=FALSE; \
        export OPENBLAS_NUM_THREADS=1; export MKL_NUM_THREADS=1

 **Step :three:**: The folder `projects/des_cluster` contains examples. So, run the `cobaya-run` on the first example following the commands below.

The examples numbered `1` run 4x2pt + N, and the examples numbered `2` run 6x2pt + N. The examples of this section compute the Boltzmann inputs with CAMB (Takahashi Halofit).

> [!Warning]
> (Linux only) In some HPC nodes, `numa` can cause you problems. If that is the case,
> replace `numa` with `slot`

- **One model evaluation**:

  - Linux

        "${CONDA_PREFIX}"/bin/mpirun -n 1 --oversubscribe \
          --mca pml ob1 --mca btl vader,tcp,self \
          --bind-to core:overload-allowed --report-bindings \
          --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} \
          cobaya-run ./projects/des_cluster/EXAMPLE_EVALUATE1.yaml -f

  - macOS (arm)

         mpirun -n 1 --oversubscribe \
          cobaya-run ./projects/des_cluster/EXAMPLE_EVALUATE1.yaml -f

  `EXAMPLE_EVALUATE1.yaml` and `EXAMPLE_EVALUATE2.yaml` evaluate the likelihood at the point of the synthetic data vector (Table I of arXiv:2503.13631), so they return $\chi^2 \approx 0$.

- **MCMC (Metropolis-Hastings Algorithm)**:

  - Linux

        "${CONDA_PREFIX}"/bin/mpirun -n 4 --oversubscribe \
          --mca pml ob1 --mca btl vader,tcp,self \
          --bind-to core:overload-allowed --report-bindings \
          --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} \
          cobaya-run ./projects/des_cluster/EXAMPLE_MCMC1.yaml -f

  - macOS (arm)

          mpirun -n 4 --oversubscribe \
            cobaya-run ./projects/des_cluster/EXAMPLE_MCMC1.yaml -f

  `EXAMPLE_MCMC1.yaml` and `EXAMPLE_MCMC2.yaml` sample $\Lambda$CDM, as in the paper, with the neutrino mass free (`w` and `w0pwa` are fixed at -1; to sample $w_0$-$w_a$, give them a prior as in the evaluate examples). No proposal covariance is shipped for the cluster combinations, so the first run learns it.

The outputs are written to `projects/des_cluster/chains/`.

# Table of contents <a name="table_of_contents"></a>

1. [Running Hybrid Cosmolike-ML emulators](#des_cluster_examples_emul2)
2. [The notebooks and the notebook wrappers](#des_cluster_notebook)
    1. [The plotting functions](#des_cluster_plots)
    2. [The notebook wrappers](#des_cluster_wrappers)
3. [Likelihood options and nuisance parameters](#des_cluster_likelihood)
    1. [Cluster options](#des_cluster_options)
    2. [Nuisance parameters](#des_cluster_params)
4. [Synthetic data](#des_cluster_data)
    1. [Files](#des_cluster_data_files)
    2. [Regenerating the data](#des_cluster_data_scripts)
5. [Unit tests and validation tools](#des_cluster_unit_tests)
    1. [Validation tools](#des_cluster_validation)
    2. [Timing](#des_cluster_timing)
6. [Status and known limitations](#des_cluster_status)

# Running Hybrid Cosmolike-ML emulators <a name="des_cluster_examples_emul2"></a>

> [!Warning]
> The code and examples associated with this section are still in alpha stage

The hybrid approach emulates only the Boltzmann outputs (comoving distance and linear and nonlinear matter power spectra), while cosmolike computes the data vector. Changes to the modeling of nuisance parameters or to the assumed redshift distributions then do not require retraining a network, and nearly all the run time is cosmolike. This project has no emulator of the data vector itself: the likelihoods refuse `use_emulator: 1` with clusters.

Examples in the hybrid case all have the prefix **EXAMPLE_EMUL2** (note the `2`). They set `use_emulator: 2` on the likelihood and replace the `camb` theory block with the emulator blocks `emulrdrag`, `emulbaosn`, and `emulmps`. Before running them, ensure the following lines are commented out in `set_installation_options.sh` before running the `setup_cocoa.sh` and `compile_cocoa.sh`. By default, these lines should be commented out, but it is worth checking.

      [Adapted from Cocoa/set_installation_options.sh shell script]
      # insert the # symbol (i.e., unset these environmental keys  on `set_installation_options.sh`)
      #export IGNORE_EMULTRF_CODE=1              #SaraivanovZhongZhu (SZZ) transformer/CNN-based emulators
      #export IGNORE_EMULTRF_DATA=1
      #export IGNORE_NAUTILUS_SAMPLER_CODE=1     # to run EXAMPLE_EMUL2_NAUTILUS1.py
      #export IGNORE_POLYCHORD_SAMPLER_CODE=1    # to run EXAMPLE_EMUL2_POLY1.yaml

> [!NOTE]
> **The emulators are fixed at a neutrino mass of 0.06 eV.** The EMUL2 examples set `mnu` to 0.06 eV and do not sample it, while the synthetic data vector was generated with CAMB at $\Omega_\nu h^2 = 0.00083$ (`mnu` = 0.077 eV). The $\chi^2$ of `EXAMPLE_EMUL2_EVALUATE1.yaml` and `EXAMPLE_EMUL2_EVALUATE2.yaml` at the fiducial point is therefore not zero. At equal neutrino mass, the emulated inputs differ from CAMB by $\Delta\chi^2 = 1.6$ on 6x2pt + N (the cluster counts differ by 1.5% in the median).

Now, users must follow all the steps below.

 **Step :one:**: Activate the private Python environment by sourcing the script `start_cocoa.sh`

    source start_cocoa.sh

 **Step :two:**: Select the number of OpenMP cores. Below, we set it to 4 (in the table of the section [Timing](#des_cluster_timing), going from 4 to 8 threads gains a factor of 1.4 on 6x2pt + N).

  - Linux

        export OMP_NUM_THREADS=4; export OMP_PROC_BIND=close; \
        export OMP_PLACES=cores; export OMP_DYNAMIC=FALSE; \
        export OPENBLAS_NUM_THREADS=1; export MKL_NUM_THREADS=1

  - macOS (arm)

        export OMP_NUM_THREADS=4; export OMP_PROC_BIND=disabled; \
        export OMP_PLACES=cores; export OMP_DYNAMIC=FALSE; \
        export OPENBLAS_NUM_THREADS=1; export MKL_NUM_THREADS=1

 **Step :three:** Run `cobaya-run` on the first emulator example, following the commands below. As before, the examples numbered `1` run 4x2pt + N and the examples numbered `2` run 6x2pt + N.

- **One model evaluation**:

  - Linux

        "${CONDA_PREFIX}"/bin/mpirun -n 1 --oversubscribe \
          --mca pml ob1 --mca btl vader,tcp,self \
          --bind-to core:overload-allowed --report-bindings \
          --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} \
          cobaya-run ./projects/des_cluster/EXAMPLE_EMUL2_EVALUATE1.yaml -f

  - macOS (arm)

        mpirun -n 1 --oversubscribe \
          cobaya-run ./projects/des_cluster/EXAMPLE_EMUL2_EVALUATE1.yaml -f

- **Benchmark (1000 model evaluations)**:

  `EXAMPLE_EMUL2_BENCHMARK1.yaml` (4x2pt + N) and `EXAMPLE_EMUL2_BENCHMARK2.yaml` (6x2pt + N) are the workloads for profiling cosmolike. Each runs 1000 evaluations with a fixed seed, and each evaluation draws every sampled parameter afresh from its `ref` distribution, so no cosmolike table is served from a cache. With the emulated Boltzmann inputs nearly all the run time is cosmolike. To profile, put the profiler (e.g., `perf stat` on Linux) in front of `cobaya-run`.

  - Linux

        "${CONDA_PREFIX}"/bin/mpirun -n 1 --oversubscribe \
          --mca pml ob1 --mca btl vader,tcp,self \
          --bind-to core:overload-allowed --report-bindings \
          --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} \
          cobaya-run ./projects/des_cluster/EXAMPLE_EMUL2_BENCHMARK1.yaml -f

  - macOS (arm)

        mpirun -n 1 --oversubscribe \
          cobaya-run ./projects/des_cluster/EXAMPLE_EMUL2_BENCHMARK1.yaml -f

- **MCMC (Metropolis-Hastings Algorithm)**:

  - Linux

        "${CONDA_PREFIX}"/bin/mpirun -n 4 --oversubscribe \
          --mca pml ob1 --mca btl vader,tcp,self \
          --bind-to core:overload-allowed --report-bindings \
          --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} \
          cobaya-run ./projects/des_cluster/EXAMPLE_EMUL2_MCMC1.yaml -r

  - macOS (arm)

        mpirun -n 4 --oversubscribe \
          cobaya-run ./projects/des_cluster/EXAMPLE_EMUL2_MCMC1.yaml -r

  The MCMC and PolyChord examples sample $\Lambda$CDM (`w` and `w0pwa` are fixed at -1; to sample $w_0$-$w_a$, give them a prior as in the evaluate examples).

> [!NOTE]
> **Running on more than one node.** The flag `--mca btl vader,tcp,self` works unchanged across
> nodes: Open MPI picks the transport per pair of ranks, using shared memory (`vader`) within a
> node and TCP between nodes. Three things deserve attention on multi-node runs:
>
> 1. **Network interface.** The TCP layer must not select an interface that is not routable
>    between compute nodes. The flag `--mca btl_tcp_if_exclude lo,docker0,virbr0,ib0` excludes
>    the common offenders. TCP bandwidth is not a limitation for our workloads, which exchange
>    small, infrequent MPI messages.
>
> 2. **Environment.** Ranks on remote nodes must see Cocoa's environment (`ROOTDIR`, `PATH`,
>    `LD_LIBRARY_PATH`, `PYTHONPATH`, `CONDA_PREFIX`, the OpenMP/BLAS thread settings, and
>    `CLIK_PATH`/`CLIK_DATA`/`CLIK_PLUGIN`). Slurm forwards the submitting environment
>    automatically; the explicit `-x` flags in our sbatch templates repeat this so the
>    scripts also work under ssh-based launchers. No other Cocoa installation flags are read at runtime.
>
> 3. **Slurm geometry.** Keep `ntasks-per-node` × `cpus-per-task` no larger than the cores per
>    node, and use `--map-by numa:pe=${OMP_NUM_THREADS}` so each rank reserves the cores its
>    OpenMP threads will use.

> [!NOTE]
> **Note on core oversubscription**: an MPI process that is waiting still burns 100% of its
> core, checking for messages in a loop. With more processes than cores, this stalls the
> processes doing real work. Open MPI usually detects this and makes waiting processes give
> up the CPU, but its detection can be fooled. Adding `--mca mpi_yield_when_idle 1` forces
> that behavior; it is harmless otherwise.

The `Nautilus`, `Minimizer`, and `Profile` scripts below contain an internally
defined `yaml_string` that specifies priors,
likelihoods, and the theory code, all following Cobaya Conventions.
The `PolyChord` example, in contrast, is configured directly by the YAML file `EXAMPLE_EMUL2_POLY1.yaml`.

- **Nautilus**:

  - Linux

        export OMP_NUM_THREADS=1

        "${CONDA_PREFIX}"/bin/mpirun -n 96 --oversubscribe --mca pml ob1 --mca btl vader,tcp,self \
          -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x CONDA_PREFIX -x ROOTDIR \
          -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES -x OMP_DYNAMIC \
          -x OPENBLAS_NUM_THREADS -x MKL_NUM_THREADS -x CLIK_PATH -x CLIK_DATA \
          -x CLIK_PLUGIN --mca mpi_yield_when_idle 1 \
          --mca btl_tcp_if_exclude lo,docker0,virbr0,ib0 \
          --bind-to core:overload-allowed --report-bindings \
          --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} \
          python -m mpi4py.futures ./projects/des_cluster/EXAMPLE_EMUL2_NAUTILUS1.py \
            --root ./projects/des_cluster/ --outroot "EXAMPLE_EMUL2_NAUTILUS1"  \
            --maxfeval 750000 --nlive 2048 --neff 15000 \
            --flive 0.01 --nnetworks 5

  - macOS (arm)

        export OMP_NUM_THREADS=1

        mpirun -n 12 --oversubscribe \
          python -m mpi4py.futures ./projects/des_cluster/EXAMPLE_EMUL2_NAUTILUS1.py \
            --root ./projects/des_cluster/ \
            --outroot "EXAMPLE_EMUL2_NAUTILUS1" \
            --maxfeval 750000 --nlive 2048 --neff 15000 \
            --flive 0.01 --nnetworks 5

  The script writes the weighted samples to `chains/EXAMPLE_EMUL2_NAUTILUS1.1.txt`, together with the `.ranges`, `.paramnames`, and `.covmat` files, and keeps a checkpoint at `chains/EXAMPLE_EMUL2_NAUTILUS1_checkpoint.hdf5`, from which a new run resumes.

- **PolyChord**:

  - Linux (assuming node with 96 cores)

        export OMP_NUM_THREADS=4

        "${CONDA_PREFIX}"/bin/mpirun -n 24 --oversubscribe --mca pml ob1 --mca btl vader,tcp,self \
          -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x CONDA_PREFIX -x ROOTDIR \
          -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES -x OMP_DYNAMIC \
          -x OPENBLAS_NUM_THREADS -x MKL_NUM_THREADS -x CLIK_PATH -x CLIK_DATA \
          -x CLIK_PLUGIN --mca mpi_yield_when_idle 1 \
          --mca btl_tcp_if_exclude lo,docker0,virbr0,ib0 \
          --bind-to core:overload-allowed --report-bindings \
          --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} \
          cobaya-run ./projects/des_cluster/EXAMPLE_EMUL2_POLY1.yaml -r

  - macOS (arm)

        export OMP_NUM_THREADS=1

        mpirun -n 12 --oversubscribe \
          cobaya-run ./projects/des_cluster/EXAMPLE_EMUL2_POLY1.yaml -r

- **Global Minimizer**:

  Our minimizer is a reimplementation of `Procoli`, developed by Karwal et al (arXiv:2401.14225)

  - Linux (assuming node with 96 cores)

        export OMP_NUM_THREADS=4

        "${CONDA_PREFIX}"/bin/mpirun -n 24 --oversubscribe --mca pml ob1 --mca btl vader,tcp,self \
          -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x CONDA_PREFIX -x ROOTDIR \
          -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES -x OMP_DYNAMIC \
          -x OPENBLAS_NUM_THREADS -x MKL_NUM_THREADS -x CLIK_PATH -x CLIK_DATA \
          -x CLIK_PLUGIN --mca mpi_yield_when_idle 1 \
          --mca btl_tcp_if_exclude lo,docker0,virbr0,ib0 \
          --bind-to core:overload-allowed --report-bindings \
          --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} \
          python ./projects/des_cluster/EXAMPLE_EMUL2_MINIMIZE1.py \
            --root ./projects/des_cluster/ \
            --outroot "EXAMPLE_EMUL2_MIN1" \
            --nstw 350

  - macOS (arm)

        export OMP_NUM_THREADS=1

        mpirun -n 12 --oversubscribe \
          python ./projects/des_cluster/EXAMPLE_EMUL2_MINIMIZE1.py \
            --root ./projects/des_cluster/ \
            --outroot "EXAMPLE_EMUL2_MIN1" \
            --nstw 350

  The number of steps per Emcee walker per temperature is $n_{\rm stw}$,
  and the number of walkers is $n_{\rm w}={\rm max}(3n_{\rm params},n_{\rm MPI})$.
  The minimum number of total evaluations is $3n_{\rm params} \times n_{\rm T} \times n_{\rm stw}$, which can be distributed among $n_{\rm MPI} = 3n_{\rm params}$ MPI processes for faster results.
  The script writes the minimum to `chains/EXAMPLE_EMUL2_MIN1.txt`.

- **Profile**:

  - Linux (assuming node with 96 cores)

        export OMP_NUM_THREADS=4

        "${CONDA_PREFIX}"/bin/mpirun -n 24 --oversubscribe --mca pml ob1 --mca btl vader,tcp,self \
          -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x CONDA_PREFIX -x ROOTDIR \
          -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES -x OMP_DYNAMIC \
          -x OPENBLAS_NUM_THREADS -x MKL_NUM_THREADS -x CLIK_PATH -x CLIK_DATA \
          -x CLIK_PLUGIN --mca mpi_yield_when_idle 1 \
          --mca btl_tcp_if_exclude lo,docker0,virbr0,ib0 \
          --bind-to core:overload-allowed --report-bindings \
          --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} \
          python ./projects/des_cluster/EXAMPLE_EMUL2_PROFILE1.py \
            --root ./projects/des_cluster/ --cov 'chains/EXAMPLE_EMUL2_MCMC1.covmat' \
            --outroot "EXAMPLE_EMUL2_PROFILE1" \
            --factor 3 --nstw 350 --numpts 10 \
            --profile 1 \
            --minfile="./projects/des_cluster/chains/EXAMPLE_EMUL2_MIN1.txt"

  - macOS (arm)

        export OMP_NUM_THREADS=1

        mpirun -n 12 --oversubscribe \
          python ./projects/des_cluster/EXAMPLE_EMUL2_PROFILE1.py \
            --root ./projects/des_cluster/ \
            --cov 'chains/EXAMPLE_EMUL2_MCMC1.covmat' \
            --outroot "EXAMPLE_EMUL2_PROFILE1" \
            --factor 3 --nstw 350 --numpts 10 --profile 1 \
            --minfile="./projects/des_cluster/chains/EXAMPLE_EMUL2_MIN1.txt"

  The commands above read two files that earlier examples write: the covariance `chains/EXAMPLE_EMUL2_MCMC1.covmat` (from the MCMC example) and the minimum `chains/EXAMPLE_EMUL2_MIN1.txt` (from the minimizer example). Without `--minfile`, the script computes the minimum itself, which is slow.

  The argument `profile` is the index of the profiled parameter in Cobaya's list of sampled parameters (the script prints its name at the start), and `numpts` sets the number of points of the profile. The argument `factor` specifies the start and end of the parameter being profiled:

      start value ~ minimum value - factor*np.sqrt(np.diag(cov))
      end   value ~ minimum value + factor*np.sqrt(np.diag(cov))

  We advise ${\rm factor} \sim 3$ for parameters that are well constrained by the data when a covariance matrix is provided.
  If `cov` is not supplied, the code estimates one internally from the prior.
  If a parameter is poorly constrained or `cov` is not given, we recommend ${\rm factor} \ll 1$.
  The script writes the profile to `chains/EXAMPLE_EMUL2_PROFILE1.<parameter name>.txt`.

> [!Warning]
> When running Profiles, you should not set flat priors on parameters that are not well constrained by the data.
> By doing that, you then risk having the minimizer select values near the boundary of parameter space. This is a big problem when using emulators, as volume near the
> boundary will be inevitable outside the training range. You can convert a flat prior to a Gaussian one by setting the standard deviation to be $\sigma^2 = (hi - lo)^2/12$,
> where $(lo, hi)$ are the flat prior boundaries. The `Nautilus`, `Minimizer`, and `Profile` scripts of this project add the following prior block
>
>      prior:
>        # These priors are meant to prevent the sampler to wander far off training
>        g1: "lambda As_1e9: stats.norm.logpdf(As_1e9, loc=2.35, scale=1.6)"
>        g2: "lambda ns: stats.norm.logpdf(ns, loc=0.96, scale=0.05)"
>        g3: "lambda H0: stats.norm.logpdf(H0, loc=70, scale=10.0)"
>        g4: "lambda omegab: stats.norm.logpdf(omegab, loc=0.045, scale=0.012)"
>        g5: "lambda omegam: stats.norm.logpdf(omegam, loc=0.3 , scale=0.25)"

Details on the matter power spectrum emulator designs will be presented in the
[emulator_code](https://github.com/CosmoLike/emulators_code) repository.

Basically, we apply standard neural network techniques to generalize
the *syren-new* Eq. 6 of [arXiv:2410.14623](https://arxiv.org/abs/2410.14623)
formula for the linear power spectrum (w0waCDM with a fixed neutrino mass of $0.06$ eV)
to new models, extended ranges, or higher precision.
Similarly, we use networks to generalize the *syren-Halofit* LCDM nonlinear
boost fit (Eq. 11 of [arXiv:2402.17492](https://arxiv.org/abs/2402.17492)).

# The notebooks and the notebook wrappers <a name="des_cluster_notebook"></a>

The notebook `EXAMPLE_EVALUATE1.ipynb` computes the cluster observables at the fiducial point with the options of `EXAMPLE_EVALUATE1.yaml` (4x2pt + N) and compares them with the synthetic data where the scale cuts keep them. It covers, in this order: the cluster counts per richness and redshift bin; the halo-model ingredients (the probability of a richness bin given the halo mass, the number density and the bias of each richness bin, the redshift selection kernels, the one-halo cluster-matter power spectrum); cluster lensing, both as $\gamma_t$ and as the $\Sigma = Y\gamma_t$ statistic of the data vector; $w_{cc}$; $w_{cg}$; the Limber spectra $C_\ell^{cs}$, $C_\ell^{cc}$, and $C_\ell^{cg}$; the response of the counts and of $\gamma_t$ to the mass-observable relation; and the $\chi^2$ against the synthetic data vector.

The notebook `EXAMPLE_EVALUATE2.ipynb` runs 6x2pt + N with the options of `EXAMPLE_EVALUATE2.yaml`, in the form of parameter sweeps. It computes a reference model at the fiducial point, then varies one parameter at a time (five values each) and shows every block as a ratio to the reference, with the curves colored by the parameter value. The sweeps cover $\Omega_m$ and $A_s$; two parameters of the mass-observable relation ($\ln\lambda_0$ and $\sigma_{\rm int}$); two parameters of the selection bias ($b_{s2}$ and $r_0$); the number of angular bins; and the accuracy setting of cosmolike. The $\Omega_m$ sweep also shows the 3x2pt blocks ($\xi_\pm$, galaxy-galaxy lensing, and $w(\theta)$). The notebook ends with the $\chi^2$ against the synthetic data vector.

> [!NOTE]
> The notebooks load their support functions from two places. The functions shared by
> all projects (the CAMB run packaged for cosmolike, the plotting functions) come from
> `Cocoa/external_modules/code/cosmolike_core/cosmolike_notebook_utils/`. The
> functions specific to this project (fiducial values, the compiled-interface calls,
> the cluster observables) come from
> `interface/cosmolike_des_cluster_notebook_wrappers.py`.

## The plotting functions <a name="des_cluster_plots"></a>

Every data-vector figure of the two notebooks comes from a plotting function. The cluster blocks use `plot_datavectors_cluster.py` of `cosmolike_notebook_utils`, the cluster version of the functions the other projects use for the galaxy blocks. The module is imported on its own:

    from cosmolike_notebook_utils import plot_datavectors_cluster as pdc

| function | panels | curves inside a panel |
|---|---|---|
| `plot_N_cluster` | one per cluster z bin; the x axis is the richness | one staircase per model |
| `plot_gammat_cluster_tomo`, `plot_sigma_cluster_tomo`, `plot_C_cs_tomo_limber` | one per (cluster z bin, source bin) pair: columns are cluster z bins, rows are source bins | the richness bins |
| `plot_wcc_tomo`, `plot_C_cc_tomo_limber` | one per cluster z bin | the richness pairs (by default the auto pairs) |
| `plot_wcg_tomo`, `plot_C_cg_tomo_limber` | one per (cluster z bin, lens bin) pair of the dataset | the richness bins |

The functions follow the conventions of the galaxy functions. They take a list of models; `param` and `colorbarlabel` color the list by a parameter value; a reference (the `*_ref` argument) turns every panel into the fractional difference to it; `rescale = 1` joins the panels on one y axis, with one power of ten per panel; and `show = None` returns the figure. The cluster blocks have one more index than the galaxy blocks, the richness bin. The richness bins are curves inside each panel: the color follows the model and the line style follows the richness bin. The argument `richness` (`pairs` for $w_{cc}$) selects the bins to draw, counted from 0. The argument `data` overlays the data vector with its errors and leaves out the points the scale cuts remove. The 3x2pt blocks use the galaxy functions `cnu.plot_xi`, `cnu.plot_gammat_tomo_limber`, and `cnu.plot_wtheta_tomo`. The docstrings of the module document every argument.

## The notebook wrappers <a name="des_cluster_wrappers"></a>

The wrappers module drives the compiled interface through the same steps as the likelihoods: it runs CAMB once, pushes the power spectra, growth, and distances into the interface, sets the nuisance parameters, and reads off an observable. Its usage is

    import cosmolike_des_cluster_notebook_wrappers as nw
    nw.configure(cluster_ytransform=1)       # optional: the yaml options, before init_cosmolike
    nw.init_cosmolike(CLprobe="6x2pt_N", with_data=True)
    N = nw.N_cluster()                       # (richness bin, z bin)
    theta, gammat = nw.gamma_t_cluster()     # (theta, richness, z, source)
    nw.get_chi2(omegam=0.31)

| function | returns |
|---|---|
| `configure(**overrides)` | nothing; sets the values that mirror the likelihood yaml (`lmax`, `integration_accuracy`, the `cluster_*` switches, the data path). Call it before `init_cosmolike` |
| `init_cosmolike(CLprobe, with_data, lmax)` | the parsed dataset file; runs the init sequence of the likelihood. `with_data=True` also loads the mask, the data vector, and the covariance, which `get_datavector` and `get_chi2` need |
| `dataset_info()` | the binning and the bin edges of the loaded dataset |
| `N_cluster()` | cluster counts, array (richness bin, cluster z bin) |
| `gamma_t_cluster()` | `(theta, gammat)`: cluster tangential shear before the Y transform, the selection bias, and the shear calibration; array ($\theta$, richness bin, cluster z bin, source bin) |
| `sigma_cluster()` | `(theta, sigma)`: cluster lensing as the data vector holds it, without the mask; same array layout |
| `w_cc(selection_bias)` | `(theta, wcc)`: array ($\theta$, richness bin 1, richness bin 2, cluster z bin) |
| `w_cg(selection_bias)` | `(theta, wcg)`: array ($\theta$, richness bin, cluster z bin, lens bin) |
| `C_cs_tomo_limber(ell)`, `C_cc_tomo_limber(ell)`, `C_cg_tomo_limber(ell)` | the Limber spectra at the multipoles `ell`, with the layouts above and $\ell$ on the first axis |
| `xi()`, `gamma_t()`, `w_theta()` | the 3x2pt blocks in real space: `(theta, xi_plus, xi_minus)`, `(theta, gammat)`, and `(theta, wtheta)`, in the array layouts of the galaxy plotting functions |
| `prob_richness_bin_given_m(M, z)` | probability of each richness bin given the halo mass (M200m, in $M_\odot/h$) and the redshift |
| `ncl_richness(z)`, `bcl_richness(z)` | comoving number density, in $(h/{\rm Mpc})^3$, and richness-weighted linear bias of each richness bin |
| `pcm_1h_richness(k, z)` | one-halo cluster-matter power spectrum of each richness bin, in $({\rm Mpc}/h)^3$, with `k` in $h/{\rm Mpc}$ |
| `phi_cluster(z)` | the selection kernels $\langle\phi_i\vert z\rangle$ of the cluster redshift bins |
| `nz_cluster(z)`, `W_cluster(z)` | the normalized true-redshift distribution and the radial kernels of the cluster bins |
| `get_datavector()`, `get_chi2()` | the masked joint theory vector and its $\chi^2$ against the loaded data |
| `cluster_blocks(vector)`, `data_cluster_blocks()` | the cluster blocks (counts, cluster lensing, $w_{cc}$, $w_{cg}$) of a joint vector, or of the data and their errors, as arrays in the layouts above. Entries that the mask removes or that the data vector does not hold are NaN |

Every wrapper that computes a model takes the cosmology (`omegam`, `omegab`, `H0`, `ns`, `As_1e9`, `w`, `w0pwa`, `mnu`), the accuracy settings, and the nuisance vectors (`M`, `A1`, `B1`, `MOR`, `SEL`, ...) as keyword arguments, and falls back to the project fiducial point, which the module holds as plain constants. For example, `nw.N_cluster(MOR=[4.3, 0.943, 0.15, 0.207])` changes the mass-observable relation for one call. Each wrapper sets the complete state of the interface on every call, so no call depends on which wrapper ran before it; the module keeps the last CAMB run, so repeated calls at one cosmology cost one CAMB run. The docstrings of the module document every argument and array layout.

# Likelihood options and nuisance parameters <a name="des_cluster_likelihood"></a>

The defaults of the two likelihoods are in `likelihood/combo_4x2pt_N.yaml` and `likelihood/combo_6x2pt_N.yaml`; the examples repeat the keys discussed below. The model follows arXiv:2503.13631 (the equation numbers below are those of that paper):

- The counts (eq. 16) use the Tinker mass function and halo bias (M200m) and a lognormal mass-observable relation (eqs. 18-19, pivot mass $5\times10^{14}\,M_\odot/h$).
- The cluster-lensing spectrum is the sum of a two-halo term (richness-weighted cluster bias times the nonlinear matter power spectrum) and a one-halo NFW term (eq. 22). The data vector holds $\Sigma = Y\gamma_t$, the Y transform of eq. 15, which involves no critical surface density. The last angular bin of every cluster-lensing row vanishes by construction and is always masked.
- $w_{cc}$ and $w_{cg}$ use a linear cluster bias times the nonlinear matter power spectrum.
- The selection bias (eq. 23) multiplies the data vector: $\Sigma$ and $w_{cg}$ by $b_{s1} + b_{s2}\exp(-\theta\chi(\bar z)/r_0)$, and $w_{cc}$ by its square.
- The real-space statistics are full-sky, bin-averaged Legendre projections.

## Cluster options <a name="des_cluster_options"></a>

| key | default | meaning |
|---|---|---|
| `cluster_kernel_mode` | 0 | radial kernel of the clusters in the two-point functions: 0 = volume only, $dV/dz\,\langle\phi_i\vert z\rangle$ (what DES ran); 1 = abundance weighted |
| `cluster_selection_model` | 2 | selection bias: 0 = none; 1 = Y1, mass dependent, inside the bias mass integral; 2 = Y6, scale dependent, on the data vector (eq. 23) |
| `cluster_ytransform` | 1 | cluster lensing: 1 = $\Sigma = Y\gamma_t$ (eq. 15); 0 = $\gamma_t$. The data and covariance files must be in the same space |
| `cluster_include_ia` | 1 | 1 = intrinsic alignment of the sources in the two-halo cluster-lensing term |
| `cluster_magnification` | -2.0 | cluster magnification coefficient $C_c$ (eq. 28); 0 switches it off |
| `cluster_hmf_alpha_mode` | 0 | amplitude of the Tinker 2010 mass function: 0 = 0.368 at every redshift (what DES ran); 1 = the redshift-dependent amplitude of `halo.c` (3-9% fewer clusters at z = 0.2-0.6) |
| `cluster_adopt_limber_cc`, `cluster_adopt_limber_cg` | 1 | $w_{cc}$ and $w_{cg}$: 1 = Limber at every multipole. This is the only implemented value (see [Status and known limitations](#des_cluster_status)) |

All halo statistics use cold dark matter plus baryons (cb). Massive neutrinos free-stream out of halos, so the variance integrates CAMB's `delta_nonu` spectrum at the requested redshift, and the Lagrangian smoothing radius and mass-function density use $\rho_{cb}=\rho_{crit}(\Omega_m-\Omega_\nu)$. FFTLog tabulates $\sigma^2_{cb}(M,a)$ and its mass slope; no scale-independent growth factor is applied to a present-day variance. The Bhattacharya concentration uses $D_{cb}(M,a)=\sigma_{cb}(M,a)/\sigma_{cb}(M,1)$ in its $D^{1.15}$ factor. This is the adopted neutrino extension of that fit, which was not calibrated with massive neutrinos.

The NFW truncation radius still defines an M200m halo relative to the total mean density. The lensing mass weight $M/\rho_m$, lensing kernels, and nonlinear matter spectra also retain total matter. The likelihood always supplies $\Omega_\nu h^2$ and the cb spectrum. The former `halo_matter_field` option has been retired.

The shipped synthetic data and covariance predate the evolving cb-variance calculation. Their measured change at the fiducial point is $|\Delta\chi^2|<0.152$ for the cluster combinations; the frozen tests record the current prediction against those same data.

The EMUL2 path has no separate cb spectrum and retains the approximation $P_{cb}=P_{lin}/(1-f_\nu)^2$, with $f_\nu=\Omega_\nu/\Omega_m$ and fixed neutrino mass 0.06 eV. This is the small-scale limit of the spectrum ratio; its counts error against a true cb-spectrum emulator has not been measured.

Three more keys matter for the cluster combinations:

- `integration_accuracy: 1`. The cluster combinations run with 1, while the value 0 is not accurate enough for the Y6-like inputs: the tails of the MagLim lens redshift distributions reach z = 2.99. The accuracy sweep `tests/validation/knob_sweep.py` measured $\Delta\chi^2 = 1.65$ against high-accuracy settings at 0 (galaxy-galaxy lensing 0.92, galaxy clustering 0.47) and 0.048 at 1, for about 30% more cosmolike time. The cluster blocks contribute $6\times10^{-3}$ either way.
- `accuracyboost`. Keep it at 3 or below: above 3, the integration tables of the project this one was created from (desy1xplanck) broke down, and this limit was not re-measured here.
- `IA_model: 0`. The intrinsic-alignment model is NLA: the cluster lensing code has no TATT.

Galaxy clustering is non-Limber below $\ell = 150$ (`adopt_limber_gg: 0`), and galaxy-galaxy lensing is Limber (`adopt_limber_gs: 1`). The likelihoods refuse baryon PCAs with clusters.

## Nuisance parameters <a name="des_cluster_params"></a>

The nuisance parameters and their priors come from the likelihood defaults `likelihood/params_cluster.yaml`, `likelihood/params_lens_maglim.yaml`, and `likelihood/params_source_y6.yaml`, so the examples do not list them in their `params` block. The fiducial values are those of Table I of arXiv:2503.13631, the point of the synthetic data vector.

Clusters (`params_cluster.yaml`):

| name | meaning | fiducial | prior |
|---|---|---|---|
| `DES_CL_LNLAMBDA0` | $\ln\lambda_0$, amplitude of the mass-observable relation (eq. 19) | 4.26 | flat [2, 5] |
| `DES_CL_A_LAMBDA` | $A_{\ln\lambda}$, slope of the relation in $\ln M$ | 0.943 | flat [0.1, 1.5] |
| `DES_CL_B_LAMBDA` | $B_{\ln\lambda}$, slope of the relation in $\ln(1+z)$ | 0.207 | flat [-5, 5] |
| `DES_CL_SIGMA_INT` | $\sigma_{\rm int}$, intrinsic scatter of the relation (eq. 18) | 0.15 | flat [0.1, 1] |
| `DES_CL_BS1` | $b_{s1}$, selection bias (eq. 23) | 1.1 | flat [1, 2] |
| `DES_CL_BS2` | $b_{s2}$, selection bias (eq. 23) | 0.2 | flat [-1, 1] |
| `DES_CL_R0` | $r_0$, scale of the selection bias, in comoving ${\rm Mpc}/h$ | 30 | flat [10, 60] |
| `DES_CL_BSZ` | power of $(1+\bar z)/1.45$ multiplying the selection factor; 0 is the paper's model | 0 | fixed |

Sources (`params_source_y6.yaml`):

| name | meaning | fiducial | prior |
|---|---|---|---|
| `DES_DZ_S1` ... `DES_DZ_S4` | photo-z shift of each source bin | 0.034, 0.028, 0.011, -0.010 | Gaussian, width 0.018, 0.013, 0.006, 0.013 |
| `DES_M1` ... `DES_M4` | shear calibration of each source bin | 0 | Gaussian, width 0.008, 0.013, 0.009, 0.012 |
| `DES_A1_1` | NLA amplitude $a_1$ | 0 | flat [-5, 5] |
| `DES_A1_2` | NLA redshift power $\eta_1$ | 0 | Gaussian, width 3 |
| `DES_A2_1`, `DES_A2_2`, `DES_BTA_1` | TATT terms, unused under NLA | 0, 0, 1 | fixed |

MagLim lenses (`params_lens_maglim.yaml`):

| name | meaning | fiducial | prior |
|---|---|---|---|
| `DES_DZ_L1` ... `DES_DZ_L6` | photo-z shift of each lens bin | 0.005, 0.003, 0.001, -0.002, 0.001, 0.008 | Gaussian, width 0.007, 0.011, 0.006, 0.006, 0.01, 0.01 |
| `DES_B1_1` ... `DES_B1_6` | linear galaxy bias of each lens bin | 1.42, 1.66, 1.70, 1.62, 1.78, 1.75 | flat [0.8, 3] |
| `DES_PM1` ... `DES_PM6` | point mass of galaxy-galaxy lensing, in units of $10^{13}\,M_\odot/h$ | 0 | flat [-100, 100] |
| `DES_BMAG_1` ... `DES_BMAG_6` | magnification coefficient of each lens bin | -1.57, -1.70, -0.25, 1.50, 2.22, 2.80 | fixed |
| `DES_DZ2_L1` ... `DES_DZ2_L6` | photo-z stretch of each lens bin | 1 | fixed |
| `DES_B2_1` ... `DES_B2_6` | nonlinear galaxy bias (the model is linear bias) | 0 | fixed |

4x2pt + N uses lens bins 1-3 only and has no galaxy-galaxy lensing block, so `combo_4x2pt_N.yaml` fixes the bias and the photo-z shift of lens bins 4-6 at their fiducial values and the six point masses at zero (its `fixed_params` block): these parameters enter only blocks that the 4x2pt + N mask removes. 6x2pt + N samples the bias and the photo-z shift of all six bins and the six point masses. The clusters have no photo-z nuisance parameter: their redshift selection enters through the fixed kernels $\langle\phi_i|z\rangle$ of `data/des_y6_cluster.nz`.

[likelihood/README.md](likelihood/README.md) lists, for every likelihood of this project, what its `fixed_params` block fixes and which parameters it must not vary, with the reason for each.

The fiducial cosmology of the synthetic data is `As_1e9` = 2.19, `ns` = 0.96859, `H0` = 69, `omegab` = 0.048, `omegam` = 0.3, $\Omega_\nu h^2 = 0.00083$ (`mnu` = 0.077 eV), and $w = -1$.

# Synthetic data <a name="des_cluster_data"></a>

The synthetic data set is Y6-like in the following sense: the lens and source redshift distributions are the n(z) files of the DES Y6 code comparison kept in the repository of the original CosmoLike cluster code (`lighthouse`), the binning and the scale cuts are those of arXiv:2503.13631, the parameters are those of its Table I, and the survey area is 4143 deg$^2$ (the DES Y3 cluster footprint of arXiv:2503.13632). The data vector is the model of this code at that point, without noise. The covariance is Gaussian: for the two-point blocks, the full-sky, bin-averaged Gaussian term with shape and shot noise (with the Y transform applied to the cluster-lensing rows); for the counts, Poisson plus sample variance. The cross-covariance between the counts and the two-point blocks is zero, and the covariance has no super-sample and no trispectrum term. The signal-to-noise of the data vector is 109 for 4x2pt + N and 157 for 6x2pt + N.

## Files <a name="des_cluster_data_files"></a>

The files of `data/` that the cluster likelihoods read:

| file | content |
|---|---|
| `des_cluster_y6.dataset` | the base dataset descriptor: file names, binning, cluster redshift and richness edges, survey area, and the lens bin paired with each cluster redshift bin in $w_{cg}$ |
| `des_cluster_y6_4x2ptN.dataset`, `des_cluster_y6_6x2ptN.dataset` | the descriptors the likelihoods load: the base descriptor with the mask of each combination |
| `des_cluster_y6.datavector` | the joint synthetic data vector (2812 entries; columns: index, value) |
| `des_cluster_y6_cov.npy` | the joint covariance: the upper triangle with the diagonal, row by row, as a binary float64 NumPy array (32 MB; cosmolike reads it directly, and `numpy.load` returns the packed triangle). Stored with Git LFS |
| `des_cluster_y6_4x2ptN.mask`, `des_cluster_y6_6x2ptN.mask` | the scale-cut masks of the two combinations |
| `des_y6_maglim.nz`, `des_y6_source.nz` | redshift distributions of the six MagLim lens bins and of the four source bins. The z column holds left bin edges (`photoz_zmid_convention: 0`) |
| `des_y6_cluster.nz` | the selection kernels $\langle\phi_i\vert z\rangle$ of the three cluster redshift bins: the probability that a cluster at true redshift z lands in each $z_\lambda$ bin. It is a probability, not a normalized distribution, and it is sampled at z (no half-cell offset) |
| `y3_redmapper_counts.txt` | the DES Y3 redMaPPer counts per redshift and richness bin, from the public Y3 catalog. The model does not read it: `make_synthetic_data.py` compares the model counts with it, and `make_cluster_mask.py` reads it for an optional cut on $w_{cc}$ that is off by default |

The remaining files (`des_y3_real.dataset`, `des_y3_unblinded_final.txt`, `des_y3_cov_unblinded_final.txt`, `des_y3_lens.nz`, `des_y3_source.nz`, `3x2pt_baseline.mask`, `ones.mask`, `pca.txt`, `baryons_logPkR.h5`) are a copy of the DES Y3 3x2pt data set of the project des_y3. The galaxy-only likelihoods of this project (`cosmic_shear`, `combo_3x2pt`, `combo_2x2pt`) point at it, as do their examples `EXAMPLE_EVALUATE3.yaml` (cosmic shear) and `EXAMPLE_EVALUATE4.yaml` (3x2pt) and their unit tests; the cluster likelihoods do not use it.

The scale cuts are defined at the mean redshift of each bin, at the fiducial cosmology: a point survives when its angle exceeds $R/\chi(\bar z)$, with $R = 2\,{\rm Mpc}/h$ for cluster lensing, $8\,{\rm Mpc}/h$ for $w_{cg}$ and $w_{gg}$, $16\,{\rm Mpc}/h$ for $w_{cc}$, and $6\,{\rm Mpc}/h$ for galaxy-galaxy lensing; cosmic shear uses a table of minimum angles per source pair. In cluster lensing, a (cluster bin, source bin) pair is masked entirely when the cluster bin's upper edge lies above the mean redshift of the source bin.

## Regenerating the data <a name="des_cluster_data_scripts"></a>

The scripts are in `scripts/`. We assume users are in the Conda cocoa environment, that `start_cocoa.sh` was sourced, and that the current folder is the cocoa main folder `cocoa/Cocoa`. Each script documents its options in its header.

| step | script | writes | notes |
|---|---|---|---|
| 1 | `make_des_y6_nz.py` | `des_y6_maglim.nz`, `des_y6_source.nz` | converts the lens and source n(z) of the DES Y6 code comparison of the `lighthouse` repository to the cosmolike n(z) format. The default input path is a local checkout of that repository: pass `--lens-in` and `--source-in` |
| 2 | `make_cluster_zdist.py` | `des_y6_cluster.nz` | tabulates the selection kernels for a Gaussian cluster photo-z of width $0.006\,(1+z)$ on a uniform grid (`--edges`, `--sigma0`, `--sigma-per-bin`, `--tophat`) |
| 3 | `make_y3_redmapper_counts.py` | `y3_redmapper_counts.txt` | tabulates the counts of the public DES Y3 redMaPPer catalog (`--h5` or `--npz`) |
| 4 | `make_cluster_mask.py` | the two masks | applies the scale cuts above; the radii, the rule for the bin angle, and the choice of $\bar z$ are options |
| 5 | `make_synthetic_data.py` | `des_cluster_y6.datavector`, `des_cluster_y6_cov.npy` | computes the data vector with the compiled code and the covariance with the Python reference (`tests/reference/ref_covariance_full.py`), checks the layout, the positive definiteness of the covariance under each mask, and the signal-to-noise per block, and asserts $\chi^2 < 10^{-6}$ at the fiducial for both combinations |
| 6 | `make_example_files.py` | every example except the two evaluate examples | see below |

For example, steps 4 and 5 are

    python ./projects/des_cluster/scripts/make_cluster_mask.py

and

    python ./projects/des_cluster/scripts/make_synthetic_data.py --threads 3

The two evaluate examples `EXAMPLE_EVALUATE1.yaml` and `EXAMPLE_EVALUATE2.yaml` are written by hand. The script `make_example_files.py` derives every other example from them (the MCMC examples and every `EXAMPLE_EMUL2` file), so the derived files cannot drift apart:

    python ./projects/des_cluster/scripts/make_example_files.py

It reads the emulator and sampler blocks from the projects des_y3 and desy1xplanck, which must be installed.

> [!NOTE]
> A change to the data vector, the n(z), the covariance, the examples, or the likelihood defaults also requires refreshing the snapshot of the unit tests (see [tests/README.md](tests/README.md)).

# Unit tests and validation tools <a name="des_cluster_unit_tests"></a>

The `tests/` folder holds unit tests for the two cluster likelihoods of this
project: they compare each likelihood against stored reference
values, check for race conditions from OpenMP threading, check that
every cached table is rebuilt when one of its parameters moves, and
measure the numerical error of the default accuracy settings. The
tests read nothing from the live project;
[tests/README.md](tests/README.md) describes every test, the tests'
own data snapshot, and how to refresh it.

We assume users are in the Conda cocoa environment from a previous
`conda activate cocoa` command, that the shell is bash, and that the
current folder is the cocoa main folder `cocoa/Cocoa`.

**Step :one:**: activate the private Python environment by sourcing
the script `start_cocoa.sh`

    source start_cocoa.sh

**Step :two:**: run the tests of this project

    python -m pytest ./projects/des_cluster/tests

## Validation tools <a name="des_cluster_validation"></a>

pytest does not collect the three other folders under `tests/`; each is run on its own.

- `tests/reference/` holds an independent Python reference model of the cluster observables and of the joint Gaussian covariance, with its own tests:

      python -m pytest ./projects/des_cluster/tests/reference

- `tests/validation/compare_reference.py` compares the compiled code with the Python reference, quantity by quantity (selection kernels, number density and bias per richness bin, one-halo power spectrum, counts, Limber spectra, real-space statistics, and the cluster blocks of the data vector), and checks that the data vector is bitwise identical at 1 and 8 OpenMP threads:

      python ./projects/des_cluster/tests/validation/compare_reference.py --threads 4

  [PORT_PLAN.md](PORT_PLAN.md) records the result: every cluster quantity matches the reference to $2\times10^{-5}$ or better, except cluster lensing, which differs by $10^{-3}$ in the tail of the lowest source bins behind the highest cluster bin; on the cluster-lensing block this amounts to $\Delta\chi^2 = 3.4\times10^{-5}$ with the scale cuts and $2.0\times10^{-4}$ without them.

- `tests/validation/knob_sweep.py` measures the accuracy settings one at a time with the small scales visible (the source of the `integration_accuracy` numbers above). It reuses the cache of `compare_reference.py`:

      OMP_NUM_THREADS=4 python ./projects/des_cluster/tests/validation/knob_sweep.py \
        --cache-dir <compare_reference cache> --cov ./projects/des_cluster/data/des_cluster_y6_cov.npy

- `tests/lighthouse_reference/` holds data vectors and intermediate quantities computed with the original CosmoLike cluster code (the `lighthouse` repository), kept for comparison. Its [README](tests/lighthouse_reference/README.md) documents the configuration and lists where that code deviates from the model of the paper. The port follows the paper, so this comparison is a sanity check at the percent level: the counts agree to 2.4% and the cluster bias to 0.8% (the original code integrates with a relative tolerance of $10^{-2}$).

## Timing <a name="des_cluster_timing"></a>

`tests/validation/time_cosmolike.py` times the data vector per probe set: after one warm-up, every timed evaluation moves every sampled parameter, so no table is served from a cache, and the timer of the likelihood is read apart from the timer of CAMB.

    OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
      python ./projects/des_cluster/tests/validation/time_cosmolike.py

`tests/validation/time_lighthouse.py` times the original CosmoLike cluster code in the same way; it needs the prebuilt library of that code, which is not part of Cocoa.

The table below is the one of [PORT_PLAN.md](PORT_PLAN.md) (section 7, "Timing, fresh machine"), in seconds per evaluation. **These are cosmolike-only times: CAMB is excluded.** The original code is single-threaded.

| probe set | original CosmoLike code (1 thread) | this code (4 threads) | this code (8 threads) | original / this code at 8 threads |
|---|---|---|---|---|
| 6x2pt + N | 22.64 | 0.152 | 0.108 | 210 |
| 4x2pt + N | 21.46 | 0.131 | 0.084 | 255 |
| 3x2pt | 2.02 | 0.098 | 0.068 | 30 |
| cluster lensing | 19.15 | 0.047 | 0.033 | 580 |
| $w_{cc}$ | 0.91 | 0.024 | 0.021 | 43 |
| $w_{cg}$ | 0.85 | 0.023 | 0.019 | 45 |
| N | 0.11 | 0.013 | 0.012 | 9 |

On the same machine, CAMB takes 0.63 s at 8 threads and 1.2 s at 4 threads, 6 to 8 times the cosmolike time. This is why the MCMC examples with CAMB put the cosmological parameters in the slowest block, and why the hybrid examples of the section [Running Hybrid Cosmolike-ML emulators](#des_cluster_examples_emul2) spend nearly all their time in cosmolike.

# Status and known limitations <a name="des_cluster_status"></a>

The port is complete for the model described above: the two likelihoods run, the compiled code is validated against an independent Python reference, and the unit tests cover both combinations. [PORT_PLAN.md](PORT_PLAN.md) (section 7) holds the detailed status. The items below are open.

- **The data are synthetic.** The data vector is a noiseless model at the fiducial point, and the covariance is analytic Gaussian: it has no block between the counts and the two-point functions, no super-sample covariance, and no trispectrum term. The terms beyond Gaussian and the block between the counts and the two-point functions are deferred to the planned port of CosmoCov to Cocoa.
- **$w_{cc}$ and $w_{cg}$ are Limber only.** The original CosmoLike code computes both without the Limber approximation. At the largest angular bin (225 arcmin), for the first cluster redshift bin and the lowest richness bin, the Limber result differs from the non-Limber one by about -13% for $w_{cc}$ and -16% for $w_{cg}$ ([tests/lighthouse_reference/README.md](tests/lighthouse_reference/README.md)). The keys `cluster_adopt_limber_cc` and `cluster_adopt_limber_cg` exist, but 1 is the only implemented value. Non-Limber $w_{cc}$ and $w_{cg}$ are the first physics item of the backlog.
- **Cluster lensing has NLA intrinsic alignment only.** There is no TATT in the cluster lensing code, and the examples run NLA in every block.
- **Neutrinos in the halo mass function.** The cluster combinations use the cold dark matter + baryon prescription described in [Cluster options](#des_cluster_options) with one massive neutrino state in CAMB, while DES ran three degenerate ones; the two differ by up to 2.6% in the counts at the same $\Omega_\nu h^2$. One state is the convention of the Cosmolike projects. On the emulator path, whose matter power spectrum emulator has no neutrino input, the likelihood approximates the cb spectrum as $P_{lin}/(1 - f_\nu)^2$.
- **Emulated Boltzmann inputs.** The emulators of the EMUL2 examples are fixed at a neutrino mass of 0.06 eV. At equal neutrino mass they differ from CAMB by $\Delta\chi^2 = 1.6$ on 6x2pt + N (the cluster counts differ by 1.5% in the median).
- **The Y1 switches are not validated end to end.** The options that reproduce the DES Y1 choices (`cluster_selection_model: 1`, `cluster_ytransform: 0`, and the magnification and intrinsic-alignment switches) exist, but only the defaults have been validated.
- **No parameter-recovery chain yet.** An MCMC that recovers the input parameters from the synthetic data is still open, as is an independent review of the model against the paper.
- **Thread scaling.** Going from 4 to 8 threads gains only a factor of 1.4 on 6x2pt + N; the cause has not been measured. Two optimizations of the cluster tables are left (the one-halo table, and the $w_{cc}$ and $w_{cg}$ spectra tables).
- **Angular binning in the compiled interface.** cosmolike caches the bin-averaged Legendre kernels by the number of angular bins and the table key, not by the angular range, so a call to `init_binning` with a new range and the same number of bins returns the values of the old range. The notebook wrappers avoid this by drawing a new table key on every call.
- **No response functions for the cluster blocks.** The data-vector plotting functions of `cosmolike_notebook_utils` have a cluster version (`plot_datavectors_cluster.py`); the response helpers (`plot_response.py`) do not.
- **No tagged release.** Cocoa pins this project to the branch `main`.
