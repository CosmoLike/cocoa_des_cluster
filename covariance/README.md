# Table of contents

1. [Overview](#overview)
2. [Running the covariance notebook](#running)
3. [Changing the covariance accuracy](#accuracy)
4. [Reading the figures](#figures)
5. [Running the tests](#tests)
6. [Files](#files)
7. [Cluster 6x2pt + counts](#joint)
8. [Appendix](#appendix)
   1. [FAQ: Which survey does the example use?](#survey)
   2. [FAQ: What does the calculation include?](#gaussian)
   3. [FAQ: How can users check convergence?](#convergence)
   4. [FAQ: How can users reuse the calculation?](#reuse)

# Overview <a name="overview"></a>

[EXAMPLE_EVALUATE_COVARIANCE.ipynb](../EXAMPLE_EVALUATE_COVARIANCE.ipynb)
computes real-space and Fourier-space galaxy/shear covariances with
separate Gaussian (G), super-sample (SSC), connected non-Gaussian (cNG)
and total matrices. The real-space vector contains cosmic shear,
galaxy–galaxy lensing and galaxy clustering, using 6 lens and 4
source bins. The Fourier example contains E-mode shear, galaxy–shear
and galaxy-density bandpowers.

The final section computes the angular cluster $`6\times2\mathrm{pt}+N`$
forecast, including all 12 counts and 2,800 two-point entries, under the
[cluster approximation described below](#joint).

The galaxy/shear section runs CAMB once for both spaces and several
accuracy boosts. The cluster section has its own recorded initialization.
Both report positivity and refinement diagnostics, plot the physical
components, and save NumPy archives. Neither loads the likelihood's
supplied covariance.

> [!NOTE]
> The forecast uses massless neutrinos, linear galaxy bias, zero IA,
> magnification and RSD, Limber spectra and a spherical-cap footprint.
> It includes isotropic halo SSC and five halo cNG terms. These choices
> need numerical and physical validation for the intended inference.
> See [what the calculation includes](#gaussian).

# Running the covariance notebook <a name="running"></a>

We assume Cocoa and the DES cluster galaxy–shear block project are installed, users have run
`conda activate cocoa`, the shell is Bash, and the current folder is
`cocoa/Cocoa`. The notebook uses the Python environment activated by Cocoa.

**Step :one:**: activate Cocoa's private Python environment.

    source start_cocoa.sh

**Step :two:**: compile the DES cluster galaxy–shear block interface, including the covariance components.

    unset IGNORE_COSMOLIKE_des_cluster_CODE
    source ./projects/des_cluster/scripts/compile_des_cluster.sh

**Step :three:**: start Jupyter.

    jupyter notebook --no-browser --port=8888

**Step :four:**: open the URL printed by Jupyter and select
`projects/des_cluster/EXAMPLE_EVALUATE_COVARIANCE.ipynb`.

**Step :five:**: select **Kernel → Restart Kernel and Run All Cells**.

The notebook computes the real-space and Fourier matrices for accuracy
boosts 1 and 2. It prints matrix dimensions, positivity diagnostics and
changes relative to the highest tested boost, then displays the figures.
The final cell saves these files in `projects/des_cluster/covariance/`:

| Output | Contents |
| --- | --- |
| `forecast_real.npz` | Angular G, SSC, cNG, total, row map, means and resolved settings. |
| `forecast_fourier.npz` | Fourier G, SSC, cNG, total, row map, means and resolved settings. |
| `forecast_camb.npz` | CAMB tables used for the calculation. |
| `forecast_cluster.npz` | Joint angular components, count means, row positions, Y null-mode selection and explicit model limits. |
| `forecast_cluster_camb.npz` | CAMB tables used for the separate cluster initialization. |

Rerunning the final cell replaces these computed output files.

> [!NOTE]
> The notebook assigns eight threads to CosmoLike's OpenMP loops and
> one thread to BLAS. Change `ci.set_omp_threads(n=8)` in the notebook
> if fewer cores are available. Run one calculation at a time when
> measuring execution time.

> [!TIP]
> To inspect the forecast inputs before running CAMB, see
> [which survey the example uses](#survey).

# Changing the covariance accuracy <a name="accuracy"></a>

We assume users have run `conda activate cocoa`, use Bash, and are in
`cocoa/Cocoa`. The interface must already be compiled.

**Step :one:**: activate Cocoa's private Python environment.

    source start_cocoa.sh

**Step :two:**: start Jupyter and open
`projects/des_cluster/EXAMPLE_EVALUATE_COVARIANCE.ipynb`.

    jupyter notebook --no-browser --port=8888

**Step :three:**: set the first calculation's accuracy in its configuration cell.

```python
boosts = [1, 2]
settings = survey.configuration(accuracy_boost=boosts[0])
```

`accuracy_boost` is the single user control. Supported values are
1, 2, 4 and 8. It raises the covariance's multipole cutoffs and radial,
angular, halo and lensing-window integration resolution together. It also
refines the non-Gaussian multipole table. It leaves
CAMB and data-vector accuracy settings unchanged.

**Step :four:**: choose which values to compare in the refinement cell.

```python
boosts = [1, 2, 4]
```

For the more expensive comparison, include boost 8:

```python
boosts = [1, 2, 4, 8]
```

**Step :five:**: restart the kernel and run all cells.

The calculation keeps the cosmology, galaxy distributions, noise, angular
bins and Fourier-band endpoints fixed. Only the covariance accuracy changes. The highest
boost in the comparison supplies the reference matrix for the difference
plots and variance-ratio table.

> [!NOTE]
> A larger boost is a numerical resolution, not a guaranteed survey
> accuracy. Boost 1 is a teaching example. Boost 8 uses more modes and
> quadrature nodes and can require substantially more memory and time.
> See [how to check convergence](#convergence).

# Reading the figures <a name="figures"></a>

| Figure | What it teaches |
| --- | --- |
| Split-triangle correlation matrix | Compare the initial calculation in the lower triangle with the highest tested boost in the upper triangle. Each matrix is normalized by its own diagonal. |
| G, SSC and cNG maps and histograms | Compare each component after normalization by the total diagonal variances. |
| Error changes | Compare first-source-bin standard deviations with the reference, in percent, for angles and Fourier bands. |
| Generalized-mode report | Bound variance changes over every linear combination of measurements. |

The correlation comparison follows the layout of
[Friedrich et al. (2021), Fig. 6](https://arxiv.org/abs/2012.08568).
The component maps and histogram adapt
[Barreira, Krause & Schmidt (2018), Fig. 1](https://arxiv.org/abs/1807.04266).
These figures display the notebook's calculation, not data from the papers.

Negative correlations remain visible. A grey cell in an element-ratio
map means its denominator is zero or too small for the selected cutoff.
No plot clips eigenvalues or adjusts the covariance.

# Running the tests <a name="tests"></a>

We assume users have run `conda activate cocoa`, use Bash, and are in
`cocoa/Cocoa`, with the DES cluster galaxy–shear block interface compiled.

**Step :one:**: activate Cocoa's private Python environment.

    source start_cocoa.sh

**Step :two:**: run the covariance tests.

    python -m pytest projects/des_cluster/tests/covariance

The tests check this project's input layout, real/Fourier assembly,
thread repeatability, subset positivity and saved metadata. The [covariance test guide](../tests/covariance/README.md)
describes each check.

For the ordinary data-vector tests, we again assume the activated Conda
Cocoa environment, Bash, and the current folder `cocoa/Cocoa`.

**Step :one:**: activate Cocoa's private Python environment.

    source start_cocoa.sh

**Step :two:**: run only the data-vector tests.

    python -m pytest projects/des_cluster/tests/data_vector

These tests compare likelihood predictions with the stored references.
The [data-vector test guide](../tests/data_vector/README.md) explains them.

# Files <a name="files"></a>

| File or folder | Purpose |
| --- | --- |
| [EXAMPLE_EVALUATE_COVARIANCE.ipynb](../EXAMPLE_EVALUATE_COVARIANCE.ipynb) | Run, refine and plot real/Fourier G, SSC and cNG matrices. |
| [des_cluster_covariance.py](des_cluster_covariance.py) | Specify survey inputs, initialize this project's interface and call the shared calculation. |
| [des_cluster_joint_covariance.py](des_cluster_joint_covariance.py) | Add the cluster selections, richness relation, joint row ordering and Y convention. |
| [Shared covariance package](../../../external_modules/code/cosmolike_core/cosmolike_notebook_utils/covariance/README.md) | Reuse integration preparation, Gaussian assembly, halo inputs, accuracy settings and diagnostics. |
| [Shared plotting script](../../../external_modules/code/cosmolike_core/cosmolike_notebook_utils/plot_covariances.py) | Plot covariance arrays from any project. |
| [Covariance C files](../../../external_modules/code/cosmolike_core/cosmolike/covariances/README.md) | Read the physics and each compiled component's role. |

# Cluster 6x2pt + counts <a name="joint"></a>

We assume the environment and interface are prepared using the
[notebook instructions](#running), and the notebook kernel is active.

**Step :one:**: reach the notebook section **Cluster 6x2pt + counts** and
inspect its physical choices. It uses `des_y6_cluster.nz`, three observed
redshift bins, and richness edges 20, 30, 45, 60 and 500. The lognormal
mass–richness relation is fixed to the values in the adapter. Its
environmental selection correction is zero; this is a forecast assumption.

**Step :two:**: choose the accuracy comparison with one control.

```python
cluster_boosts = [1, 2]
cluster_settings = joint_survey.configuration(
    accuracy_boost=cluster_boosts[0], ytransform=True,
)
```

The same boost refines the common matter calculation and selected-halo
response integration. Increasing it does not add missing physical terms.

**Step :three:**: run the calculation and inspect the printed model limits.
The matrix order is shear–shear, galaxy–shear, galaxy clustering,
cluster–galaxy, counts, cluster clustering, cluster lensing. Richness is
the fastest cluster-category index. All internal cross-redshift spectra
remain in Gaussian pairings, including pairs absent from measured rows.

The Gaussian component includes count Poisson noise. SSC uses common
long-mode shells for all counts and two-point functions, including the
fixed-profile abundance response of the selected cluster's own halo.
Density contrasts are normalized by observed catalog means; absolute
counts are not. The connected two-point term approximates clusters as
linearly biased tracers of the matter trispectrum. **Non-SSC count–spectrum
cross terms and selected-cluster one-halo cNG corrections are omitted.**
This is a complete matrix within these stated approximations, not a
validated replacement for the likelihood covariance. For the physical
decomposition, see [Krause & Eifler, Appendix A](https://arxiv.org/abs/1601.05779)
and [Schaan, Takada & Spergel, Eq. 35](https://arxiv.org/abs/1406.3330).

The DES Y6 model also uses non-Limber spectra in its covariance to retain
correlations between different redshift bins. The example's Limber-only
calculation therefore does not yet reproduce that DES prescription. See
[the DES Y6 modeling paper, Appendix F](https://arxiv.org/html/2503.13631v1).
The omitted selected-cluster and non-SSC count terms describe extensions
of the general halo model; their listing does not mean DES included each
one. DES selection-bias modeling is also distinct from the response of a
selection probability to a long-wavelength environmental fluctuation.

**Step :four:**: check the localized matrix on `valid_indices`.

```python
keep = joint["valid_indices"]
matrix = joint["total"][np.ix_(keep, keep)]
diagnostic = cov.covariance_modes(matrix=matrix)
print(diagnostic["positive_definite"])
```

The project uses $`Y(R)=\Sigma(R)-\Sigma(R_{\max})`$ for cluster lensing.
The mean changes by an angular matrix $`A`$, and every covariance component
changes as $`ACA^{\mathsf T}`$, including count–lensing cross blocks.
The last angular value of each of the 48 cluster-lensing rows is exactly
zero. `valid_indices` removes those defined null rows only. It does not
apply physical scale cuts or discard numerically negative modes. See
[Park, Rozo & Krause](https://arxiv.org/abs/2004.07504) and
[the DES covariance conventions, Sec. II.4](https://arxiv.org/abs/2503.13631).

**Step :five:**: compare boosts and inspect the correlation/component plots.
The notebook saves the highest tested boost with all row positions,
settings and omitted physics in `forecast_cluster.npz`. Its CAMB tables
are saved separately. These files remain separate from the likelihood's
supplied covariance. Fisher convergence and the effect of omitted terms
must be assessed before inference.

# Appendix <a name="appendix"></a>

## FAQ: Which survey does the example use? <a name="survey"></a>

The redshift files are `des_y6_maglim.nz` and `des_y6_source.nz` in `data/`.
Each column supplies a bin's radial shape. The catalog densities are
specified separately in [des_cluster_covariance.py](des_cluster_covariance.py).

The example uses the six MagLim lens and four source distributions,
number densities and shape dispersion adopted by this project's
[synthetic-data generator](../scripts/make_synthetic_data.py).
The quadrature shape dispersion 0.384666 is divided by $`\sqrt{2}`$
to obtain the per-component input.
These galaxy and shear blocks are part of the joint analysis described by
[DES, arXiv:2503.13631](https://arxiv.org/abs/2503.13631).
The final angular section adds cluster counts, cluster clustering,
cluster–galaxy correlations and cluster lensing with the assumptions
listed in the [joint forecast guide](#joint). The Fourier example remains
the galaxy/shear block. The angular bins follow the joint dataset.

| Survey input | Example choice |
| --- | --- |
| Area | 4143 square degrees |
| Lens bins | 6 |
| Source bins | 4 |
| Angular bins | 20 logarithmic bins, 2.5–250 arcminutes |
| Integer Fourier bands | 15 bands, multipoles 30–4000 |
| Real-space vector | 1,000 entries |
| Fourier vector | 600 entries |
| Joint angular vector | 2,812 entries before removing 48 defined Y null rows |
| Neutrino mass | Zero |
| Intrinsic alignment | Zero |
| Photo-z shifts | Zero |
| Magnification | Zero |

The adapter lists every density, shape dispersion, galaxy bias and
cosmological input. The notebook prints those settings and archives them
with each matrix. The examples do not replace a supplied likelihood matrix.

## FAQ: What does the calculation include? <a name="gaussian"></a>

For Gaussian fields, a four-point expectation separates into products
of two-point expectations. A covariance between measured spectra AB and
CD therefore needs AC, BD, AD and BC spectra, even if those crossed pairs
are excluded from the measured data vector. The example retains the
complete field matrix before assembling the measured rows.

The signal uses nonlinear matter power and Limber projection. Spherical
spin operators average the resulting angular spectra over each angular
bin. The signal covariance uses the $`f_{\rm sky}`$ approximation.

Pure white noise extends to arbitrarily high multipoles. The calculation
replaces that infinite sum with the analytic pair-noise expression using
a spherical-cap footprint. Signal and mixed signal-noise terms retain
the finite multipole sum. A cap correction to noise pair counts does not
make the signal covariance an exact treatment of an irregular footprint.

Fourier bandpowers include pure noise inside the finite band sum. They
average the core spectra directly. Real-space means retain Cocoa's
extra source-leg factor for matching its angular-transform convention.
Each convention is applied consistently to G, SSC and cNG signal terms;
white noise receives neither source conversion.

SSC describes correlations induced by modes larger than the survey;
cNG describes the connected four-point contribution inside it.
The notebook computes both terms for every measured cross block. SSC
uses common shell responses before their weighted outer products; cNG
uses the 1-halo, 2-halo (1+3), 2-halo (2+2), 3-halo and 4-halo terms.
These are approximations, not a simulation calibration. See
[Krause & Eifler](https://arxiv.org/abs/1601.05779) and
[Takada & Hu](https://arxiv.org/abs/1302.6994) for the physical decomposition.

## FAQ: How can users check convergence? <a name="convergence"></a>

Compare successive boosts with the same physical inputs. The notebook
prints the largest departure of the generalized covariance eigenvalues
from one. They solve

```math
C_b v = \lambda C_{\rm ref} v.
```

The smallest and largest eigenvalues bound the variance ratio for any
linear combination of measurements. This checks coupled directions that
a diagonal comparison alone can miss. Raw largest covariance eigenvalues
do not identify the most informative cosmological modes.

Every total matrix must have nonnegative variance in every direction;
an invertible total must be positive definite. A positive diagonal alone
is insufficient. The notebook reports eigenvalues without repairing them.

The reference boost is only the highest resolution tested. Establish its
stability with further refinement, then assess marginalized parameter
errors and the Fisher Figure of Merit for the intended survey. The
likelihood's data-vector $`\lvert\Delta\chi^2\rvert<0.2`$ rule is not a
covariance convergence requirement.

## FAQ: How can users reuse the calculation? <a name="reuse"></a>

A project supplies its own initialized interface, redshift files, survey
numbers and physical choices. The shared Python package prepares arrays
and asks the covariance C components to perform the integration and
projection. Copy the small survey adapter, not those algorithms.

Other project interfaces must link the same covariance sources before
running these examples. Cluster observables need their own physical
model; changing the number of galaxy bins does not create a cluster
covariance.

Rectangular projection inputs allow Python to request matrix subblocks.
The C routines use OpenMP inside one process and never start MPI work.
A future Python dispatcher can distribute those subblocks while keeping
all cross correlations in the assembled matrix.
