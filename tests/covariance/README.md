# Covariance tests

These checks exercise this project's galaxy/shear forecast adapter and
compiled covariance bindings. They use the project's redshift files and
catalog inputs, with small numerical grids and three measured rows.
Separate count-response tests use analytic shell integrals and do not
require a selected halo-population model. Cluster spectrum tests project
supplied windows and profiles, independently of the survey model tables.

We assume Cocoa is installed, users have run `conda activate cocoa`, the
shell is Bash and the current folder is `cocoa/Cocoa`.

**Step :one:**: activate Cocoa's private Python environment.

    source start_cocoa.sh

**Step :two:**: compile this project's interface.

    unset IGNORE_COSMOLIKE_des_cluster_CODE
    source projects/des_cluster/scripts/compile_des_cluster.sh

**Step :three:**: run the covariance checks.

    python -m pytest projects/des_cluster/tests/covariance

| Check | Purpose |
| --- | --- |
| Project layout | Verify the full angular and Fourier vector lengths from the measured row map. |
| Accuracy refinement | Keep measurement bins fixed while increasing quadrature resolution. |
| Real-space components | Check finite, symmetric G, SSC, cNG and total matrices for a measured subset. |
| Fourier components | Check the same properties for bandpowers. |
| Thread repeatability | Compare every component bitwise with one and eight OpenMP threads. |
| Positive total | Check variance positivity for every direction of the tested subset. |
| Output archive | Read arrays and resolved survey metadata without pickle. |
| Count shell volumes | Compare mean counts and count SSC with closed polynomial integrals. |
| Counts–two-point SSC | Check the cancellation of radial distance factors and shared-mode positivity. |
| Units and transforms | Change the length unit and transform the two-point observable while preserving the covariance. |
| Count component boundaries | Check odd array lengths, one/eight-thread repeatability, owned outputs and rejected invalid inputs. |
| Cluster one-halo projection | Verify that the own-halo profile contributes only to cluster lensing, without an extra halo bias. |
| Cluster biased-power projection | Compare all cluster–galaxy and cluster–cluster pairs with closed radial integrals. |
| Cluster spectrum contraction | Compare supplied tables with independent NumPy sums, including odd node and pair counts. |
| Cluster spectrum units | Change length units while preserving all angular spectra. |
| Cluster spectrum threading | Check bitwise repeatability at one, two, four and eight threads. |
| Cluster spectrum boundaries | Check owned outputs and rejection of malformed shapes, domains and richness indices. |
| Selected mass moments | Compare one-, two- and three-profile integrals with closed polynomials and independent NumPy sums. |
| Selection probabilities | Check that membership occurs once per shared halo and that a partition of categories recovers the unselected integral. |
| Mass-moment units and threading | Check distinct length dimensions, signed profiles, odd/even grids, bitwise 1/2/4/8-thread results, ownership and input guards. |
| Non-SSC count–matter terms | Compare closed radial integrals and exact Poisson-count expectations, including the two-halo factor of two and cancellation of survey volume. |
| Cluster-lensing localization | Check exact angular polynomials, every joint cross block, known last-bin null modes and bitwise 1/2/4/8-thread propagation. |
| Physical halo samples | Compare the covariance mass samples with scalar halo readers and the fixed-amplitude Tinker formula, including both HMF modes and the redshift-fit boundary. |
| Joint survey preparation | Check the DES count insertion, richness ordering, absolute versus normalized windows and every field spectrum across memory-block boundaries. |

The [shared component tests](../../../lsst_y1/tests/covariance/README.md)
contain independent algebra and projection references. This project check
covers its binding and inputs; it does not duplicate those references.
Use the [covariance notebook](../../EXAMPLE_EVALUATE_COVARIANCE.ipynb)
for full matrices and numerical refinement. Subset positivity alone does
not establish full-matrix positivity or parameter-error convergence.

`test_counts.py` checks the supplied-abundance component. It does not
validate a cluster mass–richness relation or generate the complete
cluster $`6\times2\mathrm{pt}+N`$ covariance.
`test_spectra_cluster.py` checks the all-pairs Limber projection. Its
mean-spectrum model and finite input tables do not certify a full
cluster covariance or its survey accuracy.
`test_moments_cluster.py` checks selected mass integrals with explicit
input profiles and weights. It does not choose the mass function,
selection response or angular normalization of a full cluster forecast.
`test_counts_matter_cross.py` checks the non-SSC count–matter cross block;
it does not cover discrete cluster partners or a full joint generator.
`test_transform_cluster.py` checks the propagation of the mean model's
angular localization through a supplied covariance. It retains the known
zero final angular bin and does not replace the physical scale selection.
`test_halo_samples_cluster.py` checks covariance-owned sampling of the
initialized halo and richness model. It shares the physical halo readers
with production and therefore tests sampling and normalization, rather
than independently calibrating those fits.
`test_survey_cluster.py` checks the shared cluster preparation helpers with
analytic catalog inputs and independent NumPy projection. The smaller
layout check verifies that excluding a measured galaxy–shear row does not
change the internal field identities.

The [data-vector tests](../data_vector/README.md) check likelihood signals
and stored reference values separately. No stored covariance or reference
value is replaced by this test.
