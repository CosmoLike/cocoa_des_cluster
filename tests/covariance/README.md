# Covariance tests

These checks exercise this project's galaxy/shear forecast adapter and
compiled covariance bindings. They use the project's redshift files and
catalog inputs, with small numerical grids and three measured rows.
Separate count-response tests use analytic shell integrals and do not
require a selected halo-population model.

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

The [shared component tests](../../../lsst_y1/tests/covariance/README.md)
contain independent algebra and projection references. This project check
covers its binding and inputs; it does not duplicate those references.
Use the [covariance notebook](../../EXAMPLE_EVALUATE_COVARIANCE.ipynb)
for full matrices and numerical refinement. Subset positivity alone does
not establish full-matrix positivity or parameter-error convergence.

`test_counts.py` checks the supplied-abundance component. It does not
validate a cluster mass–richness relation or generate the complete
cluster $`6\times2\mathrm{pt}+N`$ covariance. In particular, the non-SSC
count–spectrum contribution requires its own physical calculation.

The [data-vector tests](../data_vector/README.md) check likelihood signals
and stored reference values separately. No stored covariance or reference
value is replaced by this test.
