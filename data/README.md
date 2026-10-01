# des_cluster data

> [!Warning]
> **The cluster data in this folder are synthetic. They are not DES data.** The data vector is the model of this code at the fiducial point of Table I of arXiv:2503.13631, without noise. The covariance is an analytic Gaussian covariance, and its block between the cluster counts and the two-point functions is zero. No chain run on these files is a DES result.

This folder holds two independent data sets: the synthetic DES Y6-like data set of the cluster analysis, read by the likelihoods `combo_4x2pt_N` and `combo_6x2pt_N`, and a DES Y3 3x2pt placeholder data set, read only by the galaxy-only likelihoods `cosmic_shear`, `combo_3x2pt`, and `combo_2x2pt`. This file is a map of the folder. The section [Synthetic data](../README.md#des_cluster_data) of the project README describes the cluster data set and its scale cuts in full.

## The synthetic cluster data set

The data set is Y6-like in the following sense: the lens and source redshift distributions are the n(z) files of the DES Y6 code comparison kept in the repository of the original CosmoLike cluster code (`lighthouse`), the binning and the scale cuts are those of arXiv:2503.13631, the parameters are those of its Table I, and the survey area is 4143 deg$^2$ (the DES Y3 cluster footprint of arXiv:2503.13632).

| file | content |
|---|---|
| `des_cluster_y6.dataset` | the base dataset descriptor: file names, six lens and four source bins, 20 angular bins between 2.5 and 250 arcmin, three cluster redshift bins ($z_\lambda$ edges 0.2, 0.4, 0.55, 0.65), four richness bins ($\lambda$ edges 20, 30, 45, 60, 500), the survey area, and the lens bin paired with each cluster redshift bin in $w_{cg}$ (`cg_lens_bins`). The likelihood defaults and the examples name the two descriptors of the next row, not this file. Its `mask_file` key names the 6x2pt + N mask, so this file loaded on its own is the 6x2pt + N data set |
| `des_cluster_y6_4x2ptN.dataset`, `des_cluster_y6_6x2ptN.dataset` | the descriptors that `combo_4x2pt_N` and `combo_6x2pt_N` load: the base descriptor (`DEFAULT(des_cluster_y6.dataset)`) with the mask of each combination |
| `des_cluster_y6.datavector` | the joint synthetic data vector, shared by the two combinations (2812 entries; columns: index, value) |
| `des_cluster_y6_cov.npy` | the joint covariance of the 2812 entries, shared by the two combinations: the upper triangle with the diagonal, row by row (`C[numpy.triu_indices(2812)]`, zeros included), as a binary little-endian float64 NumPy array of 3,955,078 entries (32 MB). cosmolike reads the file directly, and `read_covariance` in `../scripts/make_synthetic_data.py` unpacks it into the full matrix. Until 2026-10-01 the covariance was a text table (columns: i, j, value) of 124 MB holding the same numbers |
| `des_cluster_y6_4x2ptN.mask`, `des_cluster_y6_6x2ptN.mask` | the scale-cut masks of the two combinations, one entry per data point (columns: index, 0 or 1). The 4x2pt + N mask keeps 889 entries and masks `ss` and `gs` entirely; the 6x2pt + N mask keeps 1429 |
| `des_y6_maglim.nz`, `des_y6_source.nz` | redshift distributions of the six MagLim lens bins and of the four source bins. The z column holds left bin edges (`photoz_zmid_convention: 0`) |
| `des_y6_cluster.nz` | the selection kernels $\langle\phi_i\vert z\rangle$ of the three cluster redshift bins: the probability that a cluster at true redshift z lands in each $z_\lambda$ bin. It is a probability, not a normalized distribution, and it is sampled at z (no half-cell offset) |
| `y3_redmapper_counts.txt` | the counts of the public DES Y3 redMaPPer catalog per redshift and richness bin. The model does not read it: `make_synthetic_data.py` compares the model counts with it, and `make_cluster_mask.py` reads it for an optional cut on $w_{cc}$ that is off by default |

The data vector has seven blocks, stored in the order `ss`, `gs`, `gg`, `cg`, `N`, `cc`, `cs` (400, 480, 120, 240, 12, 600, and 960 entries):

- `ss`: cosmic shear, $\xi_+$ then $\xi_-$, [source pair][$\theta$];
- `gs`: galaxy-galaxy lensing $\gamma_t$, [(lens, source) pair][$\theta$];
- `gg`: galaxy clustering $w_{gg}$, [lens bin][$\theta$];
- `cg`: cluster x galaxy clustering $w_{cg}$, [(cluster z bin, lens bin) pair][richness bin][$\theta$];
- `N`: cluster counts, [cluster z bin][richness bin];
- `cc`: cluster clustering $w_{cc}$, [cluster z bin][richness pair][$\theta$];
- `cs`: cluster lensing, [(cluster z bin, source bin) pair][richness bin][$\theta$]. The block holds $\Sigma = Y\gamma_t$ (the Y transform of eq. 15 of arXiv:2503.13631), not $\gamma_t$; its last angular bin vanishes by construction and is always masked.

The covariance is Gaussian. For the two-point blocks it is the full-sky, bin-averaged Gaussian term with shape and shot noise, with the Y transform applied to the cluster-lensing rows. For the counts it is Poisson plus sample variance. The block between the counts and the two-point functions is zero, and the covariance has no super-sample and no trispectrum term. The data vector and the covariance were made with the default cluster options of the two likelihoods, so a run that changes one of those switches evaluates a model that differs from the one behind the data.

## The DES Y3 3x2pt placeholder data set

The files `des_y3_real.dataset`, `des_y3_unblinded_final.txt`, `des_y3_cov_unblinded_final.txt`, `des_y3_lens.nz`, `des_y3_source.nz`, `3x2pt_baseline.mask`, `ones.mask`, `pca.txt`, and `baryons_logPkR.h5` are a copy of the DES Y3 3x2pt data set of the project des_y3: the unblinded data vector and covariance ($\xi_\pm$, $\gamma_t$, $w(\theta)$; redMaGiC lens sample with 5 bins, Metacalibration source sample with 4 bins, 20 angular bins between 2.5 and 250 arcmin; 900 entries), the 3x2pt baseline mask and an all-ones mask, the matching n(z), and the baryon PCA and simulation files.

The descriptor `des_y3_real.dataset` points the galaxy-only likelihoods of this project (`cosmic_shear`, `combo_3x2pt`, `combo_2x2pt`) at these files. The cluster likelihoods do not use them; the examples `EXAMPLE_EVALUATE3.yaml` (cosmic shear) and `EXAMPLE_EVALUATE4.yaml` (3x2pt) and the unit tests of the galaxy-only likelihoods do. The cluster descriptor also names `pca.txt` and `baryons_logPkR.h5`, but the cluster likelihoods refuse baryon PCAs, and with their default options they read neither file.

## Git LFS

Three files of this folder are stored with Git LFS (the patterns are in `../.gitattributes`): `des_cluster_y6_cov.npy`, `des_y3_cov_unblinded_final.txt`, and `baryons_logPkR.h5`. A clone without `git lfs pull` holds a pointer file in place of each.

## Regenerating the files

The scripts are in `../scripts`, and each documents its options in its header. The section [Regenerating the data](../README.md#des_cluster_data_scripts) of the project README gives the order of the steps and the commands. In that order: `make_des_y6_nz.py` writes `des_y6_maglim.nz` and `des_y6_source.nz`; `make_cluster_zdist.py` writes `des_y6_cluster.nz`; `make_y3_redmapper_counts.py` writes `y3_redmapper_counts.txt`; `make_cluster_mask.py` writes the two masks; `make_synthetic_data.py` writes `des_cluster_y6.datavector` and `des_cluster_y6.cov`, and asserts $\chi^2 < 10^{-6}$ at the fiducial point for both combinations.

The unit tests keep their own copy of the cluster data set under `../tests/frozen/data`, so a change to a file of this folder also requires refreshing that snapshot (see [tests/README.md](../tests/README.md)).
