# Likelihoods and their parameters <a name="des_cluster_likelihood_readme"></a>

This folder holds the likelihoods of the project (one yaml file with the defaults and one python file per likelihood), the parameter files the likelihoods include, and the base class `_cosmolike_prototype_base.py`. The model, the options, and the fiducial values of the nuisance parameters are described in the project [README](../README.md#des_cluster_likelihood); this page says how the parameters are organized and which ones each likelihood must not vary.

1. [The likelihoods](#des_cluster_likelihoods)
2. [The parameter files and `fixed_params`](#des_cluster_parameter_files)
3. [The masks](#des_cluster_masks)
4. [What each combination fixes](#des_cluster_fixed_params)
5. [Parameters that should not be varied](#des_cluster_not_varied)
6. [Changing the mask, the scale cuts, or the probes](#des_cluster_changing)

# The likelihoods <a name="des_cluster_likelihoods"></a>

| likelihood | blocks | data set | parameter files |
|---|---|---|---|
| `cosmic_shear` | `ss` | `des_y3_real.dataset` | `params_source.yaml` |
| `combo_2x2pt` | `gs`, `gg` | `des_y3_real.dataset` | `params_source.yaml`, `params_lens_redmagic.yaml` |
| `combo_3x2pt` | `ss`, `gs`, `gg` | `des_y3_real.dataset` | `params_source.yaml`, `params_lens_redmagic.yaml` |
| `combo_4x2pt_N` | `gg`, `cg`, `N`, `cc`, `cs` | `des_cluster_y6_4x2ptN.dataset` | `params_source_y6.yaml`, `params_lens_maglim.yaml`, `params_cluster.yaml` |
| `combo_6x2pt_N` | `ss`, `gs`, `gg`, `cg`, `N`, `cc`, `cs` | `des_cluster_y6_6x2ptN.dataset` | `params_source_y6.yaml`, `params_lens_maglim.yaml`, `params_cluster.yaml` |

The blocks are cosmic shear (`ss`), galaxy-galaxy lensing (`gs`), galaxy clustering (`gg`), cluster x galaxy clustering (`cg`), cluster counts (`N`), cluster clustering (`cc`), and cluster lensing (`cs`). The three galaxy likelihoods run on the DES Y3 files of the `data` folder; the two cluster likelihoods run on the synthetic Y6-like data vector, and differ only in the mask. Every likelihood computes only the blocks it holds: the entries of the other blocks are zero in the theory vector and removed from the covariance.

# The parameter files and `fixed_params` <a name="des_cluster_parameter_files"></a>

There is one parameter file per sample, shared by every likelihood that uses the sample:

| file | sample | parameters |
|---|---|---|
| `params_source.yaml` | DES Y3 sources, 4 bins | photo-z shift `DES_DZ_S*`, shear calibration `DES_M*`, intrinsic alignment `DES_A1_*`, `DES_A2_*`, `DES_BTA_*`, baryon PC amplitudes `DES_BARYON_Q*` |
| `params_source_y6.yaml` | Y6-like sources, 4 bins | the same families, with the Table I priors of arXiv:2503.13631 |
| `params_lens_redmagic.yaml` | DES Y3 redMaGiC lenses, 5 bins | photo-z shift `DES_DZ_L*`, photo-z stretch `DES_DZ2_L*`, linear bias `DES_B1_*`, nonlinear bias `DES_B2_*`, magnification `DES_BMAG_*`, point mass `DES_PM*` |
| `params_lens_maglim.yaml` | Y6-like MagLim lenses, 6 bins | the same families |
| `params_cluster.yaml` | clusters | mass-observable relation and selection bias `DES_CL_*` |

Each likelihood yaml includes its parameter files with cobaya's `!defaults` tag:

    [Adapted from likelihood/combo_4x2pt_N.yaml]
    params: !defaults [params_source_y6, params_lens_maglim, params_cluster]

The tag builds the whole `params` mapping from the files, so the likelihood yaml cannot change one of its entries. A combination that fixes some of these parameters lists them in its `fixed_params` block instead:

    [Adapted from likelihood/combo_4x2pt_N.yaml]
    fixed_params:
      DES_B1_4:
        value: 1.62
      (...)
      DES_PM6:
        value: 0.0

The base class applies this block to the defaults before cobaya merges them with the user's yaml. Each entry replaces the entry of the parameter file with cobaya's merge rule: a `value` drops the `prior`, the `ref`, and the `proposal`, and the `latex` label stays. The parameter stays declared, as a constant. The two cluster combinations therefore share the parameter files and differ only in their `fixed_params`.

# The masks <a name="des_cluster_masks"></a>

The layout of the joint Y6-like data vector (2812 entries) and the entries each cluster mask keeps per block are in the table of the project [README](../README.md#des_cluster_overview). Per MagLim lens bin, the masks keep these entries (`cg` pairs cluster redshift bin $i$ with lens bin $i$, so it has no entry for lens bins 4-6):

| mask | block | kept, lens bin 1 | kept, lens bin 2 | kept, lens bin 3 | kept, lens bin 4 | kept, lens bin 5 | kept, lens bin 6 |
|---|---|---|---|---|---|---|---|
| `des_cluster_y6_4x2ptN.mask` | `gs` | 0 | 0 | 0 | 0 | 0 | 0 |
| `des_cluster_y6_4x2ptN.mask` | `gg` | 9 | 10 | 12 | 0 | 0 | 0 |
| `des_cluster_y6_4x2ptN.mask` | `cg` | 36 | 44 | 48 | 0 | 0 | 0 |
| `des_cluster_y6_6x2ptN.mask` | `gs` | 40 | 48 | 52 | 56 | 56 | 60 |
| `des_cluster_y6_6x2ptN.mask` | `gg` | 9 | 10 | 12 | 12 | 13 | 13 |
| `des_cluster_y6_6x2ptN.mask` | `cg` | 36 | 44 | 48 | 0 | 0 | 0 |

The 4x2pt + N mask removes every entry of lens bins 4-6; the 6x2pt + N mask keeps entries of every lens bin. Both keep entries of every source bin (in `cs`, and in `ss` and `gs` for 6x2pt + N) and of every cluster bin. The DES Y3 mask of the galaxy likelihoods (`3x2pt_baseline.mask` of `des_y3_real.dataset`) keeps entries of every lens bin and every source bin.

# What each combination fixes <a name="des_cluster_fixed_params"></a>

| likelihood | `fixed_params` | why |
|---|---|---|
| `cosmic_shear` | none | |
| `combo_2x2pt` | none | |
| `combo_3x2pt` | none | |
| `combo_4x2pt_N` | `DES_B1_4`, `DES_B1_5`, `DES_B1_6`, `DES_DZ_L4`, `DES_DZ_L5`, `DES_DZ_L6` | the mask removes lens bins 4-6 (Table I of arXiv:2503.13631 varies them only in CL+3x2pt) |
| `combo_4x2pt_N` | `DES_PM1`, `DES_PM2`, `DES_PM3`, `DES_PM4`, `DES_PM5`, `DES_PM6` | no `gs` block: the point masses act on galaxy-galaxy lensing only |
| `combo_6x2pt_N` | none | |

The fixed values are the fiducial values of `params_lens_maglim.yaml` (Table I of arXiv:2503.13631):

| parameter | fixed value |
|---|---|
| `DES_B1_4` | 1.62 |
| `DES_B1_5` | 1.78 |
| `DES_B1_6` | 1.75 |
| `DES_DZ_L4` | -0.002 |
| `DES_DZ_L5` | 0.001 |
| `DES_DZ_L6` | 0.008 |
| `DES_PM1` ... `DES_PM6` | 0 |

# Parameters that should not be varied <a name="des_cluster_not_varied"></a>

The table lists, for every likelihood of this folder, the parameters that do not change its masked data vector, why, and who fixes them. The columns:

    Applies when   the likelihood option under which the row holds
    Response       `0`: the masked data vector does not change at all
                   `< 1e-5`: every entry changes by less than this fraction of its value
    Fixed by       `fixed_params` of the likelihood yaml; a parameter file, where the
                   parameter is a constant; user: the parameter files sample it, and the
                   user's yaml must fix it; not declared: the likelihood does not include
                   the parameter's file, so the parameter does not exist there

A `*` in a parameter name stands for every bin number or name.

| Likelihood | Parameters | Reason | Applies when | Response | Fixed by |
|---|---|---|---|---|---|
| `cosmic_shear` | `DES_DZ_L*`, `DES_DZ2_L*`, `DES_B1_*`, `DES_B2_*`, `DES_BMAG_*`, `DES_PM*` | no lens block | always | `0` | not declared (the yaml includes `params_source.yaml` only) |
| `cosmic_shear` | `DES_CL_*` | no cluster block | always | `0` | not declared (the yaml includes `params_source.yaml` only) |
| `cosmic_shear` | `DES_A2_1`, `DES_A2_2`, `DES_BTA_1` | NLA has no tidal-torque terms | `IA_model: 0` | `0` | user (`params_source.yaml` samples them) |
| `cosmic_shear` | `DES_A1_3`, `DES_A1_4`, `DES_A2_3`, `DES_A2_4`, `DES_BTA_2`, `DES_BTA_3`, `DES_BTA_4` | the redshift-evolution IA model reads only `DES_A1_1`, `DES_A1_2`, `DES_A2_1`, `DES_A2_2`, `DES_BTA_1` | `IA_redshift_evolution: 3` | `0` | `params_source.yaml` |
| `cosmic_shear` | `DES_BARYON_Q1`, `DES_BARYON_Q2`, `DES_BARYON_Q3`, `DES_BARYON_Q4` | the baryon PC amplitudes enter only when the likelihood applies the PCs | `use_baryon_pca: False` | `0` | `params_source.yaml` |
| `combo_2x2pt` | `DES_CL_*` | no cluster block | always | `0` | not declared (the yaml includes `params_source.yaml` and `params_lens_redmagic.yaml`) |
| `combo_2x2pt` | `DES_A2_1`, `DES_A2_2`, `DES_BTA_1` | NLA has no tidal-torque terms | `IA_model: 0` | `0` | user (`params_source.yaml` samples them) |
| `combo_2x2pt` | `DES_A1_3`, `DES_A1_4`, `DES_A2_3`, `DES_A2_4`, `DES_BTA_2`, `DES_BTA_3`, `DES_BTA_4` | the redshift-evolution IA model reads only `DES_A1_1`, `DES_A1_2`, `DES_A2_1`, `DES_A2_2`, `DES_BTA_1` | `IA_redshift_evolution: 3` | `0` | `params_source.yaml` |
| `combo_2x2pt` | `DES_BARYON_Q1`, `DES_BARYON_Q2`, `DES_BARYON_Q3`, `DES_BARYON_Q4` | the baryon PC amplitudes enter only when the likelihood applies the PCs | `use_baryon_pca: False` | `0` | `params_source.yaml` |
| `combo_3x2pt` | `DES_CL_*` | no cluster block | always | `0` | not declared (the yaml includes `params_source.yaml` and `params_lens_redmagic.yaml`) |
| `combo_3x2pt` | `DES_A2_1`, `DES_A2_2`, `DES_BTA_1` | NLA has no tidal-torque terms | `IA_model: 0` | `0` | user (`params_source.yaml` samples them) |
| `combo_3x2pt` | `DES_A1_3`, `DES_A1_4`, `DES_A2_3`, `DES_A2_4`, `DES_BTA_2`, `DES_BTA_3`, `DES_BTA_4` | the redshift-evolution IA model reads only `DES_A1_1`, `DES_A1_2`, `DES_A2_1`, `DES_A2_2`, `DES_BTA_1` | `IA_redshift_evolution: 3` | `0` | `params_source.yaml` |
| `combo_3x2pt` | `DES_BARYON_Q1`, `DES_BARYON_Q2`, `DES_BARYON_Q3`, `DES_BARYON_Q4` | the baryon PC amplitudes enter only when the likelihood applies the PCs | `use_baryon_pca: False` | `0` | `params_source.yaml` |
| `combo_4x2pt_N` | `DES_PM1`, `DES_PM2`, `DES_PM3`, `DES_PM4`, `DES_PM5`, `DES_PM6` | no `gs` block: the point masses act on galaxy-galaxy lensing only | always | `0` | `fixed_params` of `combo_4x2pt_N.yaml` |
| `combo_4x2pt_N` | `DES_B1_4`, `DES_B1_5`, `DES_B1_6` | the mask removes every entry of lens bins 4-6 | `data_file: des_cluster_y6_4x2ptN.dataset` | `0` | `fixed_params` of `combo_4x2pt_N.yaml` |
| `combo_4x2pt_N` | `DES_DZ_L4`, `DES_DZ_L5`, `DES_DZ_L6` | the mask removes every entry of lens bins 4-6; a shift can still move the upper redshift edge of the magnification-kernel table that every lens bin shares | `data_file: des_cluster_y6_4x2ptN.dataset` | `< 1e-5` | `fixed_params` of `combo_4x2pt_N.yaml` |
| `combo_4x2pt_N` | `DES_DZ2_L4`, `DES_DZ2_L5`, `DES_DZ2_L6` | the mask removes every entry of lens bins 4-6; a stretch can still move the upper redshift edge of the magnification-kernel table that every lens bin shares | `data_file: des_cluster_y6_4x2ptN.dataset` | `< 1e-5` | `params_lens_maglim.yaml` |
| `combo_4x2pt_N` | `DES_BMAG_4`, `DES_BMAG_5`, `DES_BMAG_6` | the mask removes every entry of lens bins 4-6 | `data_file: des_cluster_y6_4x2ptN.dataset` | `0` | `params_lens_maglim.yaml` |
| `combo_4x2pt_N` | `DES_A2_1`, `DES_A2_2`, `DES_A2_3`, `DES_A2_4`, `DES_BTA_1`, `DES_BTA_2`, `DES_BTA_3`, `DES_BTA_4` | NLA has no tidal-torque terms (and the redshift-evolution IA model reads only `DES_A2_1`, `DES_A2_2`, `DES_BTA_1` under TATT) | `IA_model: 0` | `0` | `params_source_y6.yaml` |
| `combo_4x2pt_N` | `DES_A1_3`, `DES_A1_4` | the redshift-evolution IA model reads only `DES_A1_1` and `DES_A1_2` | `IA_redshift_evolution: 3` | `0` | `params_source_y6.yaml` |
| `combo_4x2pt_N` | `DES_BARYON_Q1`, `DES_BARYON_Q2`, `DES_BARYON_Q3`, `DES_BARYON_Q4` | the baryon PC amplitudes enter only when the likelihood applies the PCs (the cluster likelihoods refuse them) | `use_baryon_pca: False` | `0` | `params_source_y6.yaml` |
| `combo_6x2pt_N` | `DES_A2_1`, `DES_A2_2`, `DES_A2_3`, `DES_A2_4`, `DES_BTA_1`, `DES_BTA_2`, `DES_BTA_3`, `DES_BTA_4` | NLA has no tidal-torque terms (and the redshift-evolution IA model reads only `DES_A2_1`, `DES_A2_2`, `DES_BTA_1` under TATT) | `IA_model: 0` | `0` | `params_source_y6.yaml` |
| `combo_6x2pt_N` | `DES_A1_3`, `DES_A1_4` | the redshift-evolution IA model reads only `DES_A1_1` and `DES_A1_2` | `IA_redshift_evolution: 3` | `0` | `params_source_y6.yaml` |
| `combo_6x2pt_N` | `DES_BARYON_Q1`, `DES_BARYON_Q2`, `DES_BARYON_Q3`, `DES_BARYON_Q4` | the baryon PC amplitudes enter only when the likelihood applies the PCs (the cluster likelihoods refuse them) | `use_baryon_pca: False` | `0` | `params_source_y6.yaml` |

The likelihood also turns `use_baryon_pca` off when `external_baryon_suppression: True` (the `bfmt` theory block) or `create_baryon_pca: True` is set, so the `DES_BARYON_Q*` rows of the galaxy likelihoods hold in those runs too.

The photo-z shift and stretch of lens bins 4-6 in 4x2pt + N are the two families with a response that is not exactly zero. The magnification kernels of all lens bins are tabulated on one redshift grid, whose upper edge is the highest redshift any shifted and stretched lens bin reaches; with the fiducial values, lens bin 6 sets it. A shift of bin 6, a larger shift of bins 4-5, or a stretch above 1 moves that edge, and the kernels of bins 1-3 change at the level of the table interpolation.

Three families are fixed by the parameter files but are not in the table, as they do change the data vector:

- `DES_B2_*` (0): a nonzero value in any lens bin switches on the one-loop bias terms of every lens bin, so even `DES_B2_4`, `DES_B2_5`, and `DES_B2_6` change the 4x2pt + N data vector although the mask removes their bins.
- `DES_CL_BSZ` (0): the redshift power of the selection bias, fixed as a modeling choice (the paper's model has none).
- `DES_DZ2_L1` ... `DES_DZ2_L3` and `DES_BMAG_1` ... `DES_BMAG_3` in 4x2pt + N (all six bins in 6x2pt + N): Table I fixes the stretch at 1 and the magnification coefficients.

# Changing the mask, the scale cuts, or the probes <a name="des_cluster_changing"></a>

> [!Warning]
> The `fixed_params` block of `combo_4x2pt_N.yaml` encodes the 4x2pt + N mask and the blocks of that combination. A user who changes the mask or the scale cuts (for example a mask that keeps lens bins 4-6), or the probes of a combination, must revisit that combination's `fixed_params`: a parameter it fixes may then act on the data vector, and the fixed value hides that dependence without any error.

A `fixed_params` block in the likelihood block of the user's yaml replaces the combination's block as a whole: `fixed_params: null` samples every parameter again, and a shorter block keeps only the entries it repeats.

The steps below assume the Conda cocoa environment is active (`conda activate cocoa`), the shell is bash, and the current folder is the cocoa main folder `cocoa/Cocoa`.

**Step :one:**: activate the private Python environment by sourcing the script `start_cocoa.sh`

    source start_cocoa.sh

**Step :two:**: in the likelihood block of the user's yaml (below, a copy of `EXAMPLE_EVALUATE1.yaml` named `MY_EVALUATE1.yaml`), set `fixed_params`; here the point masses stay fixed and the bias and photo-z shift of lens bins 4-6 are sampled

    likelihood:
      des_cluster.combo_4x2pt_N:
        path: ./external_modules/data/des_cluster
        data_file: des_cluster_y6_4x2ptN.dataset
        fixed_params:
          DES_PM1:
            value: 0.0
          (...)
          DES_PM6:
            value: 0.0

**Step :three:**: add a value for every parameter sampled again to the `sampler: evaluate: override` block (cobaya's evaluate sampler refuses an override of a parameter that is not sampled, so the block must match the sampled parameters)

    sampler:
      evaluate:
        override:
          (...)
          DES_B1_4: 1.62
          DES_DZ_L4: -0.002

**Step :four:**: run the evaluation

    cobaya-run ./projects/des_cluster/MY_EVALUATE1.yaml -f

> [!NOTE]
> A parameter declared in the `params` block of the user's yaml follows that declaration whatever `fixed_params` holds: the user's `params` entry takes precedence over the likelihood defaults. The examples of this project do not declare nuisance parameters in their `params` block.
