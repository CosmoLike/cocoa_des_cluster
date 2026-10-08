# Lighthouse reference outputs for the cluster port

This folder holds reference numbers from the original lighthouse cluster
code. They were computed with the prebuilt library of lighthouse's live
build, loaded through `ctypes`. They serve to cross-check the C port of
the cluster observables (4x2pt + N) in cosmolike_core.

The scripts modify nothing under `lighthouse/` or under `cosmolike_core/`
(the `cluster_chto` checkout lighthouse builds against): they only load
and call the library.

## 1. What ran

| item | value |
|---|---|
| library | `lighthouse/lib/like_cluster_Buzzard_y3_2_fast_v37_IA_test4.so` (arm64; sha256 below) |
| build flags | `-DCLASS_V29 -DIAC -DBuzzard -DFullsky -DDESY3 -DSELECTIONB` (Makefile target `like_cluster_fullsky_y3_IA_selectionB`) |
| C sources in the build | `lighthouse/cpp/like_real_mpp.c`, which includes `init_des_real.c`, `cluster_util.c`, `clusters_DES_nonlimber.c`, `init_cluster.c`, and cosmolike_core `theory/{cosmo3D,halo,redshift_spline,cosmo2D_fourier,cosmo2D_real,cosmo2D_exact_fft,cosmo2D_fullsky_TATT,...}.c` |
| dynamic libraries | resolved through the rpath stored in the library, a conda environment's `lib` folder (gsl 28, fftw3, openblas, libc++, libomp); no `DYLD_LIBRARY_PATH` is needed |
| Python | a Python 3.11 environment with numpy (the outputs were made with numpy 1.26.3) |
| runtime | about 30 s per data vector (single thread; the included sources have no OpenMP pragmas) |
| determinism | Three separate processes gave bitwise-identical `theory_wrapper` vectors |

Full library sha256: `efea6cbcccb781354164363ad41fb95d89c87c14cf12f7cca1a58c08af3d4dae`.

The driver `lh.py` reimplements `init_all_nuisance_param()` from
`lighthouse/python/cosmolike_libs_real_mpp_cluster.py` with the same
call order. It skips that module's `emcee`, `emcee_wrapper`,
`mpp_blinding` and `schwimmbad` imports.

## 2. How to reproduce

`lh.py` loads the library from the lighthouse checkout named by its
constant `LIGHTHOUSE`; set it first. Run the commands below from this
folder, in a Python 3.11 environment with numpy. Each variant must run
in a fresh process, because the C code keeps static caches.

    python run_reference.py main     # Ystatistics=1, data vectors + all intermediates
    python run_reference.py Yoff     # Ystatistics=0
    python run_reference.py limber   # w_gg, w_cg, w_cc replaced by their Limber versions
    python nonlimber_rescaling_check.py   # memoized vs fresh non-Limber C_ell

Logs are in `logs/`.

## 3. Configuration

The same values are also stored in `outputs/config.json` and in
`lh.CONFIG`.

### Binning and survey

- Cluster z_lambda edges: [0.2, 0.4, 0.55, 0.65].
- lambda edges: [20, 30, 45, 60, 500].
- w_cg pairs (cluster z bin, lens bin): (0,0), (1,1), (2,2).
- Angular bins: 20 log-spaced bins over 2.5–250 arcmin. The C code uses the area-weighted centre θ = 2/3 (θmax³−θmin³)/(θmax²−θmin²).
- Survey area: 4143 deg². The counts use 4π/41253 sr per deg².
- `rmarea_int` = False and `rmphotoz_int` = False: constant area, and a top-hat selection in true z for the clusters.

### Galaxy samples

- Lens n(z): `inputs/lens_y6.nz` (6 bins).
- Source n(z): `inputs/source_y6.nz` (4 bins).
- Both are byte-identical copies of `lighthouse/analysis/des_y6_code_comparison/{lens,source}.nz`.
- `ggl_overlap_cut` = 1e-5, which accepts all 24 gamma_t pairs.
- `lens_n_gal`, `source_n_gal` and `sigma_e` come from `dataY6.yaml`. They are unused by the model.

### Cosmology

- Runmode `Halofit`: EH98 wiggle transfer function plus Takahashi Halofit. CLASS is not used.
- Omega_m = 0.3, Omega_b = 0.048, h = 0.69, n_s = 0.96859, w0 = −1, wa = 0.
- omega_nu h² = 0. The EH/Halofit path and halo.c do not support massive neutrinos.
- **Normalization.** The Halofit runmode normalizes only through sigma_8 (`Delta_L_wiggle`). An A_s passed in that runmode would give sigma_8 = 0. So A_s = 2.19e-9 was converted with CAMB 1.6.5 (massless ν, Neff = 3.046, same Ωm, Ωb, h, n_s): **sigma_8 = 0.8425192400940533**. This value is passed with `A_s = 0` in the struct. With Neff = 3.044 the result is 0.84265.

### Nuisance parameters

- Galaxy b1 per lens bin: [1.42, 1.66, 1.70, 1.62, 1.78, 1.75]. Source: `lighthouse/analysis/yamlfiles/dataY6.yaml`, which matches 2503.13631 Table I. The `des_y6_code_comparison/dataY6.yaml` file has different values (1.54, 1.81, …) and was not used.
- Galaxy b2 = 0. This matters: `gbias.b2[0] != 0` turns on the cluster 1-loop terms of w_cc, and a nonzero b2 of its lens bin those of w_cg.
- `nonlinear` = 0.
- Cluster b2 is not set (`init_cluster_b2` is not called).
- MOR = [ln λ0, A, σ_int, B] = [4.26, 0.943, 0.15, 0.207], with M_piv = 5e14 Msun/h hard-coded in `cluster_util.c:613`.
- Selection = [b_s1, b_s2, r0, s3] = [1, 0, 30, 0], so b_sel = 1.
- IA off: `like.IA` = 0, `init_IA_mpp` is never called, and `nuisance.A_ia` = 0. The IAC term in C_cs therefore vanishes.
- Shear m = 0, photo-z shifts = 0, lens stretch = 1, b_mag = 0, point mass = 0.

### Probes and mask

- Probes: `xip xim gammat wtheta w_cg gamma_c cluster_N w_cc`.
- `Ystatistics` = 1, except in the `Yoff` variant.
- Mask: all ones (`inputs/ones_2812.mask`).

### Data-vector layout

The layout follows `like_real_mpp.c` lines 1041-1076. There are 2812 entries.

| block | start | end | ordering |
|---|---|---|---|
| xi+ | 0 | 200 | 10 shear pairs × 20 θ |
| xi− | 200 | 400 | 10 shear pairs × 20 θ |
| gamma_t | 400 | 880 | 24 (lens, source) pairs, lens outer, × 20 θ |
| w_gg | 880 | 1000 | 6 bins × 20 θ (non-Limber + RSD, `w_tomo_nonLimber`) |
| w_cg | 1000 | 1240 | cg pair (outer) × λ × θ |
| N | 1240 | 1252 | z_c (outer) × λ |
| w_cc | 1252 | 1852 | z_c (outer); then (λ1, λ2 ≥ λ1) row-major: (0,0), (0,1), (0,2), (0,3), (1,1), (1,2), (1,3), (2,2), (2,3), (3,3); × θ |
| gamma_c | 1852 | 2812 | 12 (z_c, z_s) pairs with n = 4 z_c + z_s, including sources in front of the clusters; × λ × θ |

## 4. Files

### `outputs/` (data vectors)

All text files have the format `index value`. Values are written `%.17e`, unless the name says `compute_data_vector`.

| file | content |
|---|---|
| `dv_main_theory_wrapper.txt` | `theory_wrapper` output. This is the model vector before the final Y transform. Its gamma_c block is T⁺ diag(b_sel) T γ_c. |
| `dv_main_Ytransformed.txt` | The same vector with `apply_Ttransform_and_applymask` applied, i.e. T acting on the gamma_c block. This is what enters χ² when Ystatistics=1. |
| `dv_main_compute_data_vector.txt` | `write_datavector_wrapper` (the `compute_data_vector` path), `%le` precision. Agrees with `dv_main_Ytransformed.txt` to ≤ 4.9e-7 relative. |
| `dv_Yoff_theory_wrapper.txt`, `dv_Yoff_compute_data_vector.txt` | Ystatistics=0. Bitwise equal to `dv_main_theory_wrapper.txt`: the T⁺T distortion stays and no T is applied. |
| `dv_limber_theory_wrapper.txt`, `dv_limber_Ytransformed.txt` | Limber-only w_gg, w_cg and w_cc. The other blocks are unchanged. |
| `lighthouse_reference_{Yoff,limber}.npz` | The same vectors in binary form |
| `config.json` | The configuration |

The Limber-only w_gg comes from `w_tomo_exact`, which uses C_cl_tomo and the same bin-averaged full-sky Legendre projection. The Limber-only w_cg and w_cc come from `C_*_tomo_tab` for ℓ = 1..99999, projected with the same weights (section 6).

### `outputs/lighthouse_reference_main.npz` (intermediates)

All intermediates were computed in the same process and at the same parameters as the main data vector.

**Data vectors and gamma_c**

- `dv_theory_wrapper`, `dv_Ytransformed`, `block_names`, `block_start`, `block_end`.
- `theta_{min,max,}_arcmin`, `logdt` (Δ ln θ).
- `gamma_c_raw` (960,): the raw flat-sky `cluster_gamma_t(θ_i, λ, zc, zs)` in data-vector order.
- `gamma_c_pairs_zc_zs`.
- `T_Ytransform` (20×20).
- `T_pinv_numpy`: the pseudo-inverse with rcond 1e-15. The C code uses its own SVD pseudo-inverse with the same rcond. The two agree: the data-vector gamma_c equals the Python T⁺T γ_raw to 1.4e-13.
- `gamma_c_TpT_python`.
- `Tfull_cs_block_check`.

**Background.** The grid `bg_z` is z ∈ [1e-4, 3], 301 points.

- `bg_chi`: χ in c/H0.
- `bg_fK`.
- `bg_dchi_da`.
- `bg_hoverh0` = 1/(a² dχ/da). This is sqrt(Ωm a⁻³ + 1−Ωm), with no radiation.
- `bg_growth_D_over_D0`.
- `bg_f_growth`.
- Accuracy: cosmolike's χ(a) table differs from exact quadrature by 2–6e-5 relative for z ≥ 0.1.

**Power spectra**

- `pk_k_hMpc` (301 points in h/Mpc) and `pk_z` = [0, 0.3, 0.475, 0.6, 1].
- `pk_lin_code` from `p_lin` and `pk_nl_code` from `Pdelta`. They are evaluated at k_code = k[h/Mpc] × 2997.92458. Values are in (c/H0)³; multiply by 2997.92458³ to get (Mpc/h)³.
- Store these tables if the port must be fed the same P(k).

**Halo model.** The grid is `halo_lgM` (79 points, log10 M from 12 to 15.9, M200m in Msun/h) × `halo_z`.

- `halo_massfunc_dndM`: dn/dM in (H0/c)³ per Msun/h. This is the Tinker10 f(ν), with a ≥ 0.25.
- `halo_B1`: Tinker10 b(ν), not normalized.
- `halo_nu`.
- `halo_conc`: Bhattacharya13, 9 ν^−0.29 D^1.15.
- `halo_sigma2_z0`.
- `halo_rDelta_code_z0` in c/H0.

**Mass-observable relation**

- `mor_P_lambda_given_M_density[z, M, λ]` = dP/dλ_obs, a lognormal in λ with σ² = σ_int² + (e^{⟨lnλ⟩}−1)/e^{2⟨lnλ⟩}. It is evaluated on the grids `mor_lambda` (200 points, 5–500) and `mor_z` = [0.3, 0.475, 0.6].
- `mor_P_bin_given_M_exact[λbin, z, M]` from `probability_observed_richness_given_mass`, and `mor_P_bin_given_M_tab` from the `_tab` version. **The model uses `_tab`.** See section 7, item 2.

**Cluster abundance and bias.** The grid is `clz_z` (0.15–0.70, 111 points). Here `_exact` is the direct integral at each z and `_tab` the 100-node a-table the model reads; both use the P(λ bin|M) table (section 7, item 2).

- `n_A_exact` and `n_A_tab` in (H0/c)³; multiply by 2997.92458⁻³ to get (h/Mpc)³.
- `b_A_tab`: `weighted_bias`, the table used by the model.
- `b_A_exact`.
- `mass_mean_A`.
- `b_A_at_centers_{tab,exact}`, evaluated at `zc_centers` = [0.3, 0.475, 0.6].
- `N_counts[zc, λ]`.
- `mean_mass_bin[zc, λ]`.

**Cluster kernels**

- `zdistr_cluster[zc, z]` on `kern_z`. This is the volume-only top hat χ²/E(z), normalized to ∫dz = 1 over the bin. It does not depend on λ.
- `norm_z_cluster`.
- `W_cluster[zc, λ, z]` = zdistr × E(z).
- `W_cl` = b_A W_cluster − 2 W_mag_cluster.
- `g_lens_cluster[zc, z]` and `W_mag_cluster[zc, z]`, both on `gl_z`.

**Galaxy samples**

- `pf_photoz_lens` and `zdistr_photoz_source` on `nz_z`.
- `zmean_lens`, `zmean_source`, `clustering_zmax`, `g_tomo_source`.

**Limber C_ℓ.** The ℓ list is `cl_ell` = 2 … 9e4.

- `cc_pairs_nz_l1_l2`.
- `C_cc_limber_{nl_exact,lin_exact,nl_tab}` from `C_clusterxclusterclustering_tomo` with linear = 0 or 1, and from `_tab`.
- `cg_pairs_zc_zg`: the (cluster z bin, lens bin) pairs of w_cg.
- `C_cg_limber_{nl_exact,lin_exact,nl_tab}[cgpair, λ, ℓ]`.
- `C_cs_limber_{exact,tab}[pair, λ, ℓ]`, including the 1-halo term.
- `_tab` agrees with `exact` to ≤ 0.3%. The data vector uses the `_tab` spectra.

**1-halo and 2-halo 3D spectra**

- `P_cm_1h_exact[λ, zc, k]`: P_cm^1h(k, z_centre) in (c/H0)³, on `p1h_k_hMpc`, at the redshifts `p1h_z`.
- `P_cc_2h_nl[λ, zc, k]` = b_A² P_NL.

**Non-Limber (exact spectra used in the data vector)**

- `Cl_cc_nonlimber_mix_l0_2000`: the C_ℓ array (ℓ = 0..2000) actually summed into w_cc. It comes from `C_clusterxclusterclustering_mix_tab`, called in the data-vector order.
- `Cl_cg_nonlimber_mix_l0_2000`: the same for w_cg.
- The matching `*_limber_tab_l0_2000`.
- `w_{cc,cg}_projected_from_mix`: Python projection of the full ℓ < 1e5 array. It reproduces the data-vector w_cc and w_cg to 3.4e-11.
- `w_{cc,cg}_limber`.
- `cc_block_offsets`.

### `outputs/nonlimber_rescaling_check.npz`

Memoized ("bias-rescaled") non-Limber C_ℓ and w versus a fresh FFTLog computation, for several (λ1, λ2) combinations (section 7, item 13).

### Scripts and inputs

- `lh.py`: ctypes bindings, configuration, init sequence, layout, and Legendre bin weights.
- `run_reference.py`, `nonlimber_rescaling_check.py`.
- `inputs/`: n(z) copies and the mask.

Two validation scripts read this folder: `../validation/compare_reference.py`
compares the port with `outputs/lighthouse_reference_main.npz` and
`outputs/config.json` (its `--skip-lighthouse` flag turns that step off),
and `../validation/time_lighthouse.py` times the original code through
`lh.py`.

## 5. Key numbers (main configuration)

### Counts N[z_c][λ]

| z_c \ λ | [20,30) | [30,45) | [45,60) | [60,500) | total |
|---|---|---|---|---|---|
| [0.20,0.40) | 4027.14 | 1848.85 | 612.28 | 527.59 | 7015.9 |
| [0.40,0.55) | 4959.61 | 2105.51 | 640.52 | 478.21 | 8183.9 |
| [0.55,0.65) | 3773.17 | 1500.80 | 425.55 | 283.64 | 5983.2 |

For comparison, the Y3 data counts are 5632, 6308 and 4551 per z bin, over the same area.

log10 ⟨M|bin⟩ [Msun/h]:

- z_c bin 0: 14.146, 14.342, 14.511, 14.748
- z_c bin 1: 14.127, 14.323, 14.493, 14.714
- z_c bin 2: 14.112, 14.309, 14.479, 14.689

### b_A at the z_c bin centres

The values below come from the `weighted_bias` table. The exact values agree to 1e-5.

| λ \ z | 0.3 | 0.475 | 0.6 |
|---|---|---|---|
| [20,30) | 2.8569 | 3.2967 | 3.6462 |
| [30,45) | 3.4527 | 4.0065 | 4.4534 |
| [45,60) | 4.1188 | 4.8027 | 5.3483 |
| [60,500) | 5.3399 | 6.1407 | 6.7859 |

### Limber C_ℓ at ℓ = 10, 100, 1000, 10000

- C_cc (z_c 0, λ 0 × 0), nonlinear: 4.1175e-4, 8.6341e-5, 5.1912e-6, 1.1065e-7.
- C_cg (z_c 0 × lens 0, λ 0): 1.6532e-4, 3.2086e-5, 1.9710e-6, 3.9695e-8.
- C_cs (z_c 0 × source 3, λ 0, 2-halo + 1-halo): 2.5485e-6, 5.6278e-7, 9.9186e-8, 8.0157e-9.

### w values at θ = 2.84, 8.97, 28.4, 89.7, 225 arcmin

- w_cc (z_c 0, λ 0 × 0): 1.2470, 0.42337, 0.17242, 0.044070, 0.0058898.
- w_cg (z_c 0, λ 0): 0.47424, 0.16005, 0.065624, 0.017447, 0.0025697.

### gamma_c (z_c 0 × source 3, λ 0) at the same θ

| version | γ_c at θ = 2.84, 8.97, 28.4, 89.7, 225′ |
|---|---|
| raw | 2.3152e-2, 5.4961e-3, 1.0212e-3, 3.5005e-4, 1.3617e-4 |
| T⁺T (data vector) | −4.3812e-3, 2.7602e-3, 6.7596e-4, 3.6331e-4, 8.6241e-5 |
| T γ (after the final Y transform) | 1.5351e-2, 2.3594e-3, 9.4056e-4, 2.1922e-4, 0 |

### T⁺T distortion

T has rank 19, because its last row is exactly zero. The same zero row makes the last point of T γ equal to 0.

So T⁺T is a projector, not the identity, even with b_sel = 1. Over all gamma_c entries, |T⁺Tγ/γ − 1| reaches:

- 124% at 2.8′
- 96% at 3.6′
- 83% at 5.7′
- 49% at 14′
- 36% at 28′
- 4–28% between 36′ and 179′
- 57% at 225′

It is a sign flip at the smallest θ. With Ystatistics=1 the χ² uses T (T⁺ b T γ), which equals T γ for b = 1, so it is consistent. With Ystatistics=0 the data vector (`dv_Yoff_*`) keeps the distorted T⁺Tγ.

### Limber vs non-Limber (data-vector level)

Values are Limber/non-Limber − 1.

| probe | Limber/non-Limber − 1 at 2.84′ | at 28.4′ | at 225′ |
|---|---|---|---|
| w_cc (z_c 0, λ 0×0) | +0.25% | +1.4% | −13% |
| w_cc (whole block) | | | range −69% … +4% |
| w_cg (z_c 0, λ 0) | −0.04% | −0.3% | −16% |
| w_cg (whole block) | | | range −77% … +1% |
| w_gg (bins 0–2) | −0.1% … −0.4% | | −13%, −29%, −91% |

At the C_ℓ level, the ratio non-Limber/Limber for w_cc (z_c 0, 0×0) is:

- ℓ = 2: 1.90
- ℓ = 5: 1.26
- ℓ = 10: 1.07
- ℓ = 50–150: 0.98–0.99
- ℓ ≥ 200: exactly Limber

For w_cg the ratio at ℓ = 2 is 2.02, and C_ℓ is pure Limber for ℓ ≥ 100, because the first 100-ℓ block already converged.

## 6. Function signatures used (ctypes, all exported by the .so)

Driver:

    void init_cosmo_runmode(char*); void init_source_sample_mpp(char*, int);
    void init_lens_sample_mpp(char*, int, double* b1, double* b2, double ggl_cut);
    void init_binning_mpp(int, double tmin_arcmin, double tmax_arcmin);
    void init_cluster_N200(int, double*); void init_tomo_cluster(int, double*);
    void set_cluster_cg_Npowerspectra(int, int* ZC, int* ZG);
    void init_survey_mpp(char*, double area, double sigma_e, double* nlens, double* nsrc, char* footprint);
    void init_cluster_area_interpolation(char*, bool); void init_cluster_photo_z(char*, bool);
    void set_nonlinear(int); void init_probes_4x2ptN(char*); void set_Ystatistics(int);
    void init_mask(char*); int getNdata(void);
    void theory_wrapper(input_cosmo_params_mpp, input_nuisance_params_mpp, double* out);   // structs by value
    void write_datavector_wrapper(char*, input_cosmo_params_mpp, input_nuisance_params_mpp);
    void set_all_parameters(input_cosmo_params_mpp, input_nuisance_params_mpp);
    void apply_Ttransform_and_applymask(double*); double T_Ytransform(int i, int j, int N, double dlnθ);
    double T_Ytransform_full(int i, int j); int ZC(int n); int ZSC(int n);
    int get_N_ggl(void); int get_N_tomo_shear(void); double get_tomo_clustering_zmax(int j);

Cluster model:

    double N_Delta_lambda_obs(int λ, int zc); double n_lambda_obs_z(int λ, double z); double n_lambda_obs_z_tab(int, double);
    double weighted_bias(int λ, double z); double weighted_bias_exact(int, double); double mass_mean(int, double);
    double mean_mass_Delta_lambda_obs(int λ, int zc);
    double int_probability_observed_richness_given_mass(double λobs, void* {M, z});
    double probability_observed_richness_given_mass(int λ, double M, double z); double probability_observed_richness_given_mass_tab(...)
    double zdistr_cluster(double z, int zc, int λ); double norm_z_cluster(int zc, int λ);
    double W_cluster(double a, int zc, int λ); double W_cl(double a, int zc, int λ);
    double W_mag_cluster(double a, double fK, int zc); double g_lens_cluster(double a, int zc, int -1);
    double C_clusterxclusterclustering_tomo(double ℓ, int λ1, int λ2, int zc, int zc, double linear); ..._tab(ℓ, λ1, λ2, zc, zc)
    double C_clusterxgalaxyclustering_tomo(double ℓ, int λ, int -1, int zc, int zg, double linear); ..._tab(ℓ, λ, -1, zc, zg)
    double C_clusterlensing_tomo(double ℓ, int λ, int -1, int zc, int zs); ..._tab(...)
    void C_clusterxclusterclustering_mix_tab(int L=0, int LMAX=1e5, int ni, int nj, int λ1, int λ2, double* Cl, double dev=0.1, double tol=0.01)
    void C_clusterxgalaxyclustering_mix_tab(int L=0, int LMAX=1e5, int zc, int zg, int λ, double* Cl, double dev=0.1, double tol=0.01)
    double P_cluster_mass_given_Dlambda_obs_1halo(double k_code, double a, int λ, int zc, int zs);
    double P_cluster_x_cluster_clustering_mass_given_Dlambda_obs(double k_code, double a, int λ1, int λ2, double linear);
    double cluster_gamma_t(double θ_rad, int λ, int zc, int zs);
    double w_tomo_exact(int nt, int ni, int ni);

Background and halo:

    chi(a), f_K(chi), dchi_da(a), growfac(a), f_growth(z), p_lin(k_code,a), Pdelta(k_code,a),
    massfunc(M,a), B1(M,a), nu(M,a), sigma2(M), conc(M,a), r_Delta(M,a), pf_photoz(z,j), zdistr_photoz(z,j),
    zmean(j), zmean_source(j), g_tomo(a,j)

The full-sky bin-averaged Legendre weight used for all w(θ) is:

    W_i(ℓ) = [P_{ℓ+1}(x_min) − P_{ℓ+1}(x_max) − P_{ℓ−1}(x_min) + P_{ℓ−1}(x_max)] / [4π (x_min − x_max)]

- x = cos θ at the bin edges.
- w_i = Σ_{ℓ=1}^{99999} W_i(ℓ) C_ℓ.
- `lh.legendre_bin_weights` implements this.

Units summary:

- χ and f_K in c/H0.
- k_code in H0/c.
- P in (c/H0)³.
- M = M200m in Msun/h.
- dn/dM in (H0/c)³/(Msun/h).
- n_A in (H0/c)³.
- θ in radians inside the library, arcmin in the outputs.
- 2997.92458 = c/H0 in Mpc/h.

## 7. Deviations of this code from the Y6 paper model (2503.13631)

The port follows the paper on these points, except the choices its
project README documents: the volume-only cluster kernel by default
(item 1's missing n_A weighting), all 12 cluster-lensing pairs kept and
masked rather than removed (item 9), and Limber-only w_cc and w_cg
(the opposite of item 12).

### Redshift kernels and counts

1. **Cluster n(z) is volume-only.** `zdistr_cluster` ∝ χ²/E(z) is a top hat in *true* z over each z_λ bin. It has no ⟨φ|z⟩ photo-z kernel (`rmphotoz_int` off) and no n_A(z) weighting inside the bin, per the comment "disregards evolution of N-M relation + mass function within redshift bin". The counts do integrate n_A(z): N = Ω_s ∫ dz χ²/E n_A(z), with Ω_s = 4π·area/41253.
2. **Coarse P(λ bin | M, z) table.** The model uses a 10 (z ∈ [0.2, 1.5]) × 50 (log10 M ∈ [12, 15.9]) table with linear interpolation (`probability_observed_richness_given_mass_tab`), not the exact erf. Relative to the exact P:
   - n_A is +0.6% to +2.3% higher (larger for higher λ and z).
   - b_A is −0.3% to −0.7% lower.

   These ratios come from the stored arrays (section 4). An exact-P port should expect counts about 1–2% lower than lighthouse.
3. **Unfilled table edges.** The z and log M fill loops accumulate `z += dz`, so the last z row (z = 1.5) and the last mass column (log10 M = 15.9) are never filled and stay at 0 (calloc). The same holds for the massfunc×P, c(M) and 1-halo mass tables. As a result, P falls linearly to 0 over 10^15.82–10^15.9. Because λ(M = 10^15.9) = 775 > 500, the effect is small here.
4. **P = 0 for z < 0.2.** In `interpol2d`, x < ax returns 0. `weighted_bias_exact` then returns 1e-2 there. The `weighted_bias` and `n_lambda_obs_z_tab` tables span z ∈ [0.15, 0.70] on 100 a-nodes. So b_A(z) in the first a-cell above z = 0.2 (Δz ≈ 0.005) is interpolated between 0.01 and about 2.6; `b_A_tab` at z = 0.2 is 1.98 against 2.64 just above.
5. **Mass integration range.** Mass integrals run only over M ∈ [1e12, 10^15.9] Msun/h.
6. **Halo-model details.**
   - Tinker10 f(ν) with a ≥ 0.25.
   - dlnν/dlnM from `gsl_deriv_central` with step 0.1·lnM.
   - δ_c = 1.686.
   - The bias is not normalized.
   - σ(M) comes from the EH P_lin.
7. **Selection bias does not touch b_A.** With `SELECTIONB`, `B1xBselection` returns the bare B1. The selection enters only at data-vector level:
   - b_sel = [b_s1 + b_s2 exp(−θ f_K(χ(z̄)) c/H0 / r0)] ((1+z̄)/1.45)^{s3}, with z̄ the bin midpoint.
   - w_cg is multiplied by b_sel and w_cc by b_sel².
   - gamma_c becomes T⁺ diag(b_sel) T γ_c, with T⁺ from SVD at rcond 1e-15.
   - T is singular (last row zero), so this is a projection even for b_sel = 1; see section 5.
   - With Ystatistics=1, T is applied again to the model and to the data (`data_read_Y`, `invcov_Y_mask`). The last θ of each gamma_c spectrum gets unit variance and zero signal.

### Cluster lensing (gamma_c)

8. **Flat-sky γ_c.** γ_c is a flat-sky Hankel J2 transform (FFTLog, N_thetaH = 2048, ℓ from 1e-4 to 5e6) of `C_clusterlensing_tomo` (exact, not the table). It is evaluated at the θ bin centre by linear interpolation in ln θ, with no bin averaging. Also:
   - No Σ conversion.
   - No miscentering, boost factor, reduced shear or point mass.
   - The paper projects in full sky with bin averaging.
9. **"All pairs" hack.** `init_tomo_cluster` computes `N_cgl`, `ZC` and `ZSC` while the cluster z edges are still zero. As a result, all 12 (z_c, z_s) pairs are kept, including sources in front of the clusters.
10. **C_cs composition.** C_cs = ∫ over the cluster-bin a-range only of [W_cl W_κ P_NL + W_cluster W_κ P_cm^1h] dχ/da / f_K², where:
    - W_cl includes the cluster magnification −2 W_mag_cluster, truncated to the bin z-range.
    - The 1-halo term is an NFW profile truncated at r200m, with Bhattacharya c(M) and no magnification.
    - The 1-halo term comes from a 200 k × 30 a table, zero outside k ∈ [0.02, 3e6] H0/c.
    - The 1-halo table is recomputed per source bin, although it does not depend on the source bin.
    - IA (IAC NLA) would subtract W_source·A_IA terms; that is zero here.

### Clustering (w_cc and w_cg)

11. **Limber w_cc and w_cg spectra.** These (`C_*_tomo`) are integrated only over the cluster bin, or the cluster ∩ lens overlap range for w_cg. They use:
    - b_A P_NL (Halofit).
    - Magnification through W_cl, limited to that same z range, so lower-z magnification is dropped.
    - No RSD.
12. **Non-Limber (DESY3 build, always on for w_cc, w_cg and w_gg; no runtime switch).** The spectrum is C_ℓ = C_ℓ^FFTLog(lin) + C_ℓ^Limber(NL) − C_ℓ^Limber(lin). In this form:
    - The FFTLog part is evaluated in blocks of 100 ℓ until the last ℓ of a block matches Limber to 1%, and never past ℓ = 199.
    - It uses P_lin(k, z = 0) × growth inside the kernels.
    - RSD enters with f = `f_growth`, not multiplied by bias.
    - Cluster magnification C = −2 is taken over all z < z_max in the FFTLog part, but only within the bin in the Limber part.
    - C_ℓ is pure Limber above the switch.
13. **Bias-rescaled non-Limber.** The FFTLog part is computed only for the first (λ1, λ2), or first λ, requested in each z bin (static `Nz` / `Nz_gal` memo). For the others it is rescaled by b_A(λ1) b_A(λ2) / b_A(0)², evaluated at a single z:
    - for w_cc, z = (z_max + z_min)/2;
    - for w_cg, the memo uses `zmean(ni)`, the mean z of **lens** bin ni, called with the cluster bin index.

    `nonlimber_rescaling_check.py` gives, for memoized w over fresh w − 1:
    - w_cc (z_c 0, 1×1): −4e-5 to +2.4%.
    - w_cc (z_c 0, 3×3): +0.13% to +4.2%.
    - w_cg (z_c 0, λ 1): +0.01% to +1.0%.
    - w_cg (z_c 0, λ 3): +0.02% to +1.5%.

    In each case the largest value is at the largest θ.
14. **Dormant magnification bug.** In `C_clusterxclusterclustering_mix_tab`, the second-λ magnification term lacks the `clusterMag` factor (+1 instead of −2). It is dormant in the data vector, because the fresh FFTLog is always done for λ1 = λ2 = 0. It would affect the fresh cross-λ values in the rescaling check (λ pairs (0,3) and (1,2)).
15. **Probe coverage (as in the paper).** w_cc is computed only for the auto z bin and all λ pairs. w_cg is computed only for the (i, i) pairs.

### Cosmology and numerics

16. **Power spectrum.** EH98 plus Takahashi Halofit, normalized with σ8, with no massive neutrinos. The paper uses CAMB/CLASS with HMcode2020 (T_AGN = 7.7) and a massive ν.
17. **No radiation in the background.** H(a) has no radiation term. χ(a) comes from a table with 2–6e-5 interpolation error.
18. **Integration precision.**
    - Counts, b_A, n_A and all C_ℓ integrals use GSL CQUAD with relative tolerance 1e-2 (`int_gsl_integrate_low_precision`).
    - P(λ|M) and the kernels use 1e-3.
    - C_ℓ tables interpolate 200 log-spaced ℓ between 0.1 and 1e5; `_tab` agrees with exact to ≤ 0.3%.
    - `zdistr_cluster` integrates to 1.001–1.003 on a 551-point trapezoid, which reflects the precision of the grid, not of the code.

    Expect about 1e-3 agreement at best against an accurate port, before the model differences listed above.
19. **No cluster 1-loop.** This requires `nonlinear` = 0 and galaxy b2 = 0. A nonzero `gbias.b2[0]` would turn the cluster 1-loop terms on for w_cc; w_cg uses the b2 of its lens bin.

## 8. Obstacles and notes

- **Python driver.** The lighthouse Python driver imports `emcee`, `mpp_blinding`, `emcee_wrapper` and `schwimmbad`; `lh.py` replicates its init sequence instead of importing it.
- **No A_s in Halofit.** The A_s path in the Halofit runmode is unavailable, since only the σ8 normalization exists. σ8 comes from CAMB, as described in section 3; the Halofit runmode needs only σ8, so CLASS is not involved.
- **CLASS-only helpers.** `p_lnM_lambd_bins` and `mass_mean_float` call `assert(0)` outside the CLASS runmodes, and the scripts do not call them; the P(λ|M) density comes from `int_probability_observed_richness_given_mass`.
- **`hoverh0` not exported.** It is `static inline`, so it is reconstructed from `dchi_da`.
- **No Limber switch.** The build has none. The Limber-only vector is assembled from the exported Limber tables with the same Legendre weights. Applied to the non-Limber spectra, that projection reproduces the data-vector w_cc and w_cg to 3e-11.
- **Ystatistics on/off.** On and off give identical `theory_wrapper` output. The only difference is the final T applied in `compute_data_vector` and in χ². `gammat` and `gamma_c` are always computed at all θ here, because the mask is all ones.
- **Stale static caches.** The C caches key on cosmology, MOR and selection only. A change of galaxy bias or n(z) does **not** refresh `C_clusterxgalaxyclustering_tomo_tab`. Always use one process per configuration.
