# Cluster port plan: 4x2pt + N in cosmolike_core

Goal: port the cluster part of the original CosmoLike (lighthouse +
cosmolike_core branch `cluster_chto`, arXiv 2008.10757) into cocoa's
cosmolike_core, optimized on the cosmo2D.c / halo.c designs, and run the
DES "CL+GC" (= 4x2pt + N) and "CL+3x2pt" (= 6x2pt + N) analyses of
arXiv 2503.13631 (Y6 methods, the target model) / 2503.13632 (Y3 data)
from this project.

Reference code: lighthouse branch `Ystatistics` (Chun-Hao To's latest
DES Y6 work, d4bb423, 2026-09-02; the local checkout `des_y6_comparison`
adds only compile/shear-calibration/comparison commits) with
cosmolike_core branch `cluster_chto` (1b861b1, 2026-08-21).

Source studies (scratchpad of the 2026-09-29 session; key facts copied
below): `papers_model_spec.md` (model spec, parameter tables, binning,
public data), the port audit (function map, compile state, bugs,
optimization patterns, interface design), the lighthouse live-path map,
`fullsky_cluster_projections.md` (full-sky projection math).

## 1. Findings that set the strategy

1. The cocoa cluster files (`cluster_util.c`, `cosmo2D_cluster.c`) are
   not a partial port: they were written against functions that are
   commented out in the current library (`W_cluster`, `zdistr_cluster`,
   `recompute_*`, `massfunc`, `B1`, `PT_*`, `interpol` 8-arg, ...).
   190 + 244 compile errors; every cluster struct field is commented
   out; ~20 real bugs (an infinite loop `j<N_z; i++`, pointer
   subtraction, `!(ntomo>0)` dropping the first pair, halo-exclusion
   double bias, write past `Cl[LMAX_NOLIMBER]`, ...). The redshift draft
   (pasted by Vivian) was a commented-out block with 17 confirmed bugs.
   `origin/dev` holds a newer rewrite (Nov-Dec 2025) that does not
   compile; it fixes the file layout and the struct naming.
   => The port is a REWRITE on the house patterns. Old files are a
   physics map; lighthouse's live build is the numbers reference.
2. No public cluster data vectors or covariances exist (Y1 or Y3; the
   Y3 release page still says "when the paper is accepted"). The public
   Y3 redMaPPer catalog reproduces the paper's counts exactly
   (5632 / 6308 / 4551 in z_lambda [0.2,0.4,0.55,0.65]); median
   sigma_z/(1+z) = 0.0060 / 0.0063 / 0.0066.
   => Synthetic data (Y6-like fiducial), Gaussian covariance.
3. Standards (Vivian): cosmo2D.c and halo.c are THE standard (Limber
   `_work` batch APIs, full-sky bin-averaged Legendre projections,
   FKEM FFTLog non-Limber). Flat-sky and the old cluster non-Limber are
   not ported. Cluster code lives in files ending in `_cluster`.
4. Lighthouse's live build differs from the published Y6 model in places
   (flat-sky gamma_t with J2 Hankel, an "all pairs" cluster x source
   hack, halo exclusion code that no paper uses). We follow the paper.

## 2. Target model (2503.13631, equation numbers of that paper)

- Blocks: N (counts, 3 z_lambda x 4 lambda bins), cs (cluster lensing,
  gamma_t -> Sigma through the Y transform, Park+2021), cc (w_cc, auto
  z bin, all richness pairs), cg (w_cg, cluster z bin i x lens bin i),
  plus ss, gs, gg of the 3x2pt. CL+GC = N + cs + cc + cg + gg (lens bins
  1-3); CL+3x2pt adds ss, gs and all lens bins.
- Counts (16): N_iA = Omega_s int dz dV/dz dOmega <phi_i|z> int dM
  P(lambda in A|M,z) dn/dM.
- Mass function and bias: Tinker (halo.c `fnu`, `hb1nu`), M200m.
- MOR (18)-(19): lognormal, <ln lambda|M> = ln lambda_0 + A ln(M/M_piv)
  + B ln((1+z)/1.45), sigma^2 = sigma_int^2 + (e^<ln lambda> - 1) /
  e^{2 <ln lambda>}, M_piv = 5e14 Msun/h (from lighthouse code; not in
  the papers). P(lambda_obs in A|M) is a closed-form erf difference.
- Cluster bias (21): richness-weighted Tinker b_h. Cluster lensing
  spectrum: b_cA P_NL (2-halo) + 1-halo NFW (22) with Bhattacharya+13
  c(M). w_cc, w_cg: linear bias x P_NL.
- Selection bias (23), at data-vector level: Sigma and w_cg times
  b_s1 + b_s2 exp(-theta chi(zbar)/r0); w_cc times its square.
- Magnification: lens C_l fixed, clusters C_c = -2. IA in cluster
  lensing: NLA now (TATT later). No cluster photo-z nuisance.
- Scale cuts: Sigma > 2 Mpc/h, w_cg and w_gg > 8, w_cc > 16 Mpc/h.
- Binning: z_lambda [0.2,0.4,0.55,0.65]; lambda [20,30,45,60,500];
  20 log theta bins 2.5-250'. Lenses MagLim (6 bins Y6, 4 bins Y3),
  sources 4 bins. => des_cluster needs the MagLim machinery back from
  desy1xplanck (stretch DZ2, fixed BMAG).
- Non-Limber: w_gg and w_cc auto spectra (FKEM split, cosmo2D.c design).
- Parameters: Table I of 2503.13631 (MOR 4, selection 3, cosmology 6,
  lens/source nuisances, IA).

## 3. Decisions (Fable review, 2026-09-29; full text in the session
scratchpad `fable_review_port_plan.md`, math in
`fullsky_cluster_projections.md`)

- Cluster radial kernel: volume-only q_i(z) ~ dV/dz <phi_i|z> by default
  (Y1 eq 15; what DES ran); abundance-weighted q_iA is a switch.
- C_cs = int dchi/fK^2 {[W_kappa - W_IA](b_A W_c + C_c W_mag,c) P_NL (C_c = -2)
  + W_kappa W_c P1h_A}; 1-halo without bias, IA or magnification; cluster
  magnification over the full foreground (lighthouse keeps it inside the
  bin).
- Limber everywhere first; FKEM non-Limber for all w_cc pairs of a z bin
  later; w_cg non-Limber as an option (paper: Limber).
- Y transform: Sigma = B(theta) . (T gamma_t) with T = 2S + SD on the
  angular grid, the same 20x20 matrix for every block; Sigma_crit never
  enters; gamma_t at all theta bins regardless of the mask; the last theta
  bin of each cs block always masked; covariance T C T^T. Never port
  lighthouse's T^+ B T. Switch `cluster.ytransform` (0 = Y1 gamma_t).
- Full-sky projections reuse the w_gammat_tomo / w_gg_tomo kernels (same
  theta binning, LMAX = 1e5 suffices for the 1-halo term at 2.5').
- Y1 switches: selection_model (1 = mass-dependent inside the bias
  integral, 2 = scale-dependent on the data vector), magnification,
  include_ia, ytransform. Y1 cluster data are not public.
- Mass range [1e12, 1e16] Msun/h; richness edges [20,30,45,60,500];
  neutrinos: whatever sigma2 reads now, a cb option later.
- Covariance: Gaussian C_l^2 + noise, full-sky bin-averaged, Y on cs;
  counts Poisson + sample variance; N x 2pt = 0 (stated omissions: SSC,
  trispectrum).
- Isolation (Vivian): cluster code only in *_cluster files; no cluster
  fields in structs.h, no cluster functions in the core; core internals
  needed by clusters are copied, not exported.
- Accuracy: cluster tables size themselves from knobs init_accuracy_boost
  already scales (halo_na_lens, halo_nm/hdi ladder, halo_nk_step,
  nz_fine_sampling_factor, N_a, N_ell): no core edit.

## 4. Architecture (contract: cosmolike_core 5be792a)

| file | content |
|---|---|
| `structs_cluster.h/.c` | global `cluster`: keys, switches, probes, richness bins, <phi_i|z>, pairs, MOR / selection parameters, mass limits |
| `redshift_spline_cluster.c/.h` | <phi_i|z> fine-z table, nz_cluster, g_cluster (magnification), pair maps |
| `radial_weights_cluster.c/.h` | W_cluster, W_mag_cluster |
| `halo_cluster.c/.h` | P(A|M,z) erf, n_A(a), b_A(a), P1h_A(k,a), cluster_warmup (replaces cluster_util.c) |
| `cosmo2D_cluster.c/.h` | Limber batches C_cs/C_cc/C_cg, Legendre w, counts |
| `generic_interface_cluster.cpp/.hpp` | setters, block sizes, masked assembly with Y and selection bias, IP-like class for the joint vector (no edit to generic_interface.*) |

Data-vector order (lighthouse): ss, gs, gg, cg, N, cc, cs; N [z][lambda];
cs [(zc,zs) pair][lambda][theta]; cc [z][lambda1<=lambda2][theta];
cg [(zc,zg) pair][lambda][theta].

## 5. Validation

- Ground truth: the independent Python reference
  (`tests/reference/`), 1e-4 per table, delta^T C^-1 delta < 0.01 per
  block (whole-code budget 0.2); cs compared in Y space.
- Lighthouse `.so` (`tests/lighthouse_reference/`): 1-3% sanity check
  (CQUAD 1e-2 integrals, flat-sky gamma_c, bin-restricted magnification).
- House protocol: IEEE default build is the reference; determinism OMP
  1/8; DEBUG and AGGRESSIVE builds; single-threaded warm-up.
- Milestones: M0 = counts + Limber w_cc end to end (chi2 on a synthetic
  vector); M1 = every 4x2pt + N block Limber + covariance + synthetic
  data + mask generator (Sigma > 2, w_cg/w_gg > 8, w_cc > 16 Mpc/h; Y3
  post-cut sanity counts 12/404/149/124/31); M2 = FKEM w_cc; M3 = options.

## 6. Work split (Phase 1b, parallel against the frozen contract)

- A: redshift_spline_cluster.c + radial_weights_cluster.c
- B: halo_cluster.c
- E: cosmo2D_cluster.c
- F: generic_interface_cluster.cpp/.hpp, des_cluster interface.cpp
  bindings, MakefileCosmolike, likelihood combos (combo_4x2pt_N,
  combo_6x2pt_N), dataset keys
- D: des_cluster inputs (Y6 n(z), MagLim restoration from desy1xplanck,
  cluster selection kernels, params_cluster.yaml, mask generator)
- C (running): Python reference + Gaussian covariance
- lighthouse reference outputs (running)
- Fable: code/physics review after Phase 1b, and at M0/M1.

## 7. Status (2026-09-30)

Done (cosmolike_core bugfix, des_cluster bugfix):
- Port: 5be792a contract; efdfc62 redshift_spline_cluster + radial
  weights; 2527d65 halo_cluster; 5032c08 cosmo2D_cluster; 34a5a26
  generic_interface_cluster; e79de2d counts fix; ea9a625 grouped node
  legs; 8c71ddd Tinker alpha switch (default fixed 0.368, what DES ran).
- Core bug found by the validation, fixed: 84c54c9 (g_tomo / g2_tomo /
  g_lens / C_ss ranges started at the UNSHIFTED n(z) top and dropped the
  mass a positive photo-z shift moves above it; bitwise when no shift is
  positive; delta chi2 at dz_s = +0.03: roman_fourier 13.8, roman_real
  0.85, desy1xplanck 0.054, des_y3 0.026, lsst_y1 0.010).
- Growth factor on the dense z grid (historical normalization kept):
  des_cluster 31bdff8 / bb2e1c4 and the six other projects (one commit
  each, 2026-09-30); frozen references not refrozen.
- M0: tests/validation/compare_reference.py - every cluster quantity
  matches the independent Python reference to <= 2e-5 (cluster lensing
  1e-3 in the n(z) tail of the lowest source bins behind the highest
  cluster bin; chi2 cs 3.4e-5 cut / 2.0e-4 uncut); bitwise deterministic
  (OMP 1 vs 8). Lighthouse: counts 2.4%, b_A 0.8% (its 1e-2 CQUAD).
- M1: scripts/make_synthetic_data.py + tests/reference/
  ref_covariance_full.py (joint Gaussian covariance, 2812 entries);
  chi2 ~ 1e-25 at the fiducial for both combos.

- Tinker alpha switch (8c71ddd; default fixed 0.368 = DES); IPCluster
  correlation-matrix PD test (17bcfc7); volume-kernel sharing and P1h
  build only with cluster lensing (cc60e5e, bitwise).
- Synthetic Y6-like data (1b0bf3f, 37a538a): chi2 = 0 at the fiducial for
  both combos; covariance with the exact noise x noise term (0ebb03e);
  S/N 109 (4x2pt + N), 157 (6x2pt + N).
- Accuracy knobs (b45ff48, knob_sweep.py): the cluster combos run
  integration_accuracy 1: delta chi2 vs high accuracy 1.65 at 0 (galaxy
  gammat 0.92, w_gg 0.47: the Y6 MagLim lens n(z) tails reach z = 2.99)
  -> 0.048 at 1; cluster blocks 6e-3 either way; boost / lmax irrelevant.
- Timing (M2, quiet, every parameter jittered, cosmolike only, hdi 1):
  6x2pt + N 0.42 s (4 threads) / 0.34 s (8); 4x2pt + N 0.38 / 0.30;
  cluster lensing 0.14 / 0.11; lighthouse (1 thread): 28.3 s / 26.6 s /
  23.8 s -> 83x / 89x / 225x faster. 4 -> 8 threads only 1.1-1.4x (cause
  not yet measured; see the profile note below).
  NOTE: those numbers were taken at load 15-20 and are ~2.7x too slow;
  the fresh-machine table below supersedes them.
- Timing, fresh machine (after a reboot, load 3-7; tests/validation/
  time_cosmolike.py and time_lighthouse.py; cosmolike only, every
  parameter jittered, hdi 1; seconds). Columns: lighthouse (1 thread) |
  4 threads before -> after the tiled sums (d92b90d) | 8 threads before
  -> after | lighthouse / 8 threads:
    6x2pt + N        22.64 | 0.198 -> 0.152 | 0.138 -> 0.108 | 210x
    4x2pt + N        21.46 | 0.174 -> 0.131 | 0.106 -> 0.084 | 255x
    3x2pt             2.02 | 0.097 -> 0.098 | 0.074 -> 0.068 |  30x
    cluster lensing  19.15 | 0.072 -> 0.047 | 0.045 -> 0.033 | 580x
    w_cc              0.91 | 0.040 -> 0.024 | 0.028 -> 0.021 |  43x
    w_cg              0.85 | 0.029 -> 0.023 | 0.023 -> 0.019 |  45x
    N                 0.11 | 0.013 -> 0.013 | 0.013 -> 0.012 |   9x
  CAMB: 0.63 s at 8 threads, 1.2 s at 4 (6-8x the cosmolike time).
- HOD NaN (include_HOD_GX = 1, des_y3 Y3 / desy1xplanck MagLim; old and
  new code alike): halo.c's HOD a grid now spans the union of the table
  range and every lens bin's [amin_lens, amax_lens] at every refill (the
  magnification-widened amax_lens overran the grid's top end), with the
  clustering photo-z key in its cache (0593d52).
- All seven projects rebuilt against 0593d52 and their suites run
  (test_accuracy_baryons excluded; frozen references untouched in every
  project): des_cluster 3, lsst_y1 49, des_y3 55, desy1xplanck 37,
  roman_kl 41, roman_fourier 37 passed; roman_real 73 passed, 3 failed:
  test_halo frozen bias_norm (2.0e-4), ngal and bgal (4.9e-5) at rtol
  1e-12. Attributed: with the pre-138696e set_cosmo_related bound
  in-process (coarse G) every halo probe is bitwise its frozen value on
  the current core, so the whole move is the dense growth grid (as that
  commit's message records; halo integrals amplify the G error), and the
  HOD a-grid fix and the tiling are bitwise there. roman_real slow tier
  (COCOA_HALO_SLOW=1: test_halo, halo/HOD cache consistency, test_hod_cell,
  halo IA accuracy/race): 55 passed, 8 failed, all TestFrozenReferences
  (the three above plus p_mm, p_my, p_yy, p_gm, p_gg); with the old G
  every frozen halo probe is within 1.4e-14 of its value (rtol 1e-12).
  The halo_reference refreeze waits for Vivian's review of the growth
  deltas (handoff).
- Optimization (d92b90d): a sampled profile (macOS `sample`, 6x2pt + N)
  put the cluster Legendre sums at 20% of the cosmolike thread time;
  4 rows x 4 theta register tiles, bitwise identical to the reference
  loop (full data vector at three points, OMP 1 / 4 / 8), microbenchmark
  39.5 -> 5.6 ms on the cluster lensing block. Timing after (same
  script and load as M2; 4 / 8 threads, seconds): 6x2pt + N 0.367 / 0.296
  (was 0.424 / 0.342), 4x2pt + N 0.310 / 0.266 (0.376 / 0.298), cluster
  lensing 0.106 / 0.087 (0.144 / 0.106), w_cc 0.053 / 0.049 (0.074 /
  0.061), w_cg 0.051 / 0.046 (0.062 / 0.054), N 0.022 / 0.022, 3x2pt
  unchanged (0.23 / 0.19). Against lighthouse (1 thread) at 8 threads:
  6x2pt + N 96x, 4x2pt + N 100x, cluster lensing 274x.
- Profile findings: the Python glue is ~40 ms per evaluation on the
  loaded machine (not the bottleneck). RETRACTED: an earlier version of
  this note said the 4 -> 8 thread scaling is capped by serial code in
  the galaxy non-Limber w_gg path of cosmo2D.c. Reading the code,
  C_gg_tomo_limber_work and cfftlog_ells_p2 are threaded throughout; the
  profile was taken at 2 threads on a loaded machine (the main thread
  waiting on barriers), and with -fomit-frame-pointer the sampler can
  drop the OpenMP-outlined frame, so threaded samples looked serial. The
  scaling is still weak on the fresh machine (6x2pt + N 0.152 -> 0.108 s,
  1.4x); its cause is not yet measured (next: time each block at 1, 2,
  4, 8 threads; classify samples by the OpenMP runtime frames).
- 2026-09-30 (morning), done:
  - cosmo2D.c: the five real-space Legendre sums grouped (4 x 4; xi+- 2
    pairs x 4 theta), a880d36; every frozen example of the six galaxy
    projects and the 6x2pt + N vector bitwise identical at OMP 1 / 4 / 8.
    Re-timed once (6x2pt + N 0.135 s at 4 threads, 0.094 s at 8; 3x2pt
    0.082 / 0.057), but another job overlapped that run: PROVISIONAL,
    re-time on an idle machine.
  - Tests (84c3db4 .. 504cef1): the examples and the suite are the cluster
    combos (example1 = 4x2pt + N, example2 = 6x2pt + N); frozen chi2 and
    race tests, accuracy advisory (delta chi2 0.004 / 0.050), and the
    cache-consistency ladder with the MOR and selection sectors and a
    fresh-process comparison. 8 tests, 5.5 min.
  - Sampler examples (7323262): MCMC (CAMB and EMUL2), PolyChord,
    minimize, profile, Nautilus for both combos; EXAMPLE_EMUL2_EVALUATE3/4
    are the 1000-evaluation perf benchmarks; scripts/make_example_files.py
    regenerates all of them. EMUL2 vs CAMB at equal mnu: delta chi2 1.6
    (counts 1.5% median).
  - Notebook layer (core ef05ab6; b40e524, 8fcc016): cosmo2D_wrapper_
    cluster / halo_wrapper_cluster, bindings, notebook wrappers,
    EXAMPLE_EVALUATE1.ipynb (runs headless from the rebuilt .so).
  - Cocoa: des_cluster and des_y6 registered in set_installation_options
    (skipped by default; des_cluster pinned to branch main).

Open:
- Cluster optimization left: the P1h table (~12% of the cosmolike thread
  time) could tabulate only the Limber k range (~1/3 fewer ln k nodes, not
  bitwise); cc/cg exact tables on N_ell_internal instead of N_ell (a knob
  test, ~3%).
- Refreeze the six projects only if a later change needs it (every suite
  passes against the current frozen references); attribute the
  pre-existing lsst_y1 drift (+0.034) first.
- Phase 4 physics: FKEM non-Limber w_cc, TATT in cluster lensing, a
  cb-neutrino option (Fable P1.2: counts 2-4% at the fiducial, >10% at
  the top of the Omega_nu h^2 prior), Y1 switches end to end, the
  parameter-recovery MCMC.

Backlog (Vivian, 2026-09-30):
- Cluster versions of the data-vector plotting scripts of
  cosmolike_core/cosmolike_notebook_utils (plot_datavectors.py,
  plot_response.py), in their own *_cluster files, so des_cluster gets
  the same notebooks as the other projects.
- Non-Limber w_cc and w_cg on the cosmo2D.c FKEM design: the original
  code runs both non-Limber (Limber is off by -13% / -16% at 225' for the
  first z bin, lowest richness). Now the first physics item.
- combo_4x2pt_N samples DES_PM1..6 although it has no gamma_t block
  (prior-only directions): fix them in its yaml.
- Core: init_binning with a new theta range and the same Ntheta returns
  the old range's real-space values (Legendre kernels keyed on Ntheta and
  Ntable.random, not on the theta range); found by the wrapper work, only
  checked on the cluster block.
- SIMDe vector paths for cosmo2D_cluster.c and halo_cluster.c, as
  cosmo2D.c and halo.c have (AVX2 on x86, NEON on Apple Silicon from one
  source, each behind a fallback guard): the cluster files rely on
  compiler auto-vectorization only. Important for MCMC runs on x86.
- Retire the pair-packed cluster Python bindings; keep the per-bin C++
  bindings the notebooks use (as the 3x2pt has), under the natural names,
  and move tests/validation to them.
- Shared-code bugs to fix: real-space values not refreshed when the theta
  range changes at fixed Ntheta; the HOD tables' missing source n(z) cache
  key; the notebook cosmology helper's coarse growth grid.
- Tests for the galaxy-only likelihoods of this project (cosmic_shear,
  combo_3x2pt, combo_2x2pt), ported from des_y3.
- Core, not cluster: extend the scale-cut code (cosmo2D_scuts.c, today
  cosmic shear and shear x CMB lensing) to galaxy-galaxy lensing and
  galaxy clustering, Limber part only. Reason: some projects use lens =
  source, so the lens sample is not narrow in redshift and a single
  angle-to-k conversion per bin is not a good scale cut.
  With it: the C++ wrappers and bindings, and the projects' notebooks
  that already show the scale-cut diagnostics for cosmic shear expanded
  to show the galaxy-galaxy lensing and clustering ones.
- NOT in this port: the covariance beyond Gaussian (super-sample,
  trispectrum) and the counts x 2pt block. They wait for the port of
  CosmoCov to Cocoa, the next big project.
- The independent check of the model against the paper (a Fable review)
  is LAST: the week of 2026-10-05.
