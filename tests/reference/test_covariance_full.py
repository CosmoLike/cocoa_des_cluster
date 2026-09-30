"""Tests of the joint-vector Gaussian covariance (ref_covariance_full.py).

Run from Cocoa/ (cocoa environment, start_cocoa.sh sourced):

    python -m pytest ./projects/des_cluster/tests/reference/test_covariance_full.py -s

One reference model (CAMB, halo model and the Limber spectra of every
field pair at the Table I fiducial, with the settings of
scripts/make_synthetic_data.py) and one set of band matrices are shared by
the module (about 3-4 minutes). The layout test builds the des_cluster
likelihood through cobaya in a subprocess (no evaluation).

  test_xi_kernels_vs_wigner        G+/- against quadrature of Wigner d^l_{2,+/-2}
  test_layout_*                    joint layout = compiled code (sizes,
                                   starts, pair maps) and = the likelihood mask
  test_symmetry                    the unsymmetrized assembly is symmetric
  test_positive_definite_on_masks  in-memory matrix and data/des_cluster_y6.cov
  test_reduction_to_ref_covariance cluster-only blocks = ref_covariance.py
  test_bruteforce_*                single elements against direct l sums
  test_selection_modes             none / jacobian / signal relations
"""

import json
import os
import subprocess
import sys

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
             "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "3")     # before numpy loads its BLAS

import numpy as np                                                    # noqa: E402
import pytest
from scipy.interpolate import CubicSpline
from scipy.special import eval_jacobi, gammaln, roots_legendre

HERE = os.path.dirname(os.path.abspath(__file__))
PRJ = os.path.dirname(os.path.dirname(HERE))
SCRIPTS = os.path.join(PRJ, "scripts")
DATA = os.path.join(PRJ, "data")
for p in (HERE, SCRIPTS):
    if p not in sys.path:
        sys.path.insert(0, p)

import make_synthetic_data as msd                                    # noqa: E402
from reference_cluster import ClusterReference                       # noqa: E402
from ref_covariance import GaussianCovariance, cc_pairs, counts_covariance  # noqa: E402
from ref_covariance_full import (FullGaussianCovariance, band_matrices,  # noqa: E402
                                 full_covariance, joint_layout, kernels,
                                 xi_pm_kernels)
from ref_projection import theta_edges                                # noqa: E402

TOL_REDUCTION = 1e-12     # |delta C_ij| / sqrt(C_ii C_jj)
TOL_BRUTE = 1e-10         # element vs direct l sum (same interpolation)
TOL_XI_KERNEL = 2e-9      # |delta G_l| / max_l |G_l| per theta bin


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def setup():
    ds = msd.load_dataset()
    lik = msd.load_yaml(os.path.join(PRJ, "likelihood", "combo_6x2pt_N.yaml"))
    settings = msd.reference_settings(ds, lik)
    ref = ClusterReference(None, settings)
    ref.spectra()
    M = band_matrices(ref.spectra()["ells"], ref.edges, settings["lmax"])
    cN, info = counts_covariance(ref)
    return dict(ds=ds, settings=settings, ref=ref, M=M, cN=cN, info=info)


def make_fc(setup, mode="signal", selection="ref", noise="exact", spectra=None):
    ref, s = setup["ref"], setup["settings"]
    B = ref.selection() if isinstance(selection, str) else selection
    return FullGaussianCovariance(
        ref.spectra() if spectra is None else spectra, ref.counts(), ref.cluster.Omega_s,
        ref.edges, ref.T, s["lmax"],
        selection=B, shear_m=ref.params["shear_m"],
        n_lens_arcmin2=ref.settings["n_lens_arcmin2"],
        n_src_arcmin2=ref.settings["n_src_arcmin2"], sigma_e=ref.settings["sigma_e"],
        cg_lens_bins=setup["ds"]["cg_lens_bins"], selection_mode=mode,
        counts_cov=setup["cN"], noise=noise)


@pytest.fixture(scope="module")
def cov_signal(setup):
    fc = make_fc(setup, "signal")
    return fc, fc.full(setup["M"])


@pytest.fixture(scope="module")
def cov_none(setup):
    fc = make_fc(setup, "none")
    return fc, fc.full(setup["M"])


def scaled(C):
    d = np.sqrt(np.abs(np.diag(C)))
    return np.outer(d, d)


def effective_mask(path, lay):
    m = np.loadtxt(path)[:, 1] > 0.5
    return msd.effective_mask(m, lay)


def obs_index(lay, block, label):
    for o, ob in enumerate(lay["obs"]):
        if ob[0] == block and ob[3] == label:
            return o
    raise KeyError((block, label))


def spline_sum(ells, nodes, Ki, Kj, lmax):
    """sum_{l=1}^{lmax-1} K_i(l) K_j(l) spline(nodes)(l)/(2l+1), directly."""
    l = np.arange(1, lmax, dtype=float)
    g = CubicSpline(np.log(ells), nodes)(np.log(l))
    return float(np.sum(Ki[1:] * Kj[1:] * g / (2.0 * l + 1.0)))


# ----------------------------------------------------------------------
# kernels
# ----------------------------------------------------------------------
def test_xi_kernels_vs_wigner():
    """G+/-_i(l) = (2l+1)/(4 pi) (l+2)!/(l-2)!/[l(l+1)]^2 <d^l_{2,+/-2}>_bin,
    d^l_{22} = ((1+x)/2)^2 P^{(0,4)}_{l-2}(x), d^l_{2,-2} = ((1-x)/2)^2
    P^{(4,0)}_{l-2}(x) (Jacobi), bin average by Gauss-Legendre in x."""
    edges = theta_edges(20, 2.5, 250.0)
    lmax = 75000
    Gp, Gm = xi_pm_kernels(edges, lmax)
    xg, wg = roots_legendre(400)
    xe = np.cos(edges)
    worst = 0.0
    for i in (0, 5, 10, 19):
        a, b = xe[i + 1], xe[i]
        x = 0.5 * (b - a) * xg + 0.5 * (b + a)
        w = 0.5 * (b - a) * wg
        for l in (2, 3, 5, 10, 30, 100, 300, 1000, 3000, 10000, 30000, lmax - 1):
            fac = ((2 * l + 1) / (4 * np.pi) * np.exp(gammaln(l + 3) - gammaln(l - 1))
                   / (l * (l + 1.0)) ** 2)
            dp = ((1 + x) / 2) ** 2 * eval_jacobi(l - 2, 0, 4, x)
            dm = ((1 - x) / 2) ** 2 * eval_jacobi(l - 2, 4, 0, x)
            for G, d in ((Gp, dp), (Gm, dm)):
                exact = fac * np.sum(w * d) / (b - a)
                worst = max(worst, abs(G[i, l] - exact) / np.max(np.abs(G[i])))
    print(f"\nxi+/- kernels vs Wigner d: max |dG|/max|G| = {worst:.2e}")
    assert worst < TOL_XI_KERNEL


# ----------------------------------------------------------------------
# layout
# ----------------------------------------------------------------------
LAYOUT_WORKER = r"""
import json, os, sys, tempfile
sys.path.insert(0, {scripts!r})
import make_synthetic_data as msd
import cosmolike_des_cluster_interface as ci
from cobaya.model import get_model
tmp = tempfile.mkdtemp(prefix="des_cluster_layout_")
name = msd.write_placeholder_dataset(tmp)
os.chdir(os.environ["ROOTDIR"])
model = get_model(msd.cobaya_info("6x2pt_N", tmp, name))
out = dict(sizes=list(ci.compute_data_vector_cluster_sizes()),
           starts=list(ci.compute_data_vector_cluster_starts()),
           mask=list(ci.get_mask_cluster()),
           gs=[list(map(int, x)) for x in ci.get_gs_redshift_bins()],
           cs=[list(map(int, x)) for x in ci.get_cs_redshift_bins()],
           cg=[list(map(int, x)) for x in ci.get_cg_redshift_bins()],
           cc=[list(map(int, x)) for x in ci.get_cc_richness_bins()])
print("LAYOUT_JSON " + json.dumps(out))
"""


@pytest.fixture(scope="module")
def compiled_layout():
    if "ROOTDIR" not in os.environ:
        pytest.skip("start_cocoa.sh not sourced")
    code = LAYOUT_WORKER.format(scripts=SCRIPTS)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                                VECLIB_MAXIMUM_THREADS="1"))
    line = [x for x in r.stdout.splitlines() if x.startswith("LAYOUT_JSON ")]
    assert r.returncode == 0 and line, r.stdout[-3000:] + r.stderr[-3000:]
    return json.loads(line[0][len("LAYOUT_JSON "):])


def test_layout_sizes_starts(compiled_layout):
    ds = msd.load_dataset()
    lay = joint_layout(ds["cluster_ntomo"], len(ds["richness_edges"]) - 1, ds["source_ntomo"],
                       ds["lens_ntomo"], ds["n_theta"], ds["cg_lens_bins"])
    blocks = ("ss", "gs", "gg", "cg", "N", "cc", "cs")
    assert compiled_layout["sizes"] == [lay["sizes"][b] for b in blocks]
    assert compiled_layout["starts"] == [lay["starts"][b] for b in blocks]
    assert lay["ndata"] == 2812
    # every 2pt entry and every count has exactly one joint index
    allidx = np.concatenate([lay["index"].ravel(), lay["N_index"].ravel()])
    assert np.array_equal(np.sort(allidx), np.arange(lay["ndata"]))
    # the all-ones mask keeps everything but the last theta bin of each cs row
    me = msd.effective_mask(np.ones(lay["ndata"], dtype=bool), lay)
    assert np.array_equal(np.asarray(compiled_layout["mask"], dtype=bool), me)


def test_layout_pair_maps(compiled_layout):
    ds = msd.load_dataset()
    lay = joint_layout(ds["cluster_ntomo"], len(ds["richness_edges"]) - 1, ds["source_ntomo"],
                       ds["lens_ntomo"], ds["n_theta"], ds["cg_lens_bins"])
    gs = [ob[3] for ob in lay["obs"] if ob[0] == "gs"]
    assert [tuple(x) for x in compiled_layout["gs"]] == gs
    cs = [(ob[3][0], ob[3][1]) for ob in lay["obs"] if ob[0] == "cs" and ob[3][2] == 0]
    assert [tuple(x) for x in compiled_layout["cs"]] == cs
    assert [tuple(x) for x in compiled_layout["cg"]] == lay["cg_pairs"]
    assert [tuple(x) for x in compiled_layout["cc"]] == cc_pairs(4)


def test_layout_masks_match_blocks():
    """The scale-cut masks switch whole probes off where the combos do."""
    ds = msd.load_dataset()
    lay = joint_layout(ds["cluster_ntomo"], len(ds["richness_edges"]) - 1, ds["source_ntomo"],
                       ds["lens_ntomo"], ds["n_theta"], ds["cg_lens_bins"])
    m4 = np.loadtxt(os.path.join(DATA, msd.MASKS["4x2pt_N"]))[:, 1]
    for b in ("ss", "gs"):
        s = lay["starts"][b]
        assert m4[s:s + lay["sizes"][b]].sum() == 0


# ----------------------------------------------------------------------
# matrix properties
# ----------------------------------------------------------------------
def test_symmetry(setup):
    fc = make_fc(setup, "signal")
    c2 = fc.twopoint(setup["M"], symmetrize=False)
    asym = np.max(np.abs(c2 - c2.T) / np.maximum(scaled(c2), 1e-300))
    print(f"\nmax |C - C^T|/sqrt(C_ii C_jj) before symmetrization: {asym:.2e}")
    assert asym < 1e-13


def test_positive_definite_on_masks(cov_signal):
    fc, cov = cov_signal
    lay = fc.layout
    covs = {"in-memory": cov}
    path = os.path.join(DATA, msd.COV_FILE)
    if os.path.exists(path):
        covs["file"] = msd.read_covariance(path, lay["ndata"])
    for label, C in covs.items():
        for combo, mf in msd.MASKS.items():
            me = effective_mask(os.path.join(DATA, mf), lay)
            idx = np.where(me)[0]
            Cm = C[np.ix_(idx, idx)]
            corr = Cm / scaled(Cm)
            ev = np.linalg.eigvalsh(corr)
            print(f"\n{label} {combo}: {idx.size} points, corr eigenvalues "
                  f"[{ev[0]:.3e}, {ev[-1]:.3e}], cond {ev[-1] / ev[0]:.3e}")
            assert ev[0] > 0
            np.linalg.cholesky(Cm)


def test_counts_block_and_zero_cross(cov_signal, setup):
    fc, cov = cov_signal
    lay = fc.layout
    iN = lay["N_index"].ravel()
    assert np.array_equal(cov[np.ix_(iN, iN)], setup["cN"])
    other = np.setdiff1d(np.arange(lay["ndata"]), iN)
    assert not np.any(cov[np.ix_(iN, other)])


# ----------------------------------------------------------------------
# reduction to ref_covariance.py on the cluster blocks
# ----------------------------------------------------------------------
def test_reduction_to_ref_covariance(setup):
    """noise = "lsum" (the LMAX-truncated noise sum of ref_covariance.py)."""
    fc = make_fc(setup, "none", noise="lsum")
    cov = fc.full(setup["M"])
    ref, s = setup["ref"], setup["settings"]
    lay = fc.layout
    cg = setup["ds"]["cg_lens_bins"]
    sp = ref.spectra()
    sub = dict(sp, C_cg=sp["C_cg"][:, :, cg], C_gg=sp["C_gg"][np.ix_(cg, cg)],
               C_gs=sp["C_gs"][cg])
    gc = GaussianCovariance(sub, ref.counts(), ref.cluster.Omega_s, ref.edges, ref.T,
                            s["lmax"],
                            n_lens_arcmin2=[ref.settings["n_lens_arcmin2"][g] for g in cg],
                            n_src_arcmin2=ref.settings["n_src_arcmin2"],
                            sigma_e=ref.settings["sigma_e"], lens_bins=list(range(len(cg))))
    c2, obs = gc.twopoint(y_transform=True)

    def label(o):
        # ref_covariance labels cg by (i, A); the joint layout by (i, g, A)
        return (o[3][0], cg[o[3][0]], o[3][1]) if o[0] == "cg" else o[3]

    idx = np.concatenate([lay["index"][obs_index(lay, o[0], label(o))] for o in obs])
    mine = cov[np.ix_(idx, idx)]
    ok = np.outer(np.diag(c2) > 0, np.diag(c2) > 0)
    d = np.max(np.abs(mine - c2)[ok] / scaled(c2)[ok])
    zero = np.max(np.abs(mine[~ok])) if np.any(~ok) else 0.0
    print(f"\ncluster 2pt blocks vs ref_covariance.py: max |dC|/sqrt(CiiCjj) = {d:.2e}; "
          f"rows with zero variance (last cs bin) max |C| = {zero:.1e}")
    assert d < TOL_REDUCTION
    assert zero == 0.0
    iN = lay["N_index"].ravel()
    cN_ref, _ = counts_covariance(ref)
    assert np.max(np.abs(cov[np.ix_(iN, iN)] - cN_ref) / scaled(cN_ref)) < TOL_REDUCTION


# ----------------------------------------------------------------------
# brute-force elements
# ----------------------------------------------------------------------
def test_bruteforce_xi_xi(setup, cov_signal):
    """xi+(s1 s1) x xi+(s1 s1) and xi+ x xi-: E-mode 2 (S + N)^2 +/- B-mode
    2 N^2. The pieces with signal are summed over every integer l < LMAX;
    the noise x noise piece (2 N^2 E + 2 N^2 B for ++, 0 for +-) is the
    all-l value, delta_ij/(8 pi^2 Delta x_i f_sky)."""
    fc, cov = cov_signal
    ref, lay = setup["ref"], fc.layout
    sp, lmax = ref.spectra(), setup["settings"]["lmax"]
    K = kernels(ref.edges, lmax)
    j = 1
    N = fc.sigma_e**2 / fc.n_src[j]
    S = sp["C_ss"][j, j]
    op = obs_index(lay, "ss", ("xip", j, j))
    om = obs_index(lay, "ss", ("xim", j, j))
    xe = np.cos(ref.edges)
    for ti, tj in ((3, 11), (3, 3)):
        for o2, K2, sgn in ((op, K["xip"], 1.0), (om, K["xim"], -1.0)):
            nodes = 2.0 * S**2 + 4.0 * S * N
            direct = spline_sum(sp["ells"], nodes, K["xip"][ti], K2[tj], lmax) / fc.fsky
            if ti == tj and sgn > 0:
                direct += 4.0 * N**2 / (8 * np.pi**2 * (xe[ti] - xe[ti + 1])) / fc.fsky
            el = cov[lay["index"][op][ti], lay["index"][o2][tj]]
            rel = abs(el / direct - 1.0)
            print(f"\nxi+ x {'xi+' if sgn > 0 else 'xi-'} (s{j}s{j}, bins {ti},{tj}): "
                  f"assembled {el:.10e}, direct {direct:.10e}, rel {rel:.1e}")
            assert rel < TOL_BRUTE
    ti, tj = 3, 11
    # interpolating the spectrum instead of the product: an accuracy statement
    l = np.arange(1, lmax, dtype=float)
    Sl = CubicSpline(np.log(sp["ells"]), S)(np.log(l))
    alt = np.sum(K["xip"][ti, 1:] * K["xip"][tj, 1:] * (2 * Sl**2 + 4 * Sl * N)
                 / (2 * l + 1)) / fc.fsky
    el = cov[lay["index"][op][ti], lay["index"][op][tj]]
    print(f"spline of C_l instead of the product: rel {abs(el / alt - 1):.1e}")
    assert abs(el / alt - 1.0) < 1e-3


def test_bruteforce_gs_cs(setup, cov_signal):
    """Cov(gamma_t[g1 s2](theta_i), Sigma[c(1,A=1) s2](theta_j)) =
    B_1(theta_j) sum_k T_jk sum_l Pg_i Pg_k [C_gc (C_ss + N_s) + C_gs C_sc]/((2l+1) f_sky)
    (the cluster leg never meets its own shot noise here, so it always
    carries B)."""
    fc, cov = cov_signal
    ref, lay = setup["ref"], fc.layout
    sp, lmax = ref.spectra(), setup["settings"]["lmax"]
    K = kernels(ref.edges, lmax)
    g, j, i, A = 1, 2, 1, 1
    Ns = fc.sigma_e**2 / fc.n_src[j]
    nodes = (sp["C_cg"][i, A, g] * (sp["C_ss"][j, j] + Ns)
             + sp["C_gs"][g, j] * sp["C_cs"][i, A, j])
    ti, tj = 7, 9
    gam = np.array([spline_sum(sp["ells"], nodes, K["gt"][ti], K["gt"][k], lmax)
                    for k in range(fc.nt)]) / fc.fsky
    direct = fc.B[i, tj] * (1.0 + fc.m[j]) * (ref.T[tj] @ gam) * (1.0 + fc.m[j])
    o1 = obs_index(lay, "gs", (g, j))
    o2 = obs_index(lay, "cs", (i, j, A))
    el = cov[lay["index"][o1][ti], lay["index"][o2][tj]]
    rel = abs(el / direct - 1.0)
    print(f"\ngs x cs element: assembled {el:.10e}, direct {direct:.10e}, rel {rel:.1e}")
    assert rel < TOL_BRUTE


def test_bruteforce_cc_selection(setup, cov_signal):
    """w_cc(i, A, A) auto: 2 (B_i(t) B_i(t') C + N)^2 per l (signal mode);
    the 2 N^2 piece is the all-l value (zero off the diagonal)."""
    fc, cov = cov_signal
    ref, lay = setup["ref"], fc.layout
    sp, lmax = ref.spectra(), setup["settings"]["lmax"]
    K = kernels(ref.edges, lmax)
    i, A = 2, 3
    S = sp["C_cc"][i, A, i, A]
    N = fc.Omega_s / fc.N[i, A]
    xe = np.cos(ref.edges)
    o = obs_index(lay, "cc", (i, A, A))
    for ti, tj in ((12, 15), (12, 12)):
        bi, bj = fc.B[i, ti], fc.B[i, tj]
        direct = (2 * bi**2 * bj**2 * spline_sum(sp["ells"], S**2, K["w"][ti], K["w"][tj], lmax)
                  + 4 * bi * bj * N * spline_sum(sp["ells"], S, K["w"][ti], K["w"][tj], lmax)
                  ) / fc.fsky
        if ti == tj:
            direct += 2 * N**2 / (8 * np.pi**2 * (xe[ti] - xe[ti + 1])) / fc.fsky
        el = cov[lay["index"][o][ti], lay["index"][o][tj]]
        rel = abs(el / direct - 1.0)
        print(f"\ncc auto element with selection (bins {ti},{tj}): assembled {el:.10e}, "
              f"direct {direct:.10e}, rel {rel:.1e}")
        assert rel < TOL_BRUTE


def test_pure_noise_is_pair_count_variance(setup):
    """With every spectrum set to zero the covariance is noise x noise
    alone and must equal the real-space pair-count variances:
      xi+/- auto (s_j s_j): 2 sigma_e^4 / (n_j^2 Omega pi Dx), cross: half;
      gamma_t (g_k s_j): sigma_e^2/(n_k n_j Omega 2 pi Dx);
      w auto: 1/(n^2 Omega pi Dx);  w_cg: 1/(n_c n_g Omega 2 pi Dx);
      Sigma: T diag(sigma_e^2/(n_c n_s Omega 2 pi Dx)) T^T,
    Dx = cos(theta_lo) - cos(theta_hi); diagonal in theta except through T."""
    ref = setup["ref"]
    zero = {k: (np.zeros_like(v) if k.startswith("C_") else v) for k, v in ref.spectra().items()}
    fc = make_fc(setup, "signal", spectra=zero)
    c2 = fc.twopoint(setup["M"]).reshape(len(fc.layout["obs"]), fc.nt,
                                         len(fc.layout["obs"]), fc.nt)
    xe = np.cos(ref.edges)
    Dx = xe[:-1] - xe[1:]
    Om = fc.Omega_s
    se2 = fc.sigma_e**2
    Nc = fc.N / Om
    worst = 0.0
    for o, (block, legs, kind, lab) in enumerate(fc.layout["obs"]):
        if block == "ss":
            _, i, j = lab
            v = se2**2 / (fc.n_src[i] * fc.n_src[j] * Om * np.pi * Dx) * (2 if i == j else 1)
        elif block == "gs":
            k, j = lab
            v = se2 / (fc.n_lens[k] * fc.n_src[j] * Om * 2 * np.pi * Dx)
        elif block == "gg":
            (k,) = lab
            v = 1 / (fc.n_lens[k] ** 2 * Om * np.pi * Dx)
        elif block == "cg":
            i, g, A = lab
            v = 1 / (Nc[i, A] * fc.n_lens[g] * Om * 2 * np.pi * Dx)
        elif block == "cc":
            i, A, B = lab
            v = 1 / (Nc[i, A] * Nc[i, B] * Om * (np.pi if A == B else 2 * np.pi) * Dx)
        else:
            i, j, A = lab
            v0 = se2 / (Nc[i, A] * fc.n_src[j] * Om * 2 * np.pi * Dx)
            expect = ref.T @ np.diag(v0) @ ref.T.T
            got = c2[o, :, o, :]
            worst = max(worst, np.max(np.abs(got - expect)) / np.max(np.abs(expect)))
            continue
        got = c2[o, :, o, :]
        worst = max(worst, np.max(np.abs(got - np.diag(v))) / np.max(v))
    print(f"\npure noise vs pair-count variances: max |dC|/max C = {worst:.2e}")
    assert worst < 1e-12


# ----------------------------------------------------------------------
# selection modes
# ----------------------------------------------------------------------
def test_selection_modes(setup, cov_signal, cov_none):
    fcs, cs = cov_signal
    fcn, cn = cov_none
    # jacobian = D (none) D with D = the model's data-vector factor
    fcj = make_fc(setup, "jacobian")
    cj = fcj.full(setup["M"])
    lay = fcj.layout
    D = np.ones(lay["ndata"])
    for o, ob in enumerate(lay["obs"]):
        f = np.ones(fcj.nt)
        for leg in ob[1]:
            if leg[0] == "c":
                f = f * fcj.B[leg[1]]
            elif leg[0] == "s":
                f = f * (1.0 + fcj.m[leg[1]])
        D[lay["index"][o]] = f
    iN = lay["N_index"].ravel()
    D[iN] = 1.0
    ok = scaled(cj) > 0
    assert np.max(np.abs(cj - D[:, None] * cn * D[None, :])[ok] / scaled(cj)[ok]) < 1e-13
    # signal with B = 1 is the selection-free covariance
    fc1 = make_fc(setup, "signal", selection=np.ones_like(fcs.B))
    c1 = fc1.full(setup["M"])
    ok = scaled(cn) > 0
    assert np.max(np.abs(c1 - cn)[ok] / scaled(cn)[ok]) < 1e-13
    # the signal mode sits between none and jacobian on the cluster diagonals
    dg = np.diag(cs)
    assert np.all(dg <= np.diag(cj) * (1 + 1e-12)) and np.all(dg >= np.diag(cn) * (1 - 1e-12))


def test_full_covariance_wrapper(setup, cov_signal):
    """full_covariance(ref) = the class assembled here."""
    fc, cov = cov_signal
    c2, info = full_covariance(setup["ref"], "signal", cg_lens_bins=setup["ds"]["cg_lens_bins"],
                               M=setup["M"])
    ok = scaled(cov) > 0
    assert np.max(np.abs(c2 - cov)[ok] / scaled(cov)[ok]) < 1e-13
