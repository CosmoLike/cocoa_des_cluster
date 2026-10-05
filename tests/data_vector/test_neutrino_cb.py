"""Neutrino-aware halos: cb variance at each redshift, counts and bias.

Massive neutrinos free-stream out of halos. The production halo model
therefore uses the cold dark matter plus baryon spectrum P_cb(k,z), with
rho_cb = rho_crit (Omega_m - Omega_nu) in the smoothing radius and mass
function. Integrating the spectrum at each z retains its scale-dependent
growth; multiplying a present-day variance by one D(z) would lose it.

Compare both diagnostic variance fields with an independent Simpson
integral; test the massless limit, input-cache invalidation, the physical
count/bias response, thread determinism and the missing-cb guard.
The total-matter comparison supplies P_cb=P_m and Omega_nu=0 to the halo
inputs while keeping the background fixed; there is no halo-field switch.

The test uses the reference's CAMB tables (three degenerate neutrinos)
through the same set_cosmology interface as the likelihood. It runs in
its own process because other cluster tests initialize different dataset
dimensions. The missing-spectrum check needs a second child process:
the C guard deliberately terminates that process with an error message.

Run from Cocoa/ after activating its environment and start_cocoa.sh:
    python -m pytest projects/des_cluster/tests/data_vector/test_neutrino_cb.py
"""

import os

# OpenMP reads OMP_NUM_THREADS when the compiled libraries load, so
# this must run before ANY cobaya/cosmolike import in the process.
os.environ["OMP_NUM_THREADS"] = "4"

import subprocess
import sys
import tempfile
import unittest

# The tests folder is not a package; put it on the import path so the
# shared harness resolves no matter where pytest was launched from.
TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TESTS_DIR)
sys.path.insert(0, os.path.join(TESTS_DIR, "validation"))
sys.path.insert(0, os.path.join(TESTS_DIR, "reference"))
import cocoa_test_utils as u

# The two neutrino densities of check 3: the Table I fiducial of arXiv
# 2503.13631 and the top of the DES Y6 prior on Omega_nu h^2, flat
# [0.0006, 0.00644].
OMNUH2_FIDUCIAL = 0.00083
OMNUH2_TOP = 0.00644

# Masses of check 3 (M_sun/h): the range the richness bins select, from
# the lowest richness bin (~1e14) to the heaviest halos, with 1e13 below.
SIGMA2_MASSES = (1.0e13, 3.0e13, 1.0e14, 3.0e14, 1.0e15)

# Tolerance includes the C table interpolation in ln M and scale factor; the
# reference integrates a 20001-node Simpson rule and a cubic spline in
# ln M. Both read the same CAMB table of P(k).
SIGMA2_RTOL = 1.0e-4

# Bands of check 4 at the fiducial (three degenerate states): the
# closed-form estimate of the neutrino design study gives counts +1% to
# +4% and the bias of the richness bins -0.6% to -0.9%. The bands are
# wider than the estimate on both sides; they catch a sign error or a
# missing factor (a cb field in sigma but rho_m in dn/dM makes the
# counts fall), not a percent-level change of the model.
COUNTS_RATIO_BAND = (1.002, 1.08)
BIAS_RATIO_BAND = (0.97, 0.999)

# The galaxy blocks of the joint vector (no halo model in them).
GALAXY_BLOCKS = ("ss", "gs", "gg")
BLOCKS = ("ss", "gs", "gg", "cg", "N", "cc", "cs")

# The message of the C abort of check 6 (cosmo3D.c, sigma2).
ABORT_MESSAGE = "cb variance needs P_cb"

# Flag of the child process of check 6.
ABORT_CHILD_FLAG = "--abort-child"


def build_state(omnuh2_values):
    """Initialize cosmolike for 4x2pt + N and run the reference CAMB.

    The init chain is the likelihood's (compare_reference.init_c), with
    an all-ones mask so every entry of the joint vector is computed
    (the model is not evaluated at masked entries) and a unit diagonal
    covariance, which the interface needs but these checks never read.

    Arguments:
      omnuh2_values = the Omega_nu h^2 values to run CAMB at (tuple of
                      floats)

    Returns:
      (ci, cfg, tables, cosmologies, workdir): the compiled interface,
      the compare_reference configuration (cb halos on both sides),
      {omnuh2: the set_cosmology arrays of compare_reference.
      cosmology_inputs on the likelihood grids}, {omnuh2: the reference
      Cosmology}, and the scratch directory of the dummy data files.
    """
    import numpy as np
    import compare_reference as cr
    from reference_cluster import FIDUCIAL, COSMO_KEYS
    from ref_cosmology import Cosmology

    cfg = cr.load_config()
    workdir = tempfile.mkdtemp(prefix="des_cluster_neutrino_cb_")
    ci, info = cr.init_c(cfg, workdir, threads=4)
    ones = np.ones(info["ndata"], dtype=int)
    ci.init_data_cluster(info["covf"], cr.write_mask(workdir, "ones.mask", ones),
                         info["dvf"])
    cr.set_nuisance_c(ci, cfg, dict(FIDUCIAL))

    tables = {}
    cosmologies = {}
    for omnuh2 in omnuh2_values:
        params = {}
        for key in COSMO_KEYS:
            params[key] = FIDUCIAL[key]
        params["Omega_nu_h2"] = omnuh2
        print(f"  CAMB at Omega_nu h^2 = {omnuh2} (three degenerate "
              "states) ...", flush=True)
        cosmo = Cosmology(params)
        cosmologies[omnuh2] = cosmo
        tables[omnuh2] = cr.cosmology_inputs(cosmo, cfg, "production")
    return ci, cfg, tables, cosmologies, workdir


def set_cosmology(ci, R, omegan2, lnPL_cb):
    """ci.set_cosmology with the tables R and the given neutrino inputs.

    Arguments:
      ci      = the compiled interface
      R       = the arrays of compare_reference.cosmology_inputs
      omegan2 = Omega_nu h^2 handed over (float)
      lnPL_cb = the ln P_cb table handed over (flattened as lnP_linear),
                or [] for none

    Returns:
      nothing; cosmolike's cosmology is set.
    """
    ci.set_cosmology(omegam=float(R["cin_omegam"]), omegab=float(R["cin_omegab"]),
                     H0=float(R["cin_H0"]), log10k_2D=R["cin_log10k_2D"],
                     z_2D=R["cin_z_2D"], lnP_linear=R["cin_lnPL"],
                     lnP_nonlinear=R["cin_lnPNL"], G=R["cin_G"], z_G=R["cin_z_G"],
                     z_1D=R["cin_z_1D"], chi=R["cin_chi"], omegan2=omegan2,
                     lnP_linear_cb=lnPL_cb)


def evaluate(ci):
    """The full joint data vector and the halo tables at fixed points.

    Returns:
      dict with "dv" (the joint vector, every entry: the mask is all
      ones), "sigma2" (sigma^2(M) at SIGMA2_MASSES), "N" (the counts,
      (cluster z bin, richness bin)) and "bias" (b_A at the middle of
      each cluster z bin, (richness bin, z bin)).
    """
    import numpy as np

    dv = np.array(ci.compute_data_vector_cluster_masked())
    sigma2 = []
    for mass in SIGMA2_MASSES:
        sigma2.append(ci.sigma2(M=mass, field=1))
    counts = np.array(ci.N_cluster_tomo()).T
    bias = []
    for richness_bin in range(counts.shape[1]):
        row = []
        for z in (0.3, 0.475, 0.6):
            row.append(ci.bcl_richness(1.0/(1.0 + z), richness_bin))
        bias.append(row)
    return dict(dv=dv, sigma2=np.array(sigma2), N=counts, bias=np.array(bias))


def block_slices(ci):
    """{block name: slice of the joint vector}."""
    sizes = ci.compute_data_vector_cluster_sizes()
    starts = ci.compute_data_vector_cluster_starts()
    out = {}
    for n, name in enumerate(BLOCKS):
        out[name] = slice(int(starts[n]), int(starts[n] + sizes[n]))
    return out


def abort_child():
    """The cb reader must stop with a clear error when P_cb is absent."""
    ci, cfg, tables, cosmologies, workdir = build_state((OMNUH2_FIDUCIAL,))
    R = tables[OMNUH2_FIDUCIAL]
    set_cosmology(ci, R, omegan2=float(R["cin_omnuh2"]), lnPL_cb=[])
    value = ci.sigma2(M=1.0e14, field=1)
    print(f"sigma2 returned {value} without a P_cb table", flush=True)
    sys.exit(0)


class TestNeutrinoCB(unittest.TestCase):
    """The cold dark matter + baryon halo field on 4x2pt + N."""

    @classmethod
    def setUpClass(cls):
        u.require_cocoa_environment()

    @u.own_process(__file__)
    def test_neutrino_cb(self):
        import numpy as np
        from ref_halo import HaloModel

        ci, cfg, tables, cosmologies, workdir = build_state(
            (OMNUH2_FIDUCIAL, OMNUH2_TOP))
        slices = block_slices(ci)
        R = tables[OMNUH2_FIDUCIAL]
        omegan2 = float(R["cin_omnuh2"])
        lnPL_cb = R["cin_lnPL_cb"]

        # --- 1. total variance is independent of the cb inputs ---
        set_cosmology(ci, R, omegan2=0.0, lnPL_cb=[])
        plain = np.array([ci.sigma2(M=mass, field=0) for mass in SIGMA2_MASSES])
        set_cosmology(ci, R, omegan2=omegan2, lnPL_cb=lnPL_cb)
        with_inputs = np.array([ci.sigma2(M=mass, field=0) for mass in SIGMA2_MASSES])
        np.testing.assert_array_equal(plain, with_inputs)

        # --- 2. the massless limit, including evolution ---
        set_cosmology(ci, R, omegan2=0.0, lnPL_cb=R["cin_lnPL"].copy())
        total = evaluate(ci)
        for a in (1.0, 0.8, 0.55, 0.3):
            for mass in SIGMA2_MASSES:
                self.assertEqual(ci.sigma2(M=mass, a=a, field=0),
                                 ci.sigma2(M=mass, a=a, field=1))

        # --- 3. sigma^2(M) against the Python reference ---
        print("  3. sigma^2(M): C / reference - 1", flush=True)
        worst = 0.0
        for omnuh2 in (OMNUH2_FIDUCIAL, OMNUH2_TOP):
            Rn = tables[omnuh2]
            for field, hmf_matter in ((1, "cb"), (0, "tot")):
                set_cosmology(ci, Rn, omegan2=float(Rn["cin_omnuh2"]),
                              lnPL_cb=Rn["cin_lnPL_cb"])
                reference = HaloModel(cosmologies[omnuh2], hmf_matter=hmf_matter)
                for a in (1.0, 0.8, 0.55, 0.3):
                    ratios = []
                    for mass in SIGMA2_MASSES:
                        sigma_ref = reference.sig.sigma(np.log(mass), z=1.0/a-1.0)
                        ratios.append(ci.sigma2(M=mass, a=a, field=field)/sigma_ref**2-1.0)
                    ratios = np.array(ratios)
                    worst = max(worst, float(np.max(np.abs(ratios))))
                    print(f"     Omega_nu h^2 = {omnuh2}, {hmf_matter}, a={a}: "
                          + " ".join(f"{r:+.2e}" for r in ratios), flush=True)
                    self.assertLess(float(np.max(np.abs(ratios))), SIGMA2_RTOL)

        # --- 4. direction and size at the fiducial ---
        # A diagnostic total-field surrogate uses P_cb=P_m and Omega_nu=0
        # in the halo inputs. The background and non-halo spectra are fixed.
        set_cosmology(ci, R, omegan2=0.0, lnPL_cb=R["cin_lnPL"])
        total = evaluate(ci)
        set_cosmology(ci, R, omegan2=omegan2, lnPL_cb=lnPL_cb)
        cb = evaluate(ci)
        counts_ratio = cb["N"]/total["N"]
        bias_ratio = cb["bias"]/total["bias"]
        print(f"  4. cb / total at the fiducial: counts {counts_ratio.min():.4f}"
              f" .. {counts_ratio.max():.4f}, richness-bin bias "
              f"{bias_ratio.min():.4f} .. {bias_ratio.max():.4f}", flush=True)
        self.assertGreater(float(counts_ratio.min()), COUNTS_RATIO_BAND[0])
        self.assertLess(float(counts_ratio.max()), COUNTS_RATIO_BAND[1])
        self.assertGreater(float(bias_ratio.min()), BIAS_RATIO_BAND[0])
        self.assertLess(float(bias_ratio.max()), BIAS_RATIO_BAND[1])
        for name in GALAXY_BLOCKS:
            same = np.array_equal(cb["dv"][slices[name]], total["dv"][slices[name]])
            self.assertTrue(same, f"block {name} moved with the halo field")

        # --- 5. the cache keys ---
        # a new Omega_nu h^2 alone: R(M) moves, so sigma^2 must
        set_cosmology(ci, R, omegan2=1.5*omegan2, lnPL_cb=lnPL_cb)
        moved_omnuh2 = evaluate(ci)
        # a new P_cb table alone
        set_cosmology(ci, R, omegan2=omegan2, lnPL_cb=lnPL_cb + 0.01)
        moved_table = evaluate(ci)
        # Return to the fiducial after both independent input changes.
        set_cosmology(ci, R, omegan2=omegan2, lnPL_cb=lnPL_cb)
        back = evaluate(ci)
        print("  5. cache: Omega_nu h^2 alone moves sigma^2 "
              f"{not np.array_equal(moved_omnuh2['sigma2'], cb['sigma2'])}; "
              "P_cb alone moves sigma^2 "
              f"{not np.array_equal(moved_table['sigma2'], cb['sigma2'])}; "
              f"input roundtrip bitwise {np.array_equal(back['dv'], cb['dv'])}",
              flush=True)
        self.assertFalse(np.array_equal(moved_omnuh2["sigma2"], cb["sigma2"]),
                         "a new Omega_nu h^2 left sigma^2 unchanged (stale cache)")
        self.assertFalse(np.array_equal(moved_table["sigma2"], cb["sigma2"]),
                         "a new P_cb table left sigma^2 unchanged (stale cache)")
        self.assertTrue(np.array_equal(back["dv"], cb["dv"]),
                        "input roundtrip did not return the data vector bit "
                        "for bit")

        # Changing thread count rebuilds work buffers, not the answer.
        for threads in (1, 8, 4):
            ci.set_omp_threads(threads)
            np.testing.assert_array_equal(evaluate(ci)["sigma2"], cb["sigma2"])

        # --- 6. no silent fallback ---
        environment = dict(os.environ)
        # the child runs its body directly (not through own_process)
        environment.pop(u.OWN_PROCESS_FLAG, None)
        completed = subprocess.run(
            [sys.executable, os.path.abspath(__file__), ABORT_CHILD_FLAG],
            env=environment, capture_output=True, text=True)
        output = completed.stdout + completed.stderr
        print(f"  6. cb field without a P_cb table: exit code "
              f"{completed.returncode}", flush=True)
        self.assertNotEqual(completed.returncode, 0,
                            "cb sigma^2 returned "
                            "without a P_cb table (silent fallback)")
        self.assertIn(ABORT_MESSAGE, output,
                      "the abort did not name the missing P_cb table")


if __name__ == "__main__":
    # the child of check 6 (see abort_child); otherwise unittest runs
    # the test named on the command line (own_process) or all of them
    if len(sys.argv) > 1 and sys.argv[1] == ABORT_CHILD_FLAG:
        abort_child()
    else:
        unittest.main()
