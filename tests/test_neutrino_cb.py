"""Unit tests: the cold dark matter + baryon halo field (halo_matter_field).

Massive neutrinos free-stream out of halos, so halos form from the cold
dark matter + baryon ("cb") field, not from the total matter. With the
likelihood key halo_matter_field = 1 (ci.init_halo_matter_field(1))
cosmolike's halo model takes

  sigma^2(M)   from the linear P_cb (CAMB's delta_nonu), with the
               Lagrangian radius R = (3 M/(4 pi rho_cb))^(1/3),
  dn/dlnM      = (rho_cb/M) nu f(nu) dln nu/dln M,

with rho_cb = rho_crit (Omega_m - Omega_nu), Omega_nu = omega_nu h^2/h^2
(set_cosmology's omegan2) and P_cb the table set_cosmology receives as
lnP_linear_cb. Everything else (r_Delta, the matter window M/rho_m, the
lensing kernels, the 2-halo spectrum) stays total matter. The default
halo_matter_field = 0 is the total-matter model, bit for bit the code
before the switch existed.

The checks, on the cluster combination 4x2pt + N:

  1. switch off: the data vector with Omega_nu h^2 and a P_cb table
     handed over equals, bit for bit, the one without them (nothing
     reads either under halo_matter_field = 0);
  2. the mnu -> 0 limit: halo_matter_field = 1 with Omega_nu h^2 = 0 and
     a P_cb table equal to P_lin reproduces halo_matter_field = 0 bit for
     bit (the cb path is the total-matter path with two inputs swapped);
  3. sigma^2(M) under halo_matter_field = 1 against the independent
     Python reference (tests/reference/ref_halo.py, HaloModel with
     hmf_matter = "cb") at five masses and two neutrino densities, the
     DES Y6 fiducial Omega_nu h^2 = 0.00083 and the top of its prior,
     0.00644; halo_matter_field = 0 against hmf_matter = "tot" as the
     control;
  4. direction and size at the fiducial: the cb counts are higher and
     the cb cluster bias lower than the total-matter ones, inside the
     bands the closed-form estimate gives, while cosmic shear,
     galaxy-galaxy lensing and galaxy clustering do not move at all;
  5. the cache keys: a new Omega_nu h^2 alone, or a new P_cb table
     alone, refills sigma^2(M); flipping the switch 1 -> 0 -> 1
     returns the first values bit for bit;
  6. no silent fallback: halo_matter_field = 1 without a P_cb table
     stops the process with a message naming the missing table (checked
     in a child process, since the C code calls exit).

The cosmology is the Python reference's CAMB run (tests/reference/
ref_cosmology.py: three degenerate massive neutrinos, P_lin, P_NL and
P_cb from delta_tot and delta_nonu) on the likelihood's (z, k) grids,
handed to cosmolike through ci.set_cosmology exactly as the likelihood
does (tests/validation/compare_reference.py: load_config, init_c,
cosmology_inputs, set_cosmology_c). No cobaya model is built.

Every check runs in a python process of its own
(cocoa_test_utils.own_process): under pytest the collecting process
also holds the cluster ladder's models, and cosmolike aborts a process
that initializes two data sets of different dimensions.

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/des_cluster/tests/test_neutrino_cb.py
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
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
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

# Tolerance of check 3. The C table integrates the lobe sums to 6e-6 of
# sigma^2 and reads it linearly in ln M between nodes 0.025 apart; the
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
ABORT_MESSAGE = "needs the linear P_cb table"

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
      the compare_reference configuration (halo_matter_field 0 on both
      sides), {omnuh2: the set_cosmology arrays of compare_reference.
      cosmology_inputs on the likelihood grids}, {omnuh2: the reference
      Cosmology}, and the scratch directory of the dummy data files.
    """
    import numpy as np
    import compare_reference as cr
    from reference_cluster import FIDUCIAL, COSMO_KEYS
    from ref_cosmology import Cosmology

    cfg = cr.load_config(halo_matter_field=0)
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
        sigma2.append(ci.sigma2(M=mass))
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
    """Check 6, child side: halo_matter_field = 1 with no P_cb table.

    Runs the init chain and one CAMB cosmology, hands the cosmology over
    WITHOUT a P_cb table, switches to the cb field and reads sigma^2.
    The C code must stop the process (exit code 1) naming the table; a
    return from ci.sigma2 is a silent fallback and exits with 0.
    """
    ci, cfg, tables, cosmologies, workdir = build_state((OMNUH2_FIDUCIAL,))
    R = tables[OMNUH2_FIDUCIAL]
    set_cosmology(ci, R, omegan2=float(R["cin_omnuh2"]), lnPL_cb=[])
    ci.init_halo_matter_field(halo_matter_field=1)
    value = ci.sigma2(M=1.0e14)
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

        # --- 1. switch off: the neutrino inputs are not read ---
        ci.init_halo_matter_field(halo_matter_field=0)
        set_cosmology(ci, R, omegan2=0.0, lnPL_cb=[])
        plain = evaluate(ci)
        set_cosmology(ci, R, omegan2=omegan2, lnPL_cb=lnPL_cb)
        with_inputs = evaluate(ci)
        print("\n  1. switch off: data vector with / without Omega_nu h^2 "
              "and P_cb bitwise equal: "
              f"{np.array_equal(plain['dv'], with_inputs['dv'])}", flush=True)
        self.assertTrue(np.array_equal(plain["dv"], with_inputs["dv"]),
                        "halo_matter_field = 0: the data vector moved when "
                        "Omega_nu h^2 and a P_cb table were handed over")

        # --- 2. the mnu -> 0 limit ---
        ci.init_halo_matter_field(halo_matter_field=1)
        set_cosmology(ci, R, omegan2=0.0, lnPL_cb=R["cin_lnPL"].copy())
        limit = evaluate(ci)
        same_dv = np.array_equal(limit["dv"], plain["dv"])
        same_sigma2 = np.array_equal(limit["sigma2"], plain["sigma2"])
        print(f"  2. mnu -> 0 limit: data vector bitwise {same_dv}, "
              f"sigma^2 bitwise {same_sigma2}", flush=True)
        self.assertTrue(same_sigma2,
                        "cb field with Omega_nu = 0 and P_cb = P_lin: sigma^2 "
                        "differs from the total-matter field")
        self.assertTrue(same_dv,
                        "cb field with Omega_nu = 0 and P_cb = P_lin: the data "
                        "vector differs from the total-matter field")

        # --- 3. sigma^2(M) against the Python reference ---
        print("  3. sigma^2(M): C / reference - 1", flush=True)
        worst = 0.0
        for omnuh2 in (OMNUH2_FIDUCIAL, OMNUH2_TOP):
            Rn = tables[omnuh2]
            for field, hmf_matter in ((1, "cb"), (0, "tot")):
                ci.init_halo_matter_field(halo_matter_field=field)
                set_cosmology(ci, Rn, omegan2=float(Rn["cin_omnuh2"]),
                              lnPL_cb=Rn["cin_lnPL_cb"])
                reference = HaloModel(cosmologies[omnuh2], hmf_matter=hmf_matter)
                ratios = []
                for mass in SIGMA2_MASSES:
                    sigma_ref = reference.sig.sigma(np.log(mass))
                    ratios.append(ci.sigma2(M=mass)/sigma_ref**2 - 1.0)
                ratios = np.array(ratios)
                worst = max(worst, float(np.max(np.abs(ratios))))
                print(f"     Omega_nu h^2 = {omnuh2}, {hmf_matter:3s}: "
                      + " ".join(f"{r:+.2e}" for r in ratios), flush=True)
                self.assertLess(
                    float(np.max(np.abs(ratios))), SIGMA2_RTOL,
                    f"sigma^2(M) at Omega_nu h^2 = {omnuh2}, field "
                    f"{hmf_matter}: C and the Python reference differ by "
                    f"more than {SIGMA2_RTOL}")

        # --- 4. direction and size at the fiducial ---
        ci.init_halo_matter_field(halo_matter_field=0)
        set_cosmology(ci, R, omegan2=omegan2, lnPL_cb=lnPL_cb)
        total = evaluate(ci)
        ci.init_halo_matter_field(halo_matter_field=1)
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
        # the switch 1 -> 0 -> 1 at the fiducial inputs
        set_cosmology(ci, R, omegan2=omegan2, lnPL_cb=lnPL_cb)
        ci.init_halo_matter_field(halo_matter_field=0)
        evaluate(ci)
        ci.init_halo_matter_field(halo_matter_field=1)
        back = evaluate(ci)
        print("  5. cache: Omega_nu h^2 alone moves sigma^2 "
              f"{not np.array_equal(moved_omnuh2['sigma2'], cb['sigma2'])}; "
              "P_cb alone moves sigma^2 "
              f"{not np.array_equal(moved_table['sigma2'], cb['sigma2'])}; "
              f"1 -> 0 -> 1 bitwise {np.array_equal(back['dv'], cb['dv'])}",
              flush=True)
        self.assertFalse(np.array_equal(moved_omnuh2["sigma2"], cb["sigma2"]),
                         "a new Omega_nu h^2 left sigma^2 unchanged (stale cache)")
        self.assertFalse(np.array_equal(moved_table["sigma2"], cb["sigma2"]),
                         "a new P_cb table left sigma^2 unchanged (stale cache)")
        self.assertTrue(np.array_equal(back["dv"], cb["dv"]),
                        "switch 1 -> 0 -> 1 did not return the data vector bit "
                        "for bit")

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
                            "sigma^2 returned under halo_matter_field = 1 "
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
