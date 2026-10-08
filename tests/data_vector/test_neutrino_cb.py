"""Neutrino-aware halos: cb variance at each redshift, counts and bias.

Massive neutrinos free-stream out of halos. The production halo model
therefore uses the cold dark matter plus baryon ("cb") linear spectrum
P_cb(k,z) (CAMB's delta_nonu), with rho_cb = rho_crit (Omega_m -
Omega_nu) in the smoothing radius R(M) = (3 M/(4 pi rho_cb))^(1/3) and
in the mass function. Integrating the spectrum at each z retains its
scale-dependent growth; multiplying a present-day variance by one D(z)
would lose it. The compiled interface (the python module
cosmolike_des_cluster_interface, built from interface/interface.cpp)
reports the variance sigma^2(M, a) of two fields: field 1, cb, the one
every halo statistic uses, and field 0, total matter (P_lin and rho_m),
a diagnostic. Its set_cosmology call hands cosmolike Omega_nu h^2
(omegan2) and the table ln P_cb (lnP_linear_cb) next to the usual
cosmology tables. There is no halo-field switch: the total-matter
comparison of check 4 supplies P_cb = P_lin and Omega_nu = 0 to the
halo inputs while the background stays fixed.

The checks, all on 4x2pt + N (cluster counts, cluster lensing, cluster
clustering, cluster x galaxy, galaxy clustering) at the fiducial point
of arXiv 2503.13631 (reference_cluster.FIDUCIAL), Omega_nu h^2 =
0.00083 unless stated:

  1. Handing over the cb inputs (Omega_nu h^2 and the P_cb table)
     leaves the total-matter variance exactly unchanged: field 0 reads
     neither.
  2. Massless limit: with Omega_nu h^2 = 0 and P_cb = P_lin, both
     fields agree exactly at a = 1, 0.8, 0.55 and 0.3, so the two code
     paths differ only through their inputs, evolution included.
  3. Both fields agree with the independent Python reference
     (tests/reference/ref_halo.py, Simpson integrals of the same CAMB
     run) to 1e-4 at five masses, the four scale factors, and
     Omega_nu h^2 = 0.00083 and 0.00644.
  4. Against the total-matter surrogate, the cb counts increase and
     the cluster bias decreases, inside bands around a closed-form
     estimate, and the galaxy blocks stay exactly unchanged.
  5. Changing Omega_nu h^2 alone, or the P_cb table alone, changes the
     cb variance (its table refills); restoring both inputs restores
     the full data vector exactly. The variance recomputed at 1, 8 and
     again 4 OpenMP threads must stay exactly the same.
  6. Asking for the cb variance without a P_cb table stops the process
     with a message naming the missing input (no silent fallback),
     checked in a child process.

The test feeds cosmolike without cobaya (the sampler framework that
runs the likelihoods): tests/validation/compare_reference.py supplies
the likelihood's init chain (init_c) and the cosmology tables, computed
from the Python reference's own CAMB runs (three degenerate massive
neutrino states) and handed over through the set_cosmology call the
likelihood makes. Those helpers read the live project files
(likelihood/combo_4x2pt_N.yaml, data/des_cluster_y6_4x2ptN.dataset and
the n(z) files it names), not the frozen copies under tests/frozen/
that the other tests read, so this test does not verify the frozen
manifest.

The test body runs in a python process of its own: the decorator
own_process of cocoa_test_utils.py restarts this file in a fresh
python that runs only this test, and the test passes when that process
exits with code 0. The test sets up cosmolike's global C state by hand
(init chain, an all-ones mask, OpenMP thread counts 1, 8 and 4); the
separate process keeps that state out of pytest's process, where
test_cache_consistency.py builds its cobaya models. Check 6 needs a
second child process: the C guard deliberately terminates the process
that asks for the missing table.

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest projects/des_cluster/tests/data_vector/test_neutrino_cb.py
"""

import os

# OpenMP (the threading library of the compiled C code) reads the
# environment variable OMP_NUM_THREADS when the compiled libraries
# load, so this must run before ANY cobaya/cosmolike import in the
# process. os.environ is this process's environment, which the
# processes it starts inherit. 4 = cocoa_testing.REQUIRED_OMP_THREADS,
# the thread count of every test process.
os.environ["OMP_NUM_THREADS"] = "4"

import subprocess
import sys
import tempfile
import unittest

# The tests folder is not a package (a folder with an __init__.py whose
# modules python imports by dotted name); put it on the import path so
# the shared harness resolves no matter where pytest was launched from.
# sys.path is the list of folders python searches on import, and
# insert(0, ...) puts a folder first, so the last insertion is searched
# first. TESTS_DIR = tests/ (dirname applied twice to this file's
# absolute path); validation/ holds compare_reference.py (the C-side
# init chain and the cosmology tables) and reference/ the independent
# Python reference (reference_cluster, ref_cosmology, ref_halo), both
# imported inside the functions below.
TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TESTS_DIR)
sys.path.insert(0, os.path.join(TESTS_DIR, "validation"))
sys.path.insert(0, os.path.join(TESTS_DIR, "reference"))
import cocoa_test_utils as u

# The two neutrino densities Omega_nu h^2: the fiducial of Table I of
# arXiv 2503.13631, whose CAMB tables every check uses, and the top of
# that table's flat prior [0.0006, 0.00644], where check 3 repeats the
# variance comparison at the largest neutrino fraction the prior allows.
OMNUH2_FIDUCIAL = 0.00083
OMNUH2_TOP = 0.00644

# Halo masses (M_sun/h) at which checks 1-3 and 5 read sigma^2(M): the
# range the richness bins select, from the lowest richness bin (~1e14)
# to the heaviest halos, with 1e13 below.
SIGMA2_MASSES = (1.0e13, 3.0e13, 1.0e14, 3.0e14, 1.0e15)

# Relative tolerance of check 3, |C/reference - 1|: the same 1e-4 as the
# table target of tests/validation/compare_reference.py (TABLE_TOL). It
# covers the interpolation of the C table in ln M and scale factor; the
# reference integrates each CAMB redshift with a 20001-node Simpson rule
# in ln k and interpolates with a bicubic spline in (z, ln M). Both
# start from the same CAMB run (the C side receives it resampled on the
# likelihood's k and z grids).
SIGMA2_RTOL = 1.0e-4

# Bands of check 4 at the fiducial (three degenerate neutrino states),
# for the ratio cb / total-matter surrogate. On cluster scales
# P_cb = P_lin/(1 - f_nu)^2 (f_nu = Omega_nu/Omega_m = 0.0058 here), so
# at a given mass sigma rises and the peak height nu = delta_c/sigma
# (delta_c = 1.686, the linear collapse threshold) falls: the
# exponential tail of the mass function rises and the bias of a halo of
# that mass falls, while rho_cb = (1 - f_nu) rho_m in R(M) and in dn/dM
# offsets part of the rise. A closed-form estimate gives counts +1% to
# +4% and richness-bin bias -0.6% to -0.9%. The bands are wider on both
# sides: a ratio of exactly 1 (cb inputs that never reach the halo
# model), counts that fall, a bias that rises, counts above +8% and a
# bias below -3% all fail; a percent-level change of the model does not.
COUNTS_RATIO_BAND = (1.002, 1.08)
BIAS_RATIO_BAND = (0.97, 0.999)

# The galaxy blocks of the joint vector: no halo statistic enters them,
# so check 4 requires them unchanged. 4x2pt + N switches ss and gs off
# (they hold zeros in every evaluation here), so gg carries that check.
GALAXY_BLOCKS = ("ss", "gs", "gg")
# The seven blocks in the order of the joint vector (block_slices).
BLOCKS = ("ss", "gs", "gg", "cg", "N", "cc", "cs")

# Part of the message the C guard of check 6 prints before it ends the
# process (cosmo3D.c, sigma2_field_read, reached through the
# interface's sigma2 with field=1).
ABORT_MESSAGE = "cb variance needs P_cb"

# The command-line flag that makes this file run abort_child (the child
# process of check 6) instead of the tests.
ABORT_CHILD_FLAG = "--abort-child"


def build_state(omnuh2_values):
    """Initialize cosmolike for 4x2pt + N and run the reference CAMB.

    The init chain is the likelihood's (compare_reference.init_c: the
    4x2pt + N probe, the binning, the accuracy settings, the n(z)
    files, and the cluster model and bins of the live dataset), with a
    zero data vector and a unit diagonal covariance, which the
    interface needs but these checks never read. A second
    init_data_cluster call replaces the dataset's mask with an all-ones
    mask, so every entry of the blocks 4x2pt + N computes is evaluated
    (the model is not evaluated at masked entries); the ss and gs
    blocks, which 4x2pt + N switches off, and the last theta bin of
    every cs row stay zero. The nuisance parameters are set at the
    fiducial point of arXiv 2503.13631 (reference_cluster.FIDUCIAL).
    Each Omega_nu h^2 value gets one CAMB run of the Python reference
    (three degenerate massive neutrino states), turned into the
    set_cosmology arrays on the likelihood's grids
    (compare_reference.cosmology_inputs, "production" variant).

    Arguments:
      omnuh2_values = the Omega_nu h^2 values to run CAMB at (tuple of
                      floats)

    Returns:
      (ci, cfg, tables, cosmologies, workdir): the compiled interface;
      the compare_reference configuration, {"dataset": the keys of the
      dataset file, "likelihood": the keys of the likelihood yaml,
      "point": {}}; {omnuh2: the set_cosmology arrays (keys cin_*) of
      compare_reference.cosmology_inputs on the likelihood grids};
      {omnuh2: the reference Cosmology, which holds its CAMB run}; and
      the scratch directory of the dummy data files and the all-ones
      mask. The scratch directory (tempfile.mkdtemp) stays on disk, and
      init_c sets cosmolike to 4 OpenMP threads.
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
        # the cosmological parameters of the fiducial point
        # (reference_cluster.FIDUCIAL), with Omega_nu h^2 replaced
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
    """Call ci.set_cosmology with the tables R and the given neutrino inputs.

    Everything except the two neutrino inputs comes from R, so a check
    can vary Omega_nu h^2 and the P_cb table while the background, the
    growth, and the linear and nonlinear total-matter spectra stay
    fixed. The setters behind the call compare the inputs with the
    values cosmolike holds and draw a new cosmology cache key
    (cosmology.random) when one changed, so every table built from the
    replaced inputs refills when it is next read.

    Arguments:
      ci      = the compiled interface
      R       = the arrays of compare_reference.cosmology_inputs (keys
                cin_*: log10 k in h/Mpc, z, ln P in (Mpc/h)^3, the
                growth G on the z_G grid, chi in Mpc/h, Omega_m,
                Omega_b, H0)
      omegan2 = Omega_nu h^2 handed over (float, >= 0)
      lnPL_cb = the ln P_cb table handed over (flattened as lnP_linear:
                k index slow, z index fast), or [] to remove the table

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
    """Compute the quantities the checks compare, at the current inputs.

    Arguments:
      ci = the compiled interface, initialized by build_state and given
           a cosmology by set_cosmology.

    Returns:
      dict with
        "dv"     = the joint data vector (numpy array, 2812 entries):
                   every entry of the 4x2pt + N blocks under the
                   all-ones mask, zeros in the switched-off ss and gs
                   blocks and in the last theta bin of every cs row;
        "sigma2" = the cb variance sigma^2(M, a = 1) at SIGMA2_MASSES
                   (5 values, dimensionless);
        "N"      = the expected cluster counts over the survey area,
                   array [cluster z bin, richness bin] = [3, 4];
        "bias"   = the richness-weighted linear bias b_A at the middle
                   of each cluster z bin, array [richness bin, z bin] =
                   [4, 3].
    """
    import numpy as np

    dv = np.array(ci.compute_data_vector_cluster_masked())
    # field=1 = cold dark matter + baryons; a keeps the interface
    # default, 1
    sigma2 = []
    for mass in SIGMA2_MASSES:
        sigma2.append(ci.sigma2(M=mass, field=1))
    # N_cluster_tomo returns [richness bin, cluster z bin]; .T
    # transposes it to [cluster z bin, richness bin]
    counts = np.array(ci.N_cluster_tomo()).T
    # z = 0.3, 0.475 and 0.6 are the middles of the cluster z bins
    # [0.2, 0.4], [0.4, 0.55] and [0.55, 0.65]; bcl_richness takes the
    # scale factor a = 1/(1 + z) and a richness bin
    bias = []
    for richness_bin in range(counts.shape[1]):
        row = []
        for z in (0.3, 0.475, 0.6):
            row.append(ci.bcl_richness(1.0/(1.0 + z), richness_bin))
        bias.append(row)
    # dict(name=value, ...) builds the dictionary from keyword arguments
    return dict(dv=dv, sigma2=np.array(sigma2), N=counts, bias=np.array(bias))


def block_slices(ci):
    """Map each block name to its index range in the joint vector.

    Arguments:
      ci = the compiled interface after its init chain (the block sizes
           follow from the dataset and the probe).

    Returns:
      {block name: slice(start, start + size)} for the seven BLOCKS; a
      slice object selects that index range, so dv[slice] is the block.
    """
    # one size and one start index per block, in BLOCKS order
    sizes = ci.compute_data_vector_cluster_sizes()
    starts = ci.compute_data_vector_cluster_starts()
    out = {}
    # enumerate yields (position, name) pairs
    for n, name in enumerate(BLOCKS):
        out[name] = slice(int(starts[n]), int(starts[n] + sizes[n]))
    return out


def abort_child():
    """Request the cb variance with no P_cb table (child of check 6).

    Builds the state at the fiducial (one CAMB run), hands over the
    fiducial Omega_nu h^2 with an empty P_cb list, which removes any
    P_cb table, and asks for sigma^2 of field 1 (cb) at 1e14 M_sun/h.
    The C guard (cosmo3D.c, sigma2_field_read) prints ABORT_MESSAGE and
    exits with code 1, so the lines after the call run only when the
    guard is missing: they print the value and exit with code 0, which
    the parent reports as a silent fallback.

    Arguments:
      none.

    Returns:
      never: the process ends inside the C guard (exit code 1) or at
      sys.exit(0).
    """
    # (OMNUH2_FIDUCIAL,) is a one-element tuple (the trailing comma makes
    # it one), the type build_state loops over
    ci, cfg, tables, cosmologies, workdir = build_state((OMNUH2_FIDUCIAL,))
    R = tables[OMNUH2_FIDUCIAL]
    set_cosmology(ci, R, omegan2=float(R["cin_omnuh2"]), lnPL_cb=[])
    value = ci.sigma2(M=1.0e14, field=1)
    print(f"sigma2 returned {value} without a P_cb table", flush=True)
    sys.exit(0)


class TestNeutrinoCB(unittest.TestCase):
    """The cold dark matter + baryon halo field on 4x2pt + N.

    unittest.TestCase is the base class of python's unittest framework:
    each method whose name starts with test_ is one test, and pytest
    collects such classes too. The one test method runs checks 1-6 of
    the module docstring in order on one cosmolike state (build_state).
    """

    # @classmethod is a decorator (a line starting with @ that wraps
    # the function defined below it): it hands the method the class
    # itself (cls), not an instance; unittest calls setUpClass once
    # before the first test of the class
    @classmethod
    def setUpClass(cls):
        """Check once that start_cocoa.sh was sourced.

        require_cocoa_environment refuses to run without ROOTDIR (the
        Cocoa/ folder, which start_cocoa.sh exports) and moves to it.
        The frozen manifest is not verified: this test reads the live
        project files, not tests/frozen/.

        Arguments:
          cls = this test class (the classmethod decorator passes the
                class itself, not an instance).

        Returns:
          nothing.

        Raises:
          RuntimeError when ROOTDIR is not set.
        """
        u.require_cocoa_environment()

    # The decorator u.own_process(__file__) wraps the method: in
    # pytest's process the wrapper starts a fresh python on this file
    # that runs only this test, and passes when that process exits with
    # code 0; inside that child (marked by the environment variable that
    # cocoa_test_utils.OWN_PROCESS_FLAG names) the wrapper runs the body
    @u.own_process(__file__)
    def test_neutrino_cb(self):
        """Run checks 1-6 of the module docstring on one cosmolike state.

        build_state runs CAMB at the fiducial Omega_nu h^2 and at the top
        of its prior and initializes cosmolike once; each check then
        hands set_cosmology the inputs it needs and reads sigma^2, the
        counts, the bias or the data vector. Checks 3-6 print their
        numbers before they assert.

        Arguments:
          none.

        Returns:
          nothing. The process sets OpenMP to 1, 8 and finally 4
          threads, starts one child process (check 6), and leaves the
          scratch directory of build_state on disk.

        Raises:
          AssertionError at the first failed check (np.testing raises
          it too); in pytest's process the wrapper raises an
          AssertionError naming the test when the child exits with a
          nonzero code (the child's report, printed above it, shows the
          failed check).
        """
        import numpy as np
        from ref_halo import HaloModel

        ci, cfg, tables, cosmologies, workdir = build_state(
            (OMNUH2_FIDUCIAL, OMNUH2_TOP))
        slices = block_slices(ci)
        # the fiducial tables and the two neutrino inputs the likelihood
        # would hand over there: Omega_nu h^2 and ln P_cb
        R = tables[OMNUH2_FIDUCIAL]
        omegan2 = float(R["cin_omnuh2"])
        lnPL_cb = R["cin_lnPL_cb"]

        # --- 1. total variance is independent of the cb inputs ---
        # sigma^2 of field 0 (total matter) at a = 1, one value per mass
        # (the list comprehension calls sigma2 for each mass), first
        # without the cb inputs, then with the fiducial ones;
        # assert_array_equal raises AssertionError unless every element
        # is exactly equal
        set_cosmology(ci, R, omegan2=0.0, lnPL_cb=[])
        plain = np.array([ci.sigma2(M=mass, field=0) for mass in SIGMA2_MASSES])
        set_cosmology(ci, R, omegan2=omegan2, lnPL_cb=lnPL_cb)
        with_inputs = np.array([ci.sigma2(M=mass, field=0) for mass in SIGMA2_MASSES])
        np.testing.assert_array_equal(plain, with_inputs)

        # --- 2. the massless limit, including evolution ---
        # Omega_nu h^2 = 0 and a cb table equal to P_lin (.copy() hands
        # over an independent array with the same values): the two
        # fields must agree exactly (assertEqual on two floats demands
        # equality) at every mass and scale factor
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
            # each pair is (interface field, the reference's matter
            # choice): 1 = cb (P_cb and rho_cb), 0 = total matter (P_lin
            # and rho_m); the C side always gets the physical inputs
            for field, hmf_matter in ((1, "cb"), (0, "tot")):
                set_cosmology(ci, Rn, omegan2=float(Rn["cin_omnuh2"]),
                              lnPL_cb=Rn["cin_lnPL_cb"])
                # HaloModel builds the reference variance table of that
                # field from the same Cosmology (the same CAMB run)
                reference = HaloModel(cosmologies[omnuh2], hmf_matter=hmf_matter)
                for a in (1.0, 0.8, 0.55, 0.3):
                    # reference.sig.sigma returns sigma (not squared) at
                    # ln M and the redshift z = 1/a - 1
                    ratios = []
                    for mass in SIGMA2_MASSES:
                        sigma_ref = reference.sig.sigma(np.log(mass), z=1.0/a-1.0)
                        ratios.append(ci.sigma2(M=mass, a=a, field=field)/sigma_ref**2-1.0)
                    ratios = np.array(ratios)
                    worst = max(worst, float(np.max(np.abs(ratios))))
                    # the generator inside join formats each ratio with
                    # its sign in scientific notation, two decimals
                    # (+1.23e-05), and join glues them with spaces
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
        # element-by-element ratios, [3, 4] for the counts and [4, 3]
        # for the bias; their smallest and largest values must sit
        # inside the bands
        counts_ratio = cb["N"]/total["N"]
        bias_ratio = cb["bias"]/total["bias"]
        print(f"  4. cb / total at the fiducial: counts {counts_ratio.min():.4f}"
              f" .. {counts_ratio.max():.4f}, richness-bin bias "
              f"{bias_ratio.min():.4f} .. {bias_ratio.max():.4f}", flush=True)
        self.assertGreater(float(counts_ratio.min()), COUNTS_RATIO_BAND[0])
        self.assertLess(float(counts_ratio.max()), COUNTS_RATIO_BAND[1])
        self.assertGreater(float(bias_ratio.min()), BIAS_RATIO_BAND[0])
        self.assertLess(float(bias_ratio.max()), BIAS_RATIO_BAND[1])
        # ss and gs hold zeros in both vectors (4x2pt + N switches them
        # off), so gg carries this comparison
        for name in GALAXY_BLOCKS:
            same = np.array_equal(cb["dv"][slices[name]], total["dv"][slices[name]])
            self.assertTrue(same, f"block {name} moved with the halo field")

        # --- 5. the cache keys ---
        # a changed Omega_nu h^2 alone: rho_cb and R(M) move, so sigma^2
        # must
        set_cosmology(ci, R, omegan2=1.5*omegan2, lnPL_cb=lnPL_cb)
        moved_omnuh2 = evaluate(ci)
        # a changed P_cb table alone: numpy adds 0.01 to every element
        # of ln P_cb, which multiplies P_cb by exp(0.01) at every k and z
        set_cosmology(ci, R, omegan2=omegan2, lnPL_cb=lnPL_cb + 0.01)
        moved_table = evaluate(ci)
        # Return to the fiducial after both independent input changes.
        set_cosmology(ci, R, omegan2=omegan2, lnPL_cb=lnPL_cb)
        back = evaluate(ci)
        # the f-string fields hold expressions: not np.array_equal(...)
        # prints True when the two arrays differ
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
        # set_omp_threads sets cosmolike's OpenMP thread count. The
        # variance table records the count it was built with
        # (cosmo3D.c, sigma2_fields_build) and is rebuilt, with its
        # per-thread work buffers, when the count changes; the rebuilt
        # values must equal the 4-thread ones exactly. The final 4
        # restores the count build_state set
        for threads in (1, 8, 4):
            ci.set_omp_threads(threads)
            np.testing.assert_array_equal(evaluate(ci)["sigma2"], cb["sigma2"])

        # --- 6. no silent fallback ---
        # dict(os.environ) is a copy of this process's environment, so
        # the edit below reaches only the child
        environment = dict(os.environ)
        # the child runs its body directly (not through own_process):
        # pop removes the own_process flag from the copy, and its second
        # argument, None, makes the removal silent when the flag is absent
        environment.pop(u.OWN_PROCESS_FLAG, None)
        # subprocess.run starts the child (abort_child, selected by
        # ABORT_CHILD_FLAG) and blocks until it exits; capture_output
        # and text hand back the child's printed output as python
        # strings instead of streaming it, so the message can be
        # searched
        completed = subprocess.run(
            [sys.executable, os.path.abspath(__file__), ABORT_CHILD_FLAG],
            env=environment, capture_output=True, text=True)
        output = completed.stdout + completed.stderr
        print(f"  6. cb field without a P_cb table: exit code "
              f"{completed.returncode}", flush=True)
        # the child must fail (a nonzero exit code), and its output must
        # contain the guard's message (assertIn tests for a substring)
        self.assertNotEqual(completed.returncode, 0,
                            "cb sigma^2 returned "
                            "without a P_cb table (silent fallback)")
        self.assertIn(ABORT_MESSAGE, output,
                      "the abort did not name the missing P_cb table")


# __name__ is "__main__" only when this file runs directly as a script
# (the processes own_process and check 6 start, or a manual run); pytest
# imports the module instead, so this block stays idle under pytest
if __name__ == "__main__":
    # the child of check 6 (see abort_child); otherwise unittest runs
    # the test named on the command line (own_process) or all of them
    if len(sys.argv) > 1 and sys.argv[1] == ABORT_CHILD_FLAG:
        abort_child()
    else:
        unittest.main()
