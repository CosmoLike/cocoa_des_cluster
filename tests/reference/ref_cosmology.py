"""Background, growth, linear/nonlinear P(k) and tomographic n(z) for the
DES cluster Python reference.

This module is the bottom layer of the independent Python reference
model of the des_cluster project (reference_cluster.py ties the modules
together): ref_halo, ref_cluster, ref_limber, ref_nonlimber and
ref_covariance read every distance, growth factor and power spectrum
from the Cosmology object built here, and the lens and source redshift
distributions n(z) from NzBins. It is not run on its own:
reference_cluster.py, the tests of this folder,
tests/validation/compare_reference.py and scripts/make_synthetic_data.py
import it.

Everything is in "h units":

  k    in h/Mpc              chi, D_M  in Mpc/h
  P(k) in (Mpc/h)^3          M         in M_sun/h
  rho  in (M_sun/h)/(Mpc/h)^3

The power spectra come from python CAMB (the Boltzmann code): the linear
spectra, and Halofit (a fitting formula for the nonlinear correction)
for the nonlinear one. CAMB is called directly: no cobaya, no cosmolike
tables.

Conventions mirrored from cocoa/cosmolike (so that the C code can be fed
the same inputs and compared number by number):

  * Omega_m is the total matter density, Omega_c + Omega_b + Omega_nu, as
    the cocoa likelihoods pass it (omegam). omch2 = Omega_m h^2 - ombh2 -
    omnuh2.
  * rho_crit = 7.4775e21 (M_sun/h)/(c/H0)^3 (cosmolike structs.c), i.e.
    2.77520e11 (M_sun/h)/(Mpc/h)^3 (CODATA gives 2.77537e11; 6e-5 apart).
  * The non-halo growth factor follows the external DES reference
    convention: D(z) = sqrt(P_lin(k0, z)/P_lin(k0, 0))
    with k0 = 0.0005 1/Mpc on the total-matter linear spectrum. It is
    scale independent by construction and used for non-halo terms (the
    NLA intrinsic-alignment amplitude of ref_limber, the linear-theory
    kernels of ref_nonlimber and of the counts covariance). Halo
    statistics integrate P_cb(k,z) at each redshift directly. The
    likelihood samples its own growth table at growth_k = 0.05/Mpc
    (likelihood/_cosmolike_prototype_base.py);
    tests/validation/compare_reference.py hands the C code the D(z) of
    this module, so both sides of that comparison use one growth factor.
  * n(z) files are read as in redshift_spline.c: the z column holds the
    left edges of the histogram cells (Z_LOW convention), the values sit
    at the cell centres, each bin is normalized by the rectangle sum
    sum_k n_k dz (not by the integral of the spline), and n(z) is a
    natural cubic spline through the centres, zero outside the open
    interval (first centre, last centre). Sources: n(z - dz_s). Lenses:
    n((z - dz_l - zbar)/s + zbar)/s with zbar the mean of the unshifted
    bin (dz_s, dz_l = photo-z shifts, s = lens stretch).
"""

import numpy as np
import camb
from scipy.interpolate import CubicSpline, RectBivariateSpline

# Constants shared by the reference modules (the trailing comments give
# each unit). RHO_CRIT converts cosmolike's 7.4775e21 (M_sun/h)/(c/H0)^3
# to (Mpc/h)^3 with COVERH0; K0_GROWTH_INV_MPC is the k0 of growth()
# (module docstring).
C_KMS = 299792.458                  # speed of light in km/s
COVERH0 = 2997.92458                # c/H0 in Mpc/h
RHO_CRIT = 7.4775e21 / COVERH0**3   # cosmolike rho_crit in (M_sun/h)/(Mpc/h)^3
K0_GROWTH_INV_MPC = 0.0005          # k (1/Mpc) of the external DES convention


def default_pk_redshifts(z_max=4.0):
    """Return the z nodes of the P(k, z) tables, dense where the clusters live.

    The cluster bins span z_lambda = 0.2-0.65 (photo-z tails reach about
    0.8), so the nodes are 4 times denser below z = 1 than on (1, 2] and
    8 times denser than above z = 2, where only the tails of the lens and
    source n(z) and the lensing kernels reach.

    Arguments:
      z_max = the last node, the top of every P table (default 4.0; the
              lens and source n(z) files of the project end at z = 3).

    Returns:
      numpy array of 121 increasing redshifts: 81 nodes on [0, 1]
      (dz = 0.0125), 20 on (1, 2] (dz = 0.05) and 20 on (2, z_max]
      (dz = (z_max - 2)/20, 0.1 for the default).
    """
    z1 = np.linspace(0.0, 1.0, 81)                  # dz = 0.0125
    # [1:] drops the first node of a segment: it is the last node of the
    # segment before
    z2 = np.linspace(1.0, 2.0, 21)[1:]              # dz = 0.05
    z3 = np.linspace(2.0, z_max, 21)[1:]            # dz = 0.1
    return np.concatenate([z1, z2, z3])


class Cosmology:
    """CAMB background and power spectra, interpolated in (z, ln k).

    One CAMB run happens at construction; every method afterwards reads
    tables. The object holds three ln P(k, z) tables on CAMB's k grid
    (k_pk, in h/Mpc) and the z nodes z_pk (default_pk_redshifts):

      "lin" = linear total-matter P (CAMB variable delta_tot),
      "nl"  = Halofit nonlinear total-matter P (delta_tot),
      "cb"  = linear P of cold dark matter plus baryons (delta_nonu),
              the spectrum of the halo statistics (ref_halo),

    each read through a 2-D spline in (z, ln k) (lnP), and the
    background: chi(z), E(z) = H(z)/H0, dchi/dz, z(chi) and the growth
    factor D(z). The constructor arguments are documented in __init__.
    """

    def __init__(self, params, kmax=100.0, halofit="takahashi",
                 num_massive_nu=3, z_nodes=None, accuracy_boost=1.0,
                 k_per_logint=None, nl_z_order=3):
        """Run CAMB once and tabulate ln P(k, z) and the background.

        Omega_c h^2 is derived as Omega_m h^2 - Omega_b h^2 - Omega_nu h^2,
        so Omega_m is the total matter density, as the cocoa likelihoods
        pass it. The neutrino mass is solved for: a trial CAMB set-up at
        mnu = 0.06 eV gives the ratio Omega_nu h^2/mnu, and the second
        set-up uses the mnu that reproduces the requested Omega_nu h^2
        (checked to 1e-10).

        Arguments:
          params = dict of the cosmology: "Omega_m" (total matter),
                   "Omega_b", "h" = H0/(100 km/s/Mpc), "A_s", "n_s";
                   optional "Omega_nu_h2" (default 0: every neutrino
                   massless), "w0" (default -1), "wa" (default 0) and
                   "tau" (default 0.0697186, the optical depth that
                   EXAMPLE_EVALUATE1.yaml and EXAMPLE_EVALUATE2.yaml fix).
          kmax = largest k in h/Mpc that CAMB computes; beyond the last
                 CAMB k node ln P continues linearly in ln k (lnP).
          halofit = CAMB's halofit_version of the nonlinear spectrum
                    ("takahashi": Takahashi et al. 2012).
          num_massive_nu = number of degenerate massive neutrino species
                           (default 3; scripts/make_synthetic_data.py
                           passes the number of the likelihood's CAMB
                           set-up, 1).
          z_nodes = z nodes of the P tables (None: default_pk_redshifts());
                    sorted ascending here.
          accuracy_boost = CAMB's AccuracyBoost (1 = CAMB's default).
          k_per_logint = CAMB's number of k points per logarithmic
                         interval of the matter power (None: CAMB's
                         default spacing).
          nl_z_order = spline order in z of the ln P_NL table (3 = cubic,
                       1 = linear between the CAMB z nodes). CAMB's
                       Halofit finds its nonlinear scale by bisection to
                       |sigma - 1| <= 1e-3 (halofit.f90), so ln P_NL
                       carries ~1e-3 node-to-node noise in z at k > k_NL;
                       a cubic spline through it and a linear read of the
                       same nodes (cosmolike's p_nonlin) then differ by
                       that much between nodes. A C comparison uses 1 and
                       hands the C side these nodes.

        Raises:
          AssertionError when the solved mnu misses Omega_nu h^2 by more
          than 1e-10 (relative), or when CAMB returns P tables on z nodes
          other than z_nodes.
        """
        p = dict(params)
        self.params = p
        self.h = h = float(p["h"])
        self.Omega_m = float(p["Omega_m"])
        self.Omega_b = float(p["Omega_b"])
        self.omnuh2 = float(p.get("Omega_nu_h2", 0.0))
        self.Omega_nu = self.omnuh2 / h**2
        self.Omega_cb = self.Omega_m - self.Omega_nu
        self.A_s = float(p["A_s"])
        self.n_s = float(p["n_s"])
        self.w0 = float(p.get("w0", -1.0))
        self.wa = float(p.get("wa", 0.0))
        self.halofit = halofit
        self.kmax = float(kmax)
        z_nodes = default_pk_redshifts() if z_nodes is None else np.asarray(z_nodes)
        self.z_pk = np.sort(z_nodes)

        # physical densities Omega h^2: cold dark matter gets what the
        # total matter leaves after baryons and massive neutrinos
        ombh2 = self.Omega_b * h**2
        omch2 = self.Omega_m * h**2 - ombh2 - self.omnuh2

        pars = camb.CAMBparams()
        # Trial neutrino mass in eV. CAMB's omnuh2 is proportional to mnu
        # at a fixed number of degenerate species, so the omnuh2 of this
        # trial set-up converts the requested Omega_nu h^2 into mnu below.
        mnu = 0.06
        pars.set_cosmology(H0=100.0 * h, ombh2=ombh2, omch2=omch2, omk=0.0,
                           mnu=mnu, num_massive_neutrinos=num_massive_nu,
                           neutrino_hierarchy="degenerate",
                           tau=float(p.get("tau", 0.0697186)))
        if self.omnuh2 > 0:
            # omnuh2 is linear in mnu: rescale to hit the target exactly
            mnu = mnu * self.omnuh2 / pars.omnuh2
            pars.set_cosmology(H0=100.0 * h, ombh2=ombh2, omch2=omch2, omk=0.0,
                               mnu=mnu, num_massive_neutrinos=num_massive_nu,
                               neutrino_hierarchy="degenerate",
                               tau=float(p.get("tau", 0.0697186)))
            assert abs(pars.omnuh2 / self.omnuh2 - 1) < 1e-10
        else:
            # Omega_nu h^2 = 0: every neutrino species massless
            pars.set_cosmology(H0=100.0 * h, ombh2=ombh2, omch2=omch2, omk=0.0,
                               mnu=0.0, num_massive_neutrinos=0,
                               tau=float(p.get("tau", 0.0697186)))
        self.mnu = mnu
        pars.InitPower.set_params(As=self.A_s, ns=self.n_s)
        # w0-wa dark energy through CAMB's PPF module (parametrized
        # post-Friedmann), which lets w cross -1; at w0 = -1, wa = 0 CAMB
        # keeps its cosmological constant
        if self.w0 != -1.0 or self.wa != 0.0:
            pars.set_dark_energy(w=self.w0, wa=self.wa, dark_energy_model="ppf")
        pars.set_accuracy(AccuracyBoost=accuracy_boost)
        # Matter power at every z node. z_pk[::-1] is the node array
        # reversed (decreasing z); table() below sorts CAMB's output rows
        # back to increasing z. CAMB's kmax is in 1/Mpc, hence kmax * h.
        pars.set_matter_power(redshifts=list(self.z_pk[::-1]), kmax=self.kmax * h,
                              nonlinear=True, k_per_logint=k_per_logint,
                              accurate_massive_neutrino_transfers=True,
                              silent=True)
        # Halofit (version `halofit`) for the nonlinear matter power;
        # WantCls = False skips the CMB (cosmic microwave background)
        # spectra
        pars.NonLinear = camb.model.NonLinear_both
        pars.NonLinearModel.set_params(halofit_version=halofit)
        pars.WantCls = False
        self.camb_pars = pars
        self.results = res = camb.get_results(pars)

        def table(var, nonlinear):
            """Return one CAMB matter-power table, rows sorted by increasing z.

            A function defined inside __init__: it reads the CAMB results
            object res of the enclosing function.

            Arguments:
              var = CAMB variable of both density fields: "delta_tot"
                    (total matter) or "delta_nonu" (cold dark matter plus
                    baryons).
              nonlinear = True for the Halofit spectrum, False for the
                          linear one.

            Returns:
              (k, z, P): k (nk,) in h/Mpc, z (nz,) increasing, P (nz, nk)
              in (Mpc/h)^3 (hubble_units and k_hunit select h units).
            """
            if nonlinear:
                k, z, pk = res.get_nonlinear_matter_power_spectrum(
                    var1=var, var2=var, hubble_units=True, k_hunit=True)
            else:
                k, z, pk = res.get_linear_matter_power_spectrum(
                    var1=var, var2=var, hubble_units=True, k_hunit=True)
            z = np.asarray(z)
            order = np.argsort(z)
            return np.asarray(k), z[order], np.asarray(pk)[order]

        # The three spectra of the class docstring, each (nz, nk); the `_`
        # names discard the k and z arrays the first call already returned.
        k, z, plin = table("delta_tot", False)
        _, _, pnl = table("delta_tot", True)
        _, _, pcb = table("delta_nonu", False)
        assert np.allclose(z, self.z_pk, atol=1e-8)
        self.k_pk = k
        self.lnk_pk = np.log(k)
        self._tabs = {"lin": np.log(plin), "nl": np.log(pnl), "cb": np.log(pcb)}
        # One 2-D spline of ln P over (z, ln k) per kind, built by a dict
        # comprehension ({key: value for ...}): cubic in ln k (ky = 3),
        # cubic in z (kx = 3) except "nl", which uses nl_z_order.
        self._spl = {key: RectBivariateSpline(self.z_pk, self.lnk_pk, tab,
                                              kx=(nl_z_order if key == "nl" else 3), ky=3)
                     for key, tab in self._tabs.items()}
        # End slopes d ln P/d ln k at every z node, from the first two and
        # from the last two k nodes: one array of length nz per kind. lnP
        # continues ln P along them outside the CAMB k range.
        self._slope_lo = {key: (tab[:, 1] - tab[:, 0]) / (self.lnk_pk[1] - self.lnk_pk[0])
                          for key, tab in self._tabs.items()}
        self._slope_hi = {key: (tab[:, -1] - tab[:, -2]) / (self.lnk_pk[-1] - self.lnk_pk[-2])
                          for key, tab in self._tabs.items()}

        # Normalization of growth(): ln P_lin(k0, z = 0), with k0 =
        # 0.0005/Mpc converted to h/Mpc (module docstring).
        k0 = K0_GROWTH_INV_MPC / h
        self._lnP0_k0 = float(self.lnP(np.array([k0]), np.array([0.0]), "lin")[0])
        self._k0 = k0

        # z(chi): a cubic spline through the points (chi(z), z) on 20001
        # nodes z in [0, 5] (dz = 2.5e-4); chi grows with z, so swapping
        # the two axes inverts chi(z).
        zz = np.linspace(0.0, 5.0, 20001)
        self._chi_tab = self.chi(zz)
        self._z_of_chi = CubicSpline(self._chi_tab, zz)

    # ------------------------------------------------------------------
    # background
    # ------------------------------------------------------------------
    def chi(self, z):
        """Return the comoving distance chi(z) in Mpc/h.

        The model is flat (omk = 0), so chi is also the comoving
        transverse distance D_M.

        Arguments:
          z = redshift, scalar or array.

        Returns:
          numpy array of the shape of z, in Mpc/h (CAMB's Mpc times h).
        """
        z = np.asarray(z, dtype=float)
        out = np.asarray(self.results.comoving_radial_distance(np.atleast_1d(z))) * self.h
        return out.reshape(z.shape)

    def E(self, z):
        """Return the dimensionless expansion rate E(z) = H(z)/H0.

        Arguments:
          z = redshift, scalar or array.

        Returns:
          numpy array of the shape of z (CAMB's H in km/s/Mpc divided by
          100 h).
        """
        z = np.asarray(z, dtype=float)
        out = np.asarray(self.results.hubble_parameter(np.atleast_1d(z))) / (100.0 * self.h)
        return out.reshape(z.shape)

    def dchi_dz(self, z):
        """Return dchi/dz = c/H(z) in Mpc/h.

        Arguments:
          z = redshift, scalar or array.

        Returns:
          numpy array of the shape of z, in Mpc/h per unit redshift.
        """
        return COVERH0 / self.E(z)

    def z_of_chi(self, chi):
        """Return the redshift at comoving distance chi (inverse of chi(z)).

        Arguments:
          chi = comoving distance in Mpc/h, scalar or array, within
                [0, chi(5)]; outside that range the spline continues its
                end cubic (scipy CubicSpline extrapolates, no error).

        Returns:
          numpy array of redshifts, the shape of chi.
        """
        return self._z_of_chi(chi)

    def growth(self, z):
        """Return the linear growth factor D(z), normalized to D(0) = 1.

        D(z) = sqrt(P_lin(k0, z)/P_lin(k0, 0)) with k0 = 0.0005/Mpc on the
        total-matter linear table (module docstring): one scale, so D is
        scale independent by construction.

        Arguments:
          z = redshift, scalar or array.

        Returns:
          numpy array of D, at least one-dimensional (a scalar z gives an
          array of length 1).
        """
        z = np.atleast_1d(np.asarray(z, dtype=float))
        lnp = self.lnP(np.full(z.shape, self._k0), z, "lin")
        return np.exp(0.5 * (lnp - self._lnP0_k0))

    # ------------------------------------------------------------------
    # power spectra
    # ------------------------------------------------------------------
    def lnP(self, k, z, kind="nl"):
        """Return ln P(k, z) of one tabulated spectrum, P in (Mpc/h)^3.

        Inside the CAMB k range the value is the 2-D spline of the table.
        Below the first or above the last k node ln P continues as a
        straight line in ln k whose slope is the end slope of the table,
        interpolated linearly in z. A z outside [z_pk[0], z_pk[-1]] is not
        extrapolated: the spline evaluation (scipy RectBivariateSpline.ev)
        clamps it to the nearest end node.

        Arguments:
          k = wavenumber in h/Mpc (> 0), scalar or array.
          z = redshift, scalar or array; numpy broadcasting combines it
              with k (arrays of compatible shapes are expanded to one
              common shape, e.g. k of shape (nk, 1) with z of shape
              (1, nz) gives (nk, nz)).
          kind = "nl" (Halofit total matter, default), "lin" (linear
                 total matter) or "cb" (linear cold dark matter plus
                 baryons).

        Returns:
          numpy array of the broadcast shape of (k, z).

        Raises:
          KeyError when kind is none of the three table names.
        """
        # one common shape for k and z, then flat 1-D copies (ravel); the
        # result gets the common shape back at the end
        k, z = np.broadcast_arrays(np.asarray(k, dtype=float), np.asarray(z, dtype=float))
        shape = k.shape
        lnk = np.log(k).ravel()
        zr = z.ravel()
        lo, hi = self.lnk_pk[0], self.lnk_pk[-1]
        # points outside the CAMB k range are read at the nearest end node
        # (np.clip) and then moved along the end slope below
        lnk_c = np.clip(lnk, lo, hi)
        out = self._spl[kind].ev(zr, lnk_c)
        # boolean masks of the points below and above the k range;
        # out[below] selects only those entries
        below = lnk < lo
        above = lnk > hi
        if below.any() or above.any():
            # end slopes interpolated linearly in z
            if below.any():
                s = np.interp(zr[below], self.z_pk, self._slope_lo[kind])
                out[below] += s * (lnk[below] - lo)
            if above.any():
                s = np.interp(zr[above], self.z_pk, self._slope_hi[kind])
                out[above] += s * (lnk[above] - hi)
        return out.reshape(shape)

    def P_lin(self, k, z, kind="lin"):
        """Return a linear power spectrum P(k, z) in (Mpc/h)^3.

        Arguments:
          k = wavenumber in h/Mpc, scalar or array.
          z = redshift, broadcast against k (lnP).
          kind = "lin" (total matter, default) or "cb" (cold dark matter
                 plus baryons, the spectrum of ref_halo's variance).

        Returns:
          numpy array of the broadcast shape of (k, z): exp(lnP).
        """
        return np.exp(self.lnP(k, z, kind))

    def P_nl(self, k, z):
        """Return the Halofit nonlinear total-matter P(k, z) in (Mpc/h)^3.

        Arguments:
          k = wavenumber in h/Mpc, scalar or array.
          z = redshift, broadcast against k (lnP).

        Returns:
          numpy array of the broadcast shape of (k, z).
        """
        return np.exp(self.lnP(k, z, "nl"))

    def export_tables(self):
        """Return the CAMB tables and the background as a flat dict.

        reference_cluster.ClusterReference.results saves this dict in its
        .npz output, so a comparison can start from the inputs of this
        reference.

        Returns:
          dict with cosmo_z_pk (nz,) and cosmo_k_pk (nk,) in h/Mpc, the
          tables cosmo_lnP_lin, cosmo_lnP_nl and cosmo_lnP_cb (nz, nk) of
          ln P in (Mpc/h)^3, and on cosmo_z_bg (4001 redshifts on [0, 4],
          step 0.001) cosmo_chi (Mpc/h), cosmo_E = H/H0 and cosmo_D (the
          growth factor).
        """
        zb = np.linspace(0.0, 4.0, 4001)
        return {
            "cosmo_z_pk": self.z_pk, "cosmo_k_pk": self.k_pk,
            "cosmo_lnP_lin": self._tabs["lin"], "cosmo_lnP_nl": self._tabs["nl"],
            "cosmo_lnP_cb": self._tabs["cb"],
            "cosmo_z_bg": zb, "cosmo_chi": self.chi(zb), "cosmo_E": self.E(zb),
            "cosmo_D": self.growth(zb),
        }


class NzBins:
    """Tomographic n(z) of a cosmolike-format text file (z, n_1, ..., n_N).

    Mirrors redshift_spline.c (see the module docstring): Z_LOW convention,
    rectangle-sum normalization, natural cubic spline through the cell
    centres. `norm="spline"` normalizes the spline itself to unit integral
    instead (not the C convention). Calling the object, nz(z, i, ...),
    evaluates bin i (__call__).
    """

    def __init__(self, path, convention="z_low", norm="rect"):
        """Read an n(z) file and build one normalized spline per bin.

        Arguments:
          path = cosmolike-format text file: column 0 = z, columns
                 1..N = n_i(z) of the N tomographic bins, in any
                 normalization.
          convention = "z_low" (default, cosmolike's Z_LOW: the z column
                       holds left cell edges, so the values sit half a
                       cell higher, at the centres); any other value puts
                       the values on the nodes zmin_all + k dz (no
                       offset).
          norm = "rect" (default: divide each bin by its rectangle sum
                 sum_k n_k dz, as redshift_spline.c does) or "spline"
                 (also rescale each spline to unit integral between the
                 first and the last node; not the C convention). Any
                 value other than "spline" acts as "rect".

        The object keeps z_nodes (one node per file row: the cell centres
        for Z_LOW), dz, nbin, norm_rect, splines (one scipy CubicSpline
        per bin) and zmean (the mean z of each unshifted bin, the pivot of
        the lens stretch).
        """
        data = np.loadtxt(path)
        zf = data[:, 0]
        raw = data[:, 1:]
        n = len(zf)
        # Table range as the C loader sets it (generic_interface.cpp): the
        # lower end floored at z = 1e-5, the upper end one z step past the
        # last row (the right edge of the last Z_LOW cell); n cells of
        # width dz fill it.
        zmin_all = max(zf[0], 1e-5)
        zmax_all = zf[-1] + (zf[-1] - zf[0]) / (n - 1.0)
        dz = (zmax_all - zmin_all) / n
        off = 0.5 if convention == "z_low" else 0.0
        self.z_nodes = zmin_all + (np.arange(n) + off) * dz
        self.dz = dz
        self.nbin = raw.shape[1]
        # rectangle sum of each bin: the sum over the rows (axis 0) of each
        # column, times the cell width
        norm_rect = raw.sum(axis=0) * dz
        self.norm_rect = norm_rect
        self.splines = []
        for i in range(self.nbin):
            spl = CubicSpline(self.z_nodes, raw[:, i] / norm_rect[i], bc_type="natural")
            if norm == "spline":
                tot = spl.integrate(self.z_nodes[0], self.z_nodes[-1])
                spl = CubicSpline(self.z_nodes, raw[:, i] / norm_rect[i] / tot,
                                  bc_type="natural")
            self.splines.append(spl)
        # zmean[i] = int z n_i dz/int n_i dz, trapezoid rule on 30001 points
        # between the first and the last node, one entry per bin (a list
        # comprehension over the splines); the lens stretch pivots about it
        zz = np.linspace(self.z_nodes[0], self.z_nodes[-1], 30001)
        self.zmean = np.array([np.trapz(zz * s(zz), zz) / np.trapz(s(zz), zz)
                               for s in self.splines])

    def support(self, i, shift=0.0, stretch=1.0):
        """Return the ends (z_lo, z_hi) of the open interval outside which
        n_i(z; shift, stretch) is zero.

        The first and the last node are moved by the shift-and-stretch
        map z -> zmean_i + stretch (z - zmean_i) + shift of __call__.

        Arguments:
          i = bin index, 0 <= i < nbin.
          shift = photo-z shift dz of the bin (default 0).
          stretch = photo-z stretch s of the bin (default 1; lenses only).

        Returns:
          tuple (z_lo, z_hi) of floats.
        """
        zb = self.zmean[i]
        # a generator expression inside tuple(): the map applied to the
        # first (j = 0) and to the last (j = -1) node
        return tuple(zb + stretch * (self.z_nodes[j] - zb) + shift for j in (0, -1))

    def __call__(self, z, i, shift=0.0, stretch=1.0):
        """Return the shifted and stretched n_i(z), per unit redshift.

        Calling the object, nz(z, i, shift, stretch), runs this method:
        n_i(z) = spline_i((z - shift - zmean_i)/stretch + zmean_i)/stretch,
        and zero where the mapped redshift lies outside the open interval
        between the first and the last node. The 1/stretch factor keeps
        the integral over z unchanged.

        Arguments:
          z = redshift, scalar or array.
          i = bin index, 0 <= i < nbin.
          shift = photo-z shift dz (default 0).
          stretch = photo-z stretch s (default 1).

        Returns:
          numpy array of the shape of z.
        """
        z = np.asarray(z, dtype=float)
        zz = (z - shift - self.zmean[i]) / stretch + self.zmean[i]
        out = self.splines[i](zz) / stretch
        inside = (zz > self.z_nodes[0]) & (zz < self.z_nodes[-1])
        return np.where(inside, out, 0.0)
