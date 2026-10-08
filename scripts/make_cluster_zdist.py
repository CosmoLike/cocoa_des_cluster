#!/usr/bin/env python
"""Tabulate the cluster selection kernels <phi_i|z> of des_cluster on a fine z grid.

The table (data/des_y6_cluster.nz by default) is the file the likelihoods
read through the key nz_cluster_file of data/des_cluster_y6.dataset (the
".dataset" descriptor: a small text file of `key = value` lines naming
the data files and the binning of the analysis). The options are parsed
by argparse, the Python standard-library parser of command-line options,
and --help prints this text.

PHYSICS (Y1 eqs 6-7 of arXiv 2008.10757; eq 16 of arXiv 2503.13631):
  A cluster at true redshift z has a redMaPPer photometric redshift z_lambda
  drawn from a Gaussian of width sigma_z(z) = sigma_0 (1 + z). The
  probability that it lands in the z_lambda bin i = [zlo_i, zhi_i) is

    <phi_i|z> = 1/2 [ erf((zhi_i - z) / (sqrt(2) sigma_z(z)))
                    - erf((zlo_i - z) / (sqrt(2) sigma_z(z))) ].

  This is a probability (0 <= <phi_i|z> <= 1), not a normalized n(z): the
  counts (eq 16) and the cluster radial kernel q_i(z) ~ dV/dz <phi_i|z>
  (Y1 eq 15) multiply it by the volume element and the abundance, so the
  table must not be normalized. Well inside the outer edges the kernels
  of all bins sum to 1: a cluster lands in exactly one z_lambda bin.

  sigma_0 = 0.006 is the Y3 redMaPPer value (arXiv 2503.13632: sigma(z)/(1+z)
  about 0.006 from 1955 spectroscopic centrals; the public Y3 catalog gives
  median z_lambda_e/(1+z_lambda) = 0.0060 / 0.0063 / 0.0066 in the three
  bins, available with --sigma-per-bin). sigma_z is evaluated at the true
  redshift z (the kernel is a function of z_true).

  --tophat gives the sigma -> 0 limit: 1 inside the bin, 0 outside, and 1/2
  on a grid node that sits exactly on an edge (the erf limit). When the
  edges sit on grid nodes (the default edges are multiples of dz), the
  trapezoid integral of each column equals the bin width. A step is not
  smooth: interpolate a top-hat table linearly, not with a cubic spline
  (the C code, redshift_spline_cluster.c, reads the table linearly).

GRID: uniform, spacing --dz (default 5e-4), covering every bin's support
[zlo - n sigma_z(zlo), zhi + n sigma_z(zhi)] with n = --nsigma (default 6);
end points snapped to multiples of dz, and no node below z = 0. With
sigma_0 = 0.006, a step of at most 5e-4 puts at least 12 nodes inside one
sigma_z = 0.006 (1 + z), the bound the C code sets for its fine z grid
(redshift_spline_cluster.c); a coarser --dz is refused. Outside its own
support a column is exactly zero. At n = 6 and sigma_0 = 0.006 the cut
removes kernel values up to 2.4e-10 below the bin and up to 3.5e-9 above
it: sigma_z grows with z, so the upper support edge lies only
n/(1 + n sigma_0) = 5.8 local widths sigma_z(z) away from zhi.

OUTPUT FORMAT: '#' comment lines, then one row per node: z, <phi_1|z>, ...,
<phi_N|z>. The values are samples at z (Z_MID semantics): a reader must
not apply the half-cell Z_LOW offset of the galaxy n(z) files (the Z_LOW
reading takes the z column as left bin edges and places each value at
the cell center z + dz/2).

Usage (from the cocoa/Cocoa folder; every option has a default):
  python ./projects/des_cluster/scripts/make_cluster_zdist.py
      [--edges 0.2,0.4,0.55,0.65] [--sigma0 0.006]
      [--sigma-per-bin 0.0060,0.0063,0.0066] [--tophat] [--out FILE]
"""
import argparse
import os
import numpy as np
from scipy.special import erf

# The project folder (the parent of scripts/): __file__ is this script's
# own path and each dirname strips one level, so the default output path
# does not depend on the folder the script is run from.
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Table I fiducial cosmology of arXiv 2503.13631 (flat; Omega_m includes the
# massive neutrinos, non-relativistic at the redshifts used here). In this
# file only the printed volume-weighted mean redshifts depend on it, not
# the kernel table. make_cluster_mask.py imports hubble_ratio and
# comoving_distance from here, so its scale-cut angles R/chi(zbar) and its
# cluster-bin mean redshifts use this cosmology too. Distances in Mpc/h
# do not depend on h except through the photon density Omega_gamma =
# 2.473e-5/h^2 (the only radiation term: massless neutrinos are left out).
OMEGA_M = 0.3
H_LITTLE = 0.69
OMEGA_GAMMA = 2.473e-5 / H_LITTLE**2  # photons, T_CMB = 2.7255 K


def parse_list(text, tp=float):
  """Split a comma- or space-separated option string into a list of numbers.

  Arguments:
    text = the option string, e.g. "0.2,0.4,0.55,0.65" or "0.2 0.4".
    tp = the conversion applied to every item: float (default) or int.

  Returns:
    list of tp values in the order of the string.

  Raises:
    ValueError when an item does not convert (e.g. "0.2;0.4").
  """
  # commas become spaces, split() cuts at every run of whitespace, and the
  # list comprehension converts each piece with tp
  return [tp(x) for x in text.replace(",", " ").split()]


def hubble_ratio(z):
  """E(z) = H(z)/H0 of the fiducial flat cosmology.

  E(z)^2 = Omega_m (1 + z)^3 + Omega_gamma (1 + z)^4 + Omega_Lambda, with
  Omega_Lambda = 1 - Omega_m - Omega_gamma (flat) and the module constants
  OMEGA_M and OMEGA_GAMMA.

  Arguments:
    z = redshift, a float or a numpy array of any shape.

  Returns:
    E(z), dimensionless, with the shape of z.
  """
  omega_lambda = 1.0 - OMEGA_M - OMEGA_GAMMA
  return np.sqrt(OMEGA_M*(1 + z)**3 + OMEGA_GAMMA*(1 + z)**4 + omega_lambda)


def comoving_distance(z):
  """chi(z) in Mpc/h on the nodes z (trapezoid on a fine grid from 0).

  chi(z) = (c/H0) int_0^z dz'/E(z'), with c/H0 = 2997.92458 Mpc/h (the
  speed of light over 100 km/s/Mpc). The integral is a cumulative
  trapezoid sum on 20001 uniform nodes between 0 and max(z), read at each
  requested z by linear interpolation (np.interp). The cosmology is flat,
  so chi is also the transverse comoving distance: an angle theta at
  redshift z spans the comoving separation R = chi(z) theta.

  Arguments:
    z = redshift, a float or a 1-D numpy array, every value >= 0.

  Returns:
    chi in Mpc/h, with the shape of z.
  """
  c_over_h0 = 2997.92458  # Mpc/h
  zfine = np.linspace(0.0, float(np.max(z)), 20001)
  integrand = 1.0 / hubble_ratio(zfine)
  # cumulative trapezoid: cumsum adds the cell areas 0.5 (f_k + f_k+1) dz
  # one by one, and the leading 0.0 is chi(0), so chi_fine[k] is the
  # integral from 0 to zfine[k]
  chi_fine = np.concatenate(([0.0], np.cumsum(
      0.5*(integrand[1:] + integrand[:-1])*np.diff(zfine))))
  return c_over_h0*np.interp(z, zfine, chi_fine)


def selection_kernel(z, zlo, zhi, sigma0, nsigma, tophat):
  """Return <phi_i|z> of one z_lambda bin [zlo, zhi) on the nodes z.

  The Gaussian kernel is the erf difference of the module docstring with
  sigma_z = sigma0 (1 + z) at the true redshift z. It is set to exactly
  zero outside the support [zlo - nsigma sigma_z(zlo), zhi + nsigma
  sigma_z(zhi)], and clipped to [0, 1] so that round-off cannot leave a
  probability outside that range. The top-hat kernel is 1 inside the bin,
  0 outside and 1/2 on a node within 1e-12 of an edge, the sigma -> 0
  limit of the erf form.

  Arguments:
    z = true-redshift nodes, 1-D numpy array.
    zlo = lower z_lambda edge of the bin.
    zhi = upper z_lambda edge of the bin, > zlo.
    sigma0 = photo-z width over (1 + z), e.g. 0.006; unused with tophat.
    nsigma = half-width of the support in units of sigma_z; unused with
      tophat.
    tophat = True for the sigma -> 0 top-hat kernel.

  Returns:
    numpy float array with the shape of z, values in [0, 1] (a
    probability per true redshift, not normalized).
  """
  if tophat:
    # np.where(condition, a, b) takes a where the condition holds and b
    # elsewhere; & and | are the element-wise AND and OR of boolean arrays
    phi = np.where((z > zlo) & (z < zhi), 1.0, 0.0)
    on_edge = np.isclose(z, zlo, rtol=0, atol=1e-12) | \
              np.isclose(z, zhi, rtol=0, atol=1e-12)
    # a boolean array as index selects the True positions: every edge node
    phi[on_edge] = 0.5
    return phi
  sigma = sigma0*(1.0 + z)
  phi = 0.5*(erf((zhi - z)/(np.sqrt(2.0)*sigma)) -
             erf((zlo - z)/(np.sqrt(2.0)*sigma)))
  support_lo = zlo - nsigma*sigma0*(1.0 + zlo)
  support_hi = zhi + nsigma*sigma0*(1.0 + zhi)
  # every node outside the support becomes an exact zero
  phi[(z < support_lo) | (z > support_hi)] = 0.0
  return np.clip(phi, 0.0, 1.0)


def main():
  """Parse the options, tabulate the kernels and write the table.

  The options are validated first (increasing edges, one sigma per bin, a
  step in (0, 5e-4]). Then the uniform grid is built, each column filled
  with selection_kernel, and the volume-weighted mean true redshift of
  each bin computed with the weight dV/dz <phi_i|z> (Y1 eq 15; dV/dz is
  proportional to chi^2/E(z) per steradian). The printed summary checks
  the table: the support of each column, the trapezoid integral of phi
  (close to the bin width), the largest value, and the sum over bins well
  inside the outer edges (1 when every cluster lands in exactly one bin).

  Arguments:
    none; the options come from the command line through argparse.

  Returns:
    nothing.

  Raises:
    SystemExit with a message when --edges is not increasing or has fewer
    than two values, when --sigma-per-bin does not give one value per
    bin, or when --dz is outside (0, 5e-4].

  Side effects:
    writes --out (default data/des_y6_cluster.nz; an existing file is
    overwritten) and prints the summary.
  """
  # __doc__ is the module docstring; RawDescriptionHelpFormatter makes
  # --help print it with its line breaks kept
  parser = argparse.ArgumentParser(description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--edges", default="0.2,0.4,0.55,0.65",
                      help="z_lambda bin edges (comma or space separated)")
  parser.add_argument("--sigma0", type=float, default=0.006,
                      help="sigma_z/(1+z) of every bin")
  parser.add_argument("--sigma-per-bin", default=None,
                      help="sigma_z/(1+z) per bin, overrides --sigma0")
  parser.add_argument("--dz", type=float, default=5e-4)
  parser.add_argument("--nsigma", type=float, default=6.0)
  # action="store_true": a flag without a value, False unless given
  parser.add_argument("--tophat", action="store_true")
  parser.add_argument("--out", default=os.path.join(PROJECT_DIR, "data",
                                                    "des_y6_cluster.nz"))
  args = parser.parse_args()

  edges = parse_list(args.edges)
  nbin = len(edges) - 1
  if nbin < 1 or np.any(np.diff(edges) <= 0):
    raise SystemExit("--edges must be increasing, at least two values")
  if args.sigma_per_bin is None:
    # list repetition: nbin copies of the one value
    sigma0 = [args.sigma0]*nbin
  else:
    sigma0 = parse_list(args.sigma_per_bin)
    if len(sigma0) != nbin:
      raise SystemExit("--sigma-per-bin needs one value per bin")
  # 5e-4 keeps at least 12 nodes per sigma_z = 0.006 (1 + z) (module
  # docstring, GRID)
  if not (0 < args.dz <= 5e-4):
    raise SystemExit("--dz must be in (0, 5e-4]")

  # uniform grid covering every bin's support, nodes on multiples of dz:
  # min(...) and max(...) run over a generator expression, giving the
  # lowest lower support edge and the highest upper one over all bins
  zmin = min(edges[i] - args.nsigma*sigma0[i]*(1 + edges[i])
             for i in range(nbin))
  zmax = max(edges[i+1] + args.nsigma*sigma0[i]*(1 + edges[i+1])
             for i in range(nbin))
  # node k sits at z = k dz; floor and ceil move the ends outward onto
  # nodes, and max(zmin, 0.0) keeps the first node at z >= 0
  kmin = int(np.floor(max(zmin, 0.0)/args.dz))
  kmax = int(np.ceil(zmax/args.dz))
  # rounding to 10 decimals removes the round-off of k*dz, so the nodes
  # are the decimal multiples of dz that the file prints
  z = np.round(np.arange(kmin, kmax + 1)*args.dz, 10)

  # phi has shape (number of nodes, number of bins): one column per bin,
  # the layout of the output rows (np.empty leaves the values unset; the
  # loop fills every column)
  phi = np.empty((z.size, nbin))
  for i in range(nbin):
    phi[:, i] = selection_kernel(z, edges[i], edges[i+1], sigma0[i],
                                 args.nsigma, args.tophat)

  # volume-weighted mean true redshift of each bin (Y1 eq 15 kernel):
  # dV/dz per steradian is (c/H0) chi^2/E(z); the constant c/H0 cancels in
  # the ratio, and so does dz on the uniform grid. The list comprehension
  # gives one mean per bin.
  dvdz = comoving_distance(z)**2/hubble_ratio(z)
  zmean = [float(np.sum(z*dvdz*phi[:, i])/np.sum(dvdz*phi[:, i]))
           for i in range(nbin)]

  # conditional expression: the first string with --tophat, else the second
  kind = "top-hat (sigma -> 0 limit)" if args.tophat else \
         "Gaussian photo-z, sigma_z = sigma_0 (1 + z_true)"
  # each entry becomes one '#' line of the file; adjacent string literals
  # are joined into one string before the % formatting applies
  header = [
    "DES cluster selection kernels <phi_i|z_true> (des_cluster, arXiv "
    "2503.13631 eq 16, arXiv 2008.10757 eqs 6-7)",
    "kernel: " + kind,
    "z_lambda edges: " + " ".join("%g" % e for e in edges),
    "sigma_0 per bin: " + ("n/a" if args.tophat else
                          " ".join("%g" % s for s in sigma0)),
    "grid: uniform dz = %g, %d nodes, z in [%g, %g]; support per bin = edges"
    " +- %g sigma_z" % (args.dz, z.size, z[0], z[-1], args.nsigma),
    "columns: z_true, then <phi_i|z> of bin 1 .. %d; a probability (<= 1),"
    " NOT normalized" % nbin,
    "values are samples AT z (Z_MID semantics, no half-cell offset)",
    "volume-weighted mean z (Table I cosmology): "
    + " ".join("%.4f" % m for m in zmean),
  ]
  # the with block closes the file when it ends, also after an error
  with open(args.out, "w") as f:
    for line in header:
      f.write("# " + line + "\n")
    for n in range(z.size):
      f.write("%.6f" % z[n])
      for i in range(nbin):
        f.write(" %.10e" % phi[n, i])
      f.write("\n")

  print(f"wrote {args.out}: {z.size} nodes, dz = {args.dz:g}, "
        f"z in [{z[0]:.4f}, {z[-1]:.4f}], {nbin} bins ({kind})")
  # trapezoid integral of every column at once: np.diff(z)[:, None] is the
  # node spacing as a column (n - 1, 1), which numpy broadcasting repeats
  # across the bins of phi[1:] + phi[:-1] (n - 1, nbin); the sum over
  # axis 0 leaves one integral per bin
  width_integral = 0.5*np.sum((phi[1:] + phi[:-1])*np.diff(z)[:, None], axis=0)
  for i in range(nbin):
    # np.nonzero returns one index array per dimension; [0] holds the
    # nodes where the column is positive, so its ends bound the support
    nonzero = np.nonzero(phi[:, i] > 0)[0]
    print(f"  bin {i+1} [{edges[i]:g}, {edges[i+1]:g}): support "
          f"[{z[nonzero[0]]:.4f}, {z[nonzero[-1]]:.4f}], "
          f"int phi dz = {width_integral[i]:.6f} (width {edges[i+1]-edges[i]:g}),"
          f" max phi = {phi[:, i].max():.6f}, volume-weighted <z> = {zmean[i]:.4f}")
  # completeness check: away from the outer edges every cluster lands in
  # one bin, so the kernels sum to 1. The 0.05 margin is 5 to 7 sigma_z
  # at sigma_0 = 0.006: there the outer erf tails lower the sum by at most
  # about 1e-7.
  total = phi.sum(axis=1)
  inside = (z > edges[0] + 0.05) & (z < edges[-1] - 0.05)
  print(f"  sum over bins inside [{edges[0]+0.05:g}, {edges[-1]-0.05:g}]: "
        f"min {total[inside].min():.10f} max {total[inside].max():.10f}")


if __name__ == "__main__":
  main()
