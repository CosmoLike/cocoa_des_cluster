#!/usr/bin/env python
"""
Tabulate the cluster redshift-selection kernels <phi_i|z_true> of the DES
cluster analysis (des_cluster project) on a uniform fine z grid.

PHYSICS (Y1 eqs 6-7 of arXiv 2008.10757; eq 16 of arXiv 2503.13631):
  A cluster at true redshift z has a redMaPPer photometric redshift z_lambda
  drawn from a Gaussian of width sigma_z(z) = sigma_0 (1 + z). The
  probability that it lands in the z_lambda bin i = [zlo_i, zhi_i) is

    <phi_i|z> = 1/2 [ erf((zhi_i - z) / (sqrt(2) sigma_z(z)))
                    - erf((zlo_i - z) / (sqrt(2) sigma_z(z))) ].

  This is a probability (0 <= <phi_i|z> <= 1), NOT a normalized n(z): the
  counts (eq 16) and the cluster radial kernel q_i(z) ~ dV/dz <phi_i|z>
  (Y1 eq 15) multiply it by the volume element and the abundance, so the
  table must not be normalized.

  sigma_0 = 0.006 is the Y3 redMaPPer value (arXiv 2503.13632: sigma(z)/(1+z)
  about 0.006 from 1955 spectroscopic centrals; the public Y3 catalog gives
  median z_lambda_e/(1+z_lambda) = 0.0060 / 0.0063 / 0.0066 in the three
  bins, available with --sigma-per-bin). sigma_z is evaluated at the true
  redshift z (the kernel is a function of z_true).

  --tophat gives the sigma -> 0 limit: 1 inside the bin, 0 outside, and 1/2
  on a grid node that sits exactly on an edge (the erf limit), so the
  trapezoid integral of each column equals the bin width. A step is not
  smooth: interpolate a top-hat table linearly, not with a cubic spline.

GRID: uniform, spacing --dz (default 5e-4), covering every bin's support
[zlo - n sigma_z(zlo), zhi + n sigma_z(zhi)] with n = --nsigma (default 6);
end points snapped to multiples of dz. Outside its own support a column is
exactly zero (the erf tail there is below 1e-9).

OUTPUT FORMAT: '#' comment lines, then one row per node: z, <phi_1|z>, ...,
<phi_N|z>. The values are samples AT z (Z_MID semantics): a reader must not
apply the half-cell Z_LOW offset of the galaxy n(z) files.

Usage:
  python make_cluster_zdist.py [--edges 0.2,0.4,0.55,0.65] [--sigma0 0.006]
      [--sigma-per-bin 0.0060,0.0063,0.0066] [--tophat] [--out FILE]
"""
import argparse
import os
import numpy as np
from scipy.special import erf

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Table I fiducial cosmology of arXiv 2503.13631 (flat; Omega_m includes the
# massive neutrinos, non-relativistic at the redshifts used here). Only the
# printed volume-weighted mean redshifts depend on it.
OMEGA_M = 0.3
H_LITTLE = 0.69
OMEGA_GAMMA = 2.473e-5 / H_LITTLE**2  # photons, T_CMB = 2.7255 K


def parse_list(text, tp=float):
  return [tp(x) for x in text.replace(",", " ").split()]


def hubble_ratio(z):
  """E(z) = H(z)/H0 of the fiducial flat cosmology."""
  omega_lambda = 1.0 - OMEGA_M - OMEGA_GAMMA
  return np.sqrt(OMEGA_M*(1 + z)**3 + OMEGA_GAMMA*(1 + z)**4 + omega_lambda)


def comoving_distance(z):
  """chi(z) in Mpc/h on the nodes z (trapezoid on a fine grid from 0)."""
  c_over_h0 = 2997.92458  # Mpc/h
  zfine = np.linspace(0.0, float(np.max(z)), 20001)
  integrand = 1.0 / hubble_ratio(zfine)
  chi_fine = np.concatenate(([0.0], np.cumsum(
      0.5*(integrand[1:] + integrand[:-1])*np.diff(zfine))))
  return c_over_h0*np.interp(z, zfine, chi_fine)


def selection_kernel(z, zlo, zhi, sigma0, nsigma, tophat):
  if tophat:
    phi = np.where((z > zlo) & (z < zhi), 1.0, 0.0)
    on_edge = np.isclose(z, zlo, rtol=0, atol=1e-12) | \
              np.isclose(z, zhi, rtol=0, atol=1e-12)
    phi[on_edge] = 0.5
    return phi
  sigma = sigma0*(1.0 + z)
  phi = 0.5*(erf((zhi - z)/(np.sqrt(2.0)*sigma)) -
             erf((zlo - z)/(np.sqrt(2.0)*sigma)))
  support_lo = zlo - nsigma*sigma0*(1.0 + zlo)
  support_hi = zhi + nsigma*sigma0*(1.0 + zhi)
  phi[(z < support_lo) | (z > support_hi)] = 0.0
  return np.clip(phi, 0.0, 1.0)


def main():
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
  parser.add_argument("--tophat", action="store_true")
  parser.add_argument("--out", default=os.path.join(PROJECT_DIR, "data",
                                                    "des_y6_cluster.nz"))
  args = parser.parse_args()

  edges = parse_list(args.edges)
  nbin = len(edges) - 1
  if nbin < 1 or np.any(np.diff(edges) <= 0):
    raise SystemExit("--edges must be increasing, at least two values")
  if args.sigma_per_bin is None:
    sigma0 = [args.sigma0]*nbin
  else:
    sigma0 = parse_list(args.sigma_per_bin)
    if len(sigma0) != nbin:
      raise SystemExit("--sigma-per-bin needs one value per bin")
  if not (0 < args.dz <= 5e-4):
    raise SystemExit("--dz must be in (0, 5e-4]")

  # uniform grid covering every bin's support, nodes on multiples of dz
  zmin = min(edges[i] - args.nsigma*sigma0[i]*(1 + edges[i])
             for i in range(nbin))
  zmax = max(edges[i+1] + args.nsigma*sigma0[i]*(1 + edges[i+1])
             for i in range(nbin))
  kmin = int(np.floor(max(zmin, 0.0)/args.dz))
  kmax = int(np.ceil(zmax/args.dz))
  z = np.round(np.arange(kmin, kmax + 1)*args.dz, 10)

  phi = np.empty((z.size, nbin))
  for i in range(nbin):
    phi[:, i] = selection_kernel(z, edges[i], edges[i+1], sigma0[i],
                                 args.nsigma, args.tophat)

  # volume-weighted mean true redshift of each bin (Y1 eq 15 kernel)
  dvdz = comoving_distance(z)**2/hubble_ratio(z)
  zmean = [float(np.sum(z*dvdz*phi[:, i])/np.sum(dvdz*phi[:, i]))
           for i in range(nbin)]

  kind = "top-hat (sigma -> 0 limit)" if args.tophat else \
         "Gaussian photo-z, sigma_z = sigma_0 (1 + z_true)"
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
  width_integral = 0.5*np.sum((phi[1:] + phi[:-1])*np.diff(z)[:, None], axis=0)
  for i in range(nbin):
    nonzero = np.nonzero(phi[:, i] > 0)[0]
    print(f"  bin {i+1} [{edges[i]:g}, {edges[i+1]:g}): support "
          f"[{z[nonzero[0]]:.4f}, {z[nonzero[-1]]:.4f}], "
          f"int phi dz = {width_integral[i]:.6f} (width {edges[i+1]-edges[i]:g}),"
          f" max phi = {phi[:, i].max():.6f}, volume-weighted <z> = {zmean[i]:.4f}")
  total = phi.sum(axis=1)
  inside = (z > edges[0] + 0.05) & (z < edges[-1] - 0.05)
  print(f"  sum over bins inside [{edges[0]+0.05:g}, {edges[-1]-0.05:g}]: "
        f"min {total[inside].min():.10f} max {total[inside].max():.10f}")


if __name__ == "__main__":
  main()
