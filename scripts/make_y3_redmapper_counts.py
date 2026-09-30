#!/usr/bin/env python
"""
Tabulate the DES Y3 redMaPPer cluster counts per (z_lambda bin, richness
bin) and the implied angular number densities n_cA, for the synthetic
covariance of the des_cluster project (cluster shot noise 1/n_cA).

INPUT: the public DES Y3 redMaPPer cosmology catalog,
  https://desdr-server.ncsa.illinois.edu/despublic/y3a2_files/y3kp_clusters/
  data/y3_redmapper_v6.4.22+2_release.h5
either the h5 file itself (--h5; needs h5py) or an npz extract of it
(--npz) holding the arrays
  z_lambda, lambda_chisq     (/catalog/cluster/...)
  select                     (/index/redmapper/lgt20/select)
The select index restricts the catalog to the cosmology sample of arXiv
2503.13632 (lambda >= 20 inside the joint 3x2pt footprint); with it the
counts are 5632 / 6308 / 4551 in z_lambda [0.2,0.4), [0.4,0.55), [0.55,0.65),
16491 in total, as in that paper.

OUTPUT: a text table (default data/y3_redmapper_counts.txt) with '#'
provenance lines and one row per (z bin, richness bin):
  z_lo z_hi lambda_lo lambda_hi N n_cA[1/sr] n_cA[1/arcmin^2]
with n_cA = N / Omega_s, Omega_s = area (deg^2) x (pi/180)^2 sr.
The area defaults to 4143 deg^2 (the Y3 cluster footprint of arXiv
2503.13632); the Y6-like lighthouse configuration uses 4125.3 deg^2.

Bins are half open: zlo <= z_lambda < zhi, lambda_lo <= lambda < lambda_hi
(the last richness edge, 500, lies above every Y3 cluster: lambda_max = 207).
Also printed: the median z_lambda_e/(1+z_lambda) per z bin when the input
carries z_lambda_e.

Usage:
  python make_y3_redmapper_counts.py --npz FILE [--area 4143] [--out FILE]
  python make_y3_redmapper_counts.py --h5 y3_redmapper_v6.4.22+2_release.h5
"""
import argparse
import os
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEG2_TO_SR = (np.pi/180.0)**2
ARCMIN2_PER_SR = (180.0*60.0/np.pi)**2


def parse_list(text, tp=float):
  return [tp(x) for x in text.replace(",", " ").split()]


def load_catalog(args):
  if args.h5 is not None:
    import h5py
    with h5py.File(args.h5, "r") as f:
      select = f["index/redmapper/lgt20/select"][:]
      cat = f["catalog/cluster"]
      z = cat["z_lambda"][:][select]
      richness = cat["lambda_chisq"][:][select]
      z_err = cat["z_lambda_e"][:][select]
    return z, richness, z_err, args.h5
  if args.npz is not None:
    data = np.load(args.npz)
    select = data["select"]
    z = data["z_lambda"][select]
    richness = data["lambda_chisq"][select]
    z_err = data["z_lambda_e"][select] if "z_lambda_e" in data.files else None
    return z, richness, z_err, args.npz
  raise SystemExit("give --npz or --h5")


def main():
  parser = argparse.ArgumentParser(description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--npz", default=None)
  parser.add_argument("--h5", default=None)
  parser.add_argument("--zbin-edges", default="0.2,0.4,0.55,0.65")
  parser.add_argument("--richness-edges", default="20,30,45,60,500")
  parser.add_argument("--area", type=float, default=4143.0,
                      help="survey area in deg^2")
  parser.add_argument("--out", default=os.path.join(PROJECT_DIR, "data",
                                                    "y3_redmapper_counts.txt"))
  args = parser.parse_args()

  z, richness, z_err, source = load_catalog(args)
  zedges = parse_list(args.zbin_edges)
  ledges = parse_list(args.richness_edges)
  counts, _, _ = np.histogram2d(z, richness, bins=[zedges, ledges])
  counts = counts.astype(int)
  omega_s = args.area*DEG2_TO_SR

  if np.any(richness >= ledges[-1]):
    print(f"WARNING: {np.sum(richness >= ledges[-1])} clusters above the "
          f"last richness edge {ledges[-1]:g} are not counted")

  header = [
    "DES Y3 redMaPPer cluster counts per (z_lambda bin, richness bin)",
    "catalog: y3_redmapper_v6.4.22+2_release.h5 (DES Y3 public release,"
    " y3kp_clusters), cosmology sample index/redmapper/lgt20/select",
    "read from: " + os.path.abspath(source),
    "sample of arXiv 2503.13632: z_lambda counts " +
    " / ".join(str(c) for c in counts.sum(axis=1)) +
    " (total %d)" % counts.sum(),
    "bins half open: z_lo <= z_lambda < z_hi, lambda_lo <= lambda_chisq <"
    " lambda_hi",
    "n_cA = N / Omega_s with area %.1f deg^2 = %.6f sr (Y3 cluster"
    " footprint; the local Y6 lighthouse config uses 4125.3 deg^2)"
    % (args.area, omega_s),
    "columns: z_lo z_hi lambda_lo lambda_hi N n_cA[1/sr] n_cA[1/arcmin^2]",
  ]
  with open(args.out, "w") as f:
    for line in header:
      f.write("# " + line + "\n")
    for i in range(len(zedges) - 1):
      for a in range(len(ledges) - 1):
        n_sr = counts[i, a]/omega_s
        f.write("%.3f %.3f %g %g %d %.6e %.6e\n"
                % (zedges[i], zedges[i+1], ledges[a], ledges[a+1],
                   counts[i, a], n_sr, n_sr/ARCMIN2_PER_SR))

  print(f"wrote {args.out}")
  print(f"area {args.area:g} deg^2 = {omega_s:.6f} sr")
  print("z bin            " + "".join("  [%g,%g)" % (ledges[a], ledges[a+1])
                                      for a in range(len(ledges) - 1))
        + "    total   n_c[1/sr]")
  for i in range(len(zedges) - 1):
    row = "".join("%10d" % c for c in counts[i])
    print(f"[{zedges[i]:.2f},{zedges[i+1]:.2f})    {row} {counts[i].sum():8d}"
          f"  {counts[i].sum()/omega_s:10.1f}")
    if z_err is not None:
      inside = (z >= zedges[i]) & (z < zedges[i+1]) & (richness >= ledges[0])
      print(f"    median z_lambda_e/(1+z_lambda) = "
            f"{np.median(z_err[inside]/(1 + z[inside])):.4f}")
  print("n_cA [1/sr]:")
  for i in range(len(zedges) - 1):
    print("   " + " ".join("%10.2f" % (c/omega_s) for c in counts[i]))


if __name__ == "__main__":
  main()
