#!/usr/bin/env python
"""Tabulate the DES Y3 redMaPPer cluster counts per (z_lambda bin, richness bin).

The table holds the counts N of the observed Y3 cluster sample and the
angular number densities n_cA = N/Omega_s they imply (the cluster shot
noise of a two-point covariance is 1/n_cA). The likelihoods do not read it:
make_synthetic_data.py compares the model counts with its column N, and
make_cluster_mask.py reads N for its optional w_cc pair-count cut (off by
default). No script of the project reads the n_cA columns; the synthetic
covariance takes its shot noise from the counts of the synthetic data
vector. The richness lambda (lambda_chisq in the catalog) is the
membership-weighted number of red-sequence galaxies redMaPPer assigns to
a cluster, the mass proxy of the analysis; z_lambda is its photometric
redshift. The options are parsed by argparse (the Python standard-library
parser of command-line options), and --help prints this text.

INPUT: the public DES Y3 redMaPPer cosmology catalog,
  https://desdr-server.ncsa.illinois.edu/despublic/y3a2_files/y3kp_clusters/
  data/y3_redmapper_v6.4.22+2_release.h5
either the h5 file itself (--h5; needs h5py) or an npz extract of it
(--npz) holding the arrays
  z_lambda, lambda_chisq     (/catalog/cluster/...)
  select                     (/index/redmapper/lgt20/select)
and optionally z_lambda_e (/catalog/cluster/z_lambda_e).
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
numpy.histogram2d, which does the binning, also counts a value equal to
the last edge of an axis (z_lambda = 0.65 or lambda = 500) in the last
bin; the half-open statement holds for every other edge.
Also printed: the median z_lambda_e/(1+z_lambda) per z bin (z_lambda_e is
the catalog's photo-z error) when the input carries z_lambda_e.

Usage (from the cocoa/Cocoa folder; one of --npz or --h5 is required):
  python ./projects/des_cluster/scripts/make_y3_redmapper_counts.py \\
      --npz FILE [--area 4143] [--out FILE]
  python ./projects/des_cluster/scripts/make_y3_redmapper_counts.py \\
      --h5 y3_redmapper_v6.4.22+2_release.h5
"""
import argparse
import os
import numpy as np

# The project folder (the parent of scripts/), found from this script's
# own path (__file__), so the default output path does not depend on the
# working directory.
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# steradians per square degree: (pi/180)^2
DEG2_TO_SR = (np.pi/180.0)**2
# square arcminutes per steradian: (180 x 60/pi)^2
ARCMIN2_PER_SR = (180.0*60.0/np.pi)**2


def parse_list(text, tp=float):
  """Split a comma- or space-separated option string into a list of numbers.

  Arguments:
    text = the option string, e.g. "20,30,45,60,500".
    tp = the conversion applied to every item: float (default) or int.

  Returns:
    list of tp values in the order of the string.

  Raises:
    ValueError when an item does not convert (e.g. "20;30").
  """
  # commas become spaces, split() cuts at every run of whitespace, and the
  # list comprehension converts each piece with tp
  return [tp(x) for x in text.replace(",", " ").split()]


def load_catalog(args):
  """Read z_lambda, richness and photo-z error of the cosmology sample.

  The select index (/index/redmapper/lgt20/select of the h5 file, or the
  "select" array of the npz extract) restricts each catalog column to the
  cosmology sample of arXiv 2503.13632. With --h5 the file is read with
  h5py, imported only in that branch so that --npz works without it;
  [:] reads a whole h5 column into memory before the selection.

  Arguments:
    args = the parsed options; uses args.h5 and args.npz (--h5 wins when
      both are given).

  Returns:
    (z, richness, z_err, source): numpy arrays z_lambda, lambda_chisq and
    z_lambda_e of the selected clusters (z_err is None for an npz extract
    without z_lambda_e), and the path that was read.

  Raises:
    SystemExit when neither --npz nor --h5 is given.
  """
  if args.h5 is not None:
    import h5py
    # the with block closes the h5 file when it ends
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
  """Count the clusters per (z_lambda, richness) bin and write the table.

  Arguments:
    none; the options come from the command line through argparse.

  Returns:
    nothing.

  Raises:
    SystemExit from load_catalog when no input is given.

  Side effects:
    writes --out (default data/y3_redmapper_counts.txt; an existing file
    is overwritten) and prints the counts, the densities n_cA and, when
    the input has z_lambda_e, the median photo-z error per z bin. A
    warning is printed when clusters lie at or above the last richness
    edge.
  """
  # __doc__ is the module docstring; RawDescriptionHelpFormatter makes
  # --help print it with its line breaks kept
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
  # histogram2d returns (counts, z edges, richness edges); "_" discards the
  # two edge arrays. counts has shape (z bins, richness bins) and float
  # type, hence astype(int).
  counts, _, _ = np.histogram2d(z, richness, bins=[zedges, ledges])
  counts = counts.astype(int)
  # survey solid angle Omega_s in steradians
  omega_s = args.area*DEG2_TO_SR

  if np.any(richness >= ledges[-1]):
    print(f"WARNING: {np.sum(richness >= ledges[-1])} clusters above the "
          f"last richness edge {ledges[-1]:g} are not counted")

  # each entry becomes one '#' line of the file; adjacent string literals
  # are joined into one string before the % formatting applies
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
  # the with block closes the file when it ends, also after an error; the
  # rows run over the richness bins inside each z bin (z bin outer), the
  # order read_counts of make_cluster_mask.py expects
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
      # clusters of this z bin above the lowest richness edge (& is the
      # element-wise AND of boolean arrays)
      inside = (z >= zedges[i]) & (z < zedges[i+1]) & (richness >= ledges[0])
      print(f"    median z_lambda_e/(1+z_lambda) = "
            f"{np.median(z_err[inside]/(1 + z[inside])):.4f}")
  print("n_cA [1/sr]:")
  for i in range(len(zedges) - 1):
    print("   " + " ".join("%10.2f" % (c/omega_s) for c in counts[i]))


if __name__ == "__main__":
  main()
