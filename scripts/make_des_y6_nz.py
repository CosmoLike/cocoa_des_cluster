#!/usr/bin/env python
"""
Write the DES Y6-like lens (MagLim, 6 bins) and source (4 bins) n(z) files
of the des_cluster project in the cosmolike n(z) text format.

OUTPUT FORMAT (the format of data/des_y3_lens.nz, read by read_nz_sample in
generic_interface.cpp): one row per redshift node, column 0 = z, column k =
n(z) of tomographic bin k (unnormalized; the reader normalizes each bin).
Lines starting with '#' are comments (read_table skips them).

INPUT: the Y6 n(z) of the lighthouse DES Y6 code comparison,
  lighthouse/analysis/des_y6_code_comparison/{lens,source}.nz
(300 rows, z = 0.00, 0.01, ..., 2.99; lens 6 columns, source 4 columns).

REDSHIFT CONVENTION (Z_LOW vs Z_MID):
  The cosmolike reader has two readings of the z column, chosen by the
  likelihood key photoz_zmid_convention (Ntable.photoz_zmid_convention):
    0 = Z_LOW: the column holds left bin edges and each value belongs to the
        cell center z + dz/2 (default of every des_cluster combo yaml);
    1 = Z_MID: the column holds the sample points themselves.
  Lighthouse reads these files as histograms with left edges (its printed
  lens means 0.311 / 0.440 / 0.629 / 0.784 / 0.909 / 1.018 are the
  z + dz/2 means of the columns), i.e. the Z_LOW reading. The z column is
  therefore copied unchanged, and the files must be read with
  photoz_zmid_convention = 0 to reproduce lighthouse. Reading them with 1
  shifts every bin down by dz/2 = 0.005.

CLEANING: a few lens entries are negative at the 1e-11 level (round-off of
the upstream resampling); they are set to zero. Nothing else is changed
(the long low-amplitude tails up to z = 2.99 are kept).

Usage:
  python make_des_y6_nz.py [--lens-in ...] [--source-in ...] [--outdir ...]
"""
import argparse
import os
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIGHTHOUSE_Y6 = ("/Users/vivianmiranda/data/COCOA/september2026/test/"
                 "lighthouse/analysis/des_y6_code_comparison")


def read_nz(path, ntomo):
  table = np.loadtxt(path)
  if table.shape[1] != ntomo + 1:
    raise SystemExit(f"{path}: expected {ntomo + 1} columns, "
                     f"found {table.shape[1]}")
  z = table[:, 0]
  dz = np.diff(z)
  if not (np.all(dz > 0) and np.allclose(dz, dz[0], rtol=1e-8, atol=1e-12)):
    raise SystemExit(f"{path}: z column is not uniform and increasing")
  return table


def clean(table):
  """Zero the negative round-off entries; return the table and a report."""
  nz = table[:, 1:]
  nneg = int(np.sum(nz < 0))
  most_negative = float(nz.min()) if nneg > 0 else 0.0
  cleaned = table.copy()
  cleaned[:, 1:] = np.clip(nz, 0.0, None)
  return cleaned, nneg, most_negative


def bin_means(table, zmid_convention):
  """Mean redshift of each bin under the chosen reading of the z column."""
  z = table[:, 0]
  dz = z[1] - z[0]
  node = z if 1 == zmid_convention else z + 0.5 * dz
  nz = table[:, 1:]
  return (nz * node[:, None]).sum(axis=0) / nz.sum(axis=0)


def write_nz(path, table, header_lines):
  ntomo = table.shape[1] - 1
  with open(path, "w") as f:
    for line in header_lines:
      f.write("# " + line + "\n")
    for row in table:
      f.write("%.6f" % row[0])
      for k in range(ntomo):
        f.write(" %.10e" % row[k + 1])
      f.write("\n")


def main():
  parser = argparse.ArgumentParser(description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--lens-in", default=os.path.join(LIGHTHOUSE_Y6, "lens.nz"))
  parser.add_argument("--source-in", default=os.path.join(LIGHTHOUSE_Y6, "source.nz"))
  parser.add_argument("--outdir", default=os.path.join(PROJECT_DIR, "data"))
  args = parser.parse_args()

  samples = [
    ("lens (MagLim)", args.lens_in, 6, "des_y6_maglim.nz"),
    ("source", args.source_in, 4, "des_y6_source.nz"),
  ]
  for name, path_in, ntomo, fname_out in samples:
    table, nneg, most_negative = clean(read_nz(path_in, ntomo))
    means_low = bin_means(table, 0)
    means_mid = bin_means(table, 1)
    header = [
      f"DES Y6-like {name} n(z), {ntomo} tomographic bins "
      "(des_cluster synthetic analysis, arXiv 2503.13631)",
      f"source: {path_in}",
      "columns: z, then n(z) of bin 1 .. %d (unnormalized)" % ntomo,
      "z column = Z_LOW (left bin edges): read with photoz_zmid_convention"
      " = 0 (values at z + dz/2), the lighthouse reading",
      "negative round-off entries set to zero: %d (most negative %.2e)"
      % (nneg, most_negative),
      "mean z per bin (Z_LOW reading): "
      + " ".join("%.4f" % m for m in means_low),
    ]
    path_out = os.path.join(args.outdir, fname_out)
    write_nz(path_out, table, header)
    print(f"{name}: wrote {path_out} ({table.shape[0]} rows, {ntomo} bins)")
    print(f"  negative entries zeroed: {nneg} (most negative {most_negative:.2e})")
    print("  mean z, Z_LOW reading (convention 0):",
          " ".join("%.4f" % m for m in means_low))
    print("  mean z, Z_MID reading (convention 1):",
          " ".join("%.4f" % m for m in means_mid))


if __name__ == "__main__":
  main()
