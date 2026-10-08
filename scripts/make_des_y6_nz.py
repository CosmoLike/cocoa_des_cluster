#!/usr/bin/env python
"""Write the DES Y6-like lens and source n(z) files of des_cluster in cosmolike format.

The lens sample is MagLim (6 tomographic bins), the source sample has 4
bins; the files are data/des_y6_maglim.nz and data/des_y6_source.nz,
named by the nz_lens_file and nz_source_file keys of
data/des_cluster_y6.dataset. The options are parsed by argparse (the
Python standard-library parser of command-line options), and --help
prints this text.

OUTPUT FORMAT (the format of data/des_y3_lens.nz, read by read_nz_sample in
generic_interface.cpp): one row per redshift node, column 0 = z, column k =
n(z) of tomographic bin k (unnormalized; the reader normalizes each bin).
Lines starting with '#' are comments (read_table skips them).

INPUT: the Y6 n(z) of the DES Y6 code comparison kept in lighthouse (the
repository of the original CosmoLike cluster code),
  lighthouse/analysis/des_y6_code_comparison/{lens,source}.nz
(300 rows, z = 0.00, 0.01, ..., 2.99; after the z column, 6 lens columns
and 4 source columns).

REDSHIFT CONVENTION (Z_LOW vs Z_MID):
  The cosmolike reader has two readings of the z column, chosen by the
  likelihood key photoz_zmid_convention (Ntable.photoz_zmid_convention):
    0 = Z_LOW: the column holds left bin edges and each value belongs to the
        cell center z + dz/2 (the value set in every likelihood yaml of
        des_cluster);
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

Usage (from the cocoa/Cocoa folder; the default inputs are a local
lighthouse checkout, so on another machine pass both input paths):
  python ./projects/des_cluster/scripts/make_des_y6_nz.py \\
      --lens-in <lighthouse>/analysis/des_y6_code_comparison/lens.nz \\
      --source-in <lighthouse>/analysis/des_y6_code_comparison/source.nz \\
      [--outdir DIR]
"""
import argparse
import os
import numpy as np

# The project folder (the parent of scripts/), found from this script's
# own path (__file__), so the default output folder <project>/data does
# not depend on the working directory.
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Default input folder: the DES Y6 code-comparison folder of a lighthouse
# checkout on the machine where the shipped files were made (the files
# record it in their "# source:" header line). On another machine this
# path normally does not exist, and the user passes --lens-in and
# --source-in.
LIGHTHOUSE_Y6 = ("/Users/vivianmiranda/data/COCOA/september2026/test/"
                 "lighthouse/analysis/des_y6_code_comparison")


def read_nz(path, ntomo):
  """Read one n(z) table and check its shape and its z grid.

  Arguments:
    path = n(z) text file: z, then one column per bin ('#' lines are
      skipped by numpy.loadtxt).
    ntomo = the number of bin columns the file must have.

  Returns:
    numpy array (number of z nodes, ntomo + 1), the file as read.

  Raises:
    SystemExit when the file does not have ntomo + 1 columns, or when its
    z column is not increasing with a uniform step (relative tolerance
    1e-8): both conventions of the module docstring assume one step dz.
  """
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
  """Zero the negative round-off entries; return the table and a report.

  Arguments:
    table = n(z) table (z column, then one column per bin); not modified.

  Returns:
    (cleaned, nneg, most_negative): a copy of table with every negative
    n(z) entry set to 0, the number of entries that were negative, and
    the most negative value (0.0 when there was none).
  """
  # table[:, 1:] is a view of the n(z) columns (no copy); the cleaned
  # values go into an explicit copy, so the input stays as read
  nz = table[:, 1:]
  nneg = int(np.sum(nz < 0))
  most_negative = float(nz.min()) if nneg > 0 else 0.0
  cleaned = table.copy()
  cleaned[:, 1:] = np.clip(nz, 0.0, None)
  return cleaned, nneg, most_negative


def bin_means(table, zmid_convention):
  """Mean redshift of each bin under the chosen reading of the z column.

  Arguments:
    table = n(z) table with a uniform z column, then one column per bin.
    zmid_convention = 0 (Z_LOW: values at z + dz/2) or 1 (Z_MID: values
      at z), the photoz_zmid_convention of the likelihood.

  Returns:
    numpy array (number of bins,) of n(z)-weighted mean redshifts.
  """
  z = table[:, 0]
  dz = z[1] - z[0]
  node = z if 1 == zmid_convention else z + 0.5 * dz
  nz = table[:, 1:]
  # node[:, None] is a column (number of z nodes, 1) that numpy
  # broadcasting repeats across the bins; the sums over axis 0 (the z
  # nodes) give one weighted mean per bin
  return (nz * node[:, None]).sum(axis=0) / nz.sum(axis=0)


def write_nz(path, table, header_lines):
  """Write an n(z) table in the cosmolike text format.

  Each header line is written after "# "; then one row per z node, z with
  six decimals and every n(z) value with 11 significant digits.

  Arguments:
    path = output file; an existing file is overwritten.
    table = n(z) table (z column, then one column per bin).
    header_lines = list of strings, the '#' comment lines.

  Returns:
    nothing.
  """
  ntomo = table.shape[1] - 1
  # the with block closes the file when it ends, also after an error
  with open(path, "w") as f:
    for line in header_lines:
      f.write("# " + line + "\n")
    for row in table:
      f.write("%.6f" % row[0])
      for k in range(ntomo):
        f.write(" %.10e" % row[k + 1])
      f.write("\n")


def main():
  """Convert the lens and source n(z) files and print their bin means.

  For each sample: read and check the input, zero the negative entries,
  compute the mean redshift of every bin under both readings of the z
  column, write the output with a '#' header recording the input path,
  the convention and the means, and print a summary.

  Arguments:
    none; the options come from the command line through argparse.

  Returns:
    nothing.

  Raises:
    SystemExit from read_nz when an input has the wrong shape or z grid
    (a missing input file raises the error of numpy.loadtxt).

  Side effects:
    writes des_y6_maglim.nz and des_y6_source.nz in --outdir (default
    the project's data/ folder), overwriting existing files.
  """
  # __doc__ is the module docstring; RawDescriptionHelpFormatter makes
  # --help print it with its line breaks kept
  parser = argparse.ArgumentParser(description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--lens-in", default=os.path.join(LIGHTHOUSE_Y6, "lens.nz"))
  parser.add_argument("--source-in", default=os.path.join(LIGHTHOUSE_Y6, "source.nz"))
  parser.add_argument("--outdir", default=os.path.join(PROJECT_DIR, "data"))
  args = parser.parse_args()

  # one (label, input file, number of bins, output file name) tuple per
  # sample; the for statement unpacks each tuple into four names
  samples = [
    ("lens (MagLim)", args.lens_in, 6, "des_y6_maglim.nz"),
    ("source", args.source_in, 4, "des_y6_source.nz"),
  ]
  for name, path_in, ntomo, fname_out in samples:
    # clean returns three values, unpacked into three names
    table, nneg, most_negative = clean(read_nz(path_in, ntomo))
    means_low = bin_means(table, 0)
    means_mid = bin_means(table, 1)
    # each entry becomes one '#' line of the file; adjacent string
    # literals are joined into one string before the % formatting applies
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
