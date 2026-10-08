#!/usr/bin/env python
"""Build the scale-cut masks of the des_cluster likelihoods 4x2pt + N and 6x2pt + N.

A mask has one entry per point of the joint data vector (the observables
of the likelihood, in the fixed order below): 1 keeps the point in the
chi2, 0 removes it. The script writes the masks of CL+GC (= 4x2pt + N)
and CL+3x2pt (= 6x2pt + N) of arXiv 2503.13631, one "index value" line
per entry, the value printed as 0.0 or 1.0. The options are parsed by
argparse (the Python standard-library parser of command-line options),
and --help prints this text. lighthouse, quoted below, is the repository
of the original CosmoLike cluster code; its configuration files hold the
cut tables reproduced here.

DATA-VECTOR LAYOUT (the table of the project README; one mask entry per
data point, full length, blocks in this order):
  ss  xi+ then xi-, each [source pair (i <= j, i outer)][theta]
  gs  [(lens l, source s) pair, lens-major, all Nl x Ns pairs][theta]
      (cosmolike's galaxy-galaxy lensing pairs: test_zoverlap admits every
       pair unless init_ggl_exclude lists some; des_cluster lists none)
  gg  [lens bin, auto only][theta]
  cg  [(cluster bin zc, lens bin cg_lens_bins[zc]) pair][lambda][theta]
  N   [cluster bin zc][lambda]
  cc  [cluster bin zc][lambda1 <= lambda2 (lambda1 outer)][theta]
  cs  [(cluster bin zc, source bin zs) pair, cluster-major][lambda][theta]
The two masks share this layout; CL+GC masks ss and gs entirely.
Legend: ss = cosmic shear, gs = galaxy-galaxy lensing gamma_t, gg = galaxy
clustering w_gg, cg = cluster x galaxy clustering w_cg, N = cluster counts,
cc = cluster clustering w_cc, cs = cluster lensing; zc = cluster redshift
(z_lambda) bin; lambda = richness bin (the redMaPPer richness, the number
of red-sequence member galaxies of a cluster, is the mass proxy).

SCALE CUTS (arXiv 2503.13631 Sec. III: "defined at the mean redshifts of
each tomographic bin, assuming a fiducial cosmology", Table I):
  a 2pt point survives when its angle theta exceeds R / chi(zbar), with
  chi the comoving distance in Mpc/h at the Table I cosmology and
    cs (Sigma = Y gamma_t)  R =  2 Mpc/h at the cluster bin zbar
    cg (w_cg)               R =  8 Mpc/h at the cluster bin zbar
                                 (--cg-zbar lens: at the lens bin mean)
    cc (w_cc)               R = 16 Mpc/h at the cluster bin zbar
    gg (w_gg)               R =  8 Mpc/h at the lens bin mean
    gs (gamma_t)            R =  6 Mpc/h at the lens bin mean (DES Y3/Y6
                                 3x2pt with point-mass marginalization)
    ss (xi+-)               tabulated theta_min per source pair (--ss-cuts):
                                 y6 = lighthouse dataY6.yaml "8_6_0.5" table
                                 (baryon-contamination cuts, 4 Y6-like source
                                 bins); y3 = the DES Y3 baseline table
                                 (lighthouse dataY3.yaml; reproduces the
                                 xi+- part of data/3x2pt_baseline.mask,
                                 166 + 61 points); none = keep all.
  The counts N have no angle: every N entry is kept.
  theta of a bin = the area-weighted center (2/3)(tmax^3 - tmin^3) /
  (tmax^2 - tmin^2) of generic_interface.cpp and lighthouse
  (--theta-rule center); --theta-rule lower instead requires the whole bin
  (its lower edge) above R/chi.
  Cluster-bin zbar: --cluster-zbar mean (default) = mean true redshift of
  dV/dz <phi_i|z> (the Y1 eq 15 kernel, from --cluster-nz); midpoint =
  (zlo + zhi)/2 (zmid_cluster of redshift_spline_cluster.c: the zbar of
  the selection-bias factor in the C code and in lighthouse).
  Lens / source bin means: mean of the n(z) file under
  --nz-zmid-convention (0 = Z_LOW, values at z + dz/2, as the likelihood).
  The distances chi come from make_cluster_zdist.py (flat, Omega_m = 0.3).
Extra cluster-lensing rules:
  - the last theta bin of every cs block is always masked. cs holds the Y
    statistic of Park, Rozo & Krause (2021, arXiv:2004.07504),
    Y(R) = Sigma(R) - Sigma(R_max), computed as Sigma = T gamma_t with
    T = 2S + SD on the theta bins (S a trapezoid integral up to R_max, D a
    derivative in ln theta). Its last row is Y(R_max) = 0 by construction:
    a zero-variance entry, which the likelihood (IPCluster::set_mask) masks
    as well;
  - --cs-front-rule (default on): a (zc, zs) pair is masked entirely when
    the cluster bin's upper edge zhi_c >= mean z of source bin zs (DES Y1
    rule, applied by the lighthouse mask generator make_mask_4x2ptN in
    python/run_4x2ptN_wrapper.py).
Optional w_cc cuts (not in the Y6 paper text; off by default):
  - --wcc-theta-max: upper angular limit of w_cc (lighthouse Y3 used 180');
  - --wcc-min-pairs P: drop w_cc points whose expected pair count is below
    P (DES Y1 used 100). The expected count of a theta bin is
    DD = N_A N_B Omega_bin/Omega_s for two richness bins A != B and
    N_A (N_A - 1)/2 Omega_bin/Omega_s for A = B, with N_A the observed count
    of the bin (--counts-file, y3_redmapper_counts.txt), Omega_bin the solid
    angle of the annulus of the theta bin and Omega_s the area (--area).

Usage (from the cocoa/Cocoa folder; the defaults write
data/des_cluster_y6_4x2ptN.mask and data/des_cluster_y6_6x2ptN.mask):
  python ./projects/des_cluster/scripts/make_cluster_mask.py
Y3-binning check against the Y3 paper counts (MagLim 4 bins):
  python ./projects/des_cluster/scripts/make_cluster_mask.py \\
      --lens-nz <Y3 MagLim nz> --lens-ntomo 4 --source-nz <Y3 source nz> \\
      --ss-cuts y3 --out-prefix <tmp>/y3 --compare-y3
"""
import argparse
import os
import sys
import numpy as np

# make_cluster_zdist.py sits in this script's folder: putting that folder
# first on the module search path (sys.path) lets the import below find
# it from any working directory. The import after code is deliberate, and
# "noqa: E402" tells the flake8 style checker so.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_cluster_zdist import comoving_distance, hubble_ratio  # noqa: E402

# The project folder (the parent of scripts/) and its data folder, found
# from this script's own path (__file__), so the default file names do
# not depend on the working directory.
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
# one arcminute in radians
ARCMIN = np.pi/180.0/60.0

# xi+- minimum angles (arcmin) per source pair (i <= j, i outer), compared
# with the bin centers: ten pairs of the four source bins, in the order
# (1,1), (1,2), (1,3), (1,4), (2,2), ..., (4,4) of the lighthouse tables.
# The tables' upper limits (250' in the Y6 table, 999' in the Y3 table)
# are not below theta_max = 250', so they remove nothing and are not kept.
XI_CUTS = {
  # lighthouse analysis/des_y6_code_comparison/dataY6.yaml, block "8_6_0.5"
  "y6": {
    "xip": [2.9697, 4.7066, 7.4595, 7.4595, 4.7066, 9.3910, 9.3910,
            18.7374, 29.6968, 18.7374],
    "xim": [23.5890, 47.0663, 74.5950, 74.5950, 47.0663, 118.2251,
            118.2251, 118.2251, 148.8366, 118.2251],
  },
  # lighthouse analysis/yamlfiles/dataY3.yaml (DES Y3 baseline)
  "y3": {
    "xip": [2.475, 6.21691892, 6.21691892, 4.93827423, 6.21691892,
            6.21691892, 6.21691892, 6.21691892, 6.21691892, 4.93827423],
    "xim": [24.75, 62.16918918, 62.16918918, 49.3827423, 62.16918918,
            78.26637209, 78.26637209, 78.26637209, 78.26637209,
            62.16918918],
  },
}

# post-cut counts of the DES Y3 CL+GC analysis (arXiv 2503.13632 Sec. III),
# printed next to the CL+GC block counts by --compare-y3
Y3_PAPER_COUNTS = {"N": 12, "cs": 404, "cc": 149, "cg": 124, "gg": 31}


def parse_list(text, tp=float):
  """Split a comma- or space-separated option string into a list of numbers.

  Arguments:
    text = the option string, e.g. "0.2,0.4,0.55,0.65" or "0 1 2".
    tp = the conversion applied to every item: float (default) or int.

  Returns:
    list of tp values in the order of the string.

  Raises:
    ValueError when an item does not convert (e.g. "0.2;0.4").
  """
  # commas become spaces, split() cuts at every run of whitespace, and the
  # list comprehension converts each piece with tp
  return [tp(x) for x in text.replace(",", " ").split()]


def nz_means(path, ntomo, zmid_convention):
  """Mean redshift of each tomographic bin of a galaxy n(z) file.

  Under the Z_LOW convention (0, the photoz_zmid_convention of the
  likelihoods) the z column holds left bin edges and each value belongs
  to the cell center z + dz/2; under Z_MID (1) it belongs to z itself.
  Negative entries are clipped to zero before the mean.

  Arguments:
    path = n(z) text file: z, then one column per bin ('#' lines are
      skipped by numpy.loadtxt); the z grid must be uniform.
    ntomo = number of bins to read: columns 1 .. ntomo.
    zmid_convention = 0 (Z_LOW, values at z + dz/2) or 1 (Z_MID, at z).

  Returns:
    numpy array (ntomo,) of mean redshifts.

  Raises:
    SystemExit when the file has fewer than ntomo bin columns.
  """
  table = np.loadtxt(path)
  if table.shape[1] < ntomo + 1:
    raise SystemExit(f"{path}: {table.shape[1] - 1} bins < ntomo = {ntomo}")
  z = table[:, 0]
  # node positions of the values; dz is read from the first two rows
  node = z if 1 == zmid_convention else z + 0.5*(z[1] - z[0])
  # clip with no upper bound (None): negative round-off becomes 0
  nz = np.clip(table[:, 1:ntomo + 1], 0.0, None)
  # node[:, None] is a column (number of z nodes, 1) that numpy
  # broadcasting repeats across the bins; the sums over axis 0 (the z
  # nodes) give one n(z)-weighted mean per bin
  return (nz*node[:, None]).sum(axis=0)/nz.sum(axis=0)


def cluster_zbar(args, zedges):
  """The redshift zbar of each cluster bin at which its scale cuts are set.

  "midpoint": (zlo + zhi)/2 of the z_lambda edges. "mean" (default): the
  mean true redshift with the weight dV/dz <phi_i|z>, the Y1 eq 15 radial
  kernel of the volume-only cluster two-point functions, read from the
  selection-kernel table of make_cluster_zdist.py (values sampled at z,
  no half-cell offset; a uniform grid, so dz cancels in the mean).

  Arguments:
    args = the parsed options; uses args.cluster_zbar ("mean" or
      "midpoint") and args.cluster_nz (the kernel table).
    zedges = the cluster z_lambda bin edges, list of ncl + 1 floats.

  Returns:
    numpy array (ncl,) of zbar values.

  Raises:
    SystemExit when the kernel table does not have one column per bin.
  """
  ncl = len(zedges) - 1
  if "midpoint" == args.cluster_zbar:
    return np.array([0.5*(zedges[i] + zedges[i+1]) for i in range(ncl)])
  table = np.loadtxt(args.cluster_nz)
  if table.shape[1] != ncl + 1:
    raise SystemExit(f"{args.cluster_nz}: {table.shape[1] - 1} columns, "
                     f"expected {ncl} cluster bins")
  z = table[:, 0]
  # dV/dz per steradian is (c/H0) chi^2/E(z); the constant c/H0 cancels
  # in the weighted mean
  dvdz = comoving_distance(z)**2/hubble_ratio(z)
  return np.array([np.sum(z*dvdz*table[:, i+1])/np.sum(dvdz*table[:, i+1])
                   for i in range(ncl)])


def read_counts(path, nz_cluster, nrichness):
  """Observed cluster counts N per (z_lambda bin, richness bin).

  Reads column 4 (N) of the table of make_y3_redmapper_counts.py, whose
  rows run over the richness bins inside each z bin (z bin outer), so a
  row-major reshape restores the (z, richness) grid. Only the optional
  w_cc pair-count cut (--wcc-min-pairs) uses these counts.

  Arguments:
    path = the counts table (default data/y3_redmapper_counts.txt).
    nz_cluster = number of cluster redshift bins.
    nrichness = number of richness bins.

  Returns:
    numpy array (nz_cluster, nrichness) of counts.

  Raises:
    SystemExit when the table does not have nz_cluster x nrichness rows.
  """
  table = np.loadtxt(path)
  if table.shape[0] != nz_cluster*nrichness:
    raise SystemExit(f"{path}: {table.shape[0]} rows, expected "
                     f"{nz_cluster*nrichness}")
  return table[:, 4].reshape(nz_cluster, nrichness)


class ThetaBins:
  """The angular bins of the two-point blocks: edges, centers, solid angles.

  The bins are uniform in ln theta between tmin and tmax, as in cosmolike
  (compute_binning_real_space of generic_interface.cpp), and each bin is
  represented by its area-weighted center (2/3)(hi^3 - lo^3)/(hi^2 - lo^2),
  the angle the C code assigns to it. An instance holds lo and hi (the bin
  edges), center (the area-weighted centers), all numpy arrays (ntheta,)
  in arcmin, and n = ntheta.
  """

  def __init__(self, ntheta, tmin, tmax):
    """Build the ntheta bins between tmin and tmax.

    Arguments:
      ntheta = number of angular bins (20 in des_cluster).
      tmin = lower edge of the first bin, arcmin (2.5 in des_cluster).
      tmax = upper edge of the last bin, arcmin (250 in des_cluster).
    """
    edges = np.exp(np.linspace(np.log(tmin), np.log(tmax), ntheta + 1))
    self.lo = edges[:-1]
    self.hi = edges[1:]
    self.center = (2.0/3.0)*(self.hi**3 - self.lo**3)/(self.hi**2 - self.lo**2)
    self.n = ntheta

  def keep_above(self, theta_min, rule):
    """1 for the bins that pass theta > theta_min (arcmin).

    Rule "center" compares the area-weighted center (center > theta_min);
    rule "lower" requires the whole bin above the cut (lo >= theta_min).

    Arguments:
      theta_min = the smallest angle kept, arcmin.
      rule = "center" or "lower"; any other value acts as "center".

    Returns:
      numpy int array (n,) of 0 and 1: astype(int) turns the booleans of
      the comparison into integers.
    """
    if "lower" == rule:
      return (self.lo >= theta_min).astype(int)
    return (self.center > theta_min).astype(int)

  def solid_angle_deg2(self):
    """Solid angle of each annulus lo < theta < hi around a point, in deg^2.

    2 pi (cos lo - cos hi) steradians (exact on the sphere), times
    (180/pi)^2 square degrees per steradian. The w_cc pair-count cut
    multiplies it by the counts and divides by the survey area.

    Returns:
      numpy array (n,) in deg^2.
    """
    return 2*np.pi*(np.cos(self.lo*ARCMIN) - np.cos(self.hi*ARCMIN)) * \
           (180.0/np.pi)**2


def theta_min_arcmin(r_mpch, z):
  """The angle (arcmin) that a comoving separation spans at redshift z.

  theta = R/chi(z) in radians (flat: the transverse comoving distance is
  chi), converted to arcmin. A point of a two-point block survives the
  scale cut when its angle exceeds this value.

  Arguments:
    r_mpch = comoving separation R, Mpc/h (e.g. 2 for cluster lensing).
    z = the redshift zbar of the bin, a float.

  Returns:
    float, arcmin.
  """
  # comoving_distance takes an array: z goes in as a one-element array
  # and element [0] of the result comes out
  return r_mpch/comoving_distance(np.array([z]))[0]/ARCMIN


def build_blocks(args, geo):
  """Return {block name: 1d int array} of the CL+3x2pt mask (full cuts).

  Each array has the length and the order of its block in the joint data
  vector (module docstring), with 1 for a kept point. The cuts are those
  of the module docstring: R/chi(zbar) per probe and bin, the ss tables,
  the zero last theta bin of every cs row, the cs source-in-front rule
  and the optional w_cc cuts. The counts N are always kept.

  Arguments:
    args = the parsed options (the radii --rmin-*, --theta-rule,
      --cs-theta-rule, --ss-cuts, --cg-zbar, --cs-front-rule,
      --wcc-theta-max, --wcc-min-pairs, --area).
    geo = the geometry dict built in main(): "theta" (ThetaBins); "nl",
      "ns", "ncl", "nlam" (numbers of lens, source, cluster redshift and
      richness bins); "zedges" (cluster z_lambda edges); "cg_lens" (the
      lens bin of each cluster bin in w_cg); the mean redshifts "zl",
      "zs", "zc" (numpy arrays); "counts" when --wcc-min-pairs > 0.

  Returns:
    dict {"ss", "gs", "gg", "cg", "N", "cc", "cs": numpy int array}.

  Raises:
    SystemExit when an ss cut table is requested for a number of source
    bins other than 4.
  """
  tb = geo["theta"]
  nt, ns, nl = tb.n, geo["ns"], geo["nl"]
  ncl, nlam = geo["ncl"], geo["nlam"]
  # "or" returns its first operand unless that is None: the cs blocks use
  # --cs-theta-rule when given and --theta-rule otherwise
  cs_rule = args.cs_theta_rule or args.theta_rule
  blocks = {}

  # ss: xi+ then xi-; ns (ns + 1)/2 source pairs i <= j (// is integer
  # division). The tables are compared with the bin centers whatever
  # --theta-rule says.
  npair_ss = ns*(ns + 1)//2
  if "none" == args.ss_cuts:
    blocks["ss"] = np.ones(2*npair_ss*nt, dtype=int)
  else:
    if 4 != ns:
      raise SystemExit("the xi+- cut tables are for 4 source bins")
    parts = []
    for stat in ("xip", "xim"):
      for p in range(npair_ss):
        parts.append((tb.center > XI_CUTS[args.ss_cuts][stat][p]).astype(int))
    blocks["ss"] = np.concatenate(parts)

  # gs: lens-major, all pairs. The cut depends on the lens bin only, so
  # [keep]*ns appends ns references to the same array (one per source
  # bin); np.concatenate copies them into one new array.
  parts = []
  for l in range(nl):
    keep = tb.keep_above(theta_min_arcmin(args.rmin_gs, geo["zl"][l]),
                         args.theta_rule)
    parts += [keep]*ns
  blocks["gs"] = np.concatenate(parts)

  # gg: autos; the list comprehension gives one theta mask per lens bin
  blocks["gg"] = np.concatenate(
    [tb.keep_above(theta_min_arcmin(args.rmin_gg, geo["zl"][l]),
                   args.theta_rule) for l in range(nl)])

  # cg: [pair][lambda][theta]; enumerate yields (position, value) pairs:
  # the cluster bin zc and the lens bin zg paired with it
  parts = []
  for zc, zg in enumerate(geo["cg_lens"]):
    # conditional expression: the cluster zbar by default, the lens mean
    # with --cg-zbar lens
    zbar = geo["zc"][zc] if "cluster" == args.cg_zbar else geo["zl"][zg]
    keep = tb.keep_above(theta_min_arcmin(args.rmin_cg, zbar), args.theta_rule)
    parts += [keep]*nlam
  blocks["cg"] = np.concatenate(parts)

  # N: always used
  blocks["N"] = np.ones(ncl*nlam, dtype=int)

  # cc: [z][lambda1 <= lambda2][theta]
  parts = []
  omega_bin = tb.solid_angle_deg2()
  for zc in range(ncl):
    keep = tb.keep_above(theta_min_arcmin(args.rmin_cc, geo["zc"][zc]),
                         args.theta_rule)
    if args.wcc_theta_max is not None:
      # the product with a boolean array zeros the bins at or above the
      # upper limit
      keep = keep*(tb.center < args.wcc_theta_max)
    for a in range(nlam):
      for b in range(a, nlam):
        # a copy: the pair-count cut of one richness pair must not change
        # the starting cut of the next pair
        k = keep.copy()
        if args.wcc_min_pairs > 0:
          n_a, n_b = geo["counts"][zc, a], geo["counts"][zc, b]
          # distinct pairs: N_A (N_A - 1)/2 within one richness bin,
          # N_A N_B across two
          npairs = n_a*(n_a - 1)/2 if a == b else n_a*n_b
          # expected pairs per theta bin (omega_bin and --area both in
          # deg^2) compared with the threshold P
          k = k*(npairs*omega_bin/args.area >= args.wcc_min_pairs)
        parts.append(k.astype(int))
  blocks["cc"] = np.concatenate(parts)

  # cs: [(zc, zs) cluster-major][lambda][theta]
  parts = []
  for zc in range(ncl):
    keep = tb.keep_above(theta_min_arcmin(args.rmin_cs, geo["zc"][zc]),
                         cs_rule)
    # Y(R_max) = Sigma(R_max) - Sigma(R_max) = 0 in every cs row: a
    # zero-variance entry that the likelihood masks as well
    keep[-1] = 0  # zero last row of the Y transform
    for zs in range(ns):
      # source-in-front rule: the cluster bin's upper edge is not below
      # the mean redshift of the source bin
      in_front = geo["zedges"][zc + 1] >= geo["zs"][zs]
      k = np.zeros(nt, dtype=int) if (args.cs_front_rule and in_front) else keep
      parts += [k]*nlam
  blocks["cs"] = np.concatenate(parts)
  return blocks


def cl_gc_blocks(blocks, geo, gg_lens_bins):
  """CL+GC: no ss, no gs, w_gg only in the lens bins of the cluster pairs.

  The CL+GC (4x2pt + N) mask is the CL+3x2pt mask with cosmic shear and
  galaxy-galaxy lensing removed and w_gg kept only in the listed lens bins
  (by default the lens bins paired with the cluster bins in w_cg: 0, 1, 2,
  the MagLim bins 1-3). The cluster blocks are the same in both masks.

  Arguments:
    blocks = the CL+3x2pt blocks returned by build_blocks.
    geo = the geometry dict of main(); only geo["theta"].n is used.
    gg_lens_bins = lens bins (counted from 0) whose w_gg rows are kept.

  Returns:
    a new dict of block arrays; the input dict and its arrays are left
    unchanged.
  """
  # dict(blocks) is a new dict holding the same arrays; the lines below
  # replace three entries with new arrays and never write into the
  # shared ones
  out = dict(blocks)
  out["ss"] = np.zeros_like(blocks["ss"])
  out["gs"] = np.zeros_like(blocks["gs"])
  nt = geo["theta"].n
  # w_gg row l holds entries l*nt .. (l + 1)*nt - 1 (the slice end is
  # exclusive)
  gg = np.zeros_like(blocks["gg"])
  for l in gg_lens_bins:
    gg[l*nt:(l + 1)*nt] = blocks["gg"][l*nt:(l + 1)*nt]
  out["gg"] = gg
  return out


# The block order of the joint data vector (the README table, and
# compute_data_vector_cluster_starts of the C code): write_mask joins the
# blocks in this order.
ORDER = ["ss", "gs", "gg", "cg", "N", "cc", "cs"]


def write_mask(path, blocks):
  """Write one mask file: the blocks in ORDER, one "index value" line each.

  The value is printed as 0.0 or 1.0 (the "%d %.1f" format of the
  lighthouse mask files); the likelihood reads the second column.

  Arguments:
    path = output file; an existing file is overwritten.
    blocks = {block name: int array} with every name of ORDER.

  Returns:
    the number of entries written (2812 in the default binning).
  """
  # the list comprehension lists the block arrays in the joint order
  mask = np.concatenate([blocks[b] for b in ORDER])
  # the with block closes the file when it ends, also after an error;
  # enumerate yields (entry index, 0 or 1) pairs
  with open(path, "w") as f:
    for i, m in enumerate(mask):
      f.write("%d %.1f\n" % (i, m))
  return mask.size


def report(title, blocks, compare_y3):
  """Print, per block, its entry range, its size and its kept entries.

  Arguments:
    title = the heading line.
    blocks = {block name: int array} of one mask.
    compare_y3 = True to print the post-cut counts of the DES Y3 CL+GC
      analysis (Y3_PAPER_COUNTS) next to the blocks they cover.

  Returns:
    nothing (prints only).
  """
  print(title)
  # running index: each block starts where the previous one ended
  start = 0
  for b in ORDER:
    size = blocks[b].size
    line = (f"  {b:3s} entries {start:5d}-{start + size - 1:5d} "
            f"(size {size:4d}): unmasked {int(blocks[b].sum()):4d}")
    if compare_y3 and b in Y3_PAPER_COUNTS:
      line += f"   [Y3 paper: {Y3_PAPER_COUNTS[b]}]"
    print(line)
    start += size
  # generator inside sum: the kept entries of all blocks
  total = sum(int(blocks[b].sum()) for b in ORDER)
  print(f"  total entries {start}, unmasked {total}")


def main():
  """Parse the options, compute the cuts and write the two masks.

  Checks --cg-lens-bins against the cluster and lens binning, builds the
  geometry (angular bins, mean redshift of every bin, the observed counts
  when the w_cc pair cut is on), prints the cut angles of every probe and
  bin, then writes <out-prefix>_4x2ptN.mask (CL+GC) and
  <out-prefix>_6x2ptN.mask (CL+3x2pt) and prints a per-block report of
  each.

  Arguments:
    none; the options come from the command line through argparse.

  Returns:
    nothing.

  Raises:
    SystemExit when --cg-lens-bins does not name one existing lens bin
    per cluster bin, or when an input table has the wrong number of
    columns or rows (nz_means, cluster_zbar, read_counts).

  Side effects:
    writes the two mask files (an existing file is overwritten) and
    prints the report.
  """
  # __doc__ is the module docstring; RawDescriptionHelpFormatter makes
  # --help print it with its line breaks kept
  parser = argparse.ArgumentParser(description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--lens-nz", default=os.path.join(DATA_DIR, "des_y6_maglim.nz"))
  parser.add_argument("--lens-ntomo", type=int, default=6)
  parser.add_argument("--source-nz", default=os.path.join(DATA_DIR, "des_y6_source.nz"))
  parser.add_argument("--source-ntomo", type=int, default=4)
  parser.add_argument("--nz-zmid-convention", type=int, default=0, choices=[0, 1])
  parser.add_argument("--cluster-nz", default=os.path.join(DATA_DIR, "des_y6_cluster.nz"))
  parser.add_argument("--cluster-zbin-edges", default="0.2,0.4,0.55,0.65")
  parser.add_argument("--richness-edges", default="20,30,45,60,500")
  parser.add_argument("--cg-lens-bins", default="0,1,2")
  parser.add_argument("--gg-lens-bins-clgc", default=None,
                      help="lens bins kept in w_gg of CL+GC (default: cg lens bins)")
  parser.add_argument("--n-theta", type=int, default=20)
  parser.add_argument("--theta-min-arcmin", type=float, default=2.5)
  parser.add_argument("--theta-max-arcmin", type=float, default=250.0)
  parser.add_argument("--rmin-cs", type=float, default=2.0)
  parser.add_argument("--rmin-cg", type=float, default=8.0)
  parser.add_argument("--rmin-gg", type=float, default=8.0)
  parser.add_argument("--rmin-cc", type=float, default=16.0)
  parser.add_argument("--rmin-gs", type=float, default=6.0)
  parser.add_argument("--cluster-zbar", default="mean", choices=["mean", "midpoint"])
  parser.add_argument("--cg-zbar", default="cluster", choices=["cluster", "lens"])
  parser.add_argument("--theta-rule", default="center", choices=["center", "lower"])
  parser.add_argument("--cs-theta-rule", default=None, choices=["center", "lower"])
  # two flags write one destination: the rule is on by default, and
  # --no-cs-front-rule (action="store_false") switches it off
  parser.add_argument("--cs-front-rule", dest="cs_front_rule", action="store_true",
                      default=True)
  parser.add_argument("--no-cs-front-rule", dest="cs_front_rule", action="store_false")
  parser.add_argument("--ss-cuts", default="y6", choices=["y6", "y3", "none"])
  parser.add_argument("--wcc-theta-max", type=float, default=None)
  parser.add_argument("--wcc-min-pairs", type=float, default=0.0)
  parser.add_argument("--counts-file", default=os.path.join(DATA_DIR, "y3_redmapper_counts.txt"))
  parser.add_argument("--area", type=float, default=4143.0,
                      help="deg^2, for the expected w_cc pair counts")
  parser.add_argument("--out-prefix", default=os.path.join(DATA_DIR, "des_cluster_y6"))
  parser.add_argument("--compare-y3", action="store_true")
  args = parser.parse_args()

  zedges = parse_list(args.cluster_zbin_edges)
  ledges = parse_list(args.richness_edges)
  cg_lens = parse_list(args.cg_lens_bins, int)
  # tuple assignment: the numbers of cluster z bins and richness bins
  ncl, nlam = len(zedges) - 1, len(ledges) - 1
  if len(cg_lens) != ncl:
    raise SystemExit("--cg-lens-bins needs one lens bin per cluster bin")
  if max(cg_lens) >= args.lens_ntomo:
    raise SystemExit("--cg-lens-bins refers to a missing lens bin")
  # conditional expression: the cg lens bins unless --gg-lens-bins-clgc
  # is given
  gg_clgc = cg_lens if args.gg_lens_bins_clgc is None else \
            parse_list(args.gg_lens_bins_clgc, int)

  # the geometry shared by build_blocks and cl_gc_blocks (keys listed in
  # the build_blocks docstring)
  geo = {
    "theta": ThetaBins(args.n_theta, args.theta_min_arcmin, args.theta_max_arcmin),
    "nl": args.lens_ntomo, "ns": args.source_ntomo,
    "ncl": ncl, "nlam": nlam, "zedges": zedges, "cg_lens": cg_lens,
    "zl": nz_means(args.lens_nz, args.lens_ntomo, args.nz_zmid_convention),
    "zs": nz_means(args.source_nz, args.source_ntomo, args.nz_zmid_convention),
    "zc": cluster_zbar(args, zedges),
  }
  if args.wcc_min_pairs > 0:
    geo["counts"] = read_counts(args.counts_file, ncl, nlam)

  print("fiducial cosmology: flat, Omega_m = 0.3 (arXiv 2503.13631 Table I)")
  print("lens mean z   :", " ".join("%.4f" % z for z in geo["zl"]))
  print("source mean z :", " ".join("%.4f" % z for z in geo["zs"]))
  print(f"cluster zbar ({args.cluster_zbar}):",
        " ".join("%.4f" % z for z in geo["zc"]))
  # one (probe, radius R, redshifts zbar) tuple per two-point cut;
  # geo["zl"][cg_lens] indexes the array with a list, giving the means of
  # the cg lens bins
  for name, r, zs in [("cs", args.rmin_cs, geo["zc"]), ("cg", args.rmin_cg,
                      geo["zc"] if "cluster" == args.cg_zbar else
                      geo["zl"][cg_lens]), ("cc", args.rmin_cc, geo["zc"]),
                      ("gg", args.rmin_gg, geo["zl"]), ("gs", args.rmin_gs, geo["zl"])]:
    print(f"theta_min {name} (R = {r:g} Mpc/h) [arcmin]:",
          " ".join("%.3f" % theta_min_arcmin(r, z) for z in zs))
  if args.cs_front_rule:
    # comprehension over both bin loops with a condition: the (cluster
    # bin, source bin) pairs, counted from 1, that the rule removes
    pairs = [(zc + 1, zs + 1) for zc in range(ncl) for zs in range(args.source_ntomo)
             if zedges[zc + 1] >= geo["zs"][zs]]
    print("cs pairs masked by the source-in-front rule (zc, zs):", pairs)

  full = build_blocks(args, geo)
  clgc = cl_gc_blocks(full, geo, gg_clgc)
  path_clgc = args.out_prefix + "_4x2ptN.mask"
  path_full = args.out_prefix + "_6x2ptN.mask"
  n1 = write_mask(path_clgc, clgc)
  n2 = write_mask(path_full, full)
  # the theta rule for the report, with the cs rule appended when given
  rule = args.theta_rule + ("" if args.cs_theta_rule is None
                            else f", cs {args.cs_theta_rule}")
  report(f"\nCL+GC (4x2pt+N) -> {path_clgc} ({n1} entries; theta rule {rule})",
         clgc, args.compare_y3)
  report(f"\nCL+3x2pt (6x2pt+N) -> {path_full} ({n2} entries; ss cuts "
         f"{args.ss_cuts})", full, False)


if __name__ == "__main__":
  main()
