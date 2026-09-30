#!/usr/bin/env python
"""
Build the scale-cut masks of the DES cluster analyses (des_cluster project):
CL+GC (= 4x2pt + N) and CL+3x2pt (= 6x2pt + N) of arXiv 2503.13631.

DATA-VECTOR LAYOUT (PORT_PLAN.md; one mask entry per data point, full
length, blocks in this order):
  ss  xi+ then xi-, each [source pair (i <= j, i outer)][theta]
  gs  [(lens l, source s) pair, lens-major, all Nl x Ns pairs][theta]
      (the core's ggl pairs: test_zoverlap admits every pair unless
       init_ggl_exclude lists some; des_cluster lists none)
  gg  [lens bin, auto only][theta]
  cg  [(cluster bin zc, lens bin cg_lens_bins[zc]) pair][lambda][theta]
  N   [cluster bin zc][lambda]
  cc  [cluster bin zc][lambda1 <= lambda2 (lambda1 outer)][theta]
  cs  [(cluster bin zc, source bin zs) pair, cluster-major][lambda][theta]
The two masks share this layout; CL+GC masks ss and gs entirely.

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
  theta of a bin = the area-weighted center (2/3)(tmax^3 - tmin^3) /
  (tmax^2 - tmin^2) of generic_interface.cpp and lighthouse
  (--theta-rule center); --theta-rule lower instead requires the whole bin
  (its lower edge) above R/chi.
  Cluster-bin zbar: --cluster-zbar mean (default) = mean true redshift of
  dV/dz <phi_i|z> (the Y1 eq 15 kernel, from --cluster-nz); midpoint =
  (zlo + zhi)/2 (the zmid_cluster of the C contract and lighthouse).
  Lens / source bin means: mean of the n(z) file under
  --nz-zmid-convention (0 = Z_LOW, values at z + dz/2, as the likelihood).
Extra cluster-lensing rules:
  - the last theta bin of every cs block is always masked: the Y transform
    (Park et al. 2021, T = 2S + SD) has a zero last row (Y(R_max) = 0);
  - --cs-front-rule (default on): a (zc, zs) pair is masked entirely when
    the cluster bin's upper edge zhi_c >= mean z of source bin zs (DES Y1
    rule, applied by the lighthouse mask generator).
Optional w_cc cuts (not in the Y6 paper text; off by default):
  - --wcc-theta-max: upper angular limit of w_cc (lighthouse Y3 used 180');
  - --wcc-min-pairs P: drop w_cc points whose expected pair count
    DD = N_A N_B Omega_bin/Omega_s (N_A (N_A - 1)/2 ... for A = B) is below
    P (DES Y1 used 100), with N_A from --counts-file (y3_redmapper_counts.txt).

Usage (Y6-like defaults write data/des_cluster_y6_{4x2ptN,6x2ptN}.mask):
  python make_cluster_mask.py
Y3-binning check against the Y3 paper counts (MagLim 4 bins):
  python make_cluster_mask.py --lens-nz <Y3 MagLim nz> --lens-ntomo 4 \
      --source-nz <Y3 source nz> --ss-cuts y3 --out-prefix <tmp>/y3 --compare-y3
"""
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_cluster_zdist import comoving_distance, hubble_ratio  # noqa: E402

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
ARCMIN = np.pi/180.0/60.0

# xi+- minimum angles (arcmin) per source pair (i <= j, i outer), compared
# with the bin centers. Upper limits are above theta_max (250').
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

# post-cut counts of the DES Y3 CL+GC analysis (arXiv 2503.13632 Sec. III)
Y3_PAPER_COUNTS = {"N": 12, "cs": 404, "cc": 149, "cg": 124, "gg": 31}


def parse_list(text, tp=float):
  return [tp(x) for x in text.replace(",", " ").split()]


def nz_means(path, ntomo, zmid_convention):
  table = np.loadtxt(path)
  if table.shape[1] < ntomo + 1:
    raise SystemExit(f"{path}: {table.shape[1] - 1} bins < ntomo = {ntomo}")
  z = table[:, 0]
  node = z if 1 == zmid_convention else z + 0.5*(z[1] - z[0])
  nz = np.clip(table[:, 1:ntomo + 1], 0.0, None)
  return (nz*node[:, None]).sum(axis=0)/nz.sum(axis=0)


def cluster_zbar(args, zedges):
  ncl = len(zedges) - 1
  if "midpoint" == args.cluster_zbar:
    return np.array([0.5*(zedges[i] + zedges[i+1]) for i in range(ncl)])
  table = np.loadtxt(args.cluster_nz)
  if table.shape[1] != ncl + 1:
    raise SystemExit(f"{args.cluster_nz}: {table.shape[1] - 1} columns, "
                     f"expected {ncl} cluster bins")
  z = table[:, 0]
  dvdz = comoving_distance(z)**2/hubble_ratio(z)
  return np.array([np.sum(z*dvdz*table[:, i+1])/np.sum(dvdz*table[:, i+1])
                   for i in range(ncl)])


def read_counts(path, nz_cluster, nrichness):
  table = np.loadtxt(path)
  if table.shape[0] != nz_cluster*nrichness:
    raise SystemExit(f"{path}: {table.shape[0]} rows, expected "
                     f"{nz_cluster*nrichness}")
  return table[:, 4].reshape(nz_cluster, nrichness)


class ThetaBins:
  def __init__(self, ntheta, tmin, tmax):
    edges = np.exp(np.linspace(np.log(tmin), np.log(tmax), ntheta + 1))
    self.lo = edges[:-1]
    self.hi = edges[1:]
    self.center = (2.0/3.0)*(self.hi**3 - self.lo**3)/(self.hi**2 - self.lo**2)
    self.n = ntheta

  def keep_above(self, theta_min, rule):
    """1 for the bins that pass theta > theta_min (arcmin)."""
    if "lower" == rule:
      return (self.lo >= theta_min).astype(int)
    return (self.center > theta_min).astype(int)

  def solid_angle_deg2(self):
    return 2*np.pi*(np.cos(self.lo*ARCMIN) - np.cos(self.hi*ARCMIN)) * \
           (180.0/np.pi)**2


def theta_min_arcmin(r_mpch, z):
  return r_mpch/comoving_distance(np.array([z]))[0]/ARCMIN


def build_blocks(args, geo):
  """Return {block name: 1d int array} of the CL+3x2pt mask (full cuts)."""
  tb = geo["theta"]
  nt, ns, nl = tb.n, geo["ns"], geo["nl"]
  ncl, nlam = geo["ncl"], geo["nlam"]
  cs_rule = args.cs_theta_rule or args.theta_rule
  blocks = {}

  # ss: xi+ then xi-
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

  # gs: lens-major, all pairs
  parts = []
  for l in range(nl):
    keep = tb.keep_above(theta_min_arcmin(args.rmin_gs, geo["zl"][l]),
                         args.theta_rule)
    parts += [keep]*ns
  blocks["gs"] = np.concatenate(parts)

  # gg: autos
  blocks["gg"] = np.concatenate(
    [tb.keep_above(theta_min_arcmin(args.rmin_gg, geo["zl"][l]),
                   args.theta_rule) for l in range(nl)])

  # cg: [pair][lambda][theta]
  parts = []
  for zc, zg in enumerate(geo["cg_lens"]):
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
      keep = keep*(tb.center < args.wcc_theta_max)
    for a in range(nlam):
      for b in range(a, nlam):
        k = keep.copy()
        if args.wcc_min_pairs > 0:
          n_a, n_b = geo["counts"][zc, a], geo["counts"][zc, b]
          npairs = n_a*(n_a - 1)/2 if a == b else n_a*n_b
          k = k*(npairs*omega_bin/args.area >= args.wcc_min_pairs)
        parts.append(k.astype(int))
  blocks["cc"] = np.concatenate(parts)

  # cs: [(zc, zs) cluster-major][lambda][theta]
  parts = []
  for zc in range(ncl):
    keep = tb.keep_above(theta_min_arcmin(args.rmin_cs, geo["zc"][zc]),
                         cs_rule)
    keep[-1] = 0  # zero last row of the Y transform
    for zs in range(ns):
      in_front = geo["zedges"][zc + 1] >= geo["zs"][zs]
      k = np.zeros(nt, dtype=int) if (args.cs_front_rule and in_front) else keep
      parts += [k]*nlam
  blocks["cs"] = np.concatenate(parts)
  return blocks


def cl_gc_blocks(blocks, geo, gg_lens_bins):
  """CL+GC: no ss, no gs, w_gg only in the lens bins of the cluster pairs."""
  out = dict(blocks)
  out["ss"] = np.zeros_like(blocks["ss"])
  out["gs"] = np.zeros_like(blocks["gs"])
  nt = geo["theta"].n
  gg = np.zeros_like(blocks["gg"])
  for l in gg_lens_bins:
    gg[l*nt:(l + 1)*nt] = blocks["gg"][l*nt:(l + 1)*nt]
  out["gg"] = gg
  return out


ORDER = ["ss", "gs", "gg", "cg", "N", "cc", "cs"]


def write_mask(path, blocks):
  mask = np.concatenate([blocks[b] for b in ORDER])
  with open(path, "w") as f:
    for i, m in enumerate(mask):
      f.write("%d %.1f\n" % (i, m))
  return mask.size


def report(title, blocks, compare_y3):
  print(title)
  start = 0
  for b in ORDER:
    size = blocks[b].size
    line = (f"  {b:3s} entries {start:5d}-{start + size - 1:5d} "
            f"(size {size:4d}): unmasked {int(blocks[b].sum()):4d}")
    if compare_y3 and b in Y3_PAPER_COUNTS:
      line += f"   [Y3 paper: {Y3_PAPER_COUNTS[b]}]"
    print(line)
    start += size
  total = sum(int(blocks[b].sum()) for b in ORDER)
  print(f"  total entries {start}, unmasked {total}")


def main():
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
  ncl, nlam = len(zedges) - 1, len(ledges) - 1
  if len(cg_lens) != ncl:
    raise SystemExit("--cg-lens-bins needs one lens bin per cluster bin")
  if max(cg_lens) >= args.lens_ntomo:
    raise SystemExit("--cg-lens-bins refers to a missing lens bin")
  gg_clgc = cg_lens if args.gg_lens_bins_clgc is None else \
            parse_list(args.gg_lens_bins_clgc, int)

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
  for name, r, zs in [("cs", args.rmin_cs, geo["zc"]), ("cg", args.rmin_cg,
                      geo["zc"] if "cluster" == args.cg_zbar else
                      geo["zl"][cg_lens]), ("cc", args.rmin_cc, geo["zc"]),
                      ("gg", args.rmin_gg, geo["zl"]), ("gs", args.rmin_gs, geo["zl"])]:
    print(f"theta_min {name} (R = {r:g} Mpc/h) [arcmin]:",
          " ".join("%.3f" % theta_min_arcmin(r, z) for z in zs))
  if args.cs_front_rule:
    pairs = [(zc + 1, zs + 1) for zc in range(ncl) for zs in range(args.source_ntomo)
             if zedges[zc + 1] >= geo["zs"][zs]]
    print("cs pairs masked by the source-in-front rule (zc, zs):", pairs)

  full = build_blocks(args, geo)
  clgc = cl_gc_blocks(full, geo, gg_clgc)
  path_clgc = args.out_prefix + "_4x2ptN.mask"
  path_full = args.out_prefix + "_6x2ptN.mask"
  n1 = write_mask(path_clgc, clgc)
  n2 = write_mask(path_full, full)
  rule = args.theta_rule + ("" if args.cs_theta_rule is None
                            else f", cs {args.cs_theta_rule}")
  report(f"\nCL+GC (4x2pt+N) -> {path_clgc} ({n1} entries; theta rule {rule})",
         clgc, args.compare_y3)
  report(f"\nCL+3x2pt (6x2pt+N) -> {path_full} ({n2} entries; ss cuts "
         f"{args.ss_cuts})", full, False)


if __name__ == "__main__":
  main()
