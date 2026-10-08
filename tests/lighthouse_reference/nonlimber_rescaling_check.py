"""Quantify lighthouse's memoized ("bias-rescaled") non-Limber C_ell.

C_clusterxclusterclustering_mix_tab / C_clusterxgalaxyclustering_mix_tab compute
the FFTLog non-Limber part only for the FIRST richness bin requested in a
redshift bin (static Nz / Nz_gal keys) and rescale it by b(lambda1) b(lambda2)
for the others. Here the non-Limber C_ell is recomputed from scratch for a few
(lambda1, lambda2) combinations by first requesting another z bin (which resets
the memo), and compared with the memoized value used in the data vector.

    python nonlimber_rescaling_check.py   (fresh process; ~1 min)

lighthouse is the repository of the original CosmoLike cluster code; lh.py
of this folder loads its prebuilt library. The script writes
outputs/nonlimber_rescaling_check.npz (C_ell for l = 0 ... 300 and the
projected w(theta) of both paths) and prints memo/fresh - 1 for w(theta)
and for C_ell at selected multipoles.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lh  # noqa: E402

# multipoles 0 ... LMAX - 1 of the C_ell arrays and of the Legendre sums
# that project them onto the angular bins
LMAX = 100000
cfg = lh.CONFIG
NT = cfg["ntheta"]

# one full data vector first, so every table is filled as in the main run
ndata = lh.init_all(Ystatistics=1)
dv = np.zeros(ndata)
lh.theory_wrapper(lh.cosmo_struct(), lh.nuisance_struct(), dv.ctypes.data_as(lh.PD))
W = lh.legendre_bin_weights(cfg, LMAX)
Cl = np.zeros(LMAX)


def cc(nz, l1, l2):
    """Return the w_cc C_ell of richness bins (l1, l2) in cluster z bin nz.

    The C function fills Cl in place: non-Limber (FFTLog) from l = 0 until
    it agrees with Limber to tolerance = 0.01 (dev = 0.1 is the starting
    deviation that enters that loop), Limber above. Its memo keeps the
    non-Limber part of the first richness pair of a redshift bin.

    Arguments:
      nz     = cluster redshift bin (zero-based).
      l1, l2 = richness bins (zero-based).

    Returns:
      a copy of Cl, float array [LMAX].
    """
    Cl[:] = 0.0
    lh.C_cc_mix_tab(0, LMAX, nz, nz, l1, l2, Cl.ctypes.data_as(lh.PD), 0.1, 0.01)
    return Cl.copy()


def cg(zc, zg, il):
    """Return the w_cg C_ell of richness bin il, cluster bin zc, lens bin zg.

    Same rules as cc above.

    Arguments:
      zc = cluster redshift bin, zg = lens bin, il = richness bin (all
           zero-based).

    Returns:
      a copy of Cl, float array [LMAX].
    """
    Cl[:] = 0.0
    lh.C_cg_mix_tab(0, LMAX, zc, zg, il, Cl.ctypes.data_as(lh.PD), 0.1, 0.01)
    return Cl.copy()


# out collects every array for np.savez_compressed; W[:, 1:] @ C[1:] is the
# bin-averaged Legendre projection of a C_ell onto the angular bins (l >= 1)
out = {}
# w_cc: memoized path (as in the data vector: (nz,0,0) first, then others rescaled)
for (l1, l2) in [(1, 1), (3, 3), (0, 3), (1, 2)]:
    cc(0, 0, 0)
    memo = cc(0, l1, l2)
    cc(1, 0, 0)            # reset memo to another z bin
    fresh = cc(0, l1, l2)  # full FFTLog non-Limber for (l1, l2)
    key = "cc_nz0_l%d_l%d" % (l1, l2)
    out[key + "_Cl_memo_l0_300"] = memo[:301]
    out[key + "_Cl_fresh_l0_300"] = fresh[:301]
    out[key + "_w_memo"] = W[:, 1:] @ memo[1:]
    out[key + "_w_fresh"] = W[:, 1:] @ fresh[1:]
for il in [1, 3]:
    cg(0, 0, 0)
    memo = cg(0, 0, il)
    cg(1, 1, 0)
    fresh = cg(0, 0, il)
    key = "cg_zc0_zg0_l%d" % il
    out[key + "_Cl_memo_l0_300"] = memo[:301]
    out[key + "_Cl_fresh_l0_300"] = fresh[:301]
    out[key + "_w_memo"] = W[:, 1:] @ memo[1:]
    out[key + "_w_fresh"] = W[:, 1:] @ fresh[1:]
np.savez_compressed(os.path.join(HERE, "outputs", "nonlimber_rescaling_check.npz"), **out)
np.set_printoptions(precision=4, linewidth=200)
for k in sorted(out):
    if k.endswith("_w_memo"):
        b = k[:-len("_w_memo")]
        r = out[b + "_w_memo"] / out[b + "_w_fresh"] - 1
        print(b, "w_memo/w_fresh-1:", r)
        c = out[b + "_Cl_memo_l0_300"][1:200] / out[b + "_Cl_fresh_l0_300"][1:200] - 1
        print("   C_l memo/fresh-1 at l=1,2,5,10,20,50,100,199:", c[[0, 1, 4, 9, 19, 49, 99, 198]])
