"""Cobaya likelihood des_cluster.combo_4x2pt_N: CL+GC = 4x2pt + N.

The cluster blocks N (cluster counts per redshift and richness bin), cs
(cluster lensing), cc (cluster clustering) and cg (cluster x galaxy
clustering), plus gg (galaxy clustering) of MagLim lens bins 1-3, of the
model of arXiv:2503.13631. The default .dataset file,
des_cluster_y6_4x2ptN.dataset (option data_file of combo_4x2pt_N.yaml),
points at the joint synthetic Y6-like data vector (2812 entries, shared
with combo_6x2pt_N) and at the 4x2pt + N mask, which keeps only these
blocks. combo_4x2pt_N.yaml also fixes the lens parameters that the mask
makes irrelevant (its fixed_params block).

The computation is in _cosmolike_prototype_base.py: this class only gives
it the probe name "4x2pt_N". start_cocoa.sh links this folder into Cobaya
as the package cobaya.likelihoods.des_cluster, so a YAML names the
likelihood des_cluster.combo_4x2pt_N (file name = class name).
"""
from cobaya.likelihoods.des_cluster._cosmolike_prototype_base import _cosmolike_prototype_base
import cosmolike_des_cluster_interface as ci
import numpy as np

class combo_4x2pt_N(_cosmolike_prototype_base):
  """4x2pt + N likelihood (probe "4x2pt_N"); see the module docstring."""
  def initialize(self):
    """Initialize the base class for the probe "4x2pt_N".

    Cobaya calls this once, after setting the options of combo_4x2pt_N.yaml
    and of the user's YAML as attributes. super(combo_4x2pt_N, self) is the
    base class, whose initialize reads the .dataset file and sets up
    CosmoLike, the cluster model included.

    Side effects:
      sets the global state of the compiled library (base-class initialize).
    """
    super(combo_4x2pt_N,self).initialize(probe="4x2pt_N")
