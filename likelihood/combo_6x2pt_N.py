"""Cobaya likelihood des_cluster.combo_6x2pt_N: CL+3x2pt = 6x2pt + N.

Every block of the joint vector of arXiv:2503.13631: cosmic shear (ss),
galaxy-galaxy lensing (gs), galaxy clustering (gg) of all six MagLim lens
bins, and the cluster blocks cg (cluster x galaxy clustering), N (cluster
counts per redshift and richness bin), cc (cluster clustering) and cs
(cluster lensing). The default .dataset file,
des_cluster_y6_6x2ptN.dataset (option data_file of combo_6x2pt_N.yaml),
points at the joint synthetic Y6-like data vector (2812 entries, shared
with combo_4x2pt_N) and at the 6x2pt + N mask.

The computation is in _cosmolike_prototype_base.py: this class only gives
it the probe name "6x2pt_N". start_cocoa.sh links this folder into Cobaya
as the package cobaya.likelihoods.des_cluster, so a YAML names the
likelihood des_cluster.combo_6x2pt_N (file name = class name).
"""
from cobaya.likelihoods.des_cluster._cosmolike_prototype_base import _cosmolike_prototype_base
import cosmolike_des_cluster_interface as ci
import numpy as np

class combo_6x2pt_N(_cosmolike_prototype_base):
  """6x2pt + N likelihood (probe "6x2pt_N"); see the module docstring."""
  def initialize(self):
    """Initialize the base class for the probe "6x2pt_N".

    Cobaya calls this once, after setting the options of combo_6x2pt_N.yaml
    and of the user's YAML as attributes. super(combo_6x2pt_N, self) is the
    base class, whose initialize reads the .dataset file and sets up
    CosmoLike, the cluster model included.

    Side effects:
      sets the global state of the compiled library (base-class initialize).
    """
    super(combo_6x2pt_N,self).initialize(probe="6x2pt_N")
