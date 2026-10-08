"""Cobaya likelihood des_cluster.combo_2x2pt: galaxy-galaxy lensing + clustering.

The blocks gs (galaxy-galaxy lensing gamma_t of each lens-source pair) and
gg (galaxy clustering w of each lens bin). The default .dataset file,
des_y3_real.dataset (option data_file of combo_2x2pt.yaml), is the DES Y3
data set copied from the project des_y3; the nuisance parameters come from
params_source.yaml and params_lens_redmagic.yaml.

The computation is in _cosmolike_prototype_base.py: this class only gives
it the probe name "2x2pt". start_cocoa.sh links this folder into Cobaya as
the package cobaya.likelihoods.des_cluster, so a YAML names the
likelihood des_cluster.combo_2x2pt (file name = class name).
"""
from cobaya.likelihoods.des_cluster._cosmolike_prototype_base import _cosmolike_prototype_base, survey
import cosmolike_des_cluster_interface as ci
import numpy as np

class combo_2x2pt(_cosmolike_prototype_base):
  """2x2pt likelihood (probe "2x2pt"); see the module docstring."""
  def initialize(self):
    """Initialize the base class for the probe "2x2pt".

    Cobaya calls this once, after setting the options of combo_2x2pt.yaml
    and of the user's YAML as attributes. super(combo_2x2pt, self) is the
    base class, whose initialize reads the .dataset file and sets up
    CosmoLike.

    Side effects:
      sets the global state of the compiled library (base-class initialize).
    """
    super(combo_2x2pt,self).initialize(probe="2x2pt")