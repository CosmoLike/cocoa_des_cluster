"""Cobaya likelihood des_cluster.cosmic_shear: the cosmic-shear block alone.

The block ss holds the shear two-point functions xi_+ and xi_- of every
source-bin pair. The default .dataset file, des_y3_real.dataset (option
data_file of cosmic_shear.yaml), is the DES Y3 data set copied from the
project des_y3; the nuisance parameters come from params_source.yaml.

The computation is in _cosmolike_prototype_base.py: this class only gives
it the probe name "xi". start_cocoa.sh links this folder into Cobaya as
the package cobaya.likelihoods.des_cluster, so a YAML names the
likelihood des_cluster.cosmic_shear (file name = class name).
"""
from cobaya.likelihoods.des_cluster._cosmolike_prototype_base import _cosmolike_prototype_base, survey
import cosmolike_des_cluster_interface as ci
import numpy as np

class cosmic_shear(_cosmolike_prototype_base):
  """Cosmic-shear likelihood (probe "xi"); see the module docstring."""
  def initialize(self):
    """Initialize the base class for the probe "xi".

    Cobaya calls this once, after setting the options of cosmic_shear.yaml
    and of the user's YAML as attributes. super(cosmic_shear, self) is the
    base class, whose initialize reads the .dataset file and sets up
    CosmoLike.

    Side effects:
      sets the global state of the compiled library (base-class initialize).
    """
    super(cosmic_shear,self).initialize(probe="xi")