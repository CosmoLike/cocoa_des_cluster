from cobaya.likelihoods.des_cluster._cosmolike_prototype_base import _cosmolike_prototype_base, survey
import cosmolike_des_cluster_interface as ci
import numpy as np

class cosmic_shear(_cosmolike_prototype_base):
  def initialize(self):
    super(cosmic_shear,self).initialize(probe="xi")