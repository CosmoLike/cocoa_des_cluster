from cobaya.likelihoods.des_cluster._cosmolike_prototype_base import _cosmolike_prototype_base
import cosmolike_des_cluster_interface as ci
import numpy as np

class combo_6x2pt_N(_cosmolike_prototype_base):
  def initialize(self):
    super(combo_6x2pt_N,self).initialize(probe="6x2pt_N")
