"""Package marker of the des_cluster likelihoods.

Python treats a folder holding an __init__.py file as a package. Cobaya
imports the likelihoods of this folder as cobaya.likelihoods.des_cluster.<name>
(start_cocoa.sh links the folder into Cobaya under that name), and
_cosmolike_prototype_base.py holds the class they share. The file has no
other content.
"""
