"""Setuptools-style loader stub of the compiled library of this project.

The compiled CosmoLike library is the extension module
cosmolike_des_cluster_interface.so in this folder (built from interface.cpp
by scripts/compile_des_cluster.sh), and start_des_cluster.sh puts this
folder on PYTHONPATH. In each folder Python looks for an extension module
(.so) before a .py file of the same name, so `import
cosmolike_des_cluster_interface` loads the .so directly and this file does
not run in that layout. The stub is the loader that setuptools writes for
egg packages: it finds the .so with pkg_resources and loads it with
imp.load_dynamic, two modules that are deprecated (imp no longer exists
from Python 3.12 on).
"""
def __bootstrap__():
   """Load cosmolike_des_cluster_interface.so in place of this module.

   The global statement makes the assignments below rebind the module-level
   names; the function then deletes itself and __loader__, sets __file__ to
   the path of the .so file, and loads the extension module under this
   module's name.
   """
   global __bootstrap__, __loader__, __file__
   import sys, pkg_resources, imp
   __file__ = pkg_resources.resource_filename(__name__,'cosmolike_des_cluster_interface.so')
   __loader__ = None; del __bootstrap__, __loader__
   imp.load_dynamic(__name__,__file__)
__bootstrap__()
