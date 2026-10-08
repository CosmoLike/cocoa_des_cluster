"""Locate the project interface for the optional covariance test sector.

pytest reads a file named conftest.py before it collects the tests of
the same folder, so the code here runs first for every test of
tests/covariance. It puts interface/ (the folder of the compiled library
cosmolike_des_cluster_interface) first on the import path and defines one
fixture that skips the whole folder when the library was compiled
without its covariance code. A fixture is a function pytest runs around
the tests that use it; this one is used automatically (autouse=True) and
runs once per pytest session (scope="session").

The covariance code is optional at compile time: it is left out while
the environment variable IGNORE_COSMOLIKE_DES_CLUSTER_COVARIANCE is set
(tests/covariance/README.md shows how to rebuild with it).
"""

from pathlib import Path
import sys

# parents[2] of this file's path is the project folder projects/des_cluster
project = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project/"interface"))

import pytest
import cosmolike_des_cluster_interface as ci


@pytest.fixture(scope="session", autouse=True)
def covariance_build():
    """Skip this sector when covariance generation was intentionally omitted.

    The compiled library sets has_covariance only when its covariance code
    was built; getattr(..., False) reads the flag and gives False when the
    attribute is absent.

    Arguments:
      none.

    Returns:
      nothing.

    Side effects:
      pytest.skip marks every test of this folder as skipped, with the
      message below, when the covariance code is absent.
    """
    if not getattr(ci, "has_covariance", False):
        pytest.skip(
            "Covariance generation is disabled. Unset "
            "IGNORE_COSMOLIKE_DES_CLUSTER_COVARIANCE after start_cocoa.sh, "
            "then recompile this project."
        )
