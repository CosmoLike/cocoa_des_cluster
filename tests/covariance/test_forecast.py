"""Check this project's covariance adapter with the shared forecast check.

The check is check_project_forecast of the shared module
cocoa_covariance_testing.py (in cosmolike_core). With this project's
survey adapter (under covariance/) it verifies the full layout lengths
of both measurement spaces, that a refinement of the accuracy keeps
the measurement bins fixed, that the G, SSC, cNG and total matrices of
a measured subset are finite and symmetric in real and Fourier space,
that every component is bitwise identical with one and eight OpenMP
threads, that the subset's total has positive variances in every
direction, and that the saved archive reads back.

The check uses small numerical settings and a subset of the measured
rows, so it does not certify survey convergence. Run it
separately from tests/data_vector (the two sectors initialize
different compiled-library state), from the Cocoa/ folder after
`source start_cocoa.sh`, with the project built with covariance
support:

    python -m pytest projects/des_cluster/tests/covariance
"""

from pathlib import Path
import sys

# parents[2] of this file is the project folder and its parents[1] is
# Cocoa/: the shared core (cocoa_covariance_testing), the compiled
# module (interface/) and the survey adapter (covariance/) go first on
# the module search path.
project = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project.parents[1]/"external_modules/code/cosmolike_core"))
sys.path.insert(0, str(project/"interface"))
sys.path.insert(0, str(project/"covariance"))

from cocoa_covariance_testing import check_project_forecast
import cosmolike_des_cluster_interface as ci
import des_cluster_covariance as survey


def test_forecast_adapter(tmp_path):
    """Real and Fourier components repeat at one/eight threads and save intact.

    Arguments:
      tmp_path = a temporary folder pytest creates for this test (a
                 built-in fixture, passed by the argument's name); the
                 check writes its archives there.

    Returns:
      nothing; a failed assertion inside check_project_forecast fails
      the test.
    """
    # expected_sizes = (1000, 600): the full real-space vector holds
    # xi_+ and xi_- of 10 source pairs, gamma_t of 24 lens-source
    # pairs and w of 6 lens bins in 20 angular bins
    # (400 + 480 + 120), and the Fourier vector one spectrum per
    # pair in 15 bands ((10 + 24 + 6) x 15).
    check_project_forecast(
        interface=ci, survey=survey, expected_sizes=(1000, 600),
        directory=tmp_path,
    )
