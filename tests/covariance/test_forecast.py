"""Check this project's galaxy/shear covariance interface and catalog inputs.

Run separately from tests/data_vector; the shared check uses small numerical
settings and a measured subset, so it does not certify survey convergence.

The check itself is check_project_forecast of the shared module
cocoa_covariance_testing.py (in cosmolike_core). With this project's
adapter covariance/des_cluster_covariance.py it verifies the layout of the
galaxy and shear vector, that refining the accuracy keeps the measured
bins fixed, that the G, SSC, cNG and total matrices of a measured subset
are finite, symmetric and identical bit for bit at one and eight OpenMP
threads in real and Fourier space, that the tested subset has positive
variance in every direction, and that the saved archive reads back.

Run from the cocoa/Cocoa folder after `source start_cocoa.sh`:

    python -m pytest projects/des_cluster/tests/covariance
"""

from pathlib import Path
import sys

# project = projects/des_cluster; the three folders put first on the import
# path hold cocoa_covariance_testing.py (cosmolike_core), the compiled
# library (interface/) and the survey adapter (covariance/)
project = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project.parents[1]/"external_modules/code/cosmolike_core"))
sys.path.insert(0, str(project/"interface"))
sys.path.insert(0, str(project/"covariance"))

from cocoa_covariance_testing import check_project_forecast
import cosmolike_des_cluster_interface as ci
import des_cluster_covariance as survey


def test_forecast_adapter(tmp_path):
    """Real and Fourier components repeat at one/eight threads and save intact.

    expected_sizes = (1000, 600): the full real-space vector holds xi_+ and
    xi_- of 10 source pairs, gamma_t of 24 lens-source pairs and w of 6
    lens bins in 20 angular bins (400 + 480 + 120), and the Fourier vector
    one spectrum per pair in 15 bands ((10 + 24 + 6) x 15).

    Arguments:
      tmp_path = pytest's built-in fixture: a fresh temporary directory
                 (a pathlib.Path) that receives the saved archives.

    Returns:
      nothing; a failed assertion inside check_project_forecast fails the
      test.
    """
    check_project_forecast(
        interface=ci, survey=survey, expected_sizes=(1000, 600), directory=tmp_path,
    )
