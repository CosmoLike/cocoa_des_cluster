"""Check that a compiled covariance call refuses a malformed input.

The implementation and its documentation live in
cosmolike_core/cocoa_testing.py (assert_aborts): the compiled layer
refuses a malformed input by printing one [critical] line and ending
the whole process, so the check must fork and watch the child's exit
status. The covariance tests import this module by name from their
own folder, so this file stays here as the binding to the shared
function.
"""

import os
import sys

# the core folder, computed from this file's location so the import
# works no matter where pytest was launched from (covariance/ sits
# four levels below the Cocoa/ folder)
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "..", "external_modules", "code",
    "cosmolike_core")))
# the from-import binds the shared function as this module's own
# name, so `from abort_check import assert_aborts` keeps working
from cocoa_testing import assert_aborts
