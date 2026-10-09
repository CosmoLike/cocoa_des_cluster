"""Check that a compiled covariance call refuses a malformed input.

The covariance C++ layer validates its inputs like the data-vector
layer: a failed check prints one line through the shared logger
([critical] with the function name and the reason) and ends the whole
process with exit(1). It never raises a Python exception, so
pytest.raises cannot observe a refusal - the exit would take the test
runner down with it.

assert_aborts therefore runs the call in a forked copy of the test
process. os.fork() duplicates the process at the call site, so the
child inherits the already-initialized interface, tables and test
inputs at no cost; only the one malformed call runs there. The child
must die with a nonzero exit status:

  - current layer: the validation prints [critical] and calls exit(1);
  - a Python-level check that raises instead: the uncaught exception
    also ends the child with status 1.

Either way the parent's waitpid sees a nonzero status, so this check
does not depend on which layer catches the bad input. A child that
reaches os._exit(0) means the malformed input was ACCEPTED, and the
test fails.

Forking a process that already started OpenMP threads is safe here
because a refused call ends before any parallel region starts; an
accepted call could misbehave in the child, but acceptance is already
the failure this check exists to catch.
"""

import os


def assert_aborts(attempt):
    """Run attempt() in a forked child; require a nonzero exit status.

    Arguments:
      attempt = argument-free function performing exactly one call with
        one malformed input. It runs only in the child process.

    Returns:
      nothing; an AssertionError in the parent fails the test when the
      child accepted the input (exit status 0) or died by a signal.
    """
    child = os.fork()
    if child == 0:
        # Child: the expected [critical] line is noise in the test log,
        # so stdout and stderr go to /dev/null before the call.
        quiet = os.open(os.devnull, os.O_WRONLY)
        os.dup2(quiet, 1)
        os.dup2(quiet, 2)
        # The child must never return into the test runner it inherited
        # from the fork: every outcome of attempt() ends in os._exit,
        # which leaves immediately without unwinding into pytest.
        # BaseException also catches SystemExit, so a Python-level
        # refusal (raise) and an accepted call both take a defined exit.
        try:
            attempt()
        except BaseException:
            os._exit(1)
        os._exit(0)
    status = os.waitpid(child, 0)[1]
    # WIFEXITED is False when a signal killed the child (for example a
    # crash inside the call): that is a bug, not a refusal.
    assert os.WIFEXITED(status), "malformed-input child died by a signal"
    assert os.WEXITSTATUS(status) != 0, "malformed input was accepted"
