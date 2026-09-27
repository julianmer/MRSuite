####################################################################################################
#                                            conftest.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-08-08                                                                              #
#                                                                                                  #
# Purpose: Drives Octave and MATLAB from Python for the shim tests. Both engines run the           #
#          same script text and hand back a struct named ``out`` as JSON, so a case can be         #
#          written once and compared across the two.                                               #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import json
import pathlib
import subprocess
import tempfile
import textwrap

import numpy as np
import pytest

from mrsuite.octave import shim_path

from tests.conftest import MATLAB

# both engines are driven from a script file rather than an inline command: `matlab
# -batch` silently produces no output for a multi-line command string
EMIT = "fprintf('%s\\n', jsonencode(out));"


#*********#
#   run   #
#*********#
def _run(cmd):
    """Run an engine and return the ``out`` struct it printed, decoded."""
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, \
        f"{cmd[0]} failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"

    for line in reversed(proc.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise AssertionError(f"no JSON emitted by {cmd[0]}:\n{proc.stdout[-2000:]}")


#************#
#   script   #
#************#
def _script(body, datafile, prelude=""):
    """Write a runnable script to a temp .m and return (dir, stem)."""
    head = prelude + (f"DATAFILE = '{datafile}';\n" if datafile else "")
    text = f"{head}{textwrap.dedent(body).strip()}\n{EMIT}"

    directory = tempfile.mkdtemp(prefix="mrsuite_engine_")
    (pathlib.Path(directory) / "engine_case.m").write_text(text)
    return directory, "engine_case"


#************#
#   octave   #
#************#
def run_octave(body, datafile=None):
    """Run ``body`` in Octave with the shims on the path.

    Args:
        body (str): Script text that must define a struct named ``out``.
        datafile (Optional[str]): Path bound to ``DATAFILE`` before the body runs.

    Returns:
        dict: The decoded ``out`` struct.
    """
    directory, stem = _script(body, datafile, prelude=f"addpath('{shim_path()}');\n")
    return _run(["octave-cli", "--no-gui", "--quiet", "--path", directory, "--eval", stem])


#************#
#   matlab   #
#************#
def run_matlab(body, datafile=None):
    """Run ``body`` in MATLAB, with no shims on the path.

    MATLAB uses its own nlinfit/statset/sscanf, which is the entire point of comparing.

    Args:
        body (str): Script text that must define a struct named ``out``.
        datafile (Optional[str]): Path bound to ``DATAFILE`` before the body runs.

    Returns:
        dict: The decoded ``out`` struct.
    """
    directory, stem = _script(body, datafile)
    return _run([MATLAB, "-batch", f"addpath('{directory}'); {stem}"])


#******************#
#   lorentz data   #
#******************#
@pytest.fixture(scope="session")
def lorentz_data():
    """A noisy Lorentzian written to disk, for both engines to load.

    Generating it in-script would compare fits to *different* data: ``randn('seed', N)``
    does not produce the same sequence in Octave as in MATLAB, and each engine would
    then be fitting its own noise.
    """
    rng = np.random.default_rng(7)
    ppm = np.linspace(-2, 2, 400)
    amp, hwhm, ctr, off = 5.0, 0.12, 0.35, 0.2
    y = amp * hwhm ** 2 / ((ppm - ctr) ** 2 + hwhm ** 2) + off + 0.01 * rng.standard_normal(400)

    path = pathlib.Path(tempfile.mkdtemp(prefix="mrsuite_lorentz_")) / "lorentz.txt"
    np.savetxt(path, np.column_stack([ppm, y]), fmt="%.17g")
    return str(path)
