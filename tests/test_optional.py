####################################################################################################
#                                         test_optional.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-07                                                                              #
#                                                                                                  #
# Purpose: MRSuite without what pip cannot install: FSL-MRS, Augmentrum (which needs it) and       #
#          MRSValidate; and without nicegui. Everything else imports, and their parts say         #
#          how to install them. Importing it starts no Octave.                                     #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import os
import subprocess
import sys
from pathlib import Path

CHECKOUT = Path(__file__).resolve().parents[1]


#*********************#
#   without fsl-mrs   #
#*********************#
def test_mrsuite_runs_without_fsl_mrs_and_says_how_to_install_it(tmp_path):
    script = """
import importlib, pkgutil, sys
for name in ("fsl_mrs", "augmentrum", "mrsvalidate", "nicegui"):
    sys.modules[name] = None                                      # as if not installed
import mrsuite
from mrsuite.datasets import mrsi_challenge
from mrsuite.gui.__main__ import main
from mrsuite.simulators import mrs_sim
uses = {"mrsuite.bridges.fsl_mrs": lambda: mrsuite.fsl_mrs,
        "mrsuite.bridges.augmentrum": lambda: mrsuite.augmentrum,
        "mrsuite.benchmark": lambda: importlib.import_module("mrsuite.benchmark"),
        "augmentrum": lambda: mrsi_challenge._module(mrsi_challenge.DATA_DIR),
        "compile_basis": lambda: mrs_sim.compile_basis("x.BASIS", "x.mat", te=30),
        "gui": lambda: main([])}
# every module, but the scripts that run in other environments and the GUI's pages
skipped = ("mrsuite.models.reference", "_driver", "gui.app", "gui.canvas", "gui.files")
for module in pkgutil.walk_packages(mrsuite.__path__, "mrsuite."):
    if module.name not in uses and not module.name.endswith(skipped):
        importlib.import_module(module.name)
for name, use in uses.items():
    try:
        use()
    except (ImportError, SystemExit) as err:
        print(name, err)
"""
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                         cwd=tmp_path)

    assert run.returncode == 0, run.stderr
    for line in ("mrsuite.bridges.fsl_mrs FSL-MRS is not installed: conda install",
                 "mrsuite.bridges.augmentrum Augmentrum is not installed: pip install augmentrum",
                 "mrsuite.benchmark MRSValidate is not installed: pip install",
                 "augmentrum Augmentrum is not installed: pip install augmentrum",
                 "compile_basis FSL-MRS is not installed: conda install",
                 "gui The GUI needs nicegui, which comes with MRSuite: pip install mrsuite"):
        assert line in run.stdout, run.stdout


#************#
#   broken   #
#************#
def test_a_broken_fsl_mrs_says_what_is_missing_not_that_fsl_mrs_is(tmp_path):
    (tmp_path / "fsl_mrs").mkdir()
    (tmp_path / "fsl_mrs" / "__init__.py").write_text("import a_dependency_it_lacks\n")
    run = subprocess.run([sys.executable, "-c", "import mrsuite.bridges.fsl_mrs"],
                         capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": str(tmp_path)})

    assert "No module named 'a_dependency_it_lacks'" in run.stderr
    assert "FSL-MRS is not installed" not in run.stderr


#***************#
#   the suite   #
#***************#
def test_the_suite_collects_without_fsl_mrs_and_mrsvalidate():
    """What needs them skips, so a machine with neither, as CI's core job, runs the rest."""
    script = """
import sys
sys.modules.update(fsl_mrs=None, mrsvalidate=None)               # as if not installed
import pytest
sys.exit(pytest.main(["--collect-only", "-q", "-p", "no:cacheprovider", "tests"]))
"""
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                         cwd=CHECKOUT)

    assert run.returncode == 0, run.stdout[-3000:]


#**************#
#   fixtures   #
#**************#
def test_the_shared_spectra_and_fit_skip_without_fsl_mrs(tmp_path):
    """They are read and fitted by FSL-MRS, so without it they skip, saying why."""
    (tmp_path / "conftest.py").write_text(
        "from tests.conftest import basis, fsl_mrs_fit, spectra, spectrum\n")
    (tmp_path / "test_uses.py").write_text("".join(
        f"def test_{name}({name}):\n    pass\n"
        for name in ("spectrum", "spectra", "basis", "fsl_mrs_fit")))
    script = """
import sys
sys.modules["fsl_mrs"] = None                                    # as if not installed
import pytest
sys.exit(pytest.main(["-q", "-rs", "-p", "no:cacheprovider", sys.argv[1]]))
"""
    run = subprocess.run([sys.executable, "-c", script, str(tmp_path)], capture_output=True,
                         text=True, cwd=CHECKOUT)

    assert "4 skipped" in run.stdout and "could not import 'fsl_mrs'" in run.stdout, run.stdout


#********************#
#   without octave   #
#********************#
def test_importing_mrsuite_leaves_octave_alone(tmp_path):
    # importing oct2py starts an Octave, or prints that there is none: only a session may
    script = """
import importlib, pkgutil, sys
for name in ("oct2py", "octave_kernel"):
    sys.modules[name] = None
import mrsuite
skipped = ("mrsuite.models.reference", "_driver", "gui.app", "gui.canvas", "gui.files")
for module in pkgutil.walk_packages(mrsuite.__path__, "mrsuite."):
    if module.name != "mrsuite.octave.sessions" and not module.name.endswith(skipped):
        try:
            importlib.import_module(module.name)
        except ImportError as err:          # FSL-MRS or MRSValidate missing, and it says so
            if "is not installed" not in str(err):
                raise
"""
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                         cwd=tmp_path)

    assert run.returncode == 0, run.stderr

