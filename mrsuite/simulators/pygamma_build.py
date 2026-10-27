####################################################################################################
#                                         pygamma_build.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 27/10/26                                                                                #
#                                                                                                  #
# Purpose: PyGAMMA, the Python wrapper of the GAMMA spin-physics library Vespa's Simulation runs,  #
#          built from its pinned source by its own Makefile on first use, into MRSuite's folder:   #
#          PyPI has no build of it for Python 3.10 on, nor for arm64.                              #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from mrsuite import externals

# GAMMA's Makefile for each system, under its platforms/
MAKEFILES = {"linux": "Linux", "darwin": "OSX"}

# what its Makefile runs, by the names it calls them
TOOLS = ("make", "g++", "swig")


#************#
#   folder   #
#************#
def folder() -> Path:
    """Where the build for this system, Python and pinned commit lies: the folder that goes
    on sys.path, holding the package ``pygamma``."""
    commit = externals.REGISTRY["pygamma"][1][:12]
    python = f"cp{sys.version_info[0]}{sys.version_info[1]}"
    return externals.local(".pygamma") / f"{sys.platform}-{platform.machine()}-{python}-{commit}"


#************#
#   ensure   #
#************#
def ensure() -> Path:
    """
    PyGAMMA importable as ``pygamma``: built on first use, its folder put on sys.path.

    Returns:
        Path: The folder on sys.path.

    Raises:
        RuntimeError: As ``build`` does.
    """
    where = folder()
    if not (where / "pygamma" / "__init__.py").is_file():
        build(where)
    if str(where) not in sys.path:
        sys.path.insert(0, str(where))
    return where


#***********#
#   build   #
#***********#
def build(dest: Path) -> None:
    """
    Build PyGAMMA as published: its Makefile's ``pysgdist`` target, on a copy of the pinned
    source (the build writes into its tree), with make's own variables set on its command
    line; the files it stages are the package, with the ``__init__.py`` its setup.py would
    copy in from ``init.py``. That setup.py is not run: it accepts Python 2.7 and 3.6-3.9
    only. A few minutes single-threaded; ``MAKEFLAGS=-j8`` in the environment runs more jobs.

    On Linux, ``-std=gnu++98`` keeps GCC 15's std::complex from clashing with GAMMA's own
    complex type. On macOS the build is unverified: its Makefile takes the architecture
    from the pointer size (x86_64 for any 64-bit Python) and targets OS X 10.9, so
    ``PYTHONARCH`` and ``MIN_OSX_VERSION`` are set for this machine and 11.0, the first
    macOS on arm64.

    Args:
        dest (Path): The folder to build into, moved in whole once built.

    Raises:
        RuntimeError: On a system other than Linux or macOS, when make, g++ or SWIG is not
            on PATH, or when the build fails, with the end of its output.
    """
    if sys.platform not in MAKEFILES:
        raise RuntimeError(f"PyGAMMA builds on Linux and macOS, not on {sys.platform}")
    missing = [tool for tool in TOOLS if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(f"building PyGAMMA needs {', '.join(missing)} on PATH (make, g++ and "
                           f"SWIG; e.g. conda install -c conda-forge make gxx swig)")
    source = externals.ensure("pygamma")
    flags = [f"PYTHON={sys.executable}", f"SWIG={shutil.which('swig')}"]
    if sys.platform == "linux":
        flags.append("OPTIMIZE=-O3 -std=gnu++98")
    else:
        flags += [f"PYTHONARCH=-arch {platform.machine()}", "MIN_OSX_VERSION=11.0"]

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = Path(tempfile.mkdtemp(prefix=".part", dir=dest.parent))
    try:
        tree = part / "gamma"
        shutil.copytree(source, tree, ignore=shutil.ignore_patterns(".git"))
        run = subprocess.run(["make", *flags, "pysgdist"], capture_output=True, text=True,
                             cwd=tree / "platforms" / MAKEFILES[sys.platform])
        staged = tree / "pygamma" / "dist_staging"
        if run.returncode != 0 or not (staged / "pygamma.py").is_file():
            raise RuntimeError(f"building PyGAMMA failed:\n{(run.stdout + run.stderr)[-2000:]}")
        shutil.copytree(staged, part / "pygamma")
        shutil.copy(tree / "pygamma" / "init.py", part / "pygamma" / "__init__.py")
        shutil.rmtree(tree)
        externals.move_in(part, dest)
    finally:
        shutil.rmtree(part, ignore_errors=True)
