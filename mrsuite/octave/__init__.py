####################################################################################################
#                                           __init__.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-08-08                                                                              #
#                                                                                                  #
# Purpose: Octave compatibility layer for the MATLAB toolboxes MRSuite calls. Holds the            #
#          shims for MATLAB-only functions Octave lacks, builds Osprey's bundled                   #
#          L-BFGS-B-C optimiser as an Octave MEX, and starts sessions against either a             #
#          local Octave or the container built from the dockerfile beside them.                    #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import atexit
import os
import shutil
import subprocess
import tempfile
from functools import lru_cache
from typing import TYPE_CHECKING, Optional

from mrsuite import externals
from mrsuite.octave import docker, matlab

if TYPE_CHECKING:                   # oct2py starts an Octave when imported, so only a session does
    from oct2py import Oct2Py

HERE = os.path.dirname(os.path.abspath(__file__))

# "local" uses the octave-cli on PATH, "docker" the container built from the
# dockerfile beside this file, and "auto" prefers local and falls back. "matlab" runs
# the toolboxes in MATLAB instead, for anyone with a licence; it is only ever chosen
# explicitly. Settable per session, or once via the environment.
RUNTIME = os.environ.get("MRSUITE_OCTAVE_RUNTIME", "auto")


#***************#
#   shim path   #
#***************#
def shim_path() -> str:
    """Directory holding the Octave shims, to prepend to the Octave path."""
    return os.path.join(HERE, "shims")


#*************#
#   scratch   #
#*************#
def scratch(prefix: str = "mrsuite_") -> str:
    """A temporary directory both runtimes can see.

    Under the repo, or an installed MRSuite's own folder, rather than the system one: the
    container has only those bind-mounted, so a /var/folders path exists on the host and
    nowhere else. Anything Octave has to open by name has to live somewhere both agree on.
    """
    docker.TEMP_DIR.mkdir(parents=True, exist_ok=True)
    return tempfile.mkdtemp(prefix=prefix, dir=str(docker.TEMP_DIR))


#*************#
#   runtime   #
#*************#
def resolve_runtime(runtime: Optional[str] = None) -> str:
    """Settle on 'local', 'docker' or 'matlab', raising if the choice cannot be honoured.

    Args:
        runtime: 'local', 'docker', 'matlab' or 'auto'. Defaults to ``RUNTIME``.

    Returns:
        The runtime to use.

    Raises:
        ValueError: If ``runtime`` is not one of the four.
        RuntimeError: If the requested runtime is unavailable, or neither is.
    """
    runtime = runtime or RUNTIME
    have_local = shutil.which("octave-cli") is not None

    if runtime == "local":
        if not have_local:
            raise RuntimeError("no octave-cli on PATH; install Octave or use runtime='docker'")
        return "local"

    if runtime == "docker":
        if not docker.available():
            raise RuntimeError("no Docker daemon reachable; start Docker or use runtime='local'")
        return "docker"

    if runtime == "matlab":
        if not matlab.available():
            raise RuntimeError("the MATLAB Engine for Python is not installed; pip install "
                               "<matlabroot>/extern/engines/python")
        return "matlab"

    if runtime != "auto":
        raise ValueError(f"unknown runtime '{runtime}'; "
                         f"expected 'local', 'docker', 'matlab' or 'auto'")

    if have_local:
        return "local"
    if docker.available():
        return "docker"
    raise RuntimeError("no Octave available: install one, or start Docker for the "
                       "containerised runtime (see mrsuite/octave/dockerfile)")


#*************#
#   session   #
#*************#
def session(root: str, runtime: Optional[str] = None, verbose: bool = False) -> "Oct2Py":
    """Start an Octave session with ``root`` and the shims on its path.

    The container is a drop-in for a local interpreter rather than a separate execution
    model, so the two runtimes differ only in which binary oct2py launches. The 'matlab'
    runtime answers the same calls from MATLAB, with only ``root`` on its path: MATLAB
    has the functions the shims stand in for, and Osprey ships its MEX for MATLAB.

    Args:
        root: Toolbox directory to add recursively.
        runtime: 'local', 'docker', 'matlab' or 'auto'. Defaults to ``RUNTIME``.
        verbose: Leave the interpreter's warnings on.

    Returns:
        A live session.
    """
    which = resolve_runtime(runtime)

    if which == "matlab":
        oc = matlab.MatlabSession(root)
        if not verbose:
            oc.eval("warning('off', 'all');")
        return oc

    from mrsuite.octave.sessions import FreshExchange, Unprompted

    if which == "docker":
        docker.start()
        docker.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        # oct2py reads OCTAVE_EXECUTABLE when it builds the engine, and nowhere else
        previous = os.environ.get("OCTAVE_EXECUTABLE")
        os.environ["OCTAVE_EXECUTABLE"] = docker.executable()
        # a folder of its own: oct2py passes every call through writer.mat and reader.mat
        # in it, so sessions sharing one overwrite each other's data
        exchange = scratch("oct2py_")
        atexit.register(shutil.rmtree, exchange, True)
        try:
            oc = FreshExchange(temp_dir=exchange)
        finally:
            if previous is None:
                os.environ.pop("OCTAVE_EXECUTABLE", None)
            else:
                os.environ["OCTAVE_EXECUTABLE"] = previous
    else:
        oc = Unprompted()

    if not verbose:
        oc.eval("warning('off', 'all');")
    # oct2py ends a session by killing Octave, which then dumps its workspace to
    # ./octave-workspace; parallel workers sharing a folder race to delete it
    oc.eval("crash_dumps_octave_core(false);")
    oc.eval(f"addpath(genpath('{root}'));")
    # shims and this runtime's MEX last: addpath prepends, so they end up at the front,
    # where they stand in for the MATLAB-only functions the toolboxes expect and shadow
    # any MEX built for the other runtime
    oc.eval(f"addpath('{shim_path()}');")
    mex = mex_path(which)
    if os.path.isdir(mex):
        oc.eval(f"addpath('{mex}');")
    return oc


#******************#
#   platform tag   #
#******************#
def platform_tag(runtime: Optional[str] = None) -> str:
    """Name for the machine and Octave a MEX will be loaded into, e.g.
    'darwin-arm64-octave9'.

    A MEX is a native shared library built against one Octave, so the local one and the
    container's are different binaries under the same filename, and one built for
    Octave 7 does not load into Octave 9. Each build is kept under its own tag and only
    the matching directory goes on the Octave path.
    """
    which = resolve_runtime(runtime)
    return _platform_tag(which, docker.image_id() if which == "docker" else "")


@lru_cache(maxsize=None)
def _platform_tag(which: str, image: str) -> str:
    """platform_tag, once per runtime and image: it asks Octave, which takes a while."""
    probe = ["uname -sm", "octave-cli --eval 'disp(OCTAVE_VERSION)'"]
    if which == "docker":
        docker.start()
        run = [["docker", "exec", docker.CONTAINER, "sh", "-c", p] for p in probe]
    else:
        run = [["sh", "-c", p] for p in probe]
    uname, version = (subprocess.run(r, capture_output=True, text=True, timeout=120)
                      .stdout.strip() for r in run)
    system, machine = (uname.split() + ["", ""])[:2]
    major = version.splitlines()[-1].split(".")[0] if version else "unknown"
    return f"{system.lower()}-{'arm64' if machine == 'aarch64' else machine}-octave{major}"


#*************#
#   mex dir   #
#*************#
def mex_path(runtime: Optional[str] = None) -> str:
    """Directory holding the MEX files built for this runtime."""
    return os.path.join(externals.local(".octave-mex"), platform_tag(runtime))


#******************#
#   build lbfgsb   #
#******************#
def build_lbfgsb(osprey_root: str, runtime: Optional[str] = None,
                 force: bool = False) -> str:
    """Build Osprey's L-BFGS-B-C wrapper as an Octave MEX.

    Osprey's core fit reaches ``lbfgsb_wrapper``, and the binaries it ships are MATLAB
    MEX files that Octave cannot load. The C sources are bundled, so the fix is to
    compile them with ``mkoctfile``.

    Built inside the container for the docker runtime, since a MEX is native code: the
    host's would be the wrong architecture. That works because the repo and Osprey are
    bind-mounted at the same absolute paths, so one command line serves both.

    The one non-obvious part is ``-DMATLAB_MEX_FILE``: ``src/lbfgsb.h`` guards
    ``#include "mex.h"`` behind that macro, and ``mkoctfile --mex`` does not define it,
    so without it every ``mxArray``/``mexErrMsgTxt`` reference fails to compile.

    Args:
        osprey_root: Path to an Osprey checkout.
        runtime: 'local', 'docker' or 'auto'. Decides where it is compiled and under
            which platform tag it is kept.
        force: Rebuild even if a MEX for this runtime is already present.

    Returns:
        Path to the built ``lbfgsb_wrapper.mex``.

    Raises:
        FileNotFoundError: If the sources or ``mkoctfile`` are missing.
        RuntimeError: If compilation fails.
    """
    which = resolve_runtime(runtime)
    lib = os.path.join(osprey_root, "libraries", "L-BFGS-B-C")
    mat, src = os.path.join(lib, "Matlab"), os.path.join(lib, "src")

    target = mex_path(which)
    out = os.path.join(target, "lbfgsb_wrapper.mex")
    # the wrapper Osprey ships, and MRSuite's variant for Osprey's quadratic subproblems
    # (see shims/lbfgsb.m), both over the same L-BFGS-B-C sources
    gateways = {out: os.path.join(mat, "lbfgsb_wrapper.c"),
                os.path.join(target, "lbfgsb_quadratic.mex"):
                    os.path.join(HERE, "src", "lbfgsb_quadratic.c")}
    missing = {o: g for o, g in gateways.items() if force or not os.path.exists(o)}
    if not missing:
        return out
    if not os.path.isdir(src):
        raise FileNotFoundError(f"L-BFGS-B-C sources not found under {lib}")
    os.makedirs(target, exist_ok=True)
    if which == "local" and shutil.which("mkoctfile") is None:
        raise FileNotFoundError(
            "mkoctfile not on PATH - install Octave development files, "
            "or use runtime='docker', which brings its own")
    if which == "docker":
        docker.start()

    units = ["lbfgsb", "linesearch", "subalgorithms", "print", "linpack",
             "miniCBLAS", "timer"]
    for built, gateway in missing.items():
        build = ["mkoctfile", "--mex", "-DMATLAB_MEX_FILE", f"-I{src}", "-UDEBUG",
                 "-o", built, gateway] + [os.path.join(src, f"{u}.c") for u in units]
        cmd = ["docker", "exec", "-w", mat, docker.CONTAINER] + build \
            if which == "docker" else build
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              cwd=None if which == "docker" else mat, timeout=1800)
        if proc.returncode != 0 or not os.path.exists(built):
            raise RuntimeError(f"L-BFGS-B-C build failed ({which}, "
                               f"{os.path.basename(built)}):\n{proc.stderr[-2000:]}")
    return out


#*****************#
#   build spm12   #
#*****************#
def build_spm12(spm_root: str, runtime: Optional[str] = None) -> str:
    """
    SPM12's MEX files built for Octave, as SPM documents it: its own Makefile, ``make
    PLATFORM=octave install`` in src/, which compiles them with mkoctfile and puts each
    beside the code that calls it, in the checkout (a class's private folder among them,
    so they cannot live elsewhere). Nothing of SPM's is changed; what is built is new files.

    SPM12's C is older than C23, which GCC 15 compiles by default and which makes ``bool``
    a keyword, so it is compiled as GNU C17, on top of the flags mkoctfile would use.

    Args:
        spm_root (str): An SPM12 checkout.
        runtime (Optional[str]): 'local', 'docker' or 'auto'; where it is compiled.

    Returns:
        The checkout, built.

    Raises:
        RuntimeError: If the build fails.
    """
    if os.path.exists(os.path.join(spm_root, "spm_bsplinc.mex")):
        return spm_root
    which = resolve_runtime(runtime)
    src = os.path.join(spm_root, "src")
    build = ("CFLAGS=\"$(mkoctfile -p CFLAGS) -std=gnu17\" make PLATFORM=octave install")
    cmd = ["docker", "exec", "-w", src, docker.CONTAINER, "sh", "-c", build] \
        if which == "docker" else ["sh", "-c", build]
    if which == "docker":
        docker.start()
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          cwd=None if which == "docker" else src, timeout=3600)
    if proc.returncode != 0 or not os.path.exists(os.path.join(spm_root, "spm_bsplinc.mex")):
        raise RuntimeError(f"SPM12's MEX build failed ({which}):\n{proc.stderr[-2000:]}")
    return spm_root
