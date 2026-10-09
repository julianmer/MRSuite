####################################################################################################
#                                           __init__.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-27                                                                              #
#                                                                                                  #
# Purpose: R for the spant bridge, run as Rscript on driver.R: a job goes in as JSON with its      #
#          spectra as NIfTI-MRS files, and the result comes back the same way. Out of process      #
#          on purpose, see driver.R. Either a local R or the container built from the dockerfile   #
#          beside this file, which mounts the repo at its own path so every path still holds.      #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import json
import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from mrsuite.octave import docker

HERE = os.path.dirname(os.path.abspath(__file__))
DRIVER = os.path.join(HERE, "driver.R")
IMAGE = "mrsuite-r"

# "local" uses the Rscript on PATH, "docker" the image built from the dockerfile beside
# this file, and "auto" prefers local and falls back. Settable per call, or via the
# environment.
RUNTIME = os.environ.get("MRSUITE_R_RUNTIME", "auto")


#*********************#
#   resolve runtime   #
#*********************#
def resolve_runtime(runtime: Optional[str] = None) -> str:
    """Settle on 'local' or 'docker', raising if neither can be honoured."""
    runtime = runtime or RUNTIME
    local = shutil.which("Rscript") is not None
    if runtime not in ("auto", "local", "docker"):
        raise ValueError(f"unknown R runtime '{runtime}'; expected 'local', 'docker' or 'auto'")
    if runtime == "local" or (runtime == "auto" and local):
        if not local:
            raise RuntimeError("no Rscript on PATH; install R, or use runtime='docker'")
        return "local"
    if not docker.available():
        raise RuntimeError("no R: install it, or start Docker for the containerised runtime")
    return "docker"


#***************#
#   available   #
#***************#
def available() -> bool:
    """Whether R can be run, locally or in Docker; spant is checked by the first job."""
    try:
        resolve_runtime()
        return True
    except RuntimeError:
        return False


#*********#
#   run   #
#*********#
def run(job: Dict[str, Any], folder: str, runtime: Optional[str] = None) -> Dict[str, Any]:
    """
    Run one job through driver.R.

    Args:
        job (Dict[str, Any]): The task and its arguments, as driver.R reads them.
        folder (str): Where the job, its spectra and its result live; inside the repo, so
            that the container sees it at the same path.
        runtime (Optional[str]): 'local', 'docker' or 'auto'. Defaults to ``RUNTIME``.

    Returns:
        Dict[str, Any]: What driver.R wrote back.

    Raises:
        RuntimeError: With R's own message, if the job failed.
    """
    path, result = os.path.join(folder, "job.json"), os.path.join(folder, "result.json")
    with open(path, "w") as fh:
        json.dump({**job, "result": result}, fh)
    command = ["Rscript", DRIVER, path]
    if resolve_runtime(runtime) == "docker":
        docker.build(image=IMAGE, dockerfile=Path(HERE) / "dockerfile")
        # as the host user, so what R writes stays the caller's; a bare uid has no HOME
        command = ["docker", "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}",
                   "-e", "HOME=/tmp", "-v", f"{docker.REPO}:{docker.REPO}", IMAGE] + command
    proc = subprocess.run(command, capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.exists(result):
        raise RuntimeError(f"spant job '{job['task']}' failed:\n{proc.stderr[-2000:]}")
    with open(result) as fh:
        return json.load(fh)


#**********#
#   info   #
#**********#
def info(runtime: Optional[str] = None) -> Dict[str, Any]:
    """spant's and R's versions and spant's exported functions, asked once per runtime."""
    out = _asked("info", resolve_runtime(runtime))
    return {**out, "functions": sorted(out["functions"])}


#***********#
#   about   #
#***********#
def about(runtime: Optional[str] = None) -> Dict[str, Any]:
    """spant's functions' arguments, a default as the R code that states it, and their help
    pages, asked once per runtime: apart from ``info``, which every spant call needs and
    which reading 400 help pages would slow by seconds."""
    return _asked("about", resolve_runtime(runtime))


#***********#
#   asked   #
#***********#
@lru_cache(maxsize=None)
def _asked(task: str, runtime: str) -> Dict[str, Any]:
    from mrsuite.octave import scratch

    folder = scratch("mrsuite_r_")
    try:
        return run({"task": task}, folder, runtime)
    finally:
        shutil.rmtree(folder, ignore_errors=True)
