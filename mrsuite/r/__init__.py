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
#          on purpose, see driver.R.                                                               #
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
from typing import Any, Dict

HERE = os.path.dirname(os.path.abspath(__file__))
DRIVER = os.path.join(HERE, "driver.R")


#***************#
#   available   #
#***************#
def available() -> bool:
    """Whether Rscript is on PATH; spant itself is checked by the first job."""
    return shutil.which("Rscript") is not None


#*********#
#   run   #
#*********#
def run(job: Dict[str, Any], folder: str) -> Dict[str, Any]:
    """
    Run one job through driver.R.

    Args:
        job (Dict[str, Any]): The task and its arguments, as driver.R reads them.
        folder (str): Where the job, its spectra and its result live.

    Returns:
        Dict[str, Any]: What driver.R wrote back.

    Raises:
        RuntimeError: With R's own message, if the job failed.
    """
    path, result = os.path.join(folder, "job.json"), os.path.join(folder, "result.json")
    with open(path, "w") as fh:
        json.dump({**job, "result": result}, fh)
    proc = subprocess.run(["Rscript", DRIVER, path], capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.exists(result):
        raise RuntimeError(f"spant job '{job['task']}' failed:\n{proc.stderr[-2000:]}")
    with open(result) as fh:
        return json.load(fh)


#**********#
#   info   #
#**********#
@lru_cache(maxsize=None)
def info() -> Dict[str, Any]:
    """spant's and R's versions and spant's exported functions, asked once."""
    from mrsuite.octave import scratch

    folder = scratch("mrsuite_r_")
    try:
        out = run({"task": "info"}, folder)
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    out["functions"] = sorted(out["functions"])
    return out
