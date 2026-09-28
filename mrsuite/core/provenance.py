####################################################################################################
#                                          provenance.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-27                                                                              #
#                                                                                                  #
# Purpose: What MRSuite records about each step, in NIfTI-MRS' own ProcessingApplied format, so    #
#          that a result can be reproduced from its history alone: which program ran, at which     #
#          exact version, with which arguments, through which MRSuite and runtime.                 #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import hashlib
import json
import os
import subprocess
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from nifti_mrs.nifti_mrs import NIFTI_MRS

from nifti_mrs_plus import NIfTI_MRS_Plus

REPO = Path(__file__).resolve().parents[2]

# arrays up to this many elements are recorded as values, larger ones by checksum
_INLINE = 64


#*****************#
#   git version   #
#*****************#
@lru_cache(maxsize=None)
def git_version(path: str) -> Optional[str]:
    """
    The exact state of a git checkout, as ``git describe``, or None if it is not one.

    ``--dirty`` because a result produced from uncommitted changes cannot be reproduced
    from the commit alone, and the history should say so.

    Args:
        path (str): Any folder inside the checkout.

    Returns:
        Optional[str]: e.g. 'v.2.6.0-130-g98b2e35', or 'v.2.6.0-130-g98b2e35-dirty'.
    """
    try:
        out = subprocess.run(["git", "-C", str(path), "describe", "--tags", "--always",
                              "--dirty"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return (out.stdout.strip() or None) if out.returncode == 0 else None


#*********************#
#   mrsuite version   #
#*********************#
def mrsuite_version() -> str:
    """MRSuite's release, and its commit when run from a checkout."""
    from mrsuite import __version__
    commit = git_version(str(REPO))
    return f"{__version__} (git {commit})" if commit else __version__


#**************#
#   checksum   #
#**************#
def checksum(array: Any) -> str:
    """SHA-256 of an array's values, independent of how it is laid out in memory."""
    array = np.ascontiguousarray(array)
    return hashlib.sha256(array.tobytes() + str((array.dtype, array.shape)).encode()).hexdigest()


#*******************#
#   file checksum   #
#*******************#
def file_checksum(path: Any) -> Dict[str, str]:
    """A file by its path and the SHA-256 of its bytes, so a changed file shows."""
    stat = os.stat(path)
    return {"path": str(path),
            "sha256": _sha256(os.path.abspath(path), stat.st_mtime_ns, stat.st_size)}


#************#
#   sha256   #
#************#
@lru_cache(maxsize=None)
def _sha256(path: str, mtime: int, size: int) -> str:
    """A file's SHA-256, read once per version of the file: a method records its basis
    for every spectrum of a batch."""
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


#**************#
#   describe   #
#**************#
def describe(value: Any) -> Any:
    """
    A JSON-serialisable record of an argument, faithful enough to replay the call.

    Small values are recorded as they are; files by their bytes as well as their path;
    arrays and spectra by checksum, shape and, for spectra, their own history, since a
    reference spectrum's processing is part of what produced the result.

    Args:
        value: Any argument a step was given.

    Returns:
        A structure json.dumps accepts.
    """
    if isinstance(value, (str, os.PathLike)) and os.path.isfile(value):
        return file_checksum(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, np.generic):
        return describe(value.item())
    if isinstance(value, NIFTI_MRS):
        return {"nifti_mrs": {"sha256": checksum(value[:]), "shape": list(value.shape),
                              "ProcessingApplied": history(value)}}
    # by type, not by attribute: oct2py's Struct answers every attribute it is asked for
    if isinstance(value, NIfTI_MRS_Plus):
        return {"nifti_mrs_plus": [describe(item) for item in value.list()]}
    if isinstance(value, np.ndarray):
        if value.size <= _INLINE and not np.iscomplexobj(value):
            return value.tolist()
        return {"array": {"sha256": checksum(value), "shape": list(value.shape),
                          "dtype": str(value.dtype)}}
    if isinstance(value, dict):
        return {str(k): describe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [describe(v) for v in value]
    if isinstance(value, Path):
        return describe(str(value))
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


#*************#
#   history   #
#*************#
def history(nifti: Any) -> List[Dict[str, Any]]:
    """A copy of the ProcessingApplied entries a spectrum carries, oldest first."""
    hdr_ext = getattr(nifti, "hdr_ext", None)
    if hdr_ext is None or "ProcessingApplied" not in hdr_ext:
        return []
    return json.loads(json.dumps(hdr_ext["ProcessingApplied"], default=str))


#***********#
#   entry   #
#***********#
def entry(program: str, version: Optional[str], method: str,
          details: Dict[str, Any]) -> Dict[str, str]:
    """
    One ProcessingApplied entry, as NIfTI-MRS defines it.

    Args:
        program (str): The software that did the work - the toolbox, not MRSuite.
        version (Optional[str]): Its exact version; 'unknown' when it cannot be told.
        method (str): What was done, e.g. the toolbox function's name.
        details (Dict[str, Any]): Everything else needed to redo it: the arguments, and
            MRSuite's own version, recorded here because it is what made the call.

    Returns:
        Dict[str, str]: With Time, Program, Version, Method and Details, the last as JSON.
    """
    details = {**details, "mrsuite": mrsuite_version()}
    return {"Time": datetime.now().isoformat(), "Program": program,
            "Version": version or "unknown", "Method": method,
            "Details": json.dumps(describe(details), sort_keys=True)}


#************#
#   append   #
#************#
def append(nifti: NIFTI_MRS, item: Dict[str, str]) -> None:
    """Add an entry to a spectrum's ProcessingApplied, keeping what was there."""
    nifti.add_hdr_field("ProcessingApplied", history(nifti) + [item])
