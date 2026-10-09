####################################################################################################
#                                            lcmodel.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-08-08                                                                              #
#                                                                                                  #
# Purpose: Fitting with LCModel, through PyLCModel. LCModel is a binary rather than a              #
#          library, so there is no function surface to dispatch onto; PyLCModel takes              #
#          NIfTI-MRS and NIfTI-MRS+ itself, so all that is left here is the mapping onto           #
#          MRSuite's uniform result.                                                               #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import shutil
import sys
from typing import Any, Dict, Optional

from mrsuite.bridges.base import Deferred
from mrsuite.core.basis import as_file
from mrsuite.core.method import method
from mrsuite.core.result import FitResult
from mrsuite.octave import scratch


#***************#
#   to result   #
#***************#
def to_result(parsed: tuple, method: str = "lcmodel:LCModel",
              scaling: str = "raw") -> FitResult:
    """
    Map one parsed LCModel ``.coord`` onto MRSuite's uniform result.

    Args:
        parsed (tuple): What ``coord.read_coord(path, meta=True)`` returned, i.e.
            (metabolites, concentrations, %SD, /Cr+PCr ratios, FWHM, S/N, shift, phase).
        method (str): Recorded on the result.
        scaling (str): Where the concentration column is filed. LCModel reports one
            column whose meaning depends on the fit: unreferenced numbers ('raw'), or
            mM when a water reference enabled its water scaling ('molarity', matching
            where FSL-MRS files water-scaled values).

    Returns:
        FitResult: With the reported and creatine-referenced concentrations, %SD as
        the uncertainty, and LCModel's own FWHM and S/N as QC.
    """
    metabs, concs, crlbs, tcr, fwhm, snr, shift, phase = parsed

    out = FitResult(
        method=method,
        metabolites=list(metabs),
        concentrations={scaling: dict(zip(metabs, (float(c) for c in concs))),
                        "internal": dict(zip(metabs, (float(t) for t in tcr)))},
        uncertainties={"percentage": dict(zip(metabs, (float(c) for c in crlbs)))},
        raw={"shift": shift, "phase": phase, "coord": parsed})

    # LCModel reports these itself, so they are its own numbers rather than ours
    for name, value in (("fwhm", fwhm), ("snr", snr)):
        if value is not None:
            out.qc[name] = float(value)
    return out


#*************#
#   version   #
#*************#
def _version() -> str:
    """PyLCModel's release, which decides how LCModel is driven."""
    from lcmodel_wrapper import __version__
    return f"PyLCModel {__version__}"


#************#
#   binary   #
#************#
def _binary() -> Dict[str, Any]:
    """The LCModel binary, which the provenance records by its bytes: LCModel is what
    fits, not the wrapper. It is the one PyLCModel resolves, which ``LCMODEL_EXEC``
    chooses, so it is the one that ran."""
    from lcmodel_wrapper import binaries
    return {"binary": binaries.resolve_executable()}


#*********#
#   fit   #
#*********#
@method("LCModel", version=_version, runtime=_binary, name="LCModel", batched=True,
        parallel=True)
def fit(data: Any, basis: Any, water: Any = None, ppmlim: tuple = (0.5, 4.2),
        control: Optional[str] = None, **kwargs) -> Any:
    """
    Fit with LCModel, as MRSuite's uniform result.

    Args:
        data: NIfTI-MRS or NIfTI-MRS+. A batch is fitted spectrum by spectrum.
        basis: A Basis (``mrsuite.io.basis``), or a path to an LCModel ``.basis`` file.
        water: Unsuppressed reference, enabling LCModel's water scaling. One per
            spectrum. With it the concentrations are mM and filed under 'molarity';
            without it they are unreferenced and filed under 'raw'.
        ppmlim (tuple): Fit range in ppm.
        control (str): An LCModel control file to fit with, as a study keeps one: complete
            (its nunfil, deltat, hzpppm and the rest), since PyLCModel only sets its basis,
            ppm range and omitted metabolites. By default PyLCModel writes one of its own.
        **kwargs: Passed to ``PyLCModel``, e.g. ``save_path``, ``conj``. The binary is
            chosen with ``LCMODEL_EXEC``, not ``path2exec``, so the provenance can tell it.
            Without a ``save_path`` each call works in a folder of its own, removed
            afterwards: PyLCModel's default is ./tmp with fixed file names, so two fits
            started from one directory at once would overwrite each other's files.
        workers (int): Processes to spread a batch over (see ``mrsuite.core.parallel``).

    Returns:
        A FitResult, or one per spectrum for a batch, each with the data's processing
        history and the fit itself as its provenance.
    """
    from lcmodel_wrapper import PyLCModel

    if "path2exec" in kwargs:
        raise TypeError("choose the LCModel binary with LCMODEL_EXEC instead of path2exec")
    own = "save_path" not in kwargs
    workdir = scratch("lcmodel_") if own else kwargs["save_path"]
    try:
        engine = PyLCModel(path2basis=str(as_file(basis)), ppmlim=ppmlim,
                           control=None if control is None else str(control),
                           **{**kwargs, "save_path": workdir})
        scaling = "raw" if water is None else "molarity"
        return [to_result(report, scaling=scaling) for report in engine.report(data, water)]
    finally:
        if own:
            shutil.rmtree(workdir, ignore_errors=True)


# ``lcmodel.step.fit(basis)``, as the bridges' fits are steps: LCModel is a binary rather
# than a library, so there is no bridge, but its fit is a pipeline step all the same
step = Deferred(sys.modules[__name__])
