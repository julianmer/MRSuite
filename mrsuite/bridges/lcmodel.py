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
import sys
from typing import Any

from mrsuite.bridges.base import Deferred
from mrsuite.core.provenance import entry, file_checksum, history
from mrsuite.core.result import FitResult


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


#*********#
#   fit   #
#*********#
def fit(data: Any, basis: str, water: Any = None, ppmlim: tuple = (0.5, 4.2),
        **kwargs) -> Any:
    """
    Fit with LCModel, as MRSuite's uniform result.

    Args:
        data: NIfTI-MRS or NIfTI-MRS+. A batch is fitted spectrum by spectrum.
        basis (str): Path to an LCModel ``.basis`` file.
        water: Unsuppressed reference, enabling LCModel's water scaling. One per
            spectrum. With it the concentrations are mM and filed under 'molarity';
            without it they are unreferenced and filed under 'raw'.
        ppmlim (tuple): Fit range in ppm.
        **kwargs: Passed to ``PyLCModel``, e.g. ``path2exec``, ``save_path``, ``conj``.

    Returns:
        A FitResult, or one per spectrum for a batch, each with the data's processing
        history and the fit itself as its provenance.
    """
    from lcmodel_wrapper import PyLCModel, __version__

    engine = PyLCModel(path2basis=str(basis), ppmlim=ppmlim, **kwargs)
    scaling = "raw" if water is None else "molarity"
    out = [to_result(report, scaling=scaling) for report in engine.report(data, water)]

    # the binary is what fits, so it is recorded by its bytes, not only by the wrapper
    options = {"basis": file_checksum(basis), "ppmlim": ppmlim, "kwargs": kwargs,
               "binary": file_checksum(engine.path2exec)}
    items = data.list() if hasattr(data, "list") else [data]
    refs = water.list() if hasattr(water, "list") else [water] * len(items)
    for result, nmrs, ref in zip(out, items, refs):
        result.provenance = history(nmrs) + [
            entry("LCModel", f"PyLCModel {__version__}", "LCModel",
                  {**options, "water": ref})]

    return out if hasattr(data, "list") else out[0]


# ``lcmodel.step.fit(basis)``, as the bridges' fits are steps: LCModel is a binary rather
# than a library, so there is no bridge, but its fit is a pipeline step all the same
step = Deferred(sys.modules[__name__])
