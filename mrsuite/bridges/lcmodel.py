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
from typing import Any

from mrsuite.core.result import FitResult


#***************#
#   to result   #
#***************#
def to_result(parsed: tuple, method: str = "lcmodel:LCModel") -> FitResult:
    """
    Map one parsed LCModel ``.coord`` onto MRSuite's uniform result.

    Args:
        parsed (tuple): What ``coord.read_coord(path, meta=True)`` returned, i.e.
            (metabolites, concentrations, %SD, /Cr+PCr ratios, FWHM, S/N, shift, phase).
        method (str): Recorded on the result.

    Returns:
        FitResult: With raw and creatine-referenced concentrations, %SD as the
        uncertainty, and LCModel's own FWHM and S/N as QC.
    """
    metabs, concs, crlbs, tcr, fwhm, snr, shift, phase = parsed

    out = FitResult(
        method=method,
        metabolites=list(metabs),
        concentrations={"raw": dict(zip(metabs, (float(c) for c in concs))),
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
        water: Unsuppressed reference, enabling LCModel's water scaling. One per spectrum.
        ppmlim (tuple): Fit range in ppm.
        **kwargs: Passed to ``PyLCModel``, e.g. ``path2exec``, ``save_path``, ``conj``.

    Returns:
        A FitResult, or one per spectrum for a batch.
    """
    from lcmodel_wrapper import PyLCModel

    engine = PyLCModel(path2basis=basis, ppmlim=ppmlim, **kwargs)
    out = [to_result(report) for report in engine.report(data, water)]

    return out if hasattr(data, "list") else out[0]
