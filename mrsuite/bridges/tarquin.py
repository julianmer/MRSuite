####################################################################################################
#                                            tarquin.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-28                                                                              #
#                                                                                                  #
# Purpose: Fitting with TARQUIN, through PyTARQUIN. Like LCModel it is a binary rather than a      #
#          library, so there is no function surface to dispatch onto; PyTARQUIN takes NIfTI-MRS    #
#          itself, writes TARQUIN's input and reads its results, so all that is left here is the   #
#          mapping onto MRSuite's uniform result.                                                  #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import sys
from functools import lru_cache
from typing import Any, Dict, Optional

import numpy as np

from mrsuite.bridges.base import Deferred
from mrsuite.core.basis import as_file
from mrsuite.core.method import method
from mrsuite.core.result import FitResult


#***************#
#   to result   #
#***************#
def to_result(report: Any, method: str = "tarquin:TARQUIN") -> FitResult:
    """
    Map one PyTARQUIN report onto MRSuite's uniform result.

    Args:
        report (tarquin_wrapper.Report): One fit, as PyTARQUIN read TARQUIN's results.
        method (str): Recorded on the result.

    Returns:
        FitResult: TARQUIN's table, whose units say the scaling - 'au' unreferenced, filed
        under 'raw', or 'mM' water-scaled, filed under 'molarity' as LCModel and FSL-MRS
        file theirs - with the combined signals (TNAA, TCr, ...) after the single ones, the
        creatine-referenced values, %SD and SD as uncertainties, TARQUIN's fit quality as
        QC under snake_case names ('snr_max', 'metab_fwhm_hz', 'q'), and the report as raw.
    """
    scaling = "molarity" if report.units == "mM" else "raw"
    conc = dict(zip(report.metabolites, report.concentrations))
    concentrations = {scaling: conc}
    tcr = conc.get("Cr", 0.0) + conc.get("PCr", 0.0)
    if tcr:
        concentrations["internal"] = {m: v / tcr for m, v in conc.items()}
    quality = report.sections.get("fit_quality", {})
    return FitResult(method=method, metabolites=list(report.metabolites),
                     concentrations=concentrations,
                     uncertainties={"percentage": dict(zip(report.metabolites, report.percent_sd)),
                                    scaling: dict(zip(report.metabolites, report.sd))},
                     qc={k: v for k, v in quality.items() if isinstance(v, float)}, raw=report)


#****************#
#   executable   #
#****************#
def _executable() -> str:
    """The TARQUIN binary PyTARQUIN resolves, which ``TARQUIN_EXEC`` chooses, so it is the
    one that ran."""
    from tarquin_wrapper import binaries
    return binaries.resolve_executable()


#*************#
#   version   #
#*************#
def _version() -> Optional[str]:
    """The binary's release, asked once per binary."""
    return _release(_executable())


#*************#
#   release   #
#*************#
@lru_cache(maxsize=None)
def _release(path: str) -> Optional[str]:
    """The release the binary names when it starts, as PyTARQUIN reads it."""
    from tarquin_wrapper import binaries
    return binaries.version(path)


#************#
#   binary   #
#************#
def _binary() -> Dict[str, Any]:
    """The binary that ran, recorded like LCModel's: TARQUIN is what fits, and PyTARQUIN
    decides how it is driven."""
    from tarquin_wrapper import __version__
    return {"binary": _executable(), "wrapper": f"PyTARQUIN {__version__}"}


#*********#
#   fit   #
#*********#
@method("TARQUIN", version=_version, runtime=_binary, name="TARQUIN", parallel=True)
def fit(data: Any, basis: Any = None, water: Any = None, te: Optional[float] = None,
        opts: Optional[Dict[str, Any]] = None) -> FitResult:
    """
    Fit with TARQUIN, as MRSuite's uniform result.

    Args:
        data: NIfTI-MRS or NIfTI-MRS+ of single spectra. A batch is fitted spectrum by
            spectrum.
        basis: A Basis (``mrsuite.io.basis``), or an LCModel ``.basis``, as every fitter
            takes. TARQUIN does not resample: its sampling frequency must match the
            data's. By default TARQUIN rescales an LCModel basis to its sIns or Scyllo
            signal and stops if it has neither; ``{"rescale_lcm_basis": False}`` turns
            that off. None uses TARQUIN's default internal basis, ``1h_brain_glth``,
            simulated for the sequence at the data's echo time; ``{"int_basis":
            "1h_brain"}`` picks another.
        water: Unsuppressed reference, one per spectrum, enabling TARQUIN's water
            scaling. With it the concentrations are mM and filed under 'molarity';
            without it they are unreferenced and filed under 'raw'.
        te (Optional[float]): Echo time in seconds; read from the header when absent.
        opts (Optional[dict]): TARQUIN's command line options by name, e.g.
            ``{"pul_seq": "slaser", "max_iters": 50}``; anything not given is TARQUIN's
            own default - PRESS, and a start point it picks itself, skipping about the
            first 10 ms of the FID to keep broad baseline out of the fit. The binary is
            chosen with ``TARQUIN_EXEC``, so the provenance can tell it; without it
            PyTARQUIN finds one (PATH, its cache, its release, a container).
        workers (int): Processes to spread a batch over (see ``mrsuite.core.parallel``).

    Returns:
        A FitResult, or one per spectrum for a batch, each with the data's processing
        history and the fit itself as its provenance.

    Raises:
        ValueError: If a spectrum is not a single FID, or the internal basis is asked for
            without an echo time to simulate it at.
    """
    from tarquin_wrapper import PyTARQUIN

    # PyTARQUIN takes every non-spectral dimension for a batch; here a spectrum is one FID
    for item in (data, water):
        if item is not None and np.prod(item.shape) != item.shape[3]:
            raise ValueError(f"TARQUIN fits single spectra, not shape {item.shape}; combine "
                             f"coils and averages first")
    echo = te if te is not None else _header(data, "EchoTime")
    if basis is None and echo is None:
        raise ValueError("TARQUIN simulates its internal basis at the data's echo time, which "
                         "the header does not state; pass te= or a basis")

    # conj: PyTARQUIN hands out NIfTI-MRS as indexed, the conjugate of what TARQUIN reads
    engine = PyTARQUIN(path2basis=as_file(basis), opts=opts, echo_time=te, conj=True)
    return to_result(engine.report(data, water)[0])


#************#
#   header   #
#************#
def _header(item: Any, field: str) -> Optional[float]:
    """A header field as a number, or None when the header does not state it."""
    hdr = getattr(item, "hdr_ext", None)
    return float(hdr[field]) if hdr is not None and field in hdr else None


# TARQUIN is a binary rather than a library, so like LCModel there is no bridge, but
# its fit is a pipeline step all the same: ``tarquin.step.fit(basis)``
step = Deferred(sys.modules[__name__])
