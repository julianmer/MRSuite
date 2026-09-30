####################################################################################################
#                                            pyamares.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-29                                                                              #
#                                                                                                  #
# Purpose: Fitting with pyAMARES (Xu et al., Diagnostics 2024). AMARES is parametric peak          #
#          fitting rather than a linear combination of basis spectra, so instead of a              #
#          basis it takes a prior-knowledge file - peaks with initial values, bounds and           #
#          expression constraints - and its CRLBs come from the same Fisher machinery as           #
#          jMRUI's. The library is one function surface deep, so like TARQUIN it needs no          #
#          Bridge: a hand-written fit maps its result table onto MRSuite's uniform result.         #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import sys
from typing import Any, Dict, Optional

import numpy as np

from mrsuite.bridges.base import Deferred
from mrsuite.core.method import method
from mrsuite.core.result import FitResult


#*************#
#   version   #
#*************#
def _version() -> Optional[str]:
    """pyAMARES' exact version, None when it is not installed."""
    try:
        import pyAMARES
        return pyAMARES.__version__
    except ImportError:
        return None


#*********#
#   fit   #
#*********#
@method("pyAMARES", version=_version, name="AMARES", parallel=True)
def fit(data: Any, prior: str, opts: Optional[Dict[str, Any]] = None) -> FitResult:
    """
    Fit with pyAMARES, as MRSuite's uniform result.

    Args:
        data: NIfTI-MRS or NIfTI-MRS+ of single spectra. A batch is fitted spectrum by
            spectrum.
        prior: pyAMARES' prior-knowledge file (.csv or .xlsx): one column per peak, with
            initial values, bounds and expression constraints. This is AMARES' analogue
            of a basis; multiplet components named like ``Glu``, ``Glu2`` are summed into
            one ``Glu`` in the result.
        opts (Optional[dict]): Options by name. ``method``, ``initialize_with_lm`` and
            ``fit_range`` steer ``fitAMARES``; everything else is passed to
            ``initialize_FID``, e.g. ``{"truncate_initial_points": 2}``.
        workers (int): Processes to spread a batch over (see ``mrsuite.core.parallel``).

    Returns:
        A FitResult, or one per spectrum for a batch. Concentrations are peak amplitudes
        ('raw', plus 'internal' over the tCr peak when the prior has one); 'percentage'
        holds the amplitude CRLBs.

    Raises:
        ValueError: If a spectrum is not a single FID.
        ImportError: If pyAMARES is not installed.
    """
    if np.prod(data.shape) != data.shape[3]:
        raise ValueError(f"pyAMARES fits single spectra, not shape {data.shape}; combine "
                         f"coils and averages first")
    try:
        import pyAMARES
    except ImportError as err:
        raise ImportError("pyAMARES is not installed: pip install pyAMARES") from err

    # the NIfTI-MRS array and pyAMARES' model share numpy's FFT sign convention, so the
    # FID goes over as stored; the carrier puts the prior's absolute ppm on the axis
    # (pinned by the physiology test, as for TARQUIN). The prior's initial amplitudes
    # and fitAMARES' tolerance (from max |FID|) assume data of order one, so the FID is
    # fitted at unit scale and the amplitudes scaled back; pyAMARES' own normalize_fid
    # divides by numpy's max of a complex array, which rotates the phase.
    fid = np.asarray(data[:], complex).reshape(-1)
    scale = float(np.abs(fid).max()) or 1.0
    options = dict(opts or {})
    fitting = {key: options.pop(key) for key in ("method", "initialize_with_lm", "fit_range")
               if key in options}
    if "fit_range" in fitting:                   # the default objective takes no range
        from pyAMARES.kernel.objective_func import objective_range
        fitting["objective_func"] = objective_range
    settings = {"MHz": float(data.spectrometer_frequency[0]), "sw": 1.0 / data.dwelltime,
                "deadtime": 0.0, "carrier": float(data.SpecFreqChemShift), "g_global": False,
                **options}
    params = pyAMARES.initialize_FID(fid / scale, priorknowledgefile=str(prior), **settings)
    fitted = pyAMARES.fitAMARES(params, params.initialParams, ifplot=False, inplace=False,
                                **fitting)
    return _result(fitted, scale)


#************#
#   result   #
#************#
def _result(fitted: Any, scale: float = 1.0) -> FitResult:
    """
    pyAMARES' result table as MRSuite's uniform result.

    ``result_sum`` (multiplets summed by base name) carries per peak the amplitude and its
    absolute SD; ``result_multiplets`` is the fallback when the prior named no peaks to
    sum. The CRLB percentage is taken as SD over amplitude: for a summed multiplet the
    table's own CRLB(%) column is its first component's. ``scale`` undoes the unit-scale
    normalization the fit ran at.

    Raises:
        ValueError: If the fit produced no peaks, rather than returning an empty result.
    """
    table = getattr(fitted, "result_sum", None)
    if table is None:
        table = getattr(fitted, "result_multiplets", None)
    if table is None or not len(table):
        raise ValueError("pyAMARES produced no result table")

    conc = {str(m): scale * float(row["amplitude"]) for m, row in table.iterrows()}
    sd = {str(m): scale * float(row["sd"]) for m, row in table.iterrows()}
    pct = {m: 100.0 * sd[m] / conc[m] if conc[m] else float("nan") for m in conc}

    qc = {}
    if "SNR" in table.columns and np.isfinite(table["SNR"]).any():
        qc["snr_max"] = float(np.nanmax(table["SNR"]))

    concentrations = {"raw": conc}
    tcr = conc.get("tCr") or conc.get("Cr", 0.0) + conc.get("PCr", 0.0)
    if tcr:
        concentrations["internal"] = {m: v / tcr for m, v in conc.items()}
    # the table rather than pyAMARES' whole namespace, which holds a pandas Styler that
    # cannot be pickled back from a worker process
    return FitResult(method="pyamares:AMARES", metabolites=list(conc),
                     concentrations=concentrations,
                     uncertainties={"percentage": pct, "raw": sd}, qc=qc, raw=table)


# pyAMARES is a library with a single entry point, so like TARQUIN there is no bridge,
# but its fit is a pipeline step all the same: ``pyamares.step.fit(prior)``
step = Deferred(sys.modules[__name__])
