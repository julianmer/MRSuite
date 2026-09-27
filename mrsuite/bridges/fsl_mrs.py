####################################################################################################
#                                            fsl_mrs.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-08-08                                                                              #
#                                                                                                  #
# Purpose: Bridge onto FSL-MRS's NIfTI-MRS processing functions. FSL-MRS already speaks            #
#          NIfTI-MRS, so no conversion is needed; the bridge adds batching over                    #
#          NIfTI-MRS+, provenance, and the preconditions that keep degenerate input from           #
#          reaching functions that cannot handle it.                                               #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import inspect
import os
from typing import Any, Callable, List, Optional

import numpy as np

from fsl_mrs.utils.preproc import nifti_mrs_proc

from mrsuite.bridges.base import Bridge
from mrsuite.core.provenance import checksum, entry, file_checksum, history
from mrsuite.core.result import FitResult


#**************************************************************************************************#
#                                        Class FSLMRSBridge                                        #
#**************************************************************************************************#
#                                                                                                  #
# Dispatches onto FSL-MRS's NIfTI-MRS processing functions.                                        #
#                                                                                                  #
#**************************************************************************************************#
class FSLMRSBridge(Bridge):
    """Dispatch onto FSL-MRS's NIfTI-MRS processing functions."""

    name = "fsl_mrs"
    program = "FSL-MRS"
    records = False           # every nifti_mrs_proc function writes its own entry

    #**********#
    #   init   #
    #**********#
    def __init__(self):
        super().__init__()

    #*************#
    #   resolve   #
    #*************#
    def _resolve(self, func: str) -> Callable:
        target = getattr(nifti_mrs_proc, func, None)
        # functions only: dir() also returns what the module imports (the NIFTI_MRS
        # class, datetime) and its exception classes, none of which are processing
        # steps. Testing for a function sorts them out without a hand-kept denylist
        # that would go stale as FSL-MRS changes.
        if not inspect.isfunction(target):
            raise AttributeError(
                f"FSL-MRS has no processing function '{func}'. "
                f"Available: {', '.join(self.available())}")
        return target

    #*****************#
    #   skip reason   #
    #*****************#
    def _skip_reason(self, func, args, kwargs):
        """Decline operations that are meaningless on degenerate input.

        Already-combined data can still declare a DIM_COIL of length 1. FSL-MRS's
        copy(remove_dim=...) slices with self[:], which drops that trailing singleton,
        so remove_dim then indexes an axis that is no longer there and raises
        AxisError. Nothing is combined either way, so decline instead.
        """
        if func != "coilcombine":
            return None
        data = args[0] if args else kwargs.get("data")
        if data is None or "DIM_COIL" not in getattr(data, "dim_tags", []):
            return "no DIM_COIL: nothing to combine"
        n = data.shape[data.dim_position("DIM_COIL")]
        if n < 2:
            return f"DIM_COIL has length {n}: already combined"
        return None

    #****************#
    #   to toolbox   #
    #****************#
    def _to_toolbox(self, obj):
        return obj          # FSL-MRS takes NIfTI-MRS directly

    #******************#
    #   from toolbox   #
    #******************#
    def _from_toolbox(self, obj):
        return obj

    #*************#
    #   version   #
    #*************#
    def version(self) -> str:
        import fsl_mrs
        return fsl_mrs.__version__

    #***************#
    #   available   #
    #***************#
    def available(self) -> List[str]:
        return sorted(n for n in dir(nifti_mrs_proc)
                      if not n.startswith("_")
                      and inspect.isfunction(getattr(nifti_mrs_proc, n)))


bridge = FSLMRSBridge()


#****************#
#   references   #
#****************#
def _references(items: List[Any], water: Any) -> List[Any]:
    """Line a water reference up with the spectra it scales.

    One per spectrum, which is how they are acquired. A single reference is not spread
    across a batch: a batch is usually separate acquisitions, and scaling them all by
    one subject's water would be wrong without being visibly wrong.
    """
    if water is None:
        return [None] * len(items)

    refs = water.list() if hasattr(water, "list") else [water]
    if len(refs) != len(items):
        raise ValueError(f"{len(refs)} water references for {len(items)} spectra; "
                         f"give one per spectrum.")
    return refs


#*********#
#   fit   #
#*********#
def fit(data: Any, basis: Any, water: Any = None, ppmlim: tuple = (0.2, 4.2),
        baseline_order: int = 2, method: str = "Newton",
        te: Optional[float] = None, tr: Optional[float] = None, **kwargs) -> Any:
    """
    Fit with FSL-MRS, as MRSuite's uniform result.

    Hand-written rather than dispatched because fit_FSLModel reads the basis off an MRS
    object that ``mrs(basis=...)`` has to build, and the bridge's converter only ever
    sees the data.

    Args:
        data: NIfTI-MRS or NIfTI-MRS+. A batch is fitted spectrum by spectrum.
        basis: A basis file or folder FSL-MRS reads (an LCModel ``.BASIS``, as every
            fitter takes), or an ``fsl_mrs`` basis from ``mrs_io.read_basis``.
        water: Unsuppressed reference, for molarity and molality scaling. Without it the
            result carries raw and internally referenced concentrations only.
        ppmlim (tuple): Fit range in ppm.
        baseline_order (int): Polynomial order of the baseline.
        method (str): 'Newton', 'MH' or 'VB'.
        te (Optional[float]): Echo time in seconds, for water scaling. Read from the
            header when absent.
        tr (Optional[float]): Repetition time in seconds, likewise.
        **kwargs: Passed through to ``fit_FSLModel``.

    Returns:
        A FitResult, or one per spectrum for a batch, each with the data's processing
        history and the fit itself as its provenance.

    Raises:
        ValueError: If water scaling is asked for but no echo or repetition time can be
            found, since the T1 and T2 corrections cannot be applied without them.
    """
    from fsl_mrs.utils import fitting, mrs_io, quantify

    if isinstance(basis, (str, os.PathLike)):
        source = file_checksum(basis)
        basis = mrs_io.read_basis(str(basis))
    else:
        source = {"names": list(basis.names), "sha256": checksum(basis.original_basis_array)}
    options = {"basis": source, "ppmlim": ppmlim, "baseline_order": baseline_order,
               "method": method, "te": te, "tr": tr, "kwargs": kwargs}

    batched = hasattr(data, "list")
    items = data.list() if batched else [data]

    out = []
    for nmrs, ref in zip(items, _references(items, water)):
        mrs = nmrs.mrs(basis=basis, ref_data=None if ref is None else ref[:])
        mrs.processForFitting()
        res = fitting.fit_FSLModel(mrs, method=method, ppmlim=ppmlim,
                                   baseline_order=baseline_order, **kwargs)

        if ref is not None:
            # hdr_ext is an Hdr_Ext, which supports `in` but has no .get
            hdr = getattr(nmrs, "hdr_ext", None)
            def header(field):
                return hdr[field] if hdr is not None and field in hdr else None

            echo = te if te is not None else header("EchoTime")
            rep = tr if tr is not None else header("RepetitionTime")
            if echo is None or rep is None:
                raise ValueError(
                    "water scaling needs an echo and repetition time; the header has "
                    f"EchoTime={header('EchoTime')} and "
                    f"RepetitionTime={header('RepetitionTime')}. Pass te= and tr=.")
            res.calculateConcScaling(mrs, quant_info=quantify.QuantificationInfo(
                echo, rep, mrs.names, mrs.centralFrequency / 1e6))

        result = to_result(res)
        result.provenance = history(nmrs) + [
            entry(bridge.program, bridge.version(), "fit_FSLModel",
                  {**options, "water": ref})]
        out.append(result)

    return out if batched else out[0]


#***************#
#   to result   #
#***************#
def to_result(res: Any) -> FitResult:
    """
    Map an FSL-MRS ``FitRes`` onto MRSuite's uniform result.

    Args:
        res: An ``fsl_mrs.utils.results.FitRes``.

    Returns:
        FitResult: The comparable fields, with ``res`` itself kept as ``raw`` so the
        curves and the sampler output are not lost.
    """
    def collect(get, kinds):
        """Keep the kinds this fit actually has.

        getConc and getUncertainties raise for a scaling that was never calculated -
        molarity needs a water reference, internal needs a reference metabolite - so
        each is probed rather than assumed. Claiming a scaling the fit does not have
        would be worse than omitting it.
        """
        out = {}
        for kind in kinds:
            try:
                out[kind] = dict(zip(res.metabs, (float(v) for v in get(kind))))
            except (ValueError, KeyError, TypeError):
                pass
        return out

    qc = {"mse": float(res.mse)}
    try:
        snr, fwhm = res.getQCParams()
        qc.update(snr=float(np.mean(snr)), fwhm=float(np.mean(fwhm)))
    except (AttributeError, ValueError, TypeError):
        pass                            # runqc=False, or a method that skips it

    return FitResult(
        method=f"fsl_mrs:{res.method}",
        metabolites=list(res.metabs),
        concentrations=collect(lambda s: res.getConc(scaling=s),
                               ("raw", "internal", "molarity", "molality")),
        uncertainties=collect(lambda k: res.getUncertainties(type=k),
                              ("percentage", "raw", "internal", "molarity", "molality")),
        qc=qc,
        raw=res)


#*************#
#   getattr   #
#*************#
def __getattr__(name: str) -> Any:
    """Module-level dispatch, so `from mrsuite.bridges.fsl_mrs import average` works."""
    return getattr(bridge, name)


#*********#
#   dir   #
#*********#
def __dir__() -> List[str]:
    return sorted(set(globals()) | set(bridge.available()))
