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
import ast
import functools
import importlib.util
import inspect
import os
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

try:
    from fsl_mrs.utils.preproc import nifti_mrs_proc
    from fsl_mrs.utils.preproc import mrsi as mrsi_proc
except ModuleNotFoundError as err:
    if (err.name or "").partition(".")[0] != "fsl_mrs":   # installed, but broken
        raise
    raise ImportError("FSL-MRS is not installed: conda install -c conda-forge -c "
                      "https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/ fsl_mrs") from err

from mrsuite.bridges.base import Bridge, defaults
from mrsuite.core.basis import as_file
from mrsuite.core.method import method
from mrsuite.core.result import FitResult

# QuantificationInfo's water scaling options that are set after it is built, not passed in
SET_AFTER = ("fractions", "densities", "add_corr", "h2o_limits")


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
        target = _source(func)
        # functions only: dir() also returns what the module imports (the NIFTI_MRS
        # class, datetime) and its exception classes, none of which are processing
        # steps. Testing for a function sorts them out without a hand-kept denylist
        # that would go stale as FSL-MRS changes.
        if not inspect.isfunction(target):
            raise AttributeError(
                f"FSL-MRS has no processing function '{func}'. "
                f"Available: {', '.join(self.available())}")
        given = _proc_defaults().get(func)
        return _defaulted(target, given) if given else target

    #*************#
    #   records   #
    #*************#
    def _records(self, func: str) -> bool:
        """The MRSI functions write no entry of their own; nifti_mrs_proc's do."""
        return func in MRSI

    #*****************#
    #   skip reason   #
    #*****************#
    def _skip_reason(self, func, args, kwargs):
        """Decline operations that are meaningless on degenerate input.

        Already-combined data can still declare a DIM_COIL of length 1. FSL-MRS's
        copy(remove_dim=...) slices with self[:], which drops that trailing singleton,
        so remove_dim then indexes an axis that is no longer there and raises
        AxisError; remove_unlike on data without averages fails with an IndexError.
        Nothing is combined or removed either way, so decline instead.
        """
        tag, doing = {"coilcombine": ("DIM_COIL", "combine"),
                      "remove_unlike": ("DIM_DYN", "remove")}.get(func, (None, None))
        if tag is None:
            return None
        data = args[0] if args else kwargs.get("data")
        if data is None or tag not in getattr(data, "dim_tags", []):
            return f"no {tag}: nothing to {doing}"
        n = data.shape[data.dim_position(tag)]
        if n < 2:
            return f"{tag} has length {n}: nothing to {doing}"
        return None

    #**************#
    #   declined   #
    #**************#
    def _declined(self, func, data, kwargs):
        """remove_unlike gives the spectra kept and those removed: none removed."""
        return (data, None) if func == "remove_unlike" else data

    #****************#
    #   to toolbox   #
    #****************#
    def _to_toolbox(self, obj):
        return _fsl(obj)

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
        """The single-voxel processing functions, and the MRSI ones (``MRSI``)."""
        return sorted({n for n in dir(nifti_mrs_proc)
                       if not n.startswith("_") and inspect.isfunction(getattr(nifti_mrs_proc, n))}
                      | set(MRSI))

    #***********#
    #   steps   #
    #***********#
    def steps(self) -> List[str]:
        """The functions that take the spectra first, leaving out helpers such as
        first_index and update_processing_prov."""
        return [name for name in self.available() if next(iter(
            inspect.signature(_source(name)).parameters), "").startswith("data")]


bridge = FSLMRSBridge()

# the MRSI functions of fsl_mrs.utils.preproc.mrsi, which take a spatial NIfTI-MRS whole:
# phase and frequency across the voxels, lipids out by L2 regularisation
MRSI = ("lipid_removal_l2", "mrsi_freq_align", "mrsi_phase_corr")


#************#
#   source   #
#************#
def _source(func: str) -> Any:
    """The FSL-MRS function behind a name: an MRSI one from its module, the rest from
    nifti_mrs_proc; None where there is none."""
    return getattr(mrsi_proc if func in MRSI else nifti_mrs_proc, func, None)


#*******************#
#   proc defaults   #
#*******************#
@lru_cache(maxsize=None)
def _proc_defaults() -> Dict[str, Dict[str, Any]]:
    """
    The defaults FSL-MRS's command line, fsl_mrs_proc, gives the arguments its processing
    functions leave without one, e.g. phase_correct's ppmlim, 2.8-3.2 ppm; by function.
    Read from fsl_mrs_proc's source - each subcommand's options, the handler it sets, and
    the option each handler hands each function argument - so a new FSL-MRS brings its
    own. A function its handlers give different defaults gets none; a script laid out
    otherwise than this reads (FSL-MRS 2.5) gives none at all, rather than stop every call.
    """
    try:
        return _read_proc_defaults()
    except (AttributeError, KeyError, IndexError, TypeError, ValueError, SyntaxError):
        return {}


#************************#
#   read proc defaults   #
#************************#
def _read_proc_defaults() -> Dict[str, Dict[str, Any]]:
    source = importlib.util.find_spec("fsl_mrs.scripts.fsl_mrs_proc").origin
    tree = ast.parse(Path(source).read_text())
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    calls = [n for n in ast.walk(functions["main"]) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)]
    parser = {}                                        # a parser or its group -> the parser
    for n in ast.walk(functions["main"]):
        if isinstance(n, ast.Assign) and n.value in calls:
            made, of = n.targets[0].id, n.value.func.value.id
            if n.value.func.attr == "add_parser":
                parser[made] = made
            elif n.value.func.attr == "add_argument_group":
                parser[made] = parser.get(of, of)
    options: Dict[str, Dict[str, Any]] = {}            # parser -> option -> its default
    handler = {}                                       # handler -> its parser
    for call in calls:
        owner, keywords = parser.get(call.func.value.id), {k.arg: k.value for k in call.keywords}
        if owner and call.func.attr == "add_argument" and "default" in keywords:
            flags = [a.value for a in call.args if isinstance(a, ast.Constant)]
            dest = ast.literal_eval(keywords["dest"]) if "dest" in keywords else next(
                (f.lstrip("-").replace("-", "_") for f in flags if f.startswith("--")),
                flags[0] if flags else None)
            try:
                options.setdefault(owner, {})[dest] = ast.literal_eval(keywords["default"])
            except ValueError:                         # not a plain value
                pass
        elif owner and call.func.attr == "set_defaults" and "func" in keywords:
            handler[keywords["func"].id] = owner
    named = {a.asname or a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
             for a in n.names if a.name == "nifti_mrs_proc"}    # e.g. 'as preproc'
    found: Dict[str, Dict[str, Any]] = {}
    clash = set()
    for name, owner in handler.items():
        for call in ast.walk(functions[name]):
            if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and getattr(call.func.value, "id", None) in named):
                continue
            target = getattr(nifti_mrs_proc, call.func.attr, None)
            if not inspect.isfunction(target):
                continue
            parameters = inspect.signature(target).parameters
            passed = [*zip(parameters, call.args), *((k.arg, k.value) for k in call.keywords)]
            for argument, value in passed:
                option = value.slice.value if isinstance(value, ast.Subscript) and \
                    getattr(value.value, "id", None) == "args" and \
                    isinstance(value.slice, ast.Constant) else None
                default = options.get(owner, {}).get(option)
                if default is None or parameters[argument].default is not inspect.Parameter.empty:
                    continue
                mine = found.setdefault(call.func.attr, {})
                if mine.get(argument, default) != default:
                    clash.add((call.func.attr, argument))
                mine[argument] = default
    for function, argument in clash:
        del found[function][argument]
    return found


#***************#
#   defaulted   #
#***************#
def _defaulted(target: Callable, given: Dict[str, Any]) -> Callable:
    """A function that takes fsl_mrs_proc's defaults for what it is not given."""
    signature = defaults(inspect.signature(target), given)
    taken = {k: p.default for k, p in signature.parameters.items() if k in given
             and p.default is not inspect.Parameter.empty}

    @functools.wraps(target)
    def call(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        return target(*bound.args, **bound.kwargs)

    call.__signature__ = signature
    if taken:
        call.__doc__ = (target.__doc__ or "").rstrip() + "\n\nDefaults from FSL-MRS's " \
            "command line (fsl_mrs_proc): " + ", ".join(f"{k}={v!r}" for k, v in taken.items())
    return call


#***********#
#   basis   #
#***********#
@lru_cache(maxsize=None)
def _basis(path: str, mtime: float) -> Any:
    """A basis file, read once per process and version of the file: a batch is fitted
    spectrum by spectrum, and reading the basis for each would dominate a quick fit."""
    from fsl_mrs.utils import mrs_io
    return mrs_io.read_basis(path)


#*********#
#   fsl   #
#*********#
def _fsl(data: Any) -> Any:
    """Any NIfTI-MRS as FSL-MRS's own class, which adds what FSL-MRS calls on it (such as
    ``mrs()``); spectra made by other methods and simulators are plain NIfTI-MRS."""
    from fsl_mrs.core.nifti_mrs import NIFTI_MRS
    return data if isinstance(data, NIFTI_MRS) else NIFTI_MRS(data)


#*********#
#   fit   #
#*********#
@method("FSL-MRS", version=lambda: bridge.version(), name="fit_FSLModel", parallel=True)
def fit(data: Any, basis: Any, water: Any = None, ppmlim: tuple = (0.2, 4.2),
        baseline_order: int = 2, method: str = "Newton",
        te: Optional[float] = None, tr: Optional[float] = None,
        quant: Optional[dict] = None, **kwargs) -> Any:
    """
    Fit with FSL-MRS, as MRSuite's uniform result.

    Hand-written rather than dispatched because fit_FSLModel reads the basis off an MRS
    object that ``mrs(basis=...)`` has to build, and the bridge's converter only ever
    sees the data.

    Args:
        data: NIfTI-MRS or NIfTI-MRS+. A batch is fitted spectrum by spectrum.
        basis: A Basis (``mrsuite.io.basis``), a basis file or folder FSL-MRS reads (an
            LCModel ``.BASIS``, as every fitter takes), or an ``fsl_mrs`` basis from
            ``mrs_io.read_basis``.
        water: Unsuppressed reference, for molarity and molality scaling, one per
            spectrum. Without it the result carries raw and internally referenced
            concentrations only.
        ppmlim (tuple): Fit range in ppm.
        baseline_order (int): Polynomial order of the baseline.
        method (str): 'Newton', 'MH' or 'VB'.
        te (Optional[float]): Echo time in seconds, for water scaling. Read from the
            header when absent.
        tr (Optional[float]): Repetition time in seconds, likewise.
        quant (Optional[dict]): FSL-MRS's own water scaling options, by the names of
            ``QuantificationInfo``: its keywords (``t1``, ``t2``, ``fa``,
            ``water_ref_metab``, ...) and what it sets (``fractions``, ``densities``,
            ``add_corr``, ``h2o_limits``). Anything not given is FSL-MRS's default.
            Used only with ``water``, as are ``te`` and ``tr``.
        **kwargs: Passed through to ``fit_FSLModel``.
        workers (int): Processes to spread a batch over (see ``mrsuite.core.parallel``).

    Returns:
        A FitResult, or one per spectrum for a batch, each with the data's processing
        history and the fit itself as its provenance.

    Raises:
        ValueError: If water scaling is asked for but no echo or repetition time can be
            found, since the T1 and T2 corrections cannot be applied without them.
    """
    from fsl_mrs.utils import fitting, quantify

    basis = as_file(basis)
    if isinstance(basis, (str, os.PathLike)):
        basis = _basis(os.path.abspath(basis), os.path.getmtime(basis))

    mrs = _fsl(data).mrs(basis=basis, ref_data=None if water is None else water[:])
    mrs.processForFitting()
    res = fitting.fit_FSLModel(mrs, method=method, ppmlim=ppmlim,
                               baseline_order=baseline_order, **kwargs)

    if water is not None:
        # hdr_ext is an Hdr_Ext, which supports `in` but has no .get
        hdr = getattr(data, "hdr_ext", None)
        def header(field):
            return hdr[field] if hdr is not None and field in hdr else None

        echo = te if te is not None else header("EchoTime")
        rep = tr if tr is not None else header("RepetitionTime")
        if echo is None or rep is None:
            raise ValueError(
                "water scaling needs an echo and repetition time; the header has "
                f"EchoTime={header('EchoTime')} and "
                f"RepetitionTime={header('RepetitionTime')}. Pass te= and tr=.")
        quant = dict(quant or {})
        set_after = {k: quant.pop(k) for k in SET_AFTER if k in quant}
        info = quantify.QuantificationInfo(echo, rep, mrs.names, mrs.centralFrequency / 1e6,
                                           **quant)
        for name, value in set_after.items():
            if name in ("fractions", "densities"):   # validated by their own setters
                getattr(info, f"set_{name}")(value)
            else:
                setattr(info, name, value)
        try:
            res.calculateConcScaling(mrs, quant_info=info)
        except res.QuantificationError as err:
            # the reference metabolite fitted to zero: a failed fit keeps its raw values;
            # the same error for a water reference with no integral stays an error
            if "Metabolite reference" not in str(err):
                raise
            warnings.warn(f"FSL-MRS could not scale this fit ({err}); it keeps its raw values")

    return to_result(res)


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


#************#
#   curves   #
#************#
def curves(res: Any) -> Dict[str, np.ndarray]:
    """What FSL-MRS fitted, as its result keeps it: the prediction, baseline and residual
    are FIDs on the ``FitRes``, laid along the ppm axis FSL-MRS builds from the bandwidth
    and centre frequency it kept, and cut to its fit range."""
    from fsl_mrs.utils.misc import H2O_PPM_TO_TMS, FIDToSpec, calculateAxes

    ppm = calculateAxes(res.bandwidth, res.hzperppm * 1e6, len(res.pred),
                        H2O_PPM_TO_TMS)["ppmshift"]
    keep = (ppm >= res.ppmlim[0]) & (ppm <= res.ppmlim[1])

    def spectrum(fid):
        return np.real(FIDToSpec(fid))[keep]

    return {"ppm": ppm[keep], "data": spectrum(res.pred + res.residuals),
            "fit": spectrum(res.pred), "baseline": spectrum(res.baseline)}


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
