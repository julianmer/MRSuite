####################################################################################################
#                                             base.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-08-08                                                                              #
#                                                                                                  #
# Purpose: Shared machinery for the toolbox bridges. A bridge exposes an external                  #
#          toolbox's whole function surface on MRSuite's data types, without a                     #
#          hand-written wrapper per function: it resolves the name, converts arguments,            #
#          calls the toolbox, converts results back, and records the call in provenance.           #
#          Bridges are organised by toolbox rather than by processing/analysis, because            #
#          that is the split the code actually has - Osprey spans loading, processing,             #
#          fitting and quantification.                                                             #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import sys
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from nifti_mrs.nifti_mrs import NIFTI_MRS

from nifti_mrs_plus import NIfTI_MRS_Plus
from mrsuite.core import provenance


#**************************************************************************************************#
#                                           Class Bridge                                           #
#**************************************************************************************************#
#                                                                                                  #
# Dynamic dispatch onto an external toolbox. Any function the toolbox                              #
# exposes is reachable, whether or not MRSuite knows about it.                                     #
#                                                                                                  #
#**************************************************************************************************#
class Bridge(ABC):
    """Dynamic dispatch onto an external toolbox.

    Attribute access returns a callable, so any function the toolbox exposes is reachable
    whether or not MRSuite knows about it. Unknown names raise ``AttributeError`` at
    lookup, not a confusing failure inside the toolbox.

    Every call that returns spectra records itself in their ProcessingApplied, as
    FSL-MRS' own functions do: the toolbox and its exact version, the function, every
    argument, and what else it ran on, so that the history alone reproduces the result.

    Attributes:
        name (str): Module name, as in ``mrsuite.<name>``.
        program (str): The toolbox's own name, credited in ProcessingApplied.
        records (bool): False for a toolbox that already records its own steps.
    """

    name: str = "bridge"
    program: str = "bridge"
    records: bool = True

    #**********#
    #   init   #
    #**********#
    def __init__(self):
        self._skipped: List[str] = []

    #*************#
    #   resolve   #
    #*************#
    @abstractmethod
    def _resolve(self, func: str) -> Callable:
        """Return the toolbox callable for ``func``, or raise AttributeError."""

    #****************#
    #   to toolbox   #
    #****************#
    @abstractmethod
    def _to_toolbox(self, obj: NIFTI_MRS) -> Any:
        """Convert one NIfTI-MRS object into the toolbox's own representation."""

    #******************#
    #   from toolbox   #
    #******************#
    @abstractmethod
    def _from_toolbox(self, obj: Any) -> Any:
        """Convert a toolbox result back, leaving non-MRS results untouched."""

    #***************#
    #   available   #
    #***************#
    def available(self) -> List[str]:
        """Names this bridge can dispatch, for discovery and error messages."""
        return []

    #*************#
    #   version   #
    #*************#
    def version(self) -> Optional[str]:
        """The toolbox's exact version, for provenance; None when it cannot be told."""
        return None

    #*************#
    #   runtime   #
    #*************#
    def runtime(self) -> Dict[str, Any]:
        """What else a call's result depends on, e.g. the interpreter that ran it."""
        return {}

    #*****************#
    #   skip reason   #
    #*****************#
    def _skip_reason(self, func: str, args, kwargs) -> Optional[str]:
        """Why ``func`` would be a no-op on this input, or None to go ahead.

        Some toolbox functions cannot handle degenerate input and fail with an obscure
        error rather than declining politely - combining a single coil, aligning a
        single average. The operation is genuinely meaningless there, so the data is
        returned untouched with an explanation. Overridden per bridge; the default
        never skips.
        """
        return None

    #*************#
    #   getattr   #
    #*************#
    def __getattr__(self, func: str) -> Callable:
        if func.startswith("_"):
            raise AttributeError(func)
        target = self._resolve(func)          # raises AttributeError if unknown

        def call(*args, **kwargs):
            return self._invoke(func, target, args, kwargs)

        call.__name__ = func
        call.__doc__ = getattr(target, "__doc__", None) or \
            f"{self.name} function '{func}', dispatched by MRSuite."
        return call

    #**********#
    #   step   #
    #**********#
    @property
    def step(self) -> "Deferred":
        """The same functions as pipeline steps: bind the arguments, take the data later.

        ``bridge.average(data, dim='DIM_DYN')`` runs now; ``bridge.step.average(
        dim='DIM_DYN')`` hands back something a pipeline can call with the data.
        """
        return Deferred(self)

    #*********#
    #   dir   #
    #*********#
    def __dir__(self):
        return sorted(set(super().__dir__()) | set(self.available()))

    #************#
    #   invoke   #
    #************#
    def _invoke(self, func, target, args, kwargs):
        """Batch over NIfTI-MRS+ if present, otherwise call once."""
        self._skipped = []
        batched = [a for a in list(args) + list(kwargs.values())
                   if isinstance(a, NIfTI_MRS_Plus)]
        if batched:
            n = len(batched[0])
            for b in batched[1:]:
                if len(b) != n:
                    raise ValueError(
                        f"{self.name}.{func}: batched arguments have different lengths "
                        f"({n} vs {len(b)}); they must line up item for item.")
            results = [self._call_one(func, target, self._pick(args, i),
                                      self._pick(kwargs, i)) for i in range(n)]
            return self._regroup(results)
        return self._call_one(func, target, args, kwargs)

    #**********#
    #   pick   #
    #**********#
    @staticmethod
    def _pick(container, i):
        """Take item i out of any batched argument, passing everything else through."""
        def one(v):
            return v[i] if isinstance(v, NIfTI_MRS_Plus) else v
        if isinstance(container, dict):
            return {k: one(v) for k, v in container.items()}
        return tuple(one(v) for v in container)

    #**************#
    #   call one   #
    #**************#
    def _call_one(self, func, target, args, kwargs):
        reason = self._skip_reason(func, args, kwargs)
        if reason is not None:
            self._skipped.append(reason)
            return args[0] if args else None

        source = next((a for a in list(args) + list(kwargs.values())
                       if isinstance(a, NIFTI_MRS)), None)
        out = target(*(self._to_toolbox(a) if isinstance(a, NIFTI_MRS) else a for a in args),
                     **{k: self._to_toolbox(v) if isinstance(v, NIFTI_MRS) else v
                        for k, v in kwargs.items()})
        outs = tuple(self._from_toolbox(o) for o in out) if isinstance(out, tuple) \
            else (self._from_toolbox(out),)

        spectra = [o for o in outs if isinstance(o, NIFTI_MRS) and o is not source]
        if source is not None and spectra:
            # the data is what the history belongs to; everything else is an argument
            item = provenance.entry(self.program, self.version(), func,
                                    {"args": args[1:], "kwargs": kwargs, **self.runtime()})
            for spectrum in spectra:
                self._keep_header(source, spectrum)
                if self.records:
                    provenance.append(spectrum, item)
        return outs if isinstance(out, tuple) else outs[0]

    #*****************#
    #   keep header   #
    #*****************#
    @staticmethod
    def _keep_header(source: NIFTI_MRS, out: NIFTI_MRS) -> None:
        """Carry the input's header fields over to the output.

        A toolbox working on its own struct hands back the samples and little else, so
        the echo time, the ppm reference and the processing history would be lost at
        the first step. Fields the output already states win, and the dimension fields
        are left to the output, since a step may have removed a dimension.
        """
        for key, value in source.hdr_ext.to_dict().items():
            if not key.startswith("dim_") and key not in out.hdr_ext:
                out.add_hdr_field(key, value)

    #*************#
    #   regroup   #
    #*************#
    def _regroup(self, results):
        """Rebuild batched results, keeping tuple-returning functions tuple-shaped."""
        if results and isinstance(results[0], tuple):
            width = len(results[0])
            return tuple(self._wrap([r[j] for r in results]) for j in range(width))
        return self._wrap(results)

    #**********#
    #   wrap   #
    #**********#
    @staticmethod
    def _wrap(items):
        if not items or not all(isinstance(i, NIFTI_MRS) for i in items):
            return items                       # non-MRS output: hand back the raw list
        return NIfTI_MRS_Plus(items)           # picks up each spectrum's ProcessingApplied


#**************************************************************************************************#
#                                          Class Deferred                                          #
#**************************************************************************************************#
#                                                                                                  #
# A bridge's functions as pipeline steps, with their arguments bound.                              #
#                                                                                                  #
#**************************************************************************************************#
class Deferred:
    """A bridge's functions as pipeline steps, with their arguments bound.

    Hand-written functions in the bridge's module are steps too, so a fit is one like
    any other: ``fsl_mrs.step.fit(basis)``, ``osprey.step.fit(basis)``. A module with no
    bridge at all, such as LCModel's, can be deferred onto directly.

    The data is a toolbox function's first argument, and ``functools.partial`` binds only
    leading ones, so there is no stdlib way to bind the rest and wait. Deferring could be
    inferred instead - no data passed, so it must be a step - but that is ambiguous for
    exactly the routing case that needs it: in ``ecc(data, reference)`` the reference is
    itself NIfTI-MRS, so ``ecc(water)`` could equally mean "run ecc on the water". One
    explicit spelling avoids guessing.

    Args:
        bridge: The bridge to defer calls to, or a module of hand-written functions.
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, bridge: Any):
        self._bridge = bridge

    #**********#
    #   repr   #
    #**********#
    def __repr__(self) -> str:
        name = getattr(self._bridge, "name", None) or self._bridge.__name__.rsplit(".", 1)[-1]
        return f"{name}.step"

    #*************#
    #   resolve   #
    #*************#
    def _resolve(self, func: str) -> Callable:
        """The bridge's function, else a hand-written one in the bridge's module."""
        try:
            return getattr(self._bridge, func)
        except AttributeError:
            if isinstance(self._bridge, Bridge):
                module = sys.modules.get(type(self._bridge).__module__)
                written = vars(module).get(func) if module else None
                if callable(written):
                    return written
            raise

    #*************#
    #   getattr   #
    #*************#
    def __getattr__(self, func: str) -> Callable:
        if func.startswith("_"):
            raise AttributeError(func)
        call = self._resolve(func)             # raises AttributeError on an unknown name

        def bind(*args, **kwargs):
            def step(data):
                return call(data, *args, **kwargs)
            step.__name__ = func               # so a pipeline files its metrics by name
            step.__doc__ = call.__doc__
            return step

        bind.__name__ = func
        return bind

    #*********#
    #   dir   #
    #*********#
    def __dir__(self):
        if isinstance(self._bridge, Bridge):
            return sorted(self._bridge.available())
        return sorted(n for n, v in vars(self._bridge).items()
                      if callable(v) and not n.startswith("_"))


#**************#
#   describe   #
#**************#
def describe(bridge: Bridge, limit: int = 20) -> str:
    """Human-readable summary of what a bridge can dispatch."""
    names = bridge.available()
    head = ", ".join(names[:limit])
    more = f" ... (+{len(names) - limit} more)" if len(names) > limit else ""
    return f"{bridge.name}: {len(names)} functions\n  {head}{more}"
