####################################################################################################
#                                            method.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-28                                                                              #
#                                                                                                  #
# Purpose: What every method in MRSuite is - a toolbox function, a fit, a trained model or a       #
#          plain function of your own - written down once, with the one class that makes it so:    #
#          spectra in, spectra or fit results out, a batch handled like one spectrum, the header   #
#          kept and every call recorded. mrsuite.testing.check tests a method against it.          #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import functools
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from nifti_mrs.definitions import standard_defined
from nifti_mrs.nifti_mrs import NIFTI_MRS

from nifti_mrs_plus import NIfTI_MRS_Plus
from mrsuite.core import provenance
from mrsuite.core.result import FitResult


#**************************************************************************************************#
#                                           Class Method                                           #
#**************************************************************************************************#
#                                                                                                  #
# A method: spectra in, spectra or fit results out, with what was done recorded.                   #
#                                                                                                  #
#**************************************************************************************************#
class Method:
    """A method: spectra in, spectra or fit results out, with what was done recorded.

    The contract, the same for a toolbox function, a fit, a trained model or your own:

    1. It is called with the data first: one spectrum as NIfTI-MRS, or a batch as
       NIfTI-MRS+.
    2. It returns what it was given: one output for one spectrum, and for a batch a
       NIfTI-MRS+ of spectra or a list of FitResult, as long as the batch and in its
       order. A tuple is an output with a measurement beside it, which a pipeline
       files under the step's name.
    3. A spectrum it returns keeps the header of the one it came from - the echo time,
       the reference, the processing history - and gains one ProcessingApplied entry:
       the program, its exact version, the method and its arguments. A FitResult's
       provenance is its spectrum's history and that entry.
    4. Its input is left as it was.
    5. Further spectra, such as a water reference, come one per spectrum: a batch as
       long as the data's, or one spectrum beside one spectrum.

    This class keeps 1 to 3 and 5; a subclass does the work, and leaves its input alone.
    It states ``program`` and implements ``run``, and where it can, ``version`` and
    ``runtime``. ``mrsuite.method`` makes a plain function one, and
    ``mrsuite.testing.check`` tests any method against the contract.

    Attributes:
        program (str): Who did the work, credited in ProcessingApplied.
        name (Optional[str]): The method as ProcessingApplied names it; by default the
            name it is called by.
        batched (bool): ``run`` takes the whole batch as NIfTI-MRS+ and returns a list,
            one output per spectrum, for work that is cheaper done for all at once.
            Otherwise it takes one spectrum and returns its output.
        parallel (bool): Offer ``workers=`` to spread a batch over that many processes
            (see ``mrsuite.core.parallel``).
        records (bool): False for a program that records its own steps in the spectra
            it returns, as FSL-MRS does.
    """

    program: str = "MRSuite"
    name: Optional[str] = None
    batched: bool = False
    parallel: bool = False
    records: bool = True

    #*************#
    #   version   #
    #*************#
    def version(self) -> Optional[str]:
        """The program's exact version, or None when it cannot be told.

        Asked after the work is done, as ``runtime`` is, so both can report what the
        work itself resolved.
        """
        return None

    #*************#
    #   runtime   #
    #*************#
    def runtime(self) -> Dict[str, Any]:
        """What else the result depends on besides the arguments, e.g. the interpreter."""
        return {}

    #*********#
    #   run   #
    #*********#
    def run(self, data: Any, *args, **kwargs) -> Any:
        """The work: one spectrum to its output, or with ``batched`` a batch to a list."""
        raise NotImplementedError

    #**********#
    #   call   #
    #**********#
    def __call__(self, *args, **kwargs) -> Any:
        """Run the method on one spectrum or a batch; anything else is a plain call."""
        data = args[0] if args else None
        if not isinstance(data, (NIFTI_MRS, NIfTI_MRS_Plus)):
            return self.run(*args, **kwargs)     # e.g. a toolbox's loader, given a path

        workers = kwargs.pop("workers", 1) if self.parallel else 1
        items = data.list() if isinstance(data, NIfTI_MRS_Plus) else [data]
        args = args[1:]
        calls = self._pair(args, kwargs, len(items))

        if workers > 1 and len(items) > 1:
            from mrsuite.core.parallel import spread
            outs = spread(self, items, args, kwargs, workers)    # each chunk runs whole
        else:
            outs = self._outputs(data, items, args, kwargs, calls)
            self._record(items, outs, calls)
        return _regroup(outs) if isinstance(data, NIfTI_MRS_Plus) else outs[0]

    #**********#
    #   step   #
    #**********#
    def step(self, *args, **kwargs) -> "Step":
        """This method as a pipeline step: the arguments bound now, the data taken later."""
        return Step(self, args, kwargs)

    #***********#
    #   label   #
    #***********#
    def _label(self) -> str:
        """The method as ProcessingApplied and error messages name it."""
        return self.name or getattr(self, "__name__", None) or type(self).__name__

    #**********#
    #   pair   #
    #**********#
    def _pair(self, args: tuple, kwargs: dict, n: int) -> List[Tuple[tuple, dict]]:
        """Each spectrum's own arguments: further spectra one per spectrum, the rest shared.

        One spectrum beside a batch is refused rather than spread over it: a batch is
        usually separate acquisitions, and scaling them all by one reference would be
        wrong without looking wrong.
        """
        named = [(f"argument {j}", v) for j, v in enumerate(args, 1)] + list(kwargs.items())
        for key, value in named:
            given = len(value) if isinstance(value, NIfTI_MRS_Plus) else 1
            if isinstance(value, (NIFTI_MRS, NIfTI_MRS_Plus)) and given != n:
                raise ValueError(f"{self._label()}: {key} gives {given} for {n} "
                                 f"spectr{'um' if n == 1 else 'a'}; give one per spectrum")

        def pick(value, i):
            return value[i] if isinstance(value, NIfTI_MRS_Plus) else value
        return [(tuple(pick(v, i) for v in args), {k: pick(v, i) for k, v in kwargs.items()})
                for i in range(n)]

    #*************#
    #   outputs   #
    #*************#
    def _outputs(self, data: Any, items: List[NIFTI_MRS], args: tuple, kwargs: dict,
                 calls: List[Tuple[tuple, dict]]) -> List[Any]:
        """One output per spectrum, from ``run`` once per spectrum or once for all."""
        if not self.batched:
            return [self.run(item, *a, **k) for item, (a, k) in zip(items, calls)]

        def batch(value):
            return NIfTI_MRS_Plus([value]) if isinstance(value, NIFTI_MRS) else value
        return list(self.run(batch(data), *map(batch, args),
                             **{k: batch(v) for k, v in kwargs.items()}))

    #************#
    #   record   #
    #************#
    def _record(self, items: List[NIFTI_MRS], outs: List[Any],
                calls: List[Tuple[tuple, dict]]) -> None:
        """Keep each spectrum's header on what came of it, and record the call there.

        The arguments recorded are the spectrum's own, so a water reference is recorded
        with the spectrum it scaled. Only the first output of a tuple came of the spectrum;
        a further one (a processed water reference, say) came of another input and keeps
        the header its method gave it, with the call recorded on it too. An input handed
        back as it was is a declined no-op, and left as it is.
        """
        version, runtime, label = self.version(), self.runtime(), self._label()
        for item, out, (args, kwargs) in zip(items, outs, calls):
            entry = provenance.entry(self.program, version, label,
                                     {"args": args, "kwargs": kwargs, **runtime})
            inputs = (item, *args, *kwargs.values())
            for k, one in enumerate(out if isinstance(out, tuple) else (out,)):
                if isinstance(one, FitResult):
                    one.provenance = provenance.history(item) + [entry]
                elif isinstance(one, NIFTI_MRS) and not any(one is x for x in inputs):
                    if k == 0:
                        keep_header(item, one)
                    if self.records:
                        provenance.append(one, entry)


#**************************************************************************************************#
#                                            Class Step                                            #
#**************************************************************************************************#
#                                                                                                  #
# A method with its arguments bound, taking the data when called: one step of a pipeline.          #
#                                                                                                  #
#**************************************************************************************************#
class Step:
    """A method with its arguments bound, taking the data when called: one step of a pipeline.

    It keeps what it calls and with what, so a pipeline can show it, change it and write
    it down as the code that makes it (see ``Pipeline.script``).

    Args:
        method (Callable): What it calls, the data first.
        args (tuple): The arguments after the data.
        kwargs (dict): The arguments by name.
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, method: Callable, args: tuple = (), kwargs: Optional[dict] = None):
        self.method, self.args, self.kwargs = method, tuple(args), dict(kwargs or {})
        # so a pipeline files what the step measured under the method's name
        self.__name__ = getattr(method, "__name__", type(method).__name__)
        self.__doc__ = getattr(method, "__doc__", None)

    #**********#
    #   call   #
    #**********#
    def __call__(self, data: Any) -> Any:
        return self.method(data, *self.args, **self.kwargs)

    #**********#
    #   repr   #
    #**********#
    def __repr__(self) -> str:
        bound = [repr(a) for a in self.args] + [f"{k}={v!r}" for k, v in self.kwargs.items()]
        return f"{self.__name__}.step({', '.join(bound)})"


#**************************************************************************************************#
#                                          Class Function                                          #
#**************************************************************************************************#
#                                                                                                  #
# A plain function as a method; see ``method``.                                                    #
#                                                                                                  #
#**************************************************************************************************#
class Function(Method):
    """A plain function as a method; see ``method``.

    It keeps the function's name, docstring and module, so it is filed, documented and
    found - by a worker process, say - as the function would be.
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, fn: Callable, program: str,
                 version: Union[str, Callable[[], Optional[str]], None] = None,
                 runtime: Optional[Callable[[], Dict[str, Any]]] = None,
                 name: Optional[str] = None, batched: bool = False, parallel: bool = False,
                 records: bool = True) -> None:
        functools.update_wrapper(self, fn)
        self.program, self.name, self.batched = program, name, batched
        self.parallel, self.records = parallel, records
        self._version, self._runtime = version, runtime

    #*************#
    #   version   #
    #*************#
    def version(self) -> Optional[str]:
        return self._version() if callable(self._version) else self._version

    #*************#
    #   runtime   #
    #*************#
    def runtime(self) -> Dict[str, Any]:
        return self._runtime() if self._runtime else {}

    #*********#
    #   run   #
    #*********#
    def run(self, *args, **kwargs) -> Any:
        return self.__wrapped__(*args, **kwargs)


#************#
#   method   #
#************#
def method(program: str, version: Union[str, Callable[[], Optional[str]], None] = None,
           runtime: Optional[Callable[[], Dict[str, Any]]] = None, name: Optional[str] = None,
           batched: bool = False, parallel: bool = False) -> Callable[[Callable], Function]:
    """
    Make a plain function a method: batches, the header and the provenance come with it.

        @mrsuite.method(program="MyLab", version="1.2")
        def denoise(spectrum, strength=1.0):
            ...                                # one NIfTI-MRS in, one NIfTI-MRS out

        Pipeline([denoise.step(strength=2), fsl_mrs.step.fit(basis)])

    Args:
        program (str): Who did the work, credited in ProcessingApplied.
        version: The program's exact version, or a function that tells it when asked.
        runtime: A function returning what else the result depends on, e.g. the
            interpreter.
        name (Optional[str]): The method as ProcessingApplied names it; the function's
            own name by default.
        batched (bool): The function takes the whole batch as NIfTI-MRS+ and returns a
            list, one output per spectrum.
        parallel (bool): Offer ``workers=``. The function must be importable by its
            module and name, and return what can be pickled, such as a FitResult.

    Returns:
        A decorator that turns the function into a ``Function``.
    """
    def wrap(fn: Callable) -> Function:
        return Function(fn, program, version, runtime, name, batched, parallel)
    return wrap


#*****************#
#   keep header   #
#*****************#
def keep_header(source: NIFTI_MRS, out: NIFTI_MRS) -> None:
    """Carry the input's header fields over to the output.

    A toolbox working on its own struct hands back the samples and little else, so the
    echo time, the ppm reference and the processing history would be lost at the first
    step. Fields the output already states win, and the dimension fields are left to
    the output, since a step may have removed a dimension, as is the spectral width,
    since a step may have resampled. A user-defined field goes with its description,
    which nifti_mrs asks of it.
    """
    for key, value in source.hdr_ext.to_dict().items():
        if key.startswith("dim_") or key in out.hdr_ext:
            continue
        if key == "SpectralWidth":
            value = 1.0 / out.dwelltime
        if key in standard_defined:
            out.add_hdr_field(key, value)
        else:
            out.add_hdr_field(key, value.get("Value", value), doc=value["Description"])


#*************#
#   regroup   #
#*************#
def _regroup(outs: List[Any]) -> Any:
    """A batch's outputs as one, keeping tuple-returning methods tuple-shaped."""
    if outs and isinstance(outs[0], tuple):
        return tuple(_wrap([out[j] for out in outs]) for j in range(len(outs[0])))
    return _wrap(outs)


#**********#
#   wrap   #
#**********#
def _wrap(items: List[Any]) -> Any:
    """Spectra as NIfTI-MRS+, which picks up each one's history; anything else as a list."""
    if not items or not all(isinstance(i, NIFTI_MRS) for i in items):
        return items
    return NIfTI_MRS_Plus(items)
