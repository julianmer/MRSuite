####################################################################################################
#                                           pipeline.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2025-07-25                                                                              #
#                                                                                                  #
# Purpose: Applies a sequence of callables to whatever is handed to it. One value in, one          #
#          out, so a water reference is processed by a pipeline of its own rather than             #
#          threaded through this one, and a pipeline is itself a step, so processing and           #
#          analysis compose into a single run.                                                     #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


#*******************#
#   holds results   #
#*******************#
def _holds_results(data: Any) -> bool:
    """Whether measurements can be filed on this object.

    A NIfTI-MRS+ carries per-spectrum metadata dictionaries; a bare NIfTI-MRS does not.
    """
    return hasattr(data, "metadata_common") and hasattr(data, "metadata_individual")


#************#
#   bucket   #
#************#
def _bucket(data: Any, index: Optional[int] = None) -> Dict[str, Any]:
    """The results dictionary for one spectrum, or for the batch.

    Kept in NIfTI-MRS+'s own metadata rather than added to that class: only this
    pipeline files measurements, so the concept belongs here. It rides along in
    metadata_individual, which is deep-copied by copy(), so results survive with the
    data without NIfTI-MRS+ knowing what a result is.
    """
    if index is None:
        return data.metadata_common.setdefault("results", {})

    while len(data.metadata_individual) < len(data):
        data.metadata_individual.append({})    # empty in volatile mode, short if passed in
    return data.metadata_individual[index].setdefault("results", {})


#*************#
#   results   #
#*************#
def results(data: Any, index: Optional[int] = None) -> Dict[str, Any]:
    """
    Everything a pipeline measured on ``data``, by step name.

    Args:
        data: What a pipeline returned.
        index (Optional[int]): Which spectrum to read. None reads the batch level.

    Returns:
        Dict[str, Any]: Results by step name; empty if nothing was measured.
    """
    return dict(_bucket(data, index)) if _holds_results(data) else {}


#****************#
#   get result   #
#****************#
def get_result(data: Any, name: str, index: Optional[int] = None) -> Any:
    """
    One measurement, or None if the pipeline did not record it.

    Args:
        data: What a pipeline returned.
        name (str): The step's name.
        index (Optional[int]): Which spectrum to read. None reads the batch level.

    Returns:
        The result as the step returned it.
    """
    return results(data, index).get(name)


#**************************************************************************************************#
#                                          Class Pipeline                                          #
#**************************************************************************************************#
#                                                                                                  #
# A sequence of callables applied to a value in order.                                             #
#                                                                                                  #
#**************************************************************************************************#
class Pipeline:
    """A sequence of callables applied to a value in order.

    A step is anything callable that takes the value and returns it: a bridged toolbox
    function, your own function, or ``bridge.step.f(args)`` to bind arguments and take
    the value later. Nothing has to be wrapped, so adding something new costs nothing.

        water = Pipeline([fsl_mrs.step.coilcombine()]).run(water)

        process = Pipeline([fsl_mrs.step.coilcombine(),
                            fid_a.step.op_rmbadaverages(4, 't'),
                            fsl_mrs.step.ecc(water)])

        analyse = Pipeline([fsl_mrs.step.fit(basis, water)])   # spectra -> results

        results = Pipeline([process, analyse]).run(data)       # a pipeline is a step

    What a step returns is what flows on, so a pipeline is not tied to spectra: it
    carries a basis set or fit results just as well, and a fit is simply the step where
    the one becomes the other. A tuple whose first element is the same kind of thing
    that went in is the exception - that is a toolbox reporting a measurement alongside
    its output, and the rest is attached to the value under the step's name, readable
    afterwards with ``get_result(name)``.

    Args:
        steps (Optional[Iterable[Callable]]): The steps, in order.

    Raises:
        TypeError: If a step is not callable. Raised on construction, so a bad pipeline
            fails when it is built rather than part-way through a long run.
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, steps: Optional[Iterable[Callable]] = None) -> None:
        self.steps: List[Callable] = list(steps or [])
        for i, step in enumerate(self.steps):
            if not callable(step):
                raise TypeError(f"step {i} is {step!r}, which is not callable")

    #**********#
    #   repr   #
    #**********#
    def __repr__(self) -> str:
        return f"Pipeline({' -> '.join(self._name(s, i) for i, s in enumerate(self.steps))})"

    #**********#
    #   name   #
    #**********#
    @staticmethod
    def _name(step: Callable, i: int) -> str:
        """What to file a step's side result under.

        ``bridge.step.f()`` carries the toolbox function's name. A lambda has none, so
        it falls back to its position.
        """
        name = getattr(step, "__name__", None)
        return f"step{i}" if not name or name == "<lambda>" else name

    #***********#
    #   split   #
    #***********#
    @staticmethod
    def _split(previous: Any, out: Any) -> Tuple[Any, Any]:
        """Separate what a step returned into the value carried on and a side result.

        One rule: what a step returns is what flows next. The exception is a tuple whose
        first element is the same kind of thing that went in, which is how a toolbox
        reports a measurement alongside its output - op_rmbadaverages hands back
        (spectra, metrics) - and there the rest is kept as a side result.

        The rule is deliberately about the *kind* of thing rather than about spectra, so
        that a pipeline can carry a basis set or fit results just as well: a fit turns
        spectra into results, and everything after it works on results.
        """
        if isinstance(out, tuple) and out and isinstance(out[0], type(previous)):
            rest = out[1:]
            return out[0], (rest[0] if len(rest) == 1 else rest) if rest else None
        return out, None

    #***********#
    #   carry   #
    #***********#
    @staticmethod
    def _carry(before: Any, after: Any) -> None:
        """Move results from the object that went in onto the one that came out.

        A step usually builds a new NIfTI-MRS+ rather than mutating its input, and a new
        one starts with empty metadata. Without this, the step after a measuring one
        would silently drop what it measured.
        """
        if after is before or not (_holds_results(before) and _holds_results(after)):
            return

        _bucket(after).update(_bucket(before))
        for i in range(min(len(before), len(after))):
            _bucket(after, i).update(_bucket(before, i))

    #************#
    #   attach   #
    #************#
    @staticmethod
    def _attach(data: Any, name: str, result: Any) -> None:
        """File a metric on the data under the step's name.

        A list as long as the batch came from a call that ran once per spectrum, so it
        is filed per spectrum; anything else describes the batch. The test is list
        against tuple, not length: the bridges return a list when they batch, and a step
        reporting several different things returns a tuple.
        """
        if not _holds_results(data):
            return                             # nowhere to put it, e.g. a bare NIfTI-MRS

        if isinstance(result, list) and len(result) == len(data):
            for i, one in enumerate(result):
                _bucket(data, i)[name] = one
        else:
            _bucket(data)[name] = result

    #*********#
    #   run   #
    #*********#
    def run(self, data: Any) -> Any:
        """
        Run every step in order.

        Args:
            data: What to run them on. Spectra for a processing pipeline, but a basis
                set or fit results work the same way.

        Returns:
            Whatever the last step produced, carrying anything measured along the way.
        """
        for i, step in enumerate(self.steps):
            before = data
            data, result = self._split(before, step(before))
            self._carry(before, data)
            if result is not None:
                self._attach(data, self._name(step, i), result)
        return data

    #**********#
    #   call   #
    #**********#
    def __call__(self, data: Any) -> Any:
        """Run the pipeline.

        A step is anything callable that takes the value and returns it, and a pipeline
        is exactly that, so this is what lets one nest inside another:
        ``Pipeline([process, analyse])``.
        """
        return self.run(data)
