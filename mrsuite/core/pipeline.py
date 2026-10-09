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
#          analysis compose into a single run. A pipeline of steps is written down as the code     #
#          that builds it, or as a recipe in JSON, and built again from either.                    #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import importlib
import keyword
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

from mrsuite.core.method import Step


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

    #************#
    #   stages   #
    #************#
    def stages(self, data: Any) -> Iterator[Any]:
        """
        Run every step in order, handing out what each one made as it goes.

        Args:
            data: What to run them on, as for ``run``.

        Yields:
            Each step's output, carrying anything measured so far: what ``run`` returns
            last, and every intermediate on the way, to look at or to run on from.
        """
        for i, step in enumerate(self.steps):
            before = data
            data, result = self._split(before, step(before))
            self._carry(before, data)
            if result is not None:
                self._attach(data, self._name(step, i), result)
            yield data

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
        for data in self.stages(data):
            pass
        return data

    #************#
    #   script   #
    #************#
    def script(self, inputs: Optional[Dict[str, Any]] = None) -> str:
        """
        The pipeline as the Python that builds it: a function ``pipeline`` taking the
        spectra its steps were given, such as a water reference, by name.

            water = ...
            pipeline = Pipeline([fsl_mrs.step.coilcombine(), fsl_mrs.step.ecc(water)])
            print(pipeline.script(inputs={"water": water}))

        Args:
            inputs (Optional[Dict[str, Any]]): The spectra the steps were given, by the
                name to give them.

        Returns:
            str: Python source.

        Raises:
            ValueError: If a step or an argument cannot be written as code: a lambda, a
                model not loaded by name, unnamed spectra.
        """
        from mrsuite.core.provenance import mrsuite_version

        writer = _Writer(inputs)
        body = writer.steps(self.steps, "        ")
        clash = [n for n in writer.names if not n.isidentifier() or keyword.iskeyword(n)
                 or n in writer.bound()]
        if clash:
            raise ValueError(f"inputs named {clash} would not be Python names of their own in "
                             f"the script; name them otherwise")
        imports = sorted(writer.imports | {"from mrsuite.core.pipeline import Pipeline"})
        return "\n".join([f"# Written by MRSuite {mrsuite_version()}: the pipeline as code.",
                          *imports, "", "",
                          f"def pipeline({', '.join(writer.names)}):",
                          "    return Pipeline([", body, "    ])", ""])

    #*************#
    #   to dict   #
    #*************#
    def to_dict(self, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        The pipeline as a recipe: JSON that names each step's method and its arguments,
        to keep with results and build the pipeline again with ``from_dict``.

        Args:
            inputs (Optional[Dict[str, Any]]): The spectra the steps were given, by the
                name to give them; the recipe takes them by that name.

        Returns:
            Dict[str, Any]: The recipe.

        Raises:
            ValueError: If a step or an argument cannot be written down, as for ``script``.
        """
        from mrsuite.core.provenance import mrsuite_version

        writer = _Writer(inputs)
        steps = writer.entries(self.steps)
        return {"format": "mrsuite.pipeline", "mrsuite": mrsuite_version(),
                "inputs": writer.names, "steps": steps}

    #***************#
    #   from dict   #
    #***************#
    @classmethod
    def from_dict(cls, recipe: Dict[str, Any], **inputs: Any) -> "Pipeline":
        """
        Build a pipeline again from its recipe.

        Args:
            recipe (Dict[str, Any]): As ``to_dict`` wrote it.
            **inputs: The spectra the recipe names, by name.

        Returns:
            Pipeline: The same steps with the same arguments.

        Raises:
            ValueError: If it is not a recipe, or an input it names is not given.
        """
        if recipe.get("format") != "mrsuite.pipeline":
            raise ValueError("not a pipeline recipe: its format is not 'mrsuite.pipeline'")
        missing = sorted(set(recipe["inputs"]) - set(inputs))
        if missing:
            raise ValueError(f"the recipe takes {missing}; give them by name")
        return _build(recipe["steps"], inputs)

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


#**************************************************************************************************#
#                                          Class _Writer                                           #
#**************************************************************************************************#
#                                                                                                  #
# Writes steps down as code or as recipe entries, naming the spectra they were given.              #
#                                                                                                  #
#**************************************************************************************************#
class _Writer:
    """Writes steps down as code or as recipe entries, naming the spectra they were given.

    What would not come back as it was is refused rather than written: a lambda, a bridge
    with settings of its own, a model loaded from a folder, a function of the script that
    is running, an argument JSON or Python cannot hold.

    Args:
        inputs (Optional[Dict[str, Any]]): The spectra, by the name to give them.

    Attributes:
        imports (set): The import lines the code needs.
        names (List[str]): The inputs the steps use, in the order they first do.
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, inputs: Optional[Dict[str, Any]]) -> None:
        self._by_id = {id(value): name for name, value in (inputs or {}).items()}
        self.imports, self.names = set(), []

    #***********#
    #   steps   #
    #***********#
    def steps(self, steps: List[Callable], indent: str, at: str = "") -> str:
        """Steps as the lines of a list, a nested pipeline as a list of its own."""
        lines = []
        for i, step in enumerate(steps):
            if isinstance(step, Pipeline):
                inner = self.steps(step.steps, indent + "    ", f"{at}{i}.")
                lines.append(f"{indent}Pipeline([\n{inner}\n{indent}]),")
                continue
            target = self._reach(step, f"{at}{i}")
            if not isinstance(step, Step):
                lines.append(f"{indent}{target},")
                continue
            bound = [self._code(a) for a in step.args]
            named = {k: v for k, v in step.kwargs.items()
                     if k.isidentifier() and not keyword.iskeyword(k)}   # lambda: spant's
            bound += [f"{k}={self._code(v)}" for k, v in named.items()]
            spelt = {k: v for k, v in step.kwargs.items() if k not in named}   # e.g. na.rm
            if spelt:
                bound.append(f"**{self._code(spelt)}")
            lines.append(f"{indent}{target}({', '.join(bound)}),")
        return "\n".join(lines)

    #*************#
    #   entries   #
    #*************#
    def entries(self, steps: List[Callable], at: str = "") -> List[Dict[str, Any]]:
        """Steps as recipe entries, a nested pipeline as a list of its own."""
        out = []
        for i, step in enumerate(steps):
            if isinstance(step, Pipeline):
                out.append({"pipeline": self.entries(step.steps, f"{at}{i}.")})
                continue
            entry = self._where(step, f"{at}{i}")
            if isinstance(step, Step):
                entry.update(args=[self._json(a) for a in step.args],
                             kwargs={k: self._json(v) for k, v in step.kwargs.items()})
            out.append(entry)
        return out

    #***********#
    #   where   #
    #***********#
    def _where(self, step: Callable, at: str) -> Dict[str, Any]:
        """Where a step's method is found again: a model by the Hub name it was loaded by,
        anything else by its module and name."""
        method = step.method if isinstance(step, Step) else step
        source = getattr(method, "source", None)
        if isinstance(step, Step) and source:
            name = str(source["name"])
            if "/" not in name or Path(name).exists():
                raise ValueError(f"step {at}, a model loaded from a folder, cannot be written "
                                 f"down; load it by its name on the Hub")
            return {"model": name, "revision": source["revision"]}
        module, name = getattr(method, "__module__", None), getattr(method, "__qualname__", "")
        found = None
        if module and module != "__main__" and name.isidentifier():
            found = getattr(importlib.import_module(module), name, None)
        bridged = bool(module) and module.startswith("mrsuite.bridges.") and name.isidentifier()
        if not (bridged or found is method):
            raise ValueError(f"step {at}, {step!r}, cannot be written down: it is not found "
                             f"again by its module and name (a lambda, a function of the "
                             f"script that is running, a bridge with settings of its own, or "
                             f"a model not loaded by name)")
        if isinstance(step, Step) and not bridged and not hasattr(method, "step"):
            raise ValueError(f"step {at} binds arguments to {name}, which is not a method; "
                             f"make it one with mrsuite.method")
        return {"module": module, "function": name}

    #***********#
    #   reach   #
    #***********#
    def _reach(self, step: Callable, at: str) -> str:
        """The code that reaches a step, with its import: bound to its arguments for a step,
        the function itself for a plain one."""
        where = self._where(step, at)
        bind = ".step" if isinstance(step, Step) else ""
        if "model" in where:
            self.imports.add("from mrsuite import models")
            return f"models.load({where['model']!r}, revision={where['revision']!r}).step"
        if where["module"].startswith("mrsuite.bridges."):
            bridge = where["module"].rsplit(".", 1)[1]
            self.imports.add(f"from mrsuite.bridges import {bridge}")
            return f"{bridge}{bind}.{where['function']}"
        # the module whole, so two functions of one name from two modules stay apart
        self.imports.add(f"import {where['module']}")
        return f"{where['module']}.{where['function']}{bind}"

    #***********#
    #   bound   #
    #***********#
    def bound(self) -> set:
        """The names the imports bind, which an input must not take."""
        return {line.split()[-1] if line.startswith("from ") else line.split()[1].split(".")[0]
                for line in self.imports} | {"Pipeline", "pipeline"}

    #**********#
    #   code   #
    #**********#
    def _code(self, value: Any) -> str:
        """An argument as Python."""
        value = self._plain(value)
        if isinstance(value, _Input):
            return value.name
        if isinstance(value, Path):
            self.imports.add("from pathlib import Path")
            return f"Path({str(value)!r})"
        if isinstance(value, tuple):
            items = [self._code(v) for v in value]
            return f"({', '.join(items)}{',' if len(items) == 1 else ''})"
        if isinstance(value, list):
            return f"[{', '.join(self._code(v) for v in value)}]"
        if isinstance(value, dict):
            return "{" + ", ".join(f"{k!r}: {self._code(v)}" for k, v in value.items()) + "}"
        if isinstance(value, float) and not np.isfinite(value):
            return f"float({str(value)!r})"
        if isinstance(value, str):
            return repr(str(value))                    # not a subclass' own repr
        return repr(value)

    #**********#
    #   json   #
    #**********#
    def _json(self, value: Any) -> Any:
        """An argument as JSON, tagging what JSON cannot tell apart: an input, a path, a
        tuple."""
        value = self._plain(value)
        if isinstance(value, _Input):
            return {"$input": value.name}
        if isinstance(value, Path):
            return {"$path": str(value)}
        if isinstance(value, tuple):
            return {"$tuple": [self._json(v) for v in value]}
        if isinstance(value, list):
            return [self._json(v) for v in value]
        if isinstance(value, dict):
            return {k: self._json(v) for k, v in value.items()}
        return str(value) if isinstance(value, str) else value

    #***********#
    #   plain   #
    #***********#
    def _plain(self, value: Any) -> Any:
        """An argument as what can be written down: spectra and a basis by their name, a
        NumPy number as a Python one; anything else that is not a literal is refused."""
        from nifti_mrs.nifti_mrs import NIFTI_MRS
        from nifti_mrs_plus import NIfTI_MRS_Plus
        from mrsuite.core.basis import Basis

        if isinstance(value, (NIFTI_MRS, NIfTI_MRS_Plus, Basis)):
            if id(value) not in self._by_id:
                raise ValueError("a step takes spectra or a basis the pipeline does not name; "
                                 "give them as inputs={'name': spectra}")
            name = self._by_id[id(value)]
            if name not in self.names:
                self.names.append(name)
            return _Input(name)
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, dict):
            if not all(isinstance(k, str) for k in value):
                raise ValueError(f"{value!r} cannot be written down: its keys are not all text")
            if any(k.startswith("$") for k in value):
                raise ValueError(f"{value!r} cannot be written down: a key starting with $ "
                                 f"reads back as something else")
        if value is None or isinstance(value, (bool, int, float, str, Path, tuple, list, dict)):
            return value
        raise ValueError(f"{value!r} cannot be written down; give arguments as numbers, text, "
                         f"paths, lists, tuples, dictionaries or named spectra")


#**************************************************************************************************#
#                                          Class _Input                                            #
#**************************************************************************************************#
#                                                                                                  #
# Spectra a step was given, by the name the pipeline's inputs give them.                           #
#                                                                                                  #
#**************************************************************************************************#
class _Input:
    """Spectra a step was given, by the name the pipeline's inputs give them."""

    #**********#
    #   init   #
    #**********#
    def __init__(self, name: str) -> None:
        self.name = name


#***********#
#   build   #
#***********#
def _build(entries: List[Dict[str, Any]], inputs: Dict[str, Any]) -> "Pipeline":
    """A pipeline from recipe entries, its inputs filled in."""
    def value(v: Any) -> Any:
        if isinstance(v, list):
            return [value(x) for x in v]
        if isinstance(v, dict):
            if set(v) == {"$input"}:
                return inputs[v["$input"]]
            if set(v) == {"$path"}:
                return Path(v["$path"])
            if set(v) == {"$tuple"}:
                return tuple(value(x) for x in v["$tuple"])
            return {k: value(x) for k, x in v.items()}
        return v

    steps = []
    for entry in entries:
        if "pipeline" in entry:
            steps.append(_build(entry["pipeline"], inputs))
            continue
        if "model" in entry:
            from mrsuite import models
            method = models.load(entry["model"], revision=entry["revision"])
        else:
            method = getattr(importlib.import_module(entry["module"]), entry["function"])
        steps.append(Step(method, value(entry["args"]), value(entry["kwargs"]))
                     if "args" in entry else method)
    return Pipeline(steps)
