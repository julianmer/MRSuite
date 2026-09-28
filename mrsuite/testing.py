####################################################################################################
#                                            testing.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-28                                                                              #
#                                                                                                  #
# Purpose: A conformance check for any method: whether it keeps the contract every MRSuite         #
#          method keeps (see mrsuite.core.method), on data it accepts. Plain assertions, so it     #
#          runs in any test framework or none; MRSuite's own tests run it on every method.         #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import json
from typing import Any, Callable, List

import numpy as np

from nifti_mrs.nifti_mrs import NIFTI_MRS

from nifti_mrs_plus import NIfTI_MRS_Plus
from mrsuite.core import provenance
from mrsuite.core.pipeline import Pipeline
from mrsuite.core.result import FitResult


#***********#
#   check   #
#***********#
def check(method: Callable, data: Any, *args, rtol: float = 1e-5, **kwargs) -> None:
    """
    Assert that a method keeps MRSuite's contract.

    The caller brings the data, because only the caller knows what the method accepts:
    a model takes spectra acquired as it was trained, a fit needs its basis among the
    arguments. A batch is checked as it is; a single spectrum as a batch of two copies.

        check(mrsuite.fsl_mrs.fit, batch, "basis.BASIS", water=references)

    Checked: one spectrum gives one output and a batch one per spectrum, in its order,
    each the same as one by one; the method runs as a pipeline step; outputs keep their
    spectrum's header and history and add one ProcessingApplied entry with a known
    version; a FitResult reports its own metabolites, in numbers; the input is left as
    it was; and further spectra that do not line up with the data are refused.

    Args:
        method (Callable): The method, called as ``method(data, *args, **kwargs)``.
        data: A NIfTI-MRS or NIfTI-MRS+ the method accepts.
        *args: Its further arguments. Further spectra follow the data: one beside a
            single spectrum, a batch as long as a batch.
        rtol (float): How closely the batch must reproduce its spectra one by one.
        **kwargs: Its further arguments by name, likewise.

    Raises:
        AssertionError: Naming the rule that was broken.
    """
    if not isinstance(data, NIfTI_MRS_Plus):
        data = _twice(data)
        args = tuple(map(_twice, args))
        kwargs = {k: _twice(v) for k, v in kwargs.items()}
    items = data.list()
    further = [v for v in (*args, *kwargs.values()) if isinstance(v, NIfTI_MRS_Plus)]
    before = [_state(s) for s in items + [s for v in further for s in v.list()]]

    def pick(value, i):
        return value[i] if isinstance(value, NIfTI_MRS_Plus) else value

    alone = []
    for i, item in enumerate(items):
        out = method(item, *(pick(a, i) for a in args),
                     **{k: pick(v, i) for k, v in kwargs.items()})
        out = out[0] if isinstance(out, tuple) else out
        _require(isinstance(out, (NIFTI_MRS, FitResult)),
                 f"one spectrum gave {type(out).__name__}, not a NIfTI-MRS or a FitResult")
        alone.append(out)

    step = method.step(*args, **kwargs) if hasattr(method, "step") else \
        (lambda d: method(d, *args, **kwargs))
    together = Pipeline([step]).run(data)
    _require(isinstance(together, NIfTI_MRS_Plus)
             or (isinstance(together, list) and all(isinstance(o, FitResult) for o in together)),
             f"a batch gave {type(together).__name__}, not a NIfTI-MRS+ or a list of FitResult")
    outs = together.list() if isinstance(together, NIfTI_MRS_Plus) else together
    _require(len(outs) == len(items), f"a batch of {len(items)} gave {len(outs)} outputs")

    for i, (item, out, one) in enumerate(zip(items, outs, alone)):
        _same(out, one, rtol, i)
        _recorded(item, out, i)
        if isinstance(out, FitResult):
            _reports(out, i)

    after = [_state(s) for s in items + [s for v in further for s in v.list()]]
    _require(after == before, "the input was changed")

    for value in further:
        short = NIfTI_MRS_Plus(value.list()[:-1])
        try:
            method(data, *(short if a is value else a for a in args),
                   **{k: short if v is value else v for k, v in kwargs.items()})
        except ValueError:
            continue
        raise AssertionError(f"{len(short)} further spectra for {len(items)} were not refused")


#***********#
#   twice   #
#***********#
def _twice(value: Any) -> Any:
    """A spectrum as a batch of itself and a copy; anything else as it is."""
    return NIfTI_MRS_Plus([value, value.copy()]) if isinstance(value, NIFTI_MRS) else value


#*************#
#   require   #
#*************#
def _require(condition: bool, message: str) -> None:
    """An assertion that stays when Python runs with -O."""
    if not condition:
        raise AssertionError(message)


#***********#
#   state   #
#***********#
def _state(spectrum: NIFTI_MRS) -> tuple:
    """A spectrum's samples and header, to tell whether a method changed them."""
    return (provenance.checksum(spectrum[:]),
            json.dumps(spectrum.hdr_ext.to_dict(), sort_keys=True, default=str))


#**********#
#   same   #
#**********#
def _same(out: Any, one: Any, rtol: float, i: int) -> None:
    """A spectrum's output in the batch against its output alone."""
    _require(type(out) is type(one), f"spectrum {i} gave a {type(one).__name__} alone "
                                     f"but a {type(out).__name__} in the batch")
    if isinstance(out, FitResult):
        pairs = [(out.concentrations, one.concentrations), (out.uncertainties, one.uncertainties)]
        for batch, alone in pairs:
            _require(_keys(batch) == _keys(alone),
                     f"spectrum {i} reports other values in the batch than alone")
            a, b = _values(batch), _values(alone)
            _require(np.allclose(a, b, rtol=rtol, atol=rtol * _scale(b), equal_nan=True),
                     f"spectrum {i} fits differently in the batch than alone")
    else:
        a, b = np.asarray(out[:]), np.asarray(one[:])
        _require(a.shape == b.shape and np.allclose(a, b, rtol=rtol, atol=rtol * _scale(b)),
                 f"spectrum {i} comes out differently in the batch than alone")


#**************#
#   recorded   #
#**************#
def _recorded(item: NIFTI_MRS, out: Any, i: int) -> None:
    """The output keeps the spectrum's header and history, and adds one entry."""
    history = provenance.history(item)
    if isinstance(out, FitResult):
        steps = out.provenance
    else:
        steps = provenance.history(out)
        lost = [k for k in item.hdr_ext.to_dict() if not k.startswith("dim_")
                and k not in out.hdr_ext]
        _require(not lost, f"spectrum {i} lost the header fields {lost}")

    _require(steps[:len(history)] == history, f"spectrum {i} lost its processing history")
    _require(len(steps) == len(history) + 1,
             f"spectrum {i} gained {len(steps) - len(history)} ProcessingApplied entries; "
             f"one call adds one")
    entry = steps[-1]
    missing = {"Time", "Program", "Version", "Method", "Details"} - set(entry)
    _require(not missing, f"the entry lacks {sorted(missing)}")
    _require(entry["Version"] not in (None, "", "unknown"),
             f"{entry['Program']} does not tell its version")


#*************#
#   reports   #
#*************#
def _reports(out: FitResult, i: int) -> None:
    """A FitResult reports numbers for its own metabolites."""
    _require(bool(out.metabolites), f"spectrum {i} gave a FitResult without metabolites")
    for kind, table in (*out.concentrations.items(), *out.uncertainties.items()):
        unknown = sorted(set(table) - set(out.metabolites))
        _require(not unknown, f"'{kind}' reports {unknown}, which are not among the metabolites")
        _require(all(isinstance(v, (int, float)) for v in table.values()),
                 f"'{kind}' holds values that are not numbers")


#**********#
#   keys   #
#**********#
def _keys(tables: dict) -> List[tuple]:
    """Every (kind, metabolite) a FitResult's tables report, in a fixed order."""
    return sorted((kind, name) for kind, table in tables.items() for name in table)


#************#
#   values   #
#************#
def _values(tables: dict) -> np.ndarray:
    """The tables' values in the order of ``_keys``."""
    return np.array([tables[kind][name] for kind, name in _keys(tables)], dtype=float)


#***********#
#   scale   #
#***********#
def _scale(values: np.ndarray) -> float:
    """The size of what is compared, so that values near zero are compared absolutely."""
    finite = np.abs(values[np.isfinite(values)])
    return float(finite.max()) if finite.size else 0.0
