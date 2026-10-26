####################################################################################################
#                                            result.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-08-08                                                                              #
#                                                                                                  #
# Purpose: One shape for what any analysis method produces, so a fit from FSL-MRS, one from        #
#          Osprey and one from a model can be compared against each other and against              #
#          ground truth. The method's own object is kept alongside, so making results              #
#          comparable never costs information.                                                     #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import hashlib
import importlib
import json
import os
import pickle
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Sequence

import numpy as np


#**************************************************************************************************#
#                                         Class FitResult                                          #
#**************************************************************************************************#
#                                                                                                  #
# What an analysis method produced, in a shape every method can fill.                              #
#                                                                                                  #
#**************************************************************************************************#
@dataclass
class FitResult:
    """What an analysis method produced, in a shape every method can fill.

    Concentrations and uncertainties are nested by scaling because the toolboxes really
    do report several: FSL-MRS has raw, internal, molarity and molality, and Osprey has
    tCr, water-scaled and tissue-corrected. Flattening to one number would decide for
    the caller which is the answer.

    Attributes:
        method (str): What produced this, as ``toolbox:algorithm``.
        metabolites (List[str]): Names, in the order the method reports them.
        concentrations (Dict[str, Dict[str, float]]): Scaling -> metabolite -> value.
        uncertainties (Dict[str, Dict[str, float]]): Kind -> metabolite -> value, in the
            units of the matching scaling. 'percentage' is CRLB as a percentage.
        qc (Dict[str, float]): Scalar quality measures, e.g. SNR, FWHM, MSE.
        provenance (List[Dict[str, str]]): How this came about, as NIfTI-MRS
            ProcessingApplied entries: the data's own processing history, then the fit,
            with its program, exact version, basis and options.
        raw (Any): The method's own result object, untouched. Anything not captured
            above lives here, including the curves; ``save`` keeps it as a pickle.
    """

    method: str
    metabolites: List[str] = field(default_factory=list)
    concentrations: Dict[str, Dict[str, float]] = field(default_factory=dict)
    uncertainties: Dict[str, Dict[str, float]] = field(default_factory=dict)
    qc: Dict[str, float] = field(default_factory=dict)
    provenance: List[Dict[str, str]] = field(default_factory=list)
    raw: Any = None

    #**********#
    #   repr   #
    #**********#
    def __repr__(self) -> str:
        return (f"FitResult({self.method}, {len(self.metabolites)} metabolites, "
                f"scalings={sorted(self.concentrations)})")

    #**********#
    #   conc   #
    #**********#
    def conc(self, scaling: str = "raw") -> Dict[str, float]:
        """
        Concentrations on one scaling.

        Args:
            scaling (str): Which scaling to read.

        Returns:
            Dict[str, float]: Metabolite to value.

        Raises:
            KeyError: If the method did not report that scaling, listing what it did.
        """
        if scaling not in self.concentrations:
            raise KeyError(f"{self.method} reports no '{scaling}' concentrations; "
                           f"it has {sorted(self.concentrations)}")
        return self.concentrations[scaling]

    #**********#
    #   crlb   #
    #**********#
    def crlb(self, kind: str = "percentage") -> Dict[str, float]:
        """
        Uncertainties of one kind.

        Args:
            kind (str): Which uncertainty to read.

        Returns:
            Dict[str, float]: Metabolite to value.

        Raises:
            KeyError: If the method did not report that kind, listing what it did.
        """
        if kind not in self.uncertainties:
            raise KeyError(f"{self.method} reports no '{kind}' uncertainties; "
                           f"it has {sorted(self.uncertainties)}")
        return self.uncertainties[kind]

    #************#
    #   curves   #
    #************#
    def curves(self) -> Dict[str, np.ndarray]:
        """
        The fit drawn: the data, the fit and its baseline over the method's own range, as
        its own object holds them. Each fitting bridge reads them off its ``raw``; the
        bridge is found by the toolbox the method names.

        Returns:
            Dict[str, np.ndarray]: 'ppm', rising, and 'data', 'fit' and 'baseline' along
            it, real, in the method's own units.

        Raises:
            ValueError: If the method keeps no curves - a model, say - or the result was
                loaded without its ``raw``.
        """
        toolbox = self.method.partition(":")[0]
        try:
            drawn = importlib.import_module(f"mrsuite.bridges.{toolbox}").curves
        except (ModuleNotFoundError, AttributeError):
            raise ValueError(f"{self.method} keeps no curves") from None
        if self.raw is None:
            raise ValueError(f"this {self.method} result was loaded without its raw "
                             "object, which holds the curves")
        curves = {k: np.asarray(v, dtype=float).ravel() for k, v in drawn(self.raw).items()}
        order = np.argsort(curves["ppm"])
        return {k: v[order] for k, v in curves.items()}

    #*************#
    #   to dict   #
    #*************#
    def to_dict(self) -> Dict[str, Any]:
        """
        Every field but ``raw``, as JSON takes it. Python's json writes a float as its
        shortest exact repr, so finite values read back to the bit; NaN and infinities it
        writes as NaN and Infinity, which Python reads back (a NaN without its sign) but
        a strict JSON reader refuses.

        Returns:
            Dict[str, Any]: Field name to value.
        """
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name != "raw"}

    #***************#
    #   from dict   #
    #***************#
    @classmethod
    def from_dict(cls, values: Dict[str, Any], raw: Any = None) -> "FitResult":
        """
        A result back from ``to_dict``.

        Args:
            values (Dict[str, Any]): What ``to_dict`` returned.
            raw (Any): The method's own object, if it was kept.

        Returns:
            FitResult: The result.
        """
        return cls(**values, raw=raw)


#**********#
#   save   #
#**********#
def save(results: Sequence[FitResult], path: str, raw: bool = True) -> None:
    """
    Results to a JSON file and, with ``raw``, the methods' own objects pickled beside it,
    as ``path`` + '.pkl'. The JSON reads back without MRSuite; the pickle only where the
    libraries that made the objects are installed, at the same versions.

    Each file is written whole or not at all, and the JSON last: where it exists, both
    are complete. The pickle holds the JSON's SHA-256, so a pickle from another save
    (one cut short before its JSON was written) is not read with these results.

    Args:
        results (Sequence[FitResult]): What to save.
        path (str): The JSON file.
        raw (bool): Keep the methods' own objects too.
    """
    content = json.dumps([r.to_dict() for r in results]).encode()
    if raw:
        _write(path + ".pkl", pickle.dumps((hashlib.sha256(content).hexdigest(),
                                            [r.raw for r in results]), pickle.HIGHEST_PROTOCOL))
    elif os.path.exists(path + ".pkl"):          # not to be read back with these results
        os.remove(path + ".pkl")
    _write(path, content)


#**********#
#   load   #
#**********#
def load(path: str, raw: bool = True) -> List[FitResult]:
    """
    Results back from ``save``, with the methods' own objects when they were kept.

    Args:
        path (str): The JSON file.
        raw (bool): Read the methods' own objects too, where they were kept.

    Returns:
        List[FitResult]: The results, in the order they were saved.

    Raises:
        ValueError: If the pickle beside it was saved with other results.
    """
    with open(path, "rb") as fh:
        content = fh.read()
    values = json.loads(content)
    raws = [None] * len(values)
    if raw and os.path.exists(path + ".pkl"):
        with open(path + ".pkl", "rb") as fh:
            sha256, raws = pickle.load(fh)
        if sha256 != hashlib.sha256(content).hexdigest():
            raise ValueError(f"{path}.pkl was saved with other results than {path}")
    return [FitResult.from_dict(v, raw) for v, raw in zip(values, raws, strict=True)]


#***********#
#   write   #
#***********#
def _write(path: str, content: bytes) -> None:
    """A file written whole or not at all: to a temporary name, then renamed."""
    with open(path + ".tmp", "wb") as fh:
        fh.write(content)
    os.replace(path + ".tmp", path)
