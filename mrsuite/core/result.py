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
from dataclasses import dataclass, field
from typing import Any, Dict, List


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
        raw (Any): The method's own result object, untouched. Anything not captured
            above lives here, including the curves, which are too large to persist.
    """

    method: str
    metabolites: List[str] = field(default_factory=list)
    concentrations: Dict[str, Dict[str, float]] = field(default_factory=dict)
    uncertainties: Dict[str, Dict[str, float]] = field(default_factory=dict)
    qc: Dict[str, float] = field(default_factory=dict)
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
