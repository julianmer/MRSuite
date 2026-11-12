####################################################################################################
#                                            scoring.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-01                                                                              #
#                                                                                                  #
# Purpose: How fits meet the ground truth: MRSuite's own part of a benchmark. Fits report          #
#          amplitudes on a scale of their own, so the two sides are paired on one named reference  #
#          - water, tCr, the optimal scale (MOSAE) or none - since which one is right depends on   #
#          the question. The statistics on the pairs are MRSValidate's (bridges/mrsvalidate.py).   #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from mrsuite.core.result import FitResult

# What fits are scored on. The right one depends on the question, so none is a default.
REFERENCES = ("none", "water", "tCr", "optimal")

# Total creatine, summed from the singles the truth has; the MRSI challenge has no PCr.
TCR = ("Cr", "PCr")

# Moieties the MRSI challenge splits off a molecule into basis functions of their own, each
# carrying the molecule's amplitude in its truth: scored as that molecule, not twice.
MOIETIES = {"Cr39": "Cr", "Lac41": "Lac", "PE40": "PE"}


#**********#
#   pair   #
#**********#
def pair(results: Sequence[FitResult], truth: Sequence[Dict[str, float]], reference: str,
         scaling: str = "raw", metabolites: Optional[Sequence[str]] = None) \
        -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Ground truth and one method's fits paired on one reference, as (spectra, metabolites)
    arrays.

    Fits report amplitudes on a scale of their own, so they meet the truth on a
    reference, and the literature picks it per question:

    - 'water': the fits' water-scaled concentrations ('molarity', mM, by each tool's own
      water constants) against the truth in mM, the 2016 fitting challenge's absolute
      comparison (Marjanska et al., MRM 2022);
    - 'tCr': both sides as ratios to total creatine, Cr + PCr summed from the singles,
      the challenge's other comparison and the most common metabolite reference (Near
      et al., NMR Biomed 2021); where tCr is one metabolite (the MRSI challenge has no
      PCr), that one is not scored, as it is 1 on both sides;
    - 'optimal': each fit times its optimal scale against its truth (optimal_scale), the
      comparison behind MOSAE (Merkofer et al., arXiv:2511.23135), which needs neither
      a water reference nor tCr;
    - 'none': both as given, for fits already on the truth's scale.

    Only single metabolites are scored. Sums are not, whether a tool's own (TARQUIN's
    TNAA, LCModel's Cr+PCr) or the truth's (the challenge's 'Cr + PCr'); coverage lists
    a tool's. A molecule split into moieties (MOIETIES) is scored once, as the mean of
    its parts on either side.

    Args:
        results (Sequence[FitResult]): One fit per spectrum.
        truth (Sequence[Dict[str, float]]): Ground truth per spectrum, in the same order.
        reference (str): One of REFERENCES.
        scaling (str): The fits' scaling read for 'tCr', 'optimal' and 'none'; 'water'
            reads 'molarity'.
        metabolites (Optional[Sequence[str]]): What to score, by molecule ('Cr', not
            'Cr39'). By default the singles every fit and every truth has,
            macromolecules left out as MOSAE and the challenge do.

    Returns:
        Tuple[np.ndarray, np.ndarray, List[str]]: The truth, the fits, and the
        metabolites, one column each.

    Raises:
        ValueError: For an unknown reference, fits and truths that do not line up, or
            nothing to score.
        KeyError: For a metabolite, or a part of tCr, that one side does not have.
    """
    if reference not in REFERENCES:
        raise ValueError(f"unknown reference '{reference}'; choose from {REFERENCES}")
    if len(results) != len(truth):
        raise ValueError(f"{len(results)} fits against {len(truth)} ground truths; "
                         f"they must line up spectrum for spectrum.")

    fits = [_read(r, reference, scaling) for r in results]
    truth = [_molecules(t) for t in truth]
    metabolites = _singles(fits, truth) if metabolites is None else list(metabolites)
    tcr = [m for m in TCR if all(m in t for t in truth)]   # both sides sum the same parts
    if reference == "tCr" and not tcr:
        raise KeyError(f"the truth has no {' or '.join(TCR)} to make tCr; "
                       f"it has {sorted(set().union(*truth))}")
    if reference == "tCr" and len(tcr) == 1:   # it is 1 on both sides, always
        metabolites = [m for m in metabolites if m not in tcr]
    if len(metabolites) == 0:
        raise ValueError(f"nothing to score: every fit has {_everywhere(fits)}, "
                         f"every truth {_everywhere(truth)}")

    estimate, true = _matrix(fits, metabolites, "fit"), _matrix(truth, metabolites, "truth")
    if reference == "tCr":
        estimate = estimate / _matrix(fits, tcr, "fit").sum(axis=1)[:, None]
        true = true / _matrix(truth, tcr, "truth").sum(axis=1)[:, None]
    elif reference == "optimal":
        estimate = estimate * optimal_scale(estimate, true)[:, None]
    return true, estimate, metabolites


#*****************#
#   metabolites   #
#*****************#
def metabolites(methods: Dict[str, Sequence[FitResult]], truth: Sequence[Dict[str, float]],
                references: Sequence[str], scaling: str = "raw",
                metabolites: Optional[Sequence[str]] = None) -> Sequence[str]:
    """
    What every method is scored on: as given, or the singles all fits of all methods and
    the truth have, so pooled scores compare like with like.

    Args:
        methods (Dict[str, Sequence[FitResult]]): Name -> one fit per spectrum.
        truth (Sequence[Dict[str, float]]): Ground truth per spectrum.
        references (Sequence[str]): The references the fits are read on, see pair.
        scaling (str): As in pair.
        metabolites (Optional[Sequence[str]]): As in pair.

    Raises:
        ValueError: When nothing is there to score.
    """
    if metabolites is not None:
        return metabolites
    fits = [_read(r, reference, scaling) for results in methods.values()
            for reference in references for r in results]
    molecules = [_molecules(t) for t in truth]
    found = _singles(fits, molecules)
    if len(found) == 0:
        raise ValueError(f"nothing to score: every fit of every method has "
                         f"{_everywhere(fits)}, every truth {_everywhere(molecules)}")
    return found


#************#
#   ratios   #
#************#
def ratios(results: Sequence[FitResult], to: str = "tCr",
           scaling: Optional[str] = None) -> List[FitResult]:
    """
    Each fit's concentrations as ratios to total creatine, Cr + PCr summed from the singles
    as ``pair`` sums them, the most common reference (Near et al., NMR Biomed 2021):
    a result per fit under the scaling 'tCr' alone, keeping the method, its QC and its
    provenance.

    Args:
        results (Sequence[FitResult]): One fit per spectrum.
        to (str): 'tCr', the one reference so far.
        scaling (Optional[str]): The fits' scaling read; a ratio is the same on any
            absolute one, so by default 'raw', else 'molarity', else the first a fit has.

    A fit that found no creatine at all (Cr + PCr = 0) has no ratios: every one NaN, as a
    metabolite a fit failed on is.

    Raises:
        ValueError: For another reference.
        KeyError: Where a fit has neither Cr nor PCr, or not the scaling.
    """
    if to != "tCr":
        raise ValueError(f"ratios go to 'tCr', not {to!r}")
    out = []
    for r in results:
        own = scaling or next((s for s in ("raw", "molarity") if s in r.concentrations),
                              next(iter(r.concentrations), "raw"))
        conc = _molecules(r.conc(own))
        parts = [m for m in TCR if np.isfinite(conc.get(m, np.nan))]
        if not parts:
            raise KeyError(f"{r.method} fitted no {' or '.join(TCR)} to reference to")
        total = sum(conc[m] for m in parts) or np.nan   # fitted none: no ratio to take
        out.append(replace(r, concentrations={to: {m: v / total for m, v in conc.items()}},
                           uncertainties={}, raw=None))
    return out


#*******************#
#   optimal scale   #
#*******************#
def optimal_scale(estimate: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """
    Each spectrum's optimal scale: the w minimising sum_m |w * estimate_m - truth_m|
    (Merkofer et al., arXiv:2511.23135, Eq. 15), kept >= 0 as the paper's code keeps
    it, which turns MAE into MOSAE (Eq. 16).

    The sum equals sum_m |estimate_m| * |w - truth_m / estimate_m|, so its minimum is
    the weighted median of those ratios, found exactly here where the paper's code
    searched for it numerically. Zero estimates and non-finite pairs (a metabolite a
    fit failed on) carry no weight, and a spectrum with none left gets 0, as every
    scale fits it equally well.

    Args:
        estimate (np.ndarray): (spectra, metabolites) fitted values.
        truth (np.ndarray): (spectra, metabolites) true values.

    Returns:
        np.ndarray: (spectra,) scales.
    """
    scales = np.zeros(len(estimate))
    for i, (e, t) in enumerate(zip(estimate, truth)):
        keep = (e != 0) & np.isfinite(e) & np.isfinite(t)
        if keep.any():
            ratio, weight = t[keep] / e[keep], np.abs(e[keep])
            order = np.argsort(ratio)
            below = np.cumsum(weight[order])
            scales[i] = ratio[order][np.searchsorted(below, below[-1] / 2)]
    return np.clip(scales, 0, None)   # the sum is convex, so a negative optimum clips to 0


#**************#
#   coverage   #
#**************#
def coverage(results: Sequence[FitResult], truth: Sequence[Dict[str, float]],
             scaling: str = "raw") -> Dict[str, List[str]]:
    """
    What the fits and the truth do not have in common.

    Scores cover what both have, so a method that fits half the metabolites would
    otherwise look strong; 'extra' also lists a tool's own sums, which are not scored.

    Args:
        results (Sequence[FitResult]): One fit per spectrum.
        truth (Sequence[Dict[str, float]]): Ground truth per spectrum.
        scaling (str): Which of the fits' scalings to read.

    Returns:
        Dict[str, List[str]]: 'missing' - in the truth but not in every fit and every
        truth, which keeps it from being scored; 'extra' - fitted but not in the truth.
    """
    fits = [_molecules(r.conc(scaling)) for r in results]
    truth = [_molecules(t) for t in truth]
    known = set().union(*truth)
    return {"missing": sorted(known - set(_everywhere([*fits, *truth]))),
            "extra": sorted(set().union(*fits) - known)}


#*************#
#   singles   #
#*************#
def _singles(fits: Sequence[Dict[str, float]], truth: Sequence[Dict[str, float]]) -> List[str]:
    """The single metabolites every fit and every truth has: sums have a '+', and the
    macromolecules go by the MLP loss' rule ('Mac', 'mm_0_9', ...)."""
    return [m for m in _everywhere([*fits, *truth])
            if "+" not in m and "mm" not in m.lower() and "mac" not in m.lower()]


#****************#
#   everywhere   #
#****************#
def _everywhere(concs: Sequence[Dict[str, float]]) -> List[str]:
    """The names every one of these spectra has."""
    return sorted(set(concs[0]).intersection(*concs))


#**********#
#   read   #
#**********#
def _read(result: FitResult, reference: str, scaling: str) -> Dict[str, float]:
    """The concentrations a reference reads off a fit: the water-scaled ones for 'water'."""
    return _molecules(result.conc("molarity" if reference == "water" else scaling))


#***************#
#   molecules   #
#***************#
def _molecules(conc: Dict[str, float]) -> Dict[str, float]:
    """One value per molecule: its moieties, those fitted, averaged under its name, so a
    moiety a fit failed on does not cost the molecule."""
    parts = {}
    for name, value in conc.items():
        parts.setdefault(MOIETIES.get(name, name), []).append(value)
    fitted = {name: [v for v in values if np.isfinite(v)] for name, values in parts.items()}
    return {name: float(np.mean(values)) if values else np.nan
            for name, values in fitted.items()}


#************#
#   matrix   #
#************#
def _matrix(concs: Sequence[Dict[str, float]], names: Sequence[str], what: str) -> np.ndarray:
    """(spectra, names) values, or a KeyError saying what one spectrum lacks."""
    for i, conc in enumerate(concs):
        missing = [m for m in names if m not in conc]
        if missing:
            raise KeyError(f"{what} {i} has no {missing}; it has {sorted(conc)}")
    return np.array([[conc[m] for m in names] for conc in concs], dtype=float)
