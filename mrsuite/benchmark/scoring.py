####################################################################################################
#                                            scoring.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-01                                                                              #
#                                                                                                  #
# Purpose: Scores fits against ground truth on MRSValidate. Both sides become its ObservationSets  #
#          on one named reference - water, tCr, the optimal scale (MOSAE) or none - since which    #
#          one is right depends on the question, and every metric runs as a step of its analysis   #
#          plan, so the numbers, their status and the run's manifest are MRSValidate's own.        #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import tempfile
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
try:
    from mrsvalidate import (ObservationSet, ParameterRole, ParameterSchema, ParameterSpec,
                             align_observations)
    from mrsvalidate.analysis import descriptive
    from mrsvalidate.pipeline import AnalysisStep, RunConfig, run_analysis_plan
    from mrsvalidate.results import IntervalResult, ScalarMetricResult
except ModuleNotFoundError as err:   # not on PyPI yet, so not a dependency
    if (err.name or "").partition(".")[0] != "mrsvalidate":   # installed, but broken
        raise
    raise ImportError("MRSValidate is not installed: pip install "
                      "git+https://github.com/JohnLaMaster/MRSValidate.git") from err

from mrsuite.core.result import FitResult

# What fits are scored on. The right one depends on the question, so none is a default.
REFERENCES = ("none", "water", "tCr", "optimal")

# MRSValidate analyses that fit score's call but say nothing of the fits: standardize gives
# z-scores (arrays a table leaves out) and the truth's own location and scale; a coefficient
# of variation of signed errors divides by their mean, near 0 for fits without bias, so it
# swings and flips sign (as MRSValidate's own cv warns).
NOT_SCORES = ("standardization.standardize", "descriptive.cv", "descriptive.robust_cv")

# Total creatine, summed from the singles the truth has; the MRSI challenge has no PCr.
TCR = ("Cr", "PCr")

# Moieties the MRSI challenge splits off a molecule into basis functions of their own, each
# carrying the molecule's amplitude in its truth: scored as that molecule, not twice.
MOIETIES = {"Cr39": "Cr", "Lac41": "Lac", "PE40": "PE"}


#******************#
#   observations   #
#******************#
def observations(results: Sequence[FitResult], truth: Sequence[Dict[str, float]],
                 reference: str, scaling: str = "raw",
                 metabolites: Optional[Sequence[str]] = None,
                 ids: Optional[Sequence[str]] = None) -> Tuple[ObservationSet, ObservationSet]:
    """
    Ground truth and one method's fits as MRSValidate ObservationSets on one reference.

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
        ids (Optional[Sequence[str]]): One sample ID per spectrum; by default its position.

    Returns:
        Tuple[ObservationSet, ObservationSet]: The truth, then the fits.

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
    metabolites = _singles(fits, truth) if metabolites is None else metabolites
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

    unit = "mM" if reference == "water" else None
    schema = ParameterSchema(tuple(ParameterSpec(m, ParameterRole.CONCENTRATION, unit,
                                                 metabolite=m) for m in metabolites))
    ids = np.array(range(len(truth)) if ids is None else ids).astype(str)
    return (ObservationSet(ids, true, schema, model_id="truth"),
            ObservationSet(ids, estimate, schema, model_id=results[0].method))


#************#
#   ratios   #
#************#
def ratios(results: Sequence[FitResult], to: str = "tCr",
           scaling: Optional[str] = None) -> List[FitResult]:
    """
    Each fit's concentrations as ratios to total creatine, Cr + PCr summed from the singles
    as ``observations`` sums them, the most common reference (Near et al., NMR Biomed 2021):
    a result per fit under the scaling 'tCr' alone, keeping the method, its QC and its
    provenance.

    Args:
        results (Sequence[FitResult]): One fit per spectrum.
        to (str): 'tCr', the one reference so far.
        scaling (Optional[str]): The fits' scaling read; a ratio is the same on any
            absolute one, so by default 'raw', else 'molarity', else the first a fit has.

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
        total = sum(conc[m] for m in parts)
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


#***********#
#   score   #
#***********#
def score(methods: Dict[str, Sequence[FitResult]], truth: Sequence[Dict[str, float]],
          references: Sequence[str], metrics: Dict[str, Callable[..., Any]],
          output_dir: Optional[Union[str, Path]] = None, scaling: str = "raw",
          metabolites: Optional[Sequence[str]] = None,
          ids: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """
    Every metric for every method, reference and metabolite, and pooled over the
    metabolites, run as one MRSValidate analysis plan.

    Every method is scored on the same metabolites, so pooled scores compare like
    with like. Pooled ('all') takes each metabolite of each spectrum as a sample of its
    own, so 'mae' pooled on the 'optimal' reference is the MOSAE as a mean over the
    scored metabolites; the OoD paper's code and its printed Eq. 16 divide by all M
    basis functions, the macromolecules counted as zeros, which gives (scored / M) of
    this.

    Args:
        methods (Dict[str, Sequence[FitResult]]): Name -> one fit per spectrum.
        truth (Sequence[Dict[str, float]]): Ground truth per spectrum.
        references (Sequence[str]): What to score on, see observations. There is no
            default, as the right one depends on the question.
        metrics (Dict[str, Callable[..., Any]]): Name -> an analysis, as ``analyses``
            lists them: one comparing the truth with the fits, f(truth, fits,
            sample_ids) under its own argument names (agreement.lin_ccc, errors.rmse,
            method_comparison.bland_altman, ...), or one describing a sample, f(x,
            sample_ids), run on the errors, fits minus truth (descriptive.median,
            normality.shapiro_wilk, ...); mae and bias here. Its further arguments by
            ``functools.partial``.
        output_dir (Optional[Union[str, Path]]): Where MRSValidate writes the run, in a
            folder of its own: its manifest, each result as JSON, and a summary; when None,
            a temporary folder, gone once the table is read.
        scaling (str): As in observations.
        metabolites (Optional[Sequence[str]]): As in observations; by default the
            singles every method's fits and the truth have.
        ids (Optional[Sequence[str]]): As in observations.

    Returns:
        pd.DataFrame: One row per method, reference, metabolite and metric, with
        MRSValidate's status, sample counts and the result's own fields (value, rho_c,
        r, ...); 'analysis', the MRSValidate analysis that ran; and 'score', the
        result's first number as MRSValidate declares its fields (value, rho_c, bias,
        slope, r, statistic, ...), NaN for an interval, which has two. A result's own 'method' field, which MRSValidate's
        summary writes over the analysis' name, is only in the run's JSON.
    """
    metabolites = _metabolites(methods, truth, references, scaling, metabolites)
    keys, steps = {}, []
    for method, results in methods.items():
        for reference in references:
            columns = _columns(results, truth, reference, scaling, metabolites, ids)
            for metabolite, (t, f, s) in columns.items():
                for metric, analysis in metrics.items():   # numbered, so names never clash
                    name = ".".join((str(len(steps)), method, reference, metabolite, metric))
                    keys[name] = (method, reference, metabolite, metric)
                    steps.append(AnalysisStep(name, analysis, _given(analysis, t, f, s)))

    # MRSValidate's own run IDs are to the second; two runs in one second would share one
    run_id = datetime.now(timezone.utc).strftime("run-%Y-%m-%dT%H%M%S.%fZ")
    with tempfile.TemporaryDirectory(prefix="mrsuite-score-") as temporary:
        run = run_analysis_plan(RunConfig(output_dir=output_dir or temporary, steps=tuple(steps),
                                          fail_fast=True, run_id=run_id))
        # its summary's 'method' is a result's own 'method' field where it has one
        table = (pd.read_csv(run.run_dir / "metrics" / "summary.csv",
                             float_precision="round_trip").drop(columns="method"))
    table["analysis"] = [run.results[name].meta.method for name in table["name"]]
    table["score"] = [_first(run.results[name]) for name in table["name"]]
    index = pd.DataFrame([(name, *key) for name, key in keys.items()],
                         columns=["name", "method", "reference", "metabolite", "metric"])
    return index.merge(table, on="name").drop(columns="name")


#**************#
#   analyses   #
#**************#
def analyses() -> Dict[str, Callable[..., Any]]:
    """
    Every MRSValidate analysis score runs, by module and name ('errors.rmse'): those
    comparing the truth with the fits, f(reference, fits, sample_ids, ...), and those
    describing one sample, f(x, sample_ids, ...), which score runs on the errors; their
    further arguments by name only. Left out are those that need more than fits and
    truth (uncertainties, repeated fits, clusters) or other samples (two methods'
    residuals, independent samples), those returning something else than a result, and
    those that say nothing of the fits here (NOT_SCORES).

    Returns:
        Dict[str, Callable[..., Any]]: In MRSValidate's module order.
    """
    import importlib
    import inspect
    import pkgutil

    import mrsvalidate.analysis

    found = {}
    for part in pkgutil.iter_modules(mrsvalidate.analysis.__path__):
        if part.name.startswith("_"):
            continue
        module = importlib.import_module(f"mrsvalidate.analysis.{part.name}")
        for name, f in vars(module).items():
            if name.startswith("_") or not inspect.isfunction(f) or \
                    f.__module__ != module.__name__:
                continue
            parameters = inspect.signature(f).parameters.values()
            given = [p.name for p in parameters if p.kind in (p.POSITIONAL_ONLY,
                                                              p.POSITIONAL_OR_KEYWORD)]
            returns = str(inspect.signature(f).return_annotation)
            key = f"{part.name}.{name}"
            if (given[:1] == ["reference"] and given[2:] == ["sample_ids"] or
                    given == ["x", "sample_ids"]) and returns.strip("'\"").endswith("Result") \
                    and key not in NOT_SCORES:
                found[key] = f
    return found


#************#
#   figure   #
#************#
def figure(methods: Dict[str, Sequence[FitResult]], truth: Sequence[Dict[str, float]],
           reference: str, analysis: Callable[..., Any], plot: Callable[..., Any],
           metabolite: str = "all", scaling: str = "raw",
           metabolites: Optional[Sequence[str]] = None) -> Any:
    """
    One MRSValidate plot per method, side by side: its analysis of the fits against the
    truth on one reference, drawn by its plot, e.g. method_comparison.bland_altman with
    viz.agreement.plot_bland_altman, or regression.regression_metrics with
    viz.scatter.plot_regression_result.

    Args:
        methods (Dict[str, Sequence[FitResult]]): Name -> one fit per spectrum.
        truth (Sequence[Dict[str, float]]): Ground truth per spectrum.
        reference (str): What to score on, see observations.
        analysis (Callable[..., Any]): The analysis, as score takes it.
        plot (Callable[..., Any]): Its plot, called as plot(result, ax=ax).
        metabolite (str): Which one, or 'all' pooled, as score pools them.
        scaling, metabolites: As in score.

    Returns:
        matplotlib.figure.Figure: The figure, made without pyplot, so drawing it in a
        thread is safe and nothing is left open.
    """
    from matplotlib.figure import Figure

    metabolites = _metabolites(methods, truth, [reference], scaling, metabolites)
    fig = Figure(figsize=(4.5 * len(methods), 4), layout="constrained")
    for ax, (method, results) in zip(np.atleast_1d(fig.subplots(1, len(methods))),
                                     methods.items()):
        columns = _columns(results, truth, reference, scaling, metabolites)
        if metabolite not in columns:
            raise ValueError(f"no {metabolite!r} scored; one of {list(columns)}")
        plot(analysis(**_given(analysis, *columns[metabolite])), ax=ax)
        ax.set_title(f"{method} · {metabolite}")
    return fig


#*****************#
#   metabolites   #
#*****************#
def _metabolites(methods: Dict[str, Sequence[FitResult]], truth: Sequence[Dict[str, float]],
                 references: Sequence[str], scaling: str,
                 metabolites: Optional[Sequence[str]]) -> Sequence[str]:
    """What every method is scored on: as given, or the singles all fits and the truth
    have."""
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


#*************#
#   columns   #
#*************#
def _columns(results: Sequence[FitResult], truth: Sequence[Dict[str, float]], reference: str,
             scaling: str, metabolites: Sequence[str], ids: Optional[Sequence[str]] = None) \
        -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """One method's (truth, fits, sample IDs) per metabolite, paired, and pooled as 'all'."""
    paired = align_observations(*observations(results, truth, reference, scaling,
                                              metabolites, ids), pairing="paired")
    true, fit = paired.reference, paired.candidate
    names, sample = true.parameters.names, true.sample_ids
    columns = {m: (true.column(m), fit.column(m), sample) for m in names}
    columns["all"] = (np.concatenate([true.column(m) for m in names]),
                      np.concatenate([fit.column(m) for m in names]),
                      np.array([f"{s}/{m}" for m in names for s in sample]))
    return columns


#***********#
#   given   #
#***********#
def _given(analysis: Callable[..., Any], true: np.ndarray, fit: np.ndarray,
           sample: np.ndarray) -> Dict[str, Any]:
    """An analysis' arguments by its own names: the truth and the fits, or the errors."""
    import inspect

    given = [p.name for p in inspect.signature(analysis).parameters.values()
             if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    if given[1] == "sample_ids":
        return {given[0]: fit - true, "sample_ids": sample}
    return {given[0]: true, given[1]: fit, "sample_ids": sample}


#***********#
#   first   #
#***********#
def _first(result: Any) -> float:
    """A result's first number, in the order MRSValidate declares its fields; NaN if none,
    or for an interval, which has two."""
    import dataclasses

    if isinstance(result, IntervalResult):
        return float("nan")
    for f in dataclasses.fields(result):
        value = getattr(result, f.name)
        if f.name != "meta" and isinstance(value, (int, float, np.number)) and \
                not isinstance(value, (bool, np.bool_)):
            return float(value)
    return float("nan")


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


#*********#
#   mae   #
#*********#
def mae(reference: np.ndarray, prediction: np.ndarray,
        sample_ids: np.ndarray) -> ScalarMetricResult:
    """
    Mean absolute error, as MRSValidate's own toy example computes it (it has no MAE
    analysis); pooled on the 'optimal' reference, the MOSAE (see score).

    Args:
        reference (np.ndarray): True values.
        prediction (np.ndarray): Fitted values, paired with them.
        sample_ids (np.ndarray): One ID per pair.

    Returns:
        ScalarMetricResult: The mean, with MRSValidate's status and counts.
    """
    return descriptive.mean(np.abs(prediction - reference), sample_ids)


#**********#
#   bias   #
#**********#
def bias(reference: np.ndarray, prediction: np.ndarray,
         sample_ids: np.ndarray) -> ScalarMetricResult:
    """
    Mean signed error, prediction minus truth: the systematic part that MAE hides.

    Args:
        reference (np.ndarray): True values.
        prediction (np.ndarray): Fitted values, paired with them.
        sample_ids (np.ndarray): One ID per pair.

    Returns:
        ScalarMetricResult: The mean, with MRSValidate's status and counts.
    """
    return descriptive.mean(prediction - reference, sample_ids)


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
