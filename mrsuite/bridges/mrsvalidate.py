####################################################################################################
#                                          mrsvalidate.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 11/11/26                                                                                #
#                                                                                                  #
# Purpose: Fits scored against the ground truth by MRSValidate (LaMaster et al. 2026). Both       #
#          sides, paired on a reference by MRSuite's benchmark package, become its                 #
#          ObservationSets, every metric runs as a step of its analysis plan, so the numbers,      #
#          their status and the run's manifest are MRSValidate's own, and its plots draw them.     #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

try:
    from mrsvalidate import (ObservationSet, ParameterRole, ParameterSchema, ParameterSpec,
                             align_observations)
    from mrsvalidate.analysis import descriptive
    from mrsvalidate.pipeline import AnalysisStep, RunConfig, run_analysis_plan
    from mrsvalidate.results import IntervalResult, ScalarMetricResult
except ModuleNotFoundError as err:
    if (err.name or "").partition(".")[0] != "mrsvalidate":   # installed, but broken
        raise
    raise ImportError("MRSValidate is not installed: pip install "
                      "git+https://github.com/JohnLaMaster/MRSValidate.git") from err

from mrsuite.benchmark import scoring
from mrsuite.core.result import FitResult

PROGRAM = "MRSValidate"

# MRSValidate analyses that fit score's call but say nothing of the fits: standardize gives
# z-scores (arrays a table leaves out) and the truth's own location and scale; a coefficient
# of variation of signed errors divides by their mean, near 0 for fits without bias, so it
# swings and flips sign (as MRSValidate's own cv warns).
NOT_SCORES = ("standardization.standardize", "descriptive.cv", "descriptive.robust_cv")


#******************#
#   observations   #
#******************#
def observations(results: Sequence[FitResult], truth: Sequence[Dict[str, float]],
                 reference: str, scaling: str = "raw",
                 metabolites: Optional[Sequence[str]] = None,
                 ids: Optional[Sequence[str]] = None) -> Tuple[ObservationSet, ObservationSet]:
    """
    Ground truth and one method's fits as MRSValidate ObservationSets on one reference,
    paired as ``mrsuite.benchmark.pair`` pairs them.

    Args:
        results (Sequence[FitResult]): One fit per spectrum.
        truth (Sequence[Dict[str, float]]): Ground truth per spectrum, in the same order.
        reference (str): One of ``mrsuite.benchmark.REFERENCES``.
        scaling (str): As in pair.
        metabolites (Optional[Sequence[str]]): As in pair.
        ids (Optional[Sequence[str]]): One sample ID per spectrum; by default its position.

    Returns:
        Tuple[ObservationSet, ObservationSet]: The truth, then the fits.

    Raises:
        ValueError, KeyError: As pair raises them.
    """
    true, estimate, names = scoring.pair(results, truth, reference, scaling, metabolites)
    unit = "mM" if reference == "water" else None
    schema = ParameterSchema(tuple(ParameterSpec(m, ParameterRole.CONCENTRATION, unit,
                                                 metabolite=m) for m in names))
    ids = np.array(range(len(truth)) if ids is None else ids).astype(str)
    return (ObservationSet(ids, true, schema, model_id="truth"),
            ObservationSet(ids, estimate, schema, model_id=results[0].method))


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
        references (Sequence[str]): What to score on, see ``mrsuite.benchmark.pair``.
            There is no default, as the right one depends on the question.
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
        slope, r, statistic, ...), NaN for an interval, which has two. A result's own
        'method' field, which MRSValidate's summary writes over the analysis' name, is
        only in the run's JSON.
    """
    metabolites = scoring.metabolites(methods, truth, references, scaling, metabolites)
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
        reference (str): What to score on, see ``mrsuite.benchmark.pair``.
        analysis (Callable[..., Any]): The analysis, as score takes it.
        plot (Callable[..., Any]): Its plot, called as plot(result, ax=ax).
        metabolite (str): Which one, or 'all' pooled, as score pools them.
        scaling (str): As in score.
        metabolites (Optional[Sequence[str]]): As in score.

    Returns:
        matplotlib.figure.Figure: The figure, made without pyplot, so drawing it in a
        thread is safe and nothing is left open.
    """
    from matplotlib.figure import Figure

    metabolites = scoring.metabolites(methods, truth, [reference], scaling, metabolites)
    fig = Figure(figsize=(4.5 * len(methods), 4), layout="constrained")
    for ax, (method, results) in zip(np.atleast_1d(fig.subplots(1, len(methods))),
                                     methods.items()):
        columns = _columns(results, truth, reference, scaling, metabolites)
        if metabolite not in columns:
            raise ValueError(f"no {metabolite!r} scored; one of {list(columns)}")
        plot(analysis(**_given(analysis, *columns[metabolite])), ax=ax)
        ax.set_title(f"{method} · {metabolite}")
    return fig


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
#   version   #
#*************#
def version() -> str:
    """MRSValidate's version, as installed."""
    from importlib.metadata import version as installed

    return installed("mrsvalidate")


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
