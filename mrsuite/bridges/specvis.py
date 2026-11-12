####################################################################################################
#                                            specvis.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 10/11/26                                                                                #
#                                                                                                  #
# Purpose: Fits compared with each other, without a truth, by SpecVis (Zöllner et al., NMR        #
#          Biomed 2021): the Osprey group's R functions over the results of several linear-        #
#          combination algorithms, vendors or groups, run as published in an R process of their   #
#          own. Its raincloud, box, violin, correlation and Bland-Altman plots and its             #
#          statistics take one table, which MRSuite writes from its fit results, or which SpecVis  #
#          reads from Osprey's, LCModel's and TARQUIN's own files.                                 #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import os
import shutil
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

from mrsuite import externals, r
from mrsuite.benchmark import scoring
from mrsuite.core.result import FitResult
from mrsuite.octave import scratch

PROGRAM = "SpecVis"
EXTERNAL = "specvis"
DRIVER = os.path.join(r.HERE, "specvis.R")

Limits = Optional[Sequence[Tuple[float, float]]]

# the names SpecVis's importers give the sums (Osprey's), so one table serves every fitter:
# LCModel's and TARQUIN's own names for them, and the singles they are made of (a basis set
# may spell phosphocholine PCho, as the 2016 challenge's does)
RENAMED = {"NAA+NAAG": "tNAA", "Cr+PCr": "tCr", "PCh+GPC": "tCho", "GPC+PCh": "tCho",
           "Glu+Gln": "Glx", "Glc+Tau": "GlcTau", "-CrCH2": "CrCH2", "PCho": "PCh",
           "TNAA": "tNAA", "TCr": "tCr", "TCho": "tCho", "Glth": "GSH", "PEth": "PE"}
SUMS = {"tNAA": ("NAA", "NAAG"), "tCr": ("Cr", "PCr"), "tCho": ("GPC", "PCh"),
        "Glx": ("Glu", "Gln")}

# the measures SpecVis's own example compares (Big GABA, Zöllner et al. 2021): the default
MEASURES = ("tNAA", "tCho", "Ins", "Glx")


#***********#
#   named   #
#***********#
def _named(conc: Dict[str, float]) -> Dict[str, float]:
    """One fit's concentrations under SpecVis's names, the sums made where missing."""
    out = {RENAMED.get(name, name): float(value) for name, value in conc.items()}
    for name, parts in SUMS.items():
        if name not in out and all(p in out for p in parts):
            out[name] = sum(out[p] for p in parts)
    return out


#***********#
#   table   #
#***********#
def table(methods: Union[Dict[str, Sequence[FitResult]], Sequence[FitResult]],
          scaling: str = "tCr", covariates: Any = None) -> pd.DataFrame:
    """
    SpecVis's table of fits: a row per fit, a column per metabolite under the names its
    importers give (tNAA, tCr, tCho and Glx for the sums, made from the singles where a
    fitter reports none), and a Group column naming each method; covariates beside them,
    as its stat.csv goes.

    Args:
        methods: Name -> one fit per spectrum, the same spectra in the same order under
            every name, since SpecVis pairs the rows; or the fits of one method alone,
            which get no Group.
        scaling (str): 'tCr' for ratios to total creatine, Cr + PCr, as scoring.ratios
            makes them; else a scaling the fits have, e.g. 'raw' or 'molarity'.
        covariates: A table with a row per spectrum (age, site, ...), bound to every
            method's rows.

    Returns:
        pd.DataFrame: The table.

    Raises:
        ValueError: For covariates that are not one row per spectrum.
    """
    named = methods if isinstance(methods, dict) else {None: methods}
    given = None if covariates is None else pd.DataFrame(covariates).reset_index(drop=True)
    frames = []
    for name, results in named.items():
        fits = scoring.ratios(results) if scaling == "tCr" else results
        frame = pd.DataFrame([_named(fit.conc(scaling)) for fit in fits])
        if given is not None:
            if len(given) != len(frame):
                raise ValueError(f"{len(given)} rows of covariates for {len(frame)} fits")
            frame = pd.concat([frame, given], axis=1)
        if name is not None:
            frame["Group"] = name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


#*********#
#   run   #
#*********#
def _run(job: Dict[str, Any], folder: str) -> Dict[str, Any]:
    """One job through SpecVis's driver, in its checkout."""
    return r.run({**job, "checkout": str(externals.ensure(EXTERNAL)), "folder": folder},
                 folder, driver=DRIVER, layer=EXTERNAL)


#**********#
#   load   #
#**********#
def load(path: str, ratios: bool = True, what: str = "amplitudes") -> pd.DataFrame:
    """
    SpecVis's own import: Osprey's results .csv, or a folder of LCModel .coord or TARQUIN
    .csv files, as its table.

    Args:
        path (str): The file, or the folder.
        ratios (bool): Divide by tCr, as SpecVis does by default; else the values as
            the files give them.
        what (str): 'amplitudes'; for LCModel's and TARQUIN's files also 'crlbs' and
            'diagnostics' (FWHM, SNR, the phases), tables of their own.

    Returns:
        pd.DataFrame: A row per file, in SpecVis's order (the files' names).

    Raises:
        ValueError: For a table the files do not give.
    """
    folder = scratch("mrsuite_specvis_")
    try:
        # copied in, so that files kept anywhere are visible in the container too
        given = os.path.join(folder, "given")
        (shutil.copytree if os.path.isdir(path) else shutil.copy)(path, given)
        out = _run({"task": "load", "path": given, "quant": "tCr" if ratios else "amplitudes"},
                   folder)
        if what not in out["tables"]:
            raise ValueError(f"{path} gives {out['tables']}, not {what!r}")
        return pd.read_csv(os.path.join(folder, f"{what}.csv"))
    finally:
        shutil.rmtree(folder, ignore_errors=True)


#**********#
#   plot   #
#**********#
def _plot(func: str, frames: Union[pd.DataFrame, List[pd.DataFrame]], args: Dict[str, Any],
          size: Tuple[float, float], path: Optional[str], as_list: Sequence[str] = ()) -> Any:
    """
    One of SpecVis's plots, drawn by R: its figure as a matplotlib Figure of the picture,
    and, where a path is given, written as SpecVis writes it (.pdf, .png or .svg by the
    extension, the PDF through cairo as its examples).

    Args:
        func (str): The SpecVis function.
        frames: Its data frame, or the list of them it takes.
        args (Dict[str, Any]): Its other arguments, by its names; None for R's NULL.
        size (Tuple[float, float]): The figure's width and height in inches.
        path (Optional[str]): Where to write the figure.
        as_list (Sequence[str]): Arguments R gets as a list rather than a vector, which
            the raincloud, box and violin plots tell their measures apart by.
    """
    from matplotlib.figure import Figure
    from matplotlib.image import imread

    if path and os.path.splitext(path)[1] not in (".pdf", ".png", ".svg"):
        raise ValueError(f"{path}: write a .pdf, .png or .svg")
    for frame in [frames] if isinstance(frames, pd.DataFrame) else frames:
        _has(frame, args["MeasureVar"])
    folder = scratch("mrsuite_specvis_")
    try:
        job: Dict[str, Any] = {"task": "plot", "func": func, "args": args,
                               "as_list": list(as_list), "size": list(size), "dpi": 150}
        if isinstance(frames, pd.DataFrame):
            job["frame"] = _write(frames, folder, "frame")
        else:
            job["frames"] = [_write(f, folder, f"frame{k}") for k, f in enumerate(frames)]
        picture = os.path.join(folder, "figure.png")
        made = os.path.join(folder, "figure" + os.path.splitext(path)[1]) if path else None
        job["outputs"] = [picture] + ([made] if made else [])
        _run(job, folder)
        if made:
            shutil.copy(made, path)
        figure = Figure(figsize=size)
        axes = figure.add_axes([0, 0, 1, 1])
        axes.imshow(imread(picture))
        axes.set_axis_off()
        return figure
    finally:
        shutil.rmtree(folder, ignore_errors=True)


#*********#
#   has   #
#*********#
def _has(frame: pd.DataFrame, measures: Sequence[str]) -> None:
    """A measure the table lacks is said here, not by R."""
    missing = [m for m in measures if m not in frame.columns]
    if missing:
        raise ValueError(f"no column {missing[0]!r} in the table; it has {list(frame.columns)}")


#***********#
#   write   #
#***********#
def _write(frame: pd.DataFrame, folder: str, stem: str) -> str:
    path = os.path.join(folder, f"{stem}.csv")
    frame.to_csv(path, index=False)
    return path


#************#
#   limits   #
#************#
def _limits(limits: Limits, measures: Sequence[str], what: str = "limits") -> Dict[str, Any]:
    """SpecVis's lower and upper limits, one pair per measure."""
    if limits is None:
        return {}
    if len(limits) != len(measures):
        raise ValueError(f"{len(limits)} {what} for {len(measures)} measures")
    return {"lowerLimits": [float(lo) for lo, _ in limits],
            "upperLimits": [float(hi) for _, hi in limits]}


#******************#
#   distribution   #
#******************#
def _distribution(func: str, frame: pd.DataFrame, measures: Sequence[str], quant: str,
                  by: str, limits: Limits, title: Optional[str], columns: int,
                  size: Tuple[float, float], path: Optional[str]) -> Any:
    """The raincloud, box and violin plots, which share their arguments; SpecVis groups
    them by the Group column alone, so ``by`` becomes it. Its raincloud, as published,
    sets its axes from the limits whether given or not, so each measure's range stands
    in where none are."""
    _has(frame, measures)
    if func == "spvs_RainCloud" and limits is None:
        limits = [(float(frame[m].min()), float(frame[m].max())) for m in measures]
    frame = frame.assign(Group=frame[by] if by in frame.columns else "all")
    args = {"Quant": quant, "MeasureVar": list(measures), "GroupVars": ["Group"],
            **_limits(limits, measures), "colNum": int(columns)}
    if title is not None:
        args["title"] = title
    return _plot(func, frame, args, size, path, as_list=("MeasureVar",))


#***************#
#   raincloud   #
#***************#
def raincloud(frame: pd.DataFrame, measures: Sequence[str] = MEASURES, quant: str = "/ [tCr]",
              by: str = "Group", limits: Limits = None, title: Optional[str] = None,
              columns: int = 2, size: Tuple[float, float] = (10, 10),
              path: Optional[str] = None) -> Any:
    """
    SpecVis's raincloud plot (Allen et al. 2019): every fit as a point, the boxplot, the
    distribution and the mean ± SD, per group, with its coefficient of variation; one
    facet per measure.

    Args:
        frame (pd.DataFrame): The table (``table`` or ``load``).
        measures (Sequence[str]): Its columns to draw; by default MEASURES, the four
            SpecVis's own example compares.
        quant (str): The axis label after the measure's name.
        by (str): The column that groups the rows; 'Group', the methods of ``table``.
        limits: The axis limits, one (lower, upper) per measure; by default each
            measure's range.
        title (Optional[str]): The figure's title.
        columns (int): Facets per row.
        size (Tuple[float, float]): Width and height in inches.
        path (Optional[str]): Where to write the figure as SpecVis writes it (.pdf, .png
            or .svg).

    Returns:
        matplotlib.figure.Figure: The figure, as a picture of what R drew.
    """
    return _distribution("spvs_RainCloud", frame, measures, quant, by, limits, title, columns,
                         size, path)


#*************#
#   boxplot   #
#*************#
def boxplot(frame: pd.DataFrame, measures: Sequence[str] = MEASURES, quant: str = "/ [tCr]",
            by: str = "Group", limits: Limits = None, title: Optional[str] = None,
            columns: int = 2, size: Tuple[float, float] = (10, 10),
            path: Optional[str] = None) -> Any:
    """SpecVis's boxplots with every fit as a point; the arguments of ``raincloud``."""
    return _distribution("spvs_Boxplot", frame, measures, quant, by, limits, title, columns,
                         size, path)


#************#
#   violin   #
#************#
def violin(frame: pd.DataFrame, measures: Sequence[str] = MEASURES, quant: str = "/ [tCr]",
           by: str = "Group", limits: Limits = None, title: Optional[str] = None,
           columns: int = 2, size: Tuple[float, float] = (10, 10),
           path: Optional[str] = None) -> Any:
    """SpecVis's violin plots with every fit as a point; the arguments of ``raincloud``."""
    return _distribution("spvs_Violin", frame, measures, quant, by, limits, title, columns,
                         size, path)


#***********#
#   pairs   #
#***********#
def _pairs(frame: pd.DataFrame, x: str, y: str, by: Optional[str], hue: Optional[str],
           measures: Sequence[str]) -> Tuple[List[pd.DataFrame], Dict[str, Any]]:
    """
    Two methods' rows as SpecVis's correlation and Bland-Altman plots take them: a frame
    per method and group, each named by both, the groups the values of ``by`` in order of
    appearance, and the hue a column of every frame. A pair with a measure one method has
    no value for (a fit that failed) is left out: SpecVis takes the measure's range over
    the data, and NA there halts it.
    """
    _has(frame, measures)
    if "Group" not in frame.columns:
        raise ValueError("the table has no Group column to tell the methods apart")
    for name in (x, y):
        if not (frame.Group == name).any():
            raise ValueError(f"no {name!r} among the groups {list(dict.fromkeys(frame.Group))}")
    for name in (by, hue):
        if name is not None and name not in frame.columns:
            raise ValueError(f"no column {name!r} in the table")
    groups = list(dict.fromkeys(frame[by])) if by else [""]
    frames, methods, named = [], [], []
    for method in (x, y):
        rows = frame[frame.Group == method]
        for group in groups:
            part = rows[rows[by] == group] if by else rows
            frames.append(part.reset_index(drop=True))
            methods.append(method)
            named.append(str(group))
    half = len(groups)
    for k in range(half):
        if len(frames[k]) != len(frames[k + half]):
            raise ValueError(f"{x} has {len(frames[k])} rows and {y} {len(frames[k + half])}"
                             + (f" in group {named[k]!r}" if by else "")
                             + ": SpecVis pairs them one to one")
        whole = (frames[k][list(measures)].notna().all(axis=1)
                 & frames[k + half][list(measures)].notna().all(axis=1))
        frames[k] = frames[k][whole].reset_index(drop=True)
        frames[k + half] = frames[k + half][whole].reset_index(drop=True)
    return frames, {"GroupVarNames1": methods, "GroupVarNames2": named,
                    "HueVar": [hue] * len(frames) if hue else None}


#*****************#
#   correlation   #
#*****************#
def correlation(frame: pd.DataFrame, x: str, y: str, measures: Sequence[str] = MEASURES,
                by: Optional[str] = None, hue: Optional[str] = None, quant: str = " / [tCr]",
                limits: Limits = None, title: Optional[str] = None, columns: int = 2,
                facet: bool = False, size: Tuple[float, float] = (10, 10),
                path: Optional[str] = None) -> Any:
    """
    SpecVis's correlation plot: one method's fits against another's, per measure, with
    the regression line and Pearson's r, pooled and per group, as the Osprey group
    compares algorithms (Zöllner et al. 2021, Figure 3).

    Args:
        frame (pd.DataFrame): The table, with the two methods under Group.
        x, y (str): The two methods, as Group names them; x goes along the x axis.
        measures (Sequence[str]): Its columns to draw; by default MEASURES.
        by (Optional[str]): A column that groups the rows (a vendor, a site), each group
            its own colour and correlation.
        hue (Optional[str]): A column of sub-groups within those, as the point style.
        quant (str): The axis label after the measure's name.
        limits: The axis limits, one (lower, upper) per measure.
        title (Optional[str]): The figure's title.
        columns (int): Facets per row.
        facet (bool): One facet per group, with the sub-groups' correlations.
        size (Tuple[float, float]): Width and height in inches.
        path (Optional[str]): Where to write the figure as SpecVis writes it.

    Returns:
        matplotlib.figure.Figure: The figure, as a picture of what R drew.

    A spectrum one of the two has no value for (a fit that failed, NaN) is left out of
    the pairs.

    Raises:
        ValueError: For a method or column the table lacks, rows that do not pair up,
            or limits that do not match the measures.
    """
    frames, grouped = _pairs(frame, x, y, by, hue, measures)
    args = {"Quant": quant, "MeasureVar": list(measures), **grouped,
            **_limits(limits, measures), "colNum": int(columns)}
    if title is not None:
        args["title"] = title
    return _plot("spvs_Correlation_Facet" if facet else "spvs_Correlation", frames, args,
                 size, path)


#******************#
#   bland-altman   #
#******************#
def bland_altman(frame: pd.DataFrame, x: str, y: str, measures: Sequence[str] = MEASURES,
                 by: Optional[str] = None, hue: Optional[str] = None,
                 quant: str = " / [tCr]", limits: Limits = None, differences: Limits = None,
                 title: Optional[str] = None, columns: int = 2,
                 size: Tuple[float, float] = (10, 10), path: Optional[str] = None) -> Any:
    """
    SpecVis's Bland-Altman plot: two methods' differences against their means, per
    measure, with the mean difference and its limits of agreement, pooled and per group,
    each with a distribution ellipse (Zöllner et al. 2021, Figure 4).

    Args:
        frame, x, y, measures, by, hue, quant, title, columns, size, path: As in
            ``correlation``.
        limits: The limits of the means' axis, one (lower, upper) per measure; by default
            the means' range, as SpecVis's Bland-Altman needs them.
        differences: The limits of the differences' axis, likewise; given with ``limits``.

    Returns:
        matplotlib.figure.Figure: The figure, as a picture of what R drew.
    """
    if (limits is None) != (differences is None):
        raise ValueError("limits and differences go together, one pair per measure each")
    frames, grouped = _pairs(frame, x, y, by, hue, measures)
    if limits is None:             # SpecVis's Bland-Altman, as published, halts without them
        half = len(frames) // 2
        xs, ys = (pd.concat(frames[k:k + half], ignore_index=True) for k in (0, half))
        xs, ys = xs[list(measures)], ys[list(measures)]
        means, spans = (xs + ys) / 2, ys - xs
        limits = [(float(means[m].min()), float(means[m].max())) for m in measures]
        differences = [(float(spans[m].min()), float(spans[m].max())) for m in measures]
    args = {"Quant": quant, "MeasureVar": list(measures), **grouped, "colNum": int(columns)}
    means, spans = _limits(limits, measures), _limits(differences, measures, "differences")
    args.update({k: means[k] + spans[k] for k in means})
    if title is not None:
        args["title"] = title
    return _plot("spvs_BlandAltman", frames, args, size, path)


#****************#
#   statistics   #
#****************#
def statistics(frame: pd.DataFrame, measures: Sequence[str] = MEASURES,
               paired: bool = True) -> Dict[str, Union[pd.DataFrame, str]]:
    """
    SpecVis's statistics over the groups: descriptive statistics, Shapiro-Wilk per group,
    Fligner-Killeen for the variances with its post hoc tests, the variance analysis
    those call for (ANOVA, Welch or Kruskal-Wallis) and its post hoc tests, all
    Bonferroni-corrected.

    Args:
        frame (pd.DataFrame): The table, grouped by its Group column.
        measures (Sequence[str]): Its columns to test; by default MEASURES.
        paired (bool): Paired post hoc tests, for the same spectra under every group;
            SpecVis's unpaired ones hold for two groups.

    Returns:
        Dict[str, Union[pd.DataFrame, str]]: SpecVis's tables under its own names, and a
            line where it has nothing to tabulate.
    """
    _has(frame, measures)
    folder = scratch("mrsuite_specvis_")
    try:
        out = _run({"task": "statistics", "frame": _write(frame, folder, "frame"),
                    "measures": list(measures), "paired": bool(paired)}, folder)
        tables = {name: pd.read_csv(path, index_col=0) for name, path in out["tables"].items()}
        return {**tables, **out["notes"]}
    finally:
        shutil.rmtree(folder, ignore_errors=True)


#*************#
#   version   #
#*************#
def version() -> str:
    """SpecVis's exact state, the commit MRSuite pins."""
    return externals.version(EXTERNAL)


#*************#
#   runtime   #
#*************#
def runtime() -> Dict[str, Any]:
    """R and ggplot2 as the runtime has them, and the SpecVis functions found."""
    folder = scratch("mrsuite_specvis_")
    try:
        out = _run({"task": "info"}, folder)
        return {"runtime": r.resolve_runtime(), "interpreter": out["r"],
                "ggplot2": out["ggplot2"], "functions": sorted(out["functions"])}
    finally:
        shutil.rmtree(folder, ignore_errors=True)

