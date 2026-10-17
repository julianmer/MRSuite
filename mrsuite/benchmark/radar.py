####################################################################################################
#                                             radar.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-01                                                                              #
#                                                                                                  #
# Purpose: A radar plot of benchmark scores. Like MRSValidate's own plots it only draws values     #
#          computed before: one polygon per method, one spoke per metric.                          #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd


# how a metric reads, by its name in the score tables: higher is better for agreement and
# correlation, and for Shapiro-Wilk's W; a signed error (bias, a mean or median error,
# Bland-Altman's, whose score is its bias, the equivalence tests', whose score is their
# intercept) is best at 0; a slope (and the original concise score, slope times R²) is best
# at 1; the rest are errors and spreads, lower is better
HIGHER = {"lin_ccc", "robust_ccc", "pearson_r", "r_squared_direct", "adjusted_r_squared",
          "median_centered_correlation", "concise_score_manuscript", "shapiro_wilk"}
ONE = {"passing_bablok", "deming", "theil_sen", "regression_metrics", "concise_score_original"}
SIGNED = {"bias", "mpe", "mepe", "mean", "median", "bland_altman", "equivalence_test_1",
          "equivalence_test_2", "equivalence_test_3"}
OWN = {"mae", "bias", "mosae"}                 # MRSuite's own error measures (scoring.py)
# the one analysis MRSValidate runs under a name other than its function's (meta.method)
RUN_AS = {"manuscript_robust_ccc": "robust_ccc"}
# how a metric is labelled on the radar, by its name in the score tables; others by that name
LABELS = {"mae": "MAE", "mosae": "MOSAE", "bias": "bias", "rmse": "RMSE", "mape": "MAPE",
          "lin_ccc": "Lin's CCC", "robust_ccc": "robust CCC", "pearson_r": "Pearson r",
          "r_squared_direct": "R²", "bland_altman": "Bland-Altman bias", "robust_sd": "robust SD",
          "iqr": "IQR"}
# an aspect by the module its analyses sit in, where MRSValidate's name says less
ASPECTS = {"descriptive": "Error distribution", "normality": "Normality of errors"}


#************#
#   metric   #
#************#
def _metric(measure: str) -> str:
    """The metric a measure is: its name before any reference or metabolite."""
    return measure.split(" ")[0]


#***********#
#   radar   #
#***********#
def radar(values: pd.DataFrame, limits: Optional[Dict[str, Tuple[float, float]]] = None,
          ax: Any = None) -> Any:
    """
    Draw scores on a radar plot: one polygon per row (a method), one spoke per column
    (a metric), the best method at the rim of each spoke and the worst at its centre, on a
    web of polygon rings, a point at each corner, no fill.

    It computes no statistic, as MRSValidate's plots compute none; score's table becomes
    its input by picking one reference and each metric's own field, e.g.
    ``table.query("reference == 'optimal' and metabolite == 'all'").pivot(index="method",
    columns="metric", values="value")`` for metrics filed under 'value' (Lin's CCC is
    under 'rho_c'). Each spoke runs from the worst value among the methods to the best,
    which way round the metric says (``HIGHER``, else lower is better); a signed metric
    such as bias is best at 0, so it is drawn by its magnitude (``SIGNED``), and a slope
    is best at 1, so by its distance from 1 (``ONE``). So every
    metric sits on 0-1, comparable across spokes, which is axis geometry only. Where all
    methods score the same, all sit at the rim. A column given limits (0, 1) is drawn as
    it is, labelled by its name alone.

    Args:
        values (pd.DataFrame): Methods by metrics.
        limits (Optional[Dict[str, Tuple[float, float]]]): Per metric, the values at the
            centre and at the rim instead, e.g. a lower-is-better metric's (worst, best).
        ax (Any): A polar axes to draw on; a new figure's when None.

    Returns:
        Any: The axes.
    """
    import matplotlib.pyplot as plt

    values = values.copy()
    for c in values.columns:
        if _metric(c) in SIGNED:
            values[c] = values[c].abs()
        elif _metric(c) in ONE:
            values[c] = (values[c] - 1).abs()
    limits = {c: (limits or {}).get(c, (values[c].min(), values[c].max()) if _metric(c) in HIGHER
                                   else (values[c].max(), values[c].min()))
              for c in values.columns}
    angles = np.linspace(0, 2 * np.pi, len(limits), endpoint=False)
    if ax is None:
        _, ax = plt.subplots(subplot_kw={"projection": "polar"})
    ax.set_facecolor("none")
    ax.figure.set_facecolor("none")
    closed = np.append(angles, angles[0])

    for r in (0.2, 0.4, 0.6, 0.8, 1.0):                     # a web: rings as polygons
        ax.plot(closed, [r] * len(closed), color="grey", alpha=0.55 if r == 1.0 else 0.35,
                linewidth=1.6 if r == 1.0 else 1.0, zorder=0)
    _scale(ax)
    for method, row in values.iterrows():
        radius = [(row[c] - lo) / (hi - lo) if hi != lo else 1.0 for c, (lo, hi) in limits.items()]
        radius = [max(x, 0.05) for x in radius]             # never a point at the centre
        ax.plot(closed, radius + radius[:1], label=method, linewidth=2.5, marker="o",
                markersize=7, zorder=2)

    ax.set_xticks(angles)
    ax.set_xticklabels([c if (lo, hi) == (0, 1) else f"{c}\n{lo:.3g} to {hi:.3g}"
                        for c, (lo, hi) in limits.items()], fontsize=13, color="#9aa4b5")
    ax.tick_params(axis="x", pad=28)                         # the labels clear of the rim
    for angle, label in zip(np.degrees(angles), ax.get_xticklabels()):
        shown = (90 - angle) % 360                           # the first corner on top, clockwise
        side = np.isclose(shown, 90) or np.isclose(shown, 270)
        label.set_horizontalalignment("center" if side else
                                      "right" if 90 < shown < 270 else "left")
        label.set_verticalalignment("bottom" if np.isclose(shown, 90) else
                                    "top" if np.isclose(shown, 270) else "center")
    ax.set_yticks([])
    ax.set_ylim(0, 1)
    ax.xaxis.grid(True, color="grey", alpha=0.45, linewidth=1.0)
    ax.yaxis.grid(False)
    ax.spines["polar"].set_visible(False)
    ax.set_theta_offset(np.pi / 2)                           # the first corner at the top
    ax.set_theta_direction(-1)
    if len(values):
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False,
                  labelcolor="#9aa4b5", fontsize=11, ncol=min(len(values), 3))
    return ax


#***********#
#   scale   #
#***********#
def _scale(ax: Any) -> None:
    """The web's scale, 0 at the centre to 1 at the rim, beside the first spoke."""
    for r in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        ax.text(0, r, f" {r:g}", fontsize=9, color="grey", ha="left", va="bottom", zorder=1)


#*****************#
#   plot scores   #
#*****************#
def plot_scores(scores: Union[pd.DataFrame, Dict[str, pd.DataFrame]], ax: Any = None,
                by: str = "aspect") -> Any:
    """
    Score tables on a radar plot: one corner per aspect of MRSValidate (errors, agreement,
    regression, method comparison, ...), one polygon per method, as bsuite draws its agents
    over its capabilities. Every measure in the tables is scaled from the worst method to the
    best (see ``radar``), and a corner is the mean of its aspect's measures, so a corner
    fills as more of its measures are wired in; each corner names its measures. Or, by
    measure, a spoke per measure. Every spoke runs 0 to 1 alike. The pooled rows
    (metabolite 'all') count where a table has them, else each metabolite's; a measure on
    several references counts once per reference, and is labelled with it.

    Args:
        scores (Union[pd.DataFrame, Dict[str, pd.DataFrame]]): One table of score's, or
            several by name, e.g. one per measure.
        ax (Any): A polar axes to draw on; else a new figure's, made without pyplot, so
            drawing it in a thread is safe and nothing is left open.
        by (str): 'aspect', a corner per aspect, or 'measure', a spoke per measure.

    Returns:
        Any: The axes.

    Raises:
        ValueError: For a method without a score for a measure (one that failed, or an
            interval's), or one measure given twice. No tables at all draw an empty radar,
            to be filled.
    """
    from matplotlib.figure import Figure

    tables = list(scores.values()) if isinstance(scores, dict) else [scores]
    if ax is None:
        ax = Figure(figsize=(7, 7), layout="constrained").add_subplot(projection="polar")
    if not tables:                                           # the web, waiting for corners
        ax.set_facecolor("none")
        ax.figure.set_facecolor("none")
        for r in (0.2, 0.4, 0.6, 0.8, 1.0):
            ax.plot(np.linspace(0, 2 * np.pi, 100), [r] * 100, color="grey",
                    alpha=0.55 if r == 1.0 else 0.35, linewidth=1.6 if r == 1.0 else 1.0)
        ax.set_theta_offset(np.pi / 2)                       # its scale on top, as a radar's
        _scale(ax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylim(0, 1)
        ax.spines["polar"].set_visible(False)
        ax.set_title("No scores yet: wire score tables in, a corner per aspect",
                     color="#9aa4b5", fontsize=12)
        return ax
    picked = pd.concat([t[t["metabolite"] == "all"] if (t["metabolite"] == "all").any() else t
                        for t in tables])
    picked = picked.assign(measure=picked["metric"] + " (" + picked["reference"] + ")"
                           + np.where(picked["metabolite"] == "all", "",
                                      " " + picked["metabolite"]))
    twice = picked[picked.duplicated(["method", "measure"])]["measure"].unique()
    if len(twice):
        raise ValueError(f"given twice: {', '.join(twice)}")
    values = picked.pivot(index="method", columns="measure", values="score")
    missing = [f"{measure} ({method})" for measure in values.columns
               for method in values.index[~np.isfinite(values[measure])]]
    if missing:
        raise ValueError(f"no score for {', '.join(missing)}")

    if by not in ("aspect", "measure"):
        raise ValueError(f"a radar is by 'aspect' or by 'measure', not {by!r}")
    analysis = dict(zip(picked["measure"], picked["analysis"].replace(RUN_AS)))
    several = picked["reference"].nunique() > 1
    scaled, labels = pd.DataFrame(index=values.index), {}
    for c in values.columns:                                 # each measure 0 (worst) to 1
        names = ({_metric(c)} if _metric(c) in OWN else    # MRSuite's own by its name,
                 {_metric(c), analysis[c]})                  # others as named, or as run
        v = (values[c].abs() if names & SIGNED else
             (values[c] - 1).abs() if names & ONE else values[c])
        lo, hi = (v.min(), v.max()) if names & HIGHER else (v.max(), v.min())
        scaled[c] = (v - lo) / (hi - lo) if hi != lo else 1.0
        label = LABELS.get(_metric(c), _metric(c))
        label = f"|{label}|" if names & SIGNED else f"|{label} - 1|" if names & ONE else label
        reference, _, metabolite = c.split(" ", 1)[1].partition(" ")
        labels[c] = " ".join(filter(None, [label, reference if several else "", metabolite]))
    if by == "measure":
        corners = scaled.rename(columns=labels)
    else:
        group = {c: aspect(_metric(c), analysis[c]) for c in scaled.columns}
        corners = scaled.T.groupby(group).mean().T
        corners = corners.rename(columns={a: a + "\n" + ", ".join(
            labels[c] for c in scaled.columns if group[c] == a) for a in corners.columns})
    return radar(corners, {c: (0, 1) for c in corners.columns}, ax)


#************#
#   aspect   #
#************#
def aspect(metric: str, analysis: str) -> str:
    """The aspect a measure judges, as MRSValidate groups its analyses: the module of the
    analysis that ran, or of the metric's own name; MRSuite's own errors (MAE, bias, MOSAE)
    are errors."""
    from mrsuite.benchmark.scoring import analyses

    modules = {name.split(".")[1]: name.split(".")[0] for name in analyses()}
    module = "errors" if metric in OWN else modules.get(analysis) or modules.get(metric, "errors")
    return ASPECTS.get(module, module.replace("_", " ").capitalize())
