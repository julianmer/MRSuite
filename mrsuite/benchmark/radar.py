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
# correlation; a signed error (bias, a mean or median error) is best at 0; the rest are
# errors and spreads, lower is better
HIGHER = {"lin_ccc", "robust_ccc", "pearson_r", "r_squared_direct", "adjusted_r_squared",
          "median_centered_correlation", "concise_score_manuscript", "concise_score_original"}
SIGNED = {"bias", "mpe", "mepe", "mean", "median"}
OWN = {"mae", "bias", "mosae"}                 # MRSuite's own error measures (scoring.py)


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
    such as bias is best at 0, so it is drawn by its magnitude (``SIGNED``). So every
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
    limits = {c: (limits or {}).get(c, (values[c].min(), values[c].max())
                                       if _metric(c) in HIGHER else (values[c].max(), values[c].min()))
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
    for method, row in values.iterrows():
        radius = [(row[c] - lo) / (hi - lo) if hi != lo else 1.0 for c, (lo, hi) in limits.items()]
        radius = [max(x, 0.05) for x in radius]             # never a point at the centre
        ax.plot(closed, radius + radius[:1], label=method, linewidth=2.5, marker="o",
                markersize=7, zorder=2)

    ax.set_xticks(angles)
    ax.set_xticklabels([c if (lo, hi) == (0, 1) else f"{c}\n{lo:g} to {hi:g}"
                        for c, (lo, hi) in limits.items()], fontsize=13, color="#9aa4b5")
    ax.tick_params(axis="x", pad=14)
    for angle, label in zip(np.degrees(angles) % 360, ax.get_xticklabels()):
        label.set_horizontalalignment("right" if 90 < angle < 270 else
                                      "center" if angle in (90, 270) else "left")
    ax.set_yticks([])
    ax.set_ylim(0, 1)
    ax.xaxis.grid(True, color="grey", alpha=0.45, linewidth=1.0)
    ax.yaxis.grid(False)
    ax.spines["polar"].set_visible(False)
    ax.set_theta_offset(np.pi / 2)                           # the first corner at the top
    ax.set_theta_direction(-1)
    if len(values):
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False,
                  labelcolor="#9aa4b5", fontsize=11, ncol=min(len(values), 3))
    return ax


#*****************#
#   plot scores   #
#*****************#
def plot_scores(scores: Union[pd.DataFrame, Dict[str, pd.DataFrame]], ax: Any = None) -> Any:
    """
    Score tables on a radar plot: one corner per aspect of MRSValidate (errors, agreement,
    regression, method comparison, ...), one polygon per method, as bsuite draws its agents
    over its capabilities. Every measure in the tables is scaled from the worst method to the
    best (see ``radar``), and a corner is the mean of its aspect's measures, so a corner
    fills as more of its measures are wired in. The pooled rows (metabolite 'all') count
    where a table has them, else each metabolite's; a measure on several references counts
    once per reference.

    Args:
        scores (Union[pd.DataFrame, Dict[str, pd.DataFrame]]): One table of score's, or
            several by name, e.g. one per measure.
        ax (Any): A polar axes to draw on; else a new figure's, made without pyplot, so
            drawing it in a thread is safe and nothing is left open.

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
    if not tables:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["polar"].set_visible(False)
        ax.set_title("No scores yet: wire score tables in, a corner per aspect", color="grey")
        return ax
    table = pd.concat(tables)
    pooled = table[table["metabolite"] == "all"]
    picked = pooled if not pooled.empty else table
    picked = picked.assign(measure=picked["metric"] + " (" + picked["reference"] + ")"
                           + np.where(pooled.empty, " " + picked["metabolite"], ""))
    twice = picked[picked.duplicated(["method", "measure"])]["measure"].unique()
    if len(twice):
        raise ValueError(f"given twice: {', '.join(twice)}")
    values = picked.pivot(index="method", columns="measure", values="score")
    missing = [f"{measure} ({method})" for measure in values.columns
               for method in values.index[values[measure].isna()]]
    if missing:
        raise ValueError(f"no score for {', '.join(missing)}")

    analysis = dict(zip(picked["measure"], picked["analysis"]))
    scaled = pd.DataFrame(index=values.index)
    for c in values.columns:                                 # each measure 0 (worst) to 1
        names = {_metric(c), analysis[c]}                    # the metric as named, or as run
        v = values[c].abs() if names & SIGNED else values[c]
        lo, hi = (v.min(), v.max()) if names & HIGHER else (v.max(), v.min())
        scaled[c] = (v - lo) / (hi - lo) if hi != lo else 1.0
    corners = scaled.T.groupby(lambda c: aspect(_metric(c), analysis[c])).mean().T
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
    return module.replace("_", " ").capitalize()
