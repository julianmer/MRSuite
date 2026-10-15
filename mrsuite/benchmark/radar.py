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
    (a metric), the best method at the rim of each spoke and the worst at its centre.

    It computes no statistic, as MRSValidate's plots compute none; score's table becomes
    its input by picking one reference and each metric's own field, e.g.
    ``table.query("reference == 'optimal' and metabolite == 'all'").pivot(index="method",
    columns="metric", values="value")`` for metrics filed under 'value' (Lin's CCC is
    under 'rho_c'). Each spoke runs from the worst value among the methods to the best,
    which way round the metric says (``HIGHER``, else lower is better); a signed metric
    such as bias is best at 0, so it is drawn by its magnitude (``SIGNED``). So every
    metric sits on 0-1, comparable across spokes, which is axis geometry only. Where all
    methods score the same, all sit at the rim.

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

    for r, alpha in ((0.25, 0.05), (0.5, 0.05), (0.75, 0.03), (1.0, 0.01)):   # bsuite's rings
        ax.fill(np.linspace(0, 2 * np.pi, 100), [r] * 100, color="grey", alpha=alpha,
                zorder=0)
    for method, row in values.iterrows():
        radius = [(row[c] - lo) / (hi - lo) if hi != lo else 1.0 for c, (lo, hi) in limits.items()]
        radius = [max(x, 0.05) for x in radius]             # never a point at the centre
        line, = ax.plot(np.append(angles, angles[0]), radius + radius[:1], label=method,
                        linewidth=3, alpha=0.85, zorder=2)
        ax.fill(angles, radius, color=line.get_color(), alpha=0.2, zorder=2)

    ax.set_xticks(angles)
    ax.set_xticklabels([f"{c}\n{lo:g} to {hi:g}" for c, (lo, hi) in limits.items()],
                       fontsize=12, color="grey")
    for angle, label in zip(np.degrees(angles), ax.get_xticklabels()):
        label.set_horizontalalignment("right" if 90 < angle < 270 else
                                      "center" if angle in (90, 270) else "left")
    ax.set_rlabel_position(0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0], [".25", ".5", ".75", "1"], color="grey",
                  fontsize=9, alpha=0.75)
    ax.set_ylim(0, 1)
    ax.grid(color="grey", alpha=0.6, linestyle=":", linewidth=1)
    ax.spines["polar"].set_visible(False)
    if len(values):
        ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0), frameon=False,
                  labelcolor="grey")
    return ax


#*****************#
#   plot scores   #
#*****************#
def plot_scores(scores: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                limits: Optional[Dict[str, Tuple[float, float]]] = None, ax: Any = None) -> Any:
    """
    Score tables on a radar plot: one spoke per measure, each its 'score' (score's first
    number of a result), one polygon per method, as bsuite draws its agents over its
    capabilities. Every measure in the tables is drawn: the pooled rows (metabolite 'all')
    where a table has them, else each metabolite's; a measure scored on several references
    gets a spoke per reference, named so.

    Args:
        scores (Union[pd.DataFrame, Dict[str, pd.DataFrame]]): One table of score's, or
            several by name, e.g. one per measure.
        limits (Optional[Dict[str, Tuple[float, float]]]): As in radar.
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
        ax = Figure(figsize=(7, 6), layout="constrained").add_subplot(projection="polar")
    if not tables:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("No scores yet: wire score tables in, a spoke per measure", color="grey")
        return ax
    table = pd.concat(tables)
    pooled = table[table["metabolite"] == "all"]
    picked = pooled if not pooled.empty else table
    picked = picked.assign(measure=picked["metric"] + np.where(
        picked["reference"].nunique() > 1, " (" + picked["reference"] + ")", "") + np.where(
        picked["metabolite"].nunique() > 1, " " + picked["metabolite"], ""))
    twice = picked[picked.duplicated(["method", "measure"])]["measure"].unique()
    if len(twice):
        raise ValueError(f"given twice: {', '.join(twice)}")
    values = picked.pivot(index="method", columns="measure", values="score")
    missing = [f"{measure} ({method})" for measure in values.columns
               for method in values.index[values[measure].isna()]]
    if missing:
        raise ValueError(f"no score for {', '.join(missing)}")
    return radar(values, limits, ax)
