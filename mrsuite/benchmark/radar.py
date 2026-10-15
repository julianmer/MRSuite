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
        if c in SIGNED:
            values[c] = values[c].abs()
    limits = {c: (limits or {}).get(c, (values[c].min(), values[c].max()) if c in HIGHER
                                       else (values[c].max(), values[c].min()))
              for c in values.columns}
    angles = np.linspace(0, 2 * np.pi, len(limits), endpoint=False)
    if ax is None:
        _, ax = plt.subplots(subplot_kw={"projection": "polar"})

    for method, row in values.iterrows():
        radius = [(row[c] - lo) / (hi - lo) if hi != lo else 1.0 for c, (lo, hi) in limits.items()]
        ax.plot(np.append(angles, angles[0]), radius + radius[:1], label=method)
        ax.fill(angles, radius, alpha=0.1)

    ax.set_xticks(angles, [f"{c}\n{lo:g} to {hi:g}" for c, (lo, hi) in limits.items()])
    ax.set_yticks([])
    ax.set_ylim(0, 1)
    if len(values):
        ax.legend()
    return ax


#*****************#
#   plot scores   #
#*****************#
def plot_scores(scores: Union[pd.DataFrame, Dict[str, pd.DataFrame]], reference: str,
                metabolite: str = "all",
                limits: Optional[Dict[str, Tuple[float, float]]] = None, ax: Any = None) -> Any:
    """
    Score tables on a radar plot: one spoke per metric, each its 'score' (score's first
    number of a result), one polygon per method, on one reference and metabolite.

    Args:
        scores (Union[pd.DataFrame, Dict[str, pd.DataFrame]]): One table of score's, or
            several by name, e.g. one per metric.
        reference (str): Which reference's scores.
        metabolite (str): Which metabolite's, or 'all' pooled.
        limits (Optional[Dict[str, Tuple[float, float]]]): As in radar.
        ax (Any): A polar axes to draw on; else a new figure's, made without pyplot, so
            drawing it in a thread is safe and nothing is left open.

    Returns:
        Any: The axes.

    Raises:
        ValueError: For no scores on that reference and metabolite, a method without a
            score for a metric (one that failed, or an interval's), or one metric given
            twice. No tables at all draw an empty radar, to be filled.
    """
    from matplotlib.figure import Figure

    tables = list(scores.values()) if isinstance(scores, dict) else [scores]
    if not tables:
        if ax is None:
            ax = Figure(figsize=(6, 6), layout="constrained").add_subplot(projection="polar")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("No scores yet: wire score tables in, a spoke per metric")
        return ax
    table = pd.concat(tables)
    picked = table[(table["reference"] == reference) & (table["metabolite"] == metabolite)]
    if picked.empty:
        raise ValueError(f"no scores on {reference} for {metabolite}; there are "
                         f"{sorted(set(table['reference']))} for "
                         f"{sorted(set(table['metabolite']))[:12]}")
    twice = picked[picked.duplicated(["method", "metric"])]["metric"].unique()
    if len(twice):
        raise ValueError(f"given twice: {', '.join(twice)}")
    values = picked.pivot(index="method", columns="metric", values="score")
    missing = [f"{metric} ({method})" for metric in values.columns
               for method in values.index[values[metric].isna()]]
    if missing:
        raise ValueError(f"no score for {', '.join(missing)}")
    if ax is None:
        ax = Figure(figsize=(6, 6), layout="constrained").add_subplot(projection="polar")
    return radar(values, limits, ax)
