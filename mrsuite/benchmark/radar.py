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
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


#***********#
#   radar   #
#***********#
def radar(values: pd.DataFrame, limits: Optional[Dict[str, Tuple[float, float]]] = None,
          ax: Any = None) -> Any:
    """
    Draw scores on a radar plot: one polygon per row (a method), one spoke per column
    (a metric).

    It computes no statistic, as MRSValidate's plots compute none; score's table becomes
    its input by picking one reference and each metric's own field, e.g.
    ``table.query("reference == 'optimal' and metabolite == 'all'").pivot(index="method",
    columns="metric", values="value")`` for metrics filed under 'value' (Lin's CCC is
    under 'rho_c'). Each spoke runs from its limits, by default from its lowest value to
    its highest with 0 always on it, which is axis geometry only. A signed metric such
    as bias is best at 0, inside its spoke, so it reads better as its magnitude.

    Args:
        values (pd.DataFrame): Methods by metrics.
        limits (Optional[Dict[str, Tuple[float, float]]]): Per metric, the values at the
            centre and at the rim; (worst, best) draws better outwards, so a lower-is-
            better metric such as MAE takes them high first.
        ax (Any): A polar axes to draw on; a new figure's when None.

    Returns:
        Any: The axes.
    """
    import matplotlib.pyplot as plt

    limits = {c: (limits or {}).get(c, (min(0.0, values[c].min()), max(0.0, values[c].max())))
              for c in values.columns}
    angles = np.linspace(0, 2 * np.pi, len(limits), endpoint=False)
    if ax is None:
        _, ax = plt.subplots(subplot_kw={"projection": "polar"})

    for method, row in values.iterrows():
        radius = [(row[c] - lo) / (hi - lo) for c, (lo, hi) in limits.items()]
        ax.plot(np.append(angles, angles[0]), radius + radius[:1], label=method)
        ax.fill(angles, radius, alpha=0.1)

    ax.set_xticks(angles, [f"{c}\n{lo:g} to {hi:g}" for c, (lo, hi) in limits.items()])
    ax.set_yticks([])
    ax.set_ylim(0, 1)
    ax.legend()
    return ax
