####################################################################################################
#                                           __init__.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-01                                                                              #
#                                                                                                  #
# Purpose: MRSuite's own part of a benchmark: fits and ground truth paired on a reference          #
#          the caller names (water, tCr, the optimal scale or none), ratios to creatine,           #
#          what the fits cover, and a radar of the scores. The statistics are MRSValidate's,       #
#          through its bridge.                                                                     #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from mrsuite.benchmark.radar import plot_scores, radar
from mrsuite.benchmark.scoring import (REFERENCES, coverage, metabolites, optimal_scale, pair,
                                       ratios)

__all__ = ["REFERENCES", "pair", "metabolites", "ratios", "optimal_scale", "coverage",
           "radar", "plot_scores"]
