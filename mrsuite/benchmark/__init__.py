####################################################################################################
#                                           __init__.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-01                                                                              #
#                                                                                                  #
# Purpose: Benchmarking on MRSValidate. Fits and ground truth become its ObservationSets on a      #
#          reference the caller names (water, tCr, the optimal scale or none), every metric runs   #
#          as a step of its analysis plan, and radar draws the resulting scores.                   #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from mrsuite.benchmark.radar import plot_scores, radar
from mrsuite.benchmark.scoring import (REFERENCES, analyses, bias, coverage, figure, mae,
                                       observations, optimal_scale, score)

__all__ = ["REFERENCES", "observations", "optimal_scale", "score", "analyses", "figure",
           "coverage", "mae", "bias", "radar", "plot_scores"]
