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
from mrsuite.benchmark.radar import radar
from mrsuite.benchmark.scoring import (REFERENCES, bias, coverage, mae, observations,
                                       optimal_scale, score)

__all__ = ["REFERENCES", "observations", "optimal_scale", "score", "coverage", "mae", "bias",
           "radar"]
