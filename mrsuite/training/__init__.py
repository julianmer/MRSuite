####################################################################################################
#                                           __init__.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-28                                                                              #
#                                                                                                  #
# Purpose: Training: sources of spectra with their truth, and the one loop every model learns      #
#          from them with, through Model.learn.                                                    #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from mrsuite.training.source import Source, Truth

__all__ = ["Source", "Truth"]
