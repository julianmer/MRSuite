####################################################################################################
#                                           __init__.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-28                                                                              #
#                                                                                                  #
# Purpose: Training: sources of spectra with their truth, the one loop every model learns          #
#          from them with, through Model.learn, and recipes as the MRSsynMRS JSON sidecar.         #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from mrsuite.training import recipe
from mrsuite.training.source import Source, Truth

__all__ = ["Source", "Truth", "recipe"]
