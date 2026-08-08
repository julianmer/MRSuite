####################################################################################################
#                                           __init__.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-08-08                                                                              #
#                                                                                                  #
# Purpose: Bridges onto external MRS toolboxes. Each exposes a toolbox's whole function            #
#          surface on MRSuite's data types, accepting NIfTI-MRS and NIfTI-MRS+ alike,              #
#          batching over the latter and recording each call in provenance.                         #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from mrsuite.bridges.base import Bridge, Deferred, describe

__all__ = ["Bridge", "Deferred", "describe"]
