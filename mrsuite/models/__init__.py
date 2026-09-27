####################################################################################################
#                                           __init__.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-27                                                                              #
#                                                                                                  #
# Purpose: Trained models as pipeline steps. Model is the interface, in any framework;             #
#          TorchModel adapts it to PyTorch; MLP and WAND wrap the published networks.              #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from mrsuite.models.base import Model
from mrsuite.models.mlp import MLP
from mrsuite.models.pytorch import TorchModel
from mrsuite.models.wand import WAND

__all__ = ["Model", "TorchModel", "MLP", "WAND"]
