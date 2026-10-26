####################################################################################################
#                                           nifti_mrs.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 26/10/26                                                                                #
#                                                                                                  #
# Purpose: Bridge onto nifti_mrs_tools' operations on NIfTI-MRS' higher dimensions - split,       #
#          merge, reorder, reshape, remove_dim, conjugate - as pipeline steps; the data pass as   #
#          they are, since the tools take NIfTI-MRS as MRSuite holds it.                          #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import inspect
from typing import Any, Callable, List

import nifti_mrs
from nifti_mrs import tools

from mrsuite.bridges.base import Bridge


#**************************************************************************************************#
#                                       Class NIfTIMRSBridge                                       #
#**************************************************************************************************#
#                                                                                                  #
# Dispatch onto nifti_mrs_tools, which takes NIfTI-MRS as MRSuite holds it.                        #
#                                                                                                  #
#**************************************************************************************************#
class NIfTIMRSBridge(Bridge):
    """Dispatch onto nifti_mrs_tools, which takes NIfTI-MRS as MRSuite holds it."""

    name = "nifti_mrs"
    program = "nifti_mrs_tools"
    FUNCTIONS = ("conjugate", "merge", "remove_dim", "reorder", "reshape", "split")

    #*************#
    #   resolve   #
    #*************#
    def _resolve(self, func: str) -> Callable:
        target = getattr(tools, func, None)
        if func not in self.FUNCTIONS or not inspect.isfunction(target):
            raise AttributeError(f"nifti_mrs_tools has no operation '{func}'. "
                                 f"Available: {', '.join(self.FUNCTIONS)}")
        return target

    #****************#
    #   to toolbox   #
    #****************#
    def _to_toolbox(self, obj):
        return obj

    #******************#
    #   from toolbox   #
    #******************#
    def _from_toolbox(self, obj):
        return obj

    #***************#
    #   available   #
    #***************#
    def available(self) -> List[str]:
        return list(self.FUNCTIONS)

    #***********#
    #   steps   #
    #***********#
    def steps(self) -> List[str]:
        """All but merge, which takes a list of spectra rather than the data first."""
        return [name for name in self.FUNCTIONS if name != "merge"]

    #*************#
    #   version   #
    #*************#
    def version(self) -> str:
        return nifti_mrs.__version__


bridge = NIfTIMRSBridge()


#*************#
#   getattr   #
#*************#
def __getattr__(name: str) -> Any:
    """Module-level dispatch, so `from mrsuite.bridges.nifti_mrs import split` works."""
    return getattr(bridge, name)


#*********#
#   dir   #
#*********#
def __dir__() -> List[str]:
    return sorted(set(globals()) | set(bridge.available()))
