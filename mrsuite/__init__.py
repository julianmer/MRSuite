####################################################################################################
#                                           __init__.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-08-08                                                                              #
#                                                                                                  #
# Purpose: Package entry point. Exposes the toolbox bridges directly (mrsuite.fid_a,               #
#          mrsuite.fsl_mrs, mrsuite.osprey, mrsuite.lcmodel) and mrsuite.method, which makes       #
#          a function of your own a method, and stamps the shared NIfTI-MRS+ provenance.           #
#          Bridges are imported lazily: resolving an Octave-backed one would start a               #
#          session, and the FSL-MRS one pulls in the whole FSL-MRS stack, neither of which         #
#          a caller who only wants the pipeline should pay for.                                    #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import importlib
from typing import Any

import nifti_mrs_plus

from mrsuite.core.method import Method, method

__version__ = "0.0.3"

# Same pattern as Augmentrum: stamp the shared wrapper's provenance metadata so
# processed data records which toolbox touched it.
nifti_mrs_plus.set_provenance("MRSuite", __version__)

_BRIDGES = {"augmentrum", "fid_a", "fsl_mrs", "lcmodel", "mrspeclab", "nifti_mrs", "osprey",
            "pyamares", "spant", "suspect", "tarquin", "vespa"}

__all__ = ["__version__", "Method", "method", *sorted(_BRIDGES)]


#*************#
#   getattr   #
#*************#
def __getattr__(name: str) -> Any:
    if name in _BRIDGES:
        return importlib.import_module(f"mrsuite.bridges.{name}")
    raise AttributeError(f"module 'mrsuite' has no attribute '{name}'")


#*********#
#   dir   #
#*********#
def __dir__():
    return sorted(set(globals()) | _BRIDGES)
