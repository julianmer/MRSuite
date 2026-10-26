####################################################################################################
#                                          augmentrum.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 26/10/26                                                                                #
#                                                                                                  #
# Purpose: Bridge onto Augmentrum's augmentation modules as pipeline steps: a registry name       #
#          makes the module with the arguments given and runs it on the batch, and on its water   #
#          reference where one is wired in. Augmentrum takes NIfTI-MRS+ as MRSuite holds it and   #
#          records every module in ProcessingApplied itself.                                      #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import inspect
from contextlib import contextmanager
from typing import Any, Callable, List

import nifti_mrs_plus
from nifti_mrs_plus import NIfTI_MRS_Plus

try:
    import augmentrum
    from augmentrum.core.augmentrum import Augmentrum
except ModuleNotFoundError as err:
    if (err.name or "").partition(".")[0] != "augmentrum":   # installed, but broken
        raise
    raise ImportError("Augmentrum is not installed: with FSL-MRS installed, pip install "
                      "git+https://github.com/julianmer/Augmentrum.git") from err

from mrsuite.bridges.base import Bridge


#**************************************************************************************************#
#                                      Class AugmentrumBridge                                      #
#**************************************************************************************************#
#                                                                                                  #
# Dispatch onto Augmentrum's modules, by the names its own pipelines take.                         #
#                                                                                                  #
#**************************************************************************************************#
class AugmentrumBridge(Bridge):
    """Dispatch onto Augmentrum's modules, by the names its own pipelines take.

    A name is one configuration of a module class: ``noise`` is ``Noise``, and
    ``baseline_bspline`` is ``BaselineAugmentation`` with its mode fixed. The call takes
    the batch, the module's arguments by name, and a water reference, which the module
    augments alike; the batch comes back augmented, every module having written its own
    entry in the spectra's ProcessingApplied.
    """
    name = "augmentrum"
    program = "Augmentrum"
    records = False           # every module writes its own entry
    batched = True            # Augmentrum works on the batch: its draws are per sample

    #*************#
    #   resolve   #
    #*************#
    def _resolve(self, func: str) -> Callable:
        if func not in Augmentrum.AVAILABLE_MODULES:
            raise AttributeError(f"Augmentrum has no module '{func}'. "
                                 f"Available: {', '.join(self.available())}")
        cls, fixed = Augmentrum.resolve_module(func)

        def call(data, *, water=None, **arguments):
            with _credited():
                augmented, _ = cls(**fixed, **arguments)(_apart(data), _apart(water))
            return augmented
        call.__name__, call.__doc__ = func, inspect.getdoc(cls)
        call.__signature__ = _signature(cls, fixed)
        return call

    #****************#
    #   to toolbox   #
    #****************#
    def _to_toolbox(self, obj):
        return obj

    #******************#
    #   from toolbox   #
    #******************#
    def _from_toolbox(self, obj):
        """The batch as the list of its spectra, as a batched method hands them back."""
        return obj.list() if isinstance(obj, NIfTI_MRS_Plus) else obj

    #***************#
    #   available   #
    #***************#
    def available(self) -> List[str]:
        """The registry's names, each module once: its first name, and every name that fixes
        a configuration of it; the plain aliases ('broadening' for 'line_broadening') are
        left out."""
        names, seen = [], set()
        for name, entry in Augmentrum.AVAILABLE_MODULES.items():
            cls = entry[0] if isinstance(entry, tuple) else entry
            if isinstance(entry, tuple) or cls not in seen:
                names.append(name)
            seen.add(cls)
        return names

    #*************#
    #   version   #
    #*************#
    def version(self) -> str:
        return augmentrum.__version__


#**************#
#   credited   #
#**************#
@contextmanager
def _credited():
    """The entries written meanwhile credited to Augmentrum: nifti_mrs_plus names one program
    for the whole process, whichever set it last (MRSuite at its import, Augmentrum at its),
    and the module running here is Augmentrum's."""
    before = nifti_mrs_plus.get_provenance()
    nifti_mrs_plus.set_provenance("Augmentrum", augmentrum.__version__)
    try:
        yield
    finally:
        nifti_mrs_plus.set_provenance(**before)


#***********#
#   apart   #
#***********#
def _apart(batch):
    """A copy of the batch: Augmentrum's modules write into the very spectra they are given,
    data and history, and a step leaves its input alone."""
    return NIfTI_MRS_Plus([n.copy() for n in batch.list()]) if batch is not None else None


#***************#
#   signature   #
#***************#
def _signature(cls: Any, fixed: dict) -> inspect.Signature:
    """A module's call as a step: the data, then the class's arguments by name but the ones
    the registry name fixes, then the water reference."""
    keyword = inspect.Parameter.KEYWORD_ONLY
    arguments = [p.replace(kind=keyword) for n, p in inspect.signature(cls.__init__).parameters.items()
                 if n != "self" and n not in fixed
                 and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
    return inspect.Signature([inspect.Parameter("data", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                              *arguments, inspect.Parameter("water", keyword, default=None)])


bridge = AugmentrumBridge()


#*************#
#   getattr   #
#*************#
def __getattr__(name: str) -> Any:
    """Module-level dispatch, so `from mrsuite.bridges.augmentrum import noise` works."""
    return getattr(bridge, name)


#*********#
#   dir   #
#*********#
def __dir__() -> List[str]:
    return sorted(set(globals()) | set(bridge.available()))
