####################################################################################################
#                                           __init__.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-27                                                                              #
#                                                                                                  #
# Purpose: Trained models as pipeline steps. Model is the interface, in any framework;             #
#          TorchModel adapts it to PyTorch; MLP and WAND wrap the published networks; load         #
#          brings any of them back from the Hub or a folder by name alone.                         #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import importlib
import json
import os
from typing import Any, Optional

from mrsuite.models.base import DOWNLOAD, Model
from mrsuite.models.mlp import MLP
from mrsuite.models.pytorch import TorchModel
from mrsuite.models.wand import WAND

__all__ = ["Model", "TorchModel", "MLP", "WAND", "load"]


#**********#
#   load   #
#**********#
def load(name: Any, revision: Optional[str] = None, **kwargs) -> Model:
    """
    A model from the Hub or a folder, its class picked from its config.

    Every checkpoint a Model saves records its class ('model' in config.json), so a repo
    like 'MRSuite/wand-mrsi2024' or a checkpoint folder loads by name alone; the model's
    own acquisition check then refuses data it was not trained for.

    Args:
        name: A Hub repo id or a local folder, as from_pretrained takes it.
        revision (Optional[str]): The Hub revision to pin.
        **kwargs: Passed to the class' from_pretrained.

    Returns:
        Model: The checkpoint, loaded by its own class.

    Raises:
        ValueError: If the config names no MRSuite model class.
    """
    if os.path.isdir(name):
        path = os.path.join(name, "config.json")
    else:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=str(name), filename="config.json", revision=revision,
                               **{k: kwargs[k] for k in DOWNLOAD if k in kwargs})
    with open(path) as fh:
        spec = json.load(fh).get("model", "")
    module, _, classname = str(spec).rpartition(".")
    if not module.startswith("mrsuite."):
        raise ValueError(f"{name} does not name an MRSuite model in its config "
                         f"(found {spec!r}); load it with its own class' from_pretrained")
    cls = getattr(importlib.import_module(module), classname)
    if not (isinstance(cls, type) and issubclass(cls, Model)):
        raise ValueError(f"{spec} is not an mrsuite Model")
    return cls.from_pretrained(name, revision=revision, **kwargs)
