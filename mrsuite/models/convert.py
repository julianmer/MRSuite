####################################################################################################
#                                            convert.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-27                                                                              #
#                                                                                                  #
# Purpose: An original checkpoint as a MRSuite model, ready for save_pretrained and the Hub. The   #
#          config - basis names, spectral window, acquisition - is read off the original code      #
#          itself, run in its own environment by reference.py, so nothing is re-derived here.      #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import copy
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Type

import torch

from mrsuite.core.provenance import git_version, portable, portable_file
from mrsuite.models.base import classpath
from mrsuite.models.pytorch import TorchModel

REFERENCE = Path(__file__).with_name("reference.py")

# the Python of an environment the original repositories run in
PYTHON = os.environ.get("MRSUITE_REFERENCE_PYTHON")


#*******************#
#   run reference   #
#*******************#
def run_reference(cls: Type[TorchModel], job: Dict[str, Any],
                  python: str = None) -> Dict[str, Any]:
    """
    Run a job through the model's original code.

    Args:
        cls (Type[TorchModel]): The MRSuite model; its ``repo`` is the original's
            repository, which the job runs in.
        job (Dict[str, Any]): The task and the original's settings, as reference.py reads them.
        python (str): The original environment's Python. Defaults to ``PYTHON``.

    Returns:
        Dict[str, Any]: What reference.py wrote back.
    """
    from mrsuite.octave import scratch

    python = python or PYTHON
    if not python:
        raise RuntimeError("set MRSUITE_REFERENCE_PYTHON to an environment the original runs in")
    folder = scratch("mrsuite_reference_")
    try:
        path, result = os.path.join(folder, "job.json"), os.path.join(folder, "result.json")
        with open(path, "w") as fh:
            json.dump({**job, "result": result}, fh)
        proc = subprocess.run([python, str(REFERENCE), path], cwd=cls.repo, capture_output=True,
                              text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        if proc.returncode != 0:
            raise RuntimeError(f"the original {cls.__name__} failed:\n{proc.stderr[-3000:]}")
        with open(result) as fh:
            return json.load(fh)
    finally:
        shutil.rmtree(folder, ignore_errors=True)


#*************#
#   convert   #
#*************#
def convert(cls: Type[TorchModel], checkpoint: str, job: Dict[str, Any],
            python: str = None, **kwargs) -> TorchModel:
    """
    An original Lightning checkpoint as a MRSuite model.

    Args:
        cls (Type[TorchModel]): MLP or WAND.
        checkpoint (str): The original's ``.ckpt``.
        job (Dict[str, Any]): The original's settings - its basis and the rest - as its
            test scripts pass them; see reference.py.
        python (str): The original environment's Python. Defaults to ``PYTHON``.
        **kwargs: Further constructor arguments, e.g. the MLP's width and depth.

    Returns:
        TorchModel: With the checkpoint's network weights, loaded strictly, and a recipe
        that says where they came from: the checkpoint and the job's basis by their
        portable paths (see ``provenance.portable``), the checkpoint's SHA-256, the job,
        and the original's version the config was read from.
    """
    config = run_reference(cls, {**job, "task": "config"}, python)
    model = cls(**config, **kwargs)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    model.net.load_state_dict({k[len("net."):]: v for k, v in state.items()
                               if k.startswith("net.")})
    model.recipe = {"training": {
        "model": classpath(cls), "checkpoint": portable_file(checkpoint),
        "job": {**copy.deepcopy(job), "basis": portable(Path(cls.repo, job["basis"]))},
        "converted with": {cls.repo.name: git_version(str(cls.repo))}}}
    return model
