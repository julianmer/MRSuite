####################################################################################################
#                                          test_parity.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-27                                                                              #
#                                                                                                  #
# Purpose: The wrapped models against their original frameworks, run in the originals' own        #
#          environment (MRSUITE_REFERENCE_PYTHON), on the same spectra with the same weights:      #
#          seeded random ones always, and the published checkpoints from MRSUITE_CHECKPOINTS       #
#          when that folder is there.                                                              #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from mrsuite.datasets import challenge2016
from mrsuite.models import MLP, WAND
from mrsuite.models import convert
from mrsuite.models.base import spectra
from mrsuite.octave import scratch
from tests.models.test_models import (MLP_ACQUISITION, MLP_METABOLITES, WAND_ACQUISITION,
                                      WAND_METABOLITES, _spectra)

CHECKPOINTS = Path(os.environ.get("MRSUITE_CHECKPOINTS", "/nonexistent"))
BASIS_7T = Path.home() / "Desktop/Augmentrum/data/BasisSets/7T_sLASER_OIT_TE34.basis"

# as the originals' test scripts and run configs build them
JOBS = {
    "mlp": {"model": "mlp", "basis": str(BASIS_7T), "basisFmt": "7tslaser", "ppmlim": [0.5, 4.0],
            "dataType": "aumc2", "width": 512, "depth": 3},
    "wand": {"model": "wand", "basis": str(challenge2016.ROOT / "basisset_JMRUI") + "/",
             "specType": "synth", "arch": "unet", "dataType": "std_rw_p"},
}
PUBLISHED = {"mlp": ["mlp_super/epoch=0-step=7537664.ckpt",
                     "mlp_selfsuper/epoch=0-step=3572224.ckpt"],
             "wand": ["wand/epoch=0-step=465664.ckpt"]}

needs_reference = pytest.mark.skipif(not convert.PYTHON or not BASIS_7T.exists(),
                                     reason="no MRSUITE_REFERENCE_PYTHON, or no 7 T basis")


#************#
#   models   #
#************#
def _random(kind):
    """The wrapper with seeded random weights, running statistics included."""
    torch.manual_seed(0)
    model = (MLP(MLP_METABOLITES, 602, 957, MLP_ACQUISITION) if kind == "mlp" else
             WAND(WAND_METABOLITES, 1792, 2036, WAND_ACQUISITION))
    with torch.no_grad():
        for name, tensor in model.net.state_dict().items():
            if tensor.is_floating_point():
                tensor.copy_(torch.rand_like(tensor) + 0.5 if "running_var" in name
                             else 0.1 * torch.randn_like(tensor))
    return model


#************#
#   inputs   #
#************#
def _inputs(kind):
    """Spectra as both take them: synthetic ones at 7 T, the challenge's at 3 T."""
    if kind == "mlp":
        return spectra(_spectra(8).list())
    data, _, _ = challenge2016.load(4)
    return spectra(data.list())


#************#
#   parity   #
#************#
def _parity(model, kind):
    """The wrapper's output and the original's, on the same inputs and weights."""
    x = _inputs(kind)
    folder = Path(scratch("mrsuite_parity_"))
    torch.save(model.net.state_dict(), folder / "weights.pt")
    np.save(folder / "input.npy", x)
    convert.run_reference(type(model), {**JOBS[kind], "task": "forward",
                                        "weights": str(folder / "weights.pt"),
                                        "input": str(folder / "input.npy"),
                                        "output": str(folder / "output.npy")})
    return model.predict(x).numpy(), np.load(folder / "output.npy")


#************#
#   random   #
#************#
@needs_reference
@pytest.mark.parametrize("kind", ["mlp", "wand"])
def test_same_output_as_the_original(kind):
    ours, theirs = _parity(_random(kind), kind)

    assert ours.shape == theirs.shape
    np.testing.assert_allclose(ours, theirs, rtol=1e-5, atol=1e-6 * np.abs(theirs).max())


#***************#
#   published   #
#***************#
@needs_reference
@pytest.mark.parametrize("kind, checkpoint",
                         [(k, c) for k, cs in PUBLISHED.items() for c in cs])
def test_same_output_with_the_published_weights(kind, checkpoint):
    if not (CHECKPOINTS / checkpoint).exists():
        pytest.skip(f"{CHECKPOINTS / checkpoint} not there")
    cls = MLP if kind == "mlp" else WAND
    model = convert.convert(cls, str(CHECKPOINTS / checkpoint), JOBS[kind])
    ours, theirs = _parity(model, kind)

    np.testing.assert_allclose(ours, theirs, rtol=1e-5, atol=1e-6 * np.abs(theirs).max())
