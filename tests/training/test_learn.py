####################################################################################################
#                                          test_learn.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-28                                                                              #
#                                                                                                  #
# Purpose: Learning: a source's batches reach a model as real data does, the one Keras loop        #
#          trains it in place, checkpoints load back, and a source the model cannot learn from     #
#          is refused before any step. A toy source with an exact truth stands in for simulators.  #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import os

os.environ.setdefault("KERAS_TORCH_DEVICE", "cpu")      # before anything imports keras

import numpy as np
import pytest

from nifti_mrs_plus import NIfTI_MRS_Plus
from mrsuite.models import MLP, WAND
from mrsuite.models.base import spectra
from mrsuite.training import Source, Truth

from tests.models.test_models import (MLP_ACQUISITION, MLP_METABOLITES, WAND_ACQUISITION,
                                      WAND_METABOLITES)


#**************************************************************************************************#
#                                         Class ToySource                                          #
#**************************************************************************************************#
#                                                                                                  #
# One Lorentzian line per metabolite at random amplitude, a broad baseline and noise.              #
#                                                                                                  #
#**************************************************************************************************#
class ToySource(Source):
    """One Lorentzian line per metabolite at random amplitude, a broad baseline and noise."""

    def __init__(self, metabolites, acquisition, seed=0):
        self.metabolites, self.acquisition = list(metabolites), dict(acquisition)
        self.rng = np.random.default_rng(seed)
        points, bandwidth = int(acquisition["points"]), acquisition["bandwidth"]
        self.t = np.arange(points) / bandwidth
        self.hz = np.linspace(-0.3, 0.3, len(metabolites)) * bandwidth    # one line each

    def draw(self, n):
        amps = self.rng.uniform(0, 2, (n, len(self.metabolites)))
        lines = np.exp(2j * np.pi * self.hz[:, None] * self.t - self.t / 0.05)
        components = {m: amps[:, i, None] * lines[i] for i, m in enumerate(self.metabolites)}
        components["baseline"] = self.rng.uniform(0, 1, (n, 1)) * np.exp(-self.t / 0.002)
        noise = 0.01 * (self.rng.normal(size=(n, len(self.t))) +
                        1j * self.rng.normal(size=(n, len(self.t))))
        fids = sum(components.values()) + noise
        return fids, Truth(concentrations=dict(zip(self.metabolites, amps.T)),
                           components=components)


#*************#
#   batches   #
#*************#
def test_a_batch_reaches_the_model_as_real_data_does():
    data, truth = ToySource(MLP_METABOLITES, MLP_ACQUISITION).batch(4)

    assert isinstance(data, NIfTI_MRS_Plus) and data.numpy().shape == (4, 1, 1, 1, 1024)
    assert np.allclose(spectra(data), spectra(data.list()))       # one array, or objects
    assert set(truth.concentrations) == set(MLP_METABOLITES)


#************#
#   learns   #
#************#
def test_a_model_learns_in_place(tmp_path):
    mlp = MLP(MLP_METABOLITES, 602, 957, MLP_ACQUISITION)
    before = mlp.fingerprint()

    losses = mlp.learn(ToySource(MLP_METABOLITES, MLP_ACQUISITION), steps=150, every=50,
                       lr=1e-3, to=str(tmp_path))

    assert len(losses) == 3 and losses[-1] < losses[0]
    assert mlp.fingerprint() != before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["step-100", "step-150", "step-50"]
    assert MLP.from_pretrained(tmp_path / "step-150").fingerprint() == mlp.fingerprint()


#****************#
#   components   #
#****************#
def test_wand_learns_the_components_it_separates():
    wand = WAND(WAND_METABOLITES, 1792, 2036, WAND_ACQUISITION)
    source = ToySource(WAND_METABOLITES, WAND_ACQUISITION)
    data, truth = source.batch(2)

    target = wand.target(truth, data, "components")
    assert target.shape == tuple(wand.predict(wand.prepare(data)).shape)

    losses = wand.learn(source, steps=2, batch=2)
    assert len(losses) == 1 and np.isfinite(losses[0])


#*************#
#   refused   #
#*************#
def test_what_a_model_cannot_learn_from_is_refused():
    mlp = MLP(MLP_METABOLITES, 602, 957, MLP_ACQUISITION)

    with pytest.raises(ValueError, match="points"):
        mlp.learn(ToySource(MLP_METABOLITES, WAND_ACQUISITION), steps=1)
    with pytest.raises(ValueError, match="learns from"):
        mlp.learn(ToySource(MLP_METABOLITES, MLP_ACQUISITION), steps=1, objective="spectra")
    with pytest.raises(KeyError, match="gives none for"):
        mlp.learn(ToySource(MLP_METABOLITES[:-1], MLP_ACQUISITION), steps=1)
