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
os.environ.setdefault("KERAS_BACKEND", "torch")         # as learn sets it

import json

import numpy as np
import pytest
import torch
from safetensors.numpy import load_file

from nifti_mrs_plus import NIfTI_MRS_Plus
from mrsuite.models import MLP, WAND
from mrsuite.models.base import spectra
from mrsuite.training import Source, Truth
from mrsuite.training import loop
from mrsuite.training.loop import OPTIMIZER, last_checkpoint
from tests.conftest import needs_fsl_mrs

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
        lines = np.exp(2j * np.pi * self.hz[:, None] * self.t - self.t / 0.05)
        self.basis = dict(zip(self.metabolites, lines))

    def draw(self, n):
        amps = self.rng.uniform(0, 2, (n, len(self.metabolites)))
        components = {m: amps[:, i, None] * self.basis[m]
                      for i, m in enumerate(self.metabolites)}
        components["baseline"] = self.rng.uniform(0, 1, (n, 1)) * np.exp(-self.t / 0.002)
        noise = 0.01 * (self.rng.normal(size=(n, len(self.t))) +
                        1j * self.rng.normal(size=(n, len(self.t))))
        fids = sum(components.values()) + noise
        return fids, Truth(concentrations=dict(zip(self.metabolites, amps.T)),
                           components=components)


#**************************************************************************************************#
#                                         Class CoilSource                                         #
#**************************************************************************************************#
#                                                                                                  #
# The toy spectra as two coils, each a scaled copy.                                                #
#                                                                                                  #
#**************************************************************************************************#
class CoilSource(ToySource):
    """The toy spectra as two coils, each a scaled copy."""

    dim_tags = ["DIM_COIL"]

    def draw(self, n):
        fids, truth = super().draw(n)
        coils = np.array([1.0, 0.5])
        truth.components = {k: v[..., None] * coils for k, v in truth.components.items()}
        return fids[..., None] * coils, truth


#*************#
#   batches   #
#*************#
def test_a_batch_reaches_the_model_as_real_data_does():
    data, truth = ToySource(MLP_METABOLITES, MLP_ACQUISITION).batch(4)

    assert isinstance(data, NIfTI_MRS_Plus) and data.numpy().shape == (4, 1, 1, 1, 1024)
    assert np.allclose(spectra(data), spectra(data.list()))       # one array, or objects
    assert set(truth.concentrations) == set(MLP_METABOLITES)


#***********#
#   coils   #
#***********#
def test_a_batch_takes_any_shape_nifti_mrs_allows():
    """Several coils are a fifth dimension with its tag, and the truth has it too."""
    data, truth = CoilSource(MLP_METABOLITES, MLP_ACQUISITION).batch(3)

    assert data.numpy().shape == (3, 1, 1, 1, 1024, 2)
    assert data.list()[0].dim_tags[0] == "DIM_COIL"
    assert truth.components["NAA"].shape == (3, 1024, 2)


#************#
#   scaled   #
#************#
def test_the_scaled_objective_weighs_each_error_by_its_range():
    """Each metabolite's error over the width of its range; one held fixed, unscaled."""
    import torch

    source = ToySource(MLP_METABOLITES, MLP_ACQUISITION)
    source.ranges = {m: (0.0, 2.0) for m in MLP_METABOLITES}
    source.ranges[MLP_METABOLITES[0]] = (0.5, 0.5)
    mlp = MLP(MLP_METABOLITES, 602, 957, MLP_ACQUISITION)
    mlp.learn(source, steps=2, batch=4, objective="scaled")

    n = len(MLP_METABOLITES)
    loss = mlp.loss(torch.ones(2, n), torch.zeros(2, n + 11), "scaled")
    assert float(loss) == pytest.approx((1.0 + (n - 1) * 0.5) / n)
    with pytest.raises(ValueError, match="states none"):
        mlp.learn(ToySource(MLP_METABOLITES, MLP_ACQUISITION), steps=1, objective="scaled")


#**************#
#   recycled   #
#**************#
def test_a_recycled_batch_holds_the_next_spectra_in_the_same_objects():
    source, again = (ToySource(MLP_METABOLITES, MLP_ACQUISITION, seed=3) for _ in range(2))
    first, _ = source.batch(4, recycle=True)
    second, _ = source.batch(4, recycle=True)
    again.batch(4)
    expected, _ = again.batch(4)

    assert all(a is b for a, b in zip(first.nifti_list, second.nifti_list))
    assert np.array_equal(second.numpy(), expected.numpy())
    assert np.array_equal(spectra(second), spectra(second.list()))
    assert not any(a is b for a, b in zip(again.batch(4)[0].nifti_list,
                                          again.batch(4)[0].nifti_list))


#*************#
#   in turn   #
#*************#
def test_learning_draws_its_batches_in_turn_and_no_more():
    """Prepared ahead by a thread, the batches are the ones drawn in turn: Keras asks for
    one more than the steps, and the source stands there afterwards, as without it."""
    source, again = (ToySource(MLP_METABOLITES, MLP_ACQUISITION, seed=5) for _ in range(2))
    MLP(MLP_METABOLITES, 602, 957, MLP_ACQUISITION).learn(source, steps=6, every=3, batch=4)
    for _ in range(7):
        again.draw(4)

    assert np.array_equal(source.draw(4)[0], again.draw(4)[0])


#************#
#   learns   #
#************#
def test_a_model_learns_in_place(tmp_path):
    mlp = MLP(MLP_METABOLITES, 602, 957, MLP_ACQUISITION)
    before = mlp.fingerprint()
    source = ToySource(MLP_METABOLITES, MLP_ACQUISITION)
    source.recipe = lambda: {"MRSsynMRS": {"Experiment": {"ID": "toy"}}}

    losses = mlp.learn(source, steps=150, every=50, lr=1e-3, to=str(tmp_path))

    assert len(losses) == 3 and losses[-1] < losses[0]
    assert mlp.fingerprint() != before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["step-100", "step-150", "step-50"]
    assert MLP.from_pretrained(tmp_path / "step-150").fingerprint() == mlp.fingerprint()

    # each checkpoint carries the recipe that made its data, with the training stated
    recipe = json.loads((tmp_path / "step-150" / "recipe.json").read_text())
    assert recipe["MRSsynMRS"]["Experiment"]["ID"] == "toy"
    assert recipe["training"]["steps"] == 150 and recipe["training"]["objective"]
    assert json.loads((tmp_path / "step-50" / "recipe.json").read_text())["training"][
        "steps"] == 50
    assert mlp.recipe == recipe and recipe["training"]["from"] is None

    # learning on says what the weights started from, however often it is done
    tuned = MLP.from_pretrained(tmp_path / "step-150")
    tuned.learn(source, steps=2)
    tuned.learn(source, steps=3)
    first = tuned.recipe["training"]["from"]["training"]
    assert tuned.recipe["training"]["steps"] == 3 and first["steps"] == 2
    # a folder outside the checkout by its name alone, never where it lies on this machine
    assert first["from"] == {"source": {"name": "step-150", "revision": None},
                             "training": recipe["training"]}
    assert tuned.source is None             # the weights are no longer the ones loaded


def test_learning_resumes_with_adams_state_and_counts_on(tmp_path):
    source = ToySource(MLP_METABOLITES, MLP_ACQUISITION)
    MLP(MLP_METABOLITES, 602, 957, MLP_ACQUISITION).learn(source, steps=20, every=10, lr=1e-3,
                                                          to=str(tmp_path))
    resumed = MLP.from_pretrained(tmp_path / "step-20")
    resumed.learn(source, steps=10, lr=1e-4, to=str(tmp_path), resume=str(tmp_path / "step-20"))

    def scalars(step):          # Adam's step count (an integer) and its learning rate
        state = load_file(str(tmp_path / step / OPTIMIZER))
        return {v.dtype.kind: v.item() for v in state.values() if v.ndim == 0}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["step-10", "step-20", "step-30"]
    assert scalars("step-20")["i"] == 20 and scalars("step-30")["i"] == 30
    assert scalars("step-30")["f"] == pytest.approx(1e-4)


def test_a_checkpoint_is_there_only_once_written_whole(tmp_path, monkeypatch):
    source = ToySource(MLP_METABOLITES, MLP_ACQUISITION)

    def learn():
        MLP(MLP_METABOLITES, 602, 957, MLP_ACQUISITION).learn(source, steps=2, every=2,
                                                              to=str(tmp_path))

    def stopped(*args):                     # learning stops between the weights and Adam's state
        raise RuntimeError("stopped")
    monkeypatch.setattr(loop, "save_file", stopped)
    with pytest.raises(RuntimeError, match="stopped"):
        learn()
    assert [p.name for p in tmp_path.iterdir()] == ["step-2.partial"]

    monkeypatch.undo()
    learn()
    learn()                                 # a new training into the same folder replaces it
    assert [p.name for p in tmp_path.iterdir()] == ["step-2"]
    assert (tmp_path / "step-2" / OPTIMIZER).exists()


def test_the_last_checkpoint_is_the_highest_step_written_whole(tmp_path):
    for name in ("step-10", "step-200", "step-30", "step-400.partial"):
        (tmp_path / name).mkdir()
    assert last_checkpoint(tmp_path) == tmp_path / "step-200"


def test_a_converted_model_learns_on_from_step_zero(tmp_path):
    mlp = MLP(MLP_METABOLITES, 602, 957, MLP_ACQUISITION)
    mlp.recipe = {"training": {"model": "mrsuite.models.mlp.MLP", "checkpoint": {}}}  # as convert
    mlp.learn(ToySource(MLP_METABOLITES, MLP_ACQUISITION), steps=4, every=2, to=str(tmp_path))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["step-2", "step-4"]


#***********************#
#   self - supervised   #
#***********************#
def test_wand_weighs_each_components_error_as_told():
    plain = WAND(["metabolites"], 0, 64, WAND_ACQUISITION, nuisance=["nuisance"], baseline=False)
    weighed = WAND(["metabolites"], 0, 64, WAND_ACQUISITION, nuisance=["nuisance"],
                   baseline=False, weights={"nuisance": 0.25})
    assert weighed.components == ["metabolites", "nuisance", "artifacts"]
    target = torch.zeros(1, 2, 64, 3)
    out = target.clone()
    out[..., 0], out[..., 1] = 1.0, 2.0          # the metabolites off by 1, the nuisance by 2
    assert float(plain.loss(target, out, "components")) == pytest.approx((1 + 4) / 3)
    assert float(weighed.loss(target, out, "components")) == pytest.approx((1 + 1) / 3)
    with pytest.raises(ValueError, match="no component"):
        WAND(["metabolites"], 0, 64, WAND_ACQUISITION, weights={"water": 1.0})


@needs_fsl_mrs
def test_the_mlp_learns_from_spectra_alone():
    """The 'spectra' objective fits the source's basis through the original signal
    model: no concentrations are read, and the loss falls all the same."""
    import torch

    torch.manual_seed(0)
    mlp = MLP(MLP_METABOLITES, 602, 957, MLP_ACQUISITION)
    before = mlp.fingerprint()
    source = ToySource(MLP_METABOLITES, MLP_ACQUISITION)
    source.draw = lambda n, draw=source.draw: (draw(n)[0], Truth())    # spectra only

    losses = mlp.learn(source, steps=90, every=30, lr=1e-3, objective="spectra")

    assert len(losses) == 3 and np.isfinite(losses).all() and losses[-1] < losses[0]
    assert mlp.fingerprint() != before


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
        mlp.learn(ToySource(MLP_METABOLITES, MLP_ACQUISITION), steps=1, objective="phases")
    with pytest.raises(KeyError, match="gives none for"):
        mlp.learn(ToySource(MLP_METABOLITES[:-1], MLP_ACQUISITION), steps=1)
    with pytest.raises(ValueError, match="single spectra"):
        mlp.learn(CoilSource(MLP_METABOLITES, MLP_ACQUISITION), steps=1)

    bare = ToySource(MLP_METABOLITES, MLP_ACQUISITION)
    bare.basis = None
    with pytest.raises(ValueError, match="gives none"):
        mlp.learn(bare, steps=1, objective="spectra")
    with pytest.raises(KeyError, match="lacks"):
        mlp.learn(ToySource(MLP_METABOLITES[:-1] + ["Other"], MLP_ACQUISITION), steps=1,
                  objective="spectra")
