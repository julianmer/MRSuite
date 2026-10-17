####################################################################################################
#                                    test_challenge_models.py                                      #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-02                                                                              #
#                                                                                                  #
# Purpose: The challenge models' training script, two steps of each: every run writes checkpoints  #
#          that load by name alone, with the recipe of what it learned from, and WAND learns the   #
#          MRSI challenge's own water and lipids as a component of their own.                      #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import os

os.environ.setdefault("KERAS_TORCH_DEVICE", "cpu")      # before anything imports keras

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

import mrsuite.models
from mrsuite.datasets import mrsi_challenge
from mrsuite.simulators import mrs_sim
from mrsuite.training import Source, Truth

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "train" / "challenge_models.py"
spec = importlib.util.spec_from_file_location("challenge_models", SCRIPT)
challenge_models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(challenge_models)

pytestmark = pytest.mark.skipif(not (mrs_sim.ROOT / "src").exists(),
                                reason="MRS-Sim submodule not checked out")
HAVE_MRSI = all((mrsi_challenge.DATA_DIR / s).exists() for s in ("Sub1", "Sub2"))


#***********#
#   steps   #
#***********#
def test_each_run_checkpoints_sixteen_times():
    """The published runs' steps, which the default checkpoint interval divides."""
    for _, _, _, steps in challenge_models.MODELS.values():
        assert steps % (steps // 16) == 0


@pytest.mark.skipif(not challenge_models.challenge2016.BASIS.exists(),
                    reason="challenge submodule not checked out")
def test_any_number_of_steps_checkpoints(tmp_path):
    """Steps that a sixteenth does not divide (33: every 2 would not) checkpoint at the end."""
    challenge_models.train("mlp-supervised-ismrm2016", steps=33, to=tmp_path, batch=2, chunk=4)
    assert sorted(p.name for p in (tmp_path / "mlp-supervised-ismrm2016").iterdir()) == [
        "step-33"]


@pytest.mark.skipif(not challenge_models.challenge2016.BASIS.exists(),
                    reason="challenge submodule not checked out")
def test_a_seed_replays_the_run(tmp_path):
    """Same data and same initial weights: the same weights after training."""
    runs = [challenge_models.train("mlp-supervised-ismrm2016", steps=2, to=tmp_path / str(k),
                                   batch=2, chunk=4, seed=7) for k in range(2)]
    assert runs[0].fingerprint() == runs[1].fingerprint()


#************#
#   trains   #
#************#
@pytest.mark.parametrize("name", list(challenge_models.MODELS))
def test_a_run_trains_and_its_checkpoints_load(name, tmp_path):
    challenge, kind, objective, _ = challenge_models.MODELS[name]
    if challenge in ("mrsi", *challenge_models.RELEASE) and not HAVE_MRSI:
        pytest.skip("MRSI challenge subjects Sub1 and Sub2 not on disk")

    model = challenge_models.train(name, steps=2, every=1, to=tmp_path, batch=2, chunk=4,
                                   voxels=16, subjects=("Sub1",), release=("Sub1",))

    folder = tmp_path / name
    assert sorted(p.name for p in folder.iterdir()) == ["step-1", "step-2"]
    loaded = mrsuite.models.load(str(folder / "step-2"))
    assert type(loaded) is type(model) and loaded.fingerprint() == model.fingerprint()
    assert loaded.acquisition == model.acquisition

    recipe = json.loads((folder / "step-2" / "recipe.json").read_text())
    assert {k: recipe["training"][k] for k in ("objective", "steps", "batch", "learning_rate")
            } == {"objective": objective, "steps": 2, "batch": 2, "learning_rate": 1e-4}
    if challenge in challenge_models.RELEASE:
        assert recipe["generation"]["subjects"] == ["Sub1"]
        assert recipe["generation"]["signal"] == challenge_models.RELEASE[challenge]
    if kind == "wand":
        assert ("nuisance" in loaded.components) == (challenge in ("mrsi", "release-track1")
                                                     and name not in challenge_models.TWO_MASKS)
        assert ("baseline" in loaded.components) == (challenge not in challenge_models.RELEASE)
        assert loaded.zero_fill == 1
        assert bool(loaded.weights) == (name in challenge_models.WEIGHTED)
    if kind == "wand" and challenge == "mrsi":
        water = recipe["MRSsynMRS"]["Artifacts"]["Residual Water"]
        assert loaded.components[-3:] == ["nuisance", "baseline", "artifacts"]
        assert water["Applied"] is True and recipe["generation"]["nuisance"]["subjects"] == [
            "Sub1"]
        assert "resWater_cfg" not in recipe["generation"]["config"]     # one water, the release's


@pytest.mark.skipif(not HAVE_MRSI, reason="MRSI challenge subjects Sub1 and Sub2 not on disk")
def test_the_release_mlps_learn_what_the_release_holds(tmp_path):
    """The supervised one what the release varies (it holds Lac and Lac41 at zero), the
    self-supervised one rebuilds the whole signal, macromolecules included."""
    learn = lambda name: challenge_models.train(name, steps=1, to=tmp_path, batch=2,
                                                release=("Sub1",))
    supervised = learn("mlp-supervised-mrsi2024-release-track2")
    assert supervised.metabolites == [m for m in mrsi_challenge.NAMES.values()
                                      if m not in ("Lac", "Lac41")]
    assert learn("mlp-selfsupervised-mrsi2024-release-track2").metabolites == [
        *mrsi_challenge.NAMES.values(), *mrsi_challenge.MACROMOLECULES]


#**************#
#   nuisance   #
#**************#
def test_the_nuisance_weight_brings_its_error_to_the_metabolites_size():
    """Thirty times the metabolites in the window, and far more outside it, which counts
    for nothing."""
    class Thirtyfold(Source):
        def draw(self, n):
            rng = np.random.default_rng(n)
            metabolites = rng.normal(size=(n, 64)) + 1j * rng.normal(size=(n, 64))
            outside = np.zeros((n, 64), complex)
            outside[:, 48:] = 1e3                       # on points the window leaves out
            nuisance = 30 * metabolites + np.fft.ifft(outside, axis=1)
            return metabolites + nuisance, Truth(
                components={"metabolites": metabolites, "nuisance": nuisance})

    assert challenge_models.nuisance_weight(Thirtyfold(), 8, 40) == pytest.approx(1 / 900)



@pytest.mark.skipif(not HAVE_MRSI, reason="MRSI challenge subjects Sub1 and Sub2 not on disk")
def test_the_nuisance_adds_up_and_stands_to_the_metabolites_as_in_its_voxel():
    config = challenge_models.config("mrsi", "wand", voxels=16)
    spectra = mrs_sim.Source(config, chunk=4, components=True)
    source = challenge_models.Nuisance(spectra, ("Sub1",))
    clean, _ = mrs_sim.Source(config, chunk=4, components=True).draw(3)

    fids, truth = source.draw(3)

    nuisance = truth.components["nuisance"]
    assert np.allclose(fids, clean + nuisance)                  # added to the same spectra
    metabolites = sum(truth.components[m] for m in source.metabolites)
    ratio = np.linalg.norm(nuisance, axis=1) / np.linalg.norm(metabolites, axis=1)
    voxels = np.linalg.norm(source.signals, axis=1) / source.norms
    assert np.all([np.isclose(voxels, r, rtol=1e-5).any() for r in ratio])
    assert np.median(ratio) > 100                               # far above the metabolites


#*********************#
#   best checkpoint   #
#*********************#
def test_every_checkpoint_is_scored_and_the_benchmark_takes_the_best(tmp_path, monkeypatch):
    """Each saved checkpoint of a run gets its MOSAE on held-out spectra, and the benchmark
    takes the one best.json names instead of the last."""
    name = "mlp-supervised-ismrm2016"
    challenge_models.train(name, steps=2, every=1, to=tmp_path, batch=2, chunk=4)
    best = importlib.util.spec_from_file_location("best_checkpoint",
                                                  SCRIPT.parent / "best_checkpoint.py")
    best_checkpoint = importlib.util.module_from_spec(best)
    best.loader.exec_module(best_checkpoint)

    found, score = {}, best_checkpoint.score

    def scored(name, n):
        found.update(score(name, n, runs=tmp_path))
        return found

    config = importlib.util.spec_from_file_location(
        "config", SCRIPT.parents[1] / "benchmark" / "config.py")
    benchmark = importlib.util.module_from_spec(config)
    config.loader.exec_module(benchmark)
    monkeypatch.setattr(benchmark, "RUNS", tmp_path)
    assert benchmark.weights(name).endswith("step-2/model.safetensors"), "the last, unchosen"
    monkeypatch.setattr(best_checkpoint, "RUNS", tmp_path)
    monkeypatch.setattr(best_checkpoint, "score", scored)
    best_checkpoint.main([name, "--n", "8"])

    assert list(found) == ["step-1", "step-2"] and all(v > 0 for v in found.values())
    chosen = json.loads((tmp_path / name / "best.json").read_text())["checkpoint"]
    assert chosen == min(found, key=found.get)           # the lowest MOSAE
    benchmark.weights.cache_clear()                      # a session keeps what it first took
    assert benchmark.weights(name).endswith(f"{chosen}/model.safetensors")
