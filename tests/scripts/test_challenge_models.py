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


#************#
#   trains   #
#************#
@pytest.mark.parametrize("name", list(challenge_models.MODELS))
def test_a_run_trains_and_its_checkpoints_load(name, tmp_path):
    challenge, kind, objective, _ = challenge_models.MODELS[name]
    if challenge == "mrsi" and not HAVE_MRSI:
        pytest.skip("MRSI challenge subjects Sub1 and Sub2 not on disk")

    model = challenge_models.train(name, steps=2, every=1, to=tmp_path, batch=2, chunk=4,
                                   voxels=16, subjects=("Sub1",))

    folder = tmp_path / name
    assert sorted(p.name for p in folder.iterdir()) == ["step-1", "step-2"]
    loaded = mrsuite.models.load(str(folder / "step-2"))
    assert type(loaded) is type(model) and loaded.fingerprint() == model.fingerprint()
    assert loaded.acquisition == model.acquisition

    recipe = json.loads((folder / "step-2" / "recipe.json").read_text())
    assert {k: recipe["training"][k] for k in ("objective", "steps", "batch", "learning_rate")
            } == {"objective": objective, "steps": 2, "batch": 2, "learning_rate": 1e-4}
    water = recipe["MRSsynMRS"]["Artifacts"]["Residual Water"]
    if kind == "wand":
        assert loaded.components[-3:] == ["nuisance", "baseline", "artifacts"]
        assert water["Applied"] is True and recipe["generation"]["nuisance"]["subjects"] == [
            "Sub1"]
        assert "resWater_cfg" not in recipe["generation"]["config"]     # one water, the release's


#**************#
#   nuisance   #
#**************#
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
