####################################################################################################
#                                          test_models.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-27                                                                              #
#                                                                                                  #
# Purpose: The Model interface: a model is a pipeline step in any framework, refuses data it was   #
#          not trained for, records exactly what ran, and survives the Hub's save and load.        #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from nifti_mrs.create_nmrs import gen_nifti_mrs

from nifti_mrs_plus import NIfTI_MRS_Plus
from mrsuite.core.pipeline import Pipeline
from mrsuite.core.result import FitResult
from mrsuite.datasets import challenge2016
from mrsuite.models import MLP, WAND, Model
from mrsuite.training import Truth

MLP_METABOLITES = ["Ala", "Asc", "Asp", "Cr", "GABA", "Gln", "Glu", "Gly", "GPC", "GSH", "mIns",
                   "Lac", "NAAG", "NAA", "PCh", "PCr", "PE", "Scyllo", "Ser", "Tau", "MM_CMR"]
MLP_ACQUISITION = {"points": 1024, "bandwidth": 3000.0, "frequency": 298.2, "centre": 4.65}
WAND_METABOLITES = ["Ace", "Ala", "Asc", "Asp", "Cr", "GABA", "GPC", "GSH", "Glc", "Gln", "Glu",
                    "Gly", "Ins", "Lac", "Mac", "NAA", "NAAG", "PCho", "PCr", "PE", "Tau", "sIns"]
WAND_ACQUISITION = {"points": 2048, "bandwidth": 4000.0, "frequency": 123.22, "centre": 4.65}


#*********#
#   mlp   #
#*********#
@pytest.fixture(scope="module")
def mlp():
    return MLP(MLP_METABOLITES, 602, 957, MLP_ACQUISITION)


#*************#
#   spectra   #
#*************#
def _spectra(n, points=1024, bandwidth=3000.0, frequency=298.2):
    """Decaying FIDs as NIfTI-MRS+, acquired as the MLP was trained."""
    rng = np.random.default_rng(0)
    t = np.arange(points) / bandwidth
    items = []
    for _ in range(n):
        fid = sum(rng.uniform(1, 2) * np.exp(2j * np.pi * rng.uniform(-400, 400) * t - t / 0.05)
                  for _ in range(5))
        items.append(gen_nifti_mrs(fid.reshape(1, 1, 1, -1), 1 / bandwidth, frequency))
    return NIfTI_MRS_Plus(items)


#****************#
#   quantifies   #
#****************#
def test_a_quantifier_is_a_pipeline_step(mlp):
    data = _spectra(3)
    results = Pipeline([mlp]).run(data)

    assert len(results) == 3 and all(isinstance(r, FitResult) for r in results)
    assert results[0].metabolites == MLP_METABOLITES
    assert results[0].method == "mrsuite:MLP"
    assert set(results[0].raw) >= {"gamma", "sigma", "phi0", "baseline5"}
    assert results[0].conc("internal")["Cr"] + results[0].conc("internal")["PCr"] == \
        pytest.approx(1)


#*************#
#   batches   #
#*************#
def test_batching_changes_nothing(mlp):
    """Eval mode: a spectrum's result does not depend on what it was batched with.

    Up to float32 rounding: matrix products may sum in another order for another batch size.
    """
    data = _spectra(5)
    together = mlp(data)
    mlp.batch_size = 2
    try:
        apart = mlp(data)
    finally:
        mlp.batch_size = 64
    single = mlp(data.list()[3])

    for a, t in zip(apart + [single], together + [together[3]]):
        assert a.conc() == pytest.approx(t.conc(), rel=1e-6)


#****************#
#   provenance   #
#****************#
def test_the_provenance_pins_the_model(mlp):
    result = mlp(_spectra(1).list()[0])
    item = result.provenance[-1]
    details = json.loads(item["Details"])

    assert item["Program"] == "MRSuite" and item["Method"] == "MLP"
    assert details["model"] == "mrsuite.models.mlp.MLP"
    assert details["weights"] == mlp.fingerprint() and details["torch"]
    assert details["ood-robust-mrs"]


#************#
#   refuse   #
#************#
def test_data_acquired_otherwise_is_refused(mlp):
    with pytest.raises(ValueError, match="frequency"):
        mlp(_spectra(1, frequency=123.2))
    with pytest.raises(ValueError, match="points"):
        mlp(_spectra(1, points=2048))


#*********#
#   hub   #
#*********#
def test_save_and_load_keep_the_model(mlp, tmp_path):
    mlp.save_pretrained(tmp_path)
    loaded = MLP.from_pretrained(tmp_path)

    assert loaded.fingerprint() == mlp.fingerprint()
    assert loaded.metabolites == mlp.metabolites and loaded.acquisition == mlp.acquisition
    assert loaded.source["name"] == tmp_path.name       # a folder by its portable path
    data = _spectra(2)
    assert [r.conc() for r in loaded(data)] == [r.conc() for r in mlp(data)]


#**********#
#   load   #
#**********#
def test_a_checkpoint_loads_by_name_alone(mlp, tmp_path):
    """Every save records its class, so mrsuite.models.load needs only the name."""
    import mrsuite.models

    mlp.save_pretrained(tmp_path)
    with open(tmp_path / "config.json") as fh:
        assert json.load(fh)["model"] == "mrsuite.models.mlp.MLP"

    loaded = mrsuite.models.load(tmp_path)
    assert type(loaded) is MLP and loaded.fingerprint() == mlp.fingerprint()

    stranger = tmp_path / "stranger"
    stranger.mkdir()
    (stranger / "config.json").write_text('{"model": "os.path.join"}')
    with pytest.raises(ValueError, match="MRSuite model"):
        mrsuite.models.load(stranger)


#************#
#   recipe   #
#************#
def test_the_recipe_and_its_card_travel_with_the_model(tmp_path):
    """A save writes the recipe and a card with its MRSsynMRS table; a load reads it back."""
    import mrsuite.models
    from mrsuite.training import recipe

    model = MLP(MLP_METABOLITES, 602, 957, MLP_ACQUISITION)
    model.recipe = {"MRSsynMRS": {"Experiment": {"ID": "toy", "Nucleus": "1H"}},
                    "training": {"objective": "concentrations", "steps": 3}}
    model.save_pretrained(tmp_path / "trained")

    files = {p.name for p in (tmp_path / "trained").iterdir()}
    assert files == {"config.json", "model.safetensors", "recipe.json", "README.md"}
    card = (tmp_path / "trained" / "README.md").read_text()
    assert recipe.table(model.recipe) in card and '"steps": 3' in card
    assert f'mrsuite.models.load("{tmp_path / "trained"}")' in card
    assert model.fingerprint() in card and "https://arxiv.org/abs/2511.23135" in card
    assert mrsuite.models.load(tmp_path / "trained").recipe == model.recipe

    # saved again without one, into the same folder: nothing of the first save stays
    model.recipe = None
    model.save_pretrained(tmp_path / "trained")
    assert not (tmp_path / "trained" / "recipe.json").exists()
    assert "Not recorded." in (tmp_path / "trained" / "README.md").read_text()
    assert mrsuite.models.load(tmp_path / "trained").recipe is None


#**********#
#   push   #
#**********#
def test_a_push_sends_config_weights_recipe_and_card(monkeypatch, tmp_path):
    """One commit with all four, the card loading the model by the repo's full name, and
    no recipe left behind from an earlier push."""
    from types import SimpleNamespace

    from huggingface_hub import HfApi, ModelCard

    sent = {}

    def create_repo(self, repo_id, **kwargs):          # the Hub adds the account's namespace
        return SimpleNamespace(repo_id=repo_id if "/" in repo_id else f"someone/{repo_id}")

    def upload_folder(self, *, repo_id, folder_path, delete_patterns=None, **kwargs):
        sent.update(repo=repo_id, files={p.name for p in Path(folder_path).iterdir()},
                    card=(Path(folder_path) / "README.md").read_text(), deleted=delete_patterns)
        return "commit"

    monkeypatch.setattr(HfApi, "create_repo", create_repo)
    monkeypatch.setattr(HfApi, "upload_folder", upload_folder)
    model = MLP(MLP_METABOLITES, 602, 957, MLP_ACQUISITION)
    model.recipe = {"training": {"steps": 3}}
    model.push_to_hub("mlp-test", model_card_kwargs={"license": "apache-2.0"})

    assert sent["repo"] == "someone/mlp-test"
    assert sent["files"] == {"config.json", "model.safetensors", "recipe.json", "README.md"}
    assert 'mrsuite.models.load("someone/mlp-test")' in sent["card"]
    assert not sent["deleted"]
    # the licence is the weights', given with the push, never the class'
    assert ModelCard(sent["card"]).data.license == "apache-2.0"

    model.recipe = None
    model.save_pretrained(tmp_path / "local", push_to_hub=True, repo_id="someone/other")
    assert sent["repo"] == "someone/other"
    assert 'mrsuite.models.load("someone/other")' in sent["card"]
    assert sent["deleted"] == ["recipe.json"]       # none left from an earlier push
    assert ModelCard(sent["card"]).data.license is None


#**********************#
#   batch statistics   #
#**********************#
def test_wand_normalises_with_the_batch_and_keeps_its_weights():
    """As the paper's evaluation ran it: the batch's own statistics, not the stored
    averages, and the weights exactly as they were afterwards."""
    torch.manual_seed(0)
    wand = WAND(WAND_METABOLITES, 1792, 2036, WAND_ACQUISITION)
    x = torch.randn(4, 2, 2048)
    before = wand.fingerprint()

    batch = wand.predict(x)
    wand.batch_statistics = False
    stored = wand.predict(x)

    assert not torch.allclose(batch, stored)
    assert wand.fingerprint() == before


#****************#
#   decomposes   #
#****************#
@pytest.mark.skipif(not challenge2016.BASIS.exists(), reason="challenge submodule not checked out")
def test_a_decomposition_comes_back_as_spectra():
    data, _, _ = challenge2016.load(2)
    wand = WAND(WAND_METABOLITES, 1792, 2036, WAND_ACQUISITION)
    out = wand(data)
    item, source = out.list()[0], data.list()[0]

    assert isinstance(out, NIfTI_MRS_Plus) and len(out) == 2
    assert item.shape == (1, 1, 1, 2048, len(WAND_METABOLITES) + 2)
    assert item.dim_tags[0] == "DIM_USER_0"
    assert item.hdr_ext["dim_5_header"]["Component"]["Value"][-2:] == ["baseline", "artifacts"]
    assert item.SpecFreqChemShift == source.SpecFreqChemShift
    assert item.hdr_ext["ProcessingApplied"][-1]["Method"] == "WAND"


#*****************#
#   keep/remove   #
#*****************#
@pytest.mark.skipif(not challenge2016.BASIS.exists(), reason="challenge submodule not checked out")
def test_wand_as_a_processing_step_removes_or_keeps_components():
    data, _, _ = challenge2016.load(2)
    wand = WAND(WAND_METABOLITES, 1792, 2036, WAND_ACQUISITION)

    cleaned = wand(data, remove=["artifacts"])
    untouched = wand(data, remove=[])
    everything = wand(data, keep=["metabolites", "baseline", "artifacts"])
    nothing = wand(data, remove=["metabolites", "baseline", "artifacts"])

    assert [c.shape for c in cleaned.list()] == [d.shape for d in data.list()]
    assert np.allclose(untouched[0][:], data[0][:])
    assert np.allclose(nothing[0][:], data[0][:] - everything[0][:])
    step = cleaned[0].hdr_ext["ProcessingApplied"][-1]
    assert json.loads(step["Details"])["kwargs"] == {"remove": ["artifacts"]}
    with pytest.raises(ValueError, match="no component"):
        wand(data, keep=["NotAMetabolite"])
    with pytest.raises(ValueError, match="not both"):
        wand(data, keep=["NAA"], remove=["baseline"])


#*****************#
#   zero-filled   #
#*****************#
def test_a_zero_filled_wand_reads_twice_the_points_and_answers_on_the_datas_own(tmp_path):
    """Zero-filling is undone exactly: what comes back sits on the data's own points, its
    window where the zero-filled grid puts it, and the components it learns add up to the
    zero-filled input."""
    points, first, last, rng = 256, 10, 506, np.random.default_rng(1)
    acquisition = {"points": points, "bandwidth": 1200.0, "frequency": 123.2, "centre": 4.7}
    wand = WAND(["A"], first, last, acquisition, baseline=False, zero_fill=2,
                weights={"artifacts": 0.5})
    fids = rng.normal(size=(2, points)) + 1j * rng.normal(size=(2, points))
    items = [gen_nifti_mrs(f.reshape(1, 1, 1, -1), 1 / 1200.0, 123.2) for f in fids]

    x = wand.prepare(items).numpy()
    spec = np.fft.fft(fids, n=2 * points)
    np.testing.assert_allclose(x[:, 0] + 1j * x[:, 1], spec)

    out = torch.zeros(2, 2, last - first, 2, dtype=torch.float64)    # the window all 'A'
    out[:, 0, :, 0] = torch.from_numpy(spec.real[:, first:last])
    out[:, 1, :, 0] = torch.from_numpy(spec.imag[:, first:last])
    window = np.zeros_like(spec)
    window[:, first:last] = spec[:, first:last]
    a = np.fft.ifft(window)[:, :points]                 # 'A' back on the data's points
    kept = wand.interpret(out, items, keep=["A"])
    np.testing.assert_allclose([np.asarray(k[:]).ravel() for k in kept], a, atol=1e-9)
    removed = wand.interpret(out, items, remove=["A"])
    np.testing.assert_allclose([np.asarray(r[:]).ravel() for r in removed], fids - a, atol=1e-9)
    edges = np.pad(spec[:, first:last], ((0, 0), (first, 2 * points - last)), mode="edge")
    every = np.asarray(wand.interpret(out, items)[0][:]).reshape(points, 2)
    np.testing.assert_allclose(every[:, 0], np.fft.ifft(edges[0])[:points], atol=1e-9)

    target = wand.target(Truth(components={"A": fids / 2}), NIfTI_MRS_Plus(items), "components")
    inside = x[:, :, first:last]
    np.testing.assert_allclose(target.sum(-1), inside - inside.mean(-1, keepdims=True),
                               rtol=1e-5, atol=1e-5 * np.abs(x).max())

    wand.save_pretrained(tmp_path)
    loaded = WAND.from_pretrained(tmp_path)
    assert (loaded.zero_fill, loaded.baseline, loaded.weights, loaded.components) == \
        (2, False, {"artifacts": 0.5}, ["A", "artifacts"])


#**************#
#   nuisance   #
#**************#
@pytest.mark.skipif(not challenge2016.BASIS.exists(), reason="challenge submodule not checked out")
def test_a_nuisance_is_a_component_of_its_own_and_no_metabolite():
    """Water and lipids, say: separated beside the metabolites, taken out on their own,
    and never kept as one of the metabolites."""
    data, _, _ = challenge2016.load(2)
    wand = WAND(WAND_METABOLITES, 1792, 2036, WAND_ACQUISITION, nuisance=["nuisance"])

    item = wand(data).list()[0]
    assert item.hdr_ext["dim_5_header"]["Component"]["Value"][-3:] == [
        "nuisance", "baseline", "artifacts"]

    assert np.allclose(wand(data, keep=["metabolites"])[0][:],
                       wand(data, keep=WAND_METABOLITES)[0][:])
    nuisance = wand(data, keep=["nuisance"])[0][:]
    assert np.abs(nuisance).max() > 0
    assert np.allclose(wand(data, remove=["nuisance"])[0][:], data[0][:] - nuisance)


#**************************************************************************************************#
#                                        Class NumpyModel                                          #
#**************************************************************************************************#
#                                                                                                  #
# A model with no framework at all: a linear map in NumPy, to show the interface needs none.       #
#                                                                                                  #
#**************************************************************************************************#
class NumpyModel(Model):
    """A model with no framework at all: a linear map in NumPy, to show the interface needs none."""

    def __init__(self, metabolites, points):
        self.metabolites = metabolites
        self.weights = np.random.default_rng(1).normal(size=(2 * points, len(metabolites)))

    def prepare(self, items):
        from mrsuite.models.base import spectra
        return spectra(items).reshape(len(items), -1)

    def predict(self, x):
        return x @ self.weights

    def interpret(self, out, items):
        return [FitResult(method="numpy", metabolites=self.metabolites,
                          concentrations={"raw": dict(zip(self.metabolites, row.tolist()))})
                for row in out]

    def fingerprint(self):
        return hashlib.sha256(self.weights.tobytes()).hexdigest()

    def _save_pretrained(self, folder: Path):
        np.save(folder / "weights.npy", self.weights)

    @classmethod
    def _from_pretrained(cls, *, model_id, **kwargs):
        model = cls(**{k: kwargs[k] for k in ("metabolites", "points")})
        model.weights = np.load(Path(model_id) / "weights.npy")
        return model


#***************#
#   framework   #
#***************#
def test_any_framework_fits_the_interface(tmp_path):
    """No torch anywhere: the base does the batching, the provenance and the Hub."""
    model = NumpyModel(["A", "B"], 1024)
    model.save_pretrained(tmp_path)
    loaded = NumpyModel.from_pretrained(tmp_path)
    results = Pipeline([loaded]).run(_spectra(3))

    assert [r.metabolites for r in results] == [["A", "B"]] * 3
    assert json.loads(results[0].provenance[-1]["Details"])["weights"] == model.fingerprint()
    # mrsuite.models.load takes MRSuite's own classes only, so the card uses the class
    assert f'NumpyModel.from_pretrained("{tmp_path}")' in (tmp_path / "README.md").read_text()
