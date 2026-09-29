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

from nifti_mrs.create_nmrs import gen_nifti_mrs

from nifti_mrs_plus import NIfTI_MRS_Plus
from mrsuite.core.pipeline import Pipeline
from mrsuite.core.result import FitResult
from mrsuite.datasets import challenge2016
from mrsuite.models import MLP, WAND, Model

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
    assert loaded.source["name"] == str(tmp_path)
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
