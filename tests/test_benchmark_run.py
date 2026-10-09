####################################################################################################
#                                      test_benchmark_run.py                                       #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-02                                                                              #
#                                                                                                  #
# Purpose: The benchmark scripts (scripts/benchmark) on a few spectra: a run saves chunk by        #
#          chunk and records its environment, resumes after a crash with only what is missing,     #
#          keeps going past what fails, scores end to end, and compares to the bit.                #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import copy
import json
import math
import os
import shutil
import sys
from dataclasses import replace
from functools import partial
from pathlib import Path
from unittest import mock

import pytest

import mrsuite
from nifti_mrs_plus import NIfTI_MRS_Plus
from mrsuite.benchmark import mae
from mrsuite.core.result import FitResult
from mrsuite.datasets import challenge2016, mrsi_challenge

from tests.bridges.test_tarquin import needs_tarquin

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "benchmark"))
with mock.patch.dict(os.environ):                 # run.py sets OMP_NUM_THREADS for its process
    import compare  # noqa: E402
    import config  # noqa: E402
    import run  # noqa: E402
    import score  # noqa: E402

needs_2016 = pytest.mark.skipif(not challenge2016.BASIS.exists(),
                                reason="challenge submodule not checked out")
needs_testsub10 = pytest.mark.skipif(
    not (mrsi_challenge.DATA_DIR / "TestSub10" / "TestSub10_all.mat").exists(),
    reason="MRSI challenge TestSub10 not downloaded")

# what the toy method does: raise on its n-th call, or for these spectra
CALLS, CRASH, BROKEN = [], {"at": None}, set()


#*********#
#   toy   #
#*********#
@mrsuite.method(program="test")
def toy(spectrum, basis):
    """A quick, deterministic fit: 'NAA' is the spectrum's first point."""
    value = float(abs(spectrum[:].ravel()[0]))
    CALLS.append(value)
    if len(CALLS) == CRASH["at"]:
        raise KeyboardInterrupt                   # as a killed process stops: not caught
    if value in BROKEN:
        raise ValueError("the toy cannot fit this one")
    return FitResult("toy", ["NAA", "Cr", "PCr"], {"raw": {"NAA": value, "Cr": 1.0, "PCr": 1.0}})


#*************#
#   toy run   #
#*************#
def _toy_run(folder, spectra=3, chunk=1, jobs=None, methods=None):
    """The toy on the first few 2016 spectra, a chunk per spectrum by default. Without the
    environment in the manifest, which takes seconds and is tested on the real run."""
    with mock.patch.object(run, "environment", lambda methods: {}):
        return run.run(folder, jobs or [config.Job("challenge2016", "toy")],
                       {"challenge2016": partial(config.challenge, spectra)},
                       methods or {"toy": config.Fit(toy, chunk=chunk)})


#***************#
#   reset toy   #
#***************#
@pytest.fixture(autouse=True)
def _reset_toy():
    CALLS.clear()
    CRASH["at"] = None
    BROKEN.clear()


#************#
#   resume   #
#************#
@needs_2016
def test_a_crash_loses_one_chunk_and_the_rerun_fits_only_the_rest(tmp_path):
    CRASH["at"] = 2
    with pytest.raises(KeyboardInterrupt):
        _toy_run(tmp_path / "crashed")
    job = tmp_path / "crashed" / "challenge2016" / "toy"
    assert sorted(p.name for p in job.glob("0*")) == ["00000.json", "00000.json.pkl",
                                                      "00000.meta.json"]
    saved = (job / "00000.json").stat().st_mtime_ns

    CALLS.clear()
    CRASH["at"] = None
    assert _toy_run(tmp_path / "crashed") == {}
    assert len(CALLS) == 2, "only the spectra without a result are fitted again"
    assert (job / "00000.json").stat().st_mtime_ns == saved

    sessions = json.loads((tmp_path / "crashed" / "manifest.json").read_text())["sessions"]
    assert "end" not in sessions[0] and sessions[1]["end"]

    _toy_run(tmp_path / "whole")
    table = compare.compare(tmp_path / "crashed", tmp_path / "whole")
    assert table.identical.all() and table.same_provenance.all() and table.both[0] == 3


#*************#
#   failing   #
#*************#
@needs_2016
def test_a_failed_spectrum_is_recorded_scored_as_omitted_and_retried(tmp_path):
    data, _, _ = challenge2016.load(3)
    BROKEN.add(float(abs(data.list()[1][:].ravel()[0])))

    assert _toy_run(tmp_path, chunk=3) == {}, "a spectrum failing does not fail the job"
    results, errors = run.read(tmp_path / "challenge2016" / "toy")
    assert sorted(results) == ["dataset1", "dataset3"]
    assert errors == {"dataset2": "ValueError: the toy cannot fit this one"}

    table = score.score_run(tmp_path, {"challenge2016": ("optimal",)}, {"mae": mae})
    pooled = table["challenge2016"].query("metabolite == 'all'").iloc[0]
    assert (pooled.n_used, pooled.n_omitted) == (6, 3)

    CALLS.clear()
    BROKEN.clear()
    _toy_run(tmp_path, chunk=3)
    assert len(CALLS) == 1 and run.read(tmp_path / "challenge2016" / "toy")[1] == {}


#*************#
#   changed   #
#*************#
@needs_2016
def test_a_job_that_changed_is_not_resumed_into_and_the_others_go_on(tmp_path):
    _toy_run(tmp_path)
    failed = _toy_run(tmp_path, jobs=[config.Job("challenge2016", "toy"),
                                      config.Job("challenge2016", "other")],
                      methods={"toy": config.Fit(toy, options={"x": 1}), "other": config.Fit(toy)})

    assert list(failed) == ["challenge2016/toy"] and "differs" in failed["challenge2016/toy"]
    assert len(run.read(tmp_path / "challenge2016" / "other")[0]) == 3
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert list(manifest["sessions"][1]["failed"]) == ["challenge2016/toy"]


#*************#
#   removed   #
#*************#
@needs_2016
def test_a_removed_chunk_is_fitted_again_into_a_new_one(tmp_path):
    """Numbered after the last, not by count: a gap must not overwrite another chunk."""
    _toy_run(tmp_path)
    job = tmp_path / "challenge2016" / "toy"
    for path in job.glob("00000.*"):
        path.unlink()

    _toy_run(tmp_path)

    assert sorted(run.read(job)[0]) == ["dataset1", "dataset2", "dataset3"]
    assert [p.name for p in run._chunks(job)] == ["00001.json", "00002.json", "00003.json"]


#**************#
#   all fail   #
#**************#
@needs_2016
def test_a_job_where_every_spectrum_fails_fails_and_scoring_goes_on(tmp_path):
    """Misconfigured rather than unlucky; the methods that fitted are still scored."""
    data, _, _ = challenge2016.load(3)
    BROKEN.update(float(abs(item[:].ravel()[0])) for item in data.list())

    failed = _toy_run(tmp_path, jobs=[config.Job("challenge2016", "toy")])
    assert list(failed) == ["challenge2016/toy"] and "every spectrum failed" in failed[
        "challenge2016/toy"]

    BROKEN.clear()
    _toy_run(tmp_path, jobs=[config.Job("challenge2016", "other")],
             methods={"other": config.Fit(toy)})
    table = score.score_run(tmp_path, {"challenge2016": ("optimal",)}, {"mae": mae})
    assert set(table["challenge2016"].method) == {"other"}


#*******************#
#   dataset fails   #
#*******************#
@needs_2016
def test_a_dataset_that_fails_to_load_fails_only_its_jobs(tmp_path):
    def broken():
        raise FileNotFoundError("no such subject")

    with mock.patch.object(run, "environment", lambda methods: {}):
        failed = run.run(tmp_path, [config.Job("broken", "toy"), config.Job("challenge2016", "toy")],
                         {"broken": broken, "challenge2016": partial(config.challenge, 3)},
                         {"toy": config.Fit(toy)})

    assert list(failed) == ["broken/toy"] and "no such subject" in failed["broken/toy"]
    assert len(run.read(tmp_path / "challenge2016" / "toy")[0]) == 3


#*******************#
#   changed input   #
#*******************#
def _scaled(factor):
    """A step of one name whatever it does, as a model behind an unchanged Hub name."""
    def step(data):
        items = [copy.deepcopy(item) for item in data.list()]
        for item in items:
            item[:] = item[:] * factor
        return NIfTI_MRS_Plus(items)
    return step


@needs_2016
def test_changed_water_or_step_output_is_not_resumed_into(tmp_path):
    _toy_run(tmp_path / "step", methods={"toy": config.Fit(toy, before=(_scaled(1.0),))})
    failed = _toy_run(tmp_path / "step", methods={"toy": config.Fit(toy, before=(_scaled(10.0),))})
    assert "differs" in failed["challenge2016/toy"]

    def doubled_water(n):
        dataset = config.challenge(n)
        return replace(dataset, water=_scaled(2.0)(dataset.water))

    _toy_run(tmp_path / "water")
    with mock.patch.object(run, "environment", lambda methods: {}):
        failed = run.run(tmp_path / "water", [config.Job("challenge2016", "toy")],
                         {"challenge2016": partial(doubled_water, 3)}, {"toy": config.Fit(toy)})
    assert "differs" in failed["challenge2016/toy"]


#******************#
#   more spectra   #
#******************#
@needs_2016
def test_a_subsample_larger_than_the_dataset_fits_each_spectrum_once(tmp_path):
    _toy_run(tmp_path, jobs=[config.Job("challenge2016", "toy", spectra=5)])
    assert len(CALLS) == 3


#*****************#
#   json folder   #
#*****************#
@needs_2016
def test_a_run_folder_named_like_json_resumes(tmp_path):
    _toy_run(tmp_path / "golden.json.d")
    assert _toy_run(tmp_path / "golden.json.d") == {}


#*************#
#   secrets   #
#*************#
def test_the_environment_keeps_no_secret_and_no_machine_name(tmp_path):
    with mock.patch.dict(os.environ, {"HF_TOKEN": "hf_secret", "HF_HOME": "/somewhere"}):
        recorded = run.environment({})
    assert "HF_TOKEN" not in recorded["variables"] and "HF_HOME" in recorded["variables"]
    assert "node" not in recorded["machine"]
    assert recorded["git"]["MRSuite"]["commit"]
    assert run._git(tmp_path) is None                      # no checkout


#************#
#   config   #
#************#
def test_every_stage_names_what_the_config_defines():
    for jobs in config.STAGES.values():
        for job in jobs:
            assert job.dataset in config.DATASETS and job.method in config.METHODS
            assert job.dataset in config.REFERENCES
    assert {j.method for j in config.STAGES["2016"]} == set(config.METHODS)
    assert config.METHODS["wand-tarquin-water"].before == (config.wand,)
    assert config.METHODS["tarquin-water"].water == challenge2016.WATER["tarquin"]


#**************#
#   real run   #
#**************#
@pytest.fixture(scope="module")
def real(tmp_path_factory):
    """TARQUIN with water and FSL-MRS on two 2016 spectra, FSL-MRS on two MRSI voxels."""
    folder = tmp_path_factory.mktemp("run")
    datasets = {"challenge2016": partial(config.challenge, 2),
                "TestSub10": partial(config.mrsi, "TestSub10", voxels=2)}
    methods = {name: replace(config.METHODS[name], chunk=1)
               for name in ("tarquin-water", "fsl_mrs")}
    jobs = [config.Job("challenge2016", "tarquin-water"), config.Job("challenge2016", "fsl_mrs"),
            config.Job("TestSub10", "fsl_mrs")]
    go = partial(run.run, jobs=jobs, datasets=datasets, methods=methods,
                 config=str(run.HERE / "config.py"), stage="test")
    assert go(folder) == {}
    return folder, go


#***********#
#   saved   #
#***********#
@needs_2016
@needs_testsub10
@needs_tarquin
def test_a_run_saves_each_chunk_and_records_its_environment(real):
    folder, _ = real
    for job in ("challenge2016/tarquin-water", "challenge2016/fsl_mrs", "TestSub10/fsl_mrs"):
        metas = sorted((folder / job).glob("*.meta.json"))
        assert len(metas) == 2 and all(json.loads(m.read_text())["end"] for m in metas)
        results, errors = run.read(folder / job)
        assert len(results) == 2 and not errors

    session = json.loads((folder / "manifest.json").read_text())["sessions"][0]
    env = session["environment"]
    assert len(env["git"]["MRSuite"]["commit"]) == 40 and "externals/osprey" in env["git"]
    assert set(run.SIBLINGS) <= set(env["git"])
    assert env["methods"]["mrsuite.bridges.tarquin.fit"]["binary"]["sha256"]
    assert any(p.startswith("fsl_mrs==") for p in env["versions"]) and env["pip_freeze"]
    assert "def challenge" in session["config"]["text"] and env["machine"]["cpus"]


#**************#
#   finished   #
#**************#
@needs_2016
@needs_testsub10
@needs_tarquin
def test_a_finished_run_resumes_to_nothing(real):
    folder, go = real
    def chunks():
        return sorted(str(p.relative_to(folder)) for p in folder.glob("*/*/0*.json"))
    before = chunks()
    assert go(folder) == {}
    assert chunks() == before and len(before) == 12
    assert len(json.loads((folder / "manifest.json").read_text())["sessions"]) == 2


#************#
#   scored   #
#************#
@needs_2016
@needs_testsub10
@needs_tarquin
def test_a_run_scores_end_to_end(real):
    folder, _ = real
    tables = score.score_run(folder, config.REFERENCES, config.METRICS)

    pooled = tables["challenge2016"].query("metabolite == 'all'")
    on = {r: set(pooled[pooled.reference == r].method) for r in ("water", "tCr", "optimal")}
    assert on == {"water": {"tarquin-water"}, "tCr": {"fsl_mrs"}, "optimal": {"fsl_mrs"}}
    assert set(tables["TestSub10"].reference) == {"tCr", "optimal", "none"}
    assert (folder / "scores" / "challenge2016.csv").exists()
    assert pooled.query("metric == 'mae'").value.notna().all()


#**************#
#   compared   #
#**************#
@needs_2016
@needs_testsub10
@needs_tarquin
def test_identical_runs_compare_identical_and_one_ulp_shows(real, tmp_path):
    folder, go = real
    assert go(tmp_path / "again") == {}
    table = compare.compare(folder, tmp_path / "again")
    assert table.identical.all() and table.same_provenance.all() and len(table) == 3

    shutil.copytree(folder, tmp_path / "nudged")
    path = tmp_path / "nudged" / "challenge2016" / "fsl_mrs" / "00000.json"
    values = json.loads(path.read_text())
    values[0]["concentrations"]["raw"]["NAA"] = math.nextafter(
        values[0]["concentrations"]["raw"]["NAA"], math.inf)
    path.write_text(json.dumps(values))

    row = compare.compare(folder, tmp_path / "nudged").set_index("job").loc[
        "challenge2016/fsl_mrs"]
    assert (row.differ, row.max_ulp, row.identical) == (1, 1, False)
    assert 0 < row.max_relative < 1e-15
