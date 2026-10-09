####################################################################################################
#                                        test_pipeline.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-08-08                                                                              #
#                                                                                                  #
# Purpose: Pipelines. One value in, one out, with any callable as a step, so the same class       #
#          carries spectra, a basis set or fit results. The rule under test throughout is          #
#          that what a step returns is what flows on, and a tuple starting with the same           #
#          kind of thing is a measurement reported alongside it.                                   #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import numpy as np
import pytest

from nifti_mrs_plus import NIfTI_MRS_Plus
from mrsuite import fsl_mrs
from mrsuite.core.pipeline import Pipeline, get_result, results


#**********#
#   data   #
#**********#
@pytest.fixture
def data(spectrum):
    """A fresh two-item batch per test, since steps attach results to it."""
    return NIfTI_MRS_Plus([spectrum.copy(), spectrum.copy()])


#****************#
#   processing   #
#****************#
def test_a_step_returning_spectra_advances_the_data(data):
    out = Pipeline([fsl_mrs.conjugate]).run(data)

    assert isinstance(out, NIfTI_MRS_Plus)
    assert np.allclose(np.asarray(out[0][:]).imag, -np.asarray(data[0][:]).imag)
    assert results(out) == {}, "processing attaches nothing"


#****************#
#   what flows   #
#****************#
def test_what_a_step_returns_is_what_flows_on(data):
    """One rule, and it is what lets a pipeline carry something other than spectra:
    a fit turns spectra into results, and the steps after it see results."""
    out = Pipeline([lambda d: {"snr": 12.3}]).run(data)

    assert out == {"snr": 12.3}


#***********#
#   extra   #
#***********#
def test_extra_return_values_are_kept_not_discarded(data):
    """Toolbox functions often return (spectra, metrics), as op_rmbadaverages does.

    Both halves matter: the spectra advance the pipeline and the metrics are the reason
    the step was run at all, so neither may be dropped.
    """
    out = Pipeline([lambda d: (fsl_mrs.conjugate(d), (0, 3, 7))]).run(data)

    assert isinstance(out, NIfTI_MRS_Plus)
    assert get_result(out, "step0") == (0, 3, 7)


#**************#
#   per item   #
#**************#
def test_a_list_as_long_as_the_batch_is_filed_per_spectrum(data):
    """How the bridges return a batched non-MRS output: one entry per spectrum."""
    out = Pipeline([lambda d: (d, [{"snr": 10.0}, {"snr": 20.0}])]).run(data)

    assert get_result(out, "step0", index=0) == {"snr": 10.0}
    assert get_result(out, "step0", index=1) == {"snr": 20.0}
    assert get_result(out, "step0") is None, "not also filed at batch level"


#*************#
#   a tuple   #
#*************#
def test_a_tuple_of_the_same_length_is_not_split(data):
    """List against tuple, not length: a step reporting (mean, std) over a two-item
    batch must not be mistaken for one result per spectrum."""
    out = Pipeline([lambda d: (d, (1.5, 0.2))]).run(data)

    assert get_result(out, "step0") == (1.5, 0.2)
    assert get_result(out, "step0", index=0) is None


#***********#
#   carry   #
#***********#
def test_results_survive_a_later_processing_step(data):
    """A step builds a new NIfTI-MRS+, which starts with empty metadata.

    Without carrying them across, the step after a measuring one silently drops what it
    measured, which is the most damaging way this could fail.
    """
    out = Pipeline([lambda d: (d, {"snr": 12.3}), fsl_mrs.conjugate]).run(data)

    assert out is not data, "conjugate should have produced a new object"
    assert get_result(out, "step0") == {"snr": 12.3}


#***********#
#   water   #
#***********#
def test_water_is_processed_by_its_own_pipeline(spectrum):
    """No water argument and no apply_to_water flag: a reference is just data.

    Binding the reference into a later step then uses the processed one, which is what
    ecc needs, and it is correct by construction rather than by a placeholder.
    """
    water = Pipeline([fsl_mrs.conjugate]).run(NIfTI_MRS_Plus([spectrum.copy()]))
    seen = []

    Pipeline([lambda d: (seen.append(water) or d)]).run(NIfTI_MRS_Plus([spectrum.copy()]))

    assert np.allclose(np.asarray(seen[0][0][:]).imag, -np.asarray(spectrum[:]).imag)


#**********#
#   name   #
#**********#
def test_a_named_function_is_filed_under_its_name(data):
    def snr(d):
        return d, 12.3

    out = Pipeline([snr]).run(data)

    assert get_result(out, "snr") == 12.3


#****************#
#   step names   #
#****************#
def test_a_bridge_step_keeps_the_toolbox_function_name(data):
    """bridge.step.f() names itself, so its metrics are not filed under a position."""
    out = Pipeline([fsl_mrs.step.apodize((10.0,))]).run(data)

    assert repr(Pipeline([fsl_mrs.step.apodize((10.0,))])) == "Pipeline(apodize)"
    assert isinstance(out, NIfTI_MRS_Plus)


#******************#
#   not callable   #
#******************#
def test_a_non_callable_step_fails_at_build_time():
    """Not part-way through a long run."""
    with pytest.raises(TypeError, match="step 1 is 'coilcombine'"):
        Pipeline([fsl_mrs.conjugate, "coilcombine"])


#***********#
#   order   #
#***********#
def test_a_pipeline_runs_its_steps_in_order(data):
    """Two conjugations return the original, so order and arity are both exercised."""
    original = np.asarray(data[0][:]).copy()
    out = Pipeline([fsl_mrs.conjugate, fsl_mrs.conjugate]).run(data)

    assert np.allclose(np.asarray(out[0][:]), original)


#*************#
#   one obj   #
#*************#
def test_processing_and_measurement_leave_on_one_object(data):
    """What comes out carries the spectra and everything measured on the way."""
    original = np.asarray(data[0][:]).copy()

    def snr(d):
        return d, 12.3

    out = Pipeline([fsl_mrs.conjugate, snr,
                    lambda d: (fsl_mrs.conjugate(d), 0.87)]).run(data)

    assert sorted(results(out, )) == ["snr", "step2"]
    assert get_result(out, "snr") == 12.3
    assert get_result(out, "step2") == 0.87
    assert np.allclose(np.asarray(out[0][:]), original)


#*************#
#   nesting   #
#*************#
def test_a_pipeline_is_itself_a_step(data):
    """What lets processing and analysis be assembled into one run."""
    process = Pipeline([fsl_mrs.conjugate])
    analyse = Pipeline([lambda d: {"NAA": 12.3}])

    assert Pipeline([process, analyse]).run(data) == {"NAA": 12.3}


#**************#
#   currency   #
#**************#
def test_a_pipeline_can_carry_something_other_than_spectra():
    """Basis preparation and result post-processing use the same class: whatever a step
    returns is what the next one gets."""
    out = Pipeline([lambda names: names + ["MM09"],
                    lambda names: sorted(names)]).run(["NAA", "Cr"])

    assert out == ["Cr", "MM09", "NAA"]


#**********#
#   repr   #
#**********#
def test_repr_names_the_steps():
    assert repr(Pipeline([fsl_mrs.conjugate, lambda d: d])) == "Pipeline(conjugate -> step1)"


#************#
#   stages   #
#************#
def test_stages_hand_out_every_step_and_end_where_run_does(data):
    pipeline = Pipeline([fsl_mrs.step.apodize(amount=[5.0]), fsl_mrs.step.fshift(amount=2.0)])

    stages = list(pipeline.stages(data))

    assert len(stages) == 2
    assert np.array_equal(stages[-1][0][:], pipeline.run(data)[0][:])
    assert not np.array_equal(stages[0][0][:], stages[1][0][:])


#*****************#
#   written out   #
#*****************#
@pytest.fixture
def written(spectrum):
    """A pipeline of every kind of argument: a list, a number, a tuple, a path, a nested
    pipeline, and spectra, named."""
    from pathlib import Path
    from mrsuite.bridges import lcmodel

    water = spectrum.copy()
    return water, Pipeline([fsl_mrs.step.apodize(amount=[5.0]),
                            Pipeline([fsl_mrs.step.fshift(amount=1.5)]),
                            fsl_mrs.step.ecc(water),
                            lcmodel.step.fit(Path("x.BASIS"), ppmlim=(0.5, 4.2))])


def same(a: Pipeline, b: Pipeline) -> bool:
    """The same steps, each method by its module and name, with the same arguments, nested
    pipelines alike."""
    def key(step):
        return step.method.__module__, step.method.__qualname__, step.args, step.kwargs

    return len(a.steps) == len(b.steps) and all(
        same(x, y) if isinstance(x, Pipeline) else not isinstance(y, Pipeline) and key(x) == key(y)
        for x, y in zip(a.steps, b.steps))


def test_a_pipeline_written_as_code_builds_the_same_pipeline(written):
    water, pipeline = written
    code = pipeline.script(inputs={"water": water})
    namespace = {}
    exec(code, namespace)

    again = namespace["pipeline"](water)

    assert "def pipeline(water):" in code and "fsl_mrs.step.ecc(water)" in code
    assert same(pipeline, again)
    assert again.steps[2].args[0] is water and again.steps[3].kwargs["ppmlim"] == (0.5, 4.2)


def test_a_pipeline_written_as_a_recipe_builds_the_same_pipeline(written):
    import json

    water, pipeline = written
    recipe = json.loads(json.dumps(pipeline.to_dict(inputs={"water": water})))

    again = Pipeline.from_dict(recipe, water=water)

    assert recipe["inputs"] == ["water"]
    assert same(pipeline, again)
    assert again.steps[2].args[0] is water and again.steps[3].kwargs["ppmlim"] == (0.5, 4.2)


def test_a_rebuilt_pipeline_gives_the_same_spectra(written, data):
    water, pipeline = written
    namespace = {}
    exec(pipeline.script(inputs={"water": water}), namespace)
    first = Pipeline(pipeline.steps[:3])
    again = Pipeline(namespace["pipeline"](water).steps[:3])

    assert np.array_equal(first.run(data.list()[0])[:], again.run(data.list()[0])[:])


#*************#
#   refused   #
#*************#
def test_what_cannot_be_written_down_is_refused_by_name(written):
    water, pipeline = written

    with pytest.raises(ValueError, match="does not name"):
        pipeline.script()
    with pytest.raises(ValueError, match="step 0.*cannot be written down"):
        Pipeline([lambda d: d]).to_dict()
    with pytest.raises(ValueError, match="cannot be written down; give arguments"):
        Pipeline([fsl_mrs.step.apodize(amount=object())]).script()
    with pytest.raises(ValueError, match="takes \\['water'\\]"):
        Pipeline.from_dict(pipeline.to_dict(inputs={"water": water}))


#***************#
#   as it was   #
#***************#
def test_a_bridge_function_as_a_step_is_written_as_itself(data):
    code = Pipeline([fsl_mrs.conjugate]).script()
    namespace = {}
    exec(code, namespace)

    assert "        fsl_mrs.conjugate," in code
    assert np.array_equal(namespace["pipeline"]().run(data)[0][:],
                          Pipeline([fsl_mrs.conjugate]).run(data)[0][:])


def test_two_functions_of_one_name_stay_apart(tmp_path, monkeypatch):
    for module in ("first_lab", "second_lab"):
        (tmp_path / f"{module}.py").write_text(
            "from mrsuite import method\n\n"
            f"@method('{module}')\ndef clean(spectrum):\n    return spectrum\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    import first_lab
    import second_lab

    namespace = {}
    exec(Pipeline([first_lab.clean.step(), second_lab.clean.step()]).script(), namespace)

    assert [s.method.program for s in namespace["pipeline"]().steps] == ["first_lab", "second_lab"]


def test_arguments_python_cannot_spell_plainly_come_back(spectrum):
    code = Pipeline([fsl_mrs.step.fshift(amount=float("nan"), **{"na.rm": True})]).script()
    namespace = {}
    exec(code, namespace)

    step = namespace["pipeline"]().steps[0]
    assert np.isnan(step.kwargs["amount"]) and step.kwargs["na.rm"] is True


def test_what_would_come_back_otherwise_is_refused(spectrum, monkeypatch):
    from mrsuite.bridges.fid_a import FIDABridge
    from mrsuite.core.method import Method

    model = Method()
    model.source = {"name": "2a92d3638784ad7302c42c5ecc67276d9aa01660", "revision": None}
    monkeypatch.setattr(FIDABridge, "exists", lambda self, func: True)
    own = FIDABridge(runtime="matlab")

    def written(*steps, **inputs):
        return Pipeline(list(steps)).script(inputs=inputs)

    with pytest.raises(ValueError, match="loaded from a folder"):
        written(model.step())
    with pytest.raises(ValueError, match="settings of its own"):
        written(own.op_addphase.step(10))
    with pytest.raises(ValueError, match="would not be Python names"):
        written(fsl_mrs.step.ecc(spectrum), fsl_mrs=spectrum)
    with pytest.raises(ValueError, match="cannot be written down; give"):
        written(fsl_mrs.step.fshift(amount=1 + 2j))
    with pytest.raises(ValueError, match="starting with \\$"):
        written(fsl_mrs.step.fshift(amount={"$path": "x"}))


def test_a_function_of_the_running_script_is_refused(monkeypatch):
    """Found again by name now, but in whatever script runs when the recipe is read."""
    import sys
    from mrsuite import method

    @method("here")
    def clean(spectrum):
        return spectrum
    clean.__module__, clean.__qualname__ = "__main__", "clean"
    monkeypatch.setattr(sys.modules["__main__"], "clean", clean, raising=False)

    with pytest.raises(ValueError, match="the script that is running"):
        Pipeline([clean.step()]).to_dict()


#############
#   basis   #
#############
def test_a_basis_is_named_like_spectra_and_never_printed_whole():
    """A fit's basis goes into a recipe and a script by its name, as spectra do; refused or
    shown, a basis of thousands of points stays one line."""
    from mrsuite.core.basis import Basis

    basis = Basis(["NAA", "Cr"], np.ones((2, 4096), complex), 4000.0, 123.2)
    pipeline = Pipeline([fsl_mrs.step.fit(basis)])

    recipe = pipeline.to_dict(inputs={"basis": basis})
    again = Pipeline.from_dict(recipe, basis=basis)

    assert again.steps[0].args[0] is basis
    assert "def pipeline(basis):" in pipeline.script(inputs={"basis": basis})
    assert len(repr(pipeline.steps[0])) < 100
    with pytest.raises(ValueError, match="a basis the pipeline does not name") as err:
        pipeline.to_dict()
    assert len(str(err.value)) < 200


def test_an_argument_named_like_a_python_keyword_is_written_as_code_that_runs():
    """spant's ABfit takes lambda."""
    from mrsuite.core.method import Step

    code = Pipeline([Step(fsl_mrs.fit, (), {"lambda": 1, "ppmlim": (1, 2)})]).script()
    namespace = {}
    exec(compile(code, "<script>", "exec"), namespace)

    assert namespace["pipeline"]().steps[0].kwargs == {"lambda": 1, "ppmlim": (1, 2)}
