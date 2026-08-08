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
