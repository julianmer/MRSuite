####################################################################################################
#                                          test_method.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-28                                                                              #
#                                                                                                  #
# Purpose: The base every method stands on, through plain functions made methods: batches          #
#          mirrored, further spectra lined up, headers kept and every call recorded - and the      #
#          conformance check that tells a method keeping the contract from one that does not.      #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import json

import numpy as np
import pytest

from nifti_mrs_plus import NIfTI_MRS_Plus
from mrsuite import method
from mrsuite.core.pipeline import Pipeline
from mrsuite.core.result import FitResult
from mrsuite.testing import check


#***********#
#   scale   #
#***********#
@method("test", version="1.0")
def scale(spectrum, factor=2.0):
    """A plain function, one spectrum in and one out."""
    out = spectrum.copy()
    out[:] = spectrum[:] * factor
    return out


#**********#
#   area   #
#**********#
@method("test", version="1.0")
def area(spectrum, water=None):
    """A plain function that measures, with a reference beside the data."""
    total = float(np.abs(spectrum[:]).sum())
    ref = float(np.abs(water[:]).sum()) if water is not None else 1.0
    return FitResult("test:area", metabolites=["all"], concentrations={"raw": {"all": total / ref}})


#***********#
#   batch   #
#***********#
def _batch(spectrum, n=3):
    """Spectra that differ, so an order that is not kept shows."""
    items = []
    for i in range(n):
        item = spectrum.copy()
        item[:] = spectrum[:] * (i + 1)
        items.append(item)
    return NIfTI_MRS_Plus(items)


#*************#
#   mirrors   #
#*************#
def test_one_spectrum_gives_one_and_a_batch_a_batch(spectrum):
    data = _batch(spectrum)

    one, many = scale(spectrum), scale(data)

    assert one.__class__.__name__ == "NIFTI_MRS"
    assert isinstance(many, NIfTI_MRS_Plus) and len(many) == 3
    for i, item in enumerate(many.list()):
        assert np.allclose(item[:], 2 * (i + 1) * spectrum[:])


#************#
#   record   #
#************#
def test_every_call_is_recorded_with_its_arguments(spectrum):
    out = scale(spectrum, factor=3.0)

    step = out.hdr_ext["ProcessingApplied"][-1]
    details = json.loads(step["Details"])
    assert (step["Program"], step["Version"], step["Method"]) == ("test", "1.0", "scale")
    assert details["kwargs"] == {"factor": 3.0} and details["mrsuite"]


#************#
#   header   #
#************#
def test_the_header_and_the_history_are_kept(spectrum):
    source = spectrum.copy()
    source.add_hdr_field("EchoTime", 0.03)

    out = scale(scale(source))

    assert out.hdr_ext["EchoTime"] == 0.03
    assert [s["Method"] for s in out.hdr_ext["ProcessingApplied"]] == ["scale", "scale"]


def test_a_user_defined_field_is_kept_with_its_description(spectrum):
    """As Twix read by spec2nii carries them, onto a step's output that a toolbox built from
    the samples alone: nifti_mrs takes such a field only with its description."""
    from nifti_mrs.create_nmrs import gen_nifti_mrs

    @method("test", version="1.0")
    def rebuilt(spectrum):
        return gen_nifti_mrs(np.asarray(spectrum[:]), spectrum.dwelltime,
                             spectrum.spectrometer_frequency[0])

    source = spectrum.copy()
    source.add_hdr_field("WaterSuppression", "VAPOR", doc="the scheme")

    out = rebuilt(source)

    assert out.hdr_ext["WaterSuppression"] == {"Value": "VAPOR", "Description": "the scheme"}


#*********#
#   fit   #
#*********#
def test_a_fit_result_carries_the_history_and_the_call(spectrum):
    source = scale(spectrum)

    out = area(source)

    assert [s["Method"] for s in out.provenance] == ["scale", "area"]


#************#
#   paired   #
#************#
def test_further_spectra_go_one_per_spectrum(spectrum):
    data = _batch(spectrum)

    out = area(data, water=data)

    assert [r.conc()["all"] for r in out] == pytest.approx([1.0, 1.0, 1.0])


#************#
#   spread   #
#************#
def test_one_reference_is_not_spread_over_a_batch(spectrum):
    data = _batch(spectrum)

    with pytest.raises(ValueError, match="water gives 1 for 3 spectra"):
        area(data, water=spectrum)
    with pytest.raises(ValueError, match="water gives 2 for 3 spectra"):
        area(data, water=_batch(spectrum, 2))


#*************#
#   batched   #
#*************#
def test_a_batched_function_gets_the_whole_batch(spectrum):
    seen = []

    @method("test", version="1.0", batched=True)
    def count(batch):
        seen.append(len(batch))
        return [FitResult("test", metabolites=["n"], concentrations={"raw": {"n": 1.0}})
                for _ in batch.list()]

    assert len(count(_batch(spectrum))) == 3 and isinstance(count(spectrum), FitResult)
    assert seen == [3, 1]


#***********#
#   plain   #
#***********#
def test_anything_but_spectra_is_a_plain_call():
    @method("test")
    def load(path):
        return f"loaded {path}"

    assert load("data.nii.gz") == "loaded data.nii.gz"


#**********#
#   step   #
#**********#
def test_a_step_binds_the_arguments_and_waits_for_the_data(spectrum):
    step = scale.step(factor=4.0)

    out = Pipeline([step]).run(spectrum)

    assert step.__name__ == "scale" and np.allclose(out[:], 4 * spectrum[:])


#***********#
#   check   #
#***********#
def test_the_check_passes_a_method_that_keeps_the_contract(spectrum):
    data = _batch(spectrum)

    check(scale, data, factor=3.0)
    check(area, data, water=data)
    check(area, spectrum, water=spectrum)


#************#
#   broken   #
#************#
def test_the_check_names_the_rule_a_method_breaks(spectrum):
    @method("test", version="1.0")
    def recorded_twice(item):
        return scale(item, factor=1.0)

    @method("test", version="1.0")
    def touches_its_input(item):
        item.add_hdr_field("EchoTime", 0.03)
        return item.copy()

    with pytest.raises(AssertionError, match="gained 0 ProcessingApplied entries"):
        check(lambda data: data, _batch(spectrum))           # a plain function records nothing
    with pytest.raises(AssertionError, match="gained 2 ProcessingApplied entries"):
        check(recorded_twice, _batch(spectrum))
    with pytest.raises(AssertionError, match="input was changed"):
        check(touches_its_input, _batch(spectrum))


#*************#
#   unknown   #
#*************#
def test_the_check_wants_a_version(spectrum):
    with pytest.raises(AssertionError, match="does not tell its version"):
        check(method("test")(scale.__wrapped__), _batch(spectrum))


#*****************#
#   a step says   #
#*****************#
def test_a_step_says_what_it_calls_and_with_what(spectrum):
    step = scale.step(factor=3.0)

    assert step.method is scale and step.args == () and step.kwargs == {"factor": 3.0}
    assert repr(step) == "scale.step(factor=3.0)" and step.__name__ == "scale"
    assert np.array_equal(step(spectrum)[:], scale(spectrum, factor=3.0)[:])


#*******************#
#   your own step   #
#*******************#
def test_your_own_method_is_written_out_by_its_module_and_name(spectrum):
    code = Pipeline([scale.step(factor=3.0)]).script()
    namespace = {}
    exec(code, namespace)

    assert "import tests.core.test_method" in code
    assert "tests.core.test_method.scale.step(factor=3.0)" in code
    assert namespace["pipeline"]().steps[0].kwargs == {"factor": 3.0}


#************#
#   models   #
#************#
def test_a_model_is_written_out_as_loaded_by_name():
    """A trained model is reached again the way it was loaded, by its name on the Hub."""
    from mrsuite.core.method import Method

    model = Method()
    model.source = {"name": "MRSuite/wand-mrsi2024", "revision": "abc123"}
    code = Pipeline([model.step(remove=["nuisance"])]).script()

    assert "models.load('MRSuite/wand-mrsi2024', revision='abc123').step(remove=['nuisance'])" \
        in code


#****************#
#   resampling   #
#****************#
def test_a_step_that_resamples_keeps_its_own_spectral_width(spectrum):
    """The header comes over from the input, but the spectral width describes the
    output's sampling: a downsampling step halves it, and the output must stay valid."""
    from nifti_mrs.create_nmrs import gen_nifti_mrs

    @method("test", version="1.0")
    def halve(item):
        return gen_nifti_mrs(np.asarray(item[:])[..., ::2].copy(), item.dwelltime * 2,
                             item.spectrometer_frequency[0], nucleus=item.nucleus[0], no_conj=True)

    source = spectrum.copy()
    source.add_hdr_field("SpectralWidth", 1.0 / source.dwelltime)
    source.add_hdr_field("EchoTime", 0.03)

    out = halve(source)

    assert out.hdr_ext["SpectralWidth"] == pytest.approx(1.0 / out.dwelltime)
    assert out.hdr_ext["SpectralWidth"] == pytest.approx(0.5 / source.dwelltime)
    assert out.hdr_ext["EchoTime"] == 0.03
