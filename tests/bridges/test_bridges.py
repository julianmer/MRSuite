####################################################################################################
#                                         test_bridges.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-08-08                                                                              #
#                                                                                                  #
# Purpose: Dynamic dispatch onto FSL-MRS and FID-A. The value of a bridge is breadth, so           #
#          these call functions named nowhere in MRSuite's own source: dispatch that only          #
#          worked for a curated list would still pass a handful of hand-picked calls.              #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import json

import numpy as np
import pytest

from nifti_mrs.nifti_mrs import NIFTI_MRS
from nifti_mrs_plus import NIfTI_MRS_Plus
from mrsuite import fid_a, fsl_mrs
from mrsuite.bridges import describe

from tests.conftest import needs_octave

# parametrised rather than picked: a handful of curated calls would pass even if
# dispatch only worked for the shapes those calls happen to use
FIDA_OPS = [("op_averaging", ()), ("op_addphase", (30.0,)), ("op_ampScale", (2.0,)),
            ("op_freqshift", (10.0,)), ("op_leftshift", (2,)), ("op_filter", (5.0,)),
            ("op_addNoise", (10.0,)), ("op_complexConj", ()), ("op_freqrange", (1.0, 4.0))]


#*********************#
#   fsl-mrs surface   #
#*********************#
def test_fsl_mrs_exposes_the_whole_module():
    """Dispatch must cover the module, not a curated subset of it."""
    names = fsl_mrs.bridge.available()

    assert len(names) > 15
    for expected in ("average", "align", "coilcombine", "conjugate", "ecc", "apodize"):
        assert expected in names


#**********************#
#   fsl-mrs dispatch   #
#**********************#
def test_fsl_mrs_single_and_batched(spectrum, spectra):
    single = fsl_mrs.conjugate(spectrum)
    assert single.shape == spectrum.shape

    batched = fsl_mrs.conjugate(spectra)
    assert isinstance(batched, NIfTI_MRS_Plus) and len(batched) == len(spectra)
    assert np.allclose(np.asarray(batched[0][:]), np.asarray(single[:]))


#*********************#
#   fsl-mrs applied   #
#*********************#
def test_fsl_mrs_conjugate_actually_conjugates(spectrum):
    """A bridge that quietly returned its input would pass a type check but be useless."""
    out = fsl_mrs.conjugate(spectrum)

    assert np.allclose(np.asarray(out[:]).imag, -np.asarray(spectrum[:]).imag)


#*********************#
#   fsl-mrs unknown   #
#*********************#
def test_fsl_mrs_unknown_name_is_rejected_at_lookup():
    with pytest.raises(AttributeError, match="no processing function"):
        fsl_mrs.definitely_not_a_function


#*******************#
#   fid-a surface   #
#*******************#
@needs_octave
def test_fid_a_exposes_the_whole_toolbox():
    names = fid_a.bridge.available()

    assert len(names) > 200, f"only {len(names)} discovered"
    for expected in ("op_averaging", "op_freqrange", "op_addphase", "op_rmbadaverages"):
        assert expected in names


#**********************#
#   fid-a operations   #
#**********************#
@needs_octave
@pytest.mark.parametrize("name,args", FIDA_OPS, ids=[n for n, _ in FIDA_OPS])
def test_fid_a_operations_round_trip(spectrum, name, args):
    """Every op must run in Octave and come back as NIfTI-MRS with usable data."""
    out = getattr(fid_a, name)(spectrum, *args)

    assert out.__class__.__name__ == "NIFTI_MRS"
    arr = np.asarray(out[:])
    assert arr.size > 0 and np.all(np.isfinite(arr)), f"{name}: empty or non-finite"


#*********************#
#   fid-a unwrapped   #
#*********************#
@needs_octave
def test_fid_a_dispatches_a_function_mrsuite_never_wraps(spectrum):
    """op_freqrange is named nowhere in MRSuite's source, and must still dispatch."""
    out = fid_a.op_freqrange(spectrum, 1.0, 4.0)

    assert out.shape[3] < spectrum.shape[3], "frequency range should have cropped the FID"


#*************#
#   prompts   #
#*************#
@needs_octave
def test_octave_asking_for_input_fails_at_once_rather_than_waiting(tmp_path):
    """FID-A asks interactively for some left-out arguments; nobody is there to answer."""
    from mrsuite.octave import session

    oc = session(str(tmp_path))
    try:
        with pytest.raises(Exception, match="Octave asked for input \\(a number:\\)"):
            oc.eval("x = input('a number: ');")
        assert int(oc.eval("40", nout=1)) == 40, "the next call is its own, not an answer"
        assert int(oc.eval("exist('x')", nout=1)) == 0
    finally:
        oc.exit()


#*******************#
#   fid-a batched   #
#*******************#
@needs_octave
def test_fid_a_batches_over_nifti_mrs_plus(spectra):
    out = fid_a.op_freqrange(spectra, 1.0, 4.0)

    assert isinstance(out, NIfTI_MRS_Plus) and len(out) == len(spectra)


#*****************#
#   fid-a phase   #
#*****************#
@needs_octave
def test_fid_a_addphase_changes_phase_but_not_magnitude(spectrum):
    """A numerical check rather than a type check: a pure phase shift must leave the
    magnitude spectrum alone."""
    out = fid_a.op_addphase(spectrum, 30.0)

    a, b = np.asarray(spectrum[:]).squeeze(), np.asarray(out[:]).squeeze()
    assert np.allclose(np.abs(a), np.abs(b), rtol=1e-6), "magnitude should be preserved"
    assert not np.allclose(a, b, rtol=1e-6), "phase should have changed"


#*************#
#   outputs   #
#*************#
@needs_octave
def test_fid_a_converts_the_spectra_among_several_outputs(spectrum):
    """A function with more outputs than the spectra still hands the spectra back as
    NIfTI-MRS, and the rest as they are."""
    out, filt = fid_a.op_filter(spectrum, 5.0, nout=2)

    assert isinstance(out, NIFTI_MRS)
    assert np.size(filt) == spectrum.shape[3]   # the filter itself, one value per point


#*******************#
#   fid-a unknown   #
#*******************#
@needs_octave
def test_fid_a_unknown_name_is_rejected_at_lookup():
    with pytest.raises(AttributeError, match="no function"):
        fid_a.op_definitely_not_real


#******************#
#   struct types   #
#******************#
@needs_octave
def test_struct_sizes_reach_octave_as_numbers(spectrum):
    """``sz`` must arrive as a double, not a cell.

    oct2py maps a Python tuple to an Octave cell array, and FID-A does arithmetic on
    sz (in.sz(1)*zpFactor), which fails on a cell. It breaks a wide swathe of the
    toolbox while the simplest operations keep working, so it is easy to miss.
    """
    struct = fid_a.bridge.codec.nifti2fida(spectrum)
    octave = fid_a.bridge.octave
    octave.push("s", struct["sz"])

    assert str(octave.eval("class(s)", nout=1)).strip() == "double"


#*******************#
#   preconditions   #
#*******************#
def test_coilcombine_declines_on_a_single_coil(spectra):
    """Already-combined data can still declare DIM_COIL of length 1, where
    copy(remove_dim=...) raises AxisError. The bridge must decline, not crash.
    """
    out = fsl_mrs.coilcombine(spectra)

    assert isinstance(out, NIfTI_MRS_Plus) and len(out) == len(spectra)
    assert np.allclose(np.asarray(out[0][:]), np.asarray(spectra[0][:]))
    assert fsl_mrs.bridge._skipped, "skip should be recorded, not silent"
    assert "DIM_COIL has length 1: nothing to combine" in fsl_mrs.bridge._skipped[0]


@pytest.mark.parametrize("bridge, func, why", [
    ("fsl_mrs", "remove_unlike", "no DIM_DYN: nothing to remove"),
    pytest.param("fid_a", "op_combinesubspecs", "no DIM_EDIT or DIM_ISIS: nothing to combine",
                 marks=needs_octave),
    pytest.param("osprey", "op_averaging", "no DIM_DYN: nothing to average",
                 marks=needs_octave),
])
def test_an_operation_over_a_dimension_the_data_lacks_declines(spectra, bridge, func, why):
    """Averaged, combined, unedited: nothing to remove, combine or average, where the
    toolbox itself fails (FSL-MRS: IndexError; FID-A: sum over dimension 0; Osprey:
    'Averaging has already been performed')."""
    import importlib

    module = importlib.import_module(f"mrsuite.bridges.{bridge}")
    args = ("diff",) if func == "op_combinesubspecs" else ()
    out = getattr(module.bridge, func)(spectra, *args)

    kept = out[0] if isinstance(out, tuple) else out          # remove_unlike: kept, removed
    assert np.allclose(np.asarray(kept[0][:]), np.asarray(spectra[0][:]))
    assert why in module.bridge._skipped[0]


@needs_octave
def test_a_declined_call_gives_back_as_many_outputs_as_asked_for(spectra):
    out, metric, bad = fid_a.op_rmbadaverages(spectra, 3, "t", nout=3)

    assert np.allclose(np.asarray(out[0][:]), np.asarray(spectra[0][:]))
    assert metric == bad == [None] * len(spectra), "one for each spectrum, as a run gives"


#****************#
#   provenance   #
#****************#
def test_fsl_mrs_records_each_step_once(spectra):
    """FSL-MRS writes its own entry; the bridge must not add a second one."""
    out = fsl_mrs.conjugate(spectra)

    for spectrum in out.list():
        steps = spectrum.hdr_ext["ProcessingApplied"]
        assert [s["Program"] for s in steps].count("FSL-MRS") == 1
        assert "MRSuite" not in [s["Program"] for s in steps]
    # and the NIfTI-MRS+ batch carries the same history
    assert out.metadata_individual[0]["ProcessingApplied"] == \
        out[0].hdr_ext["ProcessingApplied"]


#**********************#
#   fid-a provenance   #
#**********************#
@needs_octave
def test_fid_a_records_the_toolbox_version_and_every_argument(spectra):
    """Enough to redo the call: the toolbox at its exact version, the function, and
    each argument, with MRSuite and the runtime alongside."""
    out = fid_a.op_addphase(spectra, 30.0)

    step = out[0].hdr_ext["ProcessingApplied"][-1]
    details = json.loads(step["Details"])
    assert (step["Program"], step["Method"]) == ("FID-A", "op_addphase")
    assert step["Version"] == fid_a.bridge.version() != "unknown"
    assert details["args"] == [30.0] and details["kwargs"] == {}
    assert {"mrsuite", "runtime", "interpreter"} <= set(details)


#******************#
#   fid-a header   #
#******************#
@needs_octave
def test_transients_keep_their_dimension_through_fid_a(spectrum):
    """A struct's header comes back from Octave whole, so transients (DIM_DYN) are still
    transients after an FID-A step that keeps them: aligning them gives as many back."""
    from nifti_mrs.create_nmrs import gen_nifti_mrs

    fid = np.asarray(spectrum[:]).ravel()
    noise = np.random.default_rng(0).normal(scale=1e-3 * np.abs(fid).max(), size=(8, fid.size))
    transients = gen_nifti_mrs((fid[:, None] + noise.T).reshape(1, 1, 1, fid.size, 8),
                               spectrum.dwelltime, spectrum.spectrometer_frequency[0],
                               dim_tags=["DIM_DYN", None, None])

    out = fid_a.op_alignAverages(transients)

    assert out.shape == transients.shape and out.dim_tags[0] == "DIM_DYN"


@needs_octave
def test_a_fid_a_step_keeps_the_header_and_the_history(spectrum):
    """FID-A hands back samples, not the header, which would lose the echo time, the
    ppm reference and every earlier step at the first FID-A call."""
    before = fsl_mrs.apodize(NIfTI_MRS_Plus([spectrum.copy()]), amount=[5.0])[0]
    before.add_hdr_field("EchoTime", 0.03)
    before.add_hdr_field("SpecFreqChemShift", 4.70)

    after = fid_a.op_addphase(before, 30.0)

    assert after.hdr_ext["EchoTime"] == 0.03 and after.SpecFreqChemShift == 4.70
    assert [s["Program"] for s in after.hdr_ext["ProcessingApplied"]] == ["FSL-MRS", "FID-A"]


#**********#
#   step   #
#**********#
def test_step_binds_arguments_and_waits_for_the_data(spectrum):
    """The data is the first argument, so functools.partial cannot bind the rest."""
    step = fsl_mrs.step.apodize((10.0,))

    assert callable(step) and step.__name__ == "apodize"
    assert np.allclose(np.asarray(step(spectrum)[:]),
                       np.asarray(fsl_mrs.apodize(spectrum, (10.0,))[:]))


#****************#
#   step water   #
#****************#
def test_step_can_bind_another_spectrum_as_an_argument(spectrum):
    """The routing case: ecc(data, reference) takes a second NIfTI-MRS.

    This is why deferring is spelled out rather than inferred from "no data passed" -
    ecc(water) is indistinguishable from running ecc on the water.
    """
    step = fsl_mrs.step.ecc(spectrum)

    assert step.__name__ == "ecc"
    assert np.allclose(np.asarray(step(spectrum)[:]),
                       np.asarray(fsl_mrs.ecc(spectrum, spectrum)[:]))


#******************#
#   step unknown   #
#******************#
def test_step_rejects_an_unknown_name_at_lookup():
    with pytest.raises(AttributeError, match="no processing function"):
        fsl_mrs.step.definitely_not_a_function


#*********#
#   fit   #
#*********#
def test_fsl_mrs_fit_returns_a_uniform_result(spectrum, basis):
    """Fitting is not reachable through dispatch: that covers nifti_mrs_proc, which is
    processing only, and fit_FSLModel takes an MRS object rather than NIfTI-MRS."""
    out = fsl_mrs.fit(spectrum, basis)

    assert out.method == "fsl_mrs:Newton"
    assert out.conc("raw")["NAA"] > 0


#***************#
#   fit plain   #
#***************#
def test_fsl_mrs_fits_any_nifti_mrs(spectrum, basis):
    """Spectra from other methods and simulators are plain NIfTI-MRS, without FSL-MRS's
    additions; they fit exactly as FSL-MRS's own objects do."""
    from nifti_mrs.create_nmrs import gen_nifti_mrs

    plain = gen_nifti_mrs(np.asarray(spectrum[:]), spectrum.dwelltime,
                          spectrum.spectrometer_frequency[0], nucleus=spectrum.nucleus[0],
                          dim_tags=spectrum.dim_tags)

    assert fsl_mrs.fit(plain, basis).conc("raw") == fsl_mrs.fit(spectrum, basis).conc("raw")


#***************#
#   fit batch   #
#***************#
def test_fsl_mrs_fit_returns_one_result_per_spectrum(spectra, basis):
    out = fsl_mrs.fit(spectra, basis)

    assert isinstance(out, list) and len(out) == len(spectra)
    # a list as long as the batch is what a pipeline files per spectrum
    assert all(r.conc("raw")["NAA"] > 0 for r in out)


#*****************#
#   fit + water   #
#*****************#
def test_water_scaling_needs_an_echo_and_repetition_time(spectrum, basis):
    """jMRUI text carries neither, and the T1/T2 corrections cannot be applied without
    them. Better to say so than to report an unscaled number as a molar concentration.
    """
    with pytest.raises(ValueError, match="echo and repetition time"):
        fsl_mrs.fit(spectrum, basis, water=spectrum)


#******************#
#   fit molarity   #
#******************#
def test_water_scaling_adds_the_molar_concentrations(spectrum, basis):
    out = fsl_mrs.fit(spectrum, basis, water=spectrum, te=0.030, tr=2.0)

    assert "molarity" in out.concentrations
    assert "molality" in out.concentrations
    assert out.conc("molarity")["NAA"] > 0


#*******************#
#   fit collapsed   #
#*******************#
def test_a_fit_whose_reference_collapsed_keeps_its_raw_values(spectrum, basis, monkeypatch):
    """A fit that drives Cr+PCr to zero cannot be scaled to it; FSL-MRS raises, and in a
    batch that one fit took every other spectrum's result down with it. The fit comes
    back with what it has, raw values and no scaling it could not compute."""
    from fsl_mrs.utils.results import FitRes
    internal_only = FitRes.calculateConcScaling

    def collapsed(self, mrs, quant_info=None, **kwargs):
        if quant_info is not None:          # FSL-MRS raises only when scaling to water
            raise FitRes.QuantificationError("Metabolite reference ['Cr', 'PCr'] has not been fit")
        return internal_only(self, mrs, **kwargs)

    monkeypatch.setattr(FitRes, "calculateConcScaling", collapsed)
    with pytest.warns(UserWarning, match="raw values"):
        out = fsl_mrs.fit(spectrum, basis, water=spectrum, te=0.030, tr=2.0)

    assert out.conc("raw")["NAA"] > 0
    assert "molarity" not in out.concentrations


#*****************#
#   empty water   #
#*****************#
def test_a_water_reference_without_signal_stays_an_error(spectrum, basis, monkeypatch):
    """FSL-MRS raises the same error for a water reference with no integral, which is
    broken input, not a failed fit: it must not pass as raw values."""
    from fsl_mrs.utils.results import FitRes
    internal_only = FitRes.calculateConcScaling

    def empty(self, mrs, quant_info=None, **kwargs):
        if quant_info is not None:
            raise FitRes.QuantificationError("Water reference has zero integral. Please check "
                                             "water reference data.")
        return internal_only(self, mrs, **kwargs)

    monkeypatch.setattr(FitRes, "calculateConcScaling", empty)
    with pytest.raises(FitRes.QuantificationError, match="zero integral"):
        fsl_mrs.fit(spectrum, basis, water=spectrum, te=0.030, tr=2.0)


#**************#
#   fit refs   #
#**************#
def test_one_water_reference_is_not_spread_over_a_batch(spectra, basis, spectrum):
    """A batch is separate acquisitions, so scaling them all by one subject's water
    would be wrong without looking wrong. Rejected rather than broadcast."""
    with pytest.raises(ValueError, match="water gives 1 for 2 spectra"):
        fsl_mrs.fit(spectra, basis, water=spectrum, te=0.030, tr=2.0)


#****************#
#   fit paired   #
#****************#
def test_a_reference_per_spectrum_is_scaled(spectra, basis, spectrum):
    water = NIfTI_MRS_Plus([spectrum.copy(), spectrum.copy()])

    out = fsl_mrs.fit(spectra, basis, water=water, te=0.030, tr=2.0)

    assert len(out) == len(spectra)
    assert all("molarity" in r.concentrations for r in out)


#******************#
#   fit mismatch   #
#******************#
def test_a_mismatched_number_of_references_is_rejected(spectra, basis, spectrum):
    three = NIfTI_MRS_Plus([spectrum.copy() for _ in range(3)])

    with pytest.raises(ValueError, match="water gives 3 for 2 spectra"):
        fsl_mrs.fit(spectra, basis, water=three, te=0.030, tr=2.0)


#**************#
#   describe   #
#**************#
def test_describe_summarises_a_bridge():
    text = describe(fsl_mrs.bridge)

    assert "fsl_mrs" in text and "functions" in text


#***********#
#   about   #
#***********#
def test_a_python_toolbox_function_states_its_own_arguments():
    import inspect

    signature, help_ = fsl_mrs.bridge.about("average")

    assert list(signature.parameters)[:2] == ["data", "dim"]
    assert inspect.signature(fsl_mrs.average) == signature, "the dispatched function says so too"
    assert ":param str dim:" in help_


def test_fid_a_states_a_function_from_its_file_without_starting_octave():
    from mrsuite.bridges.fid_a import FIDABridge

    bridge = FIDABridge()
    signature, help_ = bridge.about("op_rmbadaverages")

    assert str(signature) == "(in_, nsd=3, domain='t', /, *, nout=1)"   # FID-A's 'in', spelt
    assert "Removes motion corrupted averages" in help_                # its block before
    assert bridge._octave is None


def test_osprey_help_is_the_block_after_the_function_line():
    from mrsuite.bridges.osprey import OspreyBridge

    signature, help_ = OspreyBridge().about("OspreyProcess")

    assert str(signature) == "(MRSCont, /, *, nout=1)"
    assert "pre-processes MRS data" in help_


#***********#
#   steps   #
#***********#
def test_the_steps_are_the_functions_on_spectra():
    from mrsuite.bridges.fid_a import FIDABridge

    assert "average" in fsl_mrs.bridge.steps()
    assert not {"first_index", "update_processing_prov"} & set(fsl_mrs.bridge.steps())
    steps = FIDABridge().steps()
    assert "op_averaging" in steps and all(name.startswith("op_") for name in steps)


#*************#
#   m files   #
#*************#
@pytest.mark.parametrize("bridge, func, signature", [
    ("fid_a", "select_spec", "(src, unused1, /, *, nout=1)"),        # ~ for an unused argument
    ("fid_a", "op_CSIB0Correction",                                   # continued with ...
     "(MRSIStruct, phaseMap, freqMap, plottingArguments, /, *, nout=1)"),
    ("osprey", "osp_RemovePreFix", "(Name, /, *, nout=1)"),           # function[out]
    ("fid_a", "run_simSemiLASERShaped_fast_phCyc", "None"),           # a script, its own
])                                                                    # functions not it
def test_a_function_line_is_read_as_octave_reads_it(bridge, func, signature):
    from mrsuite.bridges.fid_a import FIDABridge
    from mrsuite.bridges.osprey import OspreyBridge

    found = (FIDABridge if bridge == "fid_a" else OspreyBridge)().about(func)[0]

    assert str(found) == signature


@pytest.mark.parametrize("bridge, func, signature", [
    ("fid_a", "op_getSNR", "(in_, NAAppmmin=1.8, NAAppmmax=2.2, noiseppmmin=-2, "
                           "noiseppmmax=0, suppressPlots=False, /, *, nout=1)"),
    ("fid_a", "op_removeWater",                                        # M worked out
     "(in_, wlim=[4.4, 5], Kinit=30, M=set by the toolbox, plot_bool=0, /, *, nout=1)"),
    ("osprey", "op_robustSpecReg",                         # echo set only when nargin<2
     "(in_, seqType, echo, F0=set by the toolbox, noAlign=0, /, *, nout=1)"),
    ("fid_a", "op_autophase",                              # dimNum asked for: required
     "(in_, ppmmin, ppmmax, ph, dimNum, /, *, nout=1)"),
    ("fid_a", "op_getLW", "(in_, Refppmmin=4.4, Refppmmax=5.0, zpfactor=8, "  # in = ... later
                          "suppressPlots=False, /, *, nout=1)"),
])
def test_a_functions_own_defaults_are_read_from_its_body(bridge, func, signature):
    from mrsuite.bridges.fid_a import FIDABridge
    from mrsuite.bridges.osprey import OspreyBridge

    assert str((FIDABridge if bridge == "fid_a" else OspreyBridge)().about(func)[0]) == signature


def test_a_default_holds_only_inside_its_guard():
    from mrsuite.bridges.fid_a import _optional

    body = ["if nargin<3", "  a = 1;", "else", "  b = 2;", "end", "if ~exist('c','var')",
            "  c = 'x';", "end", "d = 4;", "if nargin<6, e = 5; end", "e = 6;",
            "if nargin<=6", "  switch d", "    case 4", "      f = 'one';", "  end", "end",
            "if nargin==7", "  g = 7;", "end"]

    found = _optional(body, ["in", "a", "b", "c", "d", "e", "f", "g"])

    assert set(found) == {"a", "c", "f"}, "e: a one-line if opens nothing; g: == misses calls"
    assert (found["a"], found["c"], type(found["f"]).__name__) == (1, "x", "Computed")


@needs_octave
def test_a_fid_a_default_read_is_the_one_fid_a_uses(spectrum):
    left_out = fid_a.op_getSNR(spectrum)

    assert left_out == fid_a.op_getSNR(spectrum, 1.8, 2.2, -2, 0, False)


def test_fsl_mrs_takes_its_command_lines_defaults(spectrum):
    """fsl_mrs_proc phase --ppm 2.8 3.2, and so on: what FSL-MRS's own command line gives
    the arguments its functions leave without one."""
    import inspect

    phase = inspect.signature(fsl_mrs.phase_correct).parameters["ppmlim"].default
    shift = inspect.signature(fsl_mrs.shift_to_reference).parameters
    assert phase == (2.8, 3.2)
    assert (shift["ppm_ref"].default, shift["peak_search"].default) == (3.027, (2.8, 3.2))
    assert "fsl_mrs_proc" in fsl_mrs.bridge.about("phase_correct")[1]
    left_out = np.asarray(fsl_mrs.phase_correct(spectrum)[:])
    assert np.array_equal(left_out, np.asarray(fsl_mrs.phase_correct(spectrum, (2.8, 3.2))[:]))


def test_an_fsl_mrs_whose_command_line_is_laid_out_otherwise_still_runs(monkeypatch, spectrum):
    from mrsuite.bridges import fsl_mrs as bridged

    def otherwise():
        raise KeyError("main")

    monkeypatch.setattr(bridged, "_read_proc_defaults", otherwise)
    bridged._proc_defaults.cache_clear()
    try:
        assert bridged._proc_defaults() == {}
        assert fsl_mrs.phase_correct(spectrum, (2.8, 3.2)) is not None
    finally:
        bridged._proc_defaults.cache_clear()


def test_the_help_is_the_files_first_comment_block():
    """FID-A's header, not a comment in the code after the function line."""
    from mrsuite.bridges.fid_a import FIDABridge

    assert FIDABridge().about("op_HSVDfit")[1].startswith("op_HSVDfit.m")
