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
    assert "already combined" in fsl_mrs.bridge._skipped[0]


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
