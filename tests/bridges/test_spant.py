####################################################################################################
#                                          test_spant.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-27                                                                              #
#                                                                                                  #
# Purpose: The spant bridge. spant writes NIfTI-MRS other readers do not accept as it is, so       #
#          what a step hands back is checked against the spectrum that went in: its shape, its     #
#          ppm reference and its history.                                                         #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import numpy as np
import pytest

from mrsuite import r

pytestmark = pytest.mark.skipif(not r.available(), reason="Rscript not on PATH")


#*************#
#   a step    #
#*************#
def test_a_step_keeps_the_spectrum_what_it_was(spectrum):
    from mrsuite import spant

    source = spectrum.copy()
    source.add_hdr_field("SpecFreqChemShift", 4.70)
    out = spant.lb(source, 5)

    assert out.shape == source.shape and out.dim_tags == source.dim_tags
    assert out.SpecFreqChemShift == 4.70
    assert not np.allclose(out[:], source[:])                # it did broaden the lines
    assert out.hdr_ext["ProcessingApplied"][-1]["Program"] == "spant"


#************#
#   values   #
#************#
def test_a_step_that_changes_nothing_gives_the_spectrum_back_as_it_was(spectrum):
    """No line broadening is no change at all: not the conjugate, which a file read back
    raw, in NIfTI-MRS' own convention, would be under nifti_mrs' conjugated one."""
    from mrsuite import spant

    out = spant.lb(spectrum.copy(), 0)

    given = np.asarray(spectrum[:]).ravel()
    assert np.allclose(np.asarray(out[:]).ravel(), given, atol=1e-6 * np.abs(given).max())


def test_a_function_that_gives_no_spectra_gives_its_value(spectrum):
    """calc_spec_snr returns a number per spectrum, not spectra."""
    from mrsuite import spant

    snr = spant.calc_spec_snr(spectrum)

    assert isinstance(snr, float) and snr > 10


def test_a_value_is_given_back_for_every_spectrum_even_none(spectra, tmp_path):
    """write_mrs returns NULL; R drops a NULL from a list, which lost spectra."""
    from mrsuite import spant

    out = spant.write_mrs(spectra, fname=str(tmp_path / "w.nii.gz"), format="nifti",
                          force=True)

    assert out == [None] * len(spectra)
    assert [type(v) for v in spant.is_fd(spectra)] == [bool] * len(spectra), "logicals stay"
    assert [type(v) for v in spant.Npts(spectra)] == [int] * len(spectra), "counts stay"


#*************#
#   unknown   #
#*************#
def test_an_unknown_name_is_rejected_at_lookup():
    from mrsuite import spant

    with pytest.raises(AttributeError, match="no function"):
        spant.not_a_spant_function


#***************#
#   untouched   #
#***************#
def test_the_callers_spectrum_is_left_alone(spectrum):
    """Saving for R must not repoint the caller's image at a file that is then deleted."""
    from mrsuite import spant

    source = spectrum.copy()
    spant.lb(source, 5)
    spant.lb(source, 5)                                       # would fail on a stale file

    assert np.array_equal(source[:], spectrum[:])


#***********#
#   about   #
#***********#
def test_spant_states_a_function_as_r_does():
    from mrsuite.bridges.spant import bridge

    signature, help_ = bridge.about("lb")

    assert str(signature) == "(x, /, *, lb, lg=1)"                    # lg's default, R's own
    assert "Apply line-broadening" in help_ and "Arguments:" in help_
    assert "lb" in bridge.steps() and "fit_mrs" not in bridge.steps()


#****************#
#   on its own   #
#****************#
def test_a_spant_call_does_not_wait_for_every_help_page(monkeypatch):
    """Arguments and help are a job of their own, which a call does not run."""
    from mrsuite.bridges.spant import bridge

    def slow(runtime=None):
        raise AssertionError("a call asked for every help page")
    monkeypatch.setattr(r, "about", slow)

    assert callable(bridge.lb) and "help" not in r.info()


def test_r_dots_first_still_take_the_data():
    from mrsuite.bridges.spant import bridge

    assert str(bridge.about("append_coils")[0]) == "(*args, **kwargs)"
