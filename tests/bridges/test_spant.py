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


#*******************#
#   changed shape   #
#*******************#
def test_a_step_that_resamples_keeps_only_the_dimensions_the_data_have():
    """spant writes a dimension for every one it knows, of length one where the data have
    none; transients downsampled come back with their one dimension, as FSL-MRS's outlier
    removal needs them."""
    from nifti_mrs.create_nmrs import gen_nifti_mrs
    from nifti_mrs_plus import NIfTI_MRS_Plus
    from mrsuite import spant

    t = np.arange(4096) * 1.25e-4
    fid = np.exp(2j * np.pi * (2.0 - 4.65) * 123.2 * t - t / 0.05)
    dyn = np.stack([fid * (1 + 0.01 * k) for k in range(4)], -1).reshape(1, 1, 1, -1, 4)
    transients = NIfTI_MRS_Plus([gen_nifti_mrs(dyn, 1.25e-4, 123.2,
                                               dim_tags=["DIM_DYN", None, None])])

    item = spant.downsample_mrs_td(transients).list()[0]

    assert item.shape == (1, 1, 1, 2048, 4) and item.dim_tags[0] == "DIM_DYN"
    assert item.dwelltime == pytest.approx(2.5e-4)


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


#************#
#   curves   #
#************#
def test_a_fit_keeps_the_curves_spant_drew(spectrum):
    """The driver hands back spant's fitted series beside the result row, scaled back to
    the data as the amplitudes are: over ABfit's range, the fit on the data."""
    from mrsuite.bridges import spant
    from tests.conftest import CHALLENGE_BASIS

    out = spant.fit(spectrum, CHALLENGE_BASIS)

    curves = out.curves()
    ppm, data, fit = curves["ppm"], curves["data"], curves["fit"]
    assert len(ppm) == len(data) == len(fit) == len(curves["baseline"]) > 100
    assert 0.15 < ppm[0] < 0.25 and 3.9 < ppm[-1] < 4.1
    assert np.linalg.norm(data - fit) < 0.2 * np.linalg.norm(data)
    peak = np.abs(np.fft.fft(np.asarray(spectrum[:]).ravel())).max()   # the data's units
    assert 0.8 * peak < data.max() < 1.25 * peak
