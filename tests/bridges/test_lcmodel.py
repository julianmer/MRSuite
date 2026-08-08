####################################################################################################
#                                        test_lcmodel.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-08-08                                                                              #
#                                                                                                  #
# Purpose: Fitting with LCModel through PyLCModel. LCModel is an external binary, so these         #
#          skip when it is not resolvable rather than failing, and the fit is asserted on          #
#          what a spectrum must physically look like rather than on stored numbers.                #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import pytest

from nifti_mrs_plus import NIfTI_MRS_Plus
from mrsuite.bridges import lcmodel
from mrsuite.core.result import FitResult

from tests.conftest import CHALLENGE_BASIS


#***********#
#   ready   #
#***********#
def _available():
    """Whether an LCModel binary can be resolved without downloading or building."""
    try:
        from lcmodel_wrapper import binaries
        binaries.resolve_executable(allow_download=False, allow_build=False)
        return True
    except Exception:                          # noqa: BLE001 - any failure means absent
        return False


needs_lcmodel = pytest.mark.skipif(not _available(), reason="no cached LCModel binary")
pytestmark = needs_lcmodel


#***********#
#   parse   #
#***********#
def test_to_result_maps_a_parsed_coord():
    """The .coord carries both scalings and LCModel's own QC, so all of it should land."""
    parsed = (["NAA", "Cr+PCr"], [8.33, 5.96], [3.0, 2.0], [1.397, 1.0],
              0.032, 35.0, -0.002, 12.0)

    out = lcmodel.to_result(parsed)

    assert isinstance(out, FitResult) and out.method == "lcmodel:LCModel"
    assert out.conc("raw")["NAA"] == 8.33
    assert out.conc("internal")["NAA"] == 1.397     # /Cr+PCr, as LCModel reports it
    assert out.crlb("percentage")["NAA"] == 3.0
    assert out.qc == {"fwhm": 0.032, "snr": 35.0}
    assert out.raw["phase"] == 12.0


#****************#
#   missing qc   #
#****************#
def test_absent_qc_is_omitted_rather_than_faked():
    parsed = (["NAA"], [8.33], [3.0], [1.4], None, None, None, None)

    assert lcmodel.to_result(parsed).qc == {}


#*********#
#   fit   #
#*********#
def test_a_fit_runs_end_to_end(spectrum):
    """Asserted on physiology, not on stored numbers: a mis-scaled or mirrored spectrum
    still fits and still returns finite values, but not with these ratios."""
    out = lcmodel.fit(spectrum, CHALLENGE_BASIS)

    assert out.method == "lcmodel:LCModel"
    tcr = out.conc("internal")
    assert 1.0 < tcr["NAA"] < 2.0, f"NAA/tCr = {tcr['NAA']}, physiologically ~1.5"
    assert tcr["Cr+PCr"] == pytest.approx(1.0), "the reference is itself by definition"
    assert 0 < out.crlb("percentage")["NAA"] < 20


#***********#
#   batch   #
#***********#
def test_a_batch_gives_one_result_per_spectrum(spectra):
    out = lcmodel.fit(spectra, CHALLENGE_BASIS)

    assert isinstance(out, list) and len(out) == len(spectra)
    # the same spectrum twice, so the fits must agree
    assert out[0].conc("raw")["NAA"] == pytest.approx(out[1].conc("raw")["NAA"])


#***********#
#   water   #
#***********#
def test_a_mismatched_number_of_references_is_rejected(spectra, spectrum):
    three = NIfTI_MRS_Plus([spectrum.copy() for _ in range(3)])

    with pytest.raises(ValueError, match="3 water references for 2 spectra"):
        lcmodel.fit(spectra, CHALLENGE_BASIS, water=three)


#***************#
#   save path   #
#***************#
def test_save_path_leaves_the_lcmodel_files_behind(spectrum, tmp_path):
    """The .coord and .ps are what you inspect when a fit looks wrong.

    save_path is PyLCModel's own option, passed straight through: there is no reason
    for MRSuite to have an opinion about where the intermediates go.
    """
    lcmodel.fit(spectrum, CHALLENGE_BASIS, save_path=str(tmp_path))

    assert (tmp_path / "temp0.coord").exists()
    assert (tmp_path / "temp0.raw").exists()
