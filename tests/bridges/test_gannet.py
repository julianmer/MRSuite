####################################################################################################
#                                          test_gannet.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 27/10/26                                                                                #
#                                                                                                  #
# Purpose: Gannet run as published under Octave: Osprey's Philips MEGA-PRESS example from load     #
#          to tissue-corrected quantification with SPM12, the HERMES acquisition of OpenNeuro      #
#          ds005371 through load and fit, and what MRSuite makes of Gannet's output.               #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import pytest

from mrsuite import externals
from mrsuite.bridges import gannet
from tests.conftest import needs_octave

MEGA = externals.path("osprey") / "exampledata" / "sdat" / "MEGA" / "sub-01" / "ses-01"
HERMES = (externals.CHECKOUT / "data" / "openneuro_ds005371" / "sub-01" / "ses-01" / "mrs"
          / "sub-01_ses-01_acq-hermespress")


#***************#
#   to result   #
#***************#
def test_gannets_output_is_filed_by_scaling():
    out = {"GABA": {"ConcIU": 2.1, "ConcCr": 0.12, "FitError": 5.5, "ConcIU_TissCorr": 3.7},
           "Glx": {"ConcIU": 5.2, "ConcCr": 0.3, "FitError": 4.0, "ConcIU_TissCorr": 9.0},
           "water": {"FWHM": 9.3, "Area": 1e6}, "tissue": {"fGM": 0.6, "fWM": 0.3, "fCSF": 0.1},
           "Cr": {"FWHM": 8.0}}

    result = gannet.to_result(out)

    assert result.metabolites == ["GABA", "Glx"], "what Gannet quantified, not water or Cr"
    assert result.conc("water") == {"GABA": 2.1, "Glx": 5.2}
    assert result.conc("internal")["GABA"] == 0.12
    assert result.conc("water_tissue_corrected")["Glx"] == 9.0
    assert result.crlb("fit_error") == {"GABA": 5.5, "Glx": 4.0}
    assert result.qc == {"water_fwhm": 9.3, "fGM": 0.6, "fWM": 0.3, "fCSF": 0.1}


#************#
#   no t1   #
#************#
def test_a_t1_without_water_is_refused_at_once():
    with pytest.raises(ValueError, match="give the water reference with the T1"):
        gannet.fit("act.sdat", t1="t1.nii")


#**********#
#   mega   #
#**********#
@needs_octave
@pytest.mark.skipif(not MEGA.is_dir(), reason="osprey submodule not checked out")
def test_mega_press_from_load_to_tissue_corrected_quantification(tmp_path):
    """The values Gannet 3.5.4 gives on this example under Octave, from its own drivers."""
    result = gannet.fit(str(MEGA / "mrs" / "sub-01_ses-01_megapress" / "sub-01_megapress_act.sdat"),
                        str(MEGA / "mrs" / "sub-01_ses-01_megapress-ref" / "sub-01_megapress_ref.sdat"),
                        t1=str(MEGA / "anat" / "sub-01" / "sub-01_ses-01_T1w.nii.gz"),
                        output=str(tmp_path))

    assert result.conc("water")["GABA"] == pytest.approx(2.0930, abs=5e-4)
    assert result.conc("internal")["GABA"] == pytest.approx(0.1191, abs=5e-5)
    assert result.crlb("fit_error")["GABA"] == pytest.approx(5.550, abs=5e-3)
    assert result.conc("water")["Glx"] == pytest.approx(5.1764, abs=5e-4)
    assert result.conc("water_tissue_corrected")["GABA"] == pytest.approx(3.7178, abs=5e-4)
    assert result.qc["water_fwhm"] == pytest.approx(9.277, abs=5e-3)
    assert (result.qc["fGM"], result.qc["fWM"], result.qc["fCSF"]) == \
        pytest.approx((0.6104, 0.2710, 0.1186), abs=5e-4)
    entry = result.provenance[0]
    assert entry["Program"] == "Gannet" and entry["Version"].endswith("f437bb7")
    assert entry["Method"].endswith("GannetQuantify") and '"spm12"' in entry["Details"]
    assert any(tmp_path.rglob("*.pdf")), "Gannet's own output kept where asked"


#************#
#   hermes   #
#************#
@needs_octave
@pytest.mark.skipif(not HERMES.with_name(HERMES.name + "_svs.nii.gz").exists(),
                    reason="ds005371 sub-01 not fetched from the annex")
def test_hermes_gaba_and_gsh_with_settings_given():
    result = gannet.fit(str(HERMES.with_name(HERMES.name + "_svs.nii.gz")),
                        str(HERMES.with_name(HERMES.name + "_mrsref.nii.gz")),
                        settings={"HERMES": 1, "target": ["GABAGlx", "GSH"]})

    assert set(result.metabolites) >= {"GABA", "Glx", "GSH"}
    assert result.conc("water")["GABA"] == pytest.approx(2.332, abs=5e-4)
    assert result.conc("water")["GSH"] == pytest.approx(0.552, abs=5e-4)
    assert result.conc("internal")["GABA"] == pytest.approx(0.185, abs=5e-4)
    assert result.conc("water")["Glx"] == pytest.approx(4.780, abs=5e-4)
