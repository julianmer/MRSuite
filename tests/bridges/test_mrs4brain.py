####################################################################################################
#                                        test_mrs4brain.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 27/10/26                                                                                #
#                                                                                                  #
# Purpose: MRS4Brain run as published under Octave: its 14.1 T SPECIAL demo study through its      #
#          FID-A preprocessing and LCModel quantification with its own demo configuration, and     #
#          its MRSI refused in one line.                                                           #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import pytest

from mrsuite import externals
from mrsuite.bridges import mrs4brain
from tests.conftest import needs_octave

DEMO = externals.path("mrs4brain") / "Demo"
SVS = DEMO / "2_SVS_Data" / "SVS_Data_14p1T"
MRSI = DEMO / "1_MRSI_Data" / "MRSI_Data_14p1T" / "20220531_122600_FID"


#*********#
#   svs   #
#*********#
@needs_octave
@pytest.mark.skipif(not SVS.is_dir(), reason="mrs4brain not fetched")
def test_the_svs_demo_through_preprocessing_and_lcmodel(tmp_path):
    """The values MRS4Brain gives on its demo under Octave, from its own GUI's calls."""
    result = mrs4brain.quantify(str(SVS / "20240306_153617_SPECIAL"), 12, 11,
                                str(SVS / "14p1T_SPECIAL_SVS_TE9ms_Demo.BASIS"),
                                config="Demo_SVS_14p1T_SPECIAL", output=str(tmp_path))
    mm, crlb = result.conc("molarity"), result.crlb()

    assert (mm["NAA+NAAG"], mm["Glu+Gln"], mm["Cr+PCr"]) == \
        pytest.approx((12.438, 16.099, 10.584), abs=5e-4)
    assert (mm["GPC+PCho"], mm["Ins"], mm["Tau"]) == pytest.approx((1.482, 11.041, 8.595),
                                                                   abs=5e-4)
    assert (crlb["NAA+NAAG"], crlb["Glu+Gln"], crlb["Cr+PCr"]) == (2, 3, 3)
    assert result.conc("internal")["Cr+PCr"] == pytest.approx(1.0), "ratios to Cr+PCr"
    assert result.conc("internal")["NAA+NAAG"] == pytest.approx(9.402 / 8, abs=1e-3)
    entry = result.provenance[0]
    assert entry["Program"] == "MRS4Brain" and entry["Version"].endswith("364e12e")
    assert '"fid-a"' in entry["Details"] and '"lcmodel"' in entry["Details"]
    assert any(tmp_path.rglob("*.mat")), "its experiment saved where asked"


#**********#
#   mrsi   #
#**********#
@pytest.mark.skipif(not MRSI.is_dir(), reason="mrs4brain not fetched")
def test_an_mrsi_experiment_is_refused_in_one_line():
    with pytest.raises(ValueError, match="MRS4Brain's MRSI does not run under Octave"):
        mrs4brain.quantify(str(MRSI), 20, 21, "basis.BASIS")
