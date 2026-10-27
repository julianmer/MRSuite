####################################################################################################
#                                           test_oryx.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 27/10/26                                                                                #
#                                                                                                  #
# Purpose: Oryx-MRSI run as published under Octave: its shipped patient K_01, LCModel-fitted, to   #
#          its metabolite maps with its GUI's defaults (about an hour), and a folder in another    #
#          layout refused in one line.                                                             #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import pytest

from mrsuite import externals
from mrsuite.bridges import oryx
from tests.conftest import needs_octave

K_01 = externals.path("oryx-mrsi") / "Dataset" / "K_01"


#************#
#   layout   #
#************#
def test_a_folder_in_another_layout_is_refused_in_one_line(tmp_path):
    with pytest.raises(ValueError, match="not in Oryx's layout"):
        oryx.maps(str(tmp_path), str(tmp_path / "out"))


#**********#
#   maps   #
#**********#
@needs_octave
@pytest.mark.skipif(not K_01.is_dir(), reason="oryx-mrsi submodule not checked out")
def test_k01_to_its_metabolite_maps(tmp_path):
    """The voxels Oryx keeps of K_01 under Octave, with its GUI's defaults."""
    maps = oryx.maps(str(K_01), str(tmp_path))

    assert list(maps.included.values()) == [240, 214, 237, 181, 8, 88, 14, 101, 247]
    assert set(maps.files["conc_maps"]) == set(oryx.METABOLITES)
    assert set(maps.files["conc_maps_to_tCr_ratio"]) == set(oryx.METABOLITES) - {"Cr+PCr"}
    assert {"fgm_maps", "fwm_maps", "fcsf_maps", "CRLB_maps", "SNR_maps", "FWHM_maps"} <= \
        set(maps.files)
    entry = maps.provenance[0]
    assert entry["Program"] == "Oryx-MRSI" and entry["Version"].endswith("0b6ae5e")
    assert '"spm12"' in entry["Details"]
