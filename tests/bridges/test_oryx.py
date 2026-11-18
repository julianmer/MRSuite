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
#          layout refused in one line; and the same voxels kept in MATLAB.                         #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import shutil
import textwrap

import pytest

from mrsuite import externals
from mrsuite.bridges import oryx
from tests.conftest import needs_matlab, needs_octave

# the voxels Oryx keeps of K_01 per metabolite, with its GUI's defaults, under Octave
INCLUDED = [240, 214, 237, 181, 8, 88, 14, 101, 247]

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
@pytest.mark.skipif(not K_01.is_dir(), reason="oryx-mrsi not fetched")
def test_k01_to_its_metabolite_maps(tmp_path):
    """The voxels Oryx keeps of K_01 under Octave, with its GUI's defaults."""
    maps = oryx.maps(str(K_01), str(tmp_path))

    assert list(maps.included.values()) == INCLUDED
    assert set(maps.files["conc_maps"]) == set(oryx.METABOLITES)
    assert set(maps.files["conc_maps_to_tCr_ratio"]) == set(oryx.METABOLITES) - {"Cr+PCr"}
    assert {"fgm_maps", "fwm_maps", "fcsf_maps", "CRLB_maps", "SNR_maps", "FWHM_maps"} <= \
        set(maps.files)
    entry = maps.provenance[0]
    assert entry["Program"] == "Oryx-MRSI" and entry["Version"].endswith("0b6ae5e")
    assert '"spm12"' in entry["Details"]


#************#
#   matlab   #
#************#
@needs_matlab
@pytest.mark.skipif(not K_01.is_dir(), reason="oryx-mrsi not fetched")
def test_k01_keeps_the_same_voxels_in_matlab(tmp_path):
    """Oryx's GUI's calls in MATLAB as they are, on a copy of K_01, with none of MRSuite's
    code on the path. SPM12 only reads and writes the images and nothing is drawn at random,
    so the voxels kept are those of the bridge under Octave."""
    from tests.octave.conftest import COMPUTER, run_matlab, spm12_for_matlab

    copy = tmp_path / "K_01"
    shutil.copytree(K_01, copy)
    spectra = copy / "exam_1" / "spectra"
    (tmp_path / "computer.m").write_text(textwrap.dedent(COMPUTER).strip() + "\n")
    # its segmentation reports progress on the GUI's figure, which MATLAB draws only in a
    # visible window; headless, a struct takes the values it sets
    (tmp_path / "uiprogressdlg.m").write_text(
        "function d = uiprogressdlg(varargin)\nd = struct('Value', 0);\nend\n")
    calls = oryx.GUI.format(
        spar=sorted(spectra.glob("*_act.spar"))[0].name, spectra=spectra,
        names="{" + " ".join(f"'{m}'" for m in oryx.METABOLITES) + "}",
        ppm="[" + " ".join(str(p) for p in oryx.METABOLITES.values()) + "]",
        fcsf=0.3, snr=8.0, fwhm=0.1, crlb=30.0)
    try:
        got = run_matlab(f"addpath(genpath('{oryx.toolbox.root}')); "
                         f"addpath('{spm12_for_matlab()}'); addpath('{tmp_path}'); cd('{copy}'); "
                         f"set(0, 'DefaultFigureVisible', 'off');\n" + textwrap.dedent(calls)
                         + "\nout = struct('included', mrsuite_included(:)');",
                         timeout=3 * 3600)
    finally:
        shutil.rmtree(copy)                    # some 4 GB, which pytest would keep

    assert got["included"] == INCLUDED
