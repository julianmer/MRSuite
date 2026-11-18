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
#          ds005371 through load and fit, and what MRSuite makes of Gannet's output; and the       #
#          same in MATLAB, the tissue step on random draws alike in both.                          #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import re
import shutil
import textwrap

import pytest

from mrsuite import externals
from mrsuite.bridges import gannet
from tests.conftest import needs_matlab, needs_octave

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
@pytest.mark.skipif(not MEGA.is_dir(), reason="osprey not fetched")
def test_mega_press_from_load_to_tissue_corrected_quantification(tmp_path):
    """The values Gannet 3.5.4 gives on this example under Octave, from its own drivers."""
    mrs = MEGA / "mrs"
    result = gannet.fit(str(mrs / "sub-01_ses-01_megapress" / "sub-01_megapress_act.sdat"),
                        str(mrs / "sub-01_ses-01_megapress-ref" / "sub-01_megapress_ref.sdat"),
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


#************#
#   matlab   #
#************#
# Gannet on its own, as its documentation runs it, with none of MRSuite's code on the path:
# the files in a folder of their own, where a GannetPreInitialise.m edited as a user edits
# it stands in front of Gannet's
NATIVE = """
    addpath(genpath(G)); addpath(W); cd(W);
    set(0, 'DefaultFigureVisible', 'off');
    MRS = GannetFit(GannetLoad({A}, {B}));
    o = MRS.out.vox1; out = struct('water_fwhm', o.water.FWHM);
    f = fieldnames(o);
    for k = 1:numel(f)
        if all(isfield(o.(f{k}), {'ConcIU', 'ConcCr', 'FitError'}))
            out.(f{k}) = [o.(f{k}).ConcIU, o.(f{k}).ConcCr, o.(f{k}).FitError];
        end
    end
"""
# and on to tissue-corrected quantification with a T1, SPM12 on the path
TISSUE = """
    addpath(S);
    MRS = GannetQuantify(GannetSegment(GannetCoRegister(MRS, {T})));
    o = MRS.out.vox1;
    out.tissue = [o.tissue.fGM, o.tissue.fWM, o.tissue.fCSF];
    out.GABA_tissue_corrected = o.GABA.ConcIU_TissCorr;
"""


def native(folder, metabolite, water, edits=(), t1=None, spm=None):
    """Gannet's load and fit in MATLAB on copies of two acquisitions and what goes with
    them: per metabolite its water-scaled value, its ratio to creatine and its fit error;
    with a T1 and an SPM12, also the voxel's tissue fractions and GABA corrected for them."""
    from tests.octave.conftest import COMPUTER, run_matlab

    for path in (metabolite, water, t1):
        for file in path.parent.glob(path.name.split(".")[0] + ".*") if path else ():
            shutil.copy(file, folder)
    settings = (externals.path("gannet") / "GannetPreInitialise.m").read_text()
    for field, value in edits:
        settings, done = re.subn(rf"(MRS_struct\.p\.{field}\s*=)[^;]*;", rf"\1 {value};", settings)
        assert done == 1, field
    (folder / "GannetPreInitialise.m").write_text(settings)
    body = textwrap.dedent(NATIVE)
    if t1:
        (folder / "computer.m").write_text(textwrap.dedent(COMPUTER).strip() + "\n")
        body += textwrap.dedent(TISSUE)
    return run_matlab(f"G = '{externals.path('gannet')}'; W = '{folder}'; A = '{metabolite.name}'; "
                      f"B = '{water.name}'; T = '{t1.name if t1 else ''}'; S = '{spm}';\n" + body)


@needs_matlab
@pytest.mark.skipif(not MEGA.is_dir(), reason="osprey not fetched")
def test_mega_press_gives_the_same_in_matlab(tmp_path):
    """Gannet in MATLAB, the interpreter it is written for (with the four toolboxes it asks
    for), against the values the bridge gives under Octave."""
    mrs = MEGA / "mrs"
    got = native(tmp_path, mrs / "sub-01_ses-01_megapress" / "sub-01_megapress_act.sdat",
                 mrs / "sub-01_ses-01_megapress-ref" / "sub-01_megapress_ref.sdat")

    assert got["GABA"] == pytest.approx((2.0930, 0.1191, 5.550), abs=5e-4)
    assert got["Glx"][0] == pytest.approx(5.1764, abs=5e-4)
    assert got["water_fwhm"] == pytest.approx(9.277, abs=5e-3)


@needs_matlab
@pytest.mark.skipif(not HERMES.with_name(HERMES.name + "_svs.nii.gz").exists(),
                    reason="ds005371 sub-01 not fetched from the annex")
def test_hermes_gives_the_same_in_matlab(tmp_path):
    got = native(tmp_path, HERMES.with_name(HERMES.name + "_svs.nii.gz"),
                 HERMES.with_name(HERMES.name + "_mrsref.nii.gz"),
                 edits=(("HERMES", "1"), ("target", "{'GABAGlx', 'GSH'}")))

    # MATLAB's nlinfit stops at its iteration limit on these fits and MRSuite's under Octave
    # stops elsewhere: GABA 2.370 here against 2.332, inside its fit error of 8 %; Glx and
    # GSH within half a percent
    assert got["GABA"][:2] == pytest.approx((2.332, 0.185), rel=0.02)
    assert got["GSH"][0] == pytest.approx(0.552, rel=0.005)
    assert got["Glx"][0] == pytest.approx(4.780, rel=0.005)


#************#
#   tissue   #
#************#
# SPM12's segmentation draws random numbers, to dither an integer image and to split its
# Gaussians, from rand('state', 0) and randn('state', 0); Octave's generators give other
# numbers for that seed than MATLAB's. So the tissue fractions differ by what was drawn:
# 0.6234 / 0.2813 / 0.0953 in MATLAB against the bridge's 0.6104 / 0.2710 / 0.1186 under
# Octave. On the same draws they agree: stand-ins for rand and randn, alike in both, in the
# folder Gannet runs from (the golden-ratio sequence, normal by Box-Muller, started over by
# any seed)
RAND = """
    function r = rand(varargin)
    persistent k
    if isempty(k), k = 0; end
    if nargin && ischar(varargin{1}), k = 0; r = []; return; end
    dims = varargin(cellfun(@isnumeric, varargin)); dims = [dims{:}];
    if isempty(dims), dims = 1; end
    if isscalar(dims), dims = [dims dims]; end
    r = reshape(mod((k + (1:prod(dims))') * 0.6180339887498949, 1), dims);
    k = k + prod(dims);
    end
"""
RANDN = """
    function r = randn(varargin)
    persistent k
    if isempty(k), k = 0; end
    if nargin && ischar(varargin{1}), k = 0; r = []; return; end
    dims = varargin(cellfun(@isnumeric, varargin)); dims = [dims{:}];
    if isempty(dims), dims = 1; end
    if isscalar(dims), dims = [dims dims]; end
    i = k + (1:prod(dims))';
    u = mod(i * 0.6180339887498949, 1); v = mod(i * 0.4142135623730951, 1);
    r = reshape(sqrt(-2 * log(u)) .* cos(2 * pi * v), dims);
    k = k + prod(dims);
    end
"""
# what the bridge gives under Octave on those draws: fGM, fWM, fCSF and GABA corrected
SAME_DRAWS = (0.6209, 0.2779, 0.1012), 3.6367


def same_draws(folder):
    for name, text in (("rand.m", RAND), ("randn.m", RANDN)):
        (folder / name).write_text(textwrap.dedent(text).strip() + "\n")


@needs_octave
@pytest.mark.skipif(not MEGA.is_dir(), reason="osprey not fetched")
def test_mega_press_tissue_on_draws_alike_in_both(tmp_path):
    mrs = MEGA / "mrs"
    same_draws(tmp_path)
    result = gannet.fit(str(mrs / "sub-01_ses-01_megapress" / "sub-01_megapress_act.sdat"),
                        str(mrs / "sub-01_ses-01_megapress-ref" / "sub-01_megapress_ref.sdat"),
                        t1=str(MEGA / "anat" / "sub-01" / "sub-01_ses-01_T1w.nii.gz"),
                        output=str(tmp_path))

    assert (result.qc["fGM"], result.qc["fWM"], result.qc["fCSF"]) == \
        pytest.approx(SAME_DRAWS[0], abs=1e-4)
    assert result.conc("water_tissue_corrected")["GABA"] == pytest.approx(SAME_DRAWS[1],
                                                                         abs=1e-4)


@needs_matlab
@pytest.mark.skipif(not MEGA.is_dir(), reason="osprey not fetched")
def test_mega_press_tissue_gives_the_same_in_matlab(tmp_path):
    """Load to tissue-corrected quantification in MATLAB, on the draws the bridge was given
    under Octave."""
    from tests.octave.conftest import spm12_for_matlab

    mrs = MEGA / "mrs"
    same_draws(tmp_path)
    got = native(tmp_path, mrs / "sub-01_ses-01_megapress" / "sub-01_megapress_act.sdat",
                 mrs / "sub-01_ses-01_megapress-ref" / "sub-01_megapress_ref.sdat",
                 t1=MEGA / "anat" / "sub-01" / "sub-01_ses-01_T1w.nii.gz",
                 spm=spm12_for_matlab())

    # a few in ten thousand apart: the arithmetic, not the draws
    assert got["tissue"] == pytest.approx(SAME_DRAWS[0], abs=1e-3)
    assert got["GABA_tissue_corrected"] == pytest.approx(SAME_DRAWS[1], rel=1e-3)
