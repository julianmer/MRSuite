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
import os
import textwrap

import pytest

from mrsuite import externals
from mrsuite.bridges import mrs4brain
from tests.conftest import needs_matlab, needs_octave

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


#************#
#   matlab   #
#************#
# MRS4Brain's SVS as its GUI's buttons call it, with none of MRSuite's code on the path. The
# steps set a message and a value on the progress dialog they are handed, so a struct
# stands in for it; and they run LCModel as ./lcmodel in the configured folder only where
# computer() says MACI64, so it is made to
COMPUTER = """
    function varargout = computer(varargin)
    [varargout{1:max(nargout, 1)}] = builtin('computer', varargin{:});
    if nargin == 0, varargout{1} = 'MACI64'; end
    end
"""
NATIVE = """
    addpath(genpath(fullfile(R, 'MRS4BRAIN_toolbox'))); addpath(genpath(F), '-end'); addpath(W);
    x = load(fullfile(R, 'MRS4BRAIN_toolbox', 'Others', 'SVS_config', ...
                      'Demo_SVS_14p1T_SPECIAL.mat'));
    f = fieldnames(x); p = x.(f{1});
    p.LCModel_path = W; p.Basis_set = fullfile(S, '14p1T_SPECIAL_SVS_TE9ms_Demo.BASIS');
    name = '20240306_153617_SPECIAL';
    s = struct('folder', fullfile(S, name), 'expnb', {11, 12}, ...
               'exp_name', {[name '_11'], [name '_12']}, ...
               'ref_bool', {true, false}, 'met_bool', {false, true});
    svs = SVSClass(fullfile(W, 'out'), 'native', s, p);
    steps = {'create_Bruker_rawstudy', 'preprocessing_SVS_FidA', 'SVS_quantification'};
    for k = 1:numel(steps)
        msg = svs.(steps{k})(struct('Message', '', 'Value', 0));
        assert(all(cellfun(@isempty, msg)), steps{k});
    end
    out = struct('names', {cellstr(svs.Met_names)}, 'molarity', svs.abs_conc_tot(:)', ...
                 'crlb', svs.crlb_tot(:)');
"""


@needs_matlab
@pytest.mark.skipif(not SVS.is_dir(), reason="mrs4brain not fetched")
def test_the_svs_demo_gives_the_same_in_matlab(tmp_path):
    """MRS4Brain on its own in MATLAB, the interpreter it is written for, against the values
    the bridge gives under Octave: the same to the third decimal, with MATLAB's own nlinfit
    aligning the averages where Octave has MRSuite's."""
    from lcmodel_wrapper import binaries

    from tests.octave.conftest import run_matlab

    (tmp_path / "computer.m").write_text(textwrap.dedent(COMPUTER).strip() + "\n")
    os.symlink(binaries.resolve_executable(), tmp_path / "lcmodel")
    got = run_matlab(f"R = '{externals.path('mrs4brain')}'; F = '{externals.path('fid-a')}'; "
                     f"W = '{tmp_path}'; S = '{SVS}';\n" + textwrap.dedent(NATIVE))
    mm, crlb = dict(zip(got["names"], got["molarity"])), dict(zip(got["names"], got["crlb"]))

    assert (mm["NAA+NAAG"], mm["Glu+Gln"], mm["Cr+PCr"]) == \
        pytest.approx((12.438, 16.099, 10.584), abs=5e-3)
    assert (mm["GPC+PCho"], mm["Ins"], mm["Tau"]) == pytest.approx((1.482, 11.041, 8.595),
                                                                   abs=5e-3)
    assert (crlb["NAA+NAAG"], crlb["Glu+Gln"], crlb["Cr+PCr"]) == \
        pytest.approx((0.02, 0.03, 0.03))


#**********#
#   mrsi   #
#**********#
@pytest.mark.skipif(not MRSI.is_dir(), reason="mrs4brain not fetched")
def test_an_mrsi_experiment_is_refused_in_one_line():
    with pytest.raises(ValueError, match="MRS4Brain's MRSI does not run under Octave"):
        mrs4brain.quantify(str(MRSI), 20, 21, "basis.BASIS")
