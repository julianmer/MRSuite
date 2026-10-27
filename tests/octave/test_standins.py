####################################################################################################
#                                         test_standins.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 27/10/26                                                                                #
#                                                                                                  #
# Purpose: The shims standing in for the MATLAB functions and classes Gannet, Oryx-MRSI,           #
#          MRS4Brain and Spinach call and Octave lacks, held to what MATLAB's documentation says   #
#          they do: the general ones on the shim path, the scoped ones from their toolbox's        #
#          folder, each case one Octave run.                                                       #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import os

import pytest

from mrsuite.octave import shim_path
from tests.conftest import needs_octave
from tests.octave.conftest import run_octave

pytestmark = needs_octave


#***********#
#   scope   #
#***********#
def scope(name: str) -> str:
    """Octave code putting a toolbox's scoped shims in front, as its session does."""
    return f"addpath('{os.path.join(shim_path(), name)}');\n"


#**********#
#   text   #
#**********#
def test_text_functions_answer_as_matlabs():
    out = run_octave("""
        out = struct('n', strlength('abc'), 'ns', strlength({'a', 'bcd'}), ...
                     'right', pad('ab', 5), 'left', pad('ab', 5, 'left'), ...
                     'both', pad('ab', 6, 'both'), 'longer', pad('abcdef', 3), ...
                     'cells', {pad({'a', 'abc'})}, 'number', string(3.14159), ...
                     'text', string('kept'), 'finite', allfinite([1 2]), ...
                     'infinite', allfinite([1 Inf]), 'nan', allfinite([NaN 1]), ...
                     'sparse', allfinite(sparse([0 1])));
    """)

    assert out["n"] == 3 and out["ns"] == [1, 3]
    assert (out["right"], out["left"], out["both"]) == ("ab   ", "   ab", "  ab  ")
    assert out["longer"] == "abcdef", "longer text is left as it is"
    assert out["cells"] == ["a  ", "abc"], "a cell array padded to its longest entry"
    assert out["number"] == "3.1416" and out["text"] == "kept"
    assert out["finite"] and not out["infinite"] and not out["nan"] and out["sparse"]


#**********#
#   time   #
#**********#
def test_datetime_and_duration_answer_as_matlabs():
    out = run_octave("""
        epoch = datetime(0, 'ConvertFrom', 'posixtime');
        t = datetime('2024-03-06 15:36:17.000', 'InputFormat', 'yyyy-MM-dd HH:mm:ss.SSS');
        later = t + days(2);
        out = struct('year', year(epoch), 'hour', hour(t), 'minute', minute(t), ...
                     'apart', days(later - t), 'hours', hours(duration(1, 30, 0)), ...
                     'longer', (later - t) > days(1), 'shorter', duration(23, 0, 0) < days(1), ...
                     'plus', datenum(t + 1) - datenum(t), 'posix', posixtime(epoch), ...
                     'text', char(datetime('2024-03-06', 'InputFormat', 'yyyy-MM-dd', ...
                                           'Format', 'y-MM-dd')), ...
                     'order', t < later);
    """)

    assert out["year"] == 1970 and out["posix"] == 0
    assert (out["hour"], out["minute"]) == (15, 36)
    assert out["apart"] == pytest.approx(2) and out["hours"] == pytest.approx(1.5)
    assert out["longer"] and out["shorter"] and out["order"]
    assert out["plus"] == pytest.approx(1), "a number added to a datetime is days"
    assert out["text"] == "2024-03-06"


#************#
#   tables   #
#************#
def test_table_cell2table_and_dictionary_answer_as_matlabs():
    out = run_octave("""
        T = table([1; 2; 3], {'a'; 'b'; 'c'}, 'VariableNames', {'n', 's'});
        T.n(2) = 20;
        T(1, :) = [];
        C = cell2table({1, 'x'; 2, 'y'});
        d = dictionary({'a', 'b'}, [1 2]);
        out = struct('height', height(T), 'width', width(T), 'n', T.n, 's', {T.s}, ...
                     'names', {T.Properties.VariableNames}, 'var1', C.Var1, ...
                     'var2', {C.Var2}, 'b', d('b'), 'has', isKey(d, 'a'), ...
                     'fallback', lookup(d, 'c', 'FallbackValue', 0), ...
                     'stacked', height([T; T]));
    """)

    assert (out["height"], out["width"], out["stacked"]) == (2, 2, 4)
    assert out["n"] == [20, 3] and out["s"] == ["b", "c"]
    assert out["names"] == ["n", "s"]
    assert out["var1"] == [1, 2], "a column of numbers becomes a numeric variable"
    assert out["var2"] == ["x", "y"], "a column of text stays a cell array"
    assert out["b"] == 2 and out["has"] and out["fallback"] == 0


#**************#
#   releases   #
#**************#
def test_release_errors_and_progress_answer_as_matlabs():
    out = run_octave("""
        try, error('mine:id', 'went wrong'); catch err, end
        d = uiprogressdlg([], 'Title', 'Segmentation', 'Message', 'wait');
        d.Value = 0.5;
        out = struct('older2025a', isMATLABReleaseOlderThan('R2025a'), ...
                     'older2024a', isMATLABReleaseOlderThan('R2024a'), ...
                     'basic', getReport(err, 'basic'), 'extended', getReport(err), ...
                     'cores', feature('numcores'), 'pid', feature('getpid') == getpid(), ...
                     'title', d.Title, 'value', d.Value, 'map', size(parula(4)));
    """)

    assert out["older2025a"] and not out["older2024a"], "Octave answers as R2024b"
    assert out["basic"] == "went wrong" and out["extended"].startswith("went wrong")
    assert out["cores"] >= 1 and out["pid"]
    assert (out["title"], out["value"]) == ("Segmentation", 0.5)
    assert out["map"] == [4, 3]


#************#
#   gannet   #
#************#
def test_gannet_sees_matlab_toolboxes_and_handles(tmp_path):
    image = tmp_path / "dot.png"
    out = run_octave(scope("gannet") + f"""
        imwrite(uint8(255 * eye(4)), '{image}');
        [img, map, alpha] = imread('{image}', 'BackgroundColor', 'none');
        v = ver();
        f = figure('visible', 'off');
        h = subplot(2, 1, 1);
        h.Position = [0.1 0.5 0.8 0.4];
        out = struct('image', size(img), 'names', {{{{v.Name}}}}, ...
                     'parent', isequal(h.Parent, f), 'position', get(h, 'pos'), ...
                     'handle', double(h) == gca());
        close(f);
    """)

    assert out["image"] == [4, 4], "the PNG read as it is, the option dropped"
    assert {"Image Processing Toolbox", "Optimization Toolbox", "Signal Processing Toolbox",
            "Statistics and Machine Learning Toolbox"} <= set(out["names"])
    assert out["parent"] and out["handle"]
    assert out["position"] == pytest.approx([0.1, 0.5, 0.8, 0.4]), "'pos' is Position"


#**********#
#   oryx   #
#**********#
def test_oryx_textread_skips_blank_lines_as_matlabs(tmp_path):
    listing = tmp_path / "names.txt"
    listing.write_text("NAA\n\nCr  PCr\n\n")
    out = run_octave(scope("oryx") + f"""
        words = textread('{listing}', '%s');
        out = struct('words', {{words}});
    """)

    assert out["words"] == ["NAA", "Cr", "PCr"]


#***************#
#   mrs4brain   #
#***************#
def test_mrs4brain_strings_numbers_and_files_answer_as_matlabs(tmp_path):
    saved = tmp_path / "saved.mat"
    out = run_octave(scope("mrs4brain") + f"""
        line = 'Signal' + newline;
        x = 3;
        save('{saved}', 'x');
        fid = fopen('{saved}', 'r'); head = fread(fid, 6, 'char=>char')'; fclose(fid);
        [~, said] = system('echo /home/lcmodel/.lcmodel/bin/lcmodel');
        name = matlab.lang.makeValidName('1st');
        out = struct('line', char(line), 'is', strcmp({{'#', '$', newline}}, char(10)), ...
                     'digits', round(1.2345, 2), 'tens', round(1234, -2), ...
                     'significant', round(1234.5, 2, 'significant'), 'plain', round(2.5), ...
                     'computer', computer(), 'head', head, 'said', strtrim(said), ...
                     'name', name{{1}});
    """)

    assert out["line"] == "Signal\n", "a string plus newline is text, not numbers"
    assert out["is"] == [False, False, True]
    assert out["digits"] == pytest.approx(1.23) and out["tens"] == 1200
    assert out["significant"] == 1200 and out["plain"] == 3
    assert out["computer"] in ("GLNXA64", "MACI64", "MACA64", "PCWIN64")
    assert out["head"] == "MATLAB", "a MAT file by default"
    assert out["said"] == "./lcmodel", "the developers' LCModel path is the configured one"
    assert out["name"] == "x1st"


#*************#
#   spinach   #
#*************#
def test_spinach_reductions_hashes_and_pool_answer_as_matlabs():
    out = run_octave(scope("spinach") + """
        A = [1 5; 7 2];
        [m, i] = max(A, [], 'all');
        [n, j] = min(A, [], 'all');
        pool = parpool();
        f = parfeval(@(a, b) a + b, 1, 2, 3);
        q = parallel.pool.DataQueue; got = [];
        afterEach(q, @(x) disp(x)); send(q, 1);
        c = parallel.pool.Constant(@() 42);
        out = struct('sum', sum(A, 'all'), 'max', [m i], 'min', [n j], ...
                     'any', any(A > 6, 'all'), 'all', all(A > 0, 'all'), ...
                     'column', sum(A), 'md5', sprintf('%02x', digestMD5(uint8('abc'))), ...
                     'workers', pool.NumWorkers, 'same', isequal(gcp('nocreate'), pool), ...
                     'fetched', fetchOutputs(f), 'constant', c.Value, ...
                     'parallel', isfolder(fullfile(matlabroot(), 'toolbox', 'parallel')), ...
                     'worker', parallel.internal.pool.isPoolWorker(), ...
                     'older', isMATLABReleaseOlderThan('R2026b'), ...
                     'bytes', isequal(serializeToBytes(A), serializeToBytes(A)), ...
                     'gpus', gpuDeviceCount());
    """)

    assert out["sum"] == 15 and out["column"] == [8, 7]
    assert out["max"] == [7, 2] and out["min"] == [1, 1], "linear indices, as MATLAB's"
    assert out["any"] and out["all"]
    assert out["md5"] == "900150983cd24fb0d6963f7d28e17f72"
    assert out["workers"] == 1 and out["same"] and out["fetched"] == 5
    assert out["constant"] == 42 and out["parallel"] and not out["worker"]
    assert not out["older"], "Spinach's release is the one it requires"
    assert out["bytes"] and out["gpus"] == 0
