####################################################################################################
#                                          test_shims.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-08-08                                                                              #
#                                                                                                  #
# Purpose: The Octave shims standing in for MATLAB-only functions Osprey calls. These run          #
#          Octave directly rather than through oct2py, since what matters is that Octave           #
#          itself resolves and executes the .m files.                                              #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import numpy as np
import pytest

from mrsuite.octave import shim_path
from tests.conftest import needs_octave
from tests.octave.conftest import run_octave

pytestmark = needs_octave

# the shape Osprey's frequency referencing actually fits, from the guess it actually uses
LORENTZIAN = """
    ppm = linspace(-2,2,400)';
    L = @(p,x) p(1).*(p(2)^2)./((x-p(3)).^2 + p(2)^2) + p(4);
    truth = [5, 0.12, 0.35, 0.2];
"""


#*************#
#   statset   #
#*************#
def test_statset_defaults_and_overrides():
    out = run_octave("""
        o = statset('nlinfit');
        o2 = statset(o, 'MaxIter', 1e4, 'TolX', 1e-10);
        out = struct('maxiter', o.MaxIter, 'maxiter2', o2.MaxIter, ...
                     'tolx2', o2.TolX, 'tolfun2', o2.TolFun);
    """)

    assert out["maxiter"] == 200                    # default preserved
    assert out["maxiter2"] == 10000                 # override applied
    assert out["tolx2"] == 1e-10
    assert out["tolfun2"] == 1e-8                   # untouched field survives


#************#
#   linear   #
#************#
def test_nlinfit_recovers_linear_coefficients():
    out = run_octave("""
        X = (0:0.1:5)'; truth = [2.5; -1.3];
        y = truth(1)*X + truth(2);
        b = nlinfit(X, y, @(p,x) p(1)*x + p(2), [0;0]);
        out = struct('err', max(abs(b - truth)));
    """)

    assert out["err"] < 1e-8


#*****************#
#   exponential   #
#*****************#
def test_nlinfit_recovers_exponential_decay():
    out = run_octave("""
        t = (0:0.05:4)'; truth = [3.0; 0.8];
        y = truth(1)*exp(-truth(2)*t);
        b = nlinfit(t, y, @(p,x) p(1)*exp(-p(2)*x), [1;0.1]);
        out = struct('err', max(abs(b - truth)));
    """)

    assert out["err"] < 1e-6


#****************#
#   lorentzian   #
#****************#
def test_nlinfit_fits_a_lorentzian_exactly():
    """Compared on |p2|, not p2: the model uses p(2)^2, so the width's sign is
    unidentifiable and both signs are exact optima. See the degeneracy test below.
    """
    out = run_octave(LORENTZIAN + """
        y = L(truth, ppm);
        o = statset(statset('nlinfit'), 'MaxIter', 1e4, 'TolX', 1e-10, 'TolFun', 1e-10);
        [b, R] = nlinfit(ppm, y, L, [1, 1/pi, 0.0, 0], o);
        out = struct('resid', norm(R), 'row', isequal(size(b), [1 4]), ...
                     'err', max(abs([b(1) abs(b(2)) b(3) b(4)] - truth)));
    """)

    assert out["resid"] < 1e-10                     # exact fit
    assert out["err"] < 1e-6                        # parameters recovered
    assert out["row"], "beta should keep beta0's orientation"


#****************#
#   degeneracy   #
#****************#
def test_lorentzian_width_sign_is_degenerate():
    """Guards a real hazard rather than a shim defect.

    Osprey computes refFWHM = 2*param(2)*dppm from the raw fitted value. Because the
    model squares p(2), a fit is free to return the negative root: identical residual,
    but it would flip the sign of the reported linewidth. Anything consuming this fit
    has to take abs() on the width.
    """
    out = run_octave(LORENTZIAN + """
        out = struct('diff', norm(L(truth, ppm) - L([5, -0.12, 0.35, 0.2], ppm)));
    """)

    assert out["diff"] < 1e-12


#*************#
#   hilbert   #
#*************#
def test_hilbert_builds_the_analytic_signal():
    """Checked against the closed form rather than another implementation.

    For a frequency sitting exactly on the FFT grid, the analytic signal of a cosine is
    cos + i*sin, so the shim can be held to an exact answer with no reference engine.
    """
    out = run_octave("""
        n = 256; t = (0:n-1)'/n;
        y = hilbert(cos(2*pi*8*t));
        out = struct('re', max(abs(real(y) - cos(2*pi*8*t))), ...
                     'im', max(abs(imag(y) - sin(2*pi*8*t))), ...
                     'row', isrow(hilbert(t')));
    """)

    assert out["re"] < 1e-12, "the real part must be the input, untouched"
    assert out["im"] < 1e-12
    assert out["row"], "a row vector must come back as a row"


#****************#
#   covariance   #
#****************#
def test_nlinfit_covariance_is_usable_under_noise():
    out = run_octave(LORENTZIAN + """
        randn('seed', 42);
        y = L(truth, ppm) + 0.02*randn(size(ppm));
        o = statset(statset('nlinfit'), 'MaxIter', 1e4, 'TolX', 1e-10, 'TolFun', 1e-10);
        [b, R, J, CovB, MSE] = nlinfit(ppm, y, L, [1, 1/pi, 0.0, 0], o);
        se = sqrt(diag(CovB));
        out = struct('centre', b(3), 'se', se(3), 'mse', MSE, ...
                     'sepos', all(se > 0), 'rshape', isequal(size(R), size(y)));
    """)

    assert abs(out["centre"] - 0.35) < 4 * out["se"]    # within four standard errors
    assert 0 < out["mse"] < 1e-2                        # close to the 4e-4 noise variance
    assert out["sepos"] and out["rshape"]


#**************#
#   fftshift   #
#**************#
def test_fftshift_is_octaves_own_bit_for_bit():
    """The shim exists for speed, so it is held to Octave's own fftshift exactly, in every
    call form Osprey uses: a matrix along dim 1, both axes, row and column vectors, 3-D."""
    out = run_octave(f"""
        shims = '{shim_path()}';
        m = complex(reshape(1:4001*5, 4001, 5), -reshape(1:4001*5, 4001, 5) / 3);
        x = reshape(1:2*3*5, 2, 3, 5); v = 1:7;
        cases = {{ {{m, 1}}, {{m, 2}}, {{m}}, {{m(:, 1)}}, {{v}}, {{v.'}}, ...
                  {{x, 1}}, {{x, 3}}, {{x}} }};
        same = true;
        for c = 1:numel(cases)
            a = fftshift(cases{{c}}{{:}});
            rmpath(shims); b = fftshift(cases{{c}}{{:}}); addpath(shims);
            same = same && isequal(a, b) && isequal(size(a), size(b)) ...
                   && strcmp(class(a), class(b)) && iscomplex(a) == iscomplex(b);
        end
        out = struct('same', same, 'shim', strcmp(fileparts(which('fftshift')), shims));
    """)

    assert out["shim"], "the shim must be the fftshift Octave resolves"
    assert out["same"]


#*********************#
#   fftshift errors   #
#*********************#
def test_fftshift_refuses_what_octaves_own_refuses():
    """Same errors, so a call Octave would stop on is not quietly run by the shim."""
    out = run_octave(f"""
        shims = '{shim_path()}';
        cases = {{ {{{{1, 2, 3}}}}, {{struct('a', 1)}}, {{magic(4), 0}}, {{magic(4), 1.5}}, ...
                  {{magic(4), [1 2]}}, {{magic(4), 3}}, {{1:5, 3}} }};
        same = true;
        for c = 1:numel(cases)
            a = ''; b = '';
            try, fftshift(cases{{c}}{{:}}); catch err, a = err.message; end
            rmpath(shims);
            try, fftshift(cases{{c}}{{:}}); catch err, b = err.message; end
            addpath(shims);
            same = same && ~isempty(b) && strcmp(a, b);
        end
        out = struct('same', same);
    """)

    assert out["same"]


#**************#
#   padarray   #
#**************#
# (Octave arguments, numpy's pad widths per dimension, constant): op_zeropad's form first
PADS = [("m, 3, 'post'", [(0, 3), (0, 0)], 0), ("m, [1 2]", [(1, 1), (2, 2)], 0),
        ("m, [2 1], 'pre'", [(2, 0), (1, 0)], 0), ("m, [1 1], 5, 'post'", [(0, 1), (0, 1)], 5),
        ("m, [1 1], 'post', 5", [(0, 1), (0, 1)], 5), ("m, [1 0 1]", [(1, 1), (0, 0), (1, 1)], 0),
        ("x, [0 0 2]", [(0, 0), (0, 0), (2, 2)], 0),
        ("v, [0 2], 'zeros', 'pre'", [(0, 0), (2, 0)], 0), ("m, 0", [(0, 0), (0, 0)], 0)]


def test_padarray_pads_as_numpy_does():
    """Held to numpy's constant pad in every call form, with no reference engine."""
    out = run_octave(f"""
        m = complex(reshape(1:12, 4, 3), -reshape(12:-1:1, 4, 3));
        x = reshape(1:12, 2, 3, 2); v = 1:4;
        calls = {{ {', '.join('@() padarray(' + call + ')' for call, _, _ in PADS)} }};
        for c = 1:numel(calls)
            b = calls{{c}}();
            got{{c}} = struct('size', size(b), 're', real(b(:)).', 'im', imag(b(:)).');
        end
        out = struct('got', {{got}}, 'int', strcmp(class(padarray(int16(v), 1)), 'int16'), ...
                     'shim', strcmp(fileparts(which('padarray')), '{shim_path()}'));
    """)
    m = (np.arange(1, 13).reshape(4, 3, order="F")
         - 1j * np.arange(12, 0, -1).reshape(4, 3, order="F"))
    inputs = {"m": m, "x": np.arange(1, 13).reshape(2, 3, 2, order="F"),
              "v": np.arange(1, 5).reshape(1, 4)}

    assert out["shim"], "the shim must be the padarray Octave resolves"
    assert out["int"], "the class is kept"
    for (call, widths, value), got in zip(PADS, out["got"]):
        a = inputs[call[0]]
        a = a.reshape(a.shape + (1,) * (len(widths) - a.ndim))
        want = np.pad(a, widths, constant_values=value)
        assert list(want.shape) == got["size"], call
        assert np.array_equal(np.ravel(want, order="F"),
                              np.array(got["re"]) + 1j * np.array(got["im"])), call


def test_padarray_is_the_image_packages_own_where_it_is_installed():
    """Where Octave's image package is installed, the shim gives what its padarray gives,
    bit for bit, and refuses what it cannot do instead of padding wrongly."""
    out = run_octave(f"""
        shims = '{shim_path()}';
        out = struct('have', true);
        try, pkg load image; catch, out.have = false; end
        if out.have
            m = complex(reshape(1:4001*2, 4001, 2), 1 ./ reshape(1:4001*2, 4001, 2));
            cases = {{ {{m, 4001, 'post'}}, {{m, [3 2]}}, {{m, [1 0 2], -1, 'pre'}}, ...
                      {{int8(magic(3)), [1 1], 'both', 7}}, {{single(1:5), [0 3], 'zeros'}} }};
            same = true;
            for c = 1:numel(cases)
                a = padarray(cases{{c}}{{:}});
                rmpath(shims); b = padarray(cases{{c}}{{:}}); addpath(shims);
                same = same && isequal(a, b) && isequal(size(a), size(b)) ...
                       && strcmp(class(a), class(b)) && iscomplex(a) == iscomplex(b);
            end
            refused = false;
            try, padarray(m, 2, 'symmetric'); catch, refused = true; end
            out = struct('have', true, 'same', same, 'refused', refused);
        end
    """)
    if not out["have"]:
        pytest.skip("Octave's image package is not installed here")

    assert out["same"]
    assert out["refused"], "a pattern pad is refused, not done as a constant one"
