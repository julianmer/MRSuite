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
