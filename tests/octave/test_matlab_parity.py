####################################################################################################
#                                      test_matlab_parity.py                                       #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-08-08                                                                              #
#                                                                                                  #
# Purpose: MRSuite's Octave shims against the MATLAB functions they replace. The shims are         #
#          independent implementations, not reproductions, so the question here is whether         #
#          the substitution changes the numbers.                                                   #
#                                                                                                  #
#          MATLAB is optional: without it the Octave half still runs against a known               #
#          analytic answer. Put `matlab` on PATH to run the comparison proper.                     #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from tests.conftest import needs_matlab, needs_octave
from tests.octave.conftest import run_matlab, run_octave

# abs() on the width throughout: the model squares p(2), so its sign is unidentifiable
# and either root is an exact optimum. Comparing the raw value would flag a sign flip as
# a parity failure when the two fits are identical.
LORENTZ = """
    d = load(DATAFILE);
    ppm = d(:,1); y = d(:,2);
    L = @(p,x) p(1).*(p(2)^2)./((x-p(3)).^2 + p(2)^2) + p(4);
"""

NLINFIT_CASE = LORENTZ + """
    o = statset(statset('nlinfit'), 'MaxIter', 1e4, 'TolX', 1e-10, 'TolFun', 1e-10);
    [b, R, J, CovB, MSE] = nlinfit(ppm, y, L, [1, 1/pi, 0.0, 0], o);
    se = sqrt(diag(CovB));
    out = struct('beta', [b(1) abs(b(2)) b(3) b(4)], 'mse', MSE, 'se', se(:)');
"""

# unbounded, and driven through optimoptions the way Osprey's multiplex fit does
LSQNONLIN_CASE = LORENTZ + """
    o = optimoptions(optimoptions(@lsqnonlin), 'Display', 'off', ...
                     'Algorithm', 'levenberg-marquardt');
    [x, resnorm] = lsqnonlin(@(p) L(p, ppm) - y, [1, 1/pi, 0.0, 0], [], [], o);
    out = struct('beta', [x(1) abs(x(2)) x(3) x(4)], 'resnorm', resnorm);
"""

# bounded, mirroring osp_CrChoReferencing's actual lsqcurvefit call shape
LSQCURVEFIT_CASE = LORENTZ + """
    o = optimset(optimset('lsqnonlin'), 'Display', 'off', 'Algorithm', 'levenberg-marquardt');
    lo = [1 0 -1 -1]; hi = [10 2 1 1];
    [x, resnorm] = lsqcurvefit(L, [1, 1/pi, 0.0, 0], ppm, y, lo, hi, o);
    out = struct('beta', [x(1) abs(x(2)) x(3) x(4)], 'resnorm', resnorm, ...
                 'inbounds', all(x >= lo - 1e-9) && all(x <= hi + 1e-9));
"""

# deterministic input, so no seeded RNG has to agree across the two engines. The shapes
# cover every branch the shim has: even and odd length, several columns, a row vector,
# and an explicit n that both pads and truncates.
HILBERT_CASE = """
    t = (0:399)'/400;
    x = sin(2*pi*3*t) + 0.5*cos(2*pi*7*t) - 0.2*t.^2;
    even = hilbert(x);
    odd  = hilbert(x(1:399));
    mat  = hilbert([x, flipud(x)]);
    row  = hilbert(x(1:16)');
    padd = hilbert(x(1:16), 32);
    trun = hilbert(x, 100);
    out = struct('even_re', real(even)', 'even_im', imag(even)', ...
                 'odd_im', imag(odd)', 'mat_im', imag(mat(:,2))', ...
                 'row_im', imag(row), 'row_isrow', isrow(row), ...
                 'pad_im', imag(padd)', 'trun_im', imag(trun)');
"""

# headers and outright junk, which MATLAB does report as errors and the shim must leave
# alone; and clean numeric lines with trailing whitespace, which it must clear
SSCANF_CASE = """
    lines = {' $SEQPAR$', ' 997.  997.  997.,', ' 1.5  2.5  3.5', 'not a number', ...
             sprintf(' 0.61690E-01  0.15881E+01\\n'), ...
             sprintf(' 1.5  2.5   \\n'), ' 1.5  2.5 junk'};
    counts = zeros(1, numel(lines));
    errs   = zeros(1, numel(lines));
    for k = 1:numel(lines)
        [~, c, e] = sscanf(lines{k}, '%f', inf);
        counts(k) = c;
        errs(k)   = ~isempty(e);
    end
    out = struct('counts', counts, 'anyerr', errs);
"""


#**************#
#   relative   #
#**************#
def _worst(a, b):
    """Largest relative deviation between two parameter vectors."""
    return max(abs(x - y) / max(abs(y), 1e-12) for x, y in zip(a, b))


#***********#
#   truth   #
#***********#
@needs_octave
def test_octave_shim_recovers_truth(lorentz_data):
    """Runs with or without MATLAB: the shim must find the right answer regardless."""
    got = run_octave(NLINFIT_CASE, lorentz_data)

    for val, truth, tol in zip(got["beta"], [5, 0.12, 0.35, 0.2],
                               [0.05, 0.005, 0.005, 0.02]):
        assert abs(val - truth) < tol, f"{val} vs {truth}"
    assert 0 < got["mse"] < 1e-3


#*************#
#   nlinfit   #
#*************#
@needs_matlab
@needs_octave
def test_nlinfit_shim_matches_matlab(lorentz_data):
    """The parity check proper: same case, both engines, compared elementwise."""
    oct_out = run_octave(NLINFIT_CASE, lorentz_data)
    mat_out = run_matlab(NLINFIT_CASE, lorentz_data)

    # measured agreement on identical input is ~1e-8 relative; the gate sits just above
    # that, so a real regression trips it rather than being absorbed by slack
    worst = _worst(oct_out["beta"], mat_out["beta"])
    assert worst < 1e-6, (f"nlinfit shim diverges from MATLAB by {worst:.2e} relative\n"
                          f"  octave: {oct_out['beta']}\n  matlab: {mat_out['beta']}")

    # MSE feeds CRLB-style error bars, and matches to ~1e-11 on identical input
    rel = abs(oct_out["mse"] - mat_out["mse"]) / max(abs(mat_out["mse"]), 1e-12)
    assert rel < 1e-8, f"MSE differs by {rel:.2e}: {oct_out['mse']} vs {mat_out['mse']}"


#***************#
#   lsqnonlin   #
#***************#
@needs_matlab
@needs_octave
def test_lsqnonlin_shim_matches_matlab(lorentz_data):
    oct_out = run_octave(LSQNONLIN_CASE, lorentz_data)
    mat_out = run_matlab(LSQNONLIN_CASE, lorentz_data)

    worst = _worst(oct_out["beta"], mat_out["beta"])
    assert worst < 1e-5, (f"lsqnonlin diverges by {worst:.2e}\n"
                          f"  octave: {oct_out['beta']}\n  matlab: {mat_out['beta']}")

    rel = abs(oct_out["resnorm"] - mat_out["resnorm"]) / max(abs(mat_out["resnorm"]), 1e-12)
    assert rel < 1e-6, f"resnorm differs by {rel:.2e}"


#*****************#
#   lsqcurvefit   #
#*****************#
@needs_matlab
@needs_octave
def test_lsqcurvefit_shim_matches_matlab_with_bounds(lorentz_data):
    """Bounds are the part most likely to diverge, so they are exercised explicitly."""
    oct_out = run_octave(LSQCURVEFIT_CASE, lorentz_data)
    mat_out = run_matlab(LSQCURVEFIT_CASE, lorentz_data)

    assert oct_out["inbounds"] and mat_out["inbounds"], "a solver left the box"
    worst = _worst(oct_out["beta"], mat_out["beta"])
    assert worst < 1e-5, (f"lsqcurvefit diverges by {worst:.2e}\n"
                          f"  octave: {oct_out['beta']}\n  matlab: {mat_out['beta']}")


#*************#
#   hilbert   #
#*************#
@needs_matlab
@needs_octave
def test_hilbert_shim_matches_matlab():
    """The shim reimplements a Signal Processing Toolbox function from its spec, so
    every shape it can be handed is compared elementwise against the real thing.
    """
    oct_out = run_octave(HILBERT_CASE)
    mat_out = run_matlab(HILBERT_CASE)

    assert oct_out["row_isrow"] == mat_out["row_isrow"], "row/column orientation differs"
    for key in ("even_re", "even_im", "odd_im", "mat_im", "row_im", "pad_im", "trun_im"):
        worst = max(abs(o - m) for o, m in zip(oct_out[key], mat_out[key]))
        assert worst < 1e-12, f"hilbert '{key}' diverges from MATLAB by {worst:.2e}"


#************#
#   sscanf   #
#************#
@needs_matlab
@needs_octave
def test_sscanf_shim_matches_matlab_on_partial_scans():
    """Adjudicates the one shim that shadows a built-in.

    It clears errmsg only when the scan stopped on trailing whitespace, which is where
    the two engines genuinely disagree: Octave flags a trailing newline as a failed
    conversion, MATLAB does not, and fgets supplies that newline on every line.
    """
    oct_out = run_octave(SSCANF_CASE)
    mat_out = run_matlab(SSCANF_CASE)

    assert oct_out["counts"] == mat_out["counts"], (
        f"conversion counts differ — octave {oct_out['counts']} vs matlab {mat_out['counts']}")
    assert oct_out["anyerr"] == mat_out["anyerr"], (
        "errmsg behaviour differs from MATLAB; the sscanf shim's premise is wrong\n"
        f"  octave errmsg set: {oct_out['anyerr']}\n"
        f"  matlab errmsg set: {mat_out['anyerr']}")


#***************#
#   toolboxes   #
#***************#
@needs_matlab
def test_matlab_has_the_required_toolboxes():
    """Fails loudly if the install is missing what Osprey needs, rather than later."""
    out = run_matlab("""
        out = struct('optim',  license('test','Optimization_Toolbox'), ...
                     'stats',  license('test','Statistics_Toolbox'), ...
                     'signal', license('test','Signal_Toolbox'));
    """)

    assert out["stats"], "Statistics and Machine Learning Toolbox missing (nlinfit/statset)"
    assert out["optim"], "Optimization Toolbox missing (lsqnonlin/lsqcurvefit)"
