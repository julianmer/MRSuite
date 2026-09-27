####################################################################################################
#                                          test_lbfgsb.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-27                                                                              #
#                                                                                                  #
# Purpose: The lbfgsb shim and its compiled fast path for Osprey's least-squares problems. The     #
#          whole point is speed without a change in results, so they are held to bit identity     #
#          with the lbfgsb.m Osprey ships, in whichever runtime the session uses.                  #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import numpy as np
import pytest

from tests.conftest import needs_octave


#*************#
#   session   #
#*************#
@pytest.fixture(scope="module")
def octave():
    """A session of its own: the tests put Osprey's lbfgsb.m in front on the path."""
    from mrsuite.bridges.osprey import OSPREY_ROOT
    from mrsuite.octave import build_lbfgsb, session

    build_lbfgsb(str(OSPREY_ROOT))
    oc = session(str(OSPREY_ROOT))
    # an Osprey-shaped problem: nearly collinear columns, the lower bounds active
    oc.eval("""
        rand('seed', 5); randn('seed', 5);
        AB = randn(1008, 1) * ones(1, 32) + 0.01 * randn(1008, 32); b = randn(1008, 1);
        A = randn(1008, 1); AtA = AB'*AB; Ab = AB'*b; l = zeros(32, 1); u = inf(32, 1);
        opts = struct('factr', 1e4, 'pgtol', 1e-8, 'm', 10); opts.printEvery = 0;
        opts.x0 = ones(32, 1);
    """)
    yield oc
    oc.exit()


#***********#
#   solve   #
#***********#
def _solve(oc, offset):
    """Solve the problem as Osprey writes it, with or without the offset term."""
    f = "norm( A + AB*x - b)^2" if offset else "norm( AB*x - b)^2"
    oc.eval(f"fcn = @(x) {f}; grad = @(x) 2*( AtA*x - Ab ); "
            f"fun = @(x)fminunc_wrapper( x, fcn, grad); [x, fx] = lbfgsb(fun, l, u, opts);")
    return np.asarray(oc.pull("x")).ravel(), float(oc.pull("fx"))


#***************#
#   identical   #
#***************#
@needs_octave
def test_the_fast_path_is_bit_identical_to_ospreys_own(octave):
    """The fast path first, then Osprey's own lbfgsb.m put back in front on the path."""
    from mrsuite.bridges.osprey import OSPREY_ROOT

    fast = [_solve(octave, offset) for offset in (False, True)]
    octave.eval(f"addpath('{OSPREY_ROOT}/libraries/L-BFGS-B-C/Matlab'); clear lbfgsb;")
    try:
        slow = [_solve(octave, offset) for offset in (False, True)]
    finally:
        # back behind the shim, not off the path: fminunc_wrapper lives there too
        octave.eval(f"rmpath('{OSPREY_ROOT}/libraries/L-BFGS-B-C/Matlab'); "
                    f"addpath('{OSPREY_ROOT}/libraries/L-BFGS-B-C/Matlab', '-end'); clear lbfgsb;")

    for (xf, ff), (xs, fs) in zip(fast, slow):
        assert np.array_equal(xf, xs) and ff == fs


#*****************#
#   calibration   #
#*****************#
@needs_octave
def test_exactly_one_norm_is_this_octaves(octave):
    """The shim only takes the fast path when one of its two norms is Octave's own."""
    octave.eval("""
        k = (1:4000)'; same = [true true];
        for j = 1:40
            v = sin(k * j + j) .* 10 .^ (mod(k * j, 17) - 8); v = v(1:(100 * j));
            for mode = [0 1]
                same(mode + 1) = same(mode + 1) && norm(v) == lbfgsb_quadratic('norm', v, mode);
            end
        end
    """)
    assert int(np.asarray(octave.pull("same")).sum()) == 1


#**************#
#   fallback   #
#**************#
@needs_octave
def test_any_other_problem_still_solves(octave):
    """f and g given separately, as a cell, is not Osprey's shape: the callback path."""
    octave.eval("[x, fx] = lbfgsb({@(x) sum((x - 3).^2), @(x) 2*(x - 3)}, -inf(4, 1), inf(4, 1));")
    assert np.allclose(np.asarray(octave.pull("x")).ravel(), 3, atol=1e-6)
