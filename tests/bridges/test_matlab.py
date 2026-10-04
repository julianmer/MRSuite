####################################################################################################
#                                          test_matlab.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-27                                                                              #
#                                                                                                  #
# Purpose: MATLAB as a runtime for the Octave bridges. The bridges were written against oct2py,    #
#          so what matters is that values cross the engine the way oct2py carries them, and that   #
#          a toolbox call run in MATLAB comes back as the same kind of result.                     #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import json

import numpy as np
import pytest

from mrsuite.octave import matlab
from mrsuite.datasets import challenge2016

pytestmark = pytest.mark.skipif(not matlab.available(),
                                reason="MATLAB Engine for Python not installed")


#*************#
#   session   #
#*************#
@pytest.fixture(scope="module")
def engine():
    from mrsuite.bridges.fid_a import FIDA_ROOT
    session = matlab.MatlabSession(str(FIDA_ROOT))
    yield session
    session.exit()


#*****************#
#   round trip    #
#*****************#
def test_values_cross_the_engine_as_oct2py_carries_them(engine):
    value = {"specs": np.arange(6.0).reshape(3, 2) + 1j, "row": np.arange(4.0),
             "names": ("A", "B"), "n": 3, "flags": {"on": 1}}
    engine.push("v", value)
    back = engine.pull("v")

    assert np.array_equal(back["specs"], value["specs"])
    assert back["row"].shape == (1, 4)                  # a 1-D array arrives as a row
    assert list(back["names"]) == ["A", "B"]            # a tuple is a cell
    assert back["n"] == 3.0 and back["flags"] == {"on": 1.0}


#************#
#   fid-a    #
#************#
def test_a_fid_a_step_runs_in_matlab(spectrum):
    from mrsuite.bridges.fid_a import FIDABridge

    bridge = FIDABridge(runtime="matlab")
    out = bridge.op_addphase(spectrum, 30.0)
    step = out.hdr_ext["ProcessingApplied"][-1]

    assert np.allclose(np.abs(out[:]), np.abs(spectrum[:]))
    assert not np.allclose(out[:], spectrum[:])
    assert json.loads(step["Details"])["interpreter"].startswith("MATLAB")
    bridge.octave.exit()


#***********#
#   chain   #
#***********#
def test_osprey_steps_chain_in_matlab(spectrum):
    from tests.bridges.test_osprey import assert_a_chain_keeps_the_history
    assert_a_chain_keeps_the_history("matlab", spectrum)


#*************#
#   osprey    #
#*************#
@pytest.mark.skipif(not challenge2016.BASIS.exists(), reason="challenge submodule not checked out")
def test_osprey_fits_in_matlab():
    from mrsuite.bridges import osprey

    bridge, osprey.bridge = osprey.bridge, osprey.OspreyBridge(runtime="matlab")
    try:
        data, _, truth = challenge2016.load(1)
        result = osprey.fit(data, str(challenge2016.BASIS))[0]
    finally:
        osprey.bridge.octave.exit()
        osprey.bridge = bridge

    assert {"NAA", "Cr", "Glu"} <= set(result.conc("raw"))
    assert json.loads(result.provenance[-1]["Details"])["interpreter"].startswith("MATLAB")
