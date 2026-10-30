####################################################################################################
#                                           test_bases.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 26/10/26                                                                                #
#                                                                                                  #
# Purpose: Basis sets simulated by FID-A, spant and Vespa come as one Basis in NIfTI-MRS'          #
#          orientation, NAA at 2.01 ppm, saying how they were made.                                #
#                                                                                                  #
####################################################################################################
import importlib.util
import shutil

import numpy as np
import pytest

from mrsuite import externals, r
from mrsuite.bridges import fid_a
from mrsuite.simulators import bases, pygamma_build
from tests.conftest import needs_octave

needs_vespa = pytest.mark.skipif(importlib.util.find_spec("vespa") is None,
                                 reason="Vespa not installed")
needs_pygamma = pytest.mark.skipif(
    not (pygamma_build.folder() / "pygamma").is_dir()
    and not all(shutil.which(tool) for tool in pygamma_build.TOOLS),
    reason="PyGAMMA not built, and make, g++ or SWIG missing to build it")


def _peak(basis, name):
    return float(basis.ppm()[np.argmax(np.abs(basis.spectra()[basis.names.index(name)]))])


#***********#
#   fid a   #
#***********#
@needs_octave
def test_fid_a_simulates_a_press_basis_in_niftis_orientation():
    basis = bases.fid_a(["NAA", "Cr"], seq="press", te1=11.0, te2=19.0, points=1024)

    assert basis.names == ["NAA", "Cr"] and basis.fids.shape == (2, 1024)
    assert _peak(basis, "NAA") == pytest.approx(2.01, abs=0.02)
    assert _peak(basis, "Cr") == pytest.approx(3.03, abs=0.02)
    assert basis.te == 30.0 and basis.seq == "press" and basis.bandwidth == 2000.0
    assert basis.source["simulated"]["function"] == "sim_lcmrawbasis"
    with pytest.raises(ValueError, match="seq must be"):
        bases.fid_a(["NAA"], seq="mega")


#***********#
#   spant   #
#***********#
@pytest.mark.skipif(not r.available(), reason="Rscript not on PATH")
def test_spant_simulates_its_brain_basis_through_an_ideal_sequence():
    basis = bases.spant(pul_seq="seq_slaser_ideal", points=1024, TE1=0.008, TE2=0.011,
                        TE3=0.009)

    assert "NAA" in basis.names and "MM09" in basis.names and basis.fids.shape[1] == 1024
    assert _peak(basis, "NAA") == pytest.approx(2.01, abs=0.02)
    assert basis.seq == "seq_slaser_ideal" and basis.centre == 4.65
    assert basis.source["simulated"]["arguments"]["TE3"] == 0.009


@needs_octave
def test_simntg_gives_the_triglyceride_and_its_systems():
    basis = bases.simntg(18, (6, 9), points=512, bandwidth=2000.0)

    assert basis.names[-1] == "TG18:2" and basis.names[:2] == ["FAdb", "FAdiallylic"]
    assert basis.fids.shape == (len(basis.names), 512)
    assert np.allclose(basis.fids[:-1].sum(axis=0), basis.fids[-1], rtol=1e-3, atol=1e-6 * np.abs(basis.fids).max()), "the sum is the whole"
    assert "simntg" not in str(fid_a.bridge.octave.eval("which('round')", nout=1)), \
        "the round shim is SimnTG's alone: off the path once it ran"


#***********#
#   vespa   #
#***********#
@needs_vespa
@needs_pygamma
def test_vespa_simulates_a_press_basis_in_niftis_orientation():
    basis = bases.vespa(["n-acetylaspartate", "creatine"], "PRESS Ideal", (15.0, 15.0),
                        frequency=127.8, points=2048, bandwidth=4000.0)

    assert basis.names == ["n-acetylaspartate", "creatine"] and basis.fids.shape == (2, 2048)
    assert _peak(basis, "n-acetylaspartate") == pytest.approx(2.01, abs=0.02)
    assert _peak(basis, "creatine") == pytest.approx(3.03, abs=0.02)
    assert basis.te == 30.0 and basis.seq == "PRESS Ideal" and basis.centre == 4.7
    simulated = basis.source["simulated"]
    assert simulated["program"] == "Vespa Simulation" and simulated["version"]
    assert simulated["engine"] == {"program": "PyGAMMA", "version": "4.3.4",
                                   "commit": externals.REGISTRY["pygamma"][1]}
    assert simulated["pulse_sequence"]["name"] == "PRESS Ideal"
    assert simulated["arguments"]["timings"] == {"TE1 [ms]": 15.0, "TE2 [ms]": 15.0}


@needs_vespa
def test_vespa_refuses_what_its_database_does_not_hold():
    with pytest.raises(ValueError, match="no pulse sequence 'MEGA'"):
        bases.vespa(["creatine"], "MEGA")
    with pytest.raises(ValueError, match=r"\['NAA'\]"):
        bases.vespa(["NAA"])
    with pytest.raises(ValueError, match=r"Spin-Echo takes a timing per loop, \['TE \[ms\]'\], not 2"):
        bases.vespa(["creatine"], "Spin-Echo", (15.0, 15.0))


#*************#
#   pygamma   #
#*************#
def test_building_pygamma_without_its_tools_says_which_in_one_line(tmp_path, monkeypatch):
    monkeypatch.setattr(pygamma_build, "folder", lambda: tmp_path / "unbuilt")
    monkeypatch.setattr(pygamma_build.shutil, "which", lambda tool: None)

    with pytest.raises(RuntimeError, match=r"^building PyGAMMA needs make, g\+\+, swig on PATH"):
        pygamma_build.ensure()
