####################################################################################################
#                                           test_bases.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 26/10/26                                                                                #
#                                                                                                  #
# Purpose: Basis sets simulated by FID-A and spant come as one Basis in NIfTI-MRS' orientation,    #
#          NAA at 2.01 ppm, saying how they were made.                                             #
#                                                                                                  #
####################################################################################################
import numpy as np
import pytest

from mrsuite import r
from mrsuite.bridges import fid_a
from mrsuite.simulators import bases
from tests.conftest import needs_octave


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


def test_simntg_gives_the_triglyceride_and_its_systems():
    basis = bases.simntg(18, (6, 9), points=512, bandwidth=2000.0)

    assert basis.names[-1] == "TG18:2" and basis.names[:2] == ["FAdb", "FAdiallylic"]
    assert basis.fids.shape == (len(basis.names), 512)
    assert np.allclose(basis.fids[:-1].sum(axis=0), basis.fids[-1], rtol=1e-3, atol=1e-6 * np.abs(basis.fids).max()), "the sum is the whole"
    assert "simntg" not in str(fid_a.bridge.octave.eval("which('round')", nout=1)), \
        "the round shim is SimnTG's alone: off the path once it ran"
