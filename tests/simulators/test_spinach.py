####################################################################################################
#                                         test_spinach.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 27/10/26                                                                                #
#                                                                                                  #
# Purpose: Spinach run as published under Octave: spin systems through its pulse-acquire in the    #
#          full Liouville space, their lines where their shifts and couplings put them, the        #
#          roof of a strongly coupled pair, and the restricted bases Octave cannot run refused.    #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import numpy as np
import pytest

from mrsuite.simulators import spinach
from tests.conftest import needs_octave

pytestmark = needs_octave

LACTATE = {"shifts": [4.0974, 1.3142, 1.3142, 1.3142],
           "couplings": {(0, 1): 6.933, (0, 2): 6.933, (0, 3): 6.933}}


#***********#
#   peaks   #
#***********#
def peaks(basis, k: int, n: int, zerofill: int = 8):
    """The n tallest local maxima of function k's real spectrum, zero-filled, in ppm."""
    points = basis.fids.shape[1] * zerofill
    spectrum = np.fft.fftshift(np.fft.fft(basis.fids[k], points)).real
    hz = np.fft.fftshift(np.fft.fftfreq(points, 1 / basis.bandwidth))
    ppm = basis.centre + hz / basis.frequency
    top = [i for i in range(1, points - 1)
           if spectrum[i] > spectrum[i - 1] and spectrum[i] >= spectrum[i + 1]]
    top = sorted(top, key=lambda i: spectrum[i])[-n:]
    return ppm[sorted(top)], spectrum[sorted(top)]


#***********#
#   lines   #
#***********#
@pytest.fixture(scope="module")
def basis():
    return spinach.simulate({"singlet": {"shifts": [2.0]}, "Lac": LACTATE,
                             "AB": {"shifts": [1.0, 1.05], "couplings": {(0, 1): 7.0}}},
                            points=4096, linewidth=1.0)


def test_lines_sit_where_shifts_and_couplings_put_them(basis):
    (singlet,), _ = peaks(basis, 0, 1)
    doublet, _ = peaks(basis, 1, 2)

    assert basis.names == ["singlet", "Lac", "AB"] and basis.fids.shape == (3, 4096)
    assert basis.frequency == pytest.approx(127.73, abs=0.01), "1H at 3 T"
    assert singlet == pytest.approx(2.0, abs=0.002), "not mirrored about the centre"
    assert doublet.mean() == pytest.approx(1.3142, abs=0.002)
    assert np.diff(doublet)[0] * basis.frequency == pytest.approx(6.933, abs=0.2)
    assert basis.spectra()[0].real.max() > 10 * abs(basis.spectra()[0].real.min()), \
        "in absorption"


def test_a_strongly_coupled_pair_leans_towards_itself(basis):
    """Two protons 6.4 Hz apart and 7 Hz coupled: four lines, the inner two tallest."""
    where, height = peaks(basis, 2, 4)

    assert len(where) == 4
    assert min(height[1], height[2]) > 2 * max(height[0], height[3])


def test_the_simulation_says_where_it_came_from(basis):
    simulated = basis.source["simulated"]

    assert simulated["program"] == "Spinach" and simulated["version"].endswith("4138eb5")
    assert simulated["arguments"]["systems"]["Lac"]["couplings"][0] == [0, 1, 6.933]
    assert simulated["runtime"]["interpreter"].startswith(("Octave", "MATLAB"))


#****************#
#   restricted   #
#****************#
def test_a_restricted_basis_is_refused_under_octave():
    if spinach.toolbox.matlab:
        pytest.skip("MATLAB runs Spinach's restricted bases")
    with pytest.raises(ValueError, match="under Octave only 'none' runs"):
        spinach.simulate({"singlet": {"shifts": [2.0]}}, approximation="IK-2")
