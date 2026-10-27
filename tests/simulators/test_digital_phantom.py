####################################################################################################
#                                     test_digital_phantom.py                                      #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 27/10/26                                                                                #
#                                                                                                  #
# Purpose: A voxel of MRS-Digital-Phantom comes back the right way round, on its basis set's grid, #
#          with the voxel's concentrations as truth and the phantom's call in its history.         #
#                                                                                                  #
####################################################################################################
import importlib.util

import numpy as np
import pytest

from mrsuite.simulators import digital_phantom

needs_phantom = pytest.mark.skipif(
    any(importlib.util.find_spec(m) is None for m in ("torch", "torchio", "fsl_mrs")),
    reason="MRS-Digital-Phantom needs torch, torchio and FSL-MRS")


@needs_phantom
def test_a_phantom_voxel_comes_the_right_way_round_with_its_truth():
    data, truth = digital_phantom.scan("PRESS", 35, size=20.0, seed=0)

    item, = data.list()
    fid = np.asarray(item[:]).ravel()
    ppm = digital_phantom.CENTRE + np.fft.fftshift(np.fft.fftfreq(fid.size, item.dwelltime)) / \
        item.spectrometer_frequency[0]
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(fid)))
    band = lambda centre: spectrum[np.abs(ppm - centre) < 0.05].max()
    assert band(2.01) > 3 * band(2 * digital_phantom.CENTRE - 2.01), "NAA, not its mirror"
    assert item.dwelltime == pytest.approx(1 / 2000) and fid.size == 2048
    assert item.hdr_ext["EchoTime"] == pytest.approx(0.035)
    assert truth[0] and all(c >= 0 for c in truth[0].values())
    step = item.hdr_ext["ProcessingApplied"][-1]
    assert step["Program"] == "MRS-Digital-Phantom" and "LCModel_Universal_Philips" in step["Details"]
