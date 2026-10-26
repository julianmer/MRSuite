####################################################################################################
#                                         test_synthetic.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 26/10/26                                                                                #
#                                                                                                  #
# Purpose: Spectra synthesised by FSL-MRS and spant come as NIfTI-MRS+ in its orientation with     #
#          their truth, their recipe in the history, the same again from the same seed.            #
#                                                                                                  #
####################################################################################################
import json
from pathlib import Path

import numpy as np
import pytest

from mrsuite import r
from mrsuite.simulators import synthetic

BASIS_2016 = (Path(__file__).resolve().parents[2] / "data" / "2016_fitting_challenge"
              / "basisset_LCModel" / "press3T_30ms.BASIS")


def _naa(item):
    spectrum = np.fft.fftshift(np.fft.fft(np.asarray(item[:]).ravel()))
    hz = np.fft.fftshift(np.fft.fftfreq(item.shape[3], item.dwelltime))
    ppm = hz / item.spectrometer_frequency[0] + item.SpecFreqChemShift
    window = (ppm > 1.8) & (ppm < 2.2)
    return float(ppm[window][np.argmax(np.abs(spectrum[window]))]), \
        float(np.abs(spectrum[window]).max() / np.abs(spectrum[(ppm > 7.1) & (ppm < 7.5)]).max())


#*************#
#   fsl mrs   #
#*************#
def test_fsl_mrs_synthesises_from_a_basis_with_its_concentrations_as_truth():
    pytest.importorskip("fsl_mrs")
    if not BASIS_2016.exists():
        pytest.skip("challenge submodule not checked out")

    data, truth = synthetic.fsl_mrs(str(BASIS_2016), n=2, seed=1)
    again, _ = synthetic.fsl_mrs(str(BASIS_2016), n=2, seed=1)
    item = data.list()[0]

    assert len(data) == len(truth) == 2 and item.shape == (1, 1, 1, 4096)
    peak, over_mirror = _naa(item)
    assert peak == pytest.approx(2.01, abs=0.02) and over_mirror > 5
    assert truth[0]["NAA"] > 0 and set(truth[0]) >= {"NAA", "Cr", "Glu"}
    step = item.hdr_ext["ProcessingApplied"][-1]
    assert step["Program"] == "FSL-MRS" and json.loads(step["Details"])["seed"] == 1
    assert np.allclose(np.asarray(again.list()[1][:]), np.asarray(data.list()[1][:]))


#***********#
#   spant   #
#***********#
@pytest.mark.skipif(not r.available(), reason="Rscript not on PATH")
def test_spant_simulates_brain_spectra_with_its_amplitudes_as_truth():
    data, truth = synthetic.spant(n=1, pul_seq="seq_press_ideal", TE1=0.01, TE2=0.02, seed=3)
    item = data.list()[0]

    assert item.shape == (1, 1, 1, 1024) and item.dwelltime == pytest.approx(5e-4)
    peak, over_mirror = _naa(item)
    assert peak == pytest.approx(2.01, abs=0.02) and over_mirror > 5
    assert truth[0]["NAA"] == 10.0 and truth[0]["Cr"] == 5.0
    step = item.hdr_ext["ProcessingApplied"][-1]
    assert step["Program"] == "spant" and json.loads(step["Details"])["TE2"] == 0.02
