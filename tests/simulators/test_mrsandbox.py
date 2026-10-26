####################################################################################################
#                                         test_mrsandbox.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 26/10/26                                                                                #
#                                                                                                  #
# Purpose: MRSandbox's virtual scanner through MRSuite: a sequence of its library, a scan as       #
#          NIfTI-MRS+ in its orientation with the drawn concentrations as truth, Pulseq in and out. #
#                                                                                                  #
####################################################################################################
import json
import os

import numpy as np
import pytest

from mrsuite.simulators import mrsandbox

pytest.importorskip("mrsandbox")
os.environ.setdefault("KERAS_BACKEND", "jax")      # MRSandbox's own default; torch lacks eigh on MPS


#**********#
#   scan   #
#**********#
def test_a_scan_gives_spectra_in_niftis_orientation_with_the_drawn_truth():
    data, truth, basis = mrsandbox.scan("press", n=2, seed=3, points=1024, te1_s=0.011,
                                        te2_s=0.019)
    item = data.list()[0]

    assert len(data) == len(truth) == 2 and item.shape == (1, 1, 1, 1024)
    assert basis.fids.shape[1] == 1024 and "NAA" in basis.names and basis.seq == "Press"
    spectrum = np.fft.fftshift(np.fft.fft(np.asarray(item[:]).ravel()))
    hz = np.fft.fftshift(np.fft.fftfreq(1024, item.dwelltime))
    ppm = hz / item.spectrometer_frequency[0] + 4.65
    window = (ppm > 1.8) & (ppm < 2.2)
    assert ppm[window][np.argmax(np.abs(spectrum[window]))] == pytest.approx(2.01, abs=0.03)
    assert np.abs(spectrum[window]).max() > 5 * np.abs(spectrum[(ppm > 7.1) & (ppm < 7.5)]).max()
    assert 5 < truth[0]["NAA"] < 20 and truth[0] != truth[1]
    step = item.hdr_ext["ProcessingApplied"][-1]
    assert step["Program"] == "MRSandbox" and json.loads(step["Details"])["seed"] == 3
    assert "SimulationGroundTruth" in item.hdr_ext and item.hdr_ext["WaterSuppressed"] is True


#*******************#
#   ideal scanner   #
#*******************#
def test_an_ideal_scanner_draws_no_imperfections():
    data, truth, basis = mrsandbox.scan("slaser", n=1, seed=2, points=512, imperfections=False,
                                        mm=False, lipids=False, te_s=0.026)
    drawn = data.list()[0].hdr_ext["SimulationGroundTruth"]["params"]
    assert drawn["acquisition"]["phase_deg"] == 0 and drawn["acquisition"]["shift_hz"] == 0
    assert drawn["spurious"]["amplitude"] == 0 and drawn["oov"]["fraction"] == 0
    assert json.loads(data.list()[0].hdr_ext["ProcessingApplied"][-1]["Details"])["mm"] is False
    assert basis.seq == "SemiLaser" and basis.te == pytest.approx(26.0)


#***********#
#   basis   #
#***********#
def test_the_basis_of_the_same_simulation_sits_on_niftis_axis():
    """The engine renders at the chemical shifts themselves; demodulated by the centre, NAA
    sits at 2.01 ppm and creatine at 3.03, each metabolite present once."""
    basis = mrsandbox.basis("press", points=1024, seed=0, te1_s=0.011, te2_s=0.019)

    assert "NAA" in basis.names and len(basis.names) == len(set(basis.names)) > 10
    assert basis.fids.shape == (len(basis.names), 1024) and basis.centre == 4.65
    for name, ppm in (("NAA", 2.01), ("Cr", 3.03)):
        k = basis.names.index(name)
        assert basis.ppm()[np.argmax(np.abs(basis.spectra()[k]))] == pytest.approx(ppm, abs=0.02)
    assert basis.te == pytest.approx(30.0) and basis.seq == "Press"
    assert basis.frequency == pytest.approx(127.73, abs=0.01)


#*************#
#   diagram   #
#*************#
def test_a_sequence_draws_as_a_diagram():
    from matplotlib.figure import Figure

    figure = mrsandbox.diagram(mrsandbox.sequence("press"))
    assert isinstance(figure, Figure) and len(figure.axes) == 4
    labels = [text.get_text() for ax in figure.axes for text in ax.texts]
    assert "90°" in labels and labels.count("180°") == 2 and "ADC" in labels


#**************#
#   sequence   #
#**************#
def test_sequences_come_from_the_library_and_pulseq_files_round_trip(tmp_path):
    pytest.importorskip("pypulseq")
    slaser = mrsandbox.sequence("slaser", te_s=0.030)
    assert type(slaser).__name__ == "SemiLaser"
    with pytest.raises(ValueError, match="kind must be"):
        mrsandbox.sequence("spiral")

    path = str(tmp_path / "press.seq")
    with pytest.raises(ValueError, match="idealization"):      # no scanner plays an ideal one
        mrsandbox.write_pulseq(mrsandbox.sequence("press"), path)


def test_a_slice_comes_as_one_spatial_nifti_with_its_truth_and_basis():
    data, truth, basis = mrsandbox.mrsi("press", matrix=(4, 3), voxel=10.0, points=256, seed=1,
                                        imperfections=False)
    item = data.list()[0]
    assert len(data) == 1 and item.shape == (4, 3, 1, 256), "the slice, voxels on the grid"
    assert item.hdr_ext["WaterSuppressed"] is True
    step = item.hdr_ext["ProcessingApplied"][-1]
    assert step["Program"] == "MRSandbox" and step["Method"] == "MRSI.acquire"
    assert len(truth) == 1 and truth[0]["NAA"] > 0, "one draw for the subject"
    assert "NAA" in basis.names and basis.fids.shape[1] == 256
