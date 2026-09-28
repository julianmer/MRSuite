####################################################################################################
#                                         test_mrs_sim.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-28                                                                              #
#                                                                                                  #
# Purpose: Simulating with MRS-Sim: the truth adds up to the data, a seed replays it, the spectra  #
#          are NIfTI-MRS in its own orientation with their provenance, and a training source       #
#          hands out batches across its chunks. Skipped without the MRS-Sim submodule.            #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import json

import numpy as np
import pytest

from fsl_mrs.utils.misc import calculateAxes

from mrsuite.simulators import mrs_sim

pytestmark = pytest.mark.skipif(not (mrs_sim.ROOT / "src").exists(),
                                reason="MRS-Sim submodule not checked out")


#***************#
#   simulated   #
#***************#
@pytest.fixture(scope="module")
def simulated():
    """Three spectra from MRS-Sim's current config, every effect on."""
    return mrs_sim.simulate(mrs_sim.config(), n=3, seed=0)


#************#
#   adds up   #
#************#
def test_the_truth_adds_up_to_the_data(simulated):
    """Each basis function, the baseline, the residual water and the noise, each taken
    through every step the data was, are the data."""
    data, truth = simulated
    fids = data.numpy().reshape(len(data), -1)

    total = sum(truth.components.values())

    assert np.allclose(total, fids, atol=1e-6 * np.abs(fids).max())

    # the noise holds no signal: a line the truth missed would show up in it
    signal = total - truth.components["noise"]
    noise = truth.components["noise"]
    for i in np.flatnonzero(truth.parameters["snr"] < 30):
        corr = abs(np.vdot(noise[i], signal[i])) / (np.linalg.norm(noise[i]) *
                                                     np.linalg.norm(signal[i]))
        assert corr < 0.05, f"spectrum {i}: the noise correlates {corr:.2f} with the signal"
    assert {"baseline", "water", "noise"} <= set(truth.components)
    assert set(truth.concentrations) <= set(truth.components)     # a signal for each
    assert set(truth.concentrations) <= set(mrs_sim.config()["metabolites"])
    assert {"phi0", "snr", "d:NAA"} <= set(truth.parameters)


#************#
#   replay   #
#************#
def test_a_seed_replays_the_spectra():
    config = mrs_sim.config()
    first, _ = mrs_sim.simulate(config, n=2, seed=7, components=False)
    again, _ = mrs_sim.simulate(config, n=2, seed=7, components=False)
    other, _ = mrs_sim.simulate(config, n=2, seed=8, components=False)

    assert np.array_equal(first.numpy(), again.numpy())
    assert not np.allclose(first.numpy(), other.numpy())


#*****************#
#   orientation   #
#*****************#
def test_the_spectra_are_nifti_mrs_in_its_orientation():
    """With every shift and phase off, NAA's singlet sits at 2.01 ppm on NIfTI-MRS' own
    axis - mirrored data would put it at 7.3."""
    config = mrs_sim.config()
    config.update(b0=False, eddy=False, phi0=False, phi1=False, fshift_g=False,
                  fshift_i=False)
    data, truth = mrs_sim.simulate(config, n=1, seed=0)
    item = data.list()[0]

    spectrum = np.fft.fftshift(np.fft.fft(truth.components["NAA"][0]))
    axis = calculateAxes(1 / item.dwelltime, item.spectrometer_frequency[0], item.shape[3],
                         item.SpecFreqChemShift)["ppmshift"]
    assert axis[np.argmax(np.abs(spectrum))] == pytest.approx(2.01, abs=0.02)
    step = item.hdr_ext["ProcessingApplied"][-1]
    assert step["Program"] == "MRS-Sim" and step["Version"] == mrs_sim.version()
    assert json.loads(step["Details"])["seed"] == 0


#***********#
#   coils   #
#***********#
def test_several_coils_are_refused_with_the_reason():
    config = mrs_sim.config()
    config["num_coils"] = 8

    with pytest.raises(ValueError, match="several coils"):
        mrs_sim.simulate(config, n=1)


#************#
#   source   #
#************#
def test_a_source_hands_out_batches_across_its_chunks():
    source = mrs_sim.Source(mrs_sim.config(), chunk=4, components=True)

    first, truth = source.batch(3)
    second, _ = source.batch(3)                    # more than the chunk has left

    points = source.acquisition["points"]
    assert first.numpy().shape == second.numpy().shape == (3, 1, 1, 1, points)
    assert not np.allclose(first.numpy(), second.numpy())
    assert set(source.acquisition) == {"points", "bandwidth", "frequency", "centre"}
    assert len(truth.components["NAA"]) == 3
