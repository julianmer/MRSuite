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
from pathlib import Path

import numpy as np
import pytest

from fsl_mrs.utils.misc import calculateAxes

from mrsuite.simulators import mrs_sim
from mrsuite.training import recipe

pytestmark = pytest.mark.skipif(not (mrs_sim.ROOT / "src").exists(),
                                reason="MRS-Sim submodule not checked out")

BASIS_2016 = (Path(__file__).resolve().parents[2] / "data" / "2016_fitting_challenge"
              / "basisset_LCModel" / "press3T_30ms.BASIS")


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
    assert {"phi0", "snr", "NAA.d"} <= set(truth.parameters)


#****************#
#   amplitudes   #
#****************#
def test_the_concentrations_are_the_amplitudes_in_the_data():
    """At the first point, with phases and eddy currents off, each basis function's
    signal is its concentration times its basis FID: no per-spectrum normaliser left."""
    config = mrs_sim.config()
    config.update(phi0=False, phi1=False, eddy=False, b0=False)
    _, truth = mrs_sim.simulate(config, n=3, seed=0)
    basis = mrs_sim.basis(config)

    for name in ("NAA", "Cr", "Glu"):
        in_data = (truth.components[name][:, 0] / basis[name][0]).real
        np.testing.assert_allclose(in_data, truth.concentrations[name], rtol=1e-3)


#***************#
#   noiseless   #
#***************#
def test_noise_off_gives_the_noiseless_spectra():
    config = mrs_sim.config()
    config["noise"] = False
    data, truth = mrs_sim.simulate(config, n=2, seed=0)
    fids = data.numpy().reshape(len(data), -1)

    assert np.abs(truth.components["noise"]).max() < 1e-5 * np.abs(fids).max()


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


#**************#
#   compiled   #
#**************#
@pytest.mark.skipif(not BASIS_2016.exists(), reason="challenge submodule not checked out")
def test_a_dot_basis_compiles_to_mrs_sims_own_mat(tmp_path):
    """A .BASIS becomes the basis .mat MRS-Sim reads: the same names as complex FIDs cut
    to the data's grid, in NIfTI-MRS' orientation (NAA at 2.01 ppm), header included."""
    mat = tmp_path / "challenge.mat"
    mrs_sim.compile_basis(BASIS_2016, mat, te=30.0, centre=4.65, points=2048)

    fids = mrs_sim.basis({"PM_basis_set": str(mat), "metabolites": ["NAA", "Cr"]})
    assert set(fids) == {"NAA", "Cr"}
    naa = fids["NAA"]
    assert naa.shape == (2048,) and np.iscomplexobj(naa)

    _, header = mrs_sim._basis_mat({"PM_basis_set": str(mat)})
    assert int(header["Ns"]) == 2048 and float(header["centerFreq"]) == 4.65
    axis = (np.fft.fftshift(np.fft.fftfreq(2048, 1 / float(header["spectralwidth"])))
            / float(header["carrier_frequency"]) + 4.65)
    peak = axis[np.argmax(np.abs(np.fft.fftshift(np.fft.fft(naa))))]
    assert peak == pytest.approx(2.01, abs=0.02)

    with pytest.raises(KeyError, match="has no"):
        mrs_sim.basis({"PM_basis_set": str(mat), "metabolites": ["Nope"]})
    with pytest.raises(ValueError, match="never padded"):
        mrs_sim.compile_basis(BASIS_2016, tmp_path / "no.mat", te=30.0, points=1 << 20)
    with pytest.raises(ValueError, match="mirrored"):
        mrs_sim.compile_basis(BASIS_2016.parents[1] / "basisset_JMRUI", tmp_path / "no.mat",
                              te=30.0)


#************#
#   recipe   #
#************#
def test_a_source_states_its_recipe(tmp_path):
    """The recipe is the filled MRSsynMRS report - the ranges MRS-Sim really sampled -
    plus the config and the seed, and it renders as the model card's table."""
    config = mrs_sim.config()
    source = mrs_sim.Source(config, chunk=2)

    filled = source.recipe()
    report = filled["MRSsynMRS"]
    assert list(report) == ["Experiment", "Software", "Pulse Sequence", "Metabolites",
                            "Components", "Artifacts", "Data Quality"]
    assert report["Pulse Sequence"]["Number of Points"] == source.acquisition["points"]
    ranges = report["Metabolites"]["Concentration Ranges"]
    assert set(ranges) == set(config["metabolites"])
    assert ranges["NAA"] == config["parameters"]["naa"]        # what was set is what ran
    assert "fixed" in ranges["Cr"] and "fixed" in ranges["PCr"]
    assert report["Artifacts"]["Residual Water"]["Applied"] is True
    snr = filled["generation"]["sampled_ranges"]["snr"]
    assert len(snr) == 2 and snr[0] < snr[1]
    assert filled["generation"]["seed"] == 0 and filled["generation"]["basis"]["sha256"]

    assert set(config["metabolites"]) <= set(source.basis)
    assert len(source.basis["NAA"]) == source.acquisition["points"]

    text = recipe.table(filled)
    assert "| **Data Quality** |" in text and "NAA" in text
    path = recipe.save(filled, tmp_path / "recipe.json")
    assert json.loads(Path(path).read_text())["generation"]["seed"] == 0


#*******************#
#   distributions   #
#*******************#
def test_simulations_follow_the_distributions_fitted_to_data(tmp_path):
    """MRS-Sim's own pipeline, fed values as data would give them: its fitted marginals
    and correlations come back out of the simulation, and a parameter asked to be
    independent is."""
    from scipy import io

    rng = np.random.default_rng(0)
    n = 300
    shift = rng.normal(0, 0.02, n)
    samples = {"NAA": rng.lognormal(0.5, 0.1, n), "NAA.d": rng.gamma(20, 0.5, n),
               "snr": 100 + 400 * shift + rng.normal(0, 3, n), "f_shift": shift}

    copula = mrs_sim.distributions(samples, tmp_path / "fits", independent=("NAA",))
    corr = io.loadmat(copula["correlation"], simplify_cells=True)
    names = [str(x) for x in corr["names"]]
    naa = names.index("NAA_ampl")
    assert np.allclose(np.delete(corr["corr"][naa], naa), 0)

    config = mrs_sim.config()
    config["copula"] = copula
    _, truth = mrs_sim.simulate(config, n=64, seed=0, components=False)
    p = truth.parameters
    for key in ("snr", "NAA.d"):
        assert abs(np.mean(p[key]) - np.mean(samples[key])) < 0.6 * np.std(samples[key])
    assert np.corrcoef(p["f_shift"], p["snr"])[0, 1] > 0.8


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
