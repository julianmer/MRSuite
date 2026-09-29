####################################################################################################
#                                       test_challenges.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-27                                                                              #
#                                                                                                  #
# Purpose: Both challenge datasets, loaded and fitted by every fitter. The loaders are pinned      #
#          against what the release itself states - the named ground-truth maps, the basis in      #
#          its .mat - and each fit only has to run and report the ground-truth metabolites:        #
#          how well it fits is the benchmark's question, not this one.                             #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import importlib
import json

import numpy as np
import pytest

from mrsuite import r
from mrsuite.core.pipeline import Pipeline
from mrsuite.core.result import FitResult
from mrsuite.datasets import challenge2016, mrsi_challenge
from mrsuite.simulators import mrs_sim

from tests.bridges.test_lcmodel import needs_lcmodel
from tests.conftest import needs_octave

MRSI_SUBJECT = "Sub1"

needs_2016 = pytest.mark.skipif(not challenge2016.BASIS.exists(),
                                reason="challenge submodule not checked out")
# skipped rather than fetched: a subject is a 4 GB download
needs_mrsi = pytest.mark.skipif(
    not (mrsi_challenge.DATA_DIR / MRSI_SUBJECT / f"{MRSI_SUBJECT}_all.mat").exists(),
    reason=f"MRSI challenge {MRSI_SUBJECT} not downloaded")
needs_sim = pytest.mark.skipif(not (mrs_sim.ROOT / "src").exists(),
                               reason="MRS-Sim submodule not checked out")


#***************#
#   ismrm2016   #
#***************#
@pytest.fixture(scope="module")
def ismrm2016():
    """Two datasets, their ground truth, and the challenge's own basis."""
    data, _, truth = challenge2016.load(2)
    return data, truth, str(challenge2016.BASIS)


#**********#
#   mrsi   #
#**********#
@pytest.fixture(scope="module")
def mrsi(tmp_path_factory):
    """Three brain voxels of a training subject, their ground truth, and the basis."""
    data, truth, _ = mrsi_challenge.load(MRSI_SUBJECT, voxels=3)
    path = mrsi_challenge.basis(str(tmp_path_factory.mktemp("basis") / "mrsi.BASIS"),
                                subject=MRSI_SUBJECT)
    return data, truth, path


#*********#
#   fit   #
#*********#
def _fit(fitter, data, path):
    """Fit as a pipeline step, from the one .BASIS file every fitter reads: swapping the
    fitter is swapping one name."""
    module = importlib.import_module(f"mrsuite.bridges.{fitter}")
    return Pipeline([module.step.fit(path)]).run(data)


needs_r = pytest.mark.skipif(not r.available(), reason="Rscript not on PATH")

FITTERS = [pytest.param("fsl_mrs", id="fsl_mrs"),
           pytest.param("lcmodel", id="lcmodel", marks=needs_lcmodel),
           pytest.param("osprey", id="osprey", marks=needs_octave),
           pytest.param("spant", id="spant", marks=needs_r)]


#*************#
#   covered   #
#*************#
def _assert_covers_truth(results, truth, path):
    """Every result reports every ground-truth metabolite the basis can fit, finitely.

    The ground truth also names what no basis function models (the 2016 spreadsheets
    list lipids and sums such as 'Cr + PCr'), so the expectation is its overlap with
    the basis, not the whole of it.
    """
    from fsl_mrs.utils import mrs_io
    names = set(mrs_io.read_basis(path).names)

    assert len(results) == len(truth)
    for result, concs in zip(results, truth):
        expected = set(concs) & names
        assert expected, "the ground truth and the basis share no names"
        assert isinstance(result, FitResult)
        reported = result.conc("raw")
        assert expected <= set(reported), f"{result.method} misses {expected - set(reported)}"
        assert all(np.isfinite(reported[m]) for m in expected)
        fit = json.loads(result.provenance[-1]["Details"])
        basis = fit["args"][0]
        assert isinstance(basis["path"], str) and basis["sha256"], \
            "the fit must record which basis it used"
        origin = result.provenance[0]
        assert origin["Program"] == "MRSuite" and "load" in origin["Method"]
        source = json.loads(origin["Details"])["file"]
        assert isinstance(source["path"], str) and source["sha256"], "and which file the data is"


#*************************#
#   2016 loader pairing   #
#*************************#
@needs_2016
def test_the_2016_loader_pairs_spectra_references_and_truth():
    data, water, truth = challenge2016.load()

    assert len(data) == len(water) == len(truth) == 28
    assert {"NAA", "Cr", "Mac"} <= set(truth[0])


#********************#
#   mrsi truth map   #
#********************#
@needs_mrsi
def test_the_mrsi_truth_is_the_named_map_of_that_voxel():
    """The ground truth comes from the .mat's metaMap; the named NIfTI maps are the
    release's independent statement of which channel is which metabolite, and where."""
    import nibabel as nib

    _, truth, index = mrsi_challenge.load(MRSI_SUBJECT, voxels=5)
    folder = mrsi_challenge.DATA_DIR / MRSI_SUBJECT
    for name in mrsi_challenge.METABOLITES:
        named = np.asarray(nib.load(folder / f"{MRSI_SUBJECT}_mrs_metab_result_{name}.nii.gz")
                           .dataobj).squeeze()
        assert [t[mrsi_challenge.NAMES[name]] for t in truth] == \
            [float(named[tuple(i)]) for i in index], name


#****************#
#   mrsi basis   #
#****************#
@needs_mrsi
def test_the_mrsi_basis_round_trips_to_the_release_basis(tmp_path):
    """Read back as a fitter reads it, the .BASIS holds VtMeta and VtMM from the data's
    first sample on: a conjugation or a row offset would leave a large residual."""
    import h5py
    from fsl_mrs.utils import mrs_io

    back = mrs_io.read_basis(mrsi_challenge.basis(str(tmp_path / "mrsi.BASIS"),
                                                  subject=MRSI_SUBJECT))
    release = {}
    with h5py.File(mrsi_challenge.DATA_DIR / MRSI_SUBJECT / f"{MRSI_SUBJECT}_all.mat") as f:
        for key, names in (("VtMeta", list(mrsi_challenge.NAMES.values())),
                           ("VtMM", mrsi_challenge.MACROMOLECULES)):
            raw = f[key][()]
            for k, name in enumerate(names):
                release[name] = (raw["real"] + 1j * raw["imag"])[2:386, k]

    assert sorted(back.names) == sorted(release)
    for name, want in release.items():
        got = back.original_basis_array[:len(want), back.names.index(name)]
        gain = np.vdot(want, got) / np.vdot(want, want)
        assert np.linalg.norm(got - gain * want) / np.linalg.norm(got) < 1e-4, name


#*********************#
#   simulation grid   #
#*********************#
def _assert_simulates_the_challenge(config, points, bandwidth, frequency, centre):
    """The config simulates on the challenge's own grid, in NIfTI-MRS' orientation, and
    the truth stays within the stated ranges."""
    quiet = {**config, "phi0": False, "phi1": False, "fshift_g": False, "fshift_i": False,
             "b0": False, "eddy": False}
    data, truth = mrs_sim.simulate(quiet, n=2, seed=0)
    item = data.list()[0]

    assert item.shape[3] == points
    assert 1 / item.dwelltime == pytest.approx(bandwidth, rel=1e-6)
    assert item.spectrometer_frequency[0] == pytest.approx(frequency, abs=0.01)
    assert item.SpecFreqChemShift == pytest.approx(centre)
    assert set(truth.concentrations) == set(config["metabolites"])

    # the concentrations are amplitudes in the data; relative to the reference they are
    # the ratios MRS-Sim sampled, fixed for the reference and within range for the rest
    reference = config["wrt_metab"].split(",")
    unit = truth.concentrations[reference[0]] / config["wrt_metab_ratio"][0]
    for name, values in truth.concentrations.items():
        ratios = values / unit
        if name in reference:
            assert np.allclose(ratios, config["wrt_metab_ratio"][reference.index(name)])
        else:
            low, high = config["parameters"][name.lower()]
            assert (low - 1e-6 <= ratios).all() and (ratios <= high + 1e-6).all(), name

    from fsl_mrs.utils.misc import calculateAxes
    spectrum = np.fft.fftshift(np.fft.fft(truth.components["NAA"][0]))
    axis = calculateAxes(1 / item.dwelltime, item.spectrometer_frequency[0], points,
                         item.SpecFreqChemShift)["ppmshift"]
    assert axis[np.argmax(np.abs(spectrum))] == pytest.approx(2.01, abs=0.02)


#*****************#
#   config 2016   #
#*****************#
@needs_2016
@needs_sim
def test_the_2016_config_simulates_on_the_challenge_grid(tmp_path):
    config = challenge2016.mrs_sim_config(folder=tmp_path)

    assert len(config["metabolites"]) == 21 and "NAA" in config["metabolites"]
    assert config["wrt_metab"] == "Cr,PCr"
    assert 0 < config["parameters"]["naa"][1] < 5                     # from the ground truth

    _assert_simulates_the_challenge(config, points=2048, bandwidth=4000.0,
                                    frequency=123.26, centre=4.65)

    # measured the challenge's way, the simulated SNR covers the challenge's 20-161 (and a
    # little more: MRS-Sim sets noise against Cr + PCr, the challenge measures its top peak)
    simulated, _ = mrs_sim.simulate(config, n=16, seed=1, components=False)
    ratios = challenge2016.snr(simulated)
    assert 10 < ratios.min() and ratios.max() < 250 and np.ptp(ratios) > 50


#*********#
#   snr   #
#*********#
@needs_2016
def test_the_2016_snr_is_the_challenges_own():
    """The paper's values: SNR ~86 for dataset 1, 20, 30, 43 and 161 for 12, 15, 18, 21."""
    data, _, _ = challenge2016.load()
    ratios = challenge2016.snr(data)

    np.testing.assert_allclose(ratios[[0, 11, 14, 17, 20]], [86, 20, 30, 43, 161], rtol=0.05)


#****************#
#   truth mrsi   #
#****************#
@needs_mrsi
def test_the_mrsi_truth_gives_the_papers_lineshape():
    """Read off the noiseless truth, the voxels' decay is the one the descriptor paper
    states: TG 114 +/- 2 ms throughout, T2' with a median of 112-132 ms."""
    samples = mrsi_challenge.truth_parameters((MRSI_SUBJECT,), voxels=200)

    assert 0.112 < np.median(samples["TG"]) < 0.116
    t2_prime = 1 / (samples["Cr.d"] + 1 - (1 / 0.160 + 1 / 0.177) / 2)   # tCr's own T2 out
    assert 0.100 < np.median(t2_prime) < 0.140
    assert set(samples) >= {"NAA", "NAA.d", "snr", "f_shift"}
    assert "Cr" not in samples and "Lac" not in samples      # fixed reference, zero throughout


#*****************#
#   config mrsi   #
#*****************#
@needs_mrsi
@needs_sim
def test_the_mrsi_config_simulates_on_the_challenge_grid(tmp_path):
    config = mrsi_challenge.mrs_sim_config(folder=tmp_path, subjects=(MRSI_SUBJECT,),
                                           voxels=300)

    assert set(config["metabolites"]) == (set(mrsi_challenge.NAMES.values())
                                          | set(mrsi_challenge.MACROMOLECULES))
    assert config["wrt_metab"] == "Cr,Cr39"
    assert 1 < config["parameters"]["naa"][1] < 3                     # from the metabolite maps

    _assert_simulates_the_challenge(config, points=384, bandwidth=1 / 0.00083,
                                    frequency=123.24, centre=4.7)


#**************#
#   fit 2016   #
#**************#
@needs_2016
@pytest.mark.parametrize("fitter", FITTERS)
def test_every_fitter_covers_the_2016_ground_truth(fitter, ismrm2016):
    data, truth, path = ismrm2016
    _assert_covers_truth(_fit(fitter, data, path), truth, path)


#**************#
#   fit mrsi   #
#**************#
@needs_mrsi
@pytest.mark.parametrize("fitter", FITTERS)
def test_every_fitter_covers_the_mrsi_ground_truth(fitter, mrsi):
    data, truth, path = mrsi
    _assert_covers_truth(_fit(fitter, data, path), truth, path)
