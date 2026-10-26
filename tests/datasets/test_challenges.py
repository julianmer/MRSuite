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
from tests.bridges.test_tarquin import needs_tarquin
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
           pytest.param("spant", id="spant", marks=needs_r),
           pytest.param("tarquin", id="tarquin", marks=needs_tarquin)]


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
        origin = next(e for e in result.provenance if e["Program"] == "MRSuite")  # after spec2nii's
        assert "load" in origin["Method"]
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


#******************#
#   2016 release   #
#******************#
def _release(tmp_path, monkeypatch, md5=None):
    """The challenge as an installed MRSuite finds it: no checkout, a release as published
    (a zip of zips, one without a folder of its own, the truth renamed) to download."""
    import hashlib
    import zipfile

    def zipped(path, files):
        with zipfile.ZipFile(path, "w") as z:
            for name, source in files.items():
                z.write(source, name)
        return path

    inner = tmp_path / "inner"
    inner.mkdir()
    spectra = challenge2016.ROOT / "datasets_JMRUI"
    zipped(inner / "datasets_JMRUI.zip", {f"datasets_JMRUI/{f}": spectra / f
                                         for f in ("dataset1_WS.txt", "dataset1_nWS.txt")})
    zipped(inner / "basisset_text_noTMS_20160929.zip",
           {"Cr.txt": challenge2016.ROOT / "basisset_text_noTMS" / "Cr.txt"})
    release = zipped(tmp_path / "release.zip", {
        "datasets/datasets_JMRUI.zip": inner / "datasets_JMRUI.zip",
        "basis_sets/basisset_text_noTMS_20160929.zip": inner / "basisset_text_noTMS_20160929.zip",
        "ground_truth/dataset1_answer.xlsx": challenge2016.TRUTH / "dataset1.xlsx"})
    data = tmp_path / "home" / "data"
    monkeypatch.setattr(challenge2016.externals, "CHECKOUT", tmp_path / "site-packages")
    monkeypatch.setattr(challenge2016, "DATA", data)
    monkeypatch.setattr(challenge2016, "ROOT", data / "2016_fitting_challenge")
    monkeypatch.setattr(challenge2016, "TRUTH", data / "2016_fitting_challenge_gts")
    monkeypatch.setattr(challenge2016, "RELEASE", (release.as_uri(), release.stat().st_size,
                        md5 or hashlib.md5(release.read_bytes()).hexdigest()))
    return data


@needs_2016
def test_an_install_downloads_the_release_once_and_reads_it_as_the_checkout(tmp_path,
                                                                            monkeypatch):
    checkout = challenge2016.load(n=1)
    data = _release(tmp_path, monkeypatch)

    challenge2016.fetch()

    root, truth = challenge2016.ROOT, challenge2016.TRUTH
    assert (root / "basisset_text_noTMS" / "Cr.txt").is_file(), "a flat zip in its folder"
    assert [p.name for p in truth.iterdir()] == ["dataset1_answer.xlsx"]
    assert sorted(p.name for p in data.iterdir()) == [root.name, truth.name], "no part left"
    got = challenge2016.load(n=1, root=root, truth=truth)
    assert np.array_equal(got[0].list()[0][:], checkout[0].list()[0][:])
    assert got[2] == checkout[2], "the truth by either name"

    monkeypatch.setattr(challenge2016.externals, "download", None)    # never again
    challenge2016.fetch()


@needs_2016
def test_two_processes_fetching_the_release_at_once_both_finish(tmp_path, monkeypatch):
    _release(tmp_path, monkeypatch)
    download = challenge2016.externals.download

    def raced(url, dest, size=None, md5=None):            # the other one finishes meanwhile
        (challenge2016.ROOT / "datasets_JMRUI").mkdir(parents=True)
        challenge2016.TRUTH.mkdir()
        (challenge2016.TRUTH / "theirs.xlsx").write_bytes(b"")
        return download(url, dest, size, md5)

    monkeypatch.setattr(challenge2016.externals, "download", raced)
    challenge2016.fetch()

    assert [p.name for p in challenge2016.TRUTH.iterdir()] == ["theirs.xlsx"], "theirs stands"
    assert sorted(p.name for p in challenge2016.DATA.iterdir()) == \
        sorted([challenge2016.ROOT.name, challenge2016.TRUTH.name]), "no part left"


@needs_2016
def test_a_release_not_as_published_is_not_kept(tmp_path, monkeypatch):
    data = _release(tmp_path, monkeypatch, md5="0" * 32)

    with pytest.raises(RuntimeError, match="MD5"):
        challenge2016.fetch()
    assert not list(data.iterdir())


#******************#
#   mrsi release   #
#******************#
def test_mrsi_subjects_missing_are_fetched_first_and_said(tmp_path, monkeypatch):
    challenge = pytest.importorskip("augmentrum.dataset.mrsi_challenge")
    said = []

    class Release:
        def __init__(self, data_dir, source):
            self.data_dir = data_dir

        def available(self, subject):
            return subject == "Sub1"

        def fetch(self, subjects, root):
            said.append((subjects, root, mrsi_challenge.externals.now()))

    monkeypatch.setattr(challenge, "MRSIChallengeDataModule", Release)
    mrsi_challenge._module(tmp_path, "Sub1", "Sub2")

    assert said == [(["Sub2"], str(tmp_path),
                     f"Downloading the MRSI challenge's Sub2 from Zenodo into {tmp_path}…")]
    mrsi_challenge._module(tmp_path, "Sub1")
    assert len(said) == 1, "nothing missing, nothing fetched"


#*****************#
#   voxel index   #
#*****************#
@needs_mrsi
def test_the_voxel_index_is_the_loaders_without_loading():
    index = mrsi_challenge.voxel_index(MRSI_SUBJECT, voxels=3)
    _, _, loaded = mrsi_challenge.load(MRSI_SUBJECT, voxels=3)
    assert index.shape == (3, 3) and (index == loaded).all()
    assert len(mrsi_challenge.voxel_index(MRSI_SUBJECT)) > 1000


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


#*************#
#   release   #
#*************#
@needs_mrsi
def test_the_release_keeps_the_loaders_voxels_in_its_order():
    """What a model learns from is what it is tested on: the voxels load gives, as they are."""
    module = mrsi_challenge._module(mrsi_challenge.DATA_DIR)
    voxels = mrsi_challenge._brain(module, MRSI_SUBJECT, "all", mrsi_challenge.DATA_DIR)
    data, _, _ = mrsi_challenge.load(MRSI_SUBJECT, voxels=5)
    rows = np.linspace(0, len(voxels) - 1, 5).astype(int)
    np.testing.assert_array_equal(voxels[rows], [np.asarray(s[:]).ravel() for s in data.list()])


@needs_mrsi
def test_a_drawn_voxel_comes_with_its_own_truth_and_parts():
    module = mrsi_challenge._module(mrsi_challenge.DATA_DIR)
    brain = {part: mrsi_challenge._brain(module, MRSI_SUBJECT, part, mrsi_challenge.DATA_DIR)
             for part in ("all", "meta", "nuisance")}
    aux = module.read_aux(MRSI_SUBJECT, signal="meta")
    maps = aux["metaMap"][aux["brainMask"] > 0]

    source = mrsi_challenge.Release((MRSI_SUBJECT,), "all", components=True, seed=3)
    fids, truth = source.draw(6)
    rows = [int(np.flatnonzero((brain["all"] == fid).all(axis=1))[0]) for fid in fids]
    for i, name in enumerate(mrsi_challenge.NAMES.values()):
        np.testing.assert_array_equal(truth.concentrations[name], maps[rows, i])
    np.testing.assert_array_equal(truth.components["metabolites"], brain["meta"][rows])
    np.testing.assert_array_equal(truth.components["nuisance"], brain["nuisance"][rows])

    clean, clean_truth = mrsi_challenge.Release((MRSI_SUBJECT,), "all-nuisance",
                                                components=True, seed=3).draw(6)
    np.testing.assert_allclose(clean, fids - truth.components["nuisance"],   # complex64,
                               atol=1e-6 * np.abs(fids).max())     # as Augmentrum caches it
    assert set(clean_truth.components) == {"metabolites"}       # no nuisance left to give
    assert source.acquisition["points"] == fids.shape[1] == len(source.basis["NAA"])
    assert all(low <= high for low, high in source.ranges.values())


def test_the_release_has_two_signals_only():
    with pytest.raises(ValueError, match="'all' or 'all-nuisance'"):
        mrsi_challenge.Release((MRSI_SUBJECT,), "meta")


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


#***********#
#   suite   #
#***********#
def _water_over_naa(data):
    """Each spectrum's peak at 4.4-5.0 ppm over its peak at 1.9-2.1 ppm."""
    out = []
    for s in data.list():
        fid = np.asarray(s[:]).ravel()
        spec = np.abs(np.fft.fftshift(np.fft.fft(fid)))
        ppm = 4.65 + np.fft.fftshift(np.fft.fftfreq(fid.size, s.dwelltime)) \
            / s.spectrometer_frequency[0]
        out.append(spec[(ppm > 4.4) & (ppm < 5.0)].max() / spec[(ppm > 1.9) & (ppm < 2.1)].max())
    return np.array(out)


@needs_2016
@needs_sim
def test_a_suite_level_moves_its_effect_out_of_the_training_distribution():
    """The SNR level is that SNR as the challenge measures it; the strongest residual water
    is as large as datasets 23 and 24 carry, where the base has none; concentrations above
    the trained range are above it; a level draws the same spectra every time, and
    another level others."""
    base = challenge2016.suite_config()
    ten, _ = challenge2016.suite("snr", "10", n=32)
    assert 0.7 < np.median(challenge2016.snr(ten)) / 10 < 1.4

    plain, _ = challenge2016.suite(n=8)
    water, _ = challenge2016.suite("residual-water", "20.0x-tcr", n=8)
    assert np.median(_water_over_naa(plain)) < 0.5 < 2 < np.median(_water_over_naa(water))

    above = challenge2016.suite_config("concentrations", "above-1-1.5x")["parameters"]
    assert all(above[m][0] >= base["parameters"][m][1] for m in ("naa", "glu", "ins"))

    def samples(data):
        return np.stack([np.asarray(s[:]).ravel() for s in data.list()])

    again, truth = challenge2016.suite("snr", "10", n=32)
    assert np.array_equal(samples(again), samples(ten)) and len(truth) == 32
    assert not np.array_equal(samples(challenge2016.suite("snr", "5", n=32)[0]), samples(ten))


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


#*****************#
#   water scale   #
#*****************#
@needs_2016
@pytest.mark.parametrize("fitter, extra, tolerance", [
    pytest.param("fsl_mrs", {}, 0.01, id="fsl_mrs"),
    pytest.param("tarquin", {}, 0.02, id="tarquin", marks=needs_tarquin),
    # LCModel needs its referencing shift kept small on a creatine-only spectrum, and
    # its Lorentzian bias put the challenge's 10.83 mM at 10 (Marjanska et al., 2022)
    pytest.param("lcmodel", {"params": {"shifmn(2)": -0.05, "shifmx(2)": 0.05}}, 0.10,
                 id="lcmodel", marks=needs_lcmodel)])
def test_the_challenges_water_scaling_gets_its_creatine_phantom_right(fitter, extra, tolerance):
    """The challenge's 10 mM creatine test, really 10.83 mM (Marjanska et al., MRM 2022),
    through each fitter's own water scaling on the challenge's settings."""
    from fsl_mrs.utils import mrs_io

    folder = challenge2016.ROOT / "Cr_water_scaling_JMRUI"
    data, water = (mrs_io.read_FID(str(folder / f"Cr_10mM_test_water_scaling_{s}.txt"))
                   for s in ("WS", "nWS"))
    options = challenge2016.WATER[fitter]
    options = {**options, **{k: {**options.get(k, {}), **v} for k, v in extra.items()}}
    module = importlib.import_module(f"mrsuite.bridges.{fitter}")
    result = module.fit(data, str(challenge2016.BASIS), water=water, **options)

    assert result.conc("molarity")["Cr"] == pytest.approx(10.83, rel=tolerance)


#**************#
#   fit mrsi   #
#**************#
@needs_mrsi
@pytest.mark.parametrize("fitter", FITTERS)
def test_every_fitter_covers_the_mrsi_ground_truth(fitter, mrsi):
    data, truth, path = mrsi
    _assert_covers_truth(_fit(fitter, data, path), truth, path)


#************#
#   volume   #
#************#
@needs_mrsi
def test_a_subject_loads_whole_as_one_spatial_nifti_with_its_truth_maps():
    data, truth, mask = mrsi_challenge.volume(MRSI_SUBJECT, "all", z=16)
    item = data.list()[0]
    assert len(data) == 1 and item.shape[:3] == (64, 64, 1) and item.SpecFreqChemShift == 4.7
    assert set(truth) == set(mrsi_challenge.NAMES.values()) and truth["NAA"].shape == (64, 64, 1)
    assert mask.shape == (64, 64, 1) and 500 < mask.sum() < 3000
    assert (truth["NAA"][mask] > 0).mean() > 0.9, "the truth lives in the brain"
    step = item.hdr_ext["ProcessingApplied"][-1]
    assert step["Method"].endswith("mrsi_challenge.volume") and '"z": 16' in step["Details"]


#*************#
#   upright   #
#*************#
@needs_mrsi
def test_the_release_is_turned_to_its_basis():
    """The release's FIDs are 180 degrees from the VtMeta basis; MRSuite turns every array,
    so the NAA line points up in the metabolite signals and in the full one alike."""
    def line(item, x, y, centre=2.01):
        fid = np.asarray(item[:]).reshape(*item.shape[:3], -1)[x, y, 0]
        spec = np.fft.fftshift(np.fft.fft(fid)).real
        ppm = np.fft.fftshift(np.fft.fftfreq(len(fid), item.dwelltime)) / item.spectrometer_frequency[0] + 4.7
        at = lambda p: spec[np.abs(ppm - p).argmin()]
        peak = max((at(p) for p in np.linspace(centre - 0.06, centre + 0.06, 9)), key=abs)
        return peak - 0.5 * (at(centre - 0.18) + at(centre + 0.18))      # above its shoulders
    for signal in ("meta", "all-nuisance"):       # in 'all' the water and lipid tails hide the line
        data, _, _ = mrsi_challenge.volume(MRSI_SUBJECT, signal, z=16)
        assert line(data.list()[0], 48, 19) > 0, signal
    one, _, index = mrsi_challenge.load(MRSI_SUBJECT, "meta", voxels=50)
    k = int(np.argmax([np.abs(np.asarray(i[:])).max() for i in one.list()]))
    assert line(one.list()[k], 0, 0) > 0, "load() turns it too"
