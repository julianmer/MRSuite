####################################################################################################
#                                        test_mrspeclab.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-02                                                                              #
#                                                                                                  #
# Purpose: The MRSpecLAB bridge: the conversion to MRSpecLAB's transients and back, its nodes      #
#          on challenge spectra and on synthetic multi-coil, multi-transient acquisitions, the     #
#          warnings for what the pinned MRSpecLAB gets wrong, and the method contract. Each        #
#          call starts MRSpecLAB's environment, a few seconds.                                     #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import json
import shutil
import subprocess

import numpy as np
import pytest

from nifti_mrs.create_nmrs import gen_nifti_mrs

from nifti_mrs_plus import NIfTI_MRS_Plus
from mrsuite.bridges import mrspeclab
from mrsuite.datasets import challenge2016
from mrsuite.testing import check

needs_mrspeclab = pytest.mark.skipif(
    not (mrspeclab.ROOT / "processing").exists()
    or not (mrspeclab.PYTHON or shutil.which("uv")),
    reason="MRSpecLAB submodule not checked out, or no uv to build its environment")
pytestmark = needs_mrspeclab

POINTS, DWELL, FREQUENCY, CENTRE = 2048, 2.5e-4, 123.2, 4.65
METABOLITES = [(1.0, 2.01), (0.8, 3.03), (0.7, 3.2)]          # NAA, Cr, Cho: (amplitude, ppm)


#*****************#
#   acquisition   #
#*****************#
def _acquisition(coils=4, transients=16, seed=0, lines=METABOLITES, drift=3.0, noise=0.02,
                 tags=("DIM_COIL", "DIM_DYN")):
    """A synthetic acquisition: every coil sees the same lines at its own gain and phase,
    the same coils in every acquisition, and each transient is shifted by its own few Hz,
    as drift shifts it."""
    coil = np.random.default_rng(42)
    gains = coil.uniform(0.5, 1.5, coils) * np.exp(1j * coil.uniform(-np.pi, np.pi, coils))
    rng = np.random.default_rng(seed)
    shifts = rng.normal(0, drift, transients)
    t = np.arange(POINTS) * DWELL
    fids = np.empty((POINTS, coils, transients), complex)
    for k, shift in enumerate(shifts):
        fid = sum(a * np.exp(2j * np.pi * ((ppm - CENTRE) * FREQUENCY + shift) * t - t / 0.08)
                  for a, ppm in lines)
        fids[:, :, k] = gains * fid[:, None] + noise * (
            rng.normal(size=(POINTS, coils)) + 1j * rng.normal(size=(POINTS, coils)))
    if tags == ("DIM_DYN", "DIM_COIL"):
        fids = fids.transpose(0, 2, 1)
    out = gen_nifti_mrs(fids.reshape((1, 1, 1) + fids.shape), DWELL, FREQUENCY,
                        dim_tags=list(tags) + [None], no_conj=False)
    out.add_hdr_field("EchoTime", 0.03)
    return out


#***********#
#   water   #
#***********#
def _water(seed=1, transients=2):
    """An unsuppressed water reference from the same coils."""
    return _acquisition(transients=transients, seed=seed, lines=[(10.0, CENTRE)], drift=0.0)


#***********#
#   peaks   #
#***********#
def _peaks(spectra):
    """Each transient's NAA frequency in Hz, from its spectrum zero-filled 16-fold."""
    fids = spectra[:].reshape(spectra.shape)[0, 0, 0].reshape(spectra.shape[3], -1)
    hz = np.fft.fftshift(np.fft.fftfreq(16 * POINTS, DWELL))
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(fids, 16 * POINTS, axis=0), axes=0))
    naa = (2.01 - CENTRE) * FREQUENCY
    window = np.abs(hz - naa) < 30
    return hz[window][np.argmax(spectrum[window], axis=0)]


#***************#
#   challenge   #
#***************#
@pytest.fixture(scope="module")
def challenge():
    """Two challenge spectra and their water references: single, combined spectra."""
    if not challenge2016.BASIS.exists():
        pytest.skip("challenge submodule not checked out")
    data, water, _ = challenge2016.load(2)
    return data, water


#**************#
#   combined   #
#**************#
@pytest.fixture(scope="module")
def combined():
    """A synthetic acquisition and its water, coil-combined by SVD."""
    return mrspeclab.CoilCombinationSVD(_acquisition(), wref=_water())


#****************#
#   round trip   #
#****************#
@pytest.mark.parametrize("tags", [("DIM_COIL", "DIM_DYN"), ("DIM_DYN", "DIM_COIL")])
def test_the_conversion_gives_back_what_went_in(challenge, tags):
    """There and back without a node: the samples, dimensions and header as they were,
    a single coil's DIM_COIL and a dimension's own info and header included."""
    data, water = challenge
    acquisition = _acquisition(coils=3, transients=5, tags=tags)
    acquisition.set_dim_tag(4 + tags.index("DIM_DYN"), "DIM_DYN", info="five transients",
                            header={"RepetitionTime": [2.0, 2.1, 2.2, 2.3, 2.4]})
    for spectra, references in ((data, water), (NIfTI_MRS_Plus([acquisition]), None)):
        outs = mrspeclab._process(None, spectra, references, {})
        for i, out in enumerate(outs):
            pairs = ([(out, spectra[i])] if references is None
                     else zip(out, (spectra[i], references[i])))
            for back, source in pairs:
                assert back.shape == source.shape and back.dim_tags == source.dim_tags
                assert np.array_equal(back[:], source[:])
                assert back.hdr_ext.to_dict() == source.hdr_ext.to_dict()


#***************#
#   averaging   #
#***************#
def test_averaging_takes_the_mean_over_the_transients(combined):
    """Average collapses DIM_DYN, which the conversion squeezes away."""
    spectra, _ = combined
    out = mrspeclab.Average(spectra)

    assert out.shape == spectra.shape[:4] and out.dim_tags == [None, None, None]
    assert np.allclose(out[:], spectra[:].mean(axis=-1))


#**********************#
#   coil combination   #
#**********************#
@pytest.mark.parametrize("node", ["CoilCombinationSVD", "CoilCombinationAdaptive"])
def test_coil_combination_recovers_the_signal(node):
    """Each transient's coils combine into the one signal they share; the water too."""
    clean = _acquisition(coils=1, noise=0.0)[:].reshape(POINTS, -1)
    spectra, water = getattr(mrspeclab, node)(_acquisition(), wref=_water())

    assert spectra.dim_tags == ["DIM_DYN", None, None] and spectra.shape[-1] == 16
    assert water.dim_tags == ["DIM_DYN", None, None] and water.shape[-1] == 2
    fids = spectra[:].reshape(POINTS, -1)
    for k in range(16):
        r = np.abs(np.vdot(fids[:, k], clean[:, k])) / (np.linalg.norm(fids[:, k])
                                                         * np.linalg.norm(clean[:, k]))
        assert r > 0.99


#***************#
#   alignment   #
#***************#
def test_alignment_reduces_the_frequency_spread(combined):
    """FreqPhaseAlignment aligns on NAA at 1.8-2.2 ppm, which it finds only if the
    conversion hands suspect the spectrum the right way round."""
    spectra, _ = combined
    out = mrspeclab.FreqPhaseAlignment(spectra)

    before, after = np.std(_peaks(spectra)), np.std(_peaks(out))
    assert before > 2.0 and after < 0.2 * before


#******************#
#   bad averages   #
#******************#
def test_a_removed_transient_leaves_the_dynamic_dimension(combined):
    """RemoveBadAverages marks a removed transient None, which leaves DIM_DYN shorter."""
    spectra, _ = combined
    fids = spectra[:].copy()
    fids[..., 5] *= 20                                       # one transient far off the rest
    corrupted = gen_nifti_mrs(fids, DWELL, FREQUENCY, dim_tags=["DIM_DYN", None, None],
                              no_conj=False)
    out = mrspeclab.RemoveBadAverages(corrupted)

    assert out.shape[-1] == 15
    assert np.array_equal(out[:], np.delete(fids, 5, axis=-1))


#********************#
#   single spectra   #
#********************#
def test_nodes_on_single_spectra(challenge):
    """LineBroadening broadens, ZeroPadding pads, and each records itself."""
    data, _ = challenge
    broadened = mrspeclab.LineBroadening(data, factor=10)
    padded = mrspeclab.ZeroPadding(data, factor=1)

    for out, source in zip(broadened.list(), data.list()):
        assert out.shape == source.shape
        assert np.abs(np.fft.fft(out[:])).max() < np.abs(np.fft.fft(source[:])).max()
        entry = out.hdr_ext["ProcessingApplied"][-1]
        details = json.loads(entry["Details"])
        assert entry["Program"] == "MRSpecLAB" and entry["Method"] == "LineBroadening"
        assert "6e71664" in entry["Version"]
        assert details["kwargs"] == {"factor": 10}
        assert details["packages"]["suspect"] == "0.6.2"
    assert padded[0].shape[3] == 2 * data[0].shape[3]


#***************#
#   reference   #
#***************#
def test_eddy_current_correction_returns_its_reference(challenge):
    """A node given a water reference hands it back as it left it, beside the spectra."""
    data, water = challenge
    spectra, references = mrspeclab.EddyCurrentCorrection(data, wref=water)

    assert len(spectra) == len(references) == len(data)
    assert not np.allclose(references[0][:], water[0][:])
    assert references[0].hdr_ext["ProcessingApplied"][-1]["Program"] == "MRSpecLAB"


#**************#
#   warnings   #
#**************#
def test_gaussian_line_broadening_warns_and_fails(challenge):
    with pytest.warns(UserWarning, match="'factor'"):
        with pytest.raises(RuntimeError, match="KeyError: 'factor'"):
            mrspeclab.LineBroadening_Gaussian(challenge[0])


#***************************#
#   choice flags warnings   #
#***************************#
def test_the_string_false_is_ignored_and_warned(combined):
    """'False' aligns all the same, and is warned about; the bool False is honoured."""
    spectra, _ = combined
    with pytest.warns(UserWarning, match="'False' is true"):
        ignored = mrspeclab.FreqPhaseAlignment(spectra, alignFreq="False", alignPhase="False")
    honoured = mrspeclab.FreqPhaseAlignment(spectra, alignFreq=False, alignPhase=False)

    assert not np.allclose(ignored[:], spectra[:])
    assert np.array_equal(honoured[:], spectra[:])


#******************#
#   sn2 warnings   #
#******************#
def test_sn2_coil_combination_warns():
    with pytest.warns(UserWarning, match="number of coils"):
        mrspeclab.CoilCombinationSN2(_acquisition(transients=2), wref=_water())


#*************#
#   unknown   #
#*************#
def test_unknown_nodes_and_parameters_are_refused(challenge):
    with pytest.raises(AttributeError, match="no node"):
        mrspeclab.NotANode
    with pytest.raises(TypeError, match="no parameter"):
        mrspeclab.LineBroadening(challenge[0], width=3)


#**************#
#   pristine   #
#**************#
def test_the_checkout_is_left_as_it_was(challenge):
    """Running nodes writes nothing into MRSpecLAB's checkout, not even bytecode."""
    mrspeclab.LineBroadening(challenge[0])
    status = subprocess.run(["git", "-C", str(mrspeclab.ROOT), "status", "--porcelain",
                             "--ignored"], capture_output=True, text=True)
    assert status.returncode == 0 and status.stdout == ""


#*****************#
#   conformance   #
#*****************#
def test_conformance(challenge):
    data, water = challenge
    check(mrspeclab.EddyCurrentCorrection, data, wref=water)
    check(mrspeclab.CoilCombinationSVD, NIfTI_MRS_Plus([_acquisition(), _acquisition(seed=2)]))


#******************#
#   water header   #
#******************#
def test_the_water_reference_keeps_its_own_header_and_history():
    """Not the spectrum's: the reference came of another input."""
    from mrsuite.core import provenance

    data, water = _acquisition(), _water()
    provenance.append(data, provenance.entry("test", "1", "earlier", {}))
    data.add_hdr_field("InversionTime", 0.5)

    out, reference = mrspeclab.CoilCombinationSVD(data, wref=water)

    assert [e["Method"] for e in provenance.history(out)][0] == "earlier"
    assert "InversionTime" not in reference.hdr_ext
    assert [e["Method"] for e in provenance.history(reference)] == ["CoilCombinationSVD"]


#******************#
#   pass-through   #
#******************#
def test_what_a_node_leaves_untouched_is_handed_back_unrecorded(challenge):
    """Eddy-current correction without a reference skips; line broadening never touches
    the reference. Each says so, and records no step on what it did not do."""
    from mrsuite.core import provenance

    data, water = challenge
    with pytest.warns(UserWarning, match="left the spectrum as it was"):
        out = mrspeclab.EddyCurrentCorrection(data[0])
    assert provenance.history(out) == provenance.history(data[0])

    with pytest.warns(UserWarning, match="left the water reference as it was"):
        out, reference = mrspeclab.LineBroadening(data[0], wref=water[0], factor=5)
    assert provenance.history(reference) == provenance.history(water[0])
    assert len(provenance.history(out)) == len(provenance.history(data[0])) + 1


#*****************#
#   driver meta   #
#*****************#
def test_the_driver_states_the_acquisition_as_suspect_reads_it():
    """Times in ms, the ppm reference, frequency and dwell; a time not stated is NaN, not
    suspect's 30 ms."""
    fids, meta = mrspeclab._transients(_acquisition(coils=2, transients=1))
    meta["tr"] = None
    code = ("import json, sys; sys.path.insert(0, sys.argv[1]); import numpy as np; "
            "from mrspeclab_driver import transients; "
            "m = transients(np.zeros((1, 2, 8), complex), json.loads(sys.argv[2]))[0]; "
            "print(json.dumps([m.te, m.tr, m.ppm0, m.f0, m.dt]))")
    out = subprocess.run([mrspeclab.python(), "-c", code, str(mrspeclab.DRIVER.parent),
                          json.dumps(meta)], capture_output=True, text=True, check=True)
    te, tr, ppm0, f0, dt = json.loads(out.stdout.replace("NaN", "null"))
    assert (te, tr, ppm0, f0, dt) == (30.0, None, CENTRE, FREQUENCY, DWELL)


#***************#
#   none left   #
#***************#
def test_removing_every_transient_is_refused_plainly(combined):
    with pytest.raises(RuntimeError, match="left no transient"):
        mrspeclab.RemoveBadAverages(combined[0], stdDevThreshold=0.001)
