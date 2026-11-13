####################################################################################################
#                                         test_vae_mrs.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-11-12                                                                              #
#                                                                                                  #
# Purpose: Test VAE-MRS on the example subject its repository ships, against the original's own    #
#          network and the numbers of its end-to-end notebook.                                     #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import json

import numpy as np
import pytest
import torch

pytest.importorskip("fsl_mrs")
from fsl_mrs.core.nifti_mrs import gen_nifti_mrs
from fsl_mrs.utils import mrs_io
from fsl_mrs.utils.misc import calculateAxes
from fsl_mrs.utils.preproc import nifti_mrs_proc as proc

from mrsuite.models import VAEMRS, load

# the notebook's reconstruction table for the example subject: the mean squared error of the
# reconstructed real spectra over 0 to 7 ppm, OFF then ON
NOTEBOOK_MSE = (2.77e-4, 2.88e-4)


#***********#
#   model   #
#***********#
@pytest.fixture(scope="module")
def vae():
    return VAEMRS.published()


#**********#
#   scan   #
#**********#
@pytest.fixture(scope="module")
def scan(vae):
    """The example subject, preprocessed as the original's MEGAPRESSDataset does: the odd
    transients negated and first along the edit dimension (its OFF), then aligned,
    eddy-current corrected, referenced to creatine and phased."""
    folder = vae.repo / "example_data" / "C008"
    act = mrs_io.read_FID(str(folder / "C008_MP_GABA_act.nii.gz"))
    ref = mrs_io.read_FID(str(folder / "C008_Water_ref.nii.gz"))
    raw = act.image.data
    act = gen_nifti_mrs(np.stack((-raw[..., 1::2], raw[..., 0::2]), axis=-1), act.dwelltime,
                        act.spectrometer_frequency[0], nucleus=act.nucleus, affine=None,
                        dim_tags=["DIM_DYN", "DIM_EDIT", None], no_conj=True)
    ref = gen_nifti_mrs(ref.image.data, ref.dwelltime, ref.spectrometer_frequency[0],
                        nucleus=ref.nucleus, affine=None, no_conj=True)
    act = proc.align(data=act, dim="DIM_DYN", ppmlim=(0.2, 4.2))
    act = proc.ecc(data=act, reference=ref)
    act = proc.shift_to_reference(data=act, ppm_ref=3.027, peak_search=(2.9, 3.1))
    return proc.phase_correct(data=act, ppmlim=(2.9, 3.1))


#**************#
#   original   #
#**************#
def _original(scan):
    """The scan as the original feeds its network: per condition (transients, points, 2),
    the spectra scaled to a maximum of one, and that maximum."""
    fids = scan.image.data[0, 0, 0]
    specs = np.fft.fftshift(np.fft.fft(fids, axis=0, norm="ortho"), axes=0)
    scale = np.abs(specs).max()
    specs = np.stack([specs.real, specs.imag], axis=-1).astype(np.float32) / scale
    return [torch.from_numpy(specs[:, :, k]).permute(1, 0, 2) for k in range(2)], scale


def _spectra(result):
    """A result's transients as the network gives them: (edit, transients, points)."""
    fids = np.asarray(result.image[:])[0, 0, 0]       # as stored, as the original reads them
    return np.fft.fftshift(np.fft.fft(fids, axis=0, norm="ortho"), axes=0).transpose(2, 1, 0)


#********************#
#   reconstruction   #
#********************#
def test_the_reconstruction_is_the_notebooks(vae, scan):
    result = vae(scan, method="reconstruction")
    inputs, scale = _original(scan)
    ppm = np.flip(calculateAxes(scan.bandwidth, scan.spectrometer_frequency[0], scan.shape[3],
                                4.65)["ppmshift"])
    window = (ppm >= 0) & (ppm <= 7)

    assert result.shape == scan.shape and list(result.dim_tags) == list(scan.dim_tags)
    for k, (label, x) in enumerate(zip(("OFF", "ON"), inputs)):
        with torch.no_grad():
            mu, _ = vae.net.encode(x, [label] * len(x))
            expected = vae.net.decode(mu, [label] * len(x)).numpy()
        got = _spectra(result)[k] / scale
        assert got.real == pytest.approx(expected[..., 0], abs=1e-5)
        assert got.imag == pytest.approx(expected[..., 1], abs=1e-5)
        error = np.mean((got.real[:, window] - x.numpy()[:, window, 0]) ** 2)
        assert error == pytest.approx(NOTEBOOK_MSE[k], rel=5e-3)


#***************#
#   synthesis   #
#***************#
@pytest.mark.parametrize("method", ["random_sampling", "interpolation", "random_interpolation"])
def test_the_synthetic_transients_are_the_originals_draw(vae, scan, method):
    """The same seed gives what the original's functions give, called in its order."""
    result = vae(scan, method=method, samples=2, seed=0)
    inputs, scale = _original(scan)
    noise = torch.tensor(vae.noise_std)

    torch.manual_seed(0)
    for k, (label, x) in enumerate(zip(("OFF", "ON"), inputs)):
        with torch.no_grad():
            z = vae.net.reparameterize(*vae.net.encode(x, [label] * len(x)))
            z = {"random_sampling": lambda: vae.net.random_sampling(z, 2, noise_std=noise),
                 "interpolation": lambda: vae.net.interpolate(z, 2),
                 "random_interpolation":
                     lambda: vae.net.interpolate_with_random_noise(z, 2, noise_std=noise)}[method]()
            expected = vae.net.decode(z, [label] * len(z)).numpy()
        got = _spectra(result)[k] / scale
        assert got.shape == (2 * scan.shape[4], scan.shape[3])
        assert got.real == pytest.approx(expected[..., 0], abs=1e-5)


def test_the_transients_given_come_first_when_asked(vae, scan):
    result = vae(scan, method="random_sampling", samples=1, originals=True, seed=0)

    assert result.shape[4] == 2 * scan.shape[4]
    assert np.asarray(result.image[:])[..., :scan.shape[4], :] == pytest.approx(
        scan.image.data, abs=1e-6 * np.abs(scan.image.data).max())
    history = scan.hdr_ext["ProcessingApplied"]          # the header goes along: its steps
    assert result.hdr_ext["ProcessingApplied"][:len(history)] == history


#****************#
#   provenance   #
#****************#
def test_the_provenance_pins_the_model(vae, scan):
    result = vae(scan, method="reconstruction")
    item = result.hdr_ext["ProcessingApplied"][-1]
    details = json.loads(item["Details"])

    assert item["Program"] == "MRSuite" and item["Method"] == "VAEMRS"
    assert details["model"] == "mrsuite.models.vae_mrs.VAEMRS"
    assert details["weights"] == vae.fingerprint() and details["vae-mrs"] == "7cbfa78"


#************#
#   refuse   #
#************#
def test_a_scan_it_was_not_trained_for_is_refused(vae, scan):
    data = scan.image.data

    def made(array, dwelltime, tags):
        return gen_nifti_mrs(array, dwelltime, scan.spectrometer_frequency[0],
                             nucleus=scan.nucleus, affine=None, dim_tags=tags, no_conj=True)

    with pytest.raises(ValueError, match="DIM_EDIT"):
        vae(made(data[..., 0], scan.dwelltime, ["DIM_DYN", None, None]))
    with pytest.raises(ValueError, match="points"):
        vae(made(data[:, :, :, :1024], scan.dwelltime, ["DIM_DYN", "DIM_EDIT", None]))
    with pytest.raises(ValueError, match="bandwidth"):
        vae(made(data, scan.dwelltime / 2, ["DIM_DYN", "DIM_EDIT", None]))
    with pytest.raises(ValueError, match="no method"):
        vae(scan, method="sampling")


#*********#
#   hub   #
#*********#
def test_save_and_load_keep_the_model(vae, scan, tmp_path):
    vae.save_pretrained(tmp_path)
    loaded = load(str(tmp_path))

    assert isinstance(loaded, VAEMRS) and loaded.fingerprint() == vae.fingerprint()
    assert loaded.noise_std == vae.noise_std and loaded.acquisition == vae.acquisition
    assert np.asarray(loaded(scan, method="reconstruction")[:]) == pytest.approx(
        np.asarray(vae(scan, method="reconstruction")[:]))
