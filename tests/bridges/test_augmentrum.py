####################################################################################################
#                                       test_augmentrum.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 26/10/26                                                                                #
#                                                                                                  #
# Purpose: Augmentrum's modules run as steps on a batch, record themselves, and are listed and    #
#          described from its own registry.                                                       #
#                                                                                                  #
####################################################################################################
import numpy as np
import pytest
from nifti_mrs.create_nmrs import gen_nifti_mrs
from nifti_mrs_plus import NIfTI_MRS_Plus

pytest.importorskip("augmentrum")
from mrsuite.bridges import augmentrum                                   # noqa: E402


def _spectra(n: int = 2) -> NIfTI_MRS_Plus:
    t = np.arange(512) * 2.5e-4
    fid = np.exp(2j * np.pi * (2.0 - 4.65) * 123.2 * t - t / 0.05)
    spectra = [gen_nifti_mrs((k + 1) * fid.reshape(1, 1, 1, -1), 2.5e-4, 123.2) for k in range(n)]
    for s in spectra:
        s.add_hdr_field("EchoTime", 0.03)
    return NIfTI_MRS_Plus(spectra)


def test_a_module_runs_on_the_batch_and_records_itself():
    clean = _spectra()
    noisy = augmentrum.bridge.noise(clean, sigma_frac=0.05, seed=0)
    before, after = np.asarray(clean.list()[0][:]), np.asarray(noisy.list()[0][:])
    assert len(noisy) == 2 and after.shape == before.shape and not np.allclose(after, before)
    assert np.allclose(np.asarray(clean.list()[0][:]), before), "the input is left alone"
    step = noisy.list()[0].hdr_ext["ProcessingApplied"][-1]
    assert step["Program"] == "Augmentrum" and step["Method"] == "Noise"
    assert "sigma_frac" in step["Details"]
    assert noisy.list()[0].hdr_ext["EchoTime"] == clean.list()[0].hdr_ext["EchoTime"]

    broad = augmentrum.bridge.step.line_broadening(lb_hz=5.0)(clean)      # as a pipeline step
    assert len(broad) == 2 and broad.list()[1].hdr_ext["ProcessingApplied"][-1]["Method"] == \
        "LineBroadening"
    many = augmentrum.bridge.transient_synthesis(clean, n_transients=4, seed=0)   # no NIfTI backend
    assert many.list()[0].shape[-1] == 4 and "DIM_DYN" in many.list()[0].dim_tags


def test_the_registry_is_the_catalogue():
    names = augmentrum.bridge.available()
    assert "noise" in names and "baseline_bspline" in names, "modules and their fixed forms"
    assert "broadening" not in names and "water" not in names, "plain aliases folded"
    assert augmentrum.bridge.steps() == names
    signature, help_ = augmentrum.bridge.about("line_broadening")
    assert list(signature.parameters) == ["data", "lb_hz", "gb_hz", "mode", "water"]
    assert "mode" not in augmentrum.bridge.about("baseline_bspline")[0].parameters, "fixed"
    assert "broadening" in help_.lower()
    with pytest.raises(AttributeError, match="no module"):
        augmentrum.bridge.nonsense
