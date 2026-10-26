####################################################################################################
#                                         test_suspect.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 26/10/26                                                                                #
#                                                                                                  #
# Purpose: suspect's functions run as steps on NIfTI-MRS, the coils and transients going over on  #
#          its axes and back, and the steps written here apply what suspect only estimates.       #
#                                                                                                  #
####################################################################################################
import numpy as np
import pytest
from nifti_mrs.create_nmrs import gen_nifti_mrs
from nifti_mrs_plus import NIfTI_MRS_Plus

pytest.importorskip("suspect")
from mrsuite.bridges import suspect                                      # noqa: E402


def _acquired(coils: int = 2, transients: int = 4, phase: float = 0.0, shift: float = 0.0):
    t = np.arange(512) * 2.5e-4
    fid = np.exp(2j * np.pi * (2.0 - 4.65 + shift) * 123.2 * t - t / 0.05) * np.exp(1j * phase)
    data = np.stack([np.stack([(c + 1) * fid for c in range(coils)], -1)] * transients, -1)
    data = data.reshape(1, 1, 1, 512, coils, transients)
    item = gen_nifti_mrs(data, 2.5e-4, 123.2, dim_tags=["DIM_COIL", "DIM_DYN", None])
    item.add_hdr_field("EchoTime", 0.03)
    return NIfTI_MRS_Plus([item])


def test_the_coils_combine_and_the_axes_come_back():
    raw = _acquired()
    combined = suspect.bridge.combine_channels(raw)
    item = combined.list()[0]
    assert item.shape == (1, 1, 1, 512, 4) and list(item.dim_tags)[:1] == ["DIM_DYN"]
    assert item.hdr_ext["EchoTime"] == 0.03, "the header carried over"
    step = item.hdr_ext["ProcessingApplied"][-1]
    assert step["Program"] == "suspect" and "combine_channels" in step["Method"]
    mrs = suspect._suspect(raw.list()[0])
    assert mrs.shape == (4, 2, 512) and mrs.te == 30.0 and mrs.ppm0 == 4.65


def test_the_written_steps_apply_what_suspect_estimates():
    turned = _acquired(coils=1, transients=1, phase=0.8)
    phased = suspect.phasing(turned, method="mag_real")
    before = np.angle(np.asarray(turned.list()[0][:]).ravel()[0])
    after = np.angle(np.asarray(phased.list()[0][:]).ravel()[0])
    assert abs(before) > 0.7 and abs(after) < abs(before) / 4
    averaged = suspect.bridge.combine_channels(_acquired())
    aligned = suspect.alignment(averaged)
    assert aligned.list()[0].shape == averaged.list()[0].shape
    assert aligned.list()[0].hdr_ext["ProcessingApplied"][-1]["Method"].endswith(
        "correct_frequency_and_phase")
    broad = suspect.apodization(averaged, line_broadening=8.0)
    assert np.abs(np.asarray(broad.list()[0][:])[..., -1, 0]).max() < \
        np.abs(np.asarray(averaged.list()[0][:])[..., -1, 0]).max()
    watered = suspect.water_removal(averaged, rank=8, window=(1.5, 2.5))   # the one peak is "water"
    assert np.abs(np.asarray(watered.list()[0][:])).max() < \
        0.2 * np.abs(np.asarray(averaged.list()[0][:])).max()


def test_the_catalogue_lists_steps_and_readers():
    assert set(suspect.WRITTEN) <= set(suspect.bridge.steps()) <= set(suspect.bridge.available())
    assert "load_twix" in suspect.bridge.available() and "rats" not in suspect.bridge.available()
    with pytest.raises(AttributeError, match="no function"):
        suspect.bridge.nonsense
