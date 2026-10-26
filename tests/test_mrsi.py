####################################################################################################
#                                           test_mrsi.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 26/10/26                                                                                #
#                                                                                                  #
# Purpose: A spatial NIfTI-MRS comes apart into placed single-voxel spectra with the header and   #
#          the voxel in their history, and fits of them go back on the grid as maps.              #
#                                                                                                  #
####################################################################################################
import numpy as np
from nifti_mrs.create_nmrs import gen_nifti_mrs
from nifti_mrs_plus import NIfTI_MRS_Plus

from mrsuite import mrsi
from mrsuite.core.result import FitResult


def _slice():
    t = np.arange(256) * 2.5e-4
    fid = np.exp(2j * np.pi * (2.0 - 4.65) * 123.2 * t - t / 0.05)
    data = np.zeros((3, 2, 1, 256), complex)
    for x in range(3):
        for y in range(2):
            data[x, y, 0] = (1 + x + 3 * y) * fid
    data[2, 1, 0] = 0                                             # an empty voxel
    affine = np.diag([10.0, 10.0, 10.0, 1.0])
    affine[:3, 3] = [-15.0, -10.0, 35.0]
    item = gen_nifti_mrs(data, 2.5e-4, 123.2, affine=affine)
    item.add_hdr_field("EchoTime", 0.03)
    return NIfTI_MRS_Plus([item])


def test_the_voxels_come_apart_placed_and_noted():
    spectra, index = mrsi.voxels(_slice())
    assert len(spectra) == 5 and index.shape == (5, 3) and [2, 1, 0] not in index.tolist()
    assert index[0].tolist() == [1, 1, 0] and index[-1].tolist() == [0, 0, 0], "strongest first"
    one = spectra.list()[-1]
    assert one.shape == (1, 1, 1, 256) and one.hdr_ext["EchoTime"] == 0.03
    assert np.allclose(np.asarray(one.voxToWorldMat)[:3, 3], [-15.0 + 10 * index[-1][0],
                                                              -10.0 + 10 * index[-1][1], 35.0])
    step = one.hdr_ext["ProcessingApplied"][-1]
    assert step["Method"] == "mrsuite.mrsi.voxels" and str(index[-1].tolist()) in step["Details"]
    few, _ = mrsi.voxels(_slice(), threshold=0.5)
    assert len(few) == 3, "the three strongest voxels"
    mask = np.zeros((3, 2, 1), bool)
    mask[0, 0, 0] = mask[2, 1, 0] = True
    masked, where = mrsi.voxels(_slice(), mask=mask)
    assert len(masked) == 2 and where.tolist() == [[0, 0, 0], [2, 1, 0]], "the mask as it is, empty last"
    mask[1, 1, 0] = True
    _, where = mrsi.voxels(_slice(), mask=mask)
    assert where.tolist() == [[1, 1, 0], [0, 0, 0], [2, 1, 0]], "strongest first, empty last"


def test_the_fits_go_back_on_the_grid_as_maps():
    spectra, index = mrsi.voxels(_slice())
    results = [FitResult("test", ["NAA", "Cr"], {"raw": {"NAA": float(k), "Cr": 1.0}})
               for k in range(len(spectra))]
    figure = mrsi.maps(results, index, shape=(3, 2, 1))
    assert len(figure.axes) >= 2 and figure.axes[0].get_title().startswith("NAA")
    image = np.ma.filled(figure.axes[0].images[0].get_array(), np.nan)
    assert image.shape == (2, 3) and np.isnan(image[1, 2]), "y by x, the empty voxel blank"
    assert mrsi.maps(results, index, metabolites=["Cr"]).axes[0].get_title() == "Cr (raw)"


def test_the_truth_goes_under_each_map():
    spectra, index = mrsi.voxels(_slice())
    results = [FitResult("test", ["NAA", "Cr"], {"raw": {"NAA": float(k), "Cr": 1.0}})
               for k in range(len(spectra))]
    truth = {"NAA": np.arange(6, dtype=float).reshape(3, 2, 1), "Cr": np.ones((3, 2, 1))}
    figure = mrsi.maps(results, index, truth=truth)
    titles = [ax.get_title() for ax in figure.axes if ax.get_title()]
    assert titles == ["NAA (raw)", "Cr (raw)", "NAA (truth)", "Cr (truth)"], titles
    shown = np.ma.filled(figure.axes[2].images[0].get_array(), np.nan)
    assert np.isnan(shown[1, 2]), "the truth only where a voxel was fitted"
