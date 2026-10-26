####################################################################################################
#                                        test_fsl_mrsi.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 26/10/26                                                                                #
#                                                                                                  #
# Purpose: FSL-MRS's MRSI fit runs on a small grid and comes back as one result per voxel, with   #
#          the grid's indices, strongest first.                                                    #
#                                                                                                  #
####################################################################################################
import numpy as np
import pytest
from nifti_mrs.create_nmrs import gen_nifti_mrs
from nifti_mrs_plus import NIfTI_MRS_Plus

from mrsuite import io
from mrsuite.bridges import fsl_mrs
from mrsuite.datasets import challenge2016

needs_2016 = pytest.mark.skipif(not challenge2016.ROOT.is_dir(), reason="the 2016 challenge is not fetched")


@needs_2016
def test_fsl_mrsi_fits_a_grid_voxel_by_voxel():
    spectra, _, _ = challenge2016.load(n=4)
    items = spectra.list()
    one = items[0]
    grid = np.stack([np.asarray(i[:]).reshape(-1) for i in items]).reshape(2, 2, 1, -1)
    grid[1, 1, 0] *= 3                                        # the strongest voxel
    item = gen_nifti_mrs(grid, one.dwelltime, one.spectrometer_frequency[0])
    mask = np.array([[1, 1], [0, 1]], bool).reshape(2, 2, 1)
    results, index = fsl_mrs.fit_mrsi(NIfTI_MRS_Plus([item]), io.basis(challenge2016.BASIS),
                                      mask, workers=2, no_rescale=True)   # a flag, through options
    assert len(results) == 3 and index.shape == (3, 3) and index[0].tolist() == [1, 1, 0]
    assert [1, 0, 0] not in index.tolist(), "the masked-out voxel"
    first = results[0]
    assert first.method == "fsl_mrs:fsl_mrsi" and first.concentrations["raw"]["NAA"] > 0
    assert set(first.concentrations) >= {"raw", "internal"}
    assert first.crlb()["NAA"] > 0 and np.isfinite(first.qc["snr"])
    assert first.provenance[-1]["Program"] == "FSL-MRS" and first.provenance[-1]["Method"] == "fsl_mrsi"
    weaker = next(r for r, at in zip(results, index.tolist()) if at == [0, 0, 0])
    assert first.concentrations["raw"]["NAA"] > 2 * weaker.concentrations["raw"]["NAA"], "scaled thrice"
