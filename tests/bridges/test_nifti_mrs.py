####################################################################################################
#                                         test_nifti_mrs.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 26/10/26                                                                                #
#                                                                                                  #
# Purpose: nifti_mrs_tools' dimension operations as MRSuite steps: dropped, split and              #
#          conjugated spectra, recorded in the history.                                            #
#                                                                                                  #
####################################################################################################
import numpy as np
from nifti_mrs.create_nmrs import gen_nifti_mrs
from nifti_mrs_plus import NIfTI_MRS_Plus

from mrsuite.bridges import nifti_mrs


#***********#
#   coils   #
#***********#
def _coils():
    t = np.arange(256) * 2.5e-4
    fid = np.exp(2j * np.pi * (2.0 - 4.65) * 123.2 * t - t / 0.05)
    fids = np.stack([fid, 2 * fid], -1).reshape(1, 1, 1, -1, 2)
    return NIfTI_MRS_Plus([gen_nifti_mrs(fids, 2.5e-4, 123.2, dim_tags=["DIM_COIL", None, None])])


#***********#
#   steps   #
#***********#
def test_the_dimension_tools_run_as_steps_and_are_recorded():
    raw = _coils()

    one = nifti_mrs.bridge.remove_dim(raw, "DIM_COIL")
    kept, rest = nifti_mrs.bridge.split(raw, "DIM_COIL", 0)
    mirrored = nifti_mrs.bridge.conjugate(raw)

    assert one.list()[0].shape == (1, 1, 1, 256) and "DIM_COIL" not in one.list()[0].dim_tags
    assert np.allclose(np.asarray(one.list()[0][:]).ravel(), np.asarray(raw.list()[0][:])[..., 0].ravel())
    assert kept.list()[0].shape[4] == 1 and rest.list()[0].shape[4] == 1
    assert np.allclose(np.asarray(mirrored.list()[0][:]).imag, -np.asarray(raw.list()[0][:]).imag)
    step = one.list()[0].hdr_ext["ProcessingApplied"][-1]
    assert step["Program"] == "nifti_mrs_tools" and "remove_dim" in step["Method"]
    assert sorted(nifti_mrs.bridge.steps()) == ["conjugate", "remove_dim", "reorder", "reshape",
                                                "split"]
