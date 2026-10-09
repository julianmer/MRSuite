####################################################################################################
#                                            conftest.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-08-08                                                                              #
#                                                                                                  #
# Purpose: Shared fixtures and skip markers for the test suite. Octave and MATLAB are both         #
#          optional: suites that need one skip cleanly without it, so the suite is usable on       #
#          a machine that has neither.                                                             #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import glob
import os
import shutil

import nibabel as nib
import numpy as np
import pytest

# a single already-averaged, already-combined single-voxel spectrum: the degenerate
# shape that exposes most conversion and dispatch problems
CHALLENGE = "data/2016_fitting_challenge/datasets_JMRUI/dataset1_WS.txt"
CHALLENGE_BASIS = "data/2016_fitting_challenge/basisset_LCModel/press3T_30ms.BASIS"

HAVE_OCTAVE = shutil.which("octave-cli") is not None


#************#
#   matlab   #
#************#
def _matlab():
    """The matlab launcher: on PATH, or else the newest macOS install in /Applications,
    which the installer does not put on PATH."""
    installed = sorted(glob.glob("/Applications/MATLAB_R*.app/bin/matlab"))
    return shutil.which("matlab") or (installed[-1] if installed else None)


MATLAB = _matlab()
HAVE_MATLAB = MATLAB is not None

needs_octave = pytest.mark.skipif(not HAVE_OCTAVE, reason="octave-cli not on PATH")
needs_matlab = pytest.mark.skipif(not HAVE_MATLAB, reason="matlab not found")


#**************#
#   spectrum   #
#**************#
@pytest.fixture(scope="session")
def spectrum():
    """One NIfTI-MRS spectrum from the challenge submodule."""
    from fsl_mrs.utils import mrs_io
    if not os.path.exists(CHALLENGE):
        pytest.skip("challenge submodule not checked out")
    return mrs_io.read_FID(CHALLENGE)


#*************#
#   spectra   #
#*************#
@pytest.fixture(scope="session")
def spectra():
    """Two spectra wrapped as NIfTI-MRS+, to exercise batching."""
    from fsl_mrs.utils import mrs_io
    from nifti_mrs_plus import NIfTI_MRS_Plus
    if not os.path.exists(CHALLENGE):
        pytest.skip("challenge submodule not checked out")
    return NIfTI_MRS_Plus([mrs_io.read_FID(CHALLENGE), mrs_io.read_FID(CHALLENGE)])


#***********#
#   basis   #
#***********#
@pytest.fixture(scope="session")
def basis():
    """The challenge's own LCModel basis set."""
    from fsl_mrs.utils import mrs_io
    if not os.path.exists(CHALLENGE_BASIS):
        pytest.skip("challenge submodule not checked out")
    return mrs_io.read_basis(CHALLENGE_BASIS)


#*****************#
#   fsl-mrs fit   #
#*****************#
@pytest.fixture(scope="session")
def fsl_mrs_fit():
    """A real FSL-MRS fit of the challenge spectrum.

    Session-scoped and shared: fitting takes seconds, and every test that needs one
    needs the same one. A synthetic stand-in would not exercise the parts of FitRes
    that only exist once a fit has actually converged.
    """
    from fsl_mrs.utils import mrs_io, fitting
    if not (os.path.exists(CHALLENGE) and os.path.exists(CHALLENGE_BASIS)):
        pytest.skip("challenge submodule not checked out")

    mrs = mrs_io.read_FID(CHALLENGE).mrs(basis=mrs_io.read_basis(CHALLENGE_BASIS))
    mrs.processForFitting()
    return fitting.fit_FSLModel(mrs, method="Newton", ppmlim=(0.2, 4.2), baseline_order=2)


#***********#
#   study   #
#***********#
@pytest.fixture
def study(tmp_path, spectrum):
    """A folder as a study comes: two BIDS subjects, a plain-named one, a T1 and
    derivatives that are not data."""
    from nifti_mrs_plus import NIfTI_MRS_Plus

    once, = NIfTI_MRS_Plus([spectrum]).save_nifti(str(tmp_path / ".once"), prefix="tmp")

    def put(rel):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(once, path)

    for rel in ["sub-01/ses-01/mrs/sub-01_ses-01_acq-press_svs.nii.gz",
                "sub-01/ses-01/mrs/sub-01_ses-01_acq-press_mrsref.nii.gz",
                "sub-02/mrs/sub-02_acq-press_svs.nii.gz",
                "sub-02/mrs/sub-02_acq-press_mrsref.nii.gz",
                "extra/press_water_PFL.nii.gz", "extra/press_metab_PFL.nii.gz",
                "extra/press_mm_PFL.nii.gz", "derivatives/osprey/sub-01_svs.nii.gz"]:
        put(rel)
    anat = tmp_path / "sub-01/ses-01/anat"
    anat.mkdir(parents=True)
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4)), np.eye(4)), anat / "sub-01_ses-01_T1w.nii.gz")
    return tmp_path
