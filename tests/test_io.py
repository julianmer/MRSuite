####################################################################################################
#                                            test_io.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-02                                                                              #
#                                                                                                  #
# Purpose: Raw scanner data through spec2nii into NIfTI-MRS+, on the GE P-file FID-A ships as      #
#          example data: every output, every dimension, the conversion in the history, and a       #
#          pipeline straight after.                                                                #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import json
from pathlib import Path

import numpy as np
import pytest

from mrsuite import fsl_mrs, io
from mrsuite.core.pipeline import Pipeline

PFILE = (Path(__file__).resolve().parents[1] / "externals" / "fid-a" / "exampleData" / "GE"
         / "sample01_press" / "press" / "P17920.7")

needs_pfile = pytest.mark.skipif(not PFILE.exists(), reason="FID-A submodule not checked out")


#**********#
#   scan   #
#**********#
@pytest.fixture(scope="module")
def scan():
    return io.load(PFILE, format="ge")


#*************#
#   outputs   #
#*************#
@needs_pfile
def test_a_pfile_gives_its_spectra_and_water_with_every_dimension(scan):
    assert sorted(scan) == ["P17920", "P17920_ref"]
    data, water = scan["P17920"].list()[0], scan["P17920_ref"].list()[0]
    assert data.shape == (1, 1, 1, 4096, 32, 16)     # every coil and transient, as acquired
    assert data.dim_tags == ["DIM_COIL", "DIM_DYN", None]
    assert water.shape[-1] == 2
    assert np.abs(np.asarray(data[:])).max() > 0     # read into memory, not left on disk


#****************#
#   provenance   #
#****************#
@needs_pfile
def test_the_conversion_starts_the_history(scan):
    import spec2nii

    nifti = scan["P17920"].list()[0]
    entry = nifti.hdr_ext["ProcessingApplied"][-1]
    assert (entry["Program"], entry["Version"]) == ("spec2nii", spec2nii.__version__)
    assert json.loads(entry["Details"])["files"][0]["sha256"], "and which raw file it was"
    # the spec2nii recorded is the one that converted, not another on PATH
    assert nifti.hdr_ext["ConversionMethod"] == f"spec2nii v{spec2nii.__version__}"


#*************#
#   options   #
#*************#
@needs_pfile
def test_spec2niis_own_options_pass_through():
    """A value, a flag and a list each reach spec2nii as it takes them, and a name with
    dots keeps the spectra apart from their water."""
    options = {"fileout": "sub-01.press", "nifti1": True, "override_nucleus": ["1H"]}
    scans = io.load(PFILE, format="ge", **options)

    assert sorted(scans) == ["sub-01.press", "sub-01.press_ref"]
    nifti = scans["sub-01.press"].list()[0]
    assert nifti.shape[-1] == 16 and nifti.header["sizeof_hdr"] == 348   # NIfTI-1
    assert nifti.hdr_ext["ResonantNucleus"] == ["1H"]
    assert json.loads(nifti.hdr_ext["ProcessingApplied"][-1]["Details"])["options"] == options


#*************#
#   failure   #
#*************#
def test_a_failed_conversion_says_why():
    with pytest.raises(RuntimeError, match="UnknownPfile"):
        io.load(__file__, format="ge")


#*********#
#   voi   #
#*********#
def test_an_output_that_is_not_spectra_comes_back_as_an_image(monkeypatch):
    """DICOM's voi=True also writes a mask of where the voxel sits, a NIfTI that is not
    NIfTI-MRS. No DICOM data is at hand, so spec2nii is stood in for by its outputs."""
    import subprocess

    import nibabel as nib
    from nifti_mrs.create_nmrs import gen_nifti_mrs
    from nifti_mrs_plus import NIfTI_MRS_Plus

    def spec2nii(args, **kwargs):
        folder = Path(args[args.index("-o") + 1])
        gen_nifti_mrs(np.ones((1, 1, 1, 64), complex), 2.5e-4, 123.2,
                      nucleus="1H").save(str(folder / "svs.nii.gz"))
        nib.save(nib.Nifti1Image(np.ones((1, 1, 1), np.uint8), np.eye(4)),
                 folder / "svs_voi.nii.gz")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(io.subprocess, "run", spec2nii)
    scans = io.load("scan", format="dicom", voi=True)

    assert isinstance(scans["svs"], NIfTI_MRS_Plus)
    assert isinstance(scans["svs_voi"], nib.Nifti1Image)
    assert np.asarray(scans["svs_voi"].dataobj).sum() == 1   # read before the folder went


#*************#
#   nothing   #
#*************#
@needs_pfile
def test_a_conversion_that_writes_nothing_is_an_error():
    """spec2nii's Bruker FID mode finds nothing in this folder and exits as if it had."""
    with pytest.raises(RuntimeError, match="wrote nothing"):
        io.load(PFILE.parents[2] / "Bruker" / "sample01_press" / "press", format="bruker",
                mode="FID")


#**************#
#   pipeline   #
#**************#
@needs_pfile
def test_a_pipeline_runs_on_what_was_loaded(scan):
    out = Pipeline([fsl_mrs.step.coilcombine(), fsl_mrs.step.average(dim="DIM_DYN")]
                   ).run(scan["P17920"])

    assert out.shape == (1, 1, 1, 1, 4096)
