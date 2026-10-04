####################################################################################################
#                                              io.py                                               #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-02                                                                              #
#                                                                                                  #
# Purpose: Reads raw scanner data into NIfTI-MRS+ through spec2nii, the reference converter to     #
#          NIfTI-MRS, so data from any vendor reaches a pipeline the same way.                     #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Union

import nibabel as nib
import numpy as np
from nifti_mrs.nifti_mrs import NIFTI_MRS, NotNIFTI_MRS
from nifti_mrs_plus import NIfTI_MRS_Plus

from mrsuite.core import provenance


#**********#
#   load   #
#**********#
def load(*files: Any, format: str = "auto",
         **options: Any) -> Dict[str, Union[NIfTI_MRS_Plus, nib.Nifti1Image]]:
    """
    Raw scanner data as NIfTI-MRS+, converted by spec2nii.

    One acquisition can give several outputs - a GE P-file holds the water-suppressed
    spectra and their water reference - so each comes back under spec2nii's name for it,
    as a batch of one with all its dimensions (coils, averages, ...) as acquired. What is
    not spectra, such as DICOM's ``voi=True`` mask of where the voxel sits, comes back as
    a plain NIfTI image.

    Args:
        *files: The raw files as spec2nii takes them for the format: a P-file, a Twix or
            RDA file, a DICOM folder, a Philips .SDAT and its .SPAR, ...
        format (str): spec2nii's subcommand, e.g. 'ge', 'twix', 'philips', 'dicom',
            'rda', 'bruker', 'varian', 'jmrui', 'text', 'raw'. 'auto' lets spec2nii tell
            Twix, RDA, SDAT/SPAR, GE and DICOM apart itself, which it calls experimental.
        **options: That subcommand's options by name, e.g. ``evalinfo="image"`` for
            Twix; True for a flag, a list for an option taking several values.

    Returns:
        Dict[str, Union[NIfTI_MRS_Plus, nib.Nifti1Image]]: Output name -> the data, its
        processing history starting with the conversion, or the image.

    Raises:
        RuntimeError: If spec2nii fails or writes nothing, with its own message.
    """
    import spec2nii

    flags = []
    for name, value in options.items():
        if value is True:
            flags.append(f"--{name}")
        elif isinstance(value, (list, tuple)):
            flags += [f"--{name}", *map(str, value)]
        elif value is not False and value is not None:
            flags += [f"--{name}", str(value)]

    with tempfile.TemporaryDirectory() as folder:
        # spec2nii is a command line tool only; run it with this Python, so the version
        # recorded below is the one that converted
        run = subprocess.run([sys.executable, "-c", "from spec2nii.spec2nii import main; main()",
                              format, *map(str, files), *flags, "-o", folder],
                             capture_output=True, text=True)
        if run.returncode != 0:
            raise RuntimeError(f"spec2nii {format} failed:\n{run.stderr.strip()}")

        out = {}
        for path in sorted(Path(folder).glob("*.nii.gz")):   # its only output, nifti1 too
            name = path.name.removesuffix(".nii.gz")
            try:
                nifti = NIFTI_MRS(str(path)).copy()   # read now: the folder goes
            except NotNIFTI_MRS:
                image = nib.load(path)
                out[name] = type(image)(np.asarray(image.dataobj), image.affine, image.header)
                continue
            provenance.append(nifti, provenance.entry(
                "spec2nii", spec2nii.__version__, f"spec2nii {format}",
                {"files": [Path(f) for f in files], "options": options}))
            out[name] = NIfTI_MRS_Plus([nifti])
    if not out:   # some modes only list what is there, some find nothing to convert
        said = "\n".join(text.strip() for text in (run.stdout, run.stderr) if text.strip())
        raise RuntimeError(f"spec2nii {format} wrote nothing for {list(map(str, files))}"
                           + (f":\n{said}" if said else ""))
    return out
