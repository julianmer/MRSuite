####################################################################################################
#                                              io.py                                               #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-02                                                                              #
#                                                                                                  #
# Purpose: Reads raw scanner data into NIfTI-MRS+ through spec2nii, the reference converter to     #
#          NIfTI-MRS, so data from any vendor reaches a pipeline the same way; NIfTI-MRS files     #
#          themselves, each spectrum's history starting with the file it came from; and basis      #
#          sets in any format through BasiC, as one Basis every fit takes.                         #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import json
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import nibabel as nib
import numpy as np
from nifti_mrs.nifti_mrs import NIFTI_MRS, NotNIFTI_MRS
from nifti_mrs_plus import NIfTI_MRS_Plus

from mrsuite import externals
from mrsuite.core import provenance
from mrsuite.core.basis import Basis

DRIVER = Path(__file__).with_name("basic_driver.py")


#**********#
#   load   #
#**********#
def load(*files: Any, format: str = "auto",
         **options: Any) -> Dict[str, Union[NIfTI_MRS_Plus, nib.Nifti1Image]]:
    """
    Raw scanner data as NIfTI-MRS+, converted by spec2nii; NIfTI-MRS files as they are.

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
        processing history starting with the conversion, or the image. NIfTI-MRS files
        need no conversion: they come back as one batch, 'spectra', through ``read``.

    Raises:
        RuntimeError: If spec2nii fails or writes nothing, with its own message.
    """
    if format == "auto" and files and all(str(f).endswith((".nii", ".nii.gz")) for f in files):
        return {"spectra": read(*files)}
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


#**********#
#   read   #
#**********#
def read(*paths: Any) -> NIfTI_MRS_Plus:
    """
    NIfTI-MRS files as one NIfTI-MRS+.

    Args:
        *paths: The files, one spectrum each, with all its dimensions.

    Returns:
        NIfTI_MRS_Plus: The spectra, in the order given, each one's processing history
        starting with the file it was read from, by its portable path and its checksum.
    """
    import nifti_mrs

    spectra = []
    for path in paths:
        spectrum = NIFTI_MRS(str(path))
        provenance.append(spectrum, provenance.entry(
            "MRSuite", provenance.mrsuite_version(), "mrsuite.io.read",
            {"file": Path(path), "reader": f"nifti-mrs {nifti_mrs.__version__}"}))
        spectra.append(spectrum)
    return NIfTI_MRS_Plus(spectra)


#***********#
#   basis   #
#***********#
def basis(files: Any, format: str = "auto", sw: Optional[float] = None,
          sf: Optional[float] = None, te: Optional[float] = None) -> Basis:
    """
    A basis set, in any format BasiC reads, as one Basis every fit takes.

    BasiC tells the format from the files themselves, as ``load`` lets spec2nii do: an
    LCModel .BASIS, jMRUI .txt, NIfTI-MRS, FID-A, Osprey, MARSS or MRSCloud .mat, ... What
    the files state wins; ``sw``, ``sf`` and ``te`` fill only what they leave out.

    Args:
        files: The basis files: one holding the whole set, or a list, one per metabolite.
        format (str): BasiC's name for the format, e.g. 'MRSCloud .mat', where it cannot
            tell; 'auto' lets it tell.
        sw (Optional[float]): Spectral width in Hz, where the files do not state it.
        sf (Optional[float]): Spectrometer frequency in MHz, where the files do not state it.
        te (Optional[float]): Echo time in ms, where the files do not state it.

    Returns:
        Basis: The basis set, with the files it came from.

    Raises:
        RuntimeError: If BasiC cannot tell the format or read the set, or the files lack
            what it needs, with what to give.
    """
    files = [_whole(f) for f in ([files] if isinstance(files, (str, Path)) else files)]
    with tempfile.TemporaryDirectory() as folder:
        result = Path(folder) / "fids.npz"
        said = _basic(task="read", files=[str(f) for f in files], format=format, sw=sw, sf=sf,
                      te=te, result=str(result))
        fids = np.load(result)["fids"]
    given = {"sw": sw, "sf": sf, "te": te}
    source = {"files": [provenance.portable_file(f) for f in files], "format": said.pop("format"),
              "reader": f"BasiC {externals.REGISTRY['basic'][1][:7]}",
              "given": {k: v for k, v in given.items() if v is not None}}
    return Basis(fids=fids, source=source, **said)


#*******************#
#   convert basis   #
#*******************#
def convert_basis(basis: Basis, folder: Any, tool: str, output: str = "default") -> List[Path]:
    """
    A basis set written for a tool BasiC writes for.

    Args:
        basis (Basis): The basis set, e.g. from ``basis``.
        folder: Where to write.
        tool (str): BasiC's id of the tool, e.g. 'osprey', 'fsLmrs', 'jmrui', 'niftimrs',
            'lcmodel', 'fida'; a wrong one is told the others.
        output (str): The tool's output format where it has several, e.g. 'basis', 'raw' or
            'both' for 'lcmodel'; 'default' is BasiC's choice. An echo time the basis does
            not hold is written as BasiC writes it (30 ms; 26 ms for OXSA).

    Returns:
        List[Path]: The files written, as absolute paths. A metabolite BasiC's writer fails
        it goes on without, and says so in a warning.

    Raises:
        RuntimeError: For a tool or output BasiC does not write, or if it wrote nothing.
    """
    folder = _whole(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        fids = Path(scratch) / "fids.npz"
        np.savez(fids, fids=basis.fids)
        written = _basic(task="write", fids=str(fids), folder=str(folder), tool=tool,
                         output=output, names=list(basis.names), bandwidth=basis.bandwidth,
                         frequency=basis.frequency, centre=basis.centre, te=basis.te,
                         seq=basis.seq)
    return [_whole(p) for p in written]


#***********#
#   basic   #
#***********#
def _basic(**job: Any) -> Any:
    """One job through basic_driver.py, without leaving bytecode in BasiC's folder; its
    answer. What BasiC warns of comes along as warnings."""
    job["root"] = str(externals.ensure("basic") / "basis_converter")
    run = subprocess.run([sys.executable, "-B", str(DRIVER), json.dumps(job, default=float)],
                         capture_output=True, text=True)
    said = run.stderr.strip().splitlines()
    for line in said:
        if "WARNING" in line:
            warnings.warn(f"BasiC: {line.strip()}")
    if run.returncode != 0:
        raise RuntimeError(f"BasiC could not {job['task']} the basis set: "
                           + (said[-1] if said else f"exit status {run.returncode}"))
    return json.loads(run.stdout)


#***********#
#   whole   #
#***********#
def _whole(path: Any) -> Path:
    """A path from the home folder or the working one, as the driver's process needs it."""
    return Path(path).expanduser().absolute()
