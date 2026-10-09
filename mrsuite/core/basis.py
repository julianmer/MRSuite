####################################################################################################
#                                             basis.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-09                                                                              #
#                                                                                                  #
# Purpose: A basis set as one object, whatever format it came in: one FID per metabolite and what  #
#          fitting it needs to know. mrsuite.io.basis reads it through BasiC; every fit takes it,  #
#          as the LCModel .BASIS MRSuite writes for it once, in its own folder.                    #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# the ppm at 0 Hz of a basis that does not state its own, as LCModel and NIfTI-MRS take 1H
CENTRE = 4.65


#**************************************************************************************************#
#                                           Class Basis                                            #
#**************************************************************************************************#
#                                                                                                  #
# A basis set: one FID per metabolite, with what fitting it needs to know.                         #
#                                                                                                  #
#**************************************************************************************************#
@dataclass(eq=False, repr=False)
class Basis:
    """A basis set: one FID per metabolite, with what fitting it needs to know.

    Attributes:
        names (List[str]): The metabolites, in the files' order.
        fids (np.ndarray): (metabolites, points), complex, as BasiC holds them: a peak at
            f Hz in their spectrum sits at ``centre + f / frequency`` ppm.
        bandwidth (float): Spectral width in Hz.
        frequency (float): Spectrometer frequency in MHz.
        centre (Optional[float]): The ppm at 0 Hz, where the files state it (Osprey's own
            sets: 3.0); None for CENTRE.
        te (Optional[float]): Echo time in ms, where stated or given.
        seq (Optional[str]): Sequence, where stated.
        source (Dict[str, Any]): Where it came from: the files (portable path and
            checksum), the format BasiC read them as, BasiC's commit, and what the caller
            gave beyond the files.
    """

    names: List[str]
    fids: np.ndarray
    bandwidth: float
    frequency: float
    centre: Optional[float] = None
    te: Optional[float] = None
    seq: Optional[str] = None
    source: Dict[str, Any] = field(default_factory=dict)

    #**********#
    #   repr   #
    #**********#
    def __repr__(self) -> str:
        return f"Basis({self.names}, {self.fids.shape[-1]} points)"

    #*************#
    #   spectra   #
    #*************#
    def spectra(self) -> np.ndarray:
        """(metabolites, points): each FID's spectrum, on ``ppm()``."""
        return np.fft.fftshift(np.fft.fft(self.fids, axis=1), axes=1)

    #*********#
    #   ppm   #
    #*********#
    def ppm(self) -> np.ndarray:
        """The spectra's ppm axis, rising."""
        hz = np.fft.fftshift(np.fft.fftfreq(self.fids.shape[1], 1 / self.bandwidth))
        return (CENTRE if self.centre is None else self.centre) + hz / self.frequency

    #**************#
    #   checksum   #
    #**************#
    def checksum(self) -> str:
        """SHA-256 of the FIDs and everything said about them but where they came from."""
        said = [self.names, self.bandwidth, self.frequency, self.centre, self.te, self.seq]
        digest = hashlib.sha256(np.ascontiguousarray(self.fids, dtype=complex).tobytes())
        digest.update(json.dumps(said, default=float).encode())   # NumPy numbers too
        return digest.hexdigest()

    #*************#
    #   lcmodel   #
    #*************#
    def lcmodel(self) -> Path:
        """
        The basis as the LCModel .BASIS every fit takes, written by BasiC once per basis
        into MRSuite's own folder, $MRSUITE_HOME (else ~/.mrsuite)/basis, by checksum and
        BasiC's commit, so a pinned BasiC that writes differently writes anew.

        Written beside its place and moved in whole, so fits running in parallel never
        read half a file.
        """
        from mrsuite import externals, io
        basic = externals.REGISTRY["basic"][1][:7]
        path = externals.home() / "basis" / f"{self.checksum()[:16]}-{basic}.BASIS"
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            part = tempfile.mkdtemp(prefix=".part", dir=path.parent)
            try:
                written, = io.convert_basis(self, part, "lcmodel", output="basis")
                os.replace(written, path)
            finally:
                shutil.rmtree(part, ignore_errors=True)
        return path


#***********#
#   added   #
#***********#
def added(basis: Basis, spectrum: Any, name: str = "MM") -> Basis:
    """
    A basis with a measured spectrum as one more function, e.g. a macromolecule
    acquisition, as LCModel's manual and FSL-MRS's basis tools take one: the spectrum put
    on the basis' own ppm axis, resampled where its bandwidth or points differ, and taken
    back to an FID at the basis' dwell time.

    Args:
        basis (Basis): The basis set.
        spectrum: One NIfTI-MRS spectrum (or a batch of one), processed as the data it
            will help fit: coil combined, averaged, its residual water removed, phased.
        name (str): The function's name.

    Returns:
        Basis: A new one; its source notes what was added and where it came from.

    Raises:
        ValueError: For a spectrum with coils, averages or other higher dimensions.
    """
    from nifti_mrs_plus import NIfTI_MRS_Plus
    from mrsuite.core import provenance
    if isinstance(spectrum, NIfTI_MRS_Plus):           # a batch of one, as the GUI hands it
        if len(spectrum.list()) != 1:
            raise ValueError(f"a basis takes one spectrum, not {len(spectrum.list())}")
        spectrum = spectrum.list()[0]
    data = np.asarray(spectrum[:])
    if data.shape[:3] != (1, 1, 1) or data.ndim != 4:
        raise ValueError(f"a basis takes one spectrum, coil combined and averaged, not "
                         f"one of shape {data.shape}")
    fid = data.reshape(-1)
    spec = np.fft.fftshift(np.fft.fft(fid))
    hz = np.fft.fftshift(np.fft.fftfreq(fid.size, spectrum.dwelltime))
    ppm = hz / spectrum.spectrometer_frequency[0] + spectrum.SpecFreqChemShift
    own = basis.ppm()                                          # zero beyond what was measured
    moved = np.interp(own, ppm, spec.real, left=0, right=0) \
        + 1j * np.interp(own, ppm, spec.imag, left=0, right=0)
    source = {**basis.source, "added": [*basis.source.get("added", []),
                                        {"name": name, **provenance.describe(spectrum)}]}
    return Basis([*basis.names, name], np.vstack([basis.fids, np.fft.ifft(np.fft.ifftshift(moved))[None]]),
                 basis.bandwidth, basis.frequency, basis.centre, basis.te, basis.seq, source)


#*************#
#   as file   #
#*************#
def as_file(basis: Any) -> Any:
    """A basis as a fit reads it: a Basis as its .BASIS, anything else as it is."""
    return basis.lcmodel() if isinstance(basis, Basis) else basis
