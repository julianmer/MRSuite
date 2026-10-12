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
import warnings
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


#************#
#   merged   #
#************#
def merged(basis: Basis, more: Dict[str, Any], keep: Optional[Dict[str, str]] = None) -> Basis:
    """
    A basis with more functions: other basis sets', and measured spectra, e.g. a
    macromolecule acquisition, as LCModel's manual and FSL-MRS's basis tools take one.
    Each is put on the basis' own ppm axis, resampled where its bandwidth or points
    differ, and taken back to an FID at the basis' dwell time, its amplitude kept.

    Args:
        basis (Basis): The basis set whose axis the others take; its functions come from
            'basis'.
        more (Dict[str, Any]): Name -> another basis set, or one NIfTI-MRS spectrum (or a
            batch of one) processed as the data it will help fit: coil combined, averaged,
            its residual water removed, phased. A spectrum's function takes its name.
        keep (Optional[Dict[str, str]]): The functions to keep, each name -> where it comes
            from: 'basis' or a name of ``more``. Every function by default, which a name
            in two places refuses.

    Returns:
        Basis: A new one; its source notes what was added and where it came from.

    Raises:
        ValueError: For a name in two places that keep does not settle, a function keep
            names that is not there, or a spectrum with coils, averages or other higher
            dimensions.
    """
    from mrsuite.core import provenance

    if "basis" in more:
        raise ValueError("'basis' names the basis set itself; give the others other names")
    own = basis.ppm()
    parts: Dict[str, Dict[str, np.ndarray]] = {"basis": dict(zip(basis.names, basis.fids))}
    added = []
    for where, value in more.items():
        if isinstance(value, Basis):
            if None not in (value.te, basis.te) and value.te != basis.te:
                warnings.warn(f"{where} is for TE {value.te} ms, the basis for {basis.te} ms")
            spectra, ppm, names = value.spectra(), value.ppm(), value.names
            bandwidth = value.bandwidth
            added.append({"from": where, "names": list(names), "source": value.source})
        else:
            spectrum = _one(value)
            fid = np.asarray(spectrum[:]).reshape(-1)
            spectra = np.fft.fftshift(np.fft.fft(fid))[None]
            hz = np.fft.fftshift(np.fft.fftfreq(fid.size, spectrum.dwelltime))
            ppm = hz / spectrum.spectrometer_frequency[0] + spectrum.SpecFreqChemShift
            names, bandwidth = [where], 1 / spectrum.dwelltime
            added.append({"from": where, "names": [where], **provenance.describe(spectrum)})
        # zero beyond what was measured; a spectrum's values grow with its bandwidth (its
        # points per second of signal), so they are scaled to the basis' to keep the FID's
        moved = [(np.interp(own, ppm, s.real, left=0, right=0)
                  + 1j * np.interp(own, ppm, s.imag, left=0, right=0))
                 * basis.bandwidth / bandwidth for s in spectra]
        parts[where] = {n: np.fft.ifft(np.fft.ifftshift(m)) for n, m in zip(names, moved)}

    if keep is None:
        places: Dict[str, List[str]] = {}
        for where, functions in parts.items():
            for name in functions:
                places.setdefault(name, []).append(where)
        twice = {n: p for n, p in places.items() if len(p) > 1}
        if twice:
            raise ValueError("a name in two places, keep one: " + "; ".join(
                f"{n} in {', '.join(p)}" for n, p in twice.items()))
        keep = {n: p[0] for n, p in places.items()}
    gone = [f"{n} in {w}" for n, w in keep.items() if n not in parts.get(w, {})]
    if gone:
        raise ValueError(f"not there to keep: {', '.join(gone)}")
    chosen = [(n, w) for w, functions in parts.items() for n in functions if keep.get(n) == w]
    source = {**basis.source, "added": [*basis.source.get("added", []), *added]}
    return Basis([n for n, _ in chosen], np.array([parts[w][n] for n, w in chosen]),
                 basis.bandwidth, basis.frequency, basis.centre, basis.te, basis.seq, source)


#*********#
#   one   #
#*********#
def _one(spectrum: Any) -> Any:
    """One spectrum, from a batch of one as the GUI hands it.

    Raises:
        ValueError: For several, or one with coils, averages or other higher dimensions.
    """
    from nifti_mrs_plus import NIfTI_MRS_Plus

    if isinstance(spectrum, NIfTI_MRS_Plus):
        if len(spectrum.list()) != 1:
            raise ValueError(f"a basis function is one spectrum, not {len(spectrum.list())}")
        spectrum = spectrum.list()[0]
    shape = np.asarray(spectrum[:]).shape
    if shape[:3] != (1, 1, 1) or len(shape) != 4:
        raise ValueError(f"a basis function is one spectrum, coil combined and averaged, not "
                         f"one of shape {shape}")
    return spectrum


#*************#
#   as file   #
#*************#
def as_file(basis: Any) -> Any:
    """A basis as a fit reads it: a Basis as its .BASIS, anything else as it is."""
    return basis.lcmodel() if isinstance(basis, Basis) else basis
