####################################################################################################
#                                            osprey.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-08-08                                                                              #
#                                                                                                  #
# Purpose: Bridge onto Osprey's Octave function surface. Osprey bundles its own copy of            #
#          FID-A and has let it drift from upstream, so the machinery is FID-A's but the           #
#          struct is not: the codec here is the delta between the two dialects.                    #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import warnings
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from nifti_mrs.nifti_mrs import NIFTI_MRS

from mrsuite import __version__
from mrsuite.bridges.fid_a import FIDABridge, FIDACodec
from mrsuite.octave import build_lbfgsb

OSPREY_ROOT = Path(__file__).resolve().parents[2] / "externals" / "osprey"


#**************************************************************************************************#
#                                        Class OspreyCodec                                         #
#**************************************************************************************************#
#                                                                                                  #
# The FID-A codec in Osprey's dialect of the same struct.                                          #
#                                                                                                  #
#**************************************************************************************************#
class OspreyCodec(FIDACodec):
    """The FID-A codec in Osprey's dialect of the same struct.

    Osprey's bundled FID-A has diverged from upstream in how it derives a spectrum from
    a FID, and the two are mirror images:

    ==============  ==========================  ==============================
    ..              upstream FID-A              Osprey's copy
    ==============  ==========================  ==============================
    ``fids``        conjugated on load          not conjugated
    ``specs``       ``fftshift(ifft(fids))``    ``fftshift(fft(fids))``
    ``ppm``         ``-f/(Bo*42.577) + 4.65``   ``f/(Bo*42.577) + centerFreq``
    ``te``/``tr``   seconds                     milliseconds
    ==============  ==========================  ==============================

    Each is self-consistent - a conjugated FID with an ifft and a descending axis
    describes the same spectrum as a plain one with an fft and an ascending axis - but a
    struct built for one is not valid for the other, and the failure is a mirrored
    spectrum rather than an error. Checked against the data rather than against either
    loader: only this combination puts the largest peak of the challenge spectrum at
    2.04 ppm, where NAA is.

    (Both loaders read ``conj(nii.img)``, so the divergence is in the NIfTI readers
    they use, not in the line that looks responsible.)
    """

    # what Osprey's io_loadspec_niimrs uses, against FSL-MRS's 4.65
    CENTRE_PPM = 4.68

    #****************#
    #   nifti2fida   #
    #****************#
    def nifti2fida(self, data: NIFTI_MRS, centre_ppm: Optional[float] = None) -> dict:
        """
        Convert NIfTI-MRS into an Osprey-flavoured FID-A struct.

        Args:
            data (NIFTI_MRS): Input MRS data.
            centre_ppm (Optional[float]): ppm at the centre of the spectrum. Defaults to
                Osprey's own 4.68.

        Returns:
            dict: The struct, in Osprey's dialect.
        """
        centre = self.CENTRE_PPM if centre_ppm is None else centre_ppm
        out = super().nifti2fida(data, centre_ppm=centre)

        fids = np.conj(np.asarray(out["fids"]))       # undo the upstream conjugation
        out["fids"] = fids
        out["specs"] = np.fft.fftshift(np.fft.fft(fids, axis=0), axes=0)

        # ascending, and rebuilt rather than negated so it matches Osprey's own formula
        sw, n_pts = float(out["spectralwidth"]), fids.shape[0]
        bo = float(np.asarray(out["Bo"]).ravel()[0])
        f = np.linspace(-sw / 2 + sw / (2 * n_pts), sw / 2 - sw / (2 * n_pts), n_pts)
        out["ppm"] = f / (bo * 42.577) + centre

        out["te"] = float(out["te"]) * 1e3
        out["tr"] = float(out["tr"]) * 1e3

        # fit_Osprey copies dataToFit.names onto the resampled basis set. Osprey's own
        # processing stage sets it; a tuple so oct2py sends a cell array, not a matrix.
        out.setdefault("names", ("A",))

        # Osprey's loader records which of its acquisition types this is, and its fit
        # branches on them. Unedited is the only one a plain single-voxel spectrum can be.
        out["flags"].update(isUnEdited=1, isMEGA=0, isHERMES=0, isHERCULES=0,
                            isMRSI=0, isPRIAM=0)

        # Every Osprey processing step records provenance through
        # op_add_analysis_provenance, which reads hdr_ext.ProcessingSoftwareVersion and
        # raises in both its try and its catch when the field is absent. It is a
        # standard NIfTI-MRS field and MRSuite is the software in question, so fill it.
        nii = out["nii_mrs"]
        hdr_ext = nii["hdr_ext"]
        hdr_ext = hdr_ext.to_dict() if hasattr(hdr_ext, "to_dict") else dict(hdr_ext)
        hdr_ext.setdefault("ProcessingSoftwareVersion", f"MRSuite {__version__}")
        nii["hdr_ext"] = hdr_ext
        return out

    #****************#
    #   fida2nifti   #
    #****************#
    def fida2nifti(self, fida: dict) -> NIFTI_MRS:
        """
        Convert an Osprey struct back to NIfTI-MRS.

        Args:
            fida (dict): A struct in Osprey's dialect.

        Returns:
            NIFTI_MRS: The spectra.
        """
        # the inherited conversion undoes an upstream-style conjugation that this
        # dialect never applied, so put it back before handing over
        fida = dict(fida)
        fida["fids"] = np.conj(np.asarray(fida["fids"]))
        return super().fida2nifti(fida)


#**************************************************************************************************#
#                                        Class OspreyBridge                                        #
#**************************************************************************************************#
#                                                                                                  #
# Dispatches onto Osprey, converting NIfTI-MRS at the boundary.                                    #
#                                                                                                  #
#**************************************************************************************************#
class OspreyBridge(FIDABridge):
    """Dispatch onto Osprey, converting NIfTI-MRS at the boundary.

    Derived from the FID-A bridge because Osprey ships FID-A inside itself and speaks
    its structs: the Octave session, the name resolution and the discovery are the same
    work against a different tree. What differs is the dialect, hence the codec.

    The two bridges keep separate Octave sessions, which matters here: putting both
    trees on one path would leave two versions of every ``op_*`` in scope, and which one
    answered would depend on the order they were added.

    Unlike FSL-MRS, fitting needs no adapter. Osprey takes the basis as an ordinary
    argument - ``fit_runFit(dataToFit, basisSet, fitModel, fitOpts)`` - so the data
    converts through the codec, the basis passes through as a struct, and the whole
    toolbox including the fit is reachable by dispatch.
    """

    name = "osprey"

    #**********#
    #   init   #
    #**********#
    def __init__(self, osprey_root: Optional[str] = None, verbose: bool = False,
                 runtime: Optional[str] = None):
        super().__init__(fida_root=str(osprey_root or OSPREY_ROOT), verbose=verbose,
                         runtime=runtime)

    #************#
    #   octave   #
    #************#
    @property
    def octave(self):
        """Octave session, with Osprey's bundled solver compiled first.

        The core fit reaches lbfgsb_wrapper, and Osprey ships only MATLAB MEX binaries
        of it. Best effort: the build needs mkoctfile, and everything outside the fit
        works without it, so a failure here is reported rather than fatal.
        """
        if self._octave is None:
            try:
                build_lbfgsb(self._root)
            except (FileNotFoundError, RuntimeError) as exc:
                warnings.warn(f"Osprey's L-BFGS-B-C solver was not built ({exc}); "
                              f"fit_runFit will fail, other functions are unaffected.",
                              stacklevel=2)
        return FIDABridge.octave.fget(self)

    #***********#
    #   codec   #
    #***********#
    @property
    def codec(self) -> OspreyCodec:
        """Converter for NIfTI-MRS <-> Osprey's FID-A structs."""
        if self._codec is None:
            self._codec = OspreyCodec()
        return self._codec

    #*****************#
    #   fit options   #
    #*****************#
    @staticmethod
    def fit_options(**overrides) -> dict:
        """
        Options for ``fit_runFit``, as Osprey's own initialisation would set them.

        ``fit_runFit`` fills in only range, rangeWater and bLineKnotSpace when no options
        are passed. The rest normally arrive from a job file by way of osp_fitInitialise,
        and fit_Osprey reads them without checking, so calling fit_runFit directly fails
        on the first one missing. GAP is the empty default osp_fitUnEdited passes, which
        is ``MRSCont.opts.fit.GAP.A``, not the whole GAP struct.

        Args:
            **overrides: Fields to change, e.g. ``range=[0.5, 4.0]``.

        Returns:
            dict: Options ready to hand to ``fit_runFit``.
        """
        opts = {"range": [0.2, 4.2], "rangeWater": [2.0, 7.4],
                "bLineKnotSpace": 0.4, "GAP": np.array([])}
        opts.update(overrides)
        return opts

    #***********#
    #   basis   #
    #***********#
    def basis(self, path: str) -> Any:
        """
        Load one of Osprey's basis sets, ready for ``fit_runFit``.

        The shipped ``.mat`` files hold only FIDs; Osprey adds the spectra and the ppm
        axis on load, which ``osp_fitInitialise`` does through this same call. Skipping
        it leaves fit_resampleBasis reading a field that is not there.

        Args:
            path (str): Path to a basis ``.mat``, absolute or relative to the repo.

        Returns:
            The basis set container, as an Octave struct.
        """
        self.octave.eval(f"__b = load('{path}'); "
                         f"__basis = osp_recalculate_basis_specs(__b.BASIS);")
        return self.octave.pull("__basis")


bridge = OspreyBridge()


#*************#
#   getattr   #
#*************#
def __getattr__(name: str) -> Any:
    """Module-level dispatch, so `from mrsuite.bridges.osprey import fit_runFit` works."""
    return getattr(bridge, name)


#*********#
#   dir   #
#*********#
def __dir__() -> List[str]:
    return sorted(set(globals()) | set(bridge.available()))
