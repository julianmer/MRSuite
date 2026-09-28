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
import os
import re
import warnings
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from mrs_tools.constants import PPM_SHIFT
from nifti_mrs.nifti_mrs import NIFTI_MRS

from mrsuite import __version__
from mrsuite.bridges.fid_a import FIDABridge, FIDACodec
from mrsuite.core.method import method
from mrsuite.core.result import FitResult
from mrsuite.octave import build_lbfgsb, resolve_runtime, scratch

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

    #**********#
    #   init   #
    #**********#
    def __init__(self, centre_ppm: Optional[float] = None):
        """
        Args:
            centre_ppm (Optional[float]): ppm at the centre of the spectrum, for every
                conversion this codec makes. None takes each spectrum's own, which is
                what it should almost always be: fit_resampleBasis interpolates the
                basis set's ppm axis onto the data's, and each side states its own
                centre (Osprey's shipped basis sets say 3, an LCModel .BASIS says what
                its PPMSEP says), so the two line up as long as neither is guessed.
        """
        self.centre_ppm = centre_ppm

    #****************#
    #   nifti2fida   #
    #****************#
    def nifti2fida(self, data: NIFTI_MRS, centre_ppm: Optional[float] = None) -> dict:
        """
        Convert NIfTI-MRS into an Osprey-flavoured FID-A struct.

        Args:
            data (NIFTI_MRS): Input MRS data.
            centre_ppm (Optional[float]): Overrides the codec's own centre for this call.

        Returns:
            dict: The struct, in Osprey's dialect.
        """
        out = super().nifti2fida(data, centre_ppm=self.centre_ppm if centre_ppm is None
                                 else centre_ppm)
        centre = out["centerFreq"]                    # resolved, from the data if not given

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
    program = "Osprey"

    #**********#
    #   init   #
    #**********#
    def __init__(self, osprey_root: Optional[str] = None, verbose: bool = False,
                 runtime: Optional[str] = None, centre_ppm: Optional[float] = None):
        super().__init__(fida_root=str(osprey_root or OSPREY_ROOT), verbose=verbose,
                         runtime=runtime)
        self.centre_ppm = centre_ppm

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
        if self._octave is None and resolve_runtime(self._runtime) != "matlab":
            try:
                build_lbfgsb(self._root, runtime=self._runtime)
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
            self._codec = OspreyCodec(centre_ppm=self.centre_ppm)
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
        self.octave.eval(f"mrsuite_b = load('{path}'); "
                         f"mrsuite_basis = osp_recalculate_basis_specs(mrsuite_b.BASIS);")
        return self.octave.pull("mrsuite_basis")

    #************************#
    #   basis from lcmodel   #
    #************************#
    def basis_from_lcmodel(self, path: str, add_mm: bool = True,
                           centre_ppm: Optional[float] = None) -> Any:
        """
        Build a basis set from an LCModel ``.BASIS`` file.

        Osprey's own path in two steps, because neither half does the job alone:
        io_readlcmraw_basis returns one FID-A struct per metabolite, and fit_makeBasis
        assembles a container but only from a folder of ``.mat`` files. So the structs
        are written out and handed straight back.

        The file is read from a copy in the scratch folder, which the container sees
        wherever the original lives. io_readlcmraw_basis takes the ppm at the centre from
        PPMSEP in the $NMUSED block MakeBasis writes, and without it fit_makeBasis puts
        the centre at 3 ppm; a file that does not state it (PyLCModel's, for one) gets
        the block in the copy, at ``centre_ppm``. Its flipud(fftshift(conj(.))) also
        lands an even-length spectrum one bin low, which is shifted back. fit_makeBasis
        saves what it builds in Octave's working folder, so it runs in the scratch one.

        Args:
            path (str): An LCModel ``.BASIS`` file.
            add_mm (bool): Add Osprey's macromolecule and lipid basis functions.
            centre_ppm (Optional[float]): ppm at the centre, for a file without PPMSEP.
                Defaults to the codec's, else to what NIfTI-MRS assumes for 1H data
                that does not state it either (4.65).

        Returns:
            The basis set container, ready for ``fit_runFit``.
        """
        folder = scratch("mrsuite_basis_")
        with open(path) as fh:
            text = fh.read()
        if "PPMSEP" not in text:
            centre = next(c for c in (centre_ppm, self.codec.centre_ppm, PPM_SHIFT["1H"])
                          if c is not None)
            text = re.sub(r"^([ \t]*\$BASIS[ \t\r]*)$",
                          f" $NMUSED\n PPMSEP = {centre},\n $END\n\\1", text, flags=re.M)
        source = os.path.join(folder, "source.basis")
        with open(source, "w") as fh:
            fh.write(text)

        # MATLAB as well as Octave: plain names, as MATLAB's cannot start with an
        # underscore, and try/catch rather than Octave's unwind_protect
        self.octave.eval(f"""
            mrsuite_raw = io_readlcmraw_basis('{source}');
            mrsuite_mets = fieldnames(mrsuite_raw);
            for mrsuite_k = 1:numel(mrsuite_mets)
                b = mrsuite_raw.(mrsuite_mets{{mrsuite_k}});
                if mod(numel(b.specs), 2) == 0
                    b.specs = circshift(b.specs, 1);
                    b.fids = ifft(ifftshift(b.specs, 1), [], 1);
                end
                save(fullfile('{folder}', [mrsuite_mets{{mrsuite_k}} '.mat']), 'b', '-v7');
            end
            mrsuite_here = pwd;
            cd('{folder}');
            try
                mrsuite_basis = fit_makeBasis('{folder}', {int(add_mm)}, 'unedited');
            catch mrsuite_error
                cd(mrsuite_here);
                rethrow(mrsuite_error);
            end
            cd(mrsuite_here);
        """)
        return self.octave.pull("mrsuite_basis")


#***************#
#   to result   #
#***************#
def to_result(params: dict, res_basis: dict, crlb: Any = None,
              method: str = "osprey:Osprey", scale: float = 1.0) -> FitResult:
    """
    Map an Osprey fit onto MRSuite's uniform result.

    Osprey reports amplitudes on the basis set's own scale and leaves referencing to
    its quantify stage, so the creatine-referenced concentrations are formed here the
    way OspreyQuantify does: divided by Cr + PCr.

    Args:
        params (dict): ``fitParams`` from ``fit_runFit``.
        res_basis (dict): The resampled basis set it returned alongside, which carries
            the metabolite names in fit order.
        crlb: Cramer-Rao lower bounds from ``fit_Osprey_CRLB``, when they were computed.
        method (str): Recorded on the result.
        scale (float): What the data were divided by before the fit; the raw amplitudes
            are multiplied back, so they stay in the units of the data passed in.

    Returns:
        FitResult: Raw and creatine-referenced concentrations, CRLBs when available, and
        the linewidth and residual Osprey reports as QC.
    """
    names = [str(n) for n in np.asarray(res_basis["name"]).ravel()]
    ampl = np.asarray(params["ampl"], dtype=float).ravel() * scale
    raw = dict(zip(names, (float(a) for a in ampl)))

    concentrations = {"raw": raw}
    tcr = raw.get("Cr", 0.0) + raw.get("PCr", 0.0)
    if tcr:
        concentrations["internal"] = {k: v / tcr for k, v in raw.items()}

    uncertainties = {}
    if crlb is not None:
        values = np.asarray(crlb, dtype=float).ravel()
        if values.size == len(names):
            uncertainties["percentage"] = dict(zip(names, (float(c) for c in values)))

    qc = {}
    for key, name in (("refFWHM", "fwhm"), ("Res", "residual")):
        if key in params:
            qc[name] = float(np.asarray(params[key]).ravel()[0])

    return FitResult(method=method, metabolites=names, concentrations=concentrations,
                     uncertainties=uncertainties, qc=qc,
                     raw={"fitParams": params, "resBasisSet": res_basis, "scale": scale})


bridge = OspreyBridge()

# basis containers built from .BASIS files in this process, by (path, mtime)
_containers = {}


#*********#
#   fit   #
#*********#
@method("Osprey", version=lambda: bridge.version(), runtime=lambda: bridge.runtime(),
        name="fit_runFit", parallel=True)      # per call: a caller may swap in a MATLAB bridge
def fit(data: Any, basis: Any, fit_opts: Optional[dict] = None,
        crlb: bool = False) -> Any:
    """
    Fit with Osprey, as MRSuite's uniform result.

    Each spectrum is divided by Osprey's own scale first, max(real(data)) over
    max(real(basis)), as osp_fitInitialise sets it for osp_fitUnEdited: the soft priors
    on linewidths and shifts are weighed against the unnormalised residual, so without
    it the fit, not just its amplitudes, depends on the scale of the data.

    Args:
        data: NIfTI-MRS or NIfTI-MRS+. A batch is fitted spectrum by spectrum.
        basis: An LCModel ``.BASIS`` file, as every fitter takes, or a basis container
            from ``bridge.basis`` or ``bridge.basis_from_lcmodel``. A file is read without
            Osprey's own macromolecules and lipids: a basis handed to all fitters brings
            its own, and adding Osprey's would fit them twice.
        fit_opts (Optional[dict]): Overrides for ``bridge.fit_options()``.
        crlb (bool): Also run fit_Osprey_CRLB. Off by default, and left that way: the
            fit does not produce bounds, because Osprey computes them in its quantify
            stage against data the fit has already zero-filled and re-splined, so the
            routine does not stand alone. An Osprey result therefore carries
            concentrations and QC but no uncertainties - enough to compare values
            against another method, not enough to weight that comparison.
        workers (int): Processes to spread a batch over (see ``mrsuite.core.parallel``).

    Returns:
        A FitResult, or one per spectrum for a batch, each with the data's processing
        history and the fit itself as its provenance.
    """
    if isinstance(basis, (str, os.PathLike)):
        # built once per file and process: a batch is fitted spectrum by spectrum
        key = (os.path.abspath(basis), os.path.getmtime(basis))
        if key not in _containers:
            _containers[key] = bridge.basis_from_lcmodel(str(basis), add_mm=False)
        basis = _containers[key]
    opts = bridge.fit_options(**(fit_opts or {}))

    struct = bridge.codec.nifti2fida(data)
    scale = (float(np.max(np.real(np.asarray(struct["specs"]))))
             / float(np.max(np.real(np.asarray(basis["specs"])))))
    struct["fids"] = np.asarray(struct["fids"]) / scale
    struct["specs"] = np.asarray(struct["specs"]) / scale

    params, res_basis = bridge.fit_runFit(struct, basis, "Osprey", opts, nout=2)
    bounds = None
    if crlb:
        _, _, bounds = bridge.fit_Osprey_CRLB(
            struct, basis, opts["bLineKnotSpace"], opts["range"], params,
            float(np.asarray(params["refShift"]).ravel()[0]), nout=3)
    return to_result(params, res_basis, bounds, scale=scale)


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
