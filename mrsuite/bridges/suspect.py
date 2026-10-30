####################################################################################################
#                                            suspect.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 26/10/26                                                                                #
#                                                                                                  #
# Purpose: Bridge onto suspect: its readers for the vendors' raw formats and its processing       #
#          functions as steps, the spectra converted to and from its MRSData (an array with the   #
#          dwell time, frequency and echo time on it). The functions that only estimate - coil    #
#          weights, water components, phases, shifts - are applied by the steps written here.     #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from nifti_mrs.create_nmrs import gen_nifti_mrs
from nifti_mrs.nifti_mrs import NIFTI_MRS
from nifti_mrs_plus import NIfTI_MRS_Plus

try:
    import suspect
    from suspect import io as _io
    from suspect import processing as _processing
except ModuleNotFoundError as err:
    if (err.name or "").partition(".")[0] != "suspect":   # installed, but broken
        raise
    raise ImportError("suspect is not installed: pip install suspect") from err

from mrsuite.bridges.base import Bridge, dispatch
from mrsuite.core import provenance
from mrsuite.core.method import method

# suspect's functions by name, where it keeps them. Left out: rats and load_pfile, which numpy
# 2 broke (np.bool, np.complex), and its TARQUIN and LCModel glue, which MRSuite has of its own.
FUNCTIONS: Dict[str, Callable] = {
    "svd_weighting": _processing.channel_combination.svd_weighting,
    "whiten": _processing.channel_combination.whiten,
    "combine_channels": _processing.channel_combination.combine_channels,
    "hsvd": _processing.water_suppression.hsvd,
    "construct_fid": _processing.water_suppression.construct_fid,
    "residual_water_alignment": _processing.frequency_correction.residual_water_alignment,
    "spectral_registration": _processing.frequency_correction.spectral_registration,
    "correct_frequency_and_phase": _processing.frequency_correction.correct_frequency_and_phase,
    "sliding_window": _processing.denoising.sliding_window,
    "sliding_gaussian": _processing.denoising.sliding_gaussian,
    "sift": _processing.denoising.sift,
    "svd": _processing.denoising.svd,
    "spline": _processing.denoising.spline,
    "wavelet": _processing.denoising.wavelet,
    "mag_real": _processing.phase.mag_real,
    "ernst": _processing.phase.ernst,
    "acme": _processing.phase.acme,
    "apodize": _processing.apodize,
    "gaussian_window": _processing.gaussian_window,
    "adjust_phase": suspect.adjust_phase,
    "adjust_frequency": suspect.adjust_frequency,
    "load_twix": _io.load_twix,
    "load_rda": _io.load_rda,
    "load_sdat": _io.load_sdat,
    "load_siemens_dicom": _io.load_siemens_dicom,
    "load_dicom": _io.load_dicom,
    "load_svs_bruker": _io.load_svs_bruker,
}
READERS = {"twix": "load_twix", "rda": "load_rda", "sdat": "load_sdat",
           "siemens_dicom": "load_siemens_dicom", "dicom": "load_dicom", "bruker": "load_svs_bruker"}
# spectra in, spectra out, as dispatched; the steps written below join them
STEPS = ("combine_channels", "whiten", "sliding_window", "sliding_gaussian", "sift", "svd",
         "spline", "wavelet", "adjust_phase", "adjust_frequency")
WRITTEN = ("alignment", "water_removal", "phasing", "apodization")


#**************************************************************************************************#
#                                       Class SuspectBridge                                        #
#**************************************************************************************************#
#                                                                                                  #
# Dispatch onto suspect, the spectra as its MRSData there and NIfTI-MRS back.                      #
#                                                                                                  #
#**************************************************************************************************#
class SuspectBridge(Bridge):
    """Dispatch onto suspect, the spectra as its MRSData there and NIfTI-MRS back.

    suspect keeps time last and the coils second to last; a spectrum goes over so, its
    other dimensions before them, and comes back on NIfTI-MRS's axes, a dimension a call
    combined away (the coils) gone. A reader's result, with no spectrum to take the header
    from, is read off the MRSData itself.
    """
    name = "suspect"
    program = "suspect"

    def __init__(self):
        super().__init__()
        self._like: Optional[NIFTI_MRS] = None       # the spectrum of the call under way

    #*************#
    #   resolve   #
    #*************#
    def _resolve(self, func: str) -> Callable:
        if func not in FUNCTIONS:
            raise AttributeError(f"suspect has no function '{func}' here. "
                                 f"Available: {', '.join(self.available())}")
        return FUNCTIONS[func]

    #****************#
    #   to toolbox   #
    #****************#
    def _to_toolbox(self, obj):
        self._like = obj
        return _suspect(obj)

    #******************#
    #   from toolbox   #
    #******************#
    def _from_toolbox(self, obj):
        if isinstance(obj, np.ndarray) and np.iscomplexobj(obj) and obj.ndim:
            return _nifti(obj, self._like) if self._like is not None else _read(obj)
        return obj

    #***************#
    #   available   #
    #***************#
    def available(self) -> List[str]:
        return [*FUNCTIONS, *WRITTEN]

    #***********#
    #   steps   #
    #***********#
    def steps(self) -> List[str]:
        return [*STEPS, *WRITTEN]

    #*************#
    #   version   #
    #*************#
    def version(self) -> str:
        return suspect.__version__


bridge = SuspectBridge()


#*************#
#   suspect   #
#*************#
def _suspect(item: NIFTI_MRS) -> Any:
    """One spectrum as suspect's MRSData: time last, the coils before it, the rest in front;
    the dwell time, frequency, echo and repetition time (ms) and ppm reference on it; and
    conjugated, since suspect counts frequency the other way round (ppm = ppm0 - Hz / f0,
    where NIfTI-MRS has ppm = Hz / f0 + centre)."""
    data = np.conj(np.asarray(item[:])[0, 0, 0])                        # (points, ...)
    tags = list(item.dim_tags)[:data.ndim - 1]
    order = [k for k, t in enumerate(tags) if t != "DIM_COIL"] + \
            [k for k, t in enumerate(tags) if t == "DIM_COIL"]
    data = np.transpose(data, [k + 1 for k in order] + [0])
    header = item.hdr_ext.to_dict()
    return suspect.MRSData(np.ascontiguousarray(data), item.dwelltime,
                           item.spectrometer_frequency[0],
                           te=1000 * header.get("EchoTime", 0.0),
                           tr=1000 * header.get("RepetitionTime", -0.001),
                           ppm0=item.SpecFreqChemShift, transform=np.asarray(item.voxToWorldMat))


#***********#
#   nifti   #
#***********#
def _nifti(out: Any, like: NIFTI_MRS) -> NIFTI_MRS:
    """A result back as NIfTI-MRS on the axes ``like`` had, and NIfTI-MRS's way round: a
    dimension combined away is the coils, which went last."""
    data = np.conj(np.asarray(out))
    tags = list(like.dim_tags)[:np.asarray(like[:]).ndim - 4]
    ordered = [t for t in tags if t != "DIM_COIL"] + [t for t in tags if t == "DIM_COIL"]
    kept = ordered[:data.ndim - 1]
    data = np.moveaxis(data, -1, 0)[None, None, None]                   # (1, 1, 1, points, ...)
    return gen_nifti_mrs(data, like.dwelltime, like.spectrometer_frequency[0],
                         nucleus=like.nucleus[0], affine=np.asarray(like.voxToWorldMat),
                         dim_tags=kept + [None] * (3 - len(kept)))


#**********#
#   read   #
#**********#
def _read(data: Any) -> NIFTI_MRS:
    """A reader's MRSData as NIfTI-MRS: time first, then the coils, then the transients, as
    suspect lays its readers' arrays out (transients, coils, points), NIfTI-MRS's way round;
    the echo and repetition time and the voxel's placement from the MRSData."""
    array = np.conj(np.asarray(data))
    tags = ["DIM_COIL", "DIM_DYN", "DIM_USER_0"][:array.ndim - 1]
    placed = np.moveaxis(array, -1, 0)
    placed = np.transpose(placed, [0] + list(range(placed.ndim - 1, 0, -1)))[None, None, None]
    transform = getattr(data, "transform", None)
    out = gen_nifti_mrs(np.ascontiguousarray(placed), data.dt, data.f0,
                        affine=np.asarray(transform) if transform is not None else None,
                        dim_tags=tags + [None] * (3 - len(tags)))
    out.add_hdr_field("EchoTime", float(data.te) / 1000)
    if data.tr is not None and data.tr > 0:
        out.add_hdr_field("RepetitionTime", float(data.tr) / 1000)
    return out


#**********#
#   read   #
#**********#
def read(files: Union[str, Path, Sequence[Union[str, Path]]], format: str = "twix") -> NIfTI_MRS_Plus:
    """
    Raw scanner data read by suspect, as NIfTI-MRS+ with the reading in its history.

    Args:
        files: The file (a Twix .dat, an .rda, a DICOM file, a Philips .SDAT with its .SPAR
            beside it, a Bruker fid) or files, one spectrum each.
        format (str): A key of READERS: 'twix', 'rda', 'sdat', 'siemens_dicom', 'dicom'
            or 'bruker'.

    Returns:
        NIfTI_MRS_Plus: One spectrum per file, as acquired (coils, transients).
    """
    items = []
    for file in [files] if isinstance(files, (str, Path)) else files:
        item = bridge._from_toolbox(FUNCTIONS[READERS[format]](str(file)))
        provenance.append(item, provenance.entry("suspect", bridge.version(), READERS[format],
                                                 {"file": str(file)}))
        items.append(item)
    return NIfTI_MRS_Plus(items)


#***************#
#   alignment   #
#***************#
@method("suspect", version=lambda: bridge.version(), name="correct_frequency_and_phase")
def alignment(data: Any, method: str = "sr",
              frequency_range: Optional[Tuple[float, float]] = None) -> Any:
    """
    Every transient aligned in frequency and phase to their mean, by spectral registration
    ('sr', Near et al. 2014) or the residual water ('rwa').

    Args:
        data: The spectra, with their transients (DIM_DYN).
        method (str): 'sr' or 'rwa'.
        frequency_range (Optional[Tuple[float, float]]): The Hz range compared, for 'sr'.
    """
    mrs = _suspect(data)
    if mrs.ndim == 1:
        return data                                                     # nothing to align
    target = mrs.inherit(np.mean(mrs, axis=tuple(range(mrs.ndim - 1))))
    options = {"frequency_range": frequency_range} if frequency_range else {}
    return _nifti(_processing.frequency_correction.correct_frequency_and_phase(
        mrs, target, method=method, **options), data)


#*******************#
#   water removal   #
#*******************#
@method("suspect", version=lambda: bridge.version(), name="hsvd")
def water_removal(data: Any, rank: int = 20, window: Tuple[float, float] = (4.4, 5.0)) -> Any:
    """
    The residual water taken out: the FID modelled as ``rank`` decaying sinusoids (HSVD),
    and those within ``window`` ppm subtracted.

    Args:
        data: The spectra.
        rank (int): Components of the model.
        window (Tuple[float, float]): The ppm range whose components are the water.
    """
    mrs = _suspect(data)
    out = mrs.copy()
    for index in np.ndindex(mrs.shape[:-1]):
        fid = mrs[index]
        low, high = sorted((fid.ppm_to_hertz(window[0]), fid.ppm_to_hertz(window[1])))
        components = _processing.water_suppression.hsvd(fid, rank)
        water = [c for c in components if low <= c["frequency"] <= high]
        out[index] = fid - _processing.water_suppression.construct_fid(water, fid.time_axis())
    return _nifti(out, data)


#*************#
#   phasing   #
#*************#
@method("suspect", version=lambda: bridge.version(), name="acme")
def phasing(data: Any, method: str = "acme") -> Any:
    """
    Zero- and first-order phase estimated and applied: by entropy minimisation ('acme'),
    Ernst's criterion ('ernst') or the magnitude-real match ('mag_real').

    Args:
        data: The spectra.
        method (str): 'acme', 'ernst' or 'mag_real'.
    """
    estimate = {"acme": _processing.phase.acme, "ernst": _processing.phase.ernst,
                "mag_real": _processing.phase.mag_real}[method]
    mrs = _suspect(data)
    out = mrs.copy()
    for index in np.ndindex(mrs.shape[:-1]):
        phi0, phi1 = estimate(mrs[index])
        out[index] = mrs[index].adjust_phase(phi0, phi1)
    return _nifti(out, data)


#*****************#
#   apodization   #
#*****************#
@method("suspect", version=lambda: bridge.version(), name="apodize")
def apodization(data: Any, line_broadening: float = 5.0) -> Any:
    """
    Gaussian line broadening of ``line_broadening`` Hz, suspect's window on the FID.

    Args:
        data: The spectra.
        line_broadening (float): In Hz.
    """
    return _nifti(_processing.apodize(_suspect(data), _processing.gaussian_window,
                                      {"line_broadening": line_broadening}), data)


#*************#
#   getattr   #
#*************#
def __getattr__(name: str) -> Any:
    """Module-level dispatch, so `from mrsuite.bridges.suspect import combine_channels` works."""
    return dispatch(bridge, name)


#*********#
#   dir   #
#*********#
def __dir__() -> List[str]:
    return sorted(set(globals()) | set(bridge.available()))
