####################################################################################################
#                                           wat_lip_l2.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 09/11/26                                                                                #
#                                                                                                  #
# Purpose: Water and lipid signal removal by L2 regularisation (Lin et al., MRM 2019), the         #
#          MRSHub toolbox Wat_Lip_Removal_L2, run as published in an Octave (or MATLAB) session    #
#          of its own: a basis of synthesised water or lipid resonances projected out of every     #
#          voxel's spectrum, and its water shift, every voxel's water put on the carrier from an   #
#          unsuppressed reference, as steps for single voxels and MRSI grids alike.                #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from nifti_mrs.create_nmrs import gen_nifti_mrs

from mrsuite.bridges.base import Deferred
from mrsuite.core.method import method
from mrsuite.octave.toolbox import Toolbox

PROGRAM = "Wat_Lip_Removal_L2"
toolbox = Toolbox("wat-lip-removal-l2", PROGRAM, drivers="wat_lip_l2")


#*************#
#   volume    #
#*************#
def _volume(item: Any) -> np.ndarray:
    """One spectrum, or a grid of them, as (x, y, z, points); the toolbox takes spectra alone."""
    data = np.asarray(item[:], complex)
    if data.ndim != 4:
        tags = [t or f"dim_{k + 5}" for k, t in enumerate(list(item.dim_tags)[:data.ndim - 4])]
        raise ValueError(f"{PROGRAM} takes spectra, not {tags} (shape {data.shape}): combine "
                         f"coils and transients first")
    return data


#***********#
#   nifti   #
#***********#
def _nifti(out: np.ndarray, like: Any) -> Any:
    """A driver's result as NIfTI-MRS on the volume ``like`` had."""
    return gen_nifti_mrs(np.asarray(out, complex).reshape(like.shape), like.dwelltime,
                         like.spectrometer_frequency[0], nucleus=like.nucleus[0],
                         affine=np.asarray(like.voxToWorldMat))


#*************#
#   removal   #
#*************#
def _removal(batch: Any, kind: str, band: Callable[[Any], Tuple[float, ...]],
             damping: Tuple[float, float], components: Optional[int], beta: float,
             echo_position: float, seed: int) -> List[Any]:
    """
    The toolbox's L2 removal with a synthesised basis of ``kind``, through the driver, for
    a batch: the spectra that share a grid, a bandwidth and a band go over together, as
    columns, so their basis is generated and its matrix inverted once, not per spectrum.

    Args:
        batch: NIfTI-MRS+.
        kind (str): 'water' or 'lipid'.
        band: The band in the toolbox's Hz for a spectrum, from its header.
    """
    items = batch.list()
    volumes = [_volume(item) for item in items]
    groups: Dict[tuple, List[int]] = {}
    for k, (item, volume) in enumerate(zip(items, volumes)):
        groups.setdefault((volume.shape, float(item.dwelltime), tuple(band(item))), []).append(k)
    outs: List[Any] = [None] * len(items)
    for (shape, dwell, hertz), members in groups.items():
        points = shape[3]
        stacked = np.concatenate([volumes[k].reshape(-1, 1, 1, points) for k in members])
        out = toolbox.octave.feval("mrsuite_l2_removal", stacked, 1.0 / dwell, kind,
                                   np.asarray(hertz, float), np.asarray(damping, float),
                                   int(components or points), float(beta), float(echo_position),
                                   int(seed), nout=1)
        out = np.asarray(out, complex).reshape(-1, points)
        voxels = int(np.prod(shape[:3]))
        for j, k in enumerate(members):
            outs[k] = _nifti(out[j * voxels:(j + 1) * voxels], items[k])
    return outs


#*******************#
#   water removal   #
#*******************#
@method(PROGRAM, version=toolbox.version, runtime=toolbox.runtime, name="watersup_sim (water)",
        batched=True)
def water_removal(data: Any, dispersion: float = 60.0,
                  damping: Tuple[float, float] = (10.0, 30.0), components: Optional[int] = None,
                  beta: float = 3.0, echo_position: float = 0.0, seed: int = 0) -> Any:
    """
    Residual water taken out of every spectrum by L2 regularisation against a synthesised
    water basis: random Lorentzians within ``dispersion`` Hz of the carrier, orthogonal to
    the metabolites (Lin et al. 2019). The defaults are the toolbox's own script's. A batch
    of alike spectra shares one basis.

    Args:
        data: Single spectra, or MRSI grids (x, y, z, points), their water at the carrier.
        dispersion (float): The water's frequency dispersion about the carrier, in Hz (the
            toolbox's wat_frq_range).
        damping (Tuple[float, float]): The basis resonances' damping, drawn between these,
            in 1/s.
        components (Optional[int]): Resonances in the basis; the spectrum's points by
            default, as the toolbox's script takes them.
        beta (float): The regularisation exponent: the toolbox's beta, 10^-beta.
        echo_position (float): Where the echo sits in the FID, 0 to 1; 0 for an FID from
            the echo top (the toolbox's script uses 0.5 for its full-echo data).
        seed (int): The random seed, so a call repeats.
    """
    return _removal(data, "water", lambda item: (dispersion,), damping, components, beta,
                    echo_position, seed)


#*******************#
#   lipid removal   #
#*******************#
@method(PROGRAM, version=toolbox.version, runtime=toolbox.runtime, name="watersup_sim (lipid)",
        batched=True)
def lipid_removal(data: Any, band: Tuple[float, float] = (0.9, 1.9),
                  damping: Tuple[float, float] = (10.0, 30.0), components: Optional[int] = None,
                  beta: float = 3.0, echo_position: float = 0.0, seed: int = 0) -> Any:
    """
    Lipid signal taken out of every spectrum by L2 regularisation against a synthesised
    lipid basis: random Lorentzians across the lipid band (Lin et al. 2019). The band is
    given in ppm and turned into the toolbox's Hz from the spectrum's carrier; the default
    is the script's 360 to 480 Hz at its 3 T. The toolbox weights the resonances by a
    Gaussian of 80 Hz width about the band's centre, so a band much wider than that, as at
    7 T, is covered faintly at its edges. A batch of alike spectra shares one basis.

    Args:
        data: Single spectra, or MRSI grids (x, y, z, points).
        band (Tuple[float, float]): The lipid band, in ppm.
        damping (Tuple[float, float]): The basis resonances' damping, drawn between these,
            in 1/s.
        components (Optional[int]): Resonances in the basis; the spectrum's points by
            default.
        beta (float): The regularisation exponent: the toolbox's beta, 10^-beta.
        echo_position (float): Where the echo sits in the FID, 0 to 1.
        seed (int): The random seed, so a call repeats.
    """
    def hertz(item: Any) -> Tuple[float, ...]:
        """The band in the toolbox's Hz, which run upfield from the spectrum's carrier."""
        frequency, centre = float(item.spectrometer_frequency[0]), float(item.SpecFreqChemShift)
        return tuple(sorted((centre - ppm) * frequency for ppm in band))
    return _removal(data, "lipid", hertz, damping, components, beta, echo_position, seed)


#*****************#
#   water shift   #
#*****************#
@method(PROGRAM, version=toolbox.version, runtime=toolbox.runtime, name="watershift")
def water_shift(data: Any, reference: Any) -> Any:
    """
    Every voxel shifted so that its water, found as the strongest point of the unsuppressed
    reference, sits at the carrier: the toolbox's B0 correction before its water removal.
    A voxel whose water lies farther than a tenth of the bandwidth is left as it is. The
    toolbox puts the water on the middle point, so the spectra need an even number of
    points.

    Args:
        data: Single spectra, or MRSI grids (x, y, z, points), of an even number of points.
        reference: The unsuppressed acquisition of the same voxels, one per spectrum.
    """
    volume, water = _volume(data), _volume(reference)
    if water.shape != volume.shape:
        raise ValueError(f"the reference's grid {water.shape} is not the data's {volume.shape}")
    if volume.shape[3] % 2:
        raise ValueError(f"{PROGRAM}'s water shift puts the water on the middle point: "
                         f"{volume.shape[3]} points have none; zero-fill or crop to an even number")
    return _nifti(toolbox.octave.feval("mrsuite_water_shift", volume, water, nout=1), data)


step = Deferred(sys.modules[__name__])
