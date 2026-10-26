####################################################################################################
#                                              mrsi.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 26/10/26                                                                                #
#                                                                                                  #
# Purpose: Spectroscopic imaging in MRSuite's own terms: a spatial NIfTI-MRS taken apart into     #
#          single-voxel spectra, which every step and fitter then takes, and fits of those        #
#          voxels put back on the grid as metabolite maps.                                        #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from nifti_mrs.create_nmrs import gen_nifti_mrs
from nifti_mrs_plus import NIfTI_MRS_Plus

from mrsuite.core import provenance
from mrsuite.core.method import keep_header


WINDOW = (1.8, 4.2)                  # the metabolites' ppm, where a voxel's signal is judged


#************#
#   signal   #
#************#
def signal(item: Any, window: Tuple[float, float] = WINDOW) -> np.ndarray:
    """
    Each voxel's signal: the magnitude spectrum summed over ``window`` ppm, at the first
    element of every higher dimension; (x, y, z).

    Args:
        item: A spatial NIfTI-MRS.
        window (Tuple[float, float]): The ppm range judged, the metabolites' by default.
    """
    data = np.asarray(item[:])
    data = data.reshape(*data.shape[:4], -1)[..., 0]                   # (x, y, z, points)
    specs = np.fft.fftshift(np.fft.fft(data, axis=3), axes=3)
    hz = np.fft.fftshift(np.fft.fftfreq(data.shape[3], item.dwelltime))
    ppm = hz / item.spectrometer_frequency[0] + item.SpecFreqChemShift
    inside = (ppm >= min(window)) & (ppm <= max(window))
    return np.abs(specs[..., inside]).sum(axis=3)


#************#
#   voxels   #
#************#
def voxels(spectra: Any, threshold: float = 0.0) -> Tuple[NIfTI_MRS_Plus, np.ndarray]:
    """
    A spatial NIfTI-MRS as single-voxel spectra, one per voxel with signal, which every
    step and fitter takes; and the voxels' indices, to put results back on the grid.

    Args:
        spectra: NIfTI-MRS+ holding the slice or volume (x, y, z voxels of one spectrum
            each, higher dimensions kept), or the NIfTI-MRS itself.
        threshold (float): Voxels whose metabolite signal (``signal``) lies below this
            fraction of the strongest voxel's are left out; 0 leaves out only empty voxels.

    Returns:
        Tuple[NIfTI_MRS_Plus, np.ndarray]: One spectrum per voxel, the header carried over
        and the voxel noted in each one's history, placed where the voxel is; and the
        (n, 3) voxel indices on the grid, in the spectra's order.
    """
    item = spectra.list()[0] if isinstance(spectra, NIfTI_MRS_Plus) else spectra
    data = np.asarray(item[:])                                  # (x, y, z, points, ...)
    strength = signal(item)
    index = np.argwhere(strength > threshold * strength.max()) if strength.max() > 0 \
        else np.argwhere(np.ones(data.shape[:3], bool))
    affine, out = np.asarray(item.voxToWorldMat), []
    for x, y, z in index:
        placed = affine.copy()
        placed[:3, 3] = (affine @ [x, y, z, 1.0])[:3]
        one = gen_nifti_mrs(data[x, y, z][None, None, None], item.dwelltime,
                            item.spectrometer_frequency[0], nucleus=item.nucleus[0],
                            affine=placed, dim_tags=list(item.dim_tags))
        keep_header(item, one)
        provenance.append(one, provenance.entry(
            "MRSuite", provenance.mrsuite_version(), "mrsuite.mrsi.voxels",
            {"voxel": [int(x), int(y), int(z)], "of": list(data.shape[:3])}))
        out.append(one)
    return NIfTI_MRS_Plus(out), index


#**********#
#   maps   #
#**********#
def maps(results: Sequence[Any], index: np.ndarray, metabolites: Optional[List[str]] = None,
         scaling: Optional[str] = None, shape: Optional[Sequence[int]] = None,
         columns: int = 4) -> Any:
    """
    Metabolite maps from fits of the voxels ``voxels`` gave: each voxel's concentration on
    the grid, one panel per metabolite, as a matplotlib Figure.

    Args:
        results (Sequence[FitResult]): One fit per row of ``index``.
        index (np.ndarray): The (n, 3) voxel indices ``voxels`` gave.
        metabolites (Optional[List[str]]): The maps to draw; the eight largest by mean
            concentration when None.
        scaling (Optional[str]): A scaling the fits carry; else raw, else molarity, else
            the first they have.
        shape (Optional[Sequence[int]]): The grid; what the index reaches when None.
        columns (int): Panels per row.

    Returns:
        matplotlib.figure.Figure: The maps of the slice holding most voxels, voxels
        without a fit left blank, each colour range the 2nd to 98th percentile of its
        map so that a few outlying voxels (lipids at the skull) do not set the scale.
    """
    from matplotlib.figure import Figure

    index = np.asarray(index)
    shape = tuple(shape) if shape is not None else tuple(index.max(axis=0) + 1)
    first = results[0].concentrations
    scaling = scaling or next((s for s in ("raw", "molarity") if s in first), next(iter(first)))
    if metabolites is None:
        means = {}
        for r in results:
            for m, c in r.concentrations.get(scaling, {}).items():
                if c is not None and np.isfinite(c):
                    means.setdefault(m, []).append(float(c))
        metabolites = sorted(means, key=lambda m: -abs(np.mean(means[m])))[:8]
    grid = {m: np.full(shape, np.nan) for m in metabolites}
    for (x, y, z), r in zip(index, results):
        for m in metabolites:
            c = r.concentrations.get(scaling, {}).get(m)
            grid[m][x, y, z] = np.nan if c is None else float(c)
    z = int(np.bincount(index[:, 2]).argmax())                 # the slice with most voxels
    rows = -(-len(metabolites) // columns)
    figure = Figure(figsize=(2.6 * columns, 2.6 * rows))
    for k, m in enumerate(metabolites):
        ax = figure.add_subplot(rows, columns, k + 1)
        values = grid[m][:, :, z]
        low, high = (np.nanpercentile(values, [2, 98]) if np.isfinite(values).any()
                     else (None, None))
        shown = ax.imshow(values.T, origin="lower", cmap="viridis", interpolation="nearest",
                          vmin=low, vmax=high)
        ax.set_title(f"{m} ({scaling})", fontsize=9)
        ax.set_xticks([]), ax.set_yticks([])
        figure.colorbar(shown, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=7)
    figure.tight_layout()
    return figure
