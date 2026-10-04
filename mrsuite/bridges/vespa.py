####################################################################################################
#                                             vespa.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-29                                                                              #
#                                                                                                  #
# Purpose: Fitting with Vespa-Analysis (Soher et al., MRM 2023). Vespa's Voigt method is           #
#          a parametric linear combination over metabolite line lists (ppm/area/phase per          #
#          transition) rather than sampled basis FIDs, normally driven by GUI-exported             #
#          presets; this runs the same chain programmatically: dataset from the array, a           #
#          Voigt fit block from a prior of line lists, chain run, and the amplitudes with          #
#          their Cramer-Rao bounds back as MRSuite's uniform result.                               #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import collections
import sys
from typing import Any, Dict, Optional

import numpy as np

from mrsuite.bridges.base import Deferred
from mrsuite.core.method import method
from mrsuite.core.result import FitResult


#*************#
#   version   #
#*************#
def _version() -> Optional[str]:
    """Vespa's exact version, None when it is not installed."""
    try:
        import vespa.common.util.misc as util_misc
        return util_misc.get_vespa_version()
    except ImportError:
        return None


#*********#
#   fit   #
#*********#
@method("Vespa", version=_version, name="Voigt", parallel=True)
def fit(data: Any, prior: Dict[str, Any], opts: Optional[Dict[str, Any]] = None) -> FitResult:
    """
    Fit with Vespa's Voigt chain, as MRSuite's uniform result.

    Args:
        data: NIfTI-MRS or NIfTI-MRS+ of single spectra. A batch is fitted spectrum by
            spectrum.
        prior: Vespa's analogue of a basis: line lists by metabolite name,
            ``{"NAA": {"ppms": [2.008], "areas": [3.0], "phases": [0.0]}, ...}``, e.g.
            from a density-matrix simulation's transition table. A metabolite may
            instead be a sampled basis function, ``{"fid": array}`` on the data's time
            points and orientation, as Vespa's own macromolecule model takes an acquired
            one: line lists carry no linewidth of their own, so a basis function with
            lines closer than the FID resolves, or a broad one, has no exact line list.
            Its lines, for Vespa's local weighting and the noise window, are the peaks of
            its spectrum above a tenth of its maximum. Optional per-metabolite keys
            steer the initial peak search: ``peak_ppm`` (defaults to the strongest line,
            or the spectrum's maximum) and ``search_ppm`` (default 0.1).
        opts (Optional[dict]): Any fit-block setting by its Vespa name, applied onto the
            block's defaults, e.g. ``{"baseline_method": "", "optimize_max_iterations":
            500}``. Two defaults deviate from Vespa's, both traps for headless runs: the
            area bounds, relative to the initial estimate, are opened from 50-150% to
            effectively non-negative (Vespa's own source keeps that variant; its
            initial search misjudges J-coupled multiplets by more than 20x, and a
            clamped area is silently wrong, which the GUI user would see and a batch
            would not), and the Cramer-Rao noise window, at -8..-6 ppm outside a
            typical sweep width, is placed 0.5-2 ppm in from whichever spectral edge
            lies farther from the prior's lines and the water (the reference ppm).
        workers (int): Processes to spread a batch over (see ``mrsuite.core.parallel``).

    Returns:
        A FitResult, or one per spectrum for a batch. Concentrations are the fitted
        areas ('raw', plus 'internal' over tCr when the prior has Cr/PCr or tCr);
        'percentage' holds the area Cramer-Rao bounds relative to the areas.

    Raises:
        ValueError: If a spectrum is not a single FID, an option is not a fit-block
            setting, a sampled basis function is not on the data's points, the noise
            window would hold signal, or the wavelet baseline (Vespa's default) is
            asked of a spectrum whose length is not a power of 2: Vespa's filter
            returns the largest power of 2 below it and the fit then fails to
            broadcast, and its zero fill multiplies by powers of 2 only, so choose
            ``"bspline_fixed_knot"`` or ``""`` (none) for such data.
        ImportError: If Vespa is not installed.
    """
    if np.prod(data.shape) != data.shape[3]:
        raise ValueError(f"Vespa fits single spectra, not shape {data.shape}; combine "
                         f"coils and averages first")
    try:
        import vespa.analysis.mrs_dataset as mrs_dataset
        import vespa.common.mrs_data_raw as mrs_data_raw
        from vespa.analysis.constants import FitBaselineMethod
        from vespa.common.mrs_prior import BasisSetItem
        from vespa.common.mrs_prior_metabolite import PriorMetabolite
    except ImportError as err:
        raise ImportError("Vespa is not installed: pip install from "
                          "https://github.com/vespa-mrs/vespa (the PyPI wheel lags)") from err

    # Vespa wants the conjugate of the NIfTI-MRS orientation, like TARQUIN's dpt
    # (pinned by the physiology test); its tolerances and initial-value search assume
    # data of order one, so the FID is fitted at unit scale and the areas scaled back
    fid = np.conj(np.asarray(data[:], complex).reshape(1, 1, 1, -1))
    scale = float(np.abs(fid).max()) or 1.0
    fid = (fid / scale).astype(np.complex64)
    sw = 1.0 / data.dwelltime
    f0 = float(data.spectrometer_frequency[0])
    resppm = float(data.SpecFreqChemShift)

    raw = mrs_data_raw.DataRaw(attributes=dict(data_sources=["mrsuite"], data=fid,
                                               sw=sw, frequency=f0, resppm=resppm))
    dataset = mrs_dataset.dataset_from_raw(raw)
    block = dataset.add_voigt()

    ppm = resppm + np.fft.fftfreq(fid.size, data.dwelltime) / f0
    prior = {m: _sampled(prior[m], ppm) if "fid" in prior[m] else prior[m] for m in prior}
    names = list(prior)
    block.set.prior.metabolites = collections.OrderedDict()
    for name in names:
        if "fid" in prior[name]:
            continue
        lines = prior[name]
        met = PriorMetabolite()
        met.dims = [name, 1, 1, 1]
        met.ppms = list(lines["ppms"])
        met.areas = list(lines["areas"])
        met.phases = list(lines["phases"])
        block.set.prior.metabolites[name] = met

    block.set.prior_list = names
    block.set.prior_peak_ppm = [_strongest(prior[m]) for m in names]
    block.set.prior_search_ppm = [prior[m].get("search_ppm", 0.1) for m in names]
    block.set.prior_area_scale = [1.0] * len(names)
    block.set.prior_db_ppm = [False] * len(names)
    block.set.prior_fix_t2 = [1000.0] * len(names)
    block.set.prior_search_ph0 = [0.0] * len(names)
    block.set.optimize_bounds_area_min = 1e-6
    block.set.optimize_bounds_area_max = 1e5

    # the noise window 0.5-2 ppm in from the edge farther from all signal: the prior's
    # lines and the water at the reference ppm
    lowest = min(min(prior[m]["ppms"]) for m in names)
    highest = max([resppm] + [max(prior[m]["ppms"]) for m in names])
    edges = resppm - sw / (2.0 * f0), resppm + sw / (2.0 * f0)
    upfield, downfield = (edges[0] + 0.5, edges[0] + 2.0), (edges[1] - 2.0, edges[1] - 0.5)
    window = upfield if lowest - upfield[1] >= downfield[0] - highest else downfield
    block.set.cramer_rao_ppm_start, block.set.cramer_rao_ppm_end = window
    for key, value in (opts or {}).items():
        if not hasattr(block.set, key):
            raise ValueError(f"'{key}' is not a Vespa fit-block setting")
        setattr(block.set, key, value)
    start, end = block.set.cramer_rao_ppm_start, block.set.cramer_rao_ppm_end
    if block.set.cramer_rao_flag and start < highest and end > lowest:
        raise ValueError(f"Vespa's Cramer-Rao noise window, {start:.2f}-{end:.2f} ppm, "
                         f"overlaps the signal at {lowest:.2f}-{highest:.2f} ppm (the prior's "
                         f"lines and the water): at this sweep width neither tail is free; "
                         f"set cramer_rao_ppm_start and _end in opts to a stretch without "
                         f"signal, or cramer_rao_flag to False for no bounds")
    points = dataset.spectral_dims[0]
    if block.set.baseline_method == FitBaselineMethod.WAVELET_FILTER_BASIC and points & (points - 1):
        raise ValueError(f"Vespa's wavelet baseline needs a spectrum of a power of 2 points, "
                         f"not {points}; set baseline_method to 'bspline_fixed_knot' or '' "
                         f"in opts")
    block.set.prior.calculate_full_basis_set(None, None, dataset)
    # sampled functions go into the basis set as Vespa builds one from lines: in its
    # orientation, and at its frequencies, which run from the spectrum's edge (a line at
    # the reference ppm sits at +sw/2, so the function is chopped, as its macromolecule
    # dataset is)
    chop = np.where(np.arange(fid.size) % 2, -1.0, 1.0)
    for name in names:
        if "fid" in prior[name]:
            item = BasisSetItem()
            item.fid = np.conj(prior[name]["fid"]) * chop
            item.peak_ppm = prior[name]["peak_ppm"]
            item.all_ppms = list(prior[name]["ppms"])
            block.set.prior.basis_set[name] = item

    voxel = (0, 0, 0)
    dataset.blocks["spectral"].chain.run([voxel], entry="all")
    dataset.blocks["fit"].chain.run([voxel], entry="initial_only")
    dataset.blocks["spectral"].set_do_fit(True, voxel)
    dataset.blocks["spectral"].chain.run([voxel], entry="all")
    dataset.blocks["fit"].chain.run([voxel], entry="full_fit", do_init=True)
    return _result(names, block.fit_results[:, 0, 0, 0], block.cramer_rao[:, 0, 0, 0], scale)


#*************#
#   sampled   #
#*************#
def _sampled(entry: Dict[str, Any], ppm: np.ndarray) -> Dict[str, Any]:
    """A sampled basis function with the lines Vespa needs of it: the peaks of its
    spectrum above a tenth of its maximum (not its extent, which a truncated FID's
    ringing spreads over the whole sweep), and the peak search at the real spectrum's
    maximum (as Vespa places a line list's), unless given."""
    fid = np.asarray(entry["fid"], complex)
    if fid.size != ppm.size:
        raise ValueError(f"a sampled basis function needs the data's {ppm.size} points, "
                         f"not {fid.size}")
    spectrum = np.fft.fft(fid)
    size = np.abs(spectrum)
    peaks = (size >= np.roll(size, 1)) & (size >= np.roll(size, -1)) & (size >= 0.1 * size.max())
    lines = ppm[peaks]
    return {"ppms": list(lines), "peak_ppm": float(ppm[np.argmax(spectrum.real)]),
            **entry, "fid": fid}


#***************#
#   strongest   #
#***************#
def _strongest(lines: Dict[str, Any]) -> float:
    """The metabolite's peak-search ppm: given, or its strongest line's."""
    if "peak_ppm" in lines:
        return float(lines["peak_ppm"])
    areas = [abs(a) for a in lines["areas"]]
    return float(lines["ppms"][areas.index(max(areas))])


#************#
#   result   #
#************#
def _result(names, params, bounds, scale: float = 1.0) -> FitResult:
    """The chain's parameter vector [areas.., freqs.., ta, tb, ph0, ph1, ..] and its
    Cramer-Rao bounds, mapped onto MRSuite's uniform result. Vespa reports the area
    bounds already as percentages of the areas; ``scale`` undoes the unit-scale
    normalization the fit ran at. Vespa leaves the bounds at zero when it computed none
    (switched off, or a singular Fisher matrix), which is not certainty: the result then
    has no uncertainties."""
    areas = scale * np.abs(np.asarray(params[:len(names)], float))
    pct = np.asarray(bounds[:len(names)], float)
    if not np.all(np.isfinite(areas)):
        raise ValueError("Vespa's fit returned non-finite areas")

    conc = dict(zip(names, areas))
    uncertainties = {}
    if np.any(pct):
        pct = np.where(pct > 0, pct, np.nan)
        uncertainties = {"percentage": dict(zip(names, pct)),
                         "raw": dict(zip(names, areas * pct / 100.0))}

    concentrations = {"raw": conc}
    tcr = conc.get("tCr") or conc.get("Cr", 0.0) + conc.get("PCr", 0.0)
    if tcr:
        concentrations["internal"] = {m: v / tcr for m, v in conc.items()}
    return FitResult(method="vespa:Voigt", metabolites=names,
                     concentrations=concentrations,
                     uncertainties=uncertainties,
                     raw={"params": np.asarray(params), "cramer_rao": np.asarray(bounds)})


# Vespa is driven through one programmatic chain, so like TARQUIN there is no bridge,
# but its fit is a pipeline step all the same: ``vespa.step.fit(prior)``
step = Deferred(sys.modules[__name__])
