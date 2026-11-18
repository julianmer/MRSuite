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


# what basis sets call the metabolites Vespa's table has under another name
NAMES = {"mI": "myo-inositol", "sI": "scyllo-inositol", "PE": "peamine", "Glc": "glucose-alpha",
         "tCr": "cr"}
# the singlets Vespa's start values hang on, by its abbreviation, and their ppm
REFERENCES = (("naa", 2.01), ("cr", 3.03), ("cho", 3.2))
# Vespa's authors' own fit settings for in vivo short-TE sLASER with a macromolecule basis,
# where they differ from the fit block's defaults (vespa/interfaces/inline/philips/presets/
# preset_philips_slaser30_cmrr_spar_metab_withMmol.xml): its manual leaves the fit to be
# tuned per protocol and saved as such a preset, which its batch scripts then apply. All
# but its Gaussian lineshape: the block's Voigt takes Gaussian and Lorentzian lines alike
# (on the 2016 challenge's Lorentzian ones, 0.114 against 0.155 mean error of the ratios)
SETTINGS = {
    "initial_small_peak_areas": "peak_3ppm",
    "initial_linewidth_method": "deconvolution", "initial_linewidth_range_end": 3.4,
    "initial_linewidth_fudge": 0.5, "initial_ko_points": 38,
    "baseline_method": "bspline_fixed_knot", "baseline_underestimate": 95,
    "baseline_underestimate_last": 75, "baseline_underestimate_steps": 2,
    "baseline_spline_spacing": 180, "baseline_spline_order": 2, "baseline_wavelet_scale": 64,
    "optimize_method": "lmfit_default", "optimize_scaling_flag": True,
    "optimize_global_iterations": 3, "optimize_weights_width_factor": 50,
    "optimize_weights_lipid_flag": True, "optimize_weights_lipid_end": 0.5,
    "optimize_weights_small_peak_factor": 0.6, "optimize_bounds_area_max_small": 250,
    "optimize_bounds_area_min_small": 40, "optimize_enable_bounds_area_small": True,
    "optimize_bounds_range_ppm": 7, "optimize_bounds_range_ppm_small": 7,
    "optimize_enable_bounds_ppm_small": True, "optimize_bounds_range_phase0": 5,
    "optimize_bounds_range_phase1": 100}
# the preset's pairs whose shifts move together, by the abbreviations of Vespa's metabolite
# table: each only where the prior has both, since Vespa looks them up and fails without
PAIRS = {"naa_naag": ("naa", "naag"), "cr_pcr": ("cr", "pcr"), "gpc_pcho": ("gpc", "pcho"),
         "cr2_pcr2": ("cr2", "pcr2"), "glu_gln": ("glu", "gln"), "tau_glc": ("tau", "glc")}


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
            or the spectrum's maximum) and ``search_ppm`` (default 0.1); ``area_scale``
            (default 1) multiplies the metabolite's Area Scale Factor as matched.
            Vespa starts NAA, Cr and Cho from a peak search and every other area from
            a reference peak's by the literature concentrations of its metabolite
            table, read by name, then searches each area only within a share of its
            start (its manual, 6.4.2 and 6.4.5); its user turns each metabolite's Area
            Scale Factor until the start lies on the data (A.6). So that is done here
            (``_matched``) before the fit. The names matter too: those of ``NAMES``
            enter the table as Vespa's own. And so does the scale, a proton having an
            area of 1 in Vespa's line lists: the prior is brought to it by its
            reference singlet (NAA's, else Cr's, else Cho's) and the areas are returned
            on the prior's own scale.
        opts (Optional[dict]): Any fit-block setting by its Vespa name, applied onto
            ``SETTINGS``, Vespa's authors' own for short-TE sLASER with macromolecules,
            and the shift constraints of ``PAIRS`` the prior has both of, e.g.
            ``{"baseline_method": "", "optimize_max_iterations": 500}``. One setting
            deviates from Vespa's, a trap for headless runs: the Cramer-Rao noise window,
            at -8..-6 ppm outside a typical sweep width, is placed 0.5-2 ppm in from
            whichever spectral edge lies farther from the prior's lines and the water
            (the reference ppm).
        workers (int): Processes to spread a batch over (see ``mrsuite.core.parallel``).

    Returns:
        A FitResult, or one per spectrum for a batch. Concentrations are the fitted
        areas ('raw', plus 'internal' over tCr when the prior has Cr/PCr or tCr);
        'percentage' holds the area Cramer-Rao bounds relative to the areas.

    Raises:
        ValueError: If a spectrum is not a single FID, an option is not a fit-block
            setting, a sampled basis function is not on the data's points, the noise
            window would hold signal, or the wavelet baseline (the fit block's default)
            is asked of a spectrum whose length is not a power of 2: Vespa's filter
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

    # Vespa reads its start values off its metabolite table by name, and takes a proton
    # for an area of 1: the names it lacks enter the table as the ones it has, and the
    # prior goes over at a proton's area in it, found on its reference singlet
    table = dataset.user_prior.metinfo
    columns = (table.abbreviations, table.spins, table.concentrations, table.t2decays)
    for name in names:
        if name.lower() not in table.full_names and name in NAMES:
            row = table.full_names.index(NAMES[name])
            table.full_names.append(name.lower())
            for column in columns:
                column.append(column[row])
    proton = _proton(prior, ppm, table)
    block.set.prior.metabolites = collections.OrderedDict()
    for name in names:
        if "fid" in prior[name]:
            continue
        lines = prior[name]
        met = PriorMetabolite()
        met.dims = [name, 1, 1, 1]
        met.ppms = list(lines["ppms"])
        met.areas = [area / proton for area in lines["areas"]]
        met.phases = list(lines["phases"])
        block.set.prior.metabolites[name] = met

    block.set.prior_list = names
    block.set.prior_peak_ppm = [_strongest(prior[m]) for m in names]
    block.set.prior_search_ppm = [prior[m].get("search_ppm", 0.1) for m in names]
    block.set.prior_area_scale = [prior[m].get("area_scale", 1.0) for m in names]
    block.set.prior_db_ppm = [False] * len(names)
    block.set.prior_fix_t2 = [1000.0] * len(names)
    block.set.prior_search_ph0 = [0.0] * len(names)

    # the noise window 0.5-2 ppm in from the edge farther from all signal: the prior's
    # lines and the water at the reference ppm
    lowest = min(min(prior[m]["ppms"]) for m in names)
    highest = max([resppm] + [max(prior[m]["ppms"]) for m in names])
    edges = resppm - sw / (2.0 * f0), resppm + sw / (2.0 * f0)
    upfield, downfield = (edges[0] + 0.5, edges[0] + 2.0), (edges[1] - 2.0, edges[1] - 0.5)
    window = upfield if lowest - upfield[1] >= downfield[0] - highest else downfield
    block.set.cramer_rao_ppm_start, block.set.cramer_rao_ppm_end = window
    labels = {a.replace("-", "_") for a in dataset.prior_list_unique}   # as Vespa labels them
    pairs = {f"optimize_constrain_ppm_{key}": set(pair) <= labels for key, pair in PAIRS.items()}
    for key, value in {**SETTINGS, **pairs, **(opts or {})}.items():
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
            item.fid = np.conj(prior[name]["fid"]) * chop / proton
            item.peak_ppm = prior[name]["peak_ppm"]
            item.all_ppms = list(prior[name]["ppms"])
            block.set.prior.basis_set[name] = item

    voxel = (0, 0, 0)
    dataset.blocks["spectral"].chain.run([voxel], entry="all")
    chain = dataset.blocks["fit"].chain
    chain.run([voxel], entry="initial_only")
    span = (lowest, max(max(prior[m]["ppms"]) for m in names))
    matched = _matched(dataset, names, block.set.prior_area_scale, span)
    block.set.prior_area_scale = [s * prior[m].get("area_scale", 1.0)
                                  for s, m in zip(matched, names)]
    dataset.blocks["spectral"].set_do_fit(True, voxel)
    dataset.blocks["spectral"].chain.run([voxel], entry="all")
    chain.run([voxel], entry="full_fit", do_init=True)
    return _result(names, block.fit_results[:, 0, 0, 0], block.cramer_rao[:, 0, 0, 0],
                   scale / proton)


#*************#
#   matched   #
#*************#
def _matched(dataset: Any, names: list, scales: list, span: tuple) -> list:
    """The Area Scale Factors that lay Vespa's start on the data, as its user turns them
    in the GUI until the start overlay matches (its manual, A.6): each metabolite as Vespa
    models it at its start (the area derivatives of its model: unit-area spectra at the
    start's lineshape, shifts and phases), scaled to the spectrum by least squares, the
    areas non-negative, beside a smooth baseline (cubic B-splines, a knot every 0.5 ppm),
    over the prior's lines. Under initial_small_peak_areas 'naa_ratio' (the fit block's
    default) Vespa starts every area but NAA's at NAA's times a literature ratio, so NAA's
    factor carries over to the others and is divided out of theirs; under 'peak_3ppm'
    (``SETTINGS``) each start stands alone. An area matched to nothing keeps a twentieth
    of its start: Vespa searches each area within a share of its start, and none of zero."""
    from scipy.interpolate import BSpline
    from scipy.optimize import lsq_linear
    from vespa.analysis.constants import FitInitialSmallPeakAreas

    chain = dataset.blocks["fit"].chain
    m = len(names)
    start = np.asarray(chain.initial_values[:m], float)
    _, pder = chain.fit_function(chain.initial_values, nobase=True)
    data = np.asarray(chain.data, complex)
    lo, hi = sorted(int(round(float(dataset.ppm2pts(p)))) for p in (span[0] - 0.2, span[1] + 0.2))
    rows = slice(max(lo, 0), min(hi, data.size - 1) + 1)
    shapes = np.asarray(pder[:m, :data.size])[:, rows].T
    x = np.arange(shapes.shape[0], dtype=float)
    knots = max(int(round((span[1] - span[0] + 0.4) / 0.5)), 1)
    edges = np.linspace(x[0], x[-1], knots + 1)
    t = np.concatenate([[x[0]] * 3, edges, [x[-1]] * 3])
    spline = BSpline.design_matrix(x, t, 3).toarray()
    columns = np.hstack([shapes, spline, 1j * spline])
    a = np.vstack([columns.real, columns.imag])
    b = np.concatenate([data[rows].real, data[rows].imag])
    lower = np.r_[np.zeros(m), np.full(2 * spline.shape[1], -np.inf)]
    areas = lsq_linear(a, b, bounds=(lower, np.inf)).x[:m]

    gain = np.where(start > 0, np.maximum(areas, start / 20.0) / np.where(start > 0, start, 1), 1.0)
    table = dataset.user_prior.metinfo
    naa = [i for i, n in enumerate(names)
           if table.get_abbreviation(n.lower()) in ("naa", "naa+naag")]
    ratio = dataset.blocks["fit"].set.initial_small_peak_areas == FitInitialSmallPeakAreas.NAA_RATIO
    if ratio and naa:
        carried = gain[naa[0]]
        gain = np.array([g if i in naa else g / carried for i, g in enumerate(gain)])
    return [float(s * g) for s, g in zip(scales, gain)]


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


#************#
#   proton   #
#************#
def _proton(prior: Dict[str, Any], ppm: np.ndarray, table: Any) -> float:
    """The area one proton has in the prior: its reference singlet's (NAA's, else Cr's,
    else Cho's, the peaks Vespa starts from) over the protons Vespa's table gives it; 1
    for a prior without one. A sampled function's singlet is twice its real spectrum's
    integral, 0.1 ppm to either side, the first point halved as for any FID."""
    for abbreviation, at in REFERENCES:
        for name, lines in prior.items():
            if table.get_abbreviation(name.lower()) != abbreviation:
                continue
            if "fid" in lines:
                fid = lines["fid"].copy()
                fid[0] *= 0.5
                area = 2.0 * np.fft.fft(fid).real[np.abs(ppm - at) < 0.1].sum() / fid.size
            else:
                area = sum(a for a, p in zip(lines["areas"], lines["ppms"]) if abs(p - at) < 0.1)
            return area / table.spins[table.abbreviations.index(abbreviation)]
    return 1.0


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
