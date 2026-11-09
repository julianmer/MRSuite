####################################################################################################
#                                             pastis.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 09/11/26                                                                                #
#                                                                                                  #
# Purpose: PASTIS (Roussel et al., MRM 2022), run from its repository at the commit MRSuite       #
#          pins: its metabolite model, simulated by GAMMA for its sequences (sLASER with shaped    #
#          adiabatic pulses among them, 1H and 31P), as a Basis; its time-domain linear-           #
#          combination fit with linked parameters and Cramer-Rao bounds; and its processing of     #
#          transients (rejection, realignment, phasing, coil combination) as steps.                #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import functools
import importlib
import importlib.util
import logging
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from mrsuite import externals
from mrsuite.bridges.base import Deferred
from mrsuite.core.basis import Basis
from mrsuite.core.method import method
from mrsuite.core.result import FitResult
from mrsuite.simulators import pygamma_build

# PASTIS's sequence classes (its sim.py), by a short name
SEQUENCES = {"fid": "mrs_seq_fid", "press": "mrs_seq_press", "steam": "mrs_seq_steam",
             "slaser": "mrs_seq_eja_svs_slaser", "eja_svs_press": "mrs_seq_eja_svs_press",
             "eja_svs_steam": "mrs_seq_eja_svs_steam", "svs_se": "mrs_seq_svs_se",
             "svs_st": "mrs_seq_svs_st", "svs_st_vapor_643": "mrs_seq_svs_st_vapor_643",
             "dw_slaser": "mrs_seq_svs_DW_slaser_b"}
# its metabolite databases, one per nucleus, each with "Template_*" sheets naming basis sets
DATABASES = {"1H": "metabolites_1H.xls", "31P": "metabolites_31P.xls"}
# what its modules import, as pip names them (setup.py leaves some out)
PACKAGES = ("suspect", "pymapvbvd", "brukerapi", "pydicom", "nibabel", "xlrd", "termcolor",
            "parse", "ipython")
MODULES = ("suspect", "mapvbvd", "brukerapi", "pydicom", "nibabel", "xlrd", "termcolor", "parse",
           "IPython")
# its data rejection criteria (reco.data_rejection_method), by name
REJECTION = {"amplitude": "AUTO_AMPLITUDE", "linewidth": "AUTO_LINEWIDTH",
             "frequency": "AUTO_FREQUENCY", "phase": "AUTO_PHASE"}


#*************#
#   version   #
#*************#
def _version() -> Optional[str]:
    """The commit MRSuite pins, None when what PASTIS imports is not installed."""
    if not all(importlib.util.find_spec(name) for name in MODULES):
        return None
    return externals.version("pastis")


#*************#
#   modules   #
#*************#
def _modules() -> Tuple[Any, Any, Any]:
    """PASTIS's sim, reco and fit, imported from its repository on first use (fetched then),
    with PyGAMMA first on the path where it is built, since sim imports it as it loads."""
    root = externals.ensure("pastis")
    built = pygamma_build.folder()
    for path in (root, built) if (built / "pygamma").is_dir() else (root,):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    if not hasattr(np, "trapz"):
        np.trapz = np.trapezoid              # removed in NumPy 2.4; PASTIS integrates areas with it
    root = logging.getLogger()               # its log module configures the root logger as it loads
    level, handlers = root.level, list(root.handlers)
    try:
        import pastis.fit as fit
        import pastis.reco as reco
        import pastis.sim as sim
    except ImportError as err:
        raise ImportError(f"PASTIS cannot be imported ({err}): pip install "
                          f"{' '.join(PACKAGES)}") from err
    for handler in root.handlers:
        if handler not in handlers:
            root.removeHandler(handler)
    root.setLevel(level)
    return sim, reco, fit


#************#
#   select   #
#************#
def _select(meta: Any) -> Any:
    """PASTIS's aliases module, the metabolite indices its modules read, regenerated for a
    basis set: PASTIS writes the file for each set but never reloads it."""
    import pastis.aliases as aliases
    meta._write_header_file()
    for name in [n for n in vars(aliases) if n[:2] in ("m_", "n_", "ll")]:
        delattr(aliases, name)               # a reload keeps what the new file does not redefine
    importlib.reload(aliases)
    if not hasattr(aliases, "m_Water"):      # PASTIS's fit tool reads it even for a template
        aliases.m_Water = -1                 # without water (31P): an index no group has
    return aliases


#***************#
#   basis set   #
#***************#
def _basis_set(sim: Any, template: str, nucleus: str) -> Any:
    """One template of PASTIS's metabolite database for a nucleus, as its basis-set object."""
    import pandas as pd
    database = externals.path("pastis") / "pastis" / DATABASES[nucleus]
    templates = [s[len("Template_"):] for s in pd.ExcelFile(database, engine="xlrd").sheet_names
                 if s.startswith("Template_")]
    if template not in templates:
        raise ValueError(f"no template {template!r} in PASTIS's {nucleus} database; it holds "
                         f"{templates}")
    meta = sim.metabolite_basis_set(template, str(database))
    _select(meta)
    return meta


#**************#
#   simulate   #
#**************#
@functools.lru_cache(maxsize=8)
def _simulate(te: float, sequence: str, frequency: float, points: int, bandwidth: float,
              template: str, nucleus: str, shaped_pulses: bool,
              sequence_args: Tuple[Tuple[str, Any], ...]) -> Tuple[Any, Any]:
    """PASTIS's sequence initialised, its metabolite signals simulated by GAMMA, with the
    basis set it was simulated for; kept per arguments, as a fit asks for the same again."""
    if sequence not in SEQUENCES:
        raise ValueError(f"no sequence {sequence!r} in PASTIS; it holds {sorted(SEQUENCES)}")
    if nucleus not in DATABASES:
        raise ValueError(f"PASTIS has a metabolite database for {sorted(DATABASES)}, not "
                         f"{nucleus!r}")
    if shaped_pulses and sequence != "slaser":
        raise ValueError("shaped pulses are PASTIS's sLASER's (its HSn adiabatic refocusing "
                         "pulses as waveforms); its other sequences use hard pulses")
    sim, _, _ = _modules()
    meta = _basis_set(sim, template, nucleus)
    pygamma_build.ensure()
    if not sim.GAMMA_LIB_LOADED:              # sim loaded before PyGAMMA was built: hand it over
        import pygamma
        sim.pg, sim.GAMMA_LIB_LOADED = pygamma, True
    seq = getattr(sim, SEQUENCES[sequence])(te=float(te), f0=float(frequency), npts=int(points),
                                            fs=float(bandwidth), nuclei=nucleus,
                                            **dict(sequence_args))
    if shaped_pulses:
        seq.pulse_rfc_real_shape_enable = True
    try:
        seq.initialize(meta)
    except Exception as err:                  # noqa: BLE001 - PASTIS's log.error raises Exception
        raise RuntimeError(f"PASTIS's simulation failed: {err}") from err
    _select(meta)
    return seq, meta


#***********#
#   basis   #
#***********#
def basis(te: float, sequence: str = "press", frequency: float = 297.206258, points: int = 16384,
          bandwidth: float = 5000.0, template: str = "Brain_7T", nucleus: str = "1H",
          shaped_pulses: bool = False, **sequence_args: Any) -> Basis:
    """
    A basis set simulated by PASTIS on GAMMA: one of its sequences run on the spin systems
    of one template of its metabolite database, each group of the template one basis
    function, unbroadened. The defaults are PASTIS's own (sim.py: a 7 T Siemens scanner,
    16384 points at 5000 Hz, its Brain_7T template). PyGAMMA is built on first use
    (``pygamma_build``).

    Args:
        te (float): Echo time in ms.
        sequence (str): 'press', 'steam', 'slaser' (Siemens eja_svs_slaser), 'fid'
            (pulse-acquire), or another of ``SEQUENCES``.
        frequency (float): Spectrometer frequency in MHz.
        points (int): Points of the FID.
        bandwidth (float): Spectral width in Hz.
        template (str): A "Template_*" sheet of the database: for 1H 'Brain_7T', 'Muscle',
            'Lipids' or 'Ethanol'; for 31P 'Brain_7T'.
        nucleus (str): '1H' or '31P', each its own database.
        shaped_pulses (bool): For 'slaser': its HSn adiabatic refocusing pulses simulated
            as waveforms (PASTIS's pulse_rfc_real_shape_enable) rather than as hard
            pulses, which changes the J-evolution of the coupled metabolites; the pulse
            is set by ``rfc_pulse_duration`` (ms), ``rfc_pulse_r``, ``rfc_pulse_n`` and
            ``rfc_pulse_voltage`` among the sequence's arguments.
        **sequence_args: Any other argument of the sequence class, e.g. ``tr``,
            ``spoiler_duration``, ``exc_pulse_duration``.

    Returns:
        Basis: The template's groups in its order, Water and the macromolecules among
        them, about PASTIS's water at 4.7 ppm; ``te`` and ``seq`` as given.

    Raises:
        ValueError: If the sequence, nucleus or template is not PASTIS's, or shaped
            pulses are asked of another sequence than sLASER.
        ImportError: If what PASTIS imports is not installed.
        RuntimeError: If PyGAMMA cannot be built, or the simulation fails, with PASTIS's
            message (e.g. an echo time too short for the pulses).
    """
    seq, meta = _simulate(float(te), sequence, float(frequency), int(points), float(bandwidth),
                          template, nucleus, bool(shaped_pulses),
                          tuple(sorted(sequence_args.items())))
    import pygamma
    # the signals as PASTIS's model sums them, with the phase its sequence adds (a spin echo's
    # inversion, additional_phi0), in suspect's orientation (ppm = 4.7 - Hz / f0): conjugated
    # for NIfTI-MRS's
    fids = np.conj(np.asarray([np.asarray(s, complex) for s in seq._meta_signals])
                   * np.exp(1j * seq.additional_phi0))
    return Basis(list(meta.keys()), fids, float(bandwidth), float(frequency),
                 centre=float(seq.ppm_water), te=float(te), seq=sequence,
                 source={"simulated": {"program": "PASTIS", "version": externals.version("pastis"),
                                       "engine": {"program": "PyGAMMA",
                                                  "version": pygamma.__version__,
                                                  "commit": externals.REGISTRY["pygamma"][1]},
                                       "function": f"{SEQUENCES[sequence]}.initialize",
                                       "arguments": {"te": te, "sequence": sequence,
                                                     "frequency": frequency, "points": points,
                                                     "bandwidth": bandwidth,
                                                     "template": template, "nucleus": nucleus,
                                                     "shaped_pulses": shaped_pulses,
                                                     **sequence_args}}})


#*********#
#   fit   #
#*********#
@method("PASTIS", version=_version, name="fit_pastis")
def fit(data: Any, basis: Basis, metabolites: Optional[Sequence[str]] = None,
        ppmlim: Optional[Tuple[float, float]] = None,
        opts: Optional[Dict[str, Any]] = None) -> FitResult:
    """
    Fit with PASTIS's time-domain linear-combination model, as MRSuite's uniform result.

    PASTIS fits the metabolite signals it simulated, so the basis is one of
    ``bases.pastis``: it names the template, the sequence and its settings, and the model
    is simulated again on the spectrum's own grid (points, bandwidth, frequency; the
    basis's echo time) where the grid differs. Each group has an amplitude, a damping, a
    frequency shift and a phase. The fit is set up as PASTIS's own example runs it: every
    parameter of the fitted groups free within its bounds (amplitude 0-200, damping from 5
    Hz, shift within 10 Hz, phase within 0.1 rad) but the phase, which they all share with
    the template's reference metabolite (NAA in Brain_7T); the template's macromolecules
    broad (damping 150-300 Hz, amplitude to 1000); Water free with its own bounds when
    fitted; the groups not fitted fixed at zero.

    Args:
        data: NIfTI-MRS or NIfTI-MRS+ of single spectra. A batch is fitted spectrum by
            spectrum.
        basis (Basis): From ``mrsuite.simulators.bases.pastis``.
        metabolites (Optional[Sequence[str]]): The template's groups to fit, by name;
            every group but Water by default.
        ppmlim (Optional[Tuple[float, float]]): The range fitted, in ppm (PASTIS's
            optim_ppm_range); the whole spectrum by default.
        opts (Optional[dict]): Any other setting of PASTIS's fit tool by its name, e.g.
            ``{"optim_method": "lm", "fqn_noise_range": [-2, -1]}``.

    Returns:
        A FitResult, or one per spectrum for a batch. Concentrations: 'raw' the
        amplitudes, 'internal' their ratio to the template's reference metabolite, when fitted;
        uncertainties: 'raw' the Cramer-Rao bounds of the amplitudes, 'percentage' their
        share of the amplitude; qc: R² in the time and the frequency domain and the FQN
        (where its noise range lies in the spectrum); raw: every group's parameters and
        bounds, and SciPy's result.

    Raises:
        ValueError: If a spectrum is not a single FID, the basis is not PASTIS's, a
            metabolite is not in the template, an option is not a setting of the fit
            tool, or the fit returns non-finite amplitudes.
        ImportError, RuntimeError: As ``bases.pastis`` raises, or with PASTIS's message
            when the fit fails.
    """
    if np.prod(data.shape) != data.shape[3]:
        raise ValueError(f"PASTIS fits single spectra, not shape {data.shape}; combine coils "
                         f"and averages first")
    simulated = (basis.source or {}).get("simulated", {}) if isinstance(basis, Basis) else {}
    if simulated.get("program") != "PASTIS":
        raise ValueError("PASTIS fits the metabolite model it simulated: pass a basis from "
                         "mrsuite.simulators.bases.pastis")
    args = dict(simulated["arguments"])
    args.update(points=int(data.shape[3]), bandwidth=1.0 / data.dwelltime,
                frequency=float(data.spectrometer_frequency[0]))
    named = {k: args.pop(k) for k in ("te", "sequence", "frequency", "points", "bandwidth",
                                      "template", "nucleus", "shaped_pulses")}
    seq, meta = _simulate(float(named["te"]), named["sequence"], named["frequency"],
                          named["points"], named["bandwidth"], named["template"],
                          named["nucleus"], bool(named["shaped_pulses"]),
                          tuple(sorted(args.items())))
    seq = seq.copy()                          # the tool may re-initialise it for a ppm range
    _, reco, pfit = _modules()
    aliases = _select(meta)
    names = list(meta.keys())
    chosen = list(metabolites) if metabolites is not None else [n for n in names if n != "Water"]
    unknown = [n for n in chosen if n not in names]
    if unknown:
        raise ValueError(f"not in PASTIS's template {named['template']!r}: {unknown}; it holds "
                         f"{names}")
    index = [names.index(n) for n in chosen]
    # the phase is shared with the template's reference metabolite, else the first fitted;
    # PASTIS's peak-area integration, which precedes its fit, takes the reference's singlet
    reference = aliases.m_Ref_MB if aliases.m_Ref_MB in index else index[0]

    mrs = _view(reco, data)
    mrs = mrs.analyze_noise_nd()              # the noise level its Cramer-Rao bounds are scaled by
    tool = pfit.fit_pastis(mrs, seq, meta)
    tool.display_enable = tool.display_CRBs = False
    tool.metabolites = index
    tool.metabolites_area_integration = [aliases.m_Ref_MB]
    tool.area_integration_peak_ranges = [0.1]
    tool.optim_ppm_range = list(ppmlim) if ppmlim else None
    for key, value in (opts or {}).items():
        if not hasattr(tool, key):
            raise ValueError(f"'{key}' is not a setting of PASTIS's fit tool")
        setattr(tool, key, value)

    cm, dd, df, dp = aliases.p_cm, aliases.p_dd, aliases.p_df, aliases.p_dp
    low = tool.params_min.set_default_min().add_macromolecules_min()
    high = tool.params_max.set_default_max().add_macromolecules_max()
    low[:, cm], high[:, cm] = 0.0, 200.0
    high[aliases.m_All_MMs, cm] = 1000.0
    low[:, dd] = 5.0
    low[aliases.m_All_MMs, dd], high[aliases.m_All_MMs, dd] = 150.0, 300.0
    low[:, df], high[:, df] = -10.0, 10.0
    low[:, dp], high[:, dp] = -0.1, 0.1
    start = (low + high) / 2.0
    start[:, cm] = 0.0
    start[index, cm] = 0.1
    start[:, dd] = low[:, dd] * 1.1
    start[:, df] = start[:, dp] = 0.0
    link = tool.params_linklock
    link[:] = aliases.ll_FIXED
    link[index, :] = aliases.ll_FREE
    link[index, dp] = aliases.ll_SLAVE1
    link[reference, dp] = aliases.ll_MASTER1
    if "Water" in chosen:
        water = names.index("Water")
        link[water, :] = aliases.ll_FREE
        high[water, cm] = 1000.0
    tool.params_min, tool.params_max, tool.params_init = low, high, start
    try:
        tool.run()
    except Exception as err:                  # noqa: BLE001 - PASTIS's log.error raises Exception
        raise RuntimeError(f"PASTIS's fit failed: {err}") from err
    r2_time, r2_frequency = tool.get_Rsq(tool.params_fit)
    return _result(names, chosen, names[aliases.m_Ref_MB], np.asarray(tool.params_fit, float),
                   np.asarray(tool.params_fit.errors, float),
                   {"r2_time": float(r2_time), "r2_frequency": float(r2_frequency),
                    "fqn": float(tool.optim_results.fqn)}, tool.optim_results)


#************#
#   result   #
#************#
def _result(names: List[str], chosen: List[str], reference: str, params: np.ndarray,
            bounds: np.ndarray, qc: Dict[str, float], optimize: Any = None) -> FitResult:
    """
    PASTIS's fitted parameters as MRSuite's uniform result.

    Args:
        names (List[str]): The template's groups, in its order.
        chosen (List[str]): The groups fitted.
        reference (str): The group the 'internal' scaling divides by.
        params (np.ndarray): (groups, 4): amplitude, damping (Hz), shift (Hz), phase (rad).
        bounds (np.ndarray): Their absolute Cramer-Rao bounds, the same shape.
        qc (Dict[str, float]): Quality measures; one not finite is left out.
        optimize: SciPy's OptimizeResult, kept raw.

    Raises:
        ValueError: If a fitted amplitude is not finite.
    """
    amplitudes = {n: float(params[names.index(n), 0]) for n in chosen}
    if not all(np.isfinite(list(amplitudes.values()))):
        raise ValueError(f"PASTIS returned non-finite amplitudes: {amplitudes}")
    crlb = {n: float(bounds[names.index(n), 0]) for n in chosen}
    concentrations = {"raw": amplitudes}
    if amplitudes.get(reference):
        concentrations["internal"] = {n: a / amplitudes[reference] for n, a in amplitudes.items()}
    return FitResult(
        method="pastis:fit_pastis",
        metabolites=chosen,
        concentrations=concentrations,
        uncertainties={"raw": crlb,
                       "percentage": {n: 100.0 * crlb[n] / a for n, a in amplitudes.items() if a}},
        qc={k: v for k, v in qc.items() if np.isfinite(v)},
        raw={"names": names, "columns": ("amplitude", "damping", "shift", "phase"),
             "parameters": params, "bounds": bounds, "optimize": optimize})


#**********#
#   view   #
#**********#
def _view(reco: Any, item: Any) -> Any:
    """One spectrum as PASTIS's MRSData2, through suspect's MRSData (PASTIS keeps suspect's
    layout and orientation), with the label its messages concatenate (its readers set one;
    an array has none)."""
    from mrsuite.bridges.suspect import _suspect
    out = _suspect(item).view(reco.MRSData2)
    out._display_label = "MRSuite"
    return out


#**************#
#   transfer   #
#**************#
def _pastis(item: Any, dims: int) -> Tuple[Any, Tuple[int, ...]]:
    """
    One spectrum as PASTIS's MRSData2 with the axes its methods want: [averages,
    channels, points] (3) or [averages, points] (2), an axis of one added where the data
    has no such dimension.

    Returns:
        The MRSData2, and the axes added, to take away from the method's result.

    Raises:
        ValueError: If the spectrum has dimensions beyond coils and transients, or coils
            where a method takes transients alone.
    """
    _, reco, _ = _modules()
    tags = [t for t in list(item.dim_tags)[:np.asarray(item[:]).ndim - 4]]
    if any(t not in ("DIM_COIL", "DIM_DYN") for t in tags):
        raise ValueError(f"PASTIS processes coils and transients, not {tags}; split or drop the "
                         f"other dimensions first")
    mrs = _view(reco, item)
    added = []
    if "DIM_DYN" not in tags:
        mrs, added = mrs[None], [0]
    if dims == 3 and "DIM_COIL" not in tags:
        mrs, added = mrs[:, None], added + [1]
    if dims == 2 and "DIM_COIL" in tags:
        raise ValueError("PASTIS takes the transients of one spectrum, [averages, points]: "
                         "combine the coils first")
    return mrs, tuple(added)


#**********#
#   back   #
#**********#
def _back(out: Any, added: Tuple[int, ...], like: Any) -> Any:
    """A method's result as NIfTI-MRS on the axes ``like`` had, the axes added taken away."""
    from mrsuite.bridges.suspect import _nifti
    for axis in sorted(added, reverse=True):
        out = np.take(out, 0, axis=axis)
    return _nifti(np.asarray(out), like)


#*************#
#   combine   #
#*************#
@method("PASTIS", version=_version, name="correct_combine_channels_3d")
def combine(data: Any, reference: Any = None, phasing: bool = False) -> Any:
    """
    The coils combined with SVD weights (suspect's), from the water reference where given,
    else from the data themselves.

    Args:
        data: The spectra with their coils (DIM_COIL), and their transients if any.
        reference: An unsuppressed reference of the same coils, one per spectrum, to take
            the weights from.
        phasing (bool): Zero-order phasing of the coils while combining.
    """
    if "DIM_COIL" not in list(data.dim_tags):
        return data                                                     # nothing to combine
    mrs, added = _pastis(data, 3)
    if reference is not None:
        mrs.data_ref = _pastis(reference, 3)[0]
    out = mrs.correct_combine_channels_3d(use_ref_data=reference is not None, phasing=phasing)
    return _back(out, added, data)


#***********#
#   phase   #
#***********#
@method("PASTIS", version=_version, name="correct_phase_3d")
def phase(data: Any, reference: Any = None, peak_range: Tuple[float, float] = (4.5, 5.0),
          order: int = 0, offset: float = 0.0, per_channel: bool = False,
          first_point: bool = False) -> Any:
    """
    Each transient and coil phased, from the reference's phase evolution where given, else
    from a peak in the range, or the first point of the FID.

    Args:
        data: The spectra, with their coils and transients if any.
        reference: An unsuppressed reference, one per spectrum, to take the phase from.
        peak_range (Tuple[float, float]): Where the peak to phase on lies, in ppm.
        order (int): 0 or 1, the order of the correction.
        offset (float): A phase added to all, in rad.
        per_channel (bool): Estimate on the average of the transients per coil, for low
            SNR, rather than on each transient.
        first_point (bool): Take the phase of the first point of the FID instead of a
            peak, for weakly suppressed water.
    """
    mrs, added = _pastis(data, 3)
    if reference is not None:
        mrs.data_ref = _pastis(reference, 3)[0]
    out = mrs.correct_phase_3d(use_ref_data=reference is not None, peak_range=list(peak_range),
                               average_per_channel_mode=per_channel,
                               first_point_fid_mode=first_point, phase_order=order,
                               phase_offset=offset, display=False)
    return _back(out, added, data)


#**********#
#   align   #
#**********#
@method("PASTIS", version=_version, name="correct_realign_2d")
def align(data: Any, peak_range: Tuple[float, float] = (4.5, 5.0), moving_averages: int = 1,
          correlation: bool = False, max_shift: float = 25.0, apodization: float = 5.0) -> Any:
    """
    The transients realigned in frequency to the first, by the peak in the range or by
    cross-correlation.

    Args:
        data: The spectra with their transients (DIM_DYN), coils combined.
        peak_range (Tuple[float, float]): Where the peak to align on lies, in ppm.
        moving_averages (int): Transients averaged, an odd number, for the estimate.
        correlation (bool): Align by cross-correlation rather than the peak.
        max_shift (float): The largest shift allowed, in Hz.
        apodization (float): Line broadening for the estimate only, in Hz; 0 for none.
    """
    if "DIM_DYN" not in list(data.dim_tags):
        return data                                                     # nothing to align
    mrs, added = _pastis(data, 2)
    out = mrs.correct_realign_2d(peak_range=list(peak_range), moving_averages=moving_averages,
                                 inter_corr_mode=correlation, freq_shift_max=max_shift,
                                 allowed_apodization=apodization, display=False)
    return _back(out, added, data)


#************#
#   reject   #
#************#
@method("PASTIS", version=_version, name="correct_analyze_and_reject_2d")
def reject(data: Any, peak_range: Tuple[float, float] = (4.5, 5.0),
           snr_range: Tuple[float, float] = (1.8, 2.2),
           linewidth_range: Tuple[float, float] = (4.5, 5.0), moving_averages: int = 1,
           criteria: Optional[Sequence[str]] = None,
           snr_change: float = 1.0, apodization: float = 0.0, **options: Any) -> Any:
    """
    Transients rejected whose peak strays in amplitude, linewidth, frequency or phase, the
    bounds adjusted by PASTIS to the SNR of the average (its spinal-cord motion
    correction, Roussel et al. 2022).

    Args:
        data: The spectra with their transients (DIM_DYN), coils combined.
        peak_range (Tuple[float, float]): Where the peak analysed lies, in ppm.
        snr_range (Tuple[float, float]): The peak the SNR is measured on, in ppm.
        linewidth_range (Tuple[float, float]): The peak the linewidth is measured on.
        moving_averages (int): Transients averaged, an odd number, for the estimate.
        criteria (Optional[Sequence[str]]): The bounds PASTIS adjusts automatically,
            among 'amplitude', 'linewidth', 'frequency' and 'phase'; None for its fixed
            bounds (``peak_properties_ranges`` among the options).
        snr_change (float): The SNR change, in %, the adjustment aims for.
        apodization (float): Line broadening for the estimate only, in Hz; 0 for none.
        **options: Any other argument of PASTIS's correct_analyze_and_reject_2d, e.g.
            ``peak_properties_ranges``, ``reject_when_linewidth_fails``.

    Raises:
        ValueError: If a criterion is not one of the four.
    """
    if "DIM_DYN" not in list(data.dim_tags):
        return data                                                     # nothing to reject
    _, reco, _ = _modules()
    unknown = [c for c in criteria or () if c not in REJECTION]
    if unknown:
        raise ValueError(f"not a rejection criterion of PASTIS: {unknown}; choose among "
                         f"{sorted(REJECTION)}")
    methods = [getattr(reco.data_rejection_method, REJECTION[c]) for c in criteria] \
        if criteria else None
    mrs, added = _pastis(data, 2)
    out = mrs.correct_analyze_and_reject_2d(
        peak_analyze_range=list(peak_range), peak_snr_range=list(snr_range),
        peak_lw_range=list(linewidth_range), moving_averages=moving_averages,
        auto_method_list=methods, auto_adjust_allowed_snr_change=snr_change,
        allowed_apodization=apodization, display=False, **options)
    return _back(out, added, data)


step = Deferred(sys.modules[__name__])
