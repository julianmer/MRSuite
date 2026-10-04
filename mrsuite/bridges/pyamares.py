####################################################################################################
#                                            pyamares.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-29                                                                              #
#                                                                                                  #
# Purpose: Fitting with pyAMARES (Xu et al., Diagnostics 2024). AMARES is parametric peak          #
#          fitting rather than a linear combination of basis spectra, so instead of a              #
#          basis it takes a prior-knowledge file - peaks with initial values, bounds and           #
#          expression constraints - and its CRLBs come from the same Fisher machinery as           #
#          jMRUI's. The library is one function surface deep, so like TARQUIN it needs no          #
#          Bridge: a hand-written fit maps its result table onto MRSuite's uniform result.         #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from mrsuite.bridges.base import Deferred
from mrsuite.core.method import method
from mrsuite.core.result import FitResult


#*************#
#   version   #
#*************#
def _version() -> Optional[str]:
    """pyAMARES' exact version, None when it is not installed."""
    try:
        import pyAMARES
        return pyAMARES.__version__
    except ImportError:
        return None


#*********#
#   fit   #
#*********#
@method("pyAMARES", version=_version, name="AMARES", parallel=True)
def fit(data: Any, prior: str, opts: Optional[Dict[str, Any]] = None) -> FitResult:
    """
    Fit with pyAMARES, as MRSuite's uniform result.

    Args:
        data: NIfTI-MRS or NIfTI-MRS+ of single spectra. A batch is fitted spectrum by
            spectrum.
        prior: pyAMARES' prior-knowledge file (.csv or .xlsx): one column per peak, with
            initial values, bounds and expression constraints. This is AMARES' analogue
            of a basis; multiplet components named like ``Glu``, ``Glu2`` are summed into
            one ``Glu`` in the result. Comment lines ``# mrsuite: Crdj = Cr39 per 0.9997``,
            as ``prior`` writes them, rename a peak group and divide its summed amplitude
            by its amplitude at unit concentration, which puts each metabolite on its
            basis function's scale.
        opts (Optional[dict]): Options by name. ``method``, ``initialize_with_lm``,
            ``fit_range`` and ``fit_kws`` steer ``fitAMARES``, the last lmfit's own, e.g.
            ``{"fit_kws": {"max_nfev": 10000}}`` past pyAMARES' 1000 evaluations, which a
            prior of many peaks exhausts; everything else is passed to ``initialize_FID``,
            e.g. ``{"truncate_initial_points": 2}``. A fit that stops before converging
            warns.
        workers (int): Processes to spread a batch over (see ``mrsuite.core.parallel``).

    Returns:
        A FitResult, or one per spectrum for a batch. Concentrations are peak amplitudes
        ('raw', plus 'internal' over the tCr peak when the prior has one); 'percentage'
        holds the amplitude CRLBs.

    Raises:
        ValueError: If a spectrum is not a single FID.
        ImportError: If pyAMARES is not installed.
    """
    if np.prod(data.shape) != data.shape[3]:
        raise ValueError(f"pyAMARES fits single spectra, not shape {data.shape}; combine "
                         f"coils and averages first")
    try:
        import pyAMARES
    except ImportError as err:
        raise ImportError("pyAMARES is not installed: pip install pyAMARES") from err

    # the NIfTI-MRS array and pyAMARES' model share numpy's FFT sign convention, so the
    # FID goes over as stored; the carrier puts the prior's absolute ppm on the axis
    # (pinned by the physiology test, as for TARQUIN). The prior's initial amplitudes
    # and fitAMARES' tolerance (from max |FID|) assume data of order one, so the FID is
    # fitted at unit scale and the amplitudes scaled back; pyAMARES' own normalize_fid
    # divides by numpy's max of a complex array, which rotates the phase.
    fid = np.asarray(data[:], complex).reshape(-1)
    scale = float(np.abs(fid).max()) or 1.0
    options = dict(opts or {})
    fitting = {key: options.pop(key)
               for key in ("method", "initialize_with_lm", "fit_range", "fit_kws")
               if key in options}
    if "fit_range" in fitting:                   # the default objective takes no range
        from pyAMARES.kernel.objective_func import objective_range
        fitting["objective_func"] = objective_range
    settings = {"MHz": float(data.spectrometer_frequency[0]), "sw": 1.0 / data.dwelltime,
                "deadtime": 0.0, "carrier": float(data.SpecFreqChemShift), "g_global": False,
                **options}
    params = pyAMARES.initialize_FID(fid / scale, priorknowledgefile=str(prior), **settings)
    fitted = pyAMARES.fitAMARES(params, params.initialParams, ifplot=False, inplace=False,
                                **fitting)
    out = getattr(fitted, "out_obj", None)
    if out is not None and not out.success:
        warnings.warn(f"pyAMARES stopped before converging: {out.message}; for a prior of "
                      f"many peaks, raise opts['fit_kws']['max_nfev']")
    return _result(fitted, scale, *_metabolites(str(prior)))


#***********#
#   prior   #
#***********#
def prior(path: str, basis: Dict[str, np.ndarray], dwell: float, frequency: float,
          centre: float, tolerance: float = 0.01) -> str:
    """
    Write a pyAMARES prior-knowledge file that reproduces basis functions.

    AMARES fits peaks, so each basis function becomes peaks with one free amplitude:
    its exponentially damped components, found by HSVD (pyAMARES' own HLSVDPRO) of the
    function, then of what is left of it, three times, amplitudes refitted by least
    squares; a growing component (HSVD's answer to lines closer than the FID resolves)
    is kept undamped. The strongest components are kept until the rest is within
    ``tolerance`` of the function's norm, or as close as the decomposition gets, with a
    warning. The strongest is the group's peak, named after the metabolite, and every
    other component follows it in amplitude ratio, offset in frequency (Hz), damping
    and phase: each group then has one amplitude, one shift (+-0.1 ppm) and one extra
    Lorentzian linewidth (up to 30 Hz; g is held at 0, the function's own lineshape
    being in its components), and all groups share one zero-order phase. Comment lines
    name each group and give its summed amplitude at unit concentration, which ``fit``
    divides by. pyAMARES takes digits in peak names as multiplet suffixes, so a name's
    digits are spelt as letters (a = 0, .., j = 9), and any other character but a letter
    or an underscore as an underscore.

    Args:
        path (str): The .csv to write.
        basis (Dict[str, np.ndarray]): Basis FIDs by metabolite, on one time axis from 0
            and in NIfTI-MRS orientation.
        dwell (float): Their dwell time in s.
        frequency (float): Spectrometer frequency in MHz.
        centre (float): The ppm at 0 Hz.
        tolerance (float): The residual each function's peaks may leave, relative to its
            norm.

    Returns:
        str: ``path``.

    Raises:
        ValueError: If two names are spelt alike.
    """
    t = np.arange(len(next(iter(basis.values())))) * dwell
    aliases = {name: re.sub(r"[^A-Za-z_]", lambda c: "abcdefghij"[int(c.group())]
                            if c.group().isdigit() else "_", name) for name in basis}
    if len(set(aliases.values())) < len(aliases):
        raise ValueError(f"names that pyAMARES would read alike: {aliases}")
    columns, header, reference = {}, [], None
    for name, fid in basis.items():
        alias = aliases[name]
        hz, rate, amp = _components(np.asarray(fid, complex), t, tolerance)
        residual = _residual(fid, t, hz, rate, amp)
        if residual > tolerance:
            warnings.warn(f"{name}'s peaks leave {residual:.1%} of it, above the tolerance "
                          f"{tolerance:.1%}")
        width, phase = rate[0] / np.pi, np.angle(amp[0])
        if reference is None:
            reference = alias, phase
            group_phase = (f"{np.degrees(phase):.4f}", "(-180, 180)")
        else:
            group_phase = (f"{reference[0]}{phase - reference[1]:+.6f}", "")
        ppm = centre + hz[0] / frequency
        columns[alias] = [f"{abs(amp[0]):.6f}", f"{ppm:.5f}", f"{width + 5.0:.4f}",
                          group_phase[0], "0", "", "(0, ", f"({ppm - 0.1:.5f}, {ppm + 0.1:.5f})",
                          f"({width:.4f}, {width + 30.0:.4f})", group_phase[1], "(0, 0)"]
        # every expression on a free parameter, as pyAMARES' Cramer-Rao matrix needs:
        # the phase on the reference group's, the lineshape held at 0 like the peak's
        for k in range(1, len(hz)):
            columns[f"{alias}{k + 1}"] = [
                f"{alias}*{abs(amp[k]) / abs(amp[0]):.8f}", f"{alias}{hz[k] - hz[0]:+.5f}Hz",
                f"{alias}{rate[k] - rate[0]:+.5f}",
                f"{reference[0]}{np.angle(amp[k]) - reference[1]:+.6f}",
                "0", "", "", "", "", "", "(0, 0)"]
        header.append(f"# mrsuite: {alias} = {name} per {np.abs(amp).sum():.8f}")

    rows = ["amplitude", "chemicalshift", "linewidth", "phase", "g", "Bounds",
            "amplitude", "chemicalshift", "linewidth", "phase", "g"]
    lines = header + ["Index," + ",".join(columns), "Initial Values" + "," * len(columns)]
    lines += [",".join([row] + [f'"{c[i]}"' if "," in c[i] else c[i] for c in columns.values()])
              for i, row in enumerate(rows)]
    Path(path).write_text("\n".join(lines) + "\n")
    return path


#****************#
#   components   #
#****************#
def _components(fid: np.ndarray, t: np.ndarray,
                tolerance: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A function's damped components, strongest first: frequencies (Hz), damping rates
    (1/s) and complex amplitudes (see ``prior``)."""
    from pyAMARES.libs.hlsvd import hlsvdpro

    def refit(hz, rate):
        model = np.exp((2j * np.pi * hz - rate) * t[:, None])
        amp = np.linalg.lstsq(model, fid, rcond=None)[0]
        return amp, np.linalg.norm(fid - model @ amp) / np.linalg.norm(fid)

    dwell = t[1] - t[0]
    hz, rate, size, rest = np.empty(0), np.empty(0), np.empty(0), fid
    for _ in range(3):
        found = hlsvdpro(rest, 50, sparse=True)
        new_hz, new_rate = found[2] / dwell, -found[3] / dwell    # negative damping decays
        # what is left of a component is found again, and two near-equal components
        # would cancel at large amplitudes: within a tenth of the resolution they are one
        known = np.abs(new_hz[:, None] - hz).min(1, initial=np.inf) < 0.1 / t[-1]
        # a growing component stands for lines closer than the FID resolves: kept undamped
        keep = ~known
        hz, rate = np.r_[hz, new_hz[keep]], np.r_[rate, np.maximum(new_rate[keep], 0.0)]
        size = np.r_[size, found[4][keep]]
        amp, residual = refit(hz, rate)
        rest = fid - np.exp((2j * np.pi * hz - rate) * t[:, None]) @ amp
        if residual <= tolerance:
            break

    # strongest first by HSVD's own amplitudes: least-squares ones of near-equal
    # components can be large and cancel
    order = np.argsort(size ** 2 * np.exp(-2.0 * rate * t[:, None]).sum(0))[::-1]
    target = max(tolerance, 1.01 * residual)
    for k in range(1, len(order) + 1):
        amp, residual = refit(hz[order[:k]], rate[order[:k]])
        if residual <= target:
            break
    return hz[order[:k]], rate[order[:k]], amp


#**************#
#   residual   #
#**************#
def _residual(fid: np.ndarray, t: np.ndarray, hz: np.ndarray, rate: np.ndarray,
              amp: np.ndarray) -> float:
    """What the components leave of a function, relative to its norm."""
    model = np.exp((2j * np.pi * hz - rate) * t[:, None]) @ amp
    return float(np.linalg.norm(fid - model) / np.linalg.norm(fid))


#*****************#
#   metabolites   #
#*****************#
def _metabolites(path: str) -> Tuple[Dict[str, str], Dict[str, float]]:
    """The peak groups a prior file renames, and their amplitudes at unit concentration,
    from its ``# mrsuite: alias = name per unit`` comment lines."""
    names, units = {}, {}
    for match in re.finditer(r"^#\s*mrsuite:\s*(\S+)\s*=\s*(\S+)\s+per\s+(\S+)",
                             Path(path).read_text(errors="ignore"), re.MULTILINE):
        names[match[1]], units[match[1]] = match[2], float(match[3])
    return names, units


#************#
#   result   #
#************#
def _result(fitted: Any, scale: float = 1.0, names: Optional[Dict[str, str]] = None,
            units: Optional[Dict[str, float]] = None) -> FitResult:
    """
    pyAMARES' result table as MRSuite's uniform result.

    ``result_sum`` (multiplets summed by base name) carries per peak the amplitude and its
    absolute SD; ``result_multiplets`` is the fallback when the prior named no peaks to
    sum. The CRLB percentage is taken as SD over amplitude: for a summed multiplet the
    table's own CRLB(%) column is its first component's. ``scale`` undoes the unit-scale
    normalization the fit ran at; ``names`` and ``units`` rename peak groups and divide
    them by their amplitude at unit concentration (see ``_metabolites``).

    Raises:
        ValueError: If the fit produced no peaks, rather than returning an empty result.
    """
    table = getattr(fitted, "result_sum", None)
    if table is None:
        table = getattr(fitted, "result_multiplets", None)
    if table is None or not len(table):
        raise ValueError("pyAMARES produced no result table")

    names, units = names or {}, units or {}
    conc, sd = {}, {}
    for peak, row in table.iterrows():
        name, per = names.get(str(peak), str(peak)), scale / units.get(str(peak), 1.0)
        conc[name], sd[name] = per * float(row["amplitude"]), per * float(row["sd"])
    pct = {m: 100.0 * sd[m] / conc[m] if conc[m] else float("nan") for m in conc}

    qc = {}
    if "SNR" in table.columns and np.isfinite(table["SNR"]).any():
        qc["snr_max"] = float(np.nanmax(table["SNR"]))

    concentrations = {"raw": conc}
    tcr = conc.get("tCr") or conc.get("Cr", 0.0) + conc.get("PCr", 0.0)
    if tcr:
        concentrations["internal"] = {m: v / tcr for m, v in conc.items()}
    # the table rather than pyAMARES' whole namespace, which holds a pandas Styler that
    # cannot be pickled back from a worker process
    return FitResult(method="pyamares:AMARES", metabolites=list(conc),
                     concentrations=concentrations,
                     uncertainties={"percentage": pct, "raw": sd}, qc=qc, raw=table)


# pyAMARES is a library with a single entry point, so like TARQUIN there is no bridge,
# but its fit is a pipeline step all the same: ``pyamares.step.fit(prior)``
step = Deferred(sys.modules[__name__])
