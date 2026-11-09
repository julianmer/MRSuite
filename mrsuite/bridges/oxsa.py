####################################################################################################
#                                              oxsa.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 09/11/26                                                                                #
#                                                                                                  #
# Purpose: Fitting with OXSA (Purvis et al., PLoS ONE 2017), the Oxford 31P toolbox, run as        #
#          published in an Octave (or MATLAB) session of its own: its AMARES, a parametric fit     #
#          of peaks with prior knowledge (Vanhamme et al. 1997) with Cramer-Rao bounds, from its   #
#          own prior-knowledge sets (cardiac 3 T and 7 T, liver 7 T, single peaks) or one given    #
#          as its three tables, as MRSuite's uniform result.                                       #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import sys
from typing import Any, Dict, List, Optional, Union

import numpy as np

from mrsuite import externals
from mrsuite.bridges.base import Deferred
from mrsuite.core.method import method
from mrsuite.core.result import FitResult
from mrsuite.octave.toolbox import Toolbox

# OXSA's repository holds its toolbox under main/, beside its sample data and a compiled GUI
toolbox = Toolbox("oxsa", "OXSA", drivers="oxsa", within="main")


#************#
#   priors   #
#************#
def priors() -> List[str]:
    """The prior-knowledge sets OXSA ships (its AMARES.priorKnowledge), by name: the ones
    that stand on their own, not its ``*_chooseInit`` makers, which take the initial values
    as arguments (a set of your own goes to ``fit`` as its tables)."""
    folder = externals.ensure("oxsa") / "main" / "+AMARES" / "+priorKnowledge"
    return sorted(path.stem for path in folder.glob("PK_*.m")
                  if not path.stem.endswith("_chooseInit"))


#*********#
#   fit   #
#*********#
@method("OXSA", version=toolbox.version, runtime=toolbox.runtime, name="AMARES")
def fit(data: Any, prior: Union[str, Dict[str, Any]], begin_time: float = 0.0,
        opts: Optional[Dict[str, Any]] = None) -> FitResult:
    """
    Fit with OXSA's AMARES, as MRSuite's uniform result.

    Each peak has an amplitude, a chemical shift, a linewidth and a phase, bounded and
    linked as the prior knowledge says (multiplets with fixed splittings and amplitude
    ratios, shared linewidths or phases, a Gaussian width for Voigt lines). OXSA counts
    chemical shifts from the carrier, as its 31P data put PCr there: a shift of 0 is the
    spectrum's centre (SpecFreqChemShift), and the spectrum goes over as NIfTI-MRS holds
    it, since OXSA shares its orientation.

    Args:
        data: NIfTI-MRS or NIfTI-MRS+ of single spectra. A batch is fitted spectrum by
            spectrum.
        prior: The name of one of OXSA's sets (``priors()``: 'PK_3T_Cardiac',
            'PK_7T_Cardiac', 'PK_7T_Liver', 'PK_SinglePeak', ...), or a set as OXSA
            writes one, its three tables as lists of one dict per peak: ``bounds``
            (peakName, chemShift, linewidth, amplitude, phase, chemShiftDelta,
            amplitudeRatio), ``initialValues`` (peakName, chemShift, linewidth,
            amplitude, phase) and ``priorKnowledge`` (peakName, multiplet,
            chemShiftDelta, amplitudeRatio, G_linewidth, G_amplitude, G_phase,
            G_chemShiftDelta, refPeak); a multiplet's peakName is a list of its
            components. Chemical shifts in ppm from the carrier, linewidths in Hz,
            phases in degrees, bounds as [low, high]; an empty list where OXSA has [].
        begin_time (float): The time of the first point in s (OXSA's beginTime, its
            first-order phase); 0 for processed spectra.
        opts (Optional[dict]): amaresFit's options by name: ``MaxIter``, ``MaxFunEvals``,
            ``TolFun``, ``noiseVariance``, ``firstOrder``.

    Returns:
        A FitResult, or one per spectrum for a batch. Concentrations: 'raw' the
        amplitudes, 'internal' their ratio to the prior's reference peak where it has
        one; uncertainties: 'raw' the Cramer-Rao bounds of the amplitudes and
        'percentage' their share; qc: the residual's norm over the data's, the
        iterations and the solver's exit flag; raw: every fitted parameter and bound per
        peak, the chemical shifts also in the spectrum's ppm.

    Raises:
        ValueError: If a spectrum is not a single FID, or the fit returns non-finite
            amplitudes.
        RuntimeError: With OXSA's message when it refuses the prior or fails to fit.
    """
    if np.prod(data.shape) != data.shape[3]:
        raise ValueError(f"OXSA fits single spectra, not shape {data.shape}; combine coils "
                         f"and averages first")
    fid = np.asarray(data[:], complex).reshape(-1, 1)
    try:                                      # the session's error, Octave's or MATLAB's
        out = toolbox.octave.feval("mrsuite_oxsa_fit", fid, float(data.dwelltime),
                                   float(data.spectrometer_frequency[0]), float(begin_time),
                                   prior, priors(), opts or {}, nout=1)
    except Exception as err:                  # noqa: BLE001 - said as OXSA's, with its message
        said = [line for line in str(err).splitlines()
                if line.startswith("error: ") and "called from" not in line]
        raise RuntimeError(f"OXSA: {said[0][7:] if said else err}") from err
    return _result(out, float(data.SpecFreqChemShift))


#************#
#   result   #
#************#
def _result(out: Dict[str, Any], centre: float = 0.0) -> FitResult:
    """
    What the driver returned, as MRSuite's uniform result.

    Args:
        out (dict): ``names`` (one per fitted component), ``results`` and ``crb`` (OXSA's
            fitResults and CRBResults: amplitude, chemShift, linewidth, phase, and sigma
            for Voigt lines, one value per component), ``reference`` (the reference
            component, 1-based, or empty), ``exitflag``, ``iterations``,
            ``relative_norm``, ``residual_norm``.
        centre (float): The spectrum's centre in ppm, which OXSA's shifts count from.

    Raises:
        ValueError: If a fitted amplitude is not finite.
    """
    # a cell of chars, or one char for one peak
    names = [str(n) for n in np.asarray(out["names"], dtype=object).reshape(-1)]
    values = {k: np.asarray(v, float).reshape(-1) for k, v in dict(out["results"]).items()}
    bounds = {k: np.asarray(v, float).reshape(-1) for k, v in dict(out["crb"]).items()
              if k != "covariance"}
    amplitudes = dict(zip(names, (float(a) for a in values["amplitude"])))
    if not all(np.isfinite(list(amplitudes.values()))):
        raise ValueError(f"OXSA returned non-finite amplitudes: {amplitudes}")
    crlb = dict(zip(names, (float(b) for b in bounds["amplitude"])))
    concentrations = {"raw": amplitudes}
    reference = np.asarray(out["reference"]).reshape(-1)
    if reference.size and amplitudes[names[int(reference[0]) - 1]]:
        ref = amplitudes[names[int(reference[0]) - 1]]
        concentrations["internal"] = {n: a / ref for n, a in amplitudes.items()}
    percentage = {n: 100.0 * crlb[n] / a for n, a in amplitudes.items()
                  if a and np.isfinite(crlb[n])}
    qc = {"relative_norm": float(out["relative_norm"]), "iterations": float(out["iterations"]),
          "exitflag": float(out["exitflag"])}
    return FitResult(
        method="oxsa:AMARES",
        metabolites=names,
        concentrations=concentrations,
        uncertainties={"raw": crlb, **({"percentage": percentage} if percentage else {})},
        qc={k: v for k, v in qc.items() if np.isfinite(v)},
        raw={"names": names, "parameters": values, "bounds": bounds,
             "ppm": dict(zip(names, (centre + float(s) for s in values["chemShift"]))),
             "covariance": np.asarray(dict(out["crb"]).get("covariance")),
             "residual_norm": float(out["residual_norm"])})


step = Deferred(sys.modules[__name__])
