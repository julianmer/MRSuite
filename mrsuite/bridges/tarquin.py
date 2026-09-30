####################################################################################################
#                                            tarquin.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-28                                                                              #
#                                                                                                  #
# Purpose: Fitting with TARQUIN. Like LCModel it is a binary rather than a library, so             #
#          there is no function surface to dispatch onto: spectra go in as TARQUIN's               #
#          plain-text FIDs, the txt results come back as MRSuite's uniform result.                 #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import os
import re
import shutil
import subprocess
import sys
import tempfile
from functools import lru_cache
from typing import Any, Dict, Optional

import numpy as np

from mrsuite.bridges.base import Deferred
from mrsuite.core.method import method
from mrsuite.core.result import FitResult


#****************#
#   executable   #
#****************#
def _executable() -> str:
    """The TARQUIN binary: ``TARQUIN_EXEC`` chooses it, PATH is the fallback."""
    path = os.environ.get("TARQUIN_EXEC") or shutil.which("tarquin")
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(
            "no TARQUIN binary: set TARQUIN_EXEC or put 'tarquin' on PATH")
    return path


#*************#
#   version   #
#*************#
def _version() -> Optional[str]:
    """The binary's release, asked once per binary."""
    return _release(_executable())


#*************#
#   release   #
#*************#
@lru_cache(maxsize=None)
def _release(path: str) -> Optional[str]:
    """TARQUIN prints 'TARQUIN Version X' only once it has loaded a data file (not for
    --help), so it is given a tiny one; whether that fit succeeds does not matter."""
    with tempfile.TemporaryDirectory(prefix="mrsuite_tarquin_") as folder:
        probe = os.path.join(folder, "probe.dpt")
        with open(probe, "w") as f:
            f.write(_dpt_text(np.ones(64, complex), 1 / 2000.0, 123.2e6, 4.65, 0.03))
        try:
            proc = subprocess.run([path, "--input", probe, "--format", "dpt", "--max_iters",
                                   "1"], capture_output=True, text=True, timeout=120)
            out = proc.stdout + proc.stderr
        except subprocess.TimeoutExpired as err:
            out = (err.stdout or b"").decode(errors="ignore")
    return _banner(out)


#************#
#   banner   #
#************#
def _banner(text: str) -> Optional[str]:
    """The release in TARQUIN's splash, 'TARQUIN Version 4.3.11'."""
    found = re.search(r"TARQUIN Version (\S+)", text)
    return found.group(1) if found else None


#************#
#   binary   #
#************#
def _binary() -> Dict[str, Any]:
    """The binary that ran, recorded like LCModel's: TARQUIN is what fits."""
    return {"binary": _executable()}


#*********#
#   fit   #
#*********#
@method("TARQUIN", version=_version, runtime=_binary, name="TARQUIN", parallel=True)
def fit(data: Any, basis: Optional[str] = None, water: Any = None, te: Optional[float] = None,
        opts: Optional[Dict[str, Any]] = None) -> FitResult:
    """
    Fit with TARQUIN, as MRSuite's uniform result.

    Args:
        data: NIfTI-MRS or NIfTI-MRS+ of single spectra. A batch is fitted spectrum by
            spectrum.
        basis: An LCModel ``.basis``, as every fitter takes. TARQUIN does not resample:
            its sampling frequency must match the data's. By default TARQUIN rescales an
            LCModel basis to its sIns or Scyllo signal and stops if it has neither;
            ``{"rescale_lcm_basis": False}`` turns that off. None uses TARQUIN's default
            internal basis, ``1h_brain_glth``, simulated for the sequence at the data's
            echo time; ``{"int_basis": "1h_brain"}`` picks another.
        water: Unsuppressed reference, one per spectrum, enabling TARQUIN's water
            scaling. With it the concentrations are mM and filed under 'molarity';
            without it they are unreferenced and filed under 'raw'.
        te (Optional[float]): Echo time in seconds; read from the header when absent.
        opts (Optional[dict]): TARQUIN's command line options by name, e.g.
            ``{"pul_seq": "slaser", "max_iters": 50}``; anything not given is TARQUIN's
            own default - PRESS, and a start point it picks itself, skipping about the
            first 10 ms of the FID to keep broad baseline out of the fit.
        workers (int): Processes to spread a batch over (see ``mrsuite.core.parallel``).

    Returns:
        A FitResult, or one per spectrum for a batch, each with the data's processing
        history and the fit itself as its provenance.

    Raises:
        ValueError: If a spectrum is not a single FID, or the internal basis is asked for
            without an echo time to simulate it at.
    """
    echo = te if te is not None else _header(data, "EchoTime")
    if basis is None and echo is None:
        raise ValueError("TARQUIN simulates its internal basis at the data's echo time, which "
                         "the header does not state; pass te= or a basis")

    with tempfile.TemporaryDirectory(prefix="mrsuite_tarquin_") as folder:
        inputs = ["--input", _dpt(data, echo, os.path.join(folder, "in.dpt"))]
        if water is not None:
            inputs += ["--input_w", _dpt(water, echo, os.path.join(folder, "w.dpt"))]
        out = os.path.join(folder, "result.txt")
        cmd = [_executable(), *inputs, "--format", "dpt", "--output_txt", out]
        cmd += [] if basis is None else ["--basis_lcm", str(basis)]
        for key, value in (opts or {}).items():
            # TARQUIN reads only the exact string 'true' as true
            cmd += [f"--{key}", str(value).lower() if isinstance(value, bool) else str(value)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if not os.path.exists(out):
            raise RuntimeError(f"TARQUIN wrote no result:\n{(proc.stdout + proc.stderr)[-1000:]}")
        return _parse(out)


#*********#
#   dpt   #
#*********#
def _dpt(item: Any, echo: Optional[float], path: str) -> str:
    """One spectrum as TARQUIN's plain-text FID. Indexing a NIfTI-MRS conjugates, so the
    conj here hands TARQUIN the orientation NIfTI-MRS stores, which TARQUIN reads with
    a forward FFT and ppm = ref - f/ft."""
    if np.prod(item.shape) != item.shape[3]:
        raise ValueError(f"TARQUIN fits single spectra, not shape {item.shape}; combine coils "
                         f"and averages first")
    fid = np.conj(np.asarray(item[:], complex).reshape(-1))
    with open(path, "w") as f:
        f.write(_dpt_text(fid, item.dwelltime, float(item.spectrometer_frequency[0]) * 1e6,
                          item.SpecFreqChemShift, echo or 0.0))
    return path


#**************#
#   dpt text   #
#**************#
def _dpt_text(fid: np.ndarray, dwell: float, transmitter: float, reference: float,
              echo: float) -> str:
    """TARQUIN's plain-text FID format: a header of 'key<TAB>value', then the samples."""
    lines = ["Dangerplot_version\t1.0",
             f"Number_of_points\t{len(fid)}",
             f"Sampling_frequency\t{1.0 / dwell:8.8e}",
             f"Transmitter_frequency\t{transmitter:8.8e}",
             "Phi0\t0", "Phi1\t0",
             f"PPM_reference\t{reference:8.8e}",
             f"Echo_time\t{echo:8.8e}",
             "Real_FID\tImag_FID\t"]
    lines += [f"{x.real:8.8e} {x.imag:8.8e}" for x in fid]
    return "\n".join(lines) + "\n"


#************#
#   header   #
#************#
def _header(item: Any, field: str) -> Optional[float]:
    """A header field as a number, or None when the header does not state it."""
    hdr = getattr(item, "hdr_ext", None)
    return float(hdr[field]) if hdr is not None and field in hdr else None


#***********#
#   parse   #
#***********#
def _parse(path: str) -> FitResult:
    """
    TARQUIN's txt results, as its ExportTxtResults writes them.

    A table of signals - name, concentration, %SD, SD - whose header states the units,
    'Conc (au)' unreferenced or 'Conc (mM)' water-scaled, with the combined signals
    (TNAA, TCr, ...) after the single ones; then blocks of 'key : value' lines, of which
    'Fit quality' is kept as QC under snake_case names ('snr_max', 'metab_fwhm_hz', 'q').

    Raises:
        ValueError: If the table holds no signals, rather than returning an empty result.
    """
    with open(path) as f:
        blocks = [b.splitlines() for b in f.read().split("\n\n") if b.strip()]

    conc, pct, sd = {}, {}, {}
    for line in blocks[0][2:]:
        parts = line.split()
        if not parts:
            continue
        # fixed-width columns: a value that fills its column runs into the next
        if len(parts) != 4:
            raise ValueError(f"TARQUIN's results row cannot be read: {line!r} ({path})")
        name, c, p, s = parts
        conc[name], pct[name], sd[name] = float(c), float(p), float(s)
    if not conc:
        raise ValueError(f"TARQUIN's results in {path} hold no signals")

    qc = {}
    quality = next((b for b in blocks if b[0].strip() == "Fit quality"), [])
    for line in quality[2:]:
        key, _, value = line.partition(":")
        try:
            qc[re.sub(r"[^a-z0-9]+", "_", key.strip().lower()).strip("_")] = float(value)
        except ValueError:
            pass

    scaling = "molarity" if "(mM)" in blocks[0][0] else "raw"
    concentrations = {scaling: conc}
    tcr = conc.get("Cr", 0.0) + conc.get("PCr", 0.0)
    if tcr:
        concentrations["internal"] = {m: v / tcr for m, v in conc.items()}
    return FitResult(method="tarquin:TARQUIN", metabolites=list(conc),
                     concentrations=concentrations,
                     uncertainties={"percentage": pct, scaling: sd}, qc=qc)


# TARQUIN is a binary rather than a library, so like LCModel there is no bridge, but
# its fit is a pipeline step all the same: ``tarquin.step.fit(basis)``
step = Deferred(sys.modules[__name__])
