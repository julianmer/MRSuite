####################################################################################################
#                                             spant.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-27                                                                              #
#                                                                                                  #
# Purpose: Bridge onto spant, the R toolbox: every exported function as a step, and ABfit as a     #
#          fitter. R runs as a process of its own (see mrsuite/r), spectra cross as NIfTI-MRS      #
#          files, which spant reads and writes natively, and a batch travels as one job, since     #
#          starting R and loading spant costs about a second.                                      #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import json
import os
import re
import shutil
from typing import Any, Callable, Dict, List, Optional

import nibabel as nib
import numpy as np

from nifti_mrs.create_nmrs import gen_nifti_mrs
from nifti_mrs.nifti_mrs import NIFTI_MRS

from nifti_mrs_plus import NIfTI_MRS_Plus
from mrsuite import r
from mrsuite.bridges.base import Bridge
from mrsuite.core import provenance
from mrsuite.core.parallel import over_processes
from mrsuite.core.result import FitResult
from mrsuite.octave import scratch


#***************#
#   read back   #
#***************#
def _read_back(path: str, source: NIFTI_MRS) -> NIFTI_MRS:
    """A spectrum spant wrote, rebuilt on the header of the one it came from.

    Only the samples and the dimensions are taken from spant's file. Its header extension
    is not NIfTI-MRS as other readers expect it - tags as lists, a tag for every possible
    dimension whether the data has it or not - so everything else comes from ``source``,
    which the bridge's header keeping then restores in full.
    """
    image = nib.load(path)
    samples = np.asarray(image.dataobj)
    ext = [e for e in image.header.extensions if e.get_code() == 44]
    fields = json.loads(ext[0].get_content()) if ext else {}

    if samples.size == np.prod(source.shape) and samples.shape[3] == source.shape[3]:
        # the dimensions are unchanged, though spant writes length-one ones away
        samples, tags = samples.reshape(source.shape), list(source.dim_tags)
    else:
        tags = []
        for n in range(5, samples.ndim + 1):
            tag = fields.get(f"dim_{n}")
            tags.append(tag[0] if isinstance(tag, list) else tag)

    # the dwell time survives only to float32 in pixdim; keep the exact one unless it changed
    dwell = float(image.header["pixdim"][4])
    if abs(dwell - source.dwelltime) <= 1e-6 * source.dwelltime:
        dwell = source.dwelltime
    tags = (tags + [None] * 3)[:3]
    return gen_nifti_mrs(samples, dwell, source.spectrometer_frequency[0],
                         nucleus=source.nucleus[0], dim_tags=tags, no_conj=False)


#**************************************************************************************************#
#                                        Class SpantBridge                                         #
#**************************************************************************************************#
#                                                                                                  #
# Dispatches onto spant's exported functions, a batch as one R job.                                #
#                                                                                                  #
#**************************************************************************************************#
class SpantBridge(Bridge):
    """Dispatch onto spant's exported functions, a batch as one R job.

    Arguments that are spectra - a reference, say - are passed by keyword, and go per
    spectrum like the data; everything else is shared by the whole batch.
    """

    name = "spant"
    program = "spant"

    #*************#
    #   resolve   #
    #*************#
    def _resolve(self, func: str) -> Callable:
        if func not in r.info()["functions"]:
            raise AttributeError(f"spant has no function '{func}'.")
        return func                     # run by _invoke, which batches the call into one job

    #****************#
    #   to toolbox   #
    #****************#
    def _to_toolbox(self, obj):
        return obj

    #******************#
    #   from toolbox   #
    #******************#
    def _from_toolbox(self, obj):
        return obj

    #***************#
    #   available   #
    #***************#
    def available(self) -> List[str]:
        return r.info()["functions"]

    #*************#
    #   version   #
    #*************#
    def version(self) -> str:
        return r.info()["spant"]

    #*************#
    #   runtime   #
    #*************#
    def runtime(self) -> Dict[str, Any]:
        return {"runtime": r.resolve_runtime(), "interpreter": r.info()["r"]}

    #************#
    #   invoke   #
    #************#
    def _invoke(self, func, target, args, kwargs):
        """Apply ``func`` to one spectrum or a batch, as a single R job."""
        self._skipped = []
        if not args or not isinstance(args[0], (NIFTI_MRS, NIfTI_MRS_Plus)):
            raise TypeError(f"spant.{func}: the data comes first")
        if any(isinstance(a, (NIFTI_MRS, NIfTI_MRS_Plus)) for a in args[1:]):
            raise TypeError(f"spant.{func}: pass further spectra by keyword")
        batched = isinstance(args[0], NIfTI_MRS_Plus)
        items = args[0].list() if batched else [args[0]]
        spectra = {k: v for k, v in kwargs.items() if isinstance(v, (NIFTI_MRS, NIfTI_MRS_Plus))}
        plain = {k: v for k, v in kwargs.items() if k not in spectra}

        folder = scratch("mrsuite_spant_")
        try:
            inputs = _write(items, folder, "in")
            outputs = [os.path.join(folder, f"out{i}.nii.gz") for i in range(len(items))]
            per_item = {k: _write(_per_item(v, len(items)), folder, k) for k, v in spectra.items()}
            r.run({"task": "call", "func": func, "inputs": inputs, "outputs": outputs,
                   "refs": [item.SpecFreqChemShift for item in items],
                   "args": _plain(list(args[1:])), "kwargs": _plain(plain),
                   "spectra": per_item}, folder)
            results = [_read_back(o, item) for o, item in zip(outputs, items)]
        finally:
            shutil.rmtree(folder, ignore_errors=True)

        item = provenance.entry(self.program, self.version(), func,
                                {"args": args[1:], "kwargs": kwargs, **self.runtime()})
        for out, source in zip(results, items):
            self._keep_header(source, out)
            provenance.append(out, item)
        return NIfTI_MRS_Plus(results) if batched else results[0]


bridge = SpantBridge()


#***********#
#   write   #
#***********#
def _write(items: List[NIFTI_MRS], folder: str, stem: str) -> List[str]:
    """Spectra as NIfTI-MRS files in ``folder``, for R to read.

    Copies are saved, not the caller's objects: nibabel repoints a saved image at its
    file, which is deleted once the job is done.
    """
    paths = [os.path.join(folder, f"{stem}{i}.nii.gz") for i in range(len(items))]
    for path, item in zip(paths, items):
        item.copy().save(path)
    return paths


#**************#
#   per item   #
#**************#
def _per_item(spectra: Any, n: int) -> List[NIFTI_MRS]:
    """One spectrum per data spectrum: a batch as it is, a single one for all."""
    if isinstance(spectra, NIfTI_MRS_Plus):
        if len(spectra) != n:
            raise ValueError(f"{len(spectra)} spectra for {n}; give one per spectrum")
        return spectra.list()
    return [spectra] * n


#***********#
#   plain   #
#***********#
def _plain(value: Any) -> Any:
    """Arguments as JSON R can read: NumPy values as numbers and lists."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


#************#
#   ppmsep   #
#************#
def _ppmsep(path: str) -> Optional[float]:
    """The ppm at the centre an LCModel .BASIS states, if it states one."""
    with open(path) as fh:
        found = re.search(r"PPMSEP\s*=\s*([-+0-9.eEdD]+)", fh.read())
    return float(found.group(1).replace("D", "e").replace("d", "e")) if found else None


#***************#
#   to result   #
#***************#
def to_result(row: Dict[str, Any], amplitudes: List[str], method: str) -> FitResult:
    """
    Map one row of spant's result table onto MRSuite's uniform result.

    Args:
        row (Dict[str, Any]): A row of ``fit_mrs``'s ``res_tab``.
        amplitudes (List[str]): Its amplitude columns: the basis, then spant's sums.
        method (str): The fitting method, e.g. 'ABFIT'.

    Returns:
        FitResult: Amplitudes, their standard deviations and CRLBs, the creatine-referenced
        values, and spant's own quality measures.
    """
    raw = {m: float(row[m]) for m in amplitudes if row.get(m) is not None}
    sd = {m: float(row[f"{m}.sd"]) for m in raw if row.get(f"{m}.sd") is not None}
    concentrations = {"raw": raw}
    if raw.get("tCr"):
        concentrations["internal"] = {m: v / raw["tCr"] for m, v in raw.items()}
    uncertainties = {"raw": sd,
                     "percentage": {m: 100 * s / raw[m] for m, s in sd.items() if raw[m]}}
    qc = {name: float(row[key]) for key, name in (("SNR", "snr"), ("lw", "fwhm"), ("FQN", "fqn"),
                                                  ("phase", "phase"), ("shift", "shift"))
          if isinstance(row.get(key), (int, float))}
    return FitResult(method=f"spant:{method}", metabolites=list(raw),
                     concentrations=concentrations, uncertainties=uncertainties, qc=qc, raw=row)


#*********#
#   fit   #
#*********#
@over_processes
def fit(data: Any, basis: str, method: str = "ABFIT", opts: Optional[dict] = None) -> Any:
    """
    Fit with spant, as MRSuite's uniform result.

    Args:
        data: NIfTI-MRS or NIfTI-MRS+. A batch is fitted spectrum by spectrum, in one job.
        basis (str): An LCModel ``.BASIS`` file, as every fitter takes; spant resamples it
            to the data. Its ppm reference is read from PPMSEP, else taken as 4.65.
        method (str): spant's fitting method.
        opts (Optional[dict]): Options for ``abfit_opts``, by name.
        workers (int): Processes to spread a batch over (see ``over_processes``).

    Returns:
        A FitResult, or one per spectrum for a batch, each with the data's processing
        history and the fit itself as its provenance.
    """
    items = data.list() if isinstance(data, NIfTI_MRS_Plus) else [data]
    reference = _ppmsep(str(basis))
    folder = scratch("mrsuite_spant_")
    try:
        # copied in, so that a basis kept outside the repo is visible in the container too
        copy = shutil.copy(basis, os.path.join(folder, "basis.BASIS"))
        out = r.run({"task": "fit", "inputs": _write(items, folder, "in"),
                     "refs": [item.SpecFreqChemShift for item in items],
                     "basis": copy,
                     "basis_ref": 4.65 if reference is None else reference,
                     "method": method, "opts": _plain(opts or {})}, folder)
    finally:
        shutil.rmtree(folder, ignore_errors=True)

    options = {"basis": provenance.file_checksum(basis), "method": method, "opts": opts,
               **bridge.runtime()}
    results = []
    for row, item in zip(out["rows"], items):
        result = to_result(row, out["amplitudes"], method)
        result.provenance = provenance.history(item) + [
            provenance.entry(bridge.program, bridge.version(), "fit_mrs", options)]
        results.append(result)
    return results if isinstance(data, NIfTI_MRS_Plus) else results[0]


#*************#
#   getattr   #
#*************#
def __getattr__(name: str) -> Any:
    """Module-level dispatch, so `from mrsuite.bridges.spant import lb` works."""
    return getattr(bridge, name)


#*********#
#   dir   #
#*********#
def __dir__() -> List[str]:
    return sorted(set(globals()) | set(bridge.available()))
