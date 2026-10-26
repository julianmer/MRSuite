####################################################################################################
#                                           mrspeclab.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-02                                                                              #
#                                                                                                  #
# Purpose: Bridge onto MRSpecLAB's processing nodes (github.com/MRSEPFL/MRSpecLAB): every node     #
#          in its registry by name, its parameters by their idnames. MRSpecLAB is a wxPython       #
#          GUI whose nodes need old NumPy and SciPy, so it runs unchanged in an environment of     #
#          its own, through mrspeclab_driver.py, a batch as one job; MRSuite never imports it.     #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import inspect
import json
import os
import shutil
import subprocess
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np

from nifti_mrs.create_nmrs import gen_nifti_mrs
from nifti_mrs.nifti_mrs import NIFTI_MRS

from nifti_mrs_plus import NIfTI_MRS_Plus
from mrsuite import externals
from mrsuite.bridges.base import Bridge
from mrsuite.core.method import keep_header
from mrsuite.core.provenance import git_version
from mrsuite.octave import scratch

ROOT = externals.path("mrspeclab")
DRIVER = Path(__file__).with_name("mrspeclab_driver.py")

# the Python MRSpecLAB runs in: one of your own, else an environment MRSuite builds with uv
PYTHON = os.environ.get("MRSUITE_MRSPECLAB_PYTHON")
ENVIRONMENT = Path.home() / ".cache" / "mrsuite" / "mrspeclab"

# what the nodes import, at the versions MRSpecLAB's requirements.txt allows (NumPy and SciPy
# capped for suspect and mapVBVD), resolved and verified with Python 3.11 on macOS; what they
# import in turn, as uv resolved it on this date
REQUIREMENTS = ("wxpython==4.2.1", "gsnodegraph==0.5.6", "gswidgetkit==0.3.3", "suspect==0.6.2",
                "numpy==1.23.5", "scipy==1.13.1", "matplotlib==3.10.9", "nibabel==5.3.3",
                "pandas==2.3.3", "pydicom==3.0.2", "pymapvbvd==0.5.7")
RESOLVED = "2026-10-03"


#************#
#   python   #
#************#
def python() -> str:
    """The interpreter MRSpecLAB runs in, building MRSuite's environment for it on first use."""
    if PYTHON:
        return PYTHON
    marker = ENVIRONMENT / "requirements.txt"         # written last: the build is complete
    wanted = "\n".join(REQUIREMENTS + (f"# resolved as of {RESOLVED}",))
    if not marker.exists() or marker.read_text() != wanted:
        uv = shutil.which("uv")
        if uv is None:
            raise RuntimeError("MRSpecLAB runs in an environment of its own, which MRSuite "
                               "builds with uv: install uv, or set MRSUITE_MRSPECLAB_PYTHON")
        for command in ([uv, "venv", "--clear", "--python", "3.11", str(ENVIRONMENT)],
                        [uv, "pip", "install", "--python", str(ENVIRONMENT / "bin" / "python"),
                         "--exclude-newer", RESOLVED, *REQUIREMENTS]):
            proc = subprocess.run(command, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(f"building MRSpecLAB's environment failed:\n"
                                   f"{proc.stderr[-3000:]}")
        marker.write_text(wanted)
    return str(ENVIRONMENT / "bin" / "python")


#************#
#   driver   #
#************#
def _driver(job: Dict[str, Any], folder: str) -> None:
    """One job through mrspeclab_driver.py, in the job's folder rather than the checkout, which
    stays as it was: a node that writes files writes them there, and the bytecode goes to the
    environment's cache."""
    path = os.path.join(folder, "job.json")
    with open(path, "w") as fh:
        json.dump({**job, "root": str(externals.ensure("mrspeclab"))}, fh)
    proc = subprocess.run(
        [python(), "-P", str(DRIVER), path], cwd=folder, capture_output=True, text=True,
        env={**os.environ, "MPLBACKEND": "Agg",
             "PYTHONPYCACHEPREFIX": str(ENVIRONMENT / "pycache")})
    if proc.returncode != 0 or not os.path.exists(job["result"]):
        raise RuntimeError(f"MRSpecLAB failed:\n{proc.stderr[-3000:]}")
    for line in proc.stdout.splitlines():               # what its nodes print, without the GUI
        if "ERROR" in line:
            warnings.warn(f"MRSpecLAB: {line.strip()}")


#**********#
#   info   #
#**********#
@lru_cache(maxsize=None)
def info() -> Dict[str, Any]:
    """MRSpecLAB's nodes with their parameters' defaults, and the Python and every package
    they run on, asked once per session."""
    folder = scratch("mrsuite_mrspeclab_")
    try:
        result = os.path.join(folder, "info.json")
        _driver({"task": "info", "result": result}, folder)
        with open(result) as fh:
            return json.load(fh)
    finally:
        shutil.rmtree(folder, ignore_errors=True)


#**********#
#   flaw   #
#**********#
def _flaw(node: str, params: Dict[str, Any]) -> Optional[str]:
    """What the node, run unchanged, gets wrong in this call; each verified at the pinned commit."""
    if node == "LineBroadening_Gaussian":
        return ("reads a parameter 'factor' it does not have, so it always fails with KeyError: "
                "'factor'; LineBroadening with apod_type='gaussian' broadens with a Gaussian")
    if node == "CoilCombinationSN2":
        return ("takes 'Noise proportion' of the number of coils, not of the points, as its "
                "noise window: the last 8 points of each FID for 32 coils at the default 0.25, "
                "and the whole FID for fewer than 4")
    if node == "FreqPhaseAlignment":
        ignored = [k for k in ("alignFreq", "alignPhase", "median") if params.get(k) == "False"]
        if ignored:
            return (f"tests {ignored} for truth, and the string 'False' is true, so it is "
                    f"ignored; the bool False is honoured")
    return None


#****************#
#   transients   #
#****************#
def _transients(spectrum: NIFTI_MRS) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    One voxel as (transients, coils, points), with what the driver's MRSData need to know.

    DIM_DYN becomes MRSpecLAB's list of transients and DIM_COIL the first axis of each, as
    its own loaders hold them; the echo and repetition times in ms, as suspect states them.
    The samples are the ones the file stores, which suspect reads the right way round:
    nifti-mrs hands them out conjugated, as FSL-MRS reads them, and suspect would put
    2.01 ppm at 7.29 (MRSpecLAB's own NIfTI loader reads the file and gets it right).
    """
    tags = [t for t in spectrum.dim_tags if t]
    other = sorted(set(tags) - {"DIM_DYN", "DIM_COIL"})
    if spectrum.shape[:3] != (1, 1, 1) or other:
        raise ValueError(f"MRSpecLAB's nodes take one voxel's transients and coils, not shape "
                         f"{spectrum.shape} with {other or 'several voxels'}")
    fids = np.conj(spectrum[:]).reshape(spectrum.shape)[0, 0, 0]
    for tag in ("DIM_DYN", "DIM_COIL"):
        if tag not in tags:
            fids, tags = fids[..., None], tags + [tag]
    fids = np.transpose(fids, (1 + tags.index("DIM_DYN"), 1 + tags.index("DIM_COIL"), 0))

    def ms(field):
        return spectrum.hdr_ext[field] * 1e3 if field in spectrum.hdr_ext else None
    return fids, {"dt": spectrum.dwelltime, "f0": spectrum.spectrometer_frequency[0],
                  "ppm0": spectrum.SpecFreqChemShift, "te": ms("EchoTime"),
                  "tr": ms("RepetitionTime")}


#**************#
#   spectrum   #
#**************#
def _spectrum(fids: np.ndarray, source: NIFTI_MRS) -> NIFTI_MRS:
    """
    (transients, coils, points) back as NIfTI-MRS, on the header of the spectrum it came from,
    conjugated back to nifti-mrs' orientation.

    A dimension a node collapsed to one (averaging, coil combination) is squeezed away; one
    that was a single entry already stays, so a round trip gives back what went in. A
    dimension kept at its length keeps its info and header.
    """
    axis, length = {"DIM_DYN": 0, "DIM_COIL": 1}, dict(zip(source.dim_tags, source.shape[4:]))
    tags = [t for t in source.dim_tags if t and (fids.shape[axis[t]] > 1 or length[t] == 1)]
    fids = np.transpose(fids, [2] + [axis[t] for t in tags]
                        + [a for t, a in axis.items() if t not in tags])
    out = gen_nifti_mrs(np.conj(fids).reshape((1, 1, 1) + fids.shape[:1 + len(tags)]),
                        source.dwelltime, source.spectrometer_frequency[0],
                        nucleus=source.nucleus[0], dim_tags=(tags + [None] * 3)[:3],
                        no_conj=False)                  # untouched; see fid_a.fida2nifti
    keep_header(source, out)
    header = source.hdr_ext.to_dict()
    for k, tag in enumerate(tags):
        i = source.dim_tags.index(tag)
        if out.shape[4 + k] == source.shape[4 + i]:
            out.set_dim_tag(4 + k, tag, info=header.get(f"dim_{5 + i}_info"),
                            header=header.get(f"dim_{5 + i}_header"))
    return out


#*************#
#   process   #
#*************#
def _process(node: Optional[str], data: NIfTI_MRS_Plus, wref: Optional[NIfTI_MRS_Plus],
             params: Dict[str, Any]) -> List[Any]:
    """
    A batch through one node, as one driver job; no node converts there and back only.

    Returns:
        List[Any]: Per spectrum what the node made of it, and with a water reference the
        pair (spectrum, reference). What the node left untouched comes back as it went in,
        with a warning, so no step is recorded on it.
    """
    items = data.list()
    references = wref.list() if wref is not None else [None] * len(items)
    folder = scratch("mrsuite_mrspeclab_")
    try:
        arrays, jobs = {}, []
        for i, (item, reference) in enumerate(zip(items, references)):
            arrays[f"data{i}"], meta = _transients(item)
            jobs.append({"data": meta, "wref": None})
            if reference is not None:
                arrays[f"wref{i}"], jobs[-1]["wref"] = _transients(reference)
        inputs, result = os.path.join(folder, "inputs.npz"), os.path.join(folder, "result.npz")
        np.savez(inputs, **arrays)
        _driver({"task": "process", "node": node, "parameters": params, "items": jobs,
                 "inputs": inputs, "result": result}, folder)
        with np.load(result) as r:
            outs = []
            for i, (item, reference) in enumerate(zip(items, references)):
                out = _untouched(node, r, f"untouched{i}", "spectrum", item) or \
                    _spectrum(r[f"data{i}"], item)
                if reference is not None:
                    out = (out, _untouched(node, r, f"wref_untouched{i}", "water reference",
                                           reference) or _spectrum(r[f"wref{i}"], reference))
                outs.append(out)
            return outs
    finally:
        shutil.rmtree(folder, ignore_errors=True)


#***************#
#   untouched   #
#***************#
def _untouched(node: Optional[str], result: Any, key: str, what: str,
               source: NIFTI_MRS) -> Optional[NIFTI_MRS]:
    """The input itself, with a warning, where the node handed it back as it was (a step
    that skipped, as EddyCurrentCorrection without a water reference); else None."""
    if node is None or not result[key]:
        return None
    warnings.warn(f"MRSpecLAB's {node} left the {what} as it was")
    return source


#**************************************************************************************************#
#                                      Class MRSpecLABBridge                                       #
#**************************************************************************************************#
#                                                                                                  #
# Dispatch onto MRSpecLAB's processing nodes, a batch as one job.                                  #
#                                                                                                  #
#**************************************************************************************************#
class MRSpecLABBridge(Bridge):
    """Dispatch onto MRSpecLAB's processing nodes, a batch as one job.

    ``mrspeclab.FreqPhaseAlignment(data, wref=None, **params)``: the spectra, a water
    reference per spectrum if the node uses one, and the node's parameters by their idnames,
    any not given at MRSpecLAB's defaults; an idname with spaces goes as
    ``**{"Block length": "8"}``. With a reference it returns (spectra, references), each
    reference as the node left it, which a pipeline files under the node's name.
    """

    name = "mrspeclab"
    program = "MRSpecLAB"
    batched = True

    #*************#
    #   resolve   #
    #*************#
    def _resolve(self, func: str) -> Callable:
        nodes = info()["nodes"]
        if func not in nodes:
            raise AttributeError(f"MRSpecLAB has no node '{func}'. "
                                 f"Available: {', '.join(sorted(nodes))}")
        node = nodes[func]

        def run(data, wref=None, **params):
            unknown = sorted(set(params) - set(node["parameters"]))
            if unknown:
                raise TypeError(f"MRSpecLAB's {func} has no parameter {unknown}; its "
                                f"parameters and their defaults: {node['parameters']}")
            flaw = _flaw(func, params)
            if flaw:
                warnings.warn(f"MRSpecLAB's {func} {flaw}")
            return _process(func, data, wref, params)

        run.__signature__, run.__doc__ = _about(node)
        return run

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
        return sorted(info()["nodes"])

    #*************#
    #   version   #
    #*************#
    def version(self) -> Optional[str]:
        """The MRSpecLAB checkout's exact state, e.g. 'v.1.0.0-8-g6e71664'."""
        return git_version(str(externals.ensure("mrspeclab")))

    #*************#
    #   runtime   #
    #*************#
    def runtime(self) -> Dict[str, Any]:
        """The Python and the packages the nodes ran on."""
        return {"python": info()["python"], "packages": info()["packages"]}


bridge = MRSpecLABBridge()


#***********#
#   kinds   #
#***********#
KINDS = {"IntegerProp": int, "TransientsProp": int, "FloatProp": float, "StringProp": str,
         "VectorProp": tuple}


#***********#
#   about   #
#***********#
def _about(node: Dict[str, Any]) -> Tuple[inspect.Signature, str]:
    """A node's arguments as the GUI states them: the spectra, their water reference, and
    each parameter by its idname, with its default, its kind and its choices; and its help,
    with each parameter's label and range. An idname Python cannot spell, such as 'Block
    length', still goes by name through the keyword arguments."""
    parameters = [inspect.Parameter("data", inspect.Parameter.POSITIONAL_ONLY),
                  inspect.Parameter("wref", inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                    default=None)]
    lines = [f"{node['label']}: {node['description']} (MRSpecLAB node by {node['author']}).",
             "", "Parameters:"]
    for name, default in node["parameters"].items():
        prop = node.get("properties", {}).get(name, {})
        kind = Literal[tuple(prop["choices"])] if prop.get("choices") else \
            KINDS.get(prop.get("kind"), inspect.Parameter.empty)
        if name.isidentifier():
            parameters.append(inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY,
                                                default=default, annotation=kind))
        says = [f"default {default!r}"]
        if prop.get("choices"):
            says.append(f"one of {prop['choices']}")
        if prop.get("min") is not None and prop.get("max") is not None:
            says.append(f"from {prop['min']} to {prop['max']}")
        lines.append(f"    {name}: {prop.get('label') or name}; {', '.join(says)}")
    if len(parameters) - 2 < len(node["parameters"]):
        parameters.append(inspect.Parameter("params", inspect.Parameter.VAR_KEYWORD))
    return inspect.Signature(parameters), "\n".join(lines)


#*************#
#   getattr   #
#*************#
def __getattr__(name: str) -> Any:
    """Module-level dispatch, so `from mrsuite.bridges.mrspeclab import Average` works."""
    return getattr(bridge, name)


#*********#
#   dir   #
#*********#
def __dir__() -> List[str]:
    return sorted(set(globals()) | set(bridge.available()))
