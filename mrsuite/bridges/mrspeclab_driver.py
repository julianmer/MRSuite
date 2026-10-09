####################################################################################################
#                                       mrspeclab_driver.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-02                                                                              #
#                                                                                                  #
# Purpose: Runs MRSpecLAB's processing nodes for MRSuite. Not imported by MRSuite: it runs in a    #
#          process of its own, in MRSpecLAB's environment, and is driven by                        #
#          mrsuite.bridges.mrspeclab. Spectra come in as arrays, become the list of suspect        #
#          MRSData MRSpecLAB's pipeline hands its nodes, go through one node unchanged, and go     #
#          back as arrays.                                                                         #
#                                                                                                  #
#          Usage: python mrspeclab_driver.py job.json                                              #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import importlib
import json
import os
import sys
from importlib import metadata

import numpy as np


#***********#
#   nodes   #
#***********#
def nodes(root):
    """Every node MRSpecLAB's GUI offers: each module in nodes/ but the private ones, which
    registers its node as it is imported."""
    sys.path.insert(0, root)
    from processing.api.registry import NODE_REGISTRY
    for name in sorted(os.listdir(os.path.join(root, "nodes"))):
        if name.endswith(".py") and not name.startswith("_"):
            importlib.import_module(f"nodes.{name[:-3]}")
    return NODE_REGISTRY


#**********#
#   node   #
#**********#
def node(registry, name, parameters):
    """A node as the GUI holds it, without the GUI: built with no node graph, which skips
    the wx part of its set-up, and its properties keyed by idname as that set-up keys them."""
    built = registry[name](None, 0)
    built.properties = {p.idname: p for p in built.parameters}
    for idname, value in parameters.items():
        built.properties[idname].value = value
    return built


#****************#
#   transients   #
#****************#
def transients(fids, meta):
    """(transients, coils, points) as MRSpecLAB's pipeline holds it: a list of MRSData, one
    per transient, (coils, points) before coil combination and (points,) after. A time the
    spectrum does not state is NaN, so a node using it fails visibly rather than taking
    suspect's default (30 ms)."""
    from suspect import MRSData

    times = {k: float("nan") if meta[k] is None else meta[k] for k in ("te", "tr")}
    return [MRSData(f if len(f) > 1 else f[0], meta["dt"], meta["f0"], ppm0=meta["ppm0"],
                    metadata={"ave_per_rep": 1}, **times) for f in fids]


#*************#
#   stacked   #
#*************#
def stacked(spectra):
    """MRSpecLAB's list back as (transients, coils, points), without the transients a node
    removed: RemoveBadAverages leaves None in their place."""
    fids = np.asarray([s for s in spectra if s is not None])
    if not len(fids):
        raise ValueError("the node left no transient")
    return fids[:, None] if fids.ndim == 2 else fids


#**********#
#   info   #
#**********#
def info(job):
    """The nodes with their parameters' defaults and what the GUI states of each - its kind,
    label, choices and range - and what they run on, as JSON."""
    registry = nodes(job["root"])
    out = {}
    for name, cls in registry.items():
        built = cls(None, 0)
        meta = built.meta_info
        out[name] = {"label": meta["label"], "author": meta["author"],
                     "description": meta["description"],
                     "parameters": {p.idname: p.value for p in built.parameters},
                     "properties": {p.idname: {"kind": type(p).__name__, "label": p.label,
                                               "choices": getattr(p, "choices", None),
                                               "min": getattr(p, "min_value", None),
                                               "max": getattr(p, "max_value", None)}
                                    for p in built.parameters}}
    with open(job["result"], "w") as fh:
        json.dump({"nodes": out, "python": sys.version.split()[0],
                   "packages": dict(sorted((d.metadata["Name"], d.version)
                                           for d in metadata.distributions()))}, fh)


#*************#
#   process   #
#*************#
def process(job):
    """Each spectrum through the node, its water reference beside it as MRSpecLAB's pipeline
    carries it: what the node made of it, else as it was, and which of the two the node
    left untouched. No node converts only."""
    registry = nodes(job["root"]) if job["node"] else None
    inputs = np.load(job["inputs"])
    result = {}
    for i, item in enumerate(job["items"]):
        data = {"input": transients(inputs[f"data{i}"], item["data"]), "wref": None,
                "output": [], "wref_output": [], "header": {}}
        if item["wref"]:
            data["wref"] = transients(inputs[f"wref{i}"], item["wref"])
        if job["node"]:
            node(registry, job["node"], job["parameters"]).process(data)
        else:
            data["output"] = data["input"]
        result[f"data{i}"] = stacked(data["output"])
        result[f"untouched{i}"] = data["output"] is data["input"]
        if item["wref"]:
            result[f"wref{i}"] = stacked(data["wref_output"] or data["wref"])
            result[f"wref_untouched{i}"] = not data["wref_output"]
    np.savez(job["result"], **result)


if __name__ == "__main__":
    with open(sys.argv[1]) as fh:
        job = json.load(fh)
    {"info": info, "process": process}[job["task"]](job)
