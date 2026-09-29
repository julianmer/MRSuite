####################################################################################################
#                                           reference.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-27                                                                              #
#                                                                                                  #
# Purpose: Runs a model's original code, for converting its checkpoints and for checking the       #
#          wrapper against it. Not imported by MRSuite: it runs in the original's own environment, #
#          with the original repository as its working directory, and is driven by                 #
#          mrsuite.models.convert.                                                                 #
#                                                                                                  #
#          Usage: python reference.py job.json                                                     #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import json
import os
import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, os.getcwd())


#***************#
#   framework   #
#***************#
def framework(job):
    """The original framework, built as its test scripts build it."""
    if job["model"] == "mlp":
        from frameworks.frameworkNN import FrameworkNN
        return FrameworkNN(job["basis"], basisFmt=job.get("basisFmt", ""), specType="auto",
                           ppmlim=tuple(job["ppmlim"]), dataType=job.get("dataType", "none"),
                           arch="mlp", width=job["width"], depth=job["depth"], dropout=0.0)

    from frameworks.frameworkWAND import FrameworkWAND
    return FrameworkWAND(job["basis"], basisFmt=job.get("basisFmt", ""),
                         specType=job["specType"], ppmlim=job.get("ppmlim"), model=job["arch"],
                         dataType=job.get("dataType", "none"))


#************#
#   config   #
#************#
def config(model):
    """What the wrapper is built from: the names, the window and the acquisition."""
    basis = model.basisObj
    # WAND appends its extra components to the basis's own list of names
    return {"metabolites": list(basis.names)[:basis.fids.shape[1]],
            "first": int(model.first), "last": int(model.last),
            "acquisition": {"points": int(basis.n), "bandwidth": float(basis.bw),
                            "frequency": float(basis.cf), "centre": 4.65}}


#*************#
#   forward   #
#*************#
def forward(model, job):
    """The original's output on the given spectra, with the given weights, in eval mode."""
    model.net.load_state_dict(torch.load(job["weights"]))
    model.eval()
    x = torch.from_numpy(np.load(job["input"]))
    with torch.no_grad():
        if job["model"] == "mlp":
            out = model.forward(x)
        else:
            out = model.forward(model.prepareData(x.clone(), None, None)[0])[0]
    np.save(job["output"], out.numpy())


#**********#
#   loss   #
#**********#
def loss(model, job):
    """The original's loss on the given spectra and parameters, with the basis and axes
    it used, so the wrapper's re-expression is held to the same numbers."""
    x = torch.from_numpy(np.load(job["input"]))
    theta = torch.from_numpy(np.load(job["theta"]))
    err = model.loss(x, None, None, theta, type=job["loss"])
    basis = model.basisObj
    np.savez(job["output"], loss=err.numpy(), basis=model.basis, t=basis.t, f=basis.f,
             first=model.first, last=model.last,
             names=np.array([str(n) for n in list(basis.names)[:model.basis.shape[1]]]))


#**********#
#   main   #
#**********#
if __name__ == "__main__":
    with open(sys.argv[1]) as fh:
        job = json.load(fh)
    model = framework(job)
    if job["task"] == "config":
        result = config(model)
    elif job["task"] == "loss":
        loss(model, job)
        result = {"ok": True}
    else:
        forward(model, job)
        result = {"ok": True}
    with open(job["result"], "w") as fh:
        json.dump(result, fh)
