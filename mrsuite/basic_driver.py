####################################################################################################
#                                         basic_driver.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-09                                                                              #
#                                                                                                  #
# Purpose: Reads and writes basis sets with BasiC, for mrsuite.io. Not imported by MRSuite: BasiC  #
#          is a folder of scripts importing each other as top-level modules (core, readers,        #
#          writers), so it runs in a process of its own. FIDs come and go as .npz; what BasiC      #
#          prints goes to stderr; the job's answer goes to stdout, as JSON; a failure ends it      #
#          with one line saying why.                                                               #
#                                                                                                  #
#          Usage: python basic_driver.py '<job as JSON>'                                           #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import contextlib
import json
import sys

import numpy as np


#**********#
#   fail   #
#**********#
def fail(message):
    sys.exit(" ".join(str(message).split()))


#**********#
#   read   #
#**********#
def read(job):
    """The files in the format given or detected, with the parameters given where they state
    none, as one array of FIDs (to ``result``) and what is said of them; the echo time and
    sequence as the basis functions hold them (a reader gives 0 for an echo time not
    stated), else as given."""
    from core.detect import detect_format
    from core.load import fill_missing, load_basis

    found = detect_format(job["files"])
    fmt = job["format"]
    if fmt == "auto":
        if found["status"] == "ambiguous":
            fail(f"{found['message']} Choose one as format: "
                 + ", ".join(map(repr, found["ambiguous_options"])))
        if found["status"] == "failed":
            fail(found["message"])
        fmt = found["format"]
    stated = found["data"]
    given = {k: job[k] for k in ("sw", "sf", "te")}
    basis = load_basis(job["files"], fmt, sw=stated.get("sw") or given["sw"],
                       sf=stated.get("sf") or given["sf"])
    if not basis:
        fail("it found no basis functions in the files")
    fill_missing(basis, given)
    lacking = [k for k in ("sw", "sf") if not all(b.get(k) for b in basis)]
    if lacking:
        fail(f"the files do not state {' or '.join(lacking)}: give "
             + ", ".join(f"{k}=" for k in lacking))
    if len({len(np.ravel(b["fid"])) for b in basis}) > 1:
        fail("its basis functions differ in length")
    np.savez(job["result"], fids=np.array([np.ravel(b["fid"]) for b in basis]))
    first = basis[0]
    return {"names": [b["name"] for b in basis], "bandwidth": float(first["sw"]),
            "frequency": float(first["sf"]), "centre": first.get("centerFreq"),
            "te": first.get("te") or given["te"], "seq": first.get("seq"), "format": fmt}


#***********#
#   write   #
#***********#
def write(job):
    """The job's basis set (FIDs from ``fids``) written for a tool BasiC writes for; the
    files written."""
    from core.formats import ALL_TOOLS
    from core.write import write_basis

    tools = {t["id"]: [f["id"] for f in t["formats"]] for t in ALL_TOOLS if not t["dimmed"]}
    if job["tool"] not in tools:
        fail(f"no tool {job['tool']!r}; one of: {', '.join(tools)}")
    outputs = tools[job["tool"]]
    if job["output"] != "default" and job["output"] not in outputs:
        fail(f"{job['tool']} writes output 'default'"
             + "".join(f", {o!r}" for o in outputs) + f", not {job['output']!r}")
    fids = np.load(job["fids"])["fids"]
    basis = [{"fid": fid, "name": name, "n": len(fid), "sw": job["bandwidth"],
              "sf": job["frequency"], "centerFreq": job["centre"], "te": job["te"],
              "seq": job["seq"]} for name, fid in zip(job["names"], fids)]
    written = write_basis(basis, job["tool"], job["output"], job["folder"], te=job["te"],
                          seq=job["seq"])
    if not written:                        # its writers go on without what they fail
        fail("it wrote nothing")
    return written


if __name__ == "__main__":
    try:
        with contextlib.redirect_stdout(sys.stderr):
            job = json.loads(sys.argv[1])
            sys.path.insert(0, job["root"])
            answer = {"read": read, "write": write}[job["task"]](job)
    except Exception as err:               # noqa: BLE001 - said, in one line
        fail(f"{type(err).__name__}: {err}")
    print(json.dumps(answer))
