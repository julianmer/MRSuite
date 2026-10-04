####################################################################################################
#                                            compare.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-02                                                                              #
#                                                                                                  #
# Purpose: Compares two run folders (run.py) field by field, job by job: whether every value is    #
#          bit-identical, and if not, how far apart, in units in the last place and relatively.    #
#          The check of the efficiency step: golden outputs before, the same run after.            #
#                                                                                                  #
#          python scripts/benchmark/compare.py data/runs/<before> data/runs/<after>                #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import argparse
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import pandas as pd

from mrsuite.core.result import FitResult

import run


#**********#
#   main   #
#**********#
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a", help="a run folder")
    parser.add_argument("b", help="another")
    args = parser.parse_args(argv)

    table = compare(args.a, args.b)
    print(table.to_string(index=False))
    return 0 if table.identical.all() else 1


#*************#
#   compare   #
#*************#
def compare(a: str, b: str) -> pd.DataFrame:
    """
    Two runs, job by job, on the spectra both fitted.

    Values are the concentrations, uncertainties and QC; the method's name, the
    metabolites and which values a result has must match too. Provenance is compared
    without its times, apart from the values, as it also records each run's own commit.

    Args:
        a (str): A run folder.
        b (str): Another.

    Returns:
        pd.DataFrame: One row per job ('dataset/method'): spectra only in a, only in b
        and in both; values compared and how many differ; the largest distance in ulp and
        the largest relative difference; 'identical' when every value is bit-identical
        and the results have the same shape; whether the provenance is the same.
    """
    jobs = sorted({str(p.parent.relative_to(folder)) for folder in (Path(a), Path(b))
                   for p in folder.glob("*/*/job.json")})
    rows = []
    for job in jobs:
        ra, _ = run.read(Path(a) / job)
        rb, _ = run.read(Path(b) / job)
        both = sorted(set(ra) & set(rb))
        same_shape, same_provenance, ulps, relatives = True, True, [], []
        for i in both:
            x, y = _values(ra[i]), _values(rb[i])
            same_shape &= (x.keys() == y.keys() and ra[i].method == rb[i].method
                           and ra[i].metabolites == rb[i].metabolites)
            same_provenance &= _timeless(ra[i].provenance) == _timeless(rb[i].provenance)
            for key in x.keys() & y.keys():
                ulps.append(_ulp(x[key], y[key]))
                relatives.append(_relative(x[key], y[key]))
        differ = sum(u != 0 for u in ulps)
        rows.append({"job": job, "only_a": len(set(ra) - set(rb)), "only_b": len(set(rb) - set(ra)),
                     "both": len(both), "values": len(ulps), "differ": differ,
                     "max_ulp": max(ulps, default=0), "max_relative": max(relatives, default=0.0),
                     "identical": same_shape and differ == 0 and len(ra) == len(rb) == len(both),
                     "same_provenance": same_provenance})
    return pd.DataFrame(rows)


#************#
#   values   #
#************#
def _values(res: FitResult) -> Dict[tuple, float]:
    """Every number of a result, by where it is."""
    out = {("qc", k): v for k, v in res.qc.items()}
    for name in ("concentrations", "uncertainties"):
        out.update({(name, kind, m): v for kind, values in getattr(res, name).items()
                    for m, v in values.items()})
    return out


#*********#
#   ulp   #
#*********#
def _ulp(x: float, y: float) -> float:
    """How many doubles apart: 0 only when bit-identical, inf for a NaN against a number."""
    def ordinal(v):
        bits = struct.unpack("<q", struct.pack("<d", v))[0]
        return bits if bits >= 0 else -(bits & 0x7FFFFFFFFFFFFFFF) - 1   # -0.0 just below 0.0
    if math.isnan(x) or math.isnan(y):
        return 0 if struct.pack("<d", x) == struct.pack("<d", y) else math.inf
    return abs(ordinal(x) - ordinal(y))


#**************#
#   relative   #
#**************#
def _relative(x: float, y: float) -> float:
    """|x - y| relative to the larger magnitude; 0 for equal values, inf for NaN or inf
    against anything else."""
    if x == y or (math.isnan(x) and math.isnan(y)):
        return 0.0
    if not (math.isfinite(x) and math.isfinite(y)):
        return math.inf
    return abs(x - y) / max(abs(x), abs(y))


#**************#
#   timeless   #
#**************#
def _timeless(value: Any) -> Any:
    """Provenance without its times, those in the JSON of each entry's Details included."""
    if isinstance(value, dict):
        return {k: _timeless(_parsed(v) if k == "Details" else v)
                for k, v in value.items() if k != "Time"}
    if isinstance(value, list):
        return [_timeless(v) for v in value]
    return value


#************#
#   parsed   #
#************#
def _parsed(text: Any) -> Any:
    """Details as the JSON MRSuite writes it, or as it is where another program wrote it."""
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return text


if __name__ == "__main__":
    sys.exit(main())
