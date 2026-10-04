####################################################################################################
#                                             score.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-02                                                                              #
#                                                                                                  #
# Purpose: Scores a run folder (run.py) on mrsuite.benchmark, dataset by dataset, on the           #
#          references and metrics of the config, and writes the tables into the folder.            #
#                                                                                                  #
#          python scripts/benchmark/score.py data/runs/<name>                                      #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from mrsuite.benchmark import observations, score
from mrsuite.core.result import FitResult

import run


#**********#
#   main   #
#**********#
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", help="a run folder")
    parser.add_argument("--config", default=str(run.HERE / "config.py"))
    args = parser.parse_args(argv)

    config = run.load_config(args.config)
    for name, table in score_run(args.folder, config.REFERENCES, config.METRICS).items():
        pooled = table[table.metabolite == "all"]
        print(f"\n{name}\n" + pooled[["method", "reference", "metric", "value", "rho_c",
                                       "n_used", "n_omitted"]].to_string(index=False))
    return 0


#***************#
#   score run   #
#***************#
def score_run(folder: str, references: Dict[str, Sequence[str]],
              metrics: Dict[str, Callable[..., Any]]) -> Dict[str, pd.DataFrame]:
    """
    Every finished job of a run, scored dataset by dataset; written to ``scores/`` in the
    run folder, one table per dataset, with MRSValidate's own run beside them.

    A method fitted on a subsample of a dataset (an evenly spread few, for a slow fitter)
    is scored on its spectra together with every method that has them, as '<dataset>-<n>'.
    A method is scored on a reference where it has the scaling the reference reads, and
    every method of a table on the same metabolites. A spectrum whose fit failed counts
    as omitted, as a fit that failed on a metabolite does.

    Args:
        folder (str): The run folder.
        references (Dict[str, Sequence[str]]): Dataset to the references to score on.
        metrics (Dict[str, Callable[..., Any]]): As mrsuite.benchmark.score takes them.

    Returns:
        Dict[str, pd.DataFrame]: Table name to mrsuite.benchmark.score's table, all
        references in one.
    """
    folder = Path(folder)
    out = folder / "scores"
    out.mkdir(exist_ok=True)
    tables = {}
    for data in sorted(p.parent for p in folder.glob("*/dataset.json")):
        record = json.loads((data / "dataset.json").read_text())
        truth = dict(zip(record["ids"], record["truth"]))
        jobs = {}
        for job in sorted(p.parent for p in data.glob("*/job.json")):
            planned = json.loads((job / "job.json").read_text())["spectra"]
            results, errors = run.read(job)
            if set(planned) <= set(results) | set(errors):
                jobs[job.name] = (planned, results)
            else:
                print(f"{data.name}/{job.name}: unfinished, not scored", flush=True)

        for ids in sorted({tuple(p) for p, _ in jobs.values()}, key=len, reverse=True):
            methods = {name: [results.get(i) for i in ids]
                       for name, (planned, results) in jobs.items() if set(ids) <= set(planned)}
            name = data.name if len(ids) == len(record["ids"]) else f"{data.name}-{len(ids)}"
            tables[name] = _score(methods, [truth[i] for i in ids], list(ids),
                                  references[data.name], metrics, out / "mrsvalidate")
            tables[name].to_csv(out / f"{name}.csv", index=False)
    return tables


#***********#
#   score   #
#***********#
def _score(methods: Dict[str, List[Optional[FitResult]]], truth: List[Dict[str, float]],
           ids: List[str], references: Sequence[str], metrics: Dict[str, Callable[..., Any]],
           output: Path) -> pd.DataFrame:
    """One table: each reference with the methods that have its scaling, on the
    metabolites all of them have; a failed fit (None) as a fit of NaNs."""
    def reads(reference):
        return "molarity" if reference == "water" else "raw"

    def fitted(results):
        return [(r, t) for r, t in zip(results, truth) if r is not None]

    on = {ref: {m: rs for m, rs in methods.items()
                if fitted(rs) and all(reads(ref) in r.concentrations for r, _ in fitted(rs))}
          for ref in references}
    shared = set.intersection(*(
        set(observations(*zip(*fitted(rs)), "water" if ref == "water" else "none")[0]
            .parameters.names)
        for ref in references for rs in on[ref].values()))

    tables = []
    for reference in references:
        if on[reference]:
            tables.append(score({m: _failed_as_nan(rs) for m, rs in on[reference].items()},
                                truth, [reference], metrics, output,
                                metabolites=sorted(shared), ids=ids))
    return pd.concat(tables, ignore_index=True)


#*******************#
#   failed as nan   #
#*******************#
def _failed_as_nan(results: List[Optional[FitResult]]) -> List[FitResult]:
    """A method's results with each failed fit as one of NaNs, shaped as its others."""
    shape = next(r for r in results if r is not None)
    nan = FitResult(shape.method, list(shape.metabolites),
                    {s: dict.fromkeys(c, np.nan) for s, c in shape.concentrations.items()})
    return [nan if r is None else r for r in results]


if __name__ == "__main__":
    sys.exit(main())
