####################################################################################################
#                                              run.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-02                                                                              #
#                                                                                                  #
# Purpose: Runs a stage of the benchmark (config.py) into a run folder, saving each chunk of       #
#          fits as it finishes, so a crash loses at most one chunk and running again resumes       #
#          where it stopped. The folder's manifest records each session's environment: packages,   #
#          commits, runtimes and binaries, machine, the config itself.                             #
#                                                                                                  #
#          python scripts/benchmark/run.py data/runs/<name> --stage 2016 --workers 8               #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import os

os.environ["OMP_NUM_THREADS"] = "1"        # before numpy: one thread per process, workers scale

import argparse
import importlib
import json
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from nifti_mrs_plus import NIfTI_MRS_Plus
from mrsuite.core import provenance, result
from mrsuite.core.result import FitResult

HERE = Path(__file__).resolve().parent

# the repositories MRSuite runs from besides its own, by the package each provides
SIBLINGS = ("lcmodel_wrapper", "tarquin_wrapper", "nifti_mrs_plus", "mrsvalidate", "augmentrum")

# environment variables that change what runs, or how; never one holding a secret
ENVIRONMENT = ("MRSUITE_", "OMP_", "MKL_", "OPENBLAS_", "VECLIB_", "LCMODEL_", "TARQUIN_",
               "KERAS_", "HF_", "OCTAVE", "PYTHONHASHSEED")
SECRET = ("TOKEN", "KEY", "SECRET", "PASSWORD")


#**********#
#   main   #
#**********#
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", help="the run folder; an existing one is resumed")
    parser.add_argument("--stage", required=True, help="a stage of the config")
    parser.add_argument("--workers", type=int, default=1, help="processes per fit")
    parser.add_argument("--no-raw", action="store_true",
                        help="keep only the uniform results, not each tool's own object")
    parser.add_argument("--config", default=str(HERE / "config.py"))
    args = parser.parse_args(argv)

    config = load_config(args.config)
    failed = run(args.folder, config.STAGES[args.stage], config.DATASETS, config.METHODS,
                 workers=args.workers, raw=not args.no_raw, config=args.config,
                 stage=args.stage)
    return 1 if failed else 0


#*****************#
#   load config   #
#*****************#
def load_config(path: str) -> Any:
    """The config module, importable by its name, as worker processes import methods."""
    path = Path(path).resolve()
    sys.path.insert(0, str(path.parent))
    return importlib.import_module(path.stem)


#*********#
#   run   #
#*********#
def run(folder: str, jobs: Sequence[Any], datasets: Dict[str, Any], methods: Dict[str, Any],
        workers: int = 1, raw: bool = True, config: Optional[str] = None,
        stage: Optional[str] = None) -> Dict[str, str]:
    """
    Run each job, dataset by dataset, skipping the spectra it already has results for.

    A job that fails is recorded and the run goes on with the next; running again retries
    it. A spectrum that fails within a chunk is fitted again alone, and if it fails alone,
    its error is recorded in place of its result (and retried by the next session).

    Args:
        folder (str): The run folder, made or resumed.
        jobs (Sequence[Job]): What to run.
        datasets (Dict[str, Callable[[], Dataset]]): Name to loader.
        methods (Dict[str, Fit]): Name to method.
        workers (int): Processes per fit.
        raw (bool): Keep each tool's own result object, pickled beside the results.
        config (Optional[str]): The config file, recorded in the manifest.
        stage (Optional[str]): The stage's name, likewise.

    Returns:
        Dict[str, str]: 'dataset/method' to the traceback of each job that failed.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    used = {name: methods[name] for name in dict.fromkeys(job.method for job in jobs)}
    session = _begin(folder, {"stage": stage, "workers": workers, "raw": raw,
                              "jobs": [vars(job) for job in jobs],
                              "config": _config(config), "environment": environment(used)})
    failed = {}
    for name in dict.fromkeys(job.dataset for job in jobs):          # each dataset loaded once
        try:
            dataset = datasets[name]()
            _record_dataset(folder / name, dataset)
        except Exception:
            dataset, error = None, traceback.format_exc()
        for job in (j for j in jobs if j.dataset == name):
            try:
                if dataset is None:
                    raise RuntimeError(f"dataset {name} failed to load:\n{error}")
                _run_job(folder / name / job.method, methods[job.method], dataset, job, workers,
                         raw, session)
            except Exception:
                failed[f"{name}/{job.method}"] = traceback.format_exc()
                print(f"{name}/{job.method} FAILED\n{failed[f'{name}/{job.method}']}", flush=True)
    _end(folder, session, failed)
    return failed


#*************#
#   run job   #
#*************#
def _run_job(folder: Path, method: Any, dataset: Any, job: Any, workers: int, raw: bool,
             session: int) -> None:
    """One method on one dataset: its steps on the whole dataset, then the spectra it has no
    result for yet, a chunk at a time, each saved as it finishes. The spectra the steps give
    are part of the job's record, so a changed step (another model behind the same name) is
    not resumed into. A job with no result at all fails: it is misconfigured, not unlucky."""
    pick = (range(len(dataset.ids)) if job.spectra is None else
            np.unique(np.linspace(0, len(dataset.ids) - 1, job.spectra).astype(int)).tolist())
    data = dataset.data
    for step in method.before:
        data = step(data)
    spec = {"fit": _name(method.fit), "args": method.args(dataset), "options": method.options,
            "water": method.water, "before": [_name(step) for step in method.before],
            "input": provenance.checksum(data.numpy()) if method.before else None,
            "spectra": [dataset.ids[i] for i in pick]}
    _same_or_new(folder / "job.json", provenance.describe(spec, portable=True))

    done = {i for path in _chunks(folder) for i in _meta(path)["ids"]}
    todo = [i for i in pick if dataset.ids[i] not in done]
    print(f"{folder.parent.name}/{folder.name}: {len(pick) - len(todo)} of {len(pick)} done",
          flush=True)
    if not todo:
        return

    items = data.list()
    water = dataset.water.list() if method.water is not None else None
    extra = {"workers": workers} if getattr(method.fit, "parallel", False) else {}

    def fit(indices: List[int]) -> List[FitResult]:
        kwargs = dict(method.options)
        if water is not None:
            kwargs.update(method.water, water=NIfTI_MRS_Plus([water[i] for i in indices]))
        return method.fit(NIfTI_MRS_Plus([items[i] for i in indices]), *method.args(dataset),
                          **kwargs, **extra)

    size = method.chunk * workers
    chunks = _chunks(folder)
    number = int(chunks[-1].stem) + 1 if chunks else 0
    fitted = 0
    for start in range(0, len(todo), size):
        indices = todo[start:start + size]
        began, clock = datetime.now().isoformat(), time.perf_counter()
        results, kept, errors = _fit_chunk(fit, indices, dataset.ids)
        meta = {"ids": kept, "failed": errors, "start": began, "end": datetime.now().isoformat(),
                "seconds": time.perf_counter() - clock, "workers": workers, "session": session}
        stem = folder / f"{number:05d}"
        _dump(Path(f"{stem}.meta.json"), meta)         # the results, saved last, mark it done
        result.save(results, f"{stem}.json", raw=raw)
        print(f"  chunk {number}: {len(kept)} fitted, {len(errors)} failed, "
              f"{meta['seconds']:.1f} s", flush=True)
        number += 1
        fitted += len(kept)
    if not done and not fitted:
        raise RuntimeError(f"every spectrum failed, e.g. {next(iter(meta['failed'].values()))}")


#***************#
#   fit chunk   #
#***************#
def _fit_chunk(fit: Any, indices: List[int], ids: List[str]
               ) -> Tuple[List[FitResult], List[str], Dict[str, str]]:
    """A chunk's fits, and where the chunk fails, each spectrum alone, keeping the errors."""
    try:
        return fit(indices), [ids[i] for i in indices], {}
    except Exception:
        results, kept, errors = [], [], {}
        for i in indices:
            try:
                results += fit([i])
                kept.append(ids[i])
            except Exception as err:
                errors[ids[i]] = f"{type(err).__name__}: {err}"
        return results, kept, errors


#**********#
#   read   #
#**********#
def read(folder: Path, raw: bool = False) -> Tuple[Dict[str, FitResult], Dict[str, str]]:
    """
    A job's results so far, by spectrum ID, and the errors of the spectra that failed and
    have no result since.

    Args:
        folder (Path): The job's folder in a run folder.
        raw (bool): Read each tool's own object too, where it was kept.

    Returns:
        Tuple[Dict[str, FitResult], Dict[str, str]]: The results, then the errors.
    """
    results, errors = {}, {}
    for path in _chunks(folder):
        meta = _meta(path)
        results.update(zip(meta["ids"], result.load(str(path), raw=raw)))
        errors.update(meta["failed"])
    return results, {k: v for k, v in errors.items() if k not in results}


#************#
#   chunks   #
#************#
def _chunks(folder: Path) -> List[Path]:
    """A job's finished chunks, in order."""
    return sorted(Path(folder).glob("[0-9][0-9][0-9][0-9][0-9].json"))


#**********#
#   meta   #
#**********#
def _meta(chunk: Path) -> Dict[str, Any]:
    """A finished chunk's record: its spectra, those that failed, when and how it ran."""
    return json.loads(chunk.with_name(f"{chunk.stem}.meta.json").read_text())


#*****************#
#   environment   #
#*****************#
def environment(methods: Dict[str, Any]) -> Dict[str, Any]:
    """
    What a run's results depend on besides the config: packages, commits, the methods'
    own versions and runtimes (interpreters, binaries by SHA-256), machine and variables.

    Args:
        methods (Dict[str, Fit]): The methods that will run, by name.

    Returns:
        Dict[str, Any]: JSON-ready.
    """
    def pip(*args):
        return subprocess.run([sys.executable, "-m", "pip", *args], capture_output=True,
                              text=True).stdout.splitlines()
    submodules = subprocess.run(["git", "-C", str(provenance.REPO), "config", "--file",
                                 ".gitmodules", "--get-regexp", "path"],
                                capture_output=True, text=True).stdout.split()[1::2]
    fits = {_name(m.fit): m.fit for m in methods.values()}
    return {
        # pip freeze gives a conda package (torch, fsl_mrs) by its build path alone
        "python": sys.version, "pip_freeze": pip("freeze"),
        "versions": pip("list", "--format=freeze"),
        "git": {"MRSuite": _git(provenance.REPO),
                **{path: _git(provenance.REPO / path) for path in submodules},
                **{name: _git(Path(importlib.import_module(name).__path__[0]))
                   for name in SIBLINGS}},
        "methods": {name: provenance.describe({"version": fit.version(), **fit.runtime()},
                                              portable=True)
                    for name, fit in fits.items()},
        "machine": {**{k: v for k, v in platform.uname()._asdict().items() if k != "node"},
                    "cpu": _cpu(), "cpus": os.cpu_count()},
        "variables": {k: v for k, v in sorted(os.environ.items())
                      if k.startswith(ENVIRONMENT) and not any(w in k for w in SECRET)},
    }


#*********#
#   git   #
#*********#
def _git(path: Path) -> Optional[Dict[str, Any]]:
    """A checkout's commit, and whether it has changes the commit does not hold; None for
    what is no checkout, such as a package installed into site-packages (whose enclosing
    repository, if any, is not the package's)."""
    def git(*args):
        return subprocess.run(["git", "-C", str(path), *args], capture_output=True,
                              text=True).stdout.strip()
    commit = git("rev-parse", "HEAD")
    if not commit or "site-packages" in Path(path).parts:
        return None
    return {"commit": commit, "describe": git("describe", "--tags", "--always"),
            "dirty": bool(git("status", "--porcelain"))}


#*********#
#   cpu   #
#*********#
def _cpu() -> str:
    """The processor's model name, which platform does not give."""
    if sys.platform == "darwin":
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                              text=True).stdout.strip()
    lines = Path("/proc/cpuinfo").read_text().splitlines() if Path("/proc/cpuinfo").exists() else []
    return next((line.split(":", 1)[1].strip() for line in lines if line.startswith("model name")),
                platform.processor())


#************#
#   config   #
#************#
def _config(path: Optional[str]) -> Optional[Dict[str, str]]:
    """The config file as it was: its text and SHA-256."""
    if path is None:
        return None
    return {**provenance.portable_file(path), "text": Path(path).read_text()}


#**********#
#   name   #
#**********#
def _name(fn: Any) -> str:
    """A function or method by where it is imported from."""
    return f"{fn.__module__}.{getattr(fn, '__name__', type(fn).__qualname__)}"


#********************#
#   record dataset   #
#********************#
def _record_dataset(folder: Path, dataset: Any) -> None:
    """The dataset's IDs, checksums of its spectra and water references, and its truth,
    which scoring reads."""
    water = None if dataset.water is None else provenance.checksum(dataset.water.numpy())
    _same_or_new(folder / "dataset.json", {"ids": dataset.ids,
                                           "sha256": provenance.checksum(dataset.data.numpy()),
                                           "water": water, "truth": dataset.truth})


#*****************#
#   same or new   #
#*****************#
def _same_or_new(path: Path, record: Dict[str, Any]) -> None:
    """Write a record, or check that the one there is the same: results made from something
    else must not be resumed into."""
    if path.exists():
        if path.read_text() != json.dumps(record, indent=1):     # as _dump wrote it
            raise ValueError(f"{path} differs from what this run would write; the dataset or "
                             f"method changed since. Use a new run folder, or remove "
                             f"{path.parent} to start it again.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    _dump(path, record)


#***********#
#   begin   #
#***********#
def _begin(folder: Path, record: Dict[str, Any]) -> int:
    """Add a session to the manifest; its number."""
    path = folder / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {"sessions": []}
    manifest["sessions"].append({"start": datetime.now().isoformat(), **record})
    _dump(path, manifest)
    return len(manifest["sessions"]) - 1


#*********#
#   end   #
#*********#
def _end(folder: Path, session: int, failed: Dict[str, str]) -> None:
    """Close a session in the manifest."""
    path = folder / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["sessions"][session].update(end=datetime.now().isoformat(), failed=failed)
    _dump(path, manifest)


#**********#
#   dump   #
#**********#
def _dump(path: Path, record: Any) -> None:
    """JSON written whole or not at all."""
    temporary = Path(f"{path}.tmp")
    temporary.write_text(json.dumps(record, indent=1))
    os.replace(temporary, path)


if __name__ == "__main__":
    sys.exit(main())
