####################################################################################################
#                                              io.py                                               #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-02                                                                              #
#                                                                                                  #
# Purpose: Reads raw scanner data into NIfTI-MRS+ through spec2nii, the reference converter to     #
#          NIfTI-MRS, so data from any vendor reaches a pipeline the same way; NIfTI-MRS files     #
#          themselves, each spectrum's history starting with the file it came from; ground truth   #
#          from the files it comes in; and basis sets in any format through BasiC, as one Basis    #
#          every fit takes.                                                                        #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import json
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import nibabel as nib
import numpy as np
from nifti_mrs.nifti_mrs import NIFTI_MRS, NotNIFTI_MRS
from nifti_mrs_plus import NIfTI_MRS_Plus

from mrsuite import externals
from mrsuite.core import provenance
from mrsuite.core.basis import Basis

DRIVER = Path(__file__).with_name("basic_driver.py")


#**********#
#   load   #
#**********#
def load(*files: Any, format: str = "auto",
         **options: Any) -> Dict[str, Union[NIfTI_MRS_Plus, nib.Nifti1Image]]:
    """
    Raw scanner data as NIfTI-MRS+, converted by spec2nii; NIfTI-MRS files as they are.

    One acquisition can give several outputs - a GE P-file holds the water-suppressed
    spectra and their water reference - so each comes back under spec2nii's name for it,
    as a batch of one with all its dimensions (coils, averages, ...) as acquired. What is
    not spectra, such as DICOM's ``voi=True`` mask of where the voxel sits, comes back as
    a plain NIfTI image.

    Args:
        *files: The raw files as spec2nii takes them for the format: a P-file, a Twix or
            RDA file, a DICOM folder, a Philips .SDAT and its .SPAR, ...
        format (str): spec2nii's subcommand, e.g. 'ge', 'twix', 'philips', 'dicom',
            'rda', 'bruker', 'varian', 'jmrui', 'text', 'raw'. 'auto' lets spec2nii tell
            Twix, RDA, SDAT/SPAR, GE and DICOM apart itself, which it calls experimental.
        **options: That subcommand's options by name, e.g. ``evalinfo="image"`` for
            Twix; True for a flag, a list for an option taking several values.

    Returns:
        Dict[str, Union[NIfTI_MRS_Plus, nib.Nifti1Image]]: Output name -> the data, its
        processing history starting with the conversion, or the image. NIfTI-MRS files
        need no conversion: they come back as one batch, 'spectra', through ``read``.

    Raises:
        RuntimeError: If spec2nii fails or writes nothing, with its own message.
    """
    if format == "auto" and files and all(str(f).endswith((".nii", ".nii.gz")) for f in files):
        return {"spectra": read(*files)}
    import spec2nii

    flags = []
    for name, value in options.items():
        if value is True:
            flags.append(f"--{name}")
        elif isinstance(value, (list, tuple)):
            flags += [f"--{name}", *map(str, value)]
        elif value is not False and value is not None:
            flags += [f"--{name}", str(value)]

    with tempfile.TemporaryDirectory() as folder:
        # spec2nii is a command line tool only; run it with this Python, so the version
        # recorded below is the one that converted
        run = subprocess.run([sys.executable, "-c", "from spec2nii.spec2nii import main; main()",
                              format, *map(str, files), *flags, "-o", folder],
                             capture_output=True, text=True)
        if run.returncode != 0:
            raise RuntimeError(f"spec2nii {format} failed:\n{run.stderr.strip()}")

        out = {}
        for path in sorted(Path(folder).glob("*.nii.gz")):   # its only output, nifti1 too
            name = path.name.removesuffix(".nii.gz")
            try:
                nifti = NIFTI_MRS(str(path)).copy()   # read now: the folder goes
            except NotNIFTI_MRS:
                image = nib.load(path)
                out[name] = type(image)(np.asarray(image.dataobj), image.affine, image.header)
                continue
            provenance.append(nifti, provenance.entry(
                "spec2nii", spec2nii.__version__, f"spec2nii {format}",
                {"files": [Path(f) for f in files], "options": options}))
            out[name] = NIfTI_MRS_Plus([nifti])
    if not out:   # some modes only list what is there, some find nothing to convert
        said = "\n".join(text.strip() for text in (run.stdout, run.stderr) if text.strip())
        raise RuntimeError(f"spec2nii {format} wrote nothing for {list(map(str, files))}"
                           + (f":\n{said}" if said else ""))
    return out


#**********#
#   read   #
#**********#
def read(*paths: Any) -> NIfTI_MRS_Plus:
    """
    NIfTI-MRS files as one NIfTI-MRS+.

    Args:
        *paths: The files, one spectrum each, with all its dimensions.

    Returns:
        NIfTI_MRS_Plus: The spectra, in the order given, each one's processing history
        starting with the file it was read from, by its portable path and its checksum.
    """
    import nifti_mrs

    spectra = []
    for path in paths:
        spectrum = NIFTI_MRS(str(path))
        provenance.append(spectrum, provenance.entry(
            "MRSuite", provenance.mrsuite_version(), "mrsuite.io.read",
            {"file": Path(path), "reader": f"nifti-mrs {nifti_mrs.__version__}"}))
        spectra.append(spectrum)
    return NIfTI_MRS_Plus(spectra)


#***********#
#   basis   #
#***********#
def basis(files: Any, format: str = "auto", sw: Optional[float] = None,
          sf: Optional[float] = None, te: Optional[float] = None) -> Basis:
    """
    A basis set, in any format BasiC reads, as one Basis every fit takes.

    BasiC tells the format from the files themselves, as ``load`` lets spec2nii do: an
    LCModel .BASIS, jMRUI .txt, NIfTI-MRS, FID-A, Osprey, MARSS or MRSCloud .mat, ... What
    the files state wins; ``sw``, ``sf`` and ``te`` fill only what they leave out.

    Args:
        files: The basis files: one holding the whole set, or a list, one per metabolite.
        format (str): BasiC's name for the format, e.g. 'MRSCloud .mat', where it cannot
            tell; 'auto' lets it tell.
        sw (Optional[float]): Spectral width in Hz, where the files do not state it.
        sf (Optional[float]): Spectrometer frequency in MHz, where the files do not state it.
        te (Optional[float]): Echo time in ms, where the files do not state it.

    Returns:
        Basis: The basis set, with the files it came from.

    Raises:
        RuntimeError: If BasiC cannot tell the format or read the set, or the files lack
            what it needs, with what to give.
    """
    files = [_whole(f) for f in ([files] if isinstance(files, (str, Path)) else files)]
    with tempfile.TemporaryDirectory() as folder:
        result = Path(folder) / "fids.npz"
        said = _basic(task="read", files=[str(f) for f in files], format=format, sw=sw, sf=sf,
                      te=te, result=str(result))
        fids = np.load(result)["fids"]
    given = {"sw": sw, "sf": sf, "te": te}
    source = {"files": [provenance.portable_file(f) for f in files], "format": said.pop("format"),
              "reader": f"BasiC {externals.version('basic')}",
              "given": {k: v for k, v in given.items() if v is not None}}
    return Basis(fids=fids, source=source, **said)


#***********#
#   truth   #
#***********#
# what ground truth comes in, by extension; and the names a file's truth goes by among others
TRUTHS = (".json", ".pkl", ".pickle", ".mat", ".csv", ".tsv", ".xlsx", ".xls", ".py")
TRUTH_WORDS = {"truth", "gt", "groundtruth", "ground_truth", "concentrations", "conc", "concs"}


def truth(path: Any, key: Optional[str] = None) -> List[Dict[str, float]]:
    """
    Ground-truth concentrations from a file, one dictionary per spectrum.

    A JSON, pickle, MATLAB (.mat, before v7.3) or Python file (run, its variables read)
    may hold one spectrum's concentrations ({'NAA': 12.0, ...}, or a variable per
    metabolite), a list of those, a mapping of each metabolite to one value per spectrum,
    or a table. A table (CSV, TSV, Excel, or a pandas DataFrame in a pickle) has a row per
    spectrum and a column per metabolite, or a metabolite column beside a concentration
    column per spectrum, as the 2016 fitting challenge's spreadsheets do below their
    header (see ``_table``).

    Args:
        path: The file.
        key (Optional[str]): The variable or field that holds the truth, where the file
            holds more; by default one named as TRUTH_WORDS, else the only one.

    Returns:
        List[Dict[str, float]]: Metabolite -> concentration, per spectrum, as scoring
        takes it; a metabolite left blank for a spectrum is left out of its dictionary.

    Raises:
        ValueError: For a file it cannot tell the concentrations in, saying what it found.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".csv", ".tsv", ".xlsx", ".xls"):
        return _concentrations(_table(path))
    if suffix == ".json":
        held = json.loads(path.read_text())
    elif suffix in (".pkl", ".pickle"):
        import pickle
        with open(path, "rb") as f:
            held = pickle.load(f)                      # noqa: S301 - the user's own file
    elif suffix == ".mat":
        from scipy.io import loadmat
        held = {k: v for k, v in loadmat(path, simplify_cells=True).items()
                if not k.startswith("__")}
    elif suffix == ".py":
        import runpy
        held = {k: v for k, v in runpy.run_path(str(path)).items()
                if not k.startswith("_") and (isinstance(v, (dict, list, tuple, np.ndarray))
                                              or hasattr(v, "to_dict") or _number(v))}
    else:
        raise ValueError(f"{path.name}: ground truth is read from {', '.join(TRUTHS)}")
    try:
        return _concentrations(_picked(held, key, variables=suffix in (".mat", ".py")))
    except ValueError as err:
        raise ValueError(f"{path.name}: {err}") from None


#************#
#   picked   #
#************#
def _picked(held: Any, key: Optional[str], variables: bool) -> Any:
    """The truth among what a file holds: by key, by its name, the only variable, or the
    variables themselves where each is a number."""
    if key is not None:
        if not isinstance(held, dict) or key not in held:
            raise ValueError(f"no {key!r} in it; it holds {_held(held)}")
        return held[key]
    if isinstance(held, dict):
        named = [k for k in held if str(k).lower() in TRUTH_WORDS]
        if len(named) == 1:
            return held[named[0]]
        if variables and held and all(_number(v) for v in held.values()):
            return held                                  # each variable a metabolite
        if variables:
            if len(held) != 1:
                raise ValueError(f"say which holds the truth (key): {_held(held)}")
            return next(iter(held.values()))
    return held


#***********#
#   table   #
#***********#
def _table(path: Path) -> Any:
    """
    A table's spectra. Below a header with a metabolite column, each concentration column
    is one spectrum (the 2016 challenge's sheets have notes above theirs, and one column);
    otherwise the first row names the metabolites and each row below is a spectrum, columns
    that name no metabolite (id, subject, spectrum, an index without a name) left out.

    Raises:
        ValueError: For a metabolite twice in one spectrum, or a first row of numbers,
            which names no metabolite.
    """
    import pandas as pd

    sheet = pd.read_excel(path, header=None) if path.suffix.lower() in (".xlsx", ".xls") else \
        pd.read_csv(path, header=None, sep="\t" if path.suffix.lower() == ".tsv" else ",")
    for row, cells in sheet.iterrows():
        words = [str(c).strip().lower() for c in cells]
        names = next((k for k, w in enumerate(words) if w.startswith("metabolite")), None)
        values = [k for k, w in enumerate(words) if w.startswith("conc")]
        if names is not None and values:
            below = sheet.iloc[row + 1:]
            told = below.iloc[:, names].map(lambda v: isinstance(v, str) and bool(v.strip()))
            below = below.iloc[:len(told) if told.all() else int(np.argmin(told.to_numpy()))]
            metabolites = below.iloc[:, names].str.strip()
            twice = sorted(set(metabolites[metabolites.duplicated()]))
            if twice:
                raise ValueError(f"{', '.join(twice)} twice in one spectrum: give a "
                                 "concentration column per spectrum, or a row per spectrum")
            return [dict(zip(metabolites, pd.to_numeric(below.iloc[:, k], errors="coerce")))
                    for k in values]
    header = [str(c).strip() for c in sheet.iloc[0]]
    if all(pd.notna(pd.to_numeric(h, errors="coerce")) for h in header if h != "nan"):
        raise ValueError("its first row is numbers: a table needs a header naming the "
                         "metabolites")
    sheet.columns = header
    named = [c for c in header if c.lower() not in _IDS and not c.lower().startswith("unnamed")]
    return sheet.iloc[1:][named].apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")


# a table's columns that name a spectrum, not a metabolite
_IDS = {"", "nan", "id", "ids", "index", "subject", "spectrum", "sample", "dataset", "name",
        "file"}


#********************#
#   concentrations   #
#********************#
def _concentrations(held: Any) -> List[Dict[str, float]]:
    """Concentrations per spectrum from what a file held (see ``truth``).

    Raises:
        ValueError: For anything else, saying what it was.
    """
    if hasattr(held, "to_dict") and hasattr(held, "columns"):         # a DataFrame
        held = held.to_dict("records")
    if isinstance(held, dict) and held:
        values = list(held.values())
        if all(_number(v) for v in values):
            return [{str(k): float(v) for k, v in held.items() if not np.isnan(float(v))}]
        if all(isinstance(v, dict) for v in values):                  # by spectrum
            return [c for v in values for c in _concentrations(v)]
        columns = [np.atleast_1d(np.asarray(v)) for v in values]
        if all(c.ndim == 1 and c.dtype.kind in "iuf" for c in columns) and \
                len({c.size for c in columns}) == 1:                  # by metabolite
            return [{str(k): float(c[j]) for k, c in zip(held, columns) if not np.isnan(c[j])}
                    for j in range(columns[0].size)]
    if isinstance(held, (list, tuple, np.ndarray)) and len(held) and \
            all(isinstance(v, dict) for v in held):
        return [c for v in held for c in _concentrations(v)]
    raise ValueError(f"no concentrations found: it holds {_held(held)}")


#************#
#   number   #
#************#
def _number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


#**********#
#   held   #
#**********#
def _held(held: Any) -> str:
    """What a file held, in a few words, for an error."""
    if isinstance(held, dict):
        return "{" + ", ".join(f"{k}: {type(v).__name__}" for k, v in list(held.items())[:8]) + \
            (", ..." if len(held) > 8 else "") + "}"
    return f"{type(held).__name__} of {len(held)}" if hasattr(held, "__len__") else \
        type(held).__name__


#*****************#
#   merge basis   #
#*****************#
def merge_basis(basis: Basis, more: Dict[str, Any],
                keep: Optional[Dict[str, str]] = None) -> Basis:
    """A basis set with other basis sets' functions and measured spectra added, e.g. a
    macromolecule acquisition as {'MM': mm}; see ``mrsuite.core.basis.merged``."""
    from mrsuite.core.basis import merged
    return merged(basis, more, keep)


#*******************#
#   convert basis   #
#*******************#
def convert_basis(basis: Basis, folder: Any, tool: str, output: str = "default") -> List[Path]:
    """
    A basis set written for a tool BasiC writes for.

    Args:
        basis (Basis): The basis set, e.g. from ``basis``.
        folder: Where to write.
        tool (str): BasiC's id of the tool, e.g. 'osprey', 'fsLmrs', 'jmrui', 'niftimrs',
            'lcmodel', 'fida'; a wrong one is told the others.
        output (str): The tool's output format where it has several, e.g. 'basis', 'raw' or
            'both' for 'lcmodel'; 'default' is BasiC's choice. An echo time the basis does
            not hold is written as BasiC writes it (30 ms; 26 ms for OXSA).

    Returns:
        List[Path]: The files written, as absolute paths. A metabolite BasiC's writer fails
        it goes on without, and says so in a warning.

    Raises:
        RuntimeError: For a tool or output BasiC does not write, or if it wrote nothing.
    """
    folder = _whole(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        fids = Path(scratch) / "fids.npz"
        np.savez(fids, fids=basis.fids)
        written = _basic(task="write", fids=str(fids), folder=str(folder), tool=tool,
                         output=output, names=list(basis.names), bandwidth=basis.bandwidth,
                         frequency=basis.frequency, centre=basis.centre, te=basis.te,
                         seq=basis.seq)
    return [_whole(p) for p in written]


#***********#
#   basic   #
#***********#
def _basic(**job: Any) -> Any:
    """One job through basic_driver.py, without leaving bytecode in BasiC's folder; its
    answer. What BasiC warns of comes along as warnings."""
    job["root"] = str(externals.ensure("basic") / "basis_converter")
    run = subprocess.run([sys.executable, "-B", str(DRIVER), json.dumps(job, default=float)],
                         capture_output=True, text=True)
    said = run.stderr.strip().splitlines()
    for line in said:
        if "WARNING" in line:
            warnings.warn(f"BasiC: {line.strip()}")
    if run.returncode != 0:
        raise RuntimeError(f"BasiC could not {job['task']} the basis set: "
                           + (said[-1] if said else f"exit status {run.returncode}"))
    return json.loads(run.stdout)


#***********#
#   whole   #
#***********#
def _whole(path: Any) -> Path:
    """A path from the home folder or the working one, as the driver's process needs it."""
    return Path(path).expanduser().absolute()
