####################################################################################################
#                                            collect.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-09                                                                              #
#                                                                                                  #
# Purpose: Data from anywhere as named batches: files and folders, every MRS file and ground truth #
#          under a folder found, each given a name (the data, its water reference, a               #
#          macromolecule acquisition, the truth, or any other) and a group from its file name,     #
#          BIDS or not, then loaded name by name for a pipeline to take.                           #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import glob
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd
from nifti_mrs.nifti_mrs import NIFTI_MRS, NotNIFTI_MRS
from nifti_mrs_plus import NIfTI_MRS_Plus

from mrsuite import io

# the names a file is told by, in the order they come out; any other name may be given. BIDS'
# suffixes become svs -> data, mrsref -> water, mrsi, unloc; mm is a measured macromolecule
# spectrum, which BIDS has no suffix for
NAMES = ("data", "water", "mm", "truth", "mrsi", "unloc")
_BIDS_NAMES = {"svs": "data", "mrsref": "water", "mrsi": "mrsi", "unloc": "unloc"}

# words in a file's name that tell its name where it is not named as BIDS does
WORDS = {"water": {"mrsref", "ref", "wref", "water", "h2o", "unsuppressed"},
         "mm": {"mm", "macromolecule", "macromolecules"},
         "mrsi": {"mrsi", "csi"}, "unloc": {"unloc"}}
_NAME_WORDS = {w for ws in WORDS.values() for w in ws} | io.TRUTH_WORDS | set(_BIDS_NAMES) | \
    {"metab", "metabolite", "metabolites"}

# what spec2nii converts, by extension; a DICOM series is a folder of such files (or of
# files without one that DICOM's preamble tells), a Varian scan a .fid folder; a Philips
# .SDAT is read with the .SPAR beside it. 'nifti' is NIfTI-MRS read as it is, 'truth' ground
# truth (io.truth); any spec2nii subcommand may be set
FORMATS = {".dat": "twix", ".rda": "rda", ".sdat": "philips", ".7": "ge"}
DICOM = {".dcm", ".ima"}
SKIPPED = {"derivatives", "code"}                # BIDS: what was made from the data, not it

_ENTITIES = r"sub-[^_]+(?:_[a-z]+-[^_]+)*"
_BIDS = re.compile(rf"^({_ENTITIES})_(svs|mrsi|unloc|mrsref)$")


#**************************************************************************************************#
#                                           Class Entry                                            #
#**************************************************************************************************#
#                                                                                                  #
# One file to load: a file, or a DICOM series' folder, with its name, group and format.            #
#                                                                                                  #
#**************************************************************************************************#
@dataclass
class Entry:
    """One file to load: a file, or a DICOM series' folder, with its name, group and format.

    Attributes:
        path (Path): The file, or the folder of a DICOM series.
        name (str): The batch it goes into, e.g. one of NAMES.
        group (str): What it belongs with: its BIDS entities (sub-01_ses-01_acq-press), or
            its name without the words that told its name, under its folder.
        format (str): spec2nii's subcommand for it, 'nifti' for NIfTI-MRS read as it is,
            'truth' for ground truth; '' where nothing tells, to be set.
    """

    path: Path
    name: str
    group: str
    format: str


#**************#
#   discover   #
#**************#
def discover(folder: Any, names: Optional[Dict[str, str]] = None,
             groups: Optional[Dict[str, str]] = None) -> List[Entry]:
    """
    Every MRS acquisition and ground truth under a folder, named by its file name: BIDS
    names say the name and the group outright; otherwise words like ref, water, h2o tell
    a water reference, mm a macromolecule acquisition, truth or gt ground truth (a file
    io.truth reads), and the rest of the name is the group. NIfTI files are read to tell
    spectra from images, so one whose content is not there (a git-annex link not
    fetched) is left out; derivatives/ and code/ are too. Raw files are listed by name.

    Args:
        folder: The study's folder.
        names (Optional[Dict[str, str]]): Names to take instead, by path relative to the
            folder, where a file name misled; '' leaves that file out.
        groups (Optional[Dict[str, str]]): Groups to take instead, the same way.

    Returns:
        List[Entry]: In group order, each group's in the order of NAMES.

    Raises:
        ValueError: If it is not a folder.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"{folder} is not a folder")
    entries = []
    for path in _walk(folder):
        parts = path.relative_to(folder).parts
        if any(p.startswith(".") or p in SKIPPED for p in parts):
            continue
        if any(p.suffix.lower() == ".fid" for p in path.relative_to(folder).parents):
            continue                                   # inside a Varian scan, read whole
        format = _format(path) or ("truth" if _truth(path) else None)
        if format:
            entries.append(_entry(path, folder, format))
    for e in entries:
        rel = e.path.relative_to(folder).as_posix()
        e.name, e.group = (names or {}).get(rel, e.name), (groups or {}).get(rel, e.group)
    return sort([e for e in entries if e.name])


#**********#
#   walk   #
#**********#
def _walk(folder: Path) -> List[Path]:
    """Every path under a folder, in order, symlinked folders followed (a study kept on
    another disk), which rglob leaves out."""
    found = []
    for dirpath, dirnames, files in os.walk(folder, followlinks=True):
        found.extend(Path(dirpath) / name for name in [*dirnames, *files])
    return sorted(found)


#************#
#   format   #
#************#
def _format(path: Path) -> Optional[str]:
    """What reads a path: spec2nii's subcommand by the extension, 'dicom' for a folder of
    DICOM files, 'nifti' for NIfTI-MRS (its header read, so an image or a git-annex link
    without content is None); None for anything else."""
    if path.is_dir():
        if path.suffix.lower() == ".fid":
            return "varian"
        files = [f for f in path.iterdir() if f.is_file() and not f.name.startswith(".")]
        return "dicom" if any(f.suffix.lower() in DICOM for f in files) or \
            any(_preamble(f) for f in files[:1] if not f.suffix) else None
    if path.name.lower().endswith((".nii", ".nii.gz")):
        if not path.exists():
            return None
        try:
            NIFTI_MRS(str(path))
        except Exception:                              # noqa: BLE001 - an image, the T1 say
            return None
        return "nifti"
    return FORMATS.get(path.suffix.lower())


#**************#
#   preamble   #
#**************#
def _preamble(path: Path) -> bool:
    """Whether a file starts as DICOM does: 128 bytes, then 'DICM'."""
    try:
        with open(path, "rb") as f:
            return f.read(132)[128:] == b"DICM"
    except OSError:
        return False


#***********#
#   truth   #
#***********#
def _truth(path: Path) -> bool:
    """Whether a file in a folder is ground truth: one io.truth reads, named so, as a BIDS
    sidecar's .json is not; never a Python file, which reading would run (one is read
    when chosen)."""
    words = {w.lower() for w in re.split(r"[^A-Za-z0-9]+", path.name.split(".")[0])}
    return path.is_file() and path.suffix.lower() in set(io.TRUTHS) - {".py"} and \
        bool(words & io.TRUTH_WORDS)


#*************#
#   entries   #
#*************#
def entries(paths: Sequence[Any]) -> List[Entry]:
    """Files or folders a format reads whole (DICOM, Varian) chosen anywhere, named by
    their file names (see ``discover``), each grouped under its own folder. A file io.truth
    reads is ground truth whatever its name; a Philips .SPAR goes with its .SDAT; one
    nothing tells has format '', to be set."""
    out = []
    for path in map(Path, paths):
        if path.suffix.lower() == ".spar" and _sidecar(path, ".sdat"):
            continue
        format = _format(path) or ("truth" if path.suffix.lower() in io.TRUTHS else "")
        out.append(_entry(path, path.parent, format))
    return out


#**************#
#   as given   #
#**************#
def as_entries(items: Sequence[Any]) -> List[Entry]:
    """Entries from (path, name, group[, format]) tuples, as a graph keeps them; the format
    as the path tells it where none is given.

    Raises:
        ValueError: For a path no format reads, or no name.
    """
    out = []
    for item in items:
        if isinstance(item, Entry):
            out.append(item)
            continue
        path, name, group, *format = item
        format = format[0] if format else _format(Path(path)) or \
            ("truth" if Path(path).suffix.lower() in io.TRUTHS else "")
        if not format:
            raise ValueError(f"nothing reads {path}: give its format")
        if not name:
            raise ValueError(f"{path} has no name")
        out.append(Entry(Path(path), name, group, format))
    return out


def as_tuples(found: Sequence[Entry]) -> List[Tuple[str, str, str, str]]:
    """Entries as (path, name, group, format) tuples, as a graph keeps them."""
    return [(str(e.path), e.name, e.group, e.format) for e in found]


#***********#
#   entry   #
#***********#
def _entry(path: Path, folder: Path, format: str) -> Entry:
    """A file's name and group from its file name: a BIDS one's from its suffix, grouped
    by its entities; another's from its words, grouped by the rest of it - by the BIDS
    entities that are left (sub-01_svs_truth with sub-01_svs), else under its folder."""
    stem = path.name.split(".")[0]
    bids = _BIDS.match(stem)
    if bids:
        return Entry(path, "truth" if format == "truth" else _BIDS_NAMES[bids[2]], bids[1],
                     format)
    words = [w for w in re.split(r"[^A-Za-z0-9]+", stem) if w]
    name = "truth" if format == "truth" else \
        next((n for n, ws in WORDS.items() if any(w.lower() in ws for w in words)), "data")
    told = "|".join(sorted(_NAME_WORDS, key=len, reverse=True))
    group = re.sub(rf"(?i)(?<![A-Za-z0-9])({told})(?![A-Za-z0-9])", "", stem)   # as named
    group = re.sub(r"([_\-. ])[_\-. ]+", r"\1", group).strip("_-. ") or stem
    parent = path.parent.relative_to(folder)
    if re.fullmatch(_ENTITIES, group) or parent == Path():
        return Entry(path, name, group, format)
    return Entry(path, name, f"{parent.as_posix()}/{group}", format)


#***********#
#   order   #
#***********#
def order(names: Iterable[str]) -> List[str]:
    """Names as they come out: those of NAMES in its order, then the rest as they came."""
    seen = list(dict.fromkeys(names))
    return [n for n in NAMES if n in seen] + [n for n in seen if n not in NAMES]


#**********#
#   sort   #
#**********#
def sort(found: Sequence[Entry]) -> List[Entry]:
    """Entries by group, numbers in it by value (dataset2 before dataset10), each group's
    by name as ``order`` has them, then by path."""
    rank = {n: k for k, n in enumerate(order(e.name for e in found))}
    return sorted(found, key=lambda e: (_natural(e.group), rank[e.name], _natural(str(e.path))))


def _natural(text: str) -> List[Any]:
    """Text as it sorts by eye: its numbers by value."""
    return [(0, int(part), "") if part.isdigit() else (1, 0, part)
            for part in re.split(r"(\d+)", text) if part]


#***********#
#   table   #
#***********#
def table(entries: List[Entry]) -> pd.DataFrame:
    """The entries as a table: group, name, format, path."""
    return pd.DataFrame([{"group": e.group, "name": e.name, "format": e.format,
                          "path": str(e.path)} for e in entries],
                        columns=["group", "name", "format", "path"])


#*************#
#   sidecar   #
#*************#
def _sidecar(path: Path, suffix: str) -> Optional[Path]:
    """The file beside one with the same stem and another suffix, in either case."""
    return next((p for p in path.parent.glob(f"{glob.escape(path.stem)}.*")
                 if p.suffix.lower() == suffix), None)


#**********#
#   load   #
#**********#
def load(what: Union[str, Path, List[Any]], names: Optional[Dict[str, str]] = None,
         groups: Optional[Dict[str, str]] = None) \
        -> Dict[str, Union[NIfTI_MRS_Plus, List[Dict[str, float]]]]:
    """
    Data by name: each name's spectra as one batch, or its ground truth as a list, in
    group order, so a reference and a truth line up with their spectrum where every group
    has each.

    Args:
        what: The study's folder (see ``discover``), or entries in the order to load them,
            e.g. renamed, as Entry objects or (path, name, group[, format]) tuples.
        names, groups: As ``discover`` takes them, for a folder.

    Returns:
        Dict[str, Union[NIfTI_MRS_Plus, List[Dict[str, float]]]]: Name -> the spectra,
        each with its file as provenance (``io.read``, or ``io.load`` through spec2nii),
        or the concentrations per spectrum (``io.truth``), in the order of ``order``. A
        raw file holding its own reference, as a GE P-file does, gives that reference as
        'water' where the file itself is 'data', else as its name and '_water'.

    Raises:
        ValueError: If nothing is found, or one name is given both spectra and truth.
    """
    entries = discover(what, names, groups) if isinstance(what, (str, Path)) else \
        as_entries(what)
    if not entries:
        raise ValueError(f"no MRS files found under {what}")
    found: Dict[str, list] = {}
    truths = {e.name for e in entries if e.format == "truth"}
    mixed = truths & {e.name for e in entries if e.format != "truth"}
    if mixed:
        raise ValueError(f"{', '.join(sorted(mixed))}: both spectra and ground truth; "
                         "give them different names")
    for e in entries:
        if e.format == "truth":
            found.setdefault(e.name, []).extend(io.truth(e.path))
            continue
        if e.format == "nifti":
            found.setdefault(e.name, []).extend(io.read(e.path).list())
            continue
        # Twix: the image data, as spec2nii's own auto mode takes it (evalinfo 'image')
        options = {"evalinfo": "image"} if e.format == "twix" else {}
        spar = [_sidecar(e.path, ".spar")] if e.format == "philips" else []
        for output, value in io.load(e.path, *filter(None, spar), format=e.format,
                                     **options).items():
            if not isinstance(value, NIfTI_MRS_Plus):     # an image, e.g. DICOM's voi
                continue
            words = set(re.split(r"[^a-z0-9]+", output.lower()))
            water = "water" if e.name == "data" else f"{e.name}_water"
            name = water if e.name not in ("water", water) and WORDS["water"] & words \
                else e.name
            found.setdefault(name, []).extend(value.list())
    counts = {name: len(items) for name, items in found.items()}
    if len(set(counts.values())) > 1:
        warnings.warn(f"the names differ in number, so they do not line up: {counts}")
    return {name: found[name] if name in truths else NIfTI_MRS_Plus(found[name])
            for name in order(found)}
