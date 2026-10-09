####################################################################################################
#                                            collect.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-09                                                                              #
#                                                                                                  #
# Purpose: A study folder as the spectra it holds: every MRS file under it found and told its      #
#          role (the spectrum, its water reference, a macromolecule acquisition) and its group     #
#          from its name, BIDS or not, then loaded role by role for a pipeline to take.            #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd
from nifti_mrs.nifti_mrs import NIFTI_MRS, NotNIFTI_MRS
from nifti_mrs_plus import NIfTI_MRS_Plus

from mrsuite import io

# the roles, as BIDS names MRS files (svs, mrsi, unloc, mrsref), plus a measured
# macromolecule spectrum, which BIDS has no suffix for
ROLES = ("svs", "mrsi", "unloc", "mrsref", "mm")

# words in a file's name that tell its role where it is not named as BIDS does
WORDS = {"mrsref": {"mrsref", "ref", "wref", "water", "h2o", "unsuppressed"},
         "mm": {"mm", "macromolecule", "macromolecules"},
         "mrsi": {"mrsi", "csi"}, "unloc": {"unloc"}}
_ROLE_WORDS = {w for ws in WORDS.values() for w in ws} | {"metab", "metabolite", "metabolites"}

# what spec2nii converts, by extension; a DICOM series is a folder of such files
FORMATS = {".dat": "twix", ".rda": "rda", ".sdat": "philips", ".7": "ge"}
DICOM = {".dcm", ".ima"}
SKIPPED = {"derivatives", "code"}                # BIDS: what was made from the data, not it

_BIDS = re.compile(r"^(sub-[^_]+(?:_[a-z]+-[^_]+)*)_(svs|mrsi|unloc|mrsref)$")


#**************************************************************************************************#
#                                           Class Entry                                            #
#**************************************************************************************************#
#                                                                                                  #
# One acquisition found: a file, or a DICOM series' folder, with its role and group.               #
#                                                                                                  #
#**************************************************************************************************#
@dataclass
class Entry:
    """One acquisition found: a file, or a DICOM series' folder, with its role and group.

    Attributes:
        path (Path): The file, or the folder of a DICOM series.
        role (str): One of ROLES.
        group (str): What it belongs with: its BIDS entities (sub-01_ses-01_acq-press), or
            its name without the words that told its role, under its folder.
        format (str): spec2nii's subcommand for it, or 'nifti' for NIfTI-MRS read as it is.
    """

    path: Path
    role: str
    group: str
    format: str


#**************#
#   discover   #
#**************#
def discover(folder: Any, roles: Optional[Dict[str, str]] = None,
             groups: Optional[Dict[str, str]] = None) -> List[Entry]:
    """
    Every MRS acquisition under a folder, by its name: BIDS names say the role and the
    group outright; otherwise words like ref, water, h2o tell a water reference, mm a
    macromolecule acquisition, and the rest of the name is the group. NIfTI files are
    read to tell spectra from images, so one whose content is not there (a git-annex link
    not fetched) is left out; derivatives/ and code/ are too. Raw files are listed by name.

    Args:
        folder: The study's folder.
        roles (Optional[Dict[str, str]]): Roles to take instead, by path relative to the
            folder, where a name misled; '' leaves that file out.
        groups (Optional[Dict[str, str]]): Groups to take instead, the same way.

    Returns:
        List[Entry]: In group order, each group's spectrum before its reference.

    Raises:
        ValueError: If it is not a folder.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"{folder} is not a folder")
    entries = []
    for path in sorted(folder.rglob("*")):
        parts = path.relative_to(folder).parts
        if any(p.startswith(".") or p in SKIPPED for p in parts):
            continue
        format = _format(path)
        if format and not (path.is_dir() and format != "dicom"):
            entries.append(_entry(path, folder, format))
    for e in entries:
        rel = e.path.relative_to(folder).as_posix()
        e.role, e.group = (roles or {}).get(rel, e.role), (groups or {}).get(rel, e.group)
    wrong = {e.role for e in entries} - set(ROLES) - {""}
    if wrong:
        raise ValueError(f"no such role: {sorted(wrong)}; one of {ROLES}")
    entries = [e for e in entries if e.role]
    return sorted(entries, key=lambda e: (e.group, ROLES.index(e.role), str(e.path)))


#************#
#   format   #
#************#
def _format(path: Path) -> Optional[str]:
    """What reads a path: spec2nii's subcommand by the extension, 'dicom' for a folder of
    DICOM files, 'nifti' for NIfTI-MRS (its header read, so an image or a git-annex link
    without content is None); None for anything else."""
    if path.is_dir():
        files = (f for f in path.iterdir() if f.is_file())
        return "dicom" if any(f.suffix.lower() in DICOM for f in files) else None
    if path.name.lower().endswith((".nii", ".nii.gz")):
        if not path.exists():
            return None
        try:
            NIFTI_MRS(str(path))
        except Exception:                              # noqa: BLE001 - an image, the T1 say
            return None
        return "nifti"
    return FORMATS.get(path.suffix.lower())


#*************#
#   entries   #
#*************#
def entries(paths: Sequence[Any]) -> List[Entry]:
    """Files or DICOM folders chosen anywhere, as entries by their names (see ``discover``),
    each grouped under its own folder; one nothing reads is left out."""
    out = []
    for path in map(Path, paths):
        format = _format(path)
        if format:
            out.append(_entry(path, path.parent, format))
    return out


#**************#
#   as given   #
#**************#
def as_entries(items: Sequence[Any]) -> List[Entry]:
    """Entries from (path, role, group) triples, as a graph keeps them.

    Raises:
        ValueError: For a path nothing reads, or a role that is not one of ROLES.
    """
    out = []
    for item in items:
        if isinstance(item, Entry):
            out.append(item)
            continue
        path, role, group = item
        format = _format(Path(path))
        if format is None:
            raise ValueError(f"nothing reads {path}")
        if role not in ROLES:
            raise ValueError(f"no such role: {role!r}; one of {ROLES}")
        out.append(Entry(Path(path), role, group, format))
    return out


def as_triples(found: Sequence[Entry]) -> List[Tuple[str, str, str]]:
    """Entries as (path, role, group) triples, as a graph keeps them."""
    return [(str(e.path), e.role, e.group) for e in found]


#***********#
#   entry   #
#***********#
def _entry(path: Path, folder: Path, format: str) -> Entry:
    """A file's role and group from its name."""
    stem = path.name.split(".")[0]
    bids = _BIDS.match(stem)
    if bids:
        return Entry(path, bids[2], bids[1], format)
    words = [w for w in re.split(r"[^A-Za-z0-9]+", stem) if w]
    role = next((r for r, ws in WORDS.items() if any(w.lower() in ws for w in words)), "svs")
    told = "|".join(sorted(_ROLE_WORDS, key=len, reverse=True))
    group = re.sub(rf"(?i)(?<![A-Za-z0-9])({told})(?![A-Za-z0-9])", "", stem)   # as named
    group = re.sub(r"([_\-. ])[_\-. ]+", r"\1", group).strip("_-. ") or stem
    parent = path.parent.relative_to(folder)
    return Entry(path, role, group if parent == Path() else f"{parent.as_posix()}/{group}",
                 format)


#***********#
#   table   #
#***********#
def table(entries: List[Entry]) -> pd.DataFrame:
    """The entries as a table: group, role, format, path."""
    return pd.DataFrame([{"group": e.group, "role": e.role, "format": e.format,
                          "path": str(e.path)} for e in entries],
                        columns=["group", "role", "format", "path"])


#**********#
#   load   #
#**********#
def load(what: Union[str, Path, List[Entry]], roles: Optional[Dict[str, str]] = None,
         groups: Optional[Dict[str, str]] = None) -> Dict[str, NIfTI_MRS_Plus]:
    """
    A study's spectra, one batch per role, in group order, so a reference lines up with
    its spectrum where every group has both.

    Args:
        what: The study's folder (see ``discover``), or entries, e.g. relabelled, as
            Entry objects or (path, role, group) triples.
        roles, groups: As ``discover`` takes them, for a folder.

    Returns:
        Dict[str, NIfTI_MRS_Plus]: Role -> the spectra, each with its file as provenance
        (``io.read``, or ``io.load`` through spec2nii). A raw file holding its own
        reference, as a GE P-file does, gives that reference as 'mrsref'.

    Raises:
        ValueError: If nothing is found.
    """
    entries = discover(what, roles, groups) if isinstance(what, (str, Path)) else as_entries(what)
    if not entries:
        raise ValueError(f"no MRS files found under {what}")
    found: Dict[str, list] = {}
    for e in entries:
        if e.format == "nifti":
            found.setdefault(e.role, []).extend(io.read(e.path).list())
            continue
        # Twix: the image data, as spec2nii's own auto mode takes it (evalinfo 'image')
        options = {"evalinfo": "image"} if e.format == "twix" else {}
        for name, value in io.load(e.path, format=e.format, **options).items():
            if not isinstance(value, NIfTI_MRS_Plus):     # an image, e.g. DICOM's voi
                continue
            words = set(re.split(r"[^a-z0-9]+", name.lower()))
            role = "mrsref" if e.role == "svs" and WORDS["mrsref"] & words else e.role
            found.setdefault(role, []).extend(value.list())
    counts = {role: len(items) for role, items in found.items()}
    if len(set(counts.values())) > 1:
        warnings.warn(f"the roles differ in number, so their spectra do not line up: {counts}")
    return {role: NIfTI_MRS_Plus(found[role]) for role in ROLES if role in found}
