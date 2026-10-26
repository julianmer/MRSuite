####################################################################################################
#                                           openneuro.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 2026-10-15                                                                              #
#                                                                                                  #
# Purpose: The OpenNeuro datasets MRSuite carries as git-annex submodules under data/: one         #
#          subject at a time, as collect finds its files, in a fixed order of names so each        #
#          dataset is a block with known outputs. The content of a subject comes with git-annex;   #
#          a subject not fetched is said so, with the command. An installed MRSuite downloads a    #
#          subject from OpenNeuro on first use, at the snapshot the submodule pins.                #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import json
import shutil
import tempfile
import urllib.request
import warnings
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np
from nifti_mrs import tools
from nifti_mrs.nifti_mrs import NIFTI_MRS
from nifti_mrs_plus import NIfTI_MRS_Plus

from mrsuite import collect, externals
from mrsuite.datasets.challenge2016 import DATA

# what a spectrum file is called: BIDS MRS NIfTI by suffix, raw scanner files by extension
SPECTRA = ("_svs.nii", "_mrsref.nii", "_mrsi.nii", "_unloc.nii")

# each dataset: what it holds, by the names collect gives its files, in the order load gives them
DATASETS = {
    "ds005371": ("data", "water"),        # Mikkelsen: HERMES and HERCULES, PRESS and sLASER, 3T GE
    "ds006812": ("data", "water", "mm", "mm_water"),   # COWS: sLASER TE 26 ms, three water
}                                                      # suppression schemes, water inside
# whose raw files hold the water reference inside the metabolite scan: spec2nii hands COWS'
# 'Phs' loop out as DIM_USER_0, the unsuppressed transient at index 0, zero-padded to the
# suppressed ones' count, and the suppressed transients at index 1; each such batch comes
# apart into the suppressed spectra under its own name and the references under the other
WATER_INSIDE = {"ds006812": {"data": "water", "mm": "mm_water"}}
# the snapshot of each that its submodule pins (a test holds them equal), and OpenNeuro's API,
# which lists a snapshot's files with a URL each, pinned to that snapshot
TAGS = {"ds005371": "2.3.1", "ds006812": "1.0.1"}
API = "https://openneuro.org/crn/graphql"


#**********#
#   load   #
#**********#
def load(dataset: str, subject: str = "sub-01") -> Tuple[NIfTI_MRS_Plus, ...]:
    """
    One subject of an OpenNeuro dataset, its spectra by what they are.

    Args:
        dataset (str): An accession number of ``DATASETS``, e.g. 'ds005371'.
        subject (str): The subject's folder, e.g. 'sub-01'; its sessions and acquisitions
            come in group order, a group per acquisition.

    Returns:
        The batches named in ``DATASETS`` for the dataset, in that order: the water-suppressed
        spectra first, then their water references, or the macromolecule acquisitions.

    Raises:
        ValueError: If the dataset is not one carried, or the subject's content is not
            fetched from the annex yet.

    Warns:
        A file spec2nii cannot read (ds006812's sub-01 acq-06, as published, is one its Twix
        reader rejects) is left out and named.
    """
    if dataset not in DATASETS:
        raise ValueError(f"no OpenNeuro dataset {dataset!r} here; there are {sorted(DATASETS)}")
    root = DATA / f"openneuro_{dataset}"
    folder = root / subject
    if not externals.checkout():                   # no submodule: OpenNeuro's own files
        _fetch(dataset, subject, folder)
    spectra = [p for p in folder.rglob("*") if p.is_symlink() and _spectrum(p)]   # the annex's
    unfetched = [p for p in spectra if not p.exists()]
    if unfetched:
        raise ValueError(f"{dataset} {subject}: {len(unfetched)} of {len(spectra)} spectra are "
                         f"not fetched yet: git -C {root} annex get {subject}")
    entries = collect.discover(folder)
    if not entries:
        raise ValueError(f"{dataset} {subject}: no spectra found under {folder}")
    found, unread = {}, []
    for e in entries:
        try:
            loaded = collect.load([e])
        except RuntimeError:                    # spec2nii failed, with its own message
            unread.append(Path(e.path).name)
            continue
        for name, value in loaded.items():
            found.setdefault(name, []).extend(value.list())
    if unread:
        warnings.warn(f"{dataset} {subject}: {len(unread)} of {len(entries)} files could not "
                      f"be read by spec2nii and are left out: {', '.join(unread)}")
    for name, water in WATER_INSIDE.get(dataset, {}).items():
        found[name], found[water] = map(list, zip(*map(_water_apart, found.get(name, [])))) \
            if found.get(name) else ([], [])
    return tuple(NIfTI_MRS_Plus(found.get(name, [])) for name in DATASETS[dataset])


#*****************#
#   water apart   #
#*****************#
def _water_apart(spectrum: NIFTI_MRS) -> Tuple[NIFTI_MRS, NIFTI_MRS]:
    """A spectrum's water-suppressed transients and its water reference, handed out together
    on DIM_USER_0 (see ``WATER_INSIDE``): the suppressed ones, and the reference without
    the zero transients it was padded with, each marked for what it is."""
    water, suppressed = (part.copy(remove_dim="DIM_USER_0")
                         for part in tools.split(spectrum, "DIM_USER_0", 0))
    if "DIM_DYN" in water.dim_tags:
        axis = 4 + list(water.dim_tags).index("DIM_DYN")
        dead = [i for i in range(water.shape[axis]) if not np.any(np.take(water[:], i, axis))]
        if dead:
            water = tools.split(water, "DIM_DYN", dead)[0]
        if water.shape[axis] == 1:
            water = water.copy(remove_dim="DIM_DYN")
    for part, suppressed_ in ((water, False), (suppressed, True)):
        if "WaterSuppressed" in part.hdr_ext:
            part.remove_hdr_field("WaterSuppressed")
        part.add_hdr_field("WaterSuppressed", suppressed_)
    return suppressed, water


#**************#
#   spectrum   #
#**************#
def _spectrum(path: Path) -> bool:
    name = path.name.lower()
    return any(s in name for s in SPECTRA) or path.suffix.lower() in collect.FORMATS


#***********#
#   fetch   #
#***********#
def _fetch(dataset: str, subject: str, folder: Path) -> None:
    """A subject's files at the pinned snapshot, downloaded into ``folder`` on first use,
    beside it and moved in once all have come, so a part never passes for the subject."""
    if folder.is_dir():
        return
    top = {f["filename"]: f for f in _files(dataset)}
    if subject not in top or not top[subject]["directory"]:
        raise ValueError(f"{dataset} {TAGS[dataset]} has no subject {subject!r}")
    files = list(_walk(dataset, top[subject]["id"]))
    total = sum(size for _, size, _ in files)
    folder.parent.mkdir(parents=True, exist_ok=True)
    part = Path(tempfile.mkdtemp(prefix=".part", dir=folder.parent))
    try:
        with externals.fetching(f"OpenNeuro {dataset} {subject} ({len(files)} files, "
                                f"{total / 1e6:,.0f} MB)", folder.parent):
            for path, size, url in files:
                externals.download(url, part / path, size)
        externals.move_in(part, folder)
    finally:
        shutil.rmtree(part, ignore_errors=True)


#**********#
#   walk   #
#**********#
def _walk(dataset: str, tree: str, at: str = "") -> Iterator[Tuple[str, int, str]]:
    """Every file under a folder of the snapshot: its path below it, size and URL."""
    for f in _files(dataset, tree):
        if f["directory"]:
            yield from _walk(dataset, f["id"], f"{at}{f['filename']}/")
        else:
            yield f"{at}{f['filename']}", f["size"], f["urls"][0]


#***********#
#   files   #
#***********#
def _files(dataset: str, tree: Optional[str] = None) -> list:
    """One folder of the pinned snapshot as OpenNeuro's API lists it: the top, or ``tree``."""
    where = f'(tree: "{tree}")' if tree else ""
    query = (f'{{ snapshot(datasetId: "{dataset}", tag: "{TAGS[dataset]}") '
             f'{{ files{where} {{ filename directory id size urls }} }} }}')
    request = urllib.request.Request(API, json.dumps({"query": query}).encode(),
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        answer = json.load(response)
    if answer.get("errors"):
        raise RuntimeError(f"OpenNeuro could not list {dataset} {TAGS[dataset]}: "
                           f"{answer['errors'][0].get('message')}")
    return answer["data"]["snapshot"]["files"]
