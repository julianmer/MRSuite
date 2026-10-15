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
#          a subject not fetched is said so, with the command.                                     #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import warnings
from pathlib import Path
from typing import Tuple

from nifti_mrs_plus import NIfTI_MRS_Plus

from mrsuite import collect
from mrsuite.datasets.challenge2016 import DATA

# what a spectrum file is called: BIDS MRS NIfTI by suffix, raw scanner files by extension
SPECTRA = ("_svs.nii", "_mrsref.nii", "_mrsi.nii", "_unloc.nii")

# each dataset: what it holds, by the names collect gives its files, in the order load gives them
DATASETS = {
    "ds005371": ("data", "water"),        # Mikkelsen: HERMES and HERCULES, PRESS and sLASER, 3T GE
    "ds006812": ("data", "mm"),           # COWS: sLASER TE 26 ms with water suppression schemes
}


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
    return tuple(NIfTI_MRS_Plus(found.get(name, [])) for name in DATASETS[dataset])


#**************#
#   spectrum   #
#**************#
def _spectrum(path: Path) -> bool:
    name = path.name.lower()
    return any(s in name for s in SPECTRA) or path.suffix.lower() in collect.FORMATS
