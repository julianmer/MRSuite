####################################################################################################
#                                         test_openneuro.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 2026-10-15                                                                              #
#                                                                                                  #
# Purpose: The OpenNeuro datasets load one subject at a time by what their files are, and say     #
#          plainly when the subject's content is not fetched from the annex.                       #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import warnings

import pytest

from mrsuite import collect
from mrsuite.datasets import openneuro


#*************#
#   fetched   #
#*************#
def _fetched(dataset: str, subject: str = "sub-01") -> bool:
    folder = openneuro.DATA / f"openneuro_{dataset}" / subject
    return folder.is_dir() and all(e.path.exists() for e in collect.discover(folder))


#**********#
#   load   #
#**********#
@pytest.mark.parametrize("dataset", sorted(openneuro.DATASETS))
def test_a_subject_loads_as_the_batches_its_dataset_holds(dataset):
    if not _fetched(dataset):
        pytest.skip(f"{dataset} sub-01 is not fetched")
    with warnings.catch_warnings(record=True) as said:
        warnings.simplefilter("always")
        batches = openneuro.load(dataset)
    entries = collect.discover(openneuro.DATA / f"openneuro_{dataset}" / "sub-01")
    left_out = sum(len(str(m.message).split(": ")[-1].split(", ")) for m in said)

    assert len(batches) == len(openneuro.DATASETS[dataset])
    assert all(len(b) for b in batches)
    assert sum(len(b) for b in batches) + left_out == len(entries)


#***************#
#   unfetched   #
#***************#
def test_a_subject_not_fetched_is_said_with_the_command(tmp_path, monkeypatch):
    root = tmp_path / "openneuro_ds005371"
    mrs = root / "sub-09" / "ses-01" / "mrs"
    mrs.mkdir(parents=True)
    for name in ("sub-09_ses-01_acq-hermespress_svs.nii.gz",
                 "sub-09_ses-01_acq-hermespress_mrsref.nii.gz"):
        (mrs / name).symlink_to(root / ".git" / "annex" / "nowhere")    # as the annex leaves it
    monkeypatch.setattr(openneuro, "DATA", tmp_path)

    (mrs / "sub-09_ses-01_T1w.nii.gz").symlink_to(root / ".git" / "annex" / "nowhere")  # no spectrum
    with pytest.raises(ValueError, match=f"2 of 2 spectra are not fetched yet: git -C {root} "
                                         "annex get sub-09"):
        openneuro.load("ds005371", "sub-09")
    with pytest.raises(ValueError, match="no OpenNeuro dataset 'ds000001'"):
        openneuro.load("ds000001")


#************#
#   unread   #
#************#
def test_a_file_spec2nii_cannot_read_is_left_out_and_named(tmp_path, monkeypatch):
    root = tmp_path / "openneuro_ds006812"
    raw = root / "sub-01" / "mrs" / "sourcedata"
    raw.mkdir(parents=True)
    names = ["sub-01_acq-01_svs_slaser_vapor7_metab_PFL.dat",
             "sub-01_acq-02_svs_slaser_vapor7_mm_PFL.dat",
             "sub-01_acq-06_svs_slaser_vapor7_metab_Occipital.dat"]
    for name in names:
        (raw / name).write_bytes(b"twix")
    monkeypatch.setattr(openneuro, "DATA", tmp_path)

    def load(entries):
        e, = entries
        if "acq-06" in e.path.name:
            raise RuntimeError("spec2nii twix failed:\nIndexError")
        return {e.name: _spectra(1)}
    monkeypatch.setattr(collect, "load", load)

    with pytest.warns(UserWarning, match="1 of 3 files could not be read by spec2nii and are "
                                         "left out: sub-01_acq-06_svs_slaser_vapor7_metab_Occipital.dat"):
        data, mm = openneuro.load("ds006812")
    assert len(data) == 1 and len(mm) == 1


def _spectra(n):
    import numpy as np
    from nifti_mrs.create_nmrs import gen_nifti_mrs
    from nifti_mrs_plus import NIfTI_MRS_Plus
    return NIfTI_MRS_Plus([gen_nifti_mrs(np.ones((1, 1, 1, 64), dtype=complex), 2.5e-4, 123.2)
                           for _ in range(n)])


#***********#
#   fetch   #
#***********#
def _snapshot(tmp_path, monkeypatch, sizes):
    """OpenNeuro as an installed MRSuite finds it: no checkout, the API listing a snapshot
    of one subject, its files served from tmp_path at the given sizes."""
    served = tmp_path / "served"
    served.mkdir()
    for name, size in sizes.items():
        (served / name).write_bytes(b"x" * size)
    tree = {None: [{"filename": "sub-01", "directory": True, "id": "s1", "size": 0, "urls": []}],
            "s1": [{"filename": "mrs", "directory": True, "id": "m1", "size": 0, "urls": []}],
            "m1": [{"filename": name, "directory": False, "id": name, "size": 1000,
                    "urls": [(served / name).as_uri()]} for name in sizes]}
    monkeypatch.setattr(openneuro, "_files", lambda dataset, at=None: tree[at])
    monkeypatch.setattr(openneuro.externals, "CHECKOUT", tmp_path / "site-packages")
    monkeypatch.setattr(openneuro, "DATA", tmp_path / "home" / "data")
    return openneuro.DATA / "openneuro_ds006812"


def test_an_install_downloads_a_subject_at_the_pinned_snapshot_once(tmp_path, monkeypatch):
    root = _snapshot(tmp_path, monkeypatch, {"a.dat": 1000, "b.dat": 1000})
    monkeypatch.setattr(collect, "discover", lambda folder: [])

    with pytest.raises(ValueError, match="no spectra found"):
        openneuro.load("ds006812")

    assert sorted(p.name for p in (root / "sub-01" / "mrs").iterdir()) == ["a.dat", "b.dat"]
    assert [p.name for p in root.iterdir()] == ["sub-01"], "no part left"
    monkeypatch.setattr(openneuro.externals, "download", None)       # never again
    with pytest.raises(ValueError, match="no spectra found"):
        openneuro.load("ds006812")
    with pytest.raises(ValueError, match="no subject 'sub-99'"):
        openneuro.load("ds006812", "sub-99")


def test_a_subject_half_downloaded_is_not_kept(tmp_path, monkeypatch):
    root = _snapshot(tmp_path, monkeypatch, {"a.dat": 1000, "b.dat": 999})

    with pytest.raises(RuntimeError, match="999 bytes, not the 1000 published"):
        openneuro.load("ds006812")
    assert not list(root.iterdir())


@pytest.mark.parametrize("dataset", sorted(openneuro.DATASETS))
def test_an_install_downloads_the_snapshot_the_submodule_pins(dataset):
    import subprocess

    root = openneuro.DATA / f"openneuro_{dataset}"
    if not (root / ".git").exists():
        pytest.skip(f"{dataset} is not a submodule here")
    tags = subprocess.run(["git", "-C", str(root), "tag", "--points-at", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.split()
    assert openneuro.TAGS[dataset] in tags
