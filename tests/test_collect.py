####################################################################################################
#                                         test_collect.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-09                                                                              #
#                                                                                                  #
# Purpose: A study folder discovered and loaded: BIDS names and plain names give each file its     #
#          role and group; images and derivatives are left out; the roles load in group order.     #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import numpy as np
import pytest

from mrsuite import collect


#**************#
#   discover   #
#**************#
def test_files_are_told_their_role_and_group_by_their_names(study):
    entries = collect.discover(study)

    assert [(e.group, e.role) for e in entries] == [
        ("extra/press_PFL", "svs"), ("extra/press_PFL", "mrsref"), ("extra/press_PFL", "mm"),
        ("sub-01_ses-01_acq-press", "svs"), ("sub-01_ses-01_acq-press", "mrsref"),
        ("sub-02_acq-press", "svs"), ("sub-02_acq-press", "mrsref")]
    assert all(e.format == "nifti" for e in entries)
    assert not any("T1w" in str(e.path) or "derivatives" in str(e.path) for e in entries)


def test_raw_files_and_dicom_folders_are_found_by_what_spec2nii_takes(tmp_path):
    (tmp_path / "s1").mkdir()
    (tmp_path / "s1" / "T1w.nii.gz").symlink_to("not/fetched/T1w.nii.gz")   # git-annex, no content
    (tmp_path / "s1" / "meas_mm.dat").symlink_to("not/fetched/meas_mm.dat")
    (tmp_path / "s1" / "meas_press_water.dat").write_bytes(b"")
    (tmp_path / "s1" / "meas_press.dat").write_bytes(b"")
    (tmp_path / "s1" / "sub-01_acq-02_svs_slaser_mm_PFL.dat").write_bytes(b"")   # as COWS names
    (tmp_path / "s1" / "series" ).mkdir()
    (tmp_path / "s1" / "series" / "0001.dcm").write_bytes(b"")
    (tmp_path / "notes.txt").write_text("")

    entries = collect.discover(tmp_path)

    assert [(e.group, e.role, e.format, e.path.name) for e in entries] == [
        ("s1/meas", "mm", "twix", "meas_mm.dat"),
        ("s1/meas_press", "svs", "twix", "meas_press.dat"),
        ("s1/meas_press", "mrsref", "twix", "meas_press_water.dat"),
        ("s1/series", "svs", "dicom", "series"),
        ("s1/sub-01_acq-02_svs_slaser_PFL", "mm", "twix", "sub-01_acq-02_svs_slaser_mm_PFL.dat")]


def test_roles_and_groups_can_be_set_right_by_hand(study):
    """A name that misled is put right by the file's relative path; '' drops the file."""
    entries = collect.discover(study, roles={"extra/press_mm_PFL.nii.gz": "svs",
                                             "extra/press_water_PFL.nii.gz": ""},
                               groups={"extra/press_mm_PFL.nii.gz": "mm-run"})

    assert [(e.group, e.role) for e in entries if e.group.startswith(("extra", "mm"))] == [
        ("extra/press_PFL", "svs"), ("mm-run", "svs")]
    with pytest.raises(ValueError, match="no such role"):
        collect.discover(study, roles={"extra/press_mm_PFL.nii.gz": "macro"})


#**********#
#   load   #
#**********#
def test_a_study_loads_role_by_role_in_group_order(study, spectrum):
    with pytest.warns(UserWarning, match="do not line up"):
        out = collect.load(study)

    assert list(out) == ["svs", "mrsref", "mm"]
    assert [len(v.list()) for v in out.values()] == [3, 3, 1]
    assert np.array_equal(out["mm"].list()[0][:], spectrum[:])
    assert collect.table(collect.discover(study)).columns.tolist() == \
        ["group", "role", "format", "path"]


def test_a_folder_without_spectra_says_so(tmp_path):
    with pytest.raises(ValueError, match="no MRS files found"):
        collect.load(tmp_path)
