####################################################################################################
#                                         test_collect.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-09                                                                              #
#                                                                                                  #
# Purpose: Data discovered and loaded: BIDS names and plain names give each file its name and     #
#          group, truth files too; images and derivatives are left out; any name may be given;    #
#          the names load in group order, spectra as batches and truth as lists.                   #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from pathlib import Path

import numpy as np
import pytest

from mrsuite import collect


#**************#
#   discover   #
#**************#
def test_files_are_told_their_name_and_group_by_their_file_names(study):
    entries = collect.discover(study)

    assert [(e.group, e.name) for e in entries] == [
        ("extra/press_PFL", "data"), ("extra/press_PFL", "water"), ("extra/press_PFL", "mm"),
        ("sub-01_ses-01_acq-press", "data"), ("sub-01_ses-01_acq-press", "water"),
        ("sub-02_acq-press", "data"), ("sub-02_acq-press", "water")]
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

    assert [(e.group, e.name, e.format, e.path.name) for e in entries] == [
        ("s1/meas", "mm", "twix", "meas_mm.dat"),
        ("s1/meas_press", "data", "twix", "meas_press.dat"),
        ("s1/meas_press", "water", "twix", "meas_press_water.dat"),
        ("s1/series", "data", "dicom", "series"),
        ("s1/sub-01_acq-02_slaser_PFL", "mm", "twix", "sub-01_acq-02_svs_slaser_mm_PFL.dat")]


def test_truth_in_a_folder_is_found_by_its_name_and_a_sidecar_is_not(study):
    (study / "sub-02" / "mrs" / "sub-02_acq-press_truth.json").write_text('{"NAA": 12.0}')
    (study / "sub-02" / "mrs" / "sub-02_acq-press_svs.json").write_text('{"EchoTime": 0.03}')

    entries = [e for e in collect.discover(study) if e.group == "sub-02_acq-press"]

    assert [(e.name, e.format, e.path.name) for e in entries] == [
        ("data", "nifti", "sub-02_acq-press_svs.nii.gz"),
        ("water", "nifti", "sub-02_acq-press_mrsref.nii.gz"),
        ("truth", "truth", "sub-02_acq-press_truth.json")]


def test_names_and_groups_can_be_set_right_by_hand(study):
    """A name that misled is put right by the file's relative path, to any name; '' drops
    the file."""
    entries = collect.discover(study, names={"extra/press_mm_PFL.nii.gz": "macro",
                                             "extra/press_water_PFL.nii.gz": ""},
                               groups={"extra/press_mm_PFL.nii.gz": "mm-run"})

    assert [(e.group, e.name) for e in entries if e.group.startswith(("extra", "mm"))] == [
        ("extra/press_PFL", "data"), ("mm-run", "macro")]


def test_files_chosen_anywhere_and_tuples_become_entries(study):
    chosen = [study / "extra/press_mm_PFL.nii.gz", study / "sub-02/mrs/sub-02_acq-press_svs.nii.gz",
              study / "nothing.txt", study / "conc.csv"]
    (study / "conc.csv").write_text("NAA\n12\n")

    found = collect.entries(chosen)

    assert [(e.group, e.name, e.format) for e in found] == [
        ("press_PFL", "mm", "nifti"), ("sub-02_acq-press", "data", "nifti"),
        ("nothing", "data", ""), ("conc", "truth", "truth")]        # a format to set, truth
    tuples = collect.as_tuples(found[:2])
    assert tuples[0] == (str(chosen[0]), "mm", "press_PFL", "nifti")
    assert collect.as_entries(tuples) == found[:2]
    assert collect.as_entries([(str(chosen[0]), "mm", "press_PFL")]) == found[:1]
    with pytest.raises(ValueError, match="nothing reads"):
        collect.as_entries([(str(chosen[2]), "data", "x")])
    with pytest.raises(ValueError, match="no name"):
        collect.as_entries([(str(chosen[0]), "", "x")])


def test_names_come_out_known_ones_first_then_as_given():
    assert collect.order(["metab", "truth", "data", "metab", "water", "extra"]) == \
        ["data", "water", "truth", "metab", "extra"]


def test_a_file_named_after_its_spectrum_groups_with_it_and_numbers_sort_by_value(tmp_path):
    """sub-1_svs_truth with sub-1_svs; sub-1_press_ref with sub-1_press, not with sub-10;
    dataset2 before dataset10, so a truth table in dataset order lines up."""
    for name in ("sub-1_svs.dat", "sub-1_svs_truth.json", "sub-10_svs.dat",
                 "sub-10_svs_truth.json", "sub-1_press.dat", "sub-1_press_ref.dat",
                 "sub-10_press.dat", "sub-10_press_ref.dat"):
        (tmp_path / "a" / name).parent.mkdir(exist_ok=True)
        (tmp_path / "a" / name).write_text("{}")
    for k in (10, 2, 1):
        (tmp_path / "b" / f"dataset{k}.dat").parent.mkdir(exist_ok=True)
        (tmp_path / "b" / f"dataset{k}.dat").write_bytes(b"")

    paired = [(e.group, e.name) for e in collect.discover(tmp_path / "a")]

    assert paired == [("sub-1", "data"), ("sub-1", "truth"), ("sub-1_press", "data"),
                      ("sub-1_press", "water"), ("sub-10", "data"), ("sub-10", "truth"),
                      ("sub-10_press", "data"), ("sub-10_press", "water")]
    assert [e.group for e in collect.discover(tmp_path / "b")] == \
        ["dataset1", "dataset2", "dataset10"]


def test_a_folder_never_runs_a_python_file_but_one_chosen_is_truth(tmp_path):
    (tmp_path / "plot_gt.py").write_text("raise SystemExit('ran')")

    assert collect.discover(tmp_path) == []
    assert [e.format for e in collect.entries([tmp_path / "plot_gt.py"])] == ["truth"]


def test_a_folder_a_format_reads_whole_is_one_entry(tmp_path):
    """A Varian .fid, a DICOM series whose files have no extension; a Philips .SPAR goes
    with its .SDAT and is read with it."""
    (tmp_path / "scan.fid").mkdir()
    (tmp_path / "scan.fid" / "fid").write_bytes(b"")
    (tmp_path / "scan.fid" / "procpar").write_text("")
    (tmp_path / "series").mkdir()
    (tmp_path / "series" / "IM0001").write_bytes(b"\0" * 128 + b"DICM")
    (tmp_path / "x.SDAT").write_bytes(b"")
    (tmp_path / "x.SPAR").write_text("")

    found = [(e.path.name, e.format) for e in collect.discover(tmp_path)]
    chosen = [(e.path.name, e.format) for e in collect.entries(sorted(tmp_path.iterdir()))]

    assert found == chosen == [("scan.fid", "varian"), ("series", "dicom"), ("x.SDAT", "philips")]


def test_raw_files_load_as_spec2nii_takes_them_and_a_reference_inside_goes_apart(tmp_path,
                                                                                 monkeypatch,
                                                                                 spectrum):
    """A Philips .SDAT with its .SPAR; a GE P-file's own reference under water, or under
    the file's name and _water when it is named otherwise."""
    from nifti_mrs_plus import NIfTI_MRS_Plus

    (tmp_path / "x.SDAT").write_bytes(b"")
    (tmp_path / "x.spar").write_text("")
    (tmp_path / "P1.7").write_bytes(b"")
    asked = []

    def load(*files, format, **options):
        asked.append(([f.name for f in files], format))
        one = NIfTI_MRS_Plus([spectrum])
        return {"P1": one, "P1_ref": one} if format == "ge" else {"x": one}
    monkeypatch.setattr(collect.io, "load", load)

    out = collect.load([(tmp_path / "x.SDAT", "data", "x"), (tmp_path / "P1.7", "metab", "p")])

    assert asked == [(["x.SDAT", "x.spar"], "philips"), (["P1.7"], "ge")]
    assert list(out) == ["data", "metab", "metab_water"]
    assert list(collect.load([(tmp_path / "P1.7", "data", "p")])) == ["data", "water"]


def test_entries_given_load_in_the_order_given(study, monkeypatch, spectrum):
    """Not sorted again: a truth table in the caller's order must stay lined up."""
    from nifti_mrs_plus import NIfTI_MRS_Plus

    read = []
    monkeypatch.setattr(collect.io, "read", lambda path: read.append(path.name) or
                        NIfTI_MRS_Plus([spectrum]))
    given = [(str(study / "sub-02/mrs/sub-02_acq-press_svs.nii.gz"), "data", "b"),
             (str(study / "extra/press_mm_PFL.nii.gz"), "data", "a")]

    collect.load(given)

    assert read == ["sub-02_acq-press_svs.nii.gz", "press_mm_PFL.nii.gz"]


#**********#
#   load   #
#**********#
def test_a_study_loads_name_by_name_in_group_order(study, spectrum):
    with pytest.warns(UserWarning, match="do not line up"):
        out = collect.load(study)

    assert list(out) == ["data", "water", "mm"]
    assert [len(v.list()) for v in out.values()] == [3, 3, 1]
    by_hand = collect.load([(str(study / "extra/press_mm_PFL.nii.gz"), "metab", "one")])
    assert list(by_hand) == ["metab"] and len(by_hand["metab"].list()) == 1
    assert np.array_equal(out["mm"].list()[0][:], spectrum[:])
    assert collect.table(collect.discover(study)).columns.tolist() == \
        ["group", "name", "format", "path"]


def test_truth_loads_as_a_list_in_group_order_beside_its_spectra(study):
    for k, sub in enumerate(("sub-01/ses-01/mrs/sub-01_ses-01", "sub-02/mrs/sub-02")):
        (study / f"{sub}_acq-press_gt.json").write_text(f'{{"NAA": {10 + k}}}')
    chosen = [e for e in collect.discover(study) if e.group.startswith("sub-")]
    assert {e.group for e in chosen if e.name == "truth"} == \
        {e.group for e in chosen if e.name == "data"}, "grouped by BIDS entities, not folder"

    out = collect.load(chosen)

    assert list(out) == ["data", "water", "truth"]
    assert out["truth"] == [{"NAA": 10.0}, {"NAA": 11.0}] and len(out["data"].list()) == 2
    with pytest.raises(ValueError, match="both spectra and ground truth"):
        collect.load([(e.path, "data", e.group, e.format) for e in chosen])


def test_a_folder_without_spectra_says_so(tmp_path):
    with pytest.raises(ValueError, match="no MRS files found"):
        collect.load(tmp_path)


#**********#
#   cows   #
#**********#
COWS = Path(__file__).resolve().parents[1] / "data" / "openneuro_ds006812" / "sub-01" / "mrs" \
    / "sourcedata"


@pytest.mark.skipif(not (COWS / "sub-01_acq-01_svs_slaser_vapor7_metab_PFL.dat").exists(),
                    reason="COWS raw data not fetched (git annex get)")
def test_a_cows_subjects_twix_files_load_by_their_names():
    """Siemens Twix through spec2nii, metabolite and macromolecule acquisitions apart."""
    entries = [e for e in collect.discover(COWS) if "acq-01_" in e.group or "acq-02_" in e.group]
    assert [e.name for e in entries] == ["data", "mm"] and all(e.format == "twix" for e in entries)

    out = collect.load(entries)

    assert list(out) == ["data", "mm"]
    svs, mm = out["data"].list()[0], out["mm"].list()[0]
    # the README's 2048 points at 4000 Hz, kept oversampled twice as Siemens stores them,
    # 32 coils, 32 averages and a loop of 2 spec2nii cannot name, as acquired
    assert svs.shape == mm.shape == (1, 1, 1, 4096, 32, 32, 2)
    assert svs.dim_tags == ["DIM_COIL", "DIM_DYN", "DIM_USER_0"]
    assert svs.dwelltime == pytest.approx(125e-6) and svs.spectrometer_frequency[0] == \
        pytest.approx(123.26, abs=0.01)
