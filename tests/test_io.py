####################################################################################################
#                                            test_io.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-02                                                                              #
#                                                                                                  #
# Purpose: Raw scanner data through spec2nii into NIfTI-MRS+, on the GE P-file FID-A ships as      #
#          example data: every output, every dimension, the conversion in the history, and a       #
#          pipeline straight after. Basis sets through BasiC, on the 2016 challenge's, Osprey's    #
#          and the samples BasiC ships, compared by FSL-MRS, and merged. Ground truth from every   #
#          file it comes in.                                                                       #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

import mrsuite
from mrsuite import externals, fsl_mrs, io, r
from mrsuite.core.pipeline import Pipeline

from tests.bridges.test_tarquin import needs_tarquin
from tests.conftest import needs_octave

PFILE = (Path(__file__).resolve().parents[1] / "externals" / "fid-a" / "exampleData" / "GE"
         / "sample01_press" / "press" / "P17920.7")

needs_pfile = pytest.mark.skipif(not PFILE.exists(), reason="FID-A submodule not checked out")

BASIS_2016 = (Path(__file__).resolve().parents[1] / "data" / "2016_fitting_challenge"
              / "basisset_LCModel" / "press3T_30ms.BASIS")
SAMPLES = externals.path("basic") / "sample_basis_sets"      # BasiC's own, one folder a format
OSPREY_BASIS = (externals.path("osprey") / "fit" / "basissets" / "3T" / "siemens" / "unedited"
                / "slaser" / "30" / "basis_siemens_slaser30.mat")        # centred at 3.0 ppm

needs_basic = pytest.mark.skipif(not SAMPLES.exists(), reason="BasiC submodule not checked out")


#**********#
#   scan   #
#**********#
@pytest.fixture(scope="module")
def scan():
    return io.load(PFILE, format="ge")


#*************#
#   outputs   #
#*************#
@needs_pfile
def test_a_pfile_gives_its_spectra_and_water_with_every_dimension(scan):
    assert sorted(scan) == ["P17920", "P17920_ref"]
    data, water = scan["P17920"].list()[0], scan["P17920_ref"].list()[0]
    assert data.shape == (1, 1, 1, 4096, 32, 16)     # every coil and transient, as acquired
    assert data.dim_tags == ["DIM_COIL", "DIM_DYN", None]
    assert water.shape[-1] == 2
    assert np.abs(np.asarray(data[:])).max() > 0     # read into memory, not left on disk


#****************#
#   provenance   #
#****************#
@needs_pfile
def test_the_conversion_starts_the_history(scan):
    import spec2nii

    nifti = scan["P17920"].list()[0]
    entry = nifti.hdr_ext["ProcessingApplied"][-1]
    assert (entry["Program"], entry["Version"]) == ("spec2nii", spec2nii.__version__)
    assert json.loads(entry["Details"])["files"][0]["sha256"], "and which raw file it was"
    # the spec2nii recorded is the one that converted, not another on PATH
    assert nifti.hdr_ext["ConversionMethod"] == f"spec2nii v{spec2nii.__version__}"


#*************#
#   options   #
#*************#
@needs_pfile
def test_spec2niis_own_options_pass_through():
    """A value, a flag and a list each reach spec2nii as it takes them, and a name with
    dots keeps the spectra apart from their water."""
    options = {"fileout": "sub-01.press", "nifti1": True, "override_nucleus": ["1H"]}
    scans = io.load(PFILE, format="ge", **options)

    assert sorted(scans) == ["sub-01.press", "sub-01.press_ref"]
    nifti = scans["sub-01.press"].list()[0]
    assert nifti.shape[-1] == 16 and nifti.header["sizeof_hdr"] == 348   # NIfTI-1
    assert nifti.hdr_ext["ResonantNucleus"] == ["1H"]
    assert json.loads(nifti.hdr_ext["ProcessingApplied"][-1]["Details"])["options"] == options


#*************#
#   failure   #
#*************#
def test_a_failed_conversion_says_why():
    with pytest.raises(RuntimeError, match="UnknownPfile"):
        io.load(__file__, format="ge")


#*********#
#   voi   #
#*********#
def test_an_output_that_is_not_spectra_comes_back_as_an_image(monkeypatch):
    """DICOM's voi=True also writes a mask of where the voxel sits, a NIfTI that is not
    NIfTI-MRS. No DICOM data is at hand, so spec2nii is stood in for by its outputs."""
    import subprocess

    import nibabel as nib
    from nifti_mrs.create_nmrs import gen_nifti_mrs
    from nifti_mrs_plus import NIfTI_MRS_Plus

    def spec2nii(args, **kwargs):
        folder = Path(args[args.index("-o") + 1])
        gen_nifti_mrs(np.ones((1, 1, 1, 64), complex), 2.5e-4, 123.2,
                      nucleus="1H").save(str(folder / "svs.nii.gz"))
        nib.save(nib.Nifti1Image(np.ones((1, 1, 1), np.uint8), np.eye(4)),
                 folder / "svs_voi.nii.gz")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(io.subprocess, "run", spec2nii)
    scans = io.load("scan", format="dicom", voi=True)

    assert isinstance(scans["svs"], NIfTI_MRS_Plus)
    assert isinstance(scans["svs_voi"], nib.Nifti1Image)
    assert np.asarray(scans["svs_voi"].dataobj).sum() == 1   # read before the folder went


#*************#
#   nothing   #
#*************#
@needs_pfile
def test_a_conversion_that_writes_nothing_is_an_error():
    """spec2nii's Bruker FID mode finds nothing in this folder and exits as if it had."""
    with pytest.raises(RuntimeError, match="wrote nothing"):
        io.load(PFILE.parents[2] / "Bruker" / "sample01_press" / "press", format="bruker",
                mode="FID")


#**************#
#   pipeline   #
#**************#
@needs_pfile
def test_a_pipeline_runs_on_what_was_loaded(scan):
    out = Pipeline([fsl_mrs.step.coilcombine(), fsl_mrs.step.average(dim="DIM_DYN")]
                   ).run(scan["P17920"])

    assert out.shape == (1, 1, 1, 1, 4096)


#**********#
#   read   #
#**********#
def test_a_nifti_mrs_file_is_read_with_where_it_came_from(spectrum, tmp_path):
    from mrsuite.core import provenance

    path = tmp_path / "spectrum.nii.gz"
    spectrum.save(str(path))
    data = io.read(path, path)

    assert len(data) == 2 and np.array_equal(data[0][:], spectrum[:])
    entry = data.list()[0].hdr_ext["ProcessingApplied"][-1]
    details = json.loads(entry["Details"])
    assert entry["Method"] == "mrsuite.io.read"
    assert details["file"] == {"path": "spectrum.nii.gz",
                               "sha256": provenance.file_checksum(path)["sha256"]}


#*************#
#   samples   #
#*************#
def samples(folder: str, pattern: str = "*") -> list:
    return sorted((SAMPLES / folder).glob(pattern))


#**********#
#   home   #
#**********#
@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    """MRSuite's own folder, where a basis set's .BASIS is written, kept to the test."""
    monkeypatch.setenv("MRSUITE_HOME", str(tmp_path / "mrsuite"))
    return tmp_path / "mrsuite"


#**********#
#   peak   #
#**********#
def peak(basis, name: str) -> float:
    """Where a metabolite's largest peak sits, on the basis' own ppm axis."""
    return basis.ppm()[np.argmax(np.abs(basis.spectra()[basis.names.index(name)]))]


#***********#
#   basis   #
#***********#
@needs_basic
@pytest.mark.skipif(not BASIS_2016.exists(), reason="challenge submodule not checked out")
def test_a_basis_set_comes_through_as_it_was(home):
    """FSL-MRS reads both, so the comparison is not BasiC's own."""
    from fsl_mrs.utils.mrs_io import read_basis
    from mrsuite.core import provenance

    basis = io.basis(BASIS_2016)
    before, after = read_basis(str(BASIS_2016)), read_basis(str(basis.lcmodel()))

    assert list(after.names) == list(before.names) and len(basis.names) == 21
    assert np.array_equal(after.original_basis_array, before.original_basis_array)
    assert (basis.bandwidth, basis.frequency, basis.te) == (4000, 123.261703, -1)
    assert basis.source["format"] == "LCModel .BASIS"
    assert basis.source["files"] == [provenance.portable_file(BASIS_2016)]
    assert basis.lcmodel().parent == home / "basis"


#*************#
#   written   #
#*************#
@needs_basic
def test_a_basis_sets_basis_file_is_written_once_and_another_gets_its_own():
    jmrui = io.basis(samples("JMRUI", "*.txt"))
    path = jmrui.lcmodel()
    written = path.stat().st_mtime_ns

    again = io.basis(samples("JMRUI", "*.txt"))

    assert again.lcmodel() == path and path.stat().st_mtime_ns == written
    assert io.basis(samples("JMRUI", "NAA.txt")).lcmodel() != path


#************#
#   merged   #
#************#
def test_a_measured_spectrum_becomes_a_basis_function_on_the_basis_own_axis():
    """Measured at twice the bandwidth and points, its peak lands where it was; one with
    coils or averages left is refused."""
    from nifti_mrs.create_nmrs import gen_nifti_mrs
    from nifti_mrs_plus import NIfTI_MRS_Plus
    from mrsuite.core.basis import merged

    basis, mm = naa_basis(), mm_spectrum()

    out = io.merge_basis(basis, {"MM": NIfTI_MRS_Plus([mm])})

    assert out.names == ["NAA", "MM"] and out.fids.shape == (2, 2048)
    assert abs(out.fids[1, 10]) == pytest.approx(np.exp(-10 / 4000 / 0.02), rel=0.02), \
        "amplitude kept"
    assert peak(out, "MM") == pytest.approx(0.9, abs=0.01) and peak(out, "NAA") == \
        pytest.approx(2.01, abs=0.01)
    assert out.source["added"][0]["names"] == ["MM"] and "nifti_mrs" in out.source["added"][0]
    assert out.checksum() != basis.checksum() and basis.names == ["NAA"]
    with pytest.raises(ValueError, match="one spectrum, coil combined"):
        merged(basis, {"MM": gen_nifti_mrs(np.stack([mm[:][..., 0, :]] * 2, -1)
                                           .reshape(1, 1, 1, -1, 2), 1 / 8000, 123.2,
                                           dim_tags=["DIM_COIL", None, None])})


def test_basis_sets_merge_on_the_first_ones_axis_and_a_shared_name_is_kept_once():
    """Another set at twice the bandwidth lands where it was; a name in both is refused
    until keep says which one, and keep leaves out what it does not name."""
    from mrsuite.core.basis import Basis, merged

    basis, sf = naa_basis(), 123.2
    t = np.arange(4096) / 8000
    other = Basis(["NAA", "Cr"], np.array([np.exp(2j * np.pi * (f - 4.65) * sf * t - t / 0.05)
                                           for f in (2.5, 3.03)]), 8000.0, sf, te=30.0)

    with pytest.raises(ValueError, match="NAA in basis, other"):
        merged(basis, {"other": other})
    out = merged(basis, {"other": other}, keep={"NAA": "other", "Cr": "other"})

    assert out.names == ["NAA", "Cr"] and out.fids.shape == (2, 2048)
    decayed = np.exp(-10 / 4000 / 0.05)                     # the lines' own, 10 points in
    assert np.abs(out.fids[:, 10]) == pytest.approx([decayed] * 2, rel=0.02), "amplitude kept"
    assert peak(out, "NAA") == pytest.approx(2.5, abs=0.01)          # the other's NAA
    assert peak(out, "Cr") == pytest.approx(3.03, abs=0.01)
    assert out.source["added"] == [{"from": "other", "names": ["NAA", "Cr"], "source": {}}]
    with pytest.raises(ValueError, match="not there to keep: Cr in basis"):
        merged(basis, {"other": other}, keep={"Cr": "basis"})
    with pytest.raises(ValueError, match="'basis' names the basis set"):
        merged(basis, {"basis": other})
    with pytest.warns(UserWarning, match="TE 30.0 ms, the basis for 20.0 ms"):
        merged(Basis(["NAA"], basis.fids, 4000.0, sf, te=20.0), {"other": other},
               keep={"NAA": "basis"})


def naa_basis():
    """One function, NAA at 2.01 ppm, 2048 points at 4000 Hz."""
    from mrsuite.core.basis import Basis

    sf, t = 123.2, np.arange(2048) / 4000
    return Basis(["NAA"], np.exp(2j * np.pi * (2.01 - 4.65) * sf * t - t / 0.05)[None],
                 4000.0, sf)


def mm_spectrum():
    """A measured spectrum, a peak at 0.9 ppm, 4096 points at 8000 Hz."""
    from nifti_mrs.create_nmrs import gen_nifti_mrs

    sf, t = 123.2, np.arange(4096) / 8000
    return gen_nifti_mrs(np.exp(2j * np.pi * (0.9 - 4.65) * sf * t - t / 0.02)
                         .reshape(1, 1, 1, -1), 1 / 8000, sf)


#***********#
#   truth   #
#***********#
def test_ground_truth_is_read_from_every_file_it_comes_in(tmp_path):
    """One spectrum's, a list, by metabolite, a table, the 2016 challenge's sheet layout,
    a MATLAB struct and a Python file, each as one dictionary per spectrum."""
    import pickle

    import pandas as pd
    from scipy.io import savemat

    two = [{"NAA": 12.0, "Cr": 8.0}, {"NAA": 11.0, "Cr": 7.5}]
    (tmp_path / "one.json").write_text('{"NAA": 12, "Cr": 8}')
    (tmp_path / "list.json").write_text(json.dumps(two))
    (tmp_path / "wrapped.json").write_text(json.dumps({"truth": two, "note": "sim"}))
    (tmp_path / "columns.pkl").write_bytes(pickle.dumps({"NAA": np.array([12, 11]),
                                                         "Cr": [8, 7.5]}))
    (tmp_path / "frame.pickle").write_bytes(pickle.dumps(pd.DataFrame(two)))
    pd.DataFrame(two).assign(spectrum=["a", "b"]).to_csv(tmp_path / "wide.csv", index=False)
    notes = pd.DataFrame([["fitting output", None], ["Metabolites", "concentration"],
                          ["NAA", 12.0], ["Cr", 8.0], [None, None], ["total", 20.0]])
    notes.to_excel(tmp_path / "sheet.xlsx", header=False, index=False)
    savemat(tmp_path / "struct.mat", {"conc": {"NAA": [12.0, 11.0], "Cr": [8.0, 7.5]}})
    (tmp_path / "config.py").write_text("import math\nscale = 2\n"
                                        "truth = {'NAA': 6 * scale, 'Cr': 8.0}\n")

    for name, expected in [("one.json", two[:1]), ("list.json", two), ("wrapped.json", two),
                           ("columns.pkl", two), ("frame.pickle", two), ("wide.csv", two),
                           ("sheet.xlsx", two[:1]), ("struct.mat", two),
                           ("config.py", two[:1])]:
        assert io.truth(tmp_path / name) == expected, name


def test_truth_tables_and_variables_in_every_layout_give_the_spectra_they_hold(tmp_path):
    """A metabolite column with a concentration column per spectrum, in CSV too; a row per
    spectrum with an index and an id column left out; a variable per metabolite."""
    import pandas as pd
    from scipy.io import savemat

    two = [{"NAA": 12.0, "Cr": 8.0}, {"NAA": 11.0, "Cr": 7.5}]
    (tmp_path / "long.csv").write_text("Metabolite,Concentration\nNAA,12.0\nCr,8\n")
    (tmp_path / "columns.tsv").write_text("metabolite\tconc_1\tconc_2\nNAA\t12\t11\n"
                                          "Cr\t8\t7.5\n")
    pd.DataFrame(two).assign(subject=[1, 2]).to_csv(tmp_path / "indexed.csv")
    savemat(tmp_path / "variables.mat", {"NAA": 12.0, "Cr": 8.0})
    (tmp_path / "variables.py").write_text("NAA = 12.0\nCr = 8\n")

    for name, expected in [("long.csv", two[:1]), ("columns.tsv", two), ("indexed.csv", two),
                           ("variables.mat", two[:1]), ("variables.py", two[:1])]:
        assert io.truth(tmp_path / name) == expected, name


def test_a_table_that_cannot_be_read_right_is_refused(tmp_path):
    (tmp_path / "repeated.csv").write_text("spectrum,metabolite,concentration\n1,NAA,12\n"
                                           "2,NAA,11\n")
    (tmp_path / "bare.csv").write_text("12.0,8.0\n11.0,7.5\n")

    with pytest.raises(ValueError, match="NAA twice in one spectrum"):
        io.truth(tmp_path / "repeated.csv")
    with pytest.raises(ValueError, match="a table needs a header"):
        io.truth(tmp_path / "bare.csv")


def test_truth_that_cannot_be_told_says_what_it_found(tmp_path):
    from scipy.io import savemat

    savemat(tmp_path / "two.mat", {"a": {"NAA": 1.0}, "b": {"NAA": 2.0}})
    (tmp_path / "words.json").write_text('["NAA", "Cr"]')
    (tmp_path / "notes.txt").write_text("")

    with pytest.raises(ValueError, match="two.mat: say which holds the truth"):
        io.truth(tmp_path / "two.mat")
    assert io.truth(tmp_path / "two.mat", key="b") == [{"NAA": 2.0}]
    with pytest.raises(ValueError, match="no concentrations found: it holds list of 2"):
        io.truth(tmp_path / "words.json")
    with pytest.raises(ValueError, match="ground truth is read from"):
        io.truth(tmp_path / "notes.txt")


#***************#
#   nifti-mrs   #
#***************#
def test_nifti_mrs_files_are_read_in_as_they_are(tmp_path, spectrum):
    """Read files takes them too: no spec2nii, one batch called spectra."""
    from nifti_mrs_plus import NIfTI_MRS_Plus

    path, = NIfTI_MRS_Plus([spectrum]).save_nifti(str(tmp_path), prefix="one")
    out = io.load(path)

    assert list(out) == ["spectra"]
    assert np.array_equal(out["spectra"].list()[0][:], spectrum[:])


#*************#
#   partial   #
#*************#
def test_a_basis_file_written_halfway_is_never_left_for_a_fit(home, monkeypatch):
    """Another fit would take whatever lies at its place as the whole basis."""
    from mrsuite.core.basis import Basis

    def halfway(basis, folder, tool, output):
        (Path(folder) / "basis.BASIS").write_text(" $SEQPAR\n")
        raise RuntimeError("BasiC stopped")
    monkeypatch.setattr(io, "convert_basis", halfway)

    with pytest.raises(RuntimeError, match="BasiC stopped"):
        Basis(["NAA"], np.ones((1, 4), complex), 4000.0, 123.2).lcmodel()
    assert list((home / "basis").iterdir()) == []


#**********#
#   fits   #
#**********#
@needs_basic
@pytest.mark.skipif(not BASIS_2016.exists(), reason="challenge submodule not checked out")
@pytest.mark.parametrize("fitter", [
    "fsl_mrs", "lcmodel", pytest.param("osprey", marks=needs_octave),
    pytest.param("spant", marks=pytest.mark.skipif(not r.available(), reason="no Rscript")),
    pytest.param("tarquin", marks=needs_tarquin)])
def test_every_fit_takes_a_basis_set_as_its_basis_file(fitter, spectrum):
    """Given first, so a fit that took it for a path fails at once."""
    fit = getattr(mrsuite, fitter).fit
    basis = io.basis(BASIS_2016)

    own, written, original = (fit(spectrum, b).conc()
                              for b in (basis, basis.lcmodel(), str(BASIS_2016)))

    assert own == written
    assert list(own) == list(original) and own, "the same metabolites, from the same functions"
    if fitter != "osprey":      # Osprey's fit swings under any change of the basis' last digit
        assert np.allclose(list(own.values()), list(original.values()), rtol=0.05)


#*************#
#   formats   #
#*************#
@needs_basic
def test_a_basis_set_in_another_format_is_lcmodels_own():
    """BasiC's jMRUI and LCModel .raw samples are the same simulation: the jMRUI one, told
    apart by BasiC and written as a .BASIS, is the .raw's FID, NAA's peak at 2.01 ppm."""
    from fsl_mrs.utils.mrs_io import lcm_io, read_basis

    basis = io.basis(samples("JMRUI", "*.txt"))
    written = read_basis(str(basis.lcmodel()))
    fid = written.original_basis_array[:, list(written.names).index("NAA")]
    raw = np.asarray(lcm_io.readLCModelRaw(str(SAMPLES / "LCModel" / "raw_files" / "NAA.raw"))[0])
    raw = raw.ravel()[:len(fid)]
    scale = np.vdot(raw, fid) / np.vdot(raw, raw)

    assert len(basis.names) == 23 and basis.source["format"] == "jMRUI .txt"
    assert np.linalg.norm(fid - scale * raw) < 1e-4 * np.linalg.norm(fid)
    assert abs(peak(basis, "NAA") - 2.01) < 0.01


#************#
#   centre   #
#************#
@needs_basic
@pytest.mark.skipif(not OSPREY_BASIS.exists(), reason="Osprey submodule not checked out")
def test_a_basis_set_centred_elsewhere_keeps_its_peaks_where_they_are():
    """Osprey's own sets put 3.0 ppm at 0 Hz; written for a fit, NAA stays at 2.01."""
    from fsl_mrs.utils.mrs_io import read_basis

    basis = io.basis(OSPREY_BASIS)
    written = read_basis(str(basis.lcmodel()))
    naa = written.original_basis_array[:, list(written.names).index("NAA")]
    at = written.original_ppm_shift_axis[np.argmax(np.abs(np.fft.fftshift(np.fft.fft(naa))))]

    assert basis.centre == 3.0 and abs(peak(basis, "NAA") - 2.01) < 0.01
    assert abs(at - 2.01) < 0.01


#***********#
#   given   #
#***********#
@needs_basic
def test_what_the_files_do_not_state_is_asked_for_and_taken_as_given():
    raw = samples("LCModel/raw_files", "*.raw")             # no $SEQPAR: no sw, sf or TE

    with pytest.raises(RuntimeError, match=r"do not state sw or sf: give sw=, sf=$"):
        io.basis(raw)
    basis = io.basis(raw, sw=4000, sf=123.2, te=30)
    fsl = io.basis(samples("FSL-MRS", "*.json"))          # basis_centre is sf, in MHz

    assert (basis.bandwidth, basis.frequency, basis.te) == (4000, 123.2, 30)
    assert " HZPPPM = 123.200000," in basis.lcmodel().read_text()
    assert fsl.frequency == 123.261803 and fsl.centre is None


#************#
#   stated   #
#************#
@needs_basic
def test_what_the_files_state_wins_over_what_is_given():
    stated = io.basis(samples("LCModel/BASIS_file", "*sLASER_105_.BASIS"), sw=1000, sf=63.9,
                      te=30)
    spin = io.basis(samples("SPINWIZARD"), sw=1000, sf=63.9)   # its reader takes them too

    assert (stated.te, stated.seq) == (105, "UnEdited")
    assert stated.frequency != 63.9 and stated.bandwidth != 1000
    assert (spin.bandwidth, spin.frequency) == (4000, 123.2618)


#****************#
#   echo times   #
#****************#
@needs_basic
def test_the_echo_time_and_sequence_the_basis_functions_hold_go_along(tmp_path):
    """Osprey's states them in its functions, not where BasiC's detection looks."""
    basis = io.basis(samples("OSPREY", "*.mat"), te=30)
    converted, = io.convert_basis(basis, tmp_path / "lcm", "lcmodel", output="basis")

    assert (basis.te, basis.seq) == (26, "unedited")
    for written in (basis.lcmodel(), converted):
        assert " ECHOT = 26.0000000," in written.read_text() and " SEQ = 'unedit'," in written.read_text()


#***************#
#   ambiguous   #
#***************#
@needs_basic
def test_a_set_basic_cannot_tell_apart_names_the_formats_to_choose():
    files = samples("MRSCLOUD", "Siemens_*.mat")             # MARSS, FID-A or MRSCloud?

    with pytest.raises(RuntimeError, match="Choose one as format: 'MARSS .mat', 'FID-A .mat', "
                                           "'MRSCloud .mat'$"):
        io.basis(files)
    basis = io.basis(files, format="MRSCloud .mat")

    assert len(basis.names) == len(files) and basis.source["format"] == "MRSCloud .mat"


#*************#
#   unknown   #
#*************#
@needs_basic
def test_files_basic_cannot_read_say_so_in_one_line(tmp_path):
    with pytest.raises(RuntimeError) as err:
        io.basis(__file__)

    assert str(err.value) == ("BasiC could not read the basis set: "
                              "File format not recognized.")
    with pytest.raises(RuntimeError, match=r"^BasiC could not read the basis set: it found "
                                           r"no basis functions in the files$"):
        io.basis(__file__, format="jMRUI .txt")
    broken = tmp_path / "two\nlines.BASIS"                  # BasiC says its name, newline too
    broken.write_text("no basis")
    with pytest.raises(RuntimeError) as err:
        io.basis(broken)
    assert "\n" not in str(err.value) and "two lines.BASIS" in str(err.value)
    with pytest.raises(RuntimeError) as err:                 # raised in BasiC, not said
        io.basis(tmp_path / "gone\nhere.raw", format="LCModel .raw")
    assert "\n" not in str(err.value) and "RuntimeError: File not found:" in str(err.value)


#***********#
#   paths   #
#***********#
@needs_basic
def test_paths_are_taken_from_home_and_given_back_whole(tmp_path, monkeypatch):
    import shutil
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "home").mkdir()
    shutil.copy(SAMPLES / "JMRUI" / "NAA.txt", tmp_path / "home")

    basis = io.basis("~/NAA.txt")
    written = io.convert_basis(basis, "here", "lcmodel")

    assert basis.names == ["NAA"]
    assert written and all(p.is_absolute() and p.parent == tmp_path / "here" for p in written)
    assert not (tmp_path / "~").exists()


#*************#
#   numbers   #
#*************#
@needs_basic
def test_numbers_from_numpy_are_taken():
    raw = samples("LCModel/raw_files", "*.raw")

    basis = io.basis(raw, sw=np.float32(4000), sf=np.float64(123.2))

    assert basis.frequency == 123.2


#*************#
#   skipped   #
#*************#
@needs_basic
def test_metabolites_a_writer_skips_are_said_and_none_written_is_an_error(tmp_path,
                                                                        monkeypatch):
    """BasiC's NIfTI-MRS writer runs spec2nii, here not on the PATH; it goes on without
    each metabolite it fails, and says so only in what it prints."""
    basis = io.basis(samples("JMRUI", "[CN]*.txt")[:2])
    monkeypatch.setenv("PATH", "/usr/bin:/bin")

    with pytest.warns(UserWarning, match="BasiC: WARNING: failed for (Cho|Cr391) "), \
            pytest.raises(RuntimeError, match="wrote nothing$"):
        io.convert_basis(basis, tmp_path, "niftimrs")


#*************#
#   convert   #
#*************#
@needs_basic
def test_a_basis_set_is_written_for_the_tool_chosen(tmp_path):
    from fsl_mrs.utils.mrs_io import read_basis

    basis = io.basis(samples("JMRUI", "*.txt"))
    written = io.convert_basis(basis, tmp_path / "fsl", "fsLmrs")

    assert sorted(p.stem for p in written) == sorted(basis.names)
    assert sorted(read_basis(str(tmp_path / "fsl")).names) == sorted(basis.names)
    assert io.convert_basis(basis, tmp_path / "osprey", "osprey") == \
        [tmp_path / "osprey" / "basis.mat"]                   # a folder made for a writer


#***********#
#   tools   #
#***********#
@needs_basic
def test_a_tool_or_output_basic_does_not_write_is_told_the_others(tmp_path):
    basis = io.basis(samples("JMRUI", "NAA.txt"))

    with pytest.raises(RuntimeError, match=r"no tool 'LCModel'; one of: lcmodel, osprey, ") \
            as err:
        io.convert_basis(basis, tmp_path / "out", "LCModel")
    assert "gannet" not in str(err.value)                   # one BasiC lists but cannot write
    with pytest.raises(RuntimeError, match=r"lcmodel writes output 'default', 'both', 'basis', "
                                           r"'raw', not 'mat'$"):
        io.convert_basis(basis, tmp_path / "out", "lcmodel", output="mat")
    assert list((tmp_path / "out").iterdir()) == []


#**************#
#   bytecode   #
#**************#
@needs_basic
def test_reading_a_basis_set_leaves_basics_folder_as_it_was():
    """No bytecode, which BasiC would show as changes, here and in any test before."""
    io.basis(samples("JMRUI", "*.txt")).lcmodel()

    status = subprocess.run(["git", "-C", str(SAMPLES.parent), "status", "--porcelain",
                             "--ignored"], capture_output=True, text=True, check=True).stdout
    assert "__pycache__" not in status


##############
#   record   #
##############
@needs_basic
def test_what_the_caller_gave_a_basis_set_is_on_its_record():
    """Two bases from the same files, one with an echo time given, have different records."""
    files = samples("JMRUI", "NAA.txt")

    assert io.basis(files).source["given"] == {}
    assert io.basis(files, te=30).source["given"] == {"te": 30}


####################
#   basic commit   #
####################
@needs_basic
def test_a_basis_file_is_written_anew_when_basic_changes(monkeypatch):
    """The .BASIS is BasiC's writing: a bumped pin must not reuse the old one."""
    from mrsuite import externals

    naa = io.basis(samples("JMRUI", "NAA.txt"))
    path = naa.lcmodel()
    monkeypatch.setitem(externals.REGISTRY, "basic", (externals.REGISTRY["basic"][0], "0" * 40))

    assert naa.lcmodel() != path and naa.lcmodel().is_file()


###############
#   by hand   #
###############
def test_a_basis_built_by_hand_from_numpy_numbers_has_a_checksum():
    from mrsuite.core.basis import Basis

    fids = np.ones((1, 4), complex)
    by_hand = Basis(["NAA"], fids, np.float32(4000), np.float64(123.25), te=np.int64(30))

    assert by_hand.checksum() == Basis(["NAA"], fids, 4000.0, 123.25, te=30.0).checksum()
