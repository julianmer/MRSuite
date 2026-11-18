####################################################################################################
#                                          test_native.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-11-16                                                                              #
#                                                                                                  #
# Purpose: The fitters through MRSuite against the fitters on their own. Each tool is run as       #
#          its authors ship it - its command line, its binary, its R functions - with no           #
#          MRSuite in between, and the bridge has to give the same concentrations for the same     #
#          spectrum: one of the 2016 fitting challenge, from its files as distributed, and one     #
#          in vivo, COWS sub-01 with its published basis set. pyAMARES fits peaks, not a basis     #
#          set, so it is held to the 31P brain example it ships.                                   #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import os
import re
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from nifti_mrs.create_nmrs import gen_nifti_mrs

pytest.importorskip("fsl_mrs")     # the spectra are read, and one of them processed, by it

from mrsuite import r
from mrsuite.bridges import fid_a, fsl_mrs, lcmodel, osprey, pyamares, spant, tarquin
from mrsuite.core.basis import as_file
from mrsuite.core.pipeline import Pipeline
from mrsuite.datasets import openneuro

from tests.bridges.test_lcmodel import needs_lcmodel
from tests.bridges.test_pyamares import needs_pyamares
from tests.bridges.test_tarquin import needs_tarquin
from tests.conftest import CHALLENGE, CHALLENGE_BASIS, needs_octave
from tests.examples.test_examples import TE26, needs_cows

# the spectrum of the ``spectrum`` fixture as the challenge hands it to LCModel, and what
# its jMRUI file states of the acquisition: an LCModel .RAW states none of it
RAW = os.path.abspath("data/2016_fitting_challenge/datasets_LCModel/dataset1.RAW")
POINTS, DWELL, MHZ = 2048, 2.5e-4, 123.22

BASIS = os.path.abspath(CHALLENGE_BASIS)

needs_fsl_mrs = pytest.mark.skipif(shutil.which("fsl_mrs") is None, reason="fsl_mrs not on PATH")
needs_r = pytest.mark.skipif(not r.available(), reason="Rscript not on PATH")


#*************#
#   in vivo   #
#*************#
def in_vivo(test):
    """A test on the ``vivo`` fixture, which takes the data and what processes them."""
    for needs in (needs_cows, needs_octave, needs_r, needs_fsl_mrs):
        test = needs(test)
    return test


#**********#
#   same   #
#**********#
def same(ours: dict, own: dict, rel: float) -> None:
    """Every value the tool gave on its own is the bridge's, to ``rel`` of the largest."""
    assert len(own) > 10 and max(own.values()) > 0, "the tool's own table was not read"
    assert ours == pytest.approx(own, rel=rel, abs=rel * max(own.values()))


#*****************#
#   own fsl-mrs   #
#*****************#
def own_fsl_mrs(data: str, basis: str, folder) -> dict:
    """FSL-MRS' command line on a file: the ratios to creatine of its summary."""
    subprocess.run(["fsl_mrs", "--data", data, "--basis", basis, "--output", str(folder)],
                   check=True, capture_output=True)
    return pd.read_csv(folder / "summary.csv", index_col="Metab")["/Cr+PCr"].to_dict()


#*****************#
#   own tarquin   #
#*****************#
def own_tarquin(given: list, basis: str, out) -> dict:
    """TARQUIN's command line on a file: the amplitudes of its results."""
    from tarquin_wrapper import binaries

    subprocess.run([binaries.resolve_executable(), *given, "--basis_lcm", basis,
                    "--output_txt", str(out)], check=True, capture_output=True)
    # 'Signal  Conc (au)  %SD  SD (au)', then a row each
    rows = re.findall(r"^(\S+) +([-\d.e+]+) +\S+ +[-\d.e+]+ *$", out.read_text(), re.MULTILINE)
    return {name: float(conc) for name, conc in rows}


#*****************#
#   own lcmodel   #
#*****************#
def own_lcmodel(raw: str, points: int, dwell: float, mhz: float, basis: str, folder) -> dict:
    """The LCModel binary on a .RAW, with a control file as a user writes one: the
    concentrations of its table. Of the fit it sets what PyLCModel's own sets: the range;
    the rest, its simulated lipids and macromolecules too, is LCModel's default."""
    from lcmodel_wrapper import binaries

    control = ["$LCMODL", f"nunfil={points}", f"deltat={dwell}", f"hzpppm={mhz}",
               "ppmst=4.2", "ppmend=0.5", f"filbas='{basis}'", f"filraw='{raw}'",
               f"filtab='{folder / 'own.table'}'", "ltable=7", f"filps='{folder / 'own.ps'}'",
               "$END"]
    subprocess.run([binaries.resolve_executable()], input="\n".join(control).encode(),
                   cwd=folder, check=True, capture_output=True)
    # 'Conc.  %SD  /Cr+PCr  Metabolite', then a row each
    rows = re.findall(r"^ *([\d.E+-]+) +\d+% +[\d.E+-]+ +(\S+) *$",
                      (folder / "own.table").read_text(), re.MULTILINE)
    return {name: float(conc) for conc, name in rows}


#***************#
#   own spant   #
#***************#
def own_spant(read: str, basis: str, out, scale: str = "1") -> pd.Series:
    """spant's functions on a spectrum read by ``read`` (R), the data times ``scale``
    (R): the first row of its result table."""
    script = f"""library(spant)
                 mrs <- {read}
                 basis <- resample_basis(read_basis("{basis}"), mrs)
                 fit <- fit_mrs(scale_mrs_amp(mrs, {scale}), basis, progress = "none")
                 write.csv(fit$res_tab, "{out}")"""
    subprocess.run(["Rscript", "-e", script], check=True, capture_output=True)
    return pd.read_csv(out).iloc[0]


#**********#
#   vivo   #
#**********#
@pytest.fixture(scope="module")
def vivo(tmp_path_factory):
    """COWS sub-01's first spectrum, processed as examples/fitters.py processes it, with
    its basis set: as MRSuite holds them, and as files the tools read on their own."""
    folder = tmp_path_factory.mktemp("vivo")
    spectra, water, *_ = openneuro.load("ds006812", "sub-01")
    spectra, water = spectra[:1], water[:1]
    reference = Pipeline([fid_a.step.op_addrcvrs(), spant.step.downsample_mrs_td()]).run(water)
    data = Pipeline([fsl_mrs.step.coilcombine(reference=water),
                     spant.step.downsample_mrs_td(),
                     fid_a.step.op_alignAverages(0.2, 'n'),
                     fid_a.step.op_rmbadaverages(3.0, 't'),
                     spant.step.mean_dyns(),
                     osprey.step.op_eccKlose(reference),
                     spant.step.hsvd_filt(),
                     fid_a.step.op_ppmref(2.9, 3.1, 3.027),
                     fid_a.step.op_autophase(2.9, 3.1)]).run(spectra)[0]
    nii = str(folder / "processed.nii.gz")
    data.copy().save(nii)
    # LCModel and TARQUIN read no NIfTI-MRS: the spectrum as a .RAW, written by spant
    raw = str(folder / "processed.RAW")
    subprocess.run(["Rscript", "-e", f"""library(spant)
                    write_mrs(read_mrs("{nii}", format = "nifti"), "{raw}", format = "lcm_raw",
                              force = TRUE)"""], check=True, capture_output=True)
    basis = openneuro.basis("ds006812")
    return SimpleNamespace(data=data, basis=basis, nii=nii, raw=raw, points=data.shape[3],
                           dwell=data.dwelltime, mhz=data.spectrometer_frequency[0],
                           file=os.path.abspath(as_file(basis)))


#*************#
#   fsl-mrs   #
#*************#
@needs_fsl_mrs
def test_fsl_mrs_gives_what_its_command_line_gives(spectrum, tmp_path):
    own = own_fsl_mrs(CHALLENGE, BASIS, tmp_path / "own")

    same(fsl_mrs.fit(spectrum, CHALLENGE_BASIS).conc("internal"), own, rel=1e-9)


@in_vivo
def test_fsl_mrs_gives_what_its_command_line_gives_in_vivo(vivo, tmp_path):
    # the basis set as published, a folder of FSL-MRS' own files; the bridge fits with it
    # as a .BASIS, whose text holds six digits
    own = own_fsl_mrs(vivo.nii, str(TE26), tmp_path / "own")

    same(fsl_mrs.fit(vivo.data, vivo.basis).conc("internal"), own, rel=1e-4)


#*************#
#   tarquin   #
#*************#
@needs_tarquin
def test_tarquin_gives_what_its_command_line_gives(spectrum, tmp_path):
    # TARQUIN asks for an echo time, which the jMRUI file does not state; the challenge's
    own = own_tarquin(["--input", CHALLENGE, "--format", "jmrui_txt", "--echo", "0.03"], BASIS,
                      tmp_path / "own")

    same(tarquin.fit(spectrum, CHALLENGE_BASIS).conc("raw"), own, rel=1e-6)


@in_vivo
@needs_tarquin
def test_tarquin_gives_what_its_command_line_gives_in_vivo(vivo, tmp_path):
    # this basis has no sIns or Scyllo signal for TARQUIN to rescale it to
    own = own_tarquin(["--input", vivo.raw, "--format", "lcm", "--fs", str(1 / vivo.dwell),
                       "--ft", str(vivo.mhz * 1e6), "--echo", "0.026",
                       "--rescale_lcm_basis", "false"], vivo.file, tmp_path / "own")

    ours = tarquin.fit(vivo.data, vivo.basis, opts={"rescale_lcm_basis": False})
    same(ours.conc("raw"), own, rel=1e-4)             # the .RAW's text holds seven digits


#*************#
#   lcmodel   #
#*************#
@needs_lcmodel
def test_lcmodel_gives_what_its_binary_gives(spectrum, tmp_path):
    own = own_lcmodel(RAW, POINTS, DWELL, MHZ, BASIS, tmp_path)

    same(lcmodel.fit(spectrum, CHALLENGE_BASIS).conc("raw"), own, rel=1e-6)


@in_vivo
@needs_lcmodel
def test_lcmodel_gives_what_its_binary_gives_in_vivo(vivo, tmp_path):
    own = own_lcmodel(vivo.raw, vivo.points, vivo.dwell, vivo.mhz, vivo.file, tmp_path)

    same(lcmodel.fit(vivo.data, vivo.basis).conc("raw"), own, rel=1e-4)


#***********#
#   spant   #
#***********#
@needs_r
def test_spant_gives_what_its_functions_give(spectrum, tmp_path):
    row = own_spant(f'read_mrs("{RAW}", format = "lcm_raw", ft = {MHZ}e6, fs = {1 / DWELL}, '
                    f'ref = 4.65)', BASIS, tmp_path / "own.csv")

    ours = spant.fit(spectrum, CHALLENGE_BASIS).conc("raw")
    # the bridge fits at the basis' scale and puts the amplitudes back, which moves ABfit's
    # last digits
    same(ours, {name: float(row[name]) for name in ours}, rel=1e-3)


@in_vivo
def test_spant_gives_what_its_functions_give_in_vivo(vivo, tmp_path):
    # ABfit's answer depends on the size of the numbers it is handed: these data, of
    # order 1e-5 as the scanner wrote them, fit up to 13 % apart (PE) from the same data
    # at 1e6 times that or more, from where it no longer changes. The bridge fits at the
    # basis' scale, so spant on its own is asked at that scale too, and the ratios compared
    row = own_spant(f'read_mrs("{vivo.nii}", format = "nifti")', vivo.file, tmp_path / "own.csv",
                    scale="max(Mod(basis$data)) / max(Mod(mrs$data))")

    ours = spant.fit(vivo.data, vivo.basis).conc("internal")
    same(ours, {name: float(row[name]) / float(row["tCr"]) for name in ours}, rel=1e-3)


#**************#
#   pyamares   #
#**************#
@needs_pyamares
def test_pyamares_gives_what_its_own_example_gives():
    import pyAMARES

    # the call of pyAMARES' README, on the 7 T brain FID and prior knowledge it ships
    examples = os.path.join(os.path.dirname(pyAMARES.__file__), "examples")
    prior = os.path.join(examples, "example_human_brain_31P_7T.csv")
    fid = pyAMARES.readmrs(os.path.join(examples, "fid.txt"))
    params = pyAMARES.initialize_FID(fid=fid, priorknowledgefile=prior, MHz=120.0, sw=10000,
                                     deadtime=300e-6, preview=False, normalize_fid=False,
                                     xlim=(10, -20))
    own = pyAMARES.fitAMARES(fid_parameters=params, fitting_parameters=params.initialParams,
                             method="least_squares", ifplot=False).result_sum["amplitude"]

    data = gen_nifti_mrs(np.asarray(fid, complex).reshape(1, 1, 1, -1), 1 / 10000, 120.0,
                         nucleus="31P")
    ours = pyamares.fit(data, prior, opts={"deadtime": 300e-6, "method": "least_squares"})
    # the bridge fits at unit scale and puts the amplitudes back, which moves the last digits
    same(ours.conc("raw"), own.to_dict(), rel=1e-4)
