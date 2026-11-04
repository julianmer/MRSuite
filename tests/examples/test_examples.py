####################################################################################################
#                                         test_examples.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-11-04                                                                              #
#                                                                                                  #
# Purpose: Every example runs as it is, at its smallest size, and writes what it says it writes;   #
#          each skips where a runtime or a dataset it needs is not here, as the suite does.        #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mrsuite import collect
from mrsuite.datasets import openneuro
from tests.bridges.test_lcmodel import needs_lcmodel
from tests.bridges.test_tarquin import needs_tarquin
from tests.conftest import needs_fsl_mrs, needs_mrsvalidate, needs_octave
from tests.datasets.test_challenges import needs_2016, needs_mrsi, needs_r, needs_sim

CHECKOUT = Path(__file__).resolve().parents[2]
COWS = openneuro.DATA / "openneuro_ds006812" / "sub-01"
TE26 = CHECKOUT / "data" / "basis_sets" / "TE26_basis_summed"

needs_cows = pytest.mark.skipif(
    not (COWS.is_dir() and all(e.path.exists() for e in collect.discover(COWS))
         and TE26.is_dir()), reason="COWS sub-01 or the TE 26 basis is not here")


#*********#
#   run   #
#*********#
def run(example: str, *args: str, timeout: int = 900) -> str:
    """An example as a user runs it, from the checkout; its output, or the failure's."""
    env = {**os.environ, "KERAS_TORCH_DEVICE": "cpu"}
    proc = subprocess.run([sys.executable, "-m", f"examples.{example}", *args], cwd=CHECKOUT,
                          env=env, capture_output=True, text=True, timeout=timeout)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return proc.stdout


#****************#
#   processing   #
#****************#
@needs_cows
@needs_octave
@needs_r
@needs_lcmodel
def test_processing_fits_one_spectrum_and_reports_it(tmp_path):
    out = run("processing", "--n", "1", "--out", str(tmp_path))

    assert "NAA" in out and "## MRSinMRS checklist" in out
    assert (tmp_path / "report.md").exists() and (tmp_path / "processed_0000.nii.gz").exists()


#*************#
#   fitters   #
#*************#
@needs_cows
@needs_octave
@needs_r
@needs_lcmodel
@needs_tarquin
@needs_fsl_mrs
def test_fitters_give_one_table_across_the_fitters(tmp_path):
    out = run("fitters", "--n", "1", "--out", str(tmp_path), timeout=1800)

    assert all(name in out for name in ("LCModel", "FSL-MRS", "spant", "TARQUIN", "Osprey"))
    assert (tmp_path / "ratios.csv").exists() and (tmp_path / "ratios.svg").exists()


#***************#
#   benchmark   #
#***************#
@needs_2016
@needs_lcmodel
@needs_r
@needs_fsl_mrs
@needs_mrsvalidate
def test_benchmark_scores_four_methods_on_a_radar(tmp_path):
    out = run("benchmark", "--n", "3", "--out", str(tmp_path))

    assert "WAND then LCModel" in out
    assert sorted(p.name for p in tmp_path.iterdir()) == ["radar.svg", "scores.csv"]


#**********#
#   mrsi   #
#**********#
@needs_mrsi
@needs_lcmodel
@needs_fsl_mrs
def test_mrsi_maps_a_few_voxels_two_ways(tmp_path):
    out = run("mrsi", "--n", "5", "--out", str(tmp_path), timeout=1800)

    assert "5 voxels by LCModel" in out
    assert (tmp_path / "maps_lcmodel.png").exists() and (tmp_path / "maps_fsl_mrsi.png").exists()


#************************#
#   simulate and train   #
#************************#
@needs_2016
@needs_sim
@needs_mrsvalidate
def test_simulate_train_learns_saves_and_scores(tmp_path):
    out = run("simulate_train", "--steps", "2", "--n", "2", "--out", str(tmp_path))

    assert "QuantMLP" in out and "MRSsynMRS" in out
    assert (tmp_path / "run" / "step-2" / "recipe.json").exists()
