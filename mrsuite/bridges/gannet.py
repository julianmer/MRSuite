####################################################################################################
#                                            gannet.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 27/10/26                                                                                #
#                                                                                                  #
# Purpose: Gannet (Mikkelsen et al.), the edited-MRS toolbox, run as published: an edited         #
#          acquisition (MEGA-PRESS, HERMES) through GannetLoad and GannetFit, and with a T1        #
#          through GannetCoRegister, GannetSegment (SPM12) and GannetQuantify, as MRSuite's        #
#          uniform result.                                                                         #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import json
import os
import shutil
from typing import Any, Dict, Optional

import numpy as np

from mrsuite import externals
from mrsuite.core import provenance
from mrsuite.core.result import FitResult
from mrsuite.octave import build_spm12, scratch
from mrsuite.octave.toolbox import Toolbox

toolbox = Toolbox("gannet", "Gannet", shims="gannet", drivers="gannet")

# Gannet's own names for what it reports per metabolite, as MRSuite files them: water-scaled
# institutional units, the ratio to creatine, then GannetQuantify's tissue corrections
SCALINGS = {"ConcIU": "water", "ConcCr": "internal", "ConcIU_TissCorr": "water_tissue_corrected",
            "ConcIU_AlphaTissCorr": "water_alpha_corrected",
            "ConcIU_AlphaTissCorr_GrpNorm": "water_alpha_corrected_group_normalised"}


#*********#
#   fit   #
#*********#
def fit(metabolite: str, water: Optional[str] = None, t1: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None, output: Optional[str] = None) -> FitResult:
    """
    One edited acquisition through Gannet, as its documentation runs it: GannetLoad and
    GannetFit; with a T1, also GannetCoRegister, GannetSegment (SPM12's segmentation) and
    GannetQuantify. Gannet reads the files itself, in any format it reads (Philips SDAT,
    GE P-file, Siemens TWIX/RDA/DICOM, NIfTI-MRS).

    Gannet takes its settings from GannetPreInitialise.m, which a user edits; here they
    are Gannet's own defaults with ``settings`` over them (MEGA-PRESS GABA+ and Glx unless
    told otherwise; ``{'HERMES': 1, 'target': ['GABAGlx', 'GSH']}`` for HERMES). Gannet
    writes its output next to the files it reads, so it reads them from a folder of their
    own. PRIAM, several files joined and phantom fits are Gannet's, but not run here.

    Args:
        metabolite (str): The water-suppressed acquisition.
        water (Optional[str]): Its water reference; without it only ratios to creatine.
        t1 (Optional[str]): A T1-weighted image (NIfTI), for tissue-corrected values.
        settings (Optional[Dict[str, Any]]): Fields of Gannet's ``MRS_struct.p`` to set.
        output (Optional[str]): A folder to keep Gannet's own output in (its PDFs, the
            voxel mask, the segmentation); by default it is removed.

    Returns:
        FitResult: Per metabolite Gannet fits (``GABA`` is GABA+, with the macromolecules
        under it), its concentrations by scaling: 'water' (i.u.), 'internal' (to Cr), and
        with a T1 the tissue-corrected ones; Gannet's fit error (%) as 'fit_error'; the
        water linewidth and the tissue fractions as QC; Gannet's ``MRS_struct.out`` as raw.

    Raises:
        ValueError: A T1 without a water reference: Gannet quantifies against water.
    """
    if t1 is not None and water is None:
        raise ValueError("Gannet corrects for tissue against water: give the water reference "
                         "with the T1")
    folder = output or scratch("gannet_")
    os.makedirs(folder, exist_ok=True)
    given = {"metabolite": metabolite, "water": water, "t1": t1}
    files = {k: _beside(v, folder) for k, v in given.items() if v is not None}
    with open(os.path.join(folder, "settings.json"), "w") as fh:
        json.dump(settings or {}, fh)

    # Gannet runs a batch, and skips a dataset that fails with a report in a dialog; the one
    # dataset here failing is said as an error, with the message it failed with
    stages = ["GannetLoad", "GannetFit"]
    reference = f", {{'{files['water']}'}}" if water else ""
    code = (f"MRS = GannetLoad(true, 'settings.json', {{'{files['metabolite']}'}}{reference}); "
            f"if ~isfield(MRS, 'spec'), error('GannetLoad failed: %s', lasterr()); end; "
            f"MRS = GannetFit(MRS); ")
    if t1 is not None:
        spm = build_spm12(str(externals.ensure("spm12")), runtime=toolbox._runtime)
        toolbox.octave.eval(f"if ~exist('spm', 'file'), addpath('{spm}'); end")
        stages += ["GannetCoRegister", "GannetSegment", "GannetQuantify"]
        # GannetSegment fills MRS.mask.<voxel>.fwd_def{ii,:}, a row per scan, which MATLAB
        # grows on assignment and Octave needs made first
        code += (f"MRS = GannetCoRegister(MRS, {{'{files['t1']}'}}); "
                 f"MRS.mask.(MRS.p.vox{{1}}).fwd_def = cell(MRS.p.numScans, 1); "
                 f"MRS = GannetSegment(MRS); MRS = GannetQuantify(MRS); ")
    code += "mrsuite_out = MRS.out.(MRS.p.vox{1});"
    try:
        toolbox.run(code, folder)
        out = toolbox.octave.pull("mrsuite_out")
    finally:
        if output is None:
            shutil.rmtree(folder, ignore_errors=True)

    details = {**given, "settings": settings or {}, "runtime": toolbox.runtime()}
    if t1 is not None:
        details["spm12"] = provenance.git_version(str(externals.ensure("spm12")))
    result = to_result(out)
    if not result.metabolites:
        raise RuntimeError(f"Gannet fitted nothing in {metabolite}")
    result.provenance = [provenance.entry("Gannet", toolbox.version(), ", ".join(stages),
                                          details)]
    return result


#************#
#   beside   #
#************#
def _beside(path: str, folder: str) -> str:
    """A file linked into ``folder``, which Gannet then writes beside instead of beside the
    original, with the files that go with it (an SDAT's SPAR, a NIfTI's JSON sidecar): every
    file of its folder whose name starts as its own does, up to the first dot. The links keep
    their names, which Gannet reads the format from."""
    source = os.path.dirname(os.path.abspath(path))
    stem = os.path.basename(path).split(".")[0]
    for name in os.listdir(source):
        link = os.path.join(folder, name)
        if name.split(".")[0] == stem and not os.path.lexists(link):
            os.symlink(os.path.join(source, name), link)
    return os.path.join(folder, os.path.basename(path))


#***************#
#   to result   #
#***************#
def to_result(out: Dict[str, Any], method: str = "gannet:Gannet") -> FitResult:
    """
    Gannet's output for one voxel, ``MRS_struct.out.<voxel>``, as MRSuite's result.

    Args:
        out (Dict[str, Any]): What Gannet reports per metabolite and of the water and tissue.
        method (str): Recorded on the result.

    Returns:
        FitResult: See ``fit``.
    """
    def scalar(value):
        return float(np.asarray(value, dtype=float).ravel()[0])

    metabolites = [m for m, v in out.items()
                   if isinstance(v, dict) and any(k in v for k in SCALINGS)]
    concentrations = {name: {m: scalar(out[m][key]) for m in metabolites if key in out[m]}
                      for key, name in SCALINGS.items()
                      if any(key in out[m] for m in metabolites)}
    errors = {m: scalar(out[m]["FitError"]) for m in metabolites if "FitError" in out[m]}
    qc = {}
    if isinstance(out.get("water"), dict) and "FWHM" in out["water"]:
        qc["water_fwhm"] = scalar(out["water"]["FWHM"])
    for fraction in ("fGM", "fWM", "fCSF"):
        if isinstance(out.get("tissue"), dict) and fraction in out["tissue"]:
            qc[fraction] = scalar(out["tissue"][fraction])
    return FitResult(method=method, metabolites=metabolites, concentrations=concentrations,
                     uncertainties={"fit_error": errors} if errors else {}, qc=qc, raw=out)
