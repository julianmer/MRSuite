####################################################################################################
#                                          mrs4brain.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 27/10/26                                                                                #
#                                                                                                  #
# Purpose: MRS4Brain (Briand, Lanz et al., EPFL CIBM), the preclinical MRS toolbox, run as         #
#          published: a Bruker single-voxel study through its FID-A preprocessing and its LCModel  #
#          quantification, the calls of its GUI's SVS buttons, as MRSuite's uniform result. Its    #
#          licence is EPFL's, for non-commercial use: MRSuite fetches it, never ships it.          #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import os
import shutil
from typing import Optional

import numpy as np

from mrsuite.core import provenance
from mrsuite.core.result import FitResult
from mrsuite.octave import scratch
from mrsuite.octave.toolbox import Toolbox

toolbox = Toolbox("mrs4brain", "MRS4Brain", shims="mrs4brain", behind=("fid-a",),
                  within="MRS4BRAIN_toolbox")


#**************#
#   quantify   #
#**************#
def quantify(study: str, metabolite: int, water: int, basis: str, config: Optional[str] = None,
             output: Optional[str] = None) -> FitResult:
    """
    A Bruker single-voxel study through MRS4Brain, as its GUI's SVS buttons run it (Choose
    selection, Data processing, Quantification): ``create_Bruker_rawstudy``, then
    ``preprocessing_SVS_FidA`` (FID-A's alignment and outlier removal, ISIS where set) and
    ``SVS_quantification`` (LCModel, water-scaled), with MRSuite's LCModel binary.

    Its parameters come from a configuration of its own, a ``svs_param`` .mat: by default
    Others/SVS_parameters.mat, the one its GUI starts with; the shipped ones by name (e.g.
    'Demo_SVS_14p1T_SPECIAL', in Others/SVS_config); or a path. The LCModel folder and
    basis set in it are its developers' and are replaced, as its parameters window does.
    MRS4Brain's MRSI is not run here: its data2RAW does not run under Octave.

    Args:
        study (str): The Bruker study folder, holding the numbered experiments.
        metabolite (int): The water-suppressed experiment's number.
        water (int): The water reference's experiment number.
        basis (str): An LCModel .BASIS for the field and sequence.
        config (Optional[str]): A configuration, by name or path.
        output (Optional[str]): A folder to keep MRS4Brain's own output in (its LCModel
            files, the saved experiment); by default it is removed.

    Returns:
        FitResult: LCModel's water-scaled concentrations as 'molarity', its ratios to the
        configuration's reference (Rel_met, Cr+PCr in the shipped ones) as 'internal', its
        CRLBs as 'percentage', MRS4Brain's combinations
        (NAA+NAAG, Glu+Gln, ...) among the metabolites; LCModel's spectra as raw.

    Raises:
        ValueError: An MRSI experiment.
        RuntimeError: A step MRS4Brain reports failing, with what it said.
    """
    from lcmodel_wrapper import binaries

    acquired = _dimensions(os.path.join(study, str(int(metabolite)), "acqp"))
    if acquired > 1:
        raise ValueError(f"experiment {metabolite} is spatially encoded (ACQ_dim={acquired}): "
                         f"MRS4Brain's MRSI does not run under Octave, only its SVS")
    shipped = os.path.join(toolbox.root, "Others", "SVS_config", f"{config}.mat")
    config = os.path.join(toolbox.root, "Others", "SVS_parameters.mat") if config is None \
        else shipped if os.path.isfile(shipped) else config
    lcmodel = binaries.resolve_executable()
    folder = output or scratch("mrs4brain_")
    os.makedirs(folder, exist_ok=True)
    # MRS4Brain runs LCModel from its configured folder, as ./lcmodel there on macOS and by
    # the shim's mapping of its developers' path on Linux
    programs = scratch("mrs4brain_lcmodel_")
    os.symlink(lcmodel, os.path.join(programs, "lcmodel"))
    name = os.path.basename(os.path.normpath(study))
    steps = ("create_Bruker_rawstudy", "preprocessing_SVS_FidA", "SVS_quantification")
    # each step catches its own errors and hands back what went wrong, empty when nothing did
    run = "".join(f"msg = svs.{s}(prog); if any(~cellfun(@isempty, msg)), "
                  f"error('MRS4Brain {s}: %s', "
                  f"strjoin(cellfun(@char, msg, 'UniformOutput', false), '; ')); end; "
                  for s in steps)
    try:
        toolbox.run(f"""
            x = load('{config}'); f = fieldnames(x); p = x.(f{{1}});
            p.LCModel_path = '{programs}'; p.Basis_set = '{os.path.abspath(basis)}';
            mrsuite_reference = p.Rel_conc;
            s = struct('folder', '{os.path.abspath(study)}', ...
                       'expnb', {{{int(water)}, {int(metabolite)}}}, ...
                       'exp_name', {{'{name}_{int(water)}', '{name}_{int(metabolite)}'}}, ...
                       'ref_bool', {{true, false}}, 'met_bool', {{false, true}});
            prog = uiprogressdlg([], 'Title', 'SVS');
            svs = SVSClass('{folder}', 'mrsuite', s, p);
            {run}
            mrsuite_names = cellstr(svs.Met_names);
            mrsuite_abs = svs.abs_conc_tot; mrsuite_rel = svs.rel_conc_tot;
            mrsuite_crlb = svs.crlb_tot;
            mrsuite_raw = struct('ppm', svs.ppm_scale, 'data', svs.data_pts_tot, ...
                                 'fit', svs.fit_pts_tot, 'baseline', svs.baseline_pts_tot);
        """, folder)
        names = [str(n) for n in np.ravel(toolbox.octave.pull("mrsuite_names"))]
        values = {k: dict(zip(names, map(float, np.ravel(toolbox.octave.pull(f"mrsuite_{k}")))))
                  for k in ("abs", "rel", "crlb")}
        raw = toolbox.octave.pull("mrsuite_raw")
        reference = float(toolbox.octave.pull("mrsuite_reference"))
    finally:
        shutil.rmtree(programs, ignore_errors=True)
        if output is None:
            shutil.rmtree(folder, ignore_errors=True)

    # LCModel's relative column puts the reference (NAMREL, Cr+PCr) at CONREL, the config's
    # Rel_conc (8 mM in the shipped ones): divided by it, the ratios to the reference
    result = FitResult(method="mrs4brain:LCModel", metabolites=names,
                       concentrations={"molarity": values["abs"],
                                       "internal": {m: c / reference
                                                    for m, c in values["rel"].items()}},
                       uncertainties={"percentage": {m: 100 * c
                                                     for m, c in values["crlb"].items()}},
                       raw=raw)
    result.provenance = [provenance.entry(
        "MRS4Brain", toolbox.version(), ", ".join(steps),
        {"study": study, "metabolite": metabolite, "water": water, "basis": basis,
         "config": config, "lcmodel": lcmodel, "runtime": toolbox.runtime()})]
    return result


#****************#
#   dimensions   #
#****************#
def _dimensions(acqp: str) -> int:
    """How many dimensions a Bruker experiment acquired, from its acqp: 1 for a single voxel,
    more where it encoded space (MRSI)."""
    with open(acqp, errors="replace") as fh:
        for line in fh:
            if line.startswith("##$ACQ_dim="):
                return int(line.split("=")[1])
    return 1
