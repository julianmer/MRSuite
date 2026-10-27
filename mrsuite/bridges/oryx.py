####################################################################################################
#                                             oryx.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 27/10/26                                                                                #
#                                                                                                  #
# Purpose: Oryx-MRSI (Computational Imaging Lab), the MRSI map maker, run as published: an exam    #
#          whose MRSI LCModel has fitted, in Oryx's layout, through its co-registration, tissue    #
#          segmentation, quality maps, exclusion criteria and metabolite maps, the calls of its    #
#          GUI's buttons with their defaults.                                                      #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import glob
import os
import shutil
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from mrsuite import externals
from mrsuite.core import provenance
from mrsuite.octave import build_spm12
from mrsuite.octave.toolbox import Toolbox

toolbox = Toolbox("oryx-mrsi", "Oryx-MRSI", shims="oryx", front=("spm12",))

# the GUI's pre-defined maps and the ppm each is drawn at (Oryx.m, lines 948 and 949)
METABOLITES = {"Cr+PCr": 3.03, "Glu+Gln": 2.25, "GPC+PCh": 3.2, "Ins": 3.52, "Lac": 1.3,
               "Lip13a": 1.3, "Lip13b": 1.3, "Lip13a+Lip13b": 1.3, "NAA+NAAG": 2.02}


#**************************************************************************************************#
#                                            Class Maps                                            #
#**************************************************************************************************#
#                                                                                                  #
# The maps Oryx made of one exam, and how.                                                         #
#                                                                                                  #
#**************************************************************************************************#
@dataclass
class Maps:
    """The maps Oryx made of one exam, and how.

    Attributes:
        folder (str): Oryx's copy of the patient folder, which holds them.
        files (Dict[str, Dict[str, str]]): By Oryx's folder of maps (conc_maps,
            conc_maps_to_tCr_ratio, CRLB_maps, fgm_maps, SNR_maps, ...), each map's NIfTI
            by metabolite, or by what it maps where it maps all voxels at once (SNR, FWHM).
        included (Dict[str, int]): Per metabolite, the voxels the exclusion criteria kept.
        provenance (List[Dict[str, str]]): How they were made, as ProcessingApplied entries.
    """

    folder: str
    files: Dict[str, Dict[str, str]] = field(default_factory=dict)
    included: Dict[str, int] = field(default_factory=dict)
    provenance: List[Dict[str, str]] = field(default_factory=list)


#**********#
#   maps   #
#**********#
def maps(patient: str, output: str, fcsf: float = 0.3, snr: float = 8.0, fwhm: float = 0.1,
         crlb: float = 30.0) -> Maps:
    """
    One patient's MRSI exam through Oryx, as its GUI's buttons run it: Coregistration (the
    FOV and each metabolite's PRESS box on the T1), Segmentation (each voxel's grey, white
    and CSF fractions, from FSL's partial-volume images), the SNR, FWHM and CRLB maps, the
    Exclusion criteria, then the concentration maps and their ratios to tCr and Ins; with
    the GUI's defaults: its pre-defined maps, RFOV direction RL, no chemical-shift
    correction, MRSI acquired after the T1, CRLBs against NAA (Oryx.m, lines 236, 948 and
    3063 to 3093). SPM12 reads and writes the images.

    The patient folder is in Oryx's layout, as its shipped Dataset/K_01:
    ``exam_1/spectra/<name>_act.spar`` with ``.sdat``, LCModel's fits of every voxel in
    ``exam_1/spectra/lcmodel_outputs/<name>_act/``, and the T1, brain-extracted and
    segmented by FSL, in ``exam_1/images/T1/nifti/<patient>_Bet*.nii.gz``. Oryx writes its
    maps into that folder, so it works on a copy, in ``output``: some 4 GB for K_01, and
    about an hour.

    Args:
        patient (str): The patient folder.
        output (str): Where the copy, and so the maps, go.
        fcsf (float): The CSF fraction above which a voxel is excluded.
        snr (float): The SNR below which it is.
        fwhm (float): The FWHM (ppm) above which it is.
        crlb (float): The CRLB (%) above which it is.

    Returns:
        Maps: The maps, the voxels kept per metabolite, and how they were made.

    Raises:
        ValueError: A folder not in Oryx's layout.
    """
    spectra = os.path.join(patient, "exam_1", "spectra")
    acts = sorted(glob.glob(os.path.join(spectra, "*_act.spar")))
    if not acts:
        raise ValueError(f"{patient} is not in Oryx's layout: no exam_1/spectra/*_act.spar")
    name = os.path.basename(os.path.normpath(patient))
    copy = os.path.join(output, name)
    shutil.copytree(patient, copy)
    spectra = os.path.join(copy, "exam_1", "spectra")
    spar = os.path.basename(acts[0])

    build_spm12(str(externals.ensure("spm12")), runtime=toolbox._runtime)
    names = "{" + " ".join(f"'{m}'" for m in METABOLITES) + "}"
    ppm = "[" + " ".join(str(p) for p in METABOLITES.values()) + "]"
    toolbox.run(f"""
        spm('defaults', 'fmri');
        Pinfo = loadfile('{spar}', ['{spectra}' filesep]);
        Metname = {names}; ppm = {ppm};
        for n = 1:numel(Metname)
            Metabolites(n).no = n; Metabolites(n).name = Metname{{n}}; Metabolites(n).ppm = ppm(n);
        end
        RFOV_dir = 'RL';
        coreg_path = [Pinfo.spectrapath filesep 'nifti' filesep 'coreg_binary_mask'];
        if ~exist(coreg_path, 'dir'), mkdir(coreg_path); end
        Pinfo.MRI = [Pinfo.mainpath 'exam_1' filesep 'images' filesep 'T1' filesep 'nifti' ...
                     filesep Pinfo.name '_Bet.nii.gz'];
        Pinfo = Mask_FOV_run_Pinfo(Pinfo, [coreg_path filesep Pinfo.sparname '_FOV_mask.nii'], ...
                                   RFOV_dir);
        for k = 1:numel(Metabolites)
            press = [coreg_path filesep Pinfo.sparname '_' Metabolites(k).name '_PressBox_mask.nii'];
            Pinfo = Mask_Press_run_Pinfo(Pinfo, k, press, 0, 0, 0, RFOV_dir);
        end
        litoutpath = [Pinfo.spectrapath filesep 'nifti' filesep 'binary_mask_littlevoxels'];
        if ~exist(litoutpath, 'dir'), mkdir(litoutpath); end
        cs = struct('no', num2cell(zeros(1, numel(Metabolites))));
        [Pinfo, AllFWHM, AllSNR, AllNumStd, AllConc] = GUI_seg([], Metabolites, Pinfo, cs, cs, ...
            cs, 'Off', 'T1w-MRI', litoutpath, RFOV_dir, 'Pre-defined maps (Above)', '', 0);
        Map_fcsf_Generation(Pinfo, Pinfo.frac_CSF, coreg_path, Metabolites);
        Map_fgm_Generation(Pinfo, Pinfo.frac_GM, coreg_path, Metabolites);
        Map_fwm_Generation(Pinfo, Pinfo.frac_WM, coreg_path, Metabolites);
        Conc_SNR_Generation(Pinfo, AllSNR, coreg_path, Metabolites);
        Conc_FWHM_Generation(Pinfo, AllFWHM, coreg_path, Metabolites);
        Conc_CRLB_Generation(Pinfo, AllNumStd, coreg_path, Metabolites, 'NAA');
        Exc_struct.value = [{float(fcsf)} {float(snr)} {float(fwhm)} {float(crlb)}];
        Exc_struct.name = {{'fcsf', 'SNR', 'FWHM', 'CRLB'}};
        outmat = [Pinfo.mainpath 'exam_1' filesep 'spectra' filesep 'MAT'];
        if ~exist(outmat, 'dir'), mkdir(outmat); end
        includedvoxels = ExclusionCriteria(Pinfo, AllFWHM, AllSNR, AllNumStd, Metname, ...
                                           Metabolites, outmat, Exc_struct);
        Conc_Map_Generation(Pinfo, AllConc, coreg_path, Metabolites, includedvoxels);
        Conc_to_Ratio_Maps(Pinfo);
        mrsuite_included = squeeze(sum(sum(sum(includedvoxels, 2), 3), 4));
    """, copy)
    included = np.ravel(toolbox.octave.pull("mrsuite_included"))

    stem = os.path.splitext(spar)[0] + "_"
    longest = sorted(METABOLITES, key=len, reverse=True)
    files: Dict[str, Dict[str, str]] = {}
    for path in sorted(glob.glob(os.path.join(spectra, "nifti", "*_maps*", stem + "*.nii"))):
        rest = os.path.basename(path)[len(stem):-len(".nii")]
        key = next((m for m in longest if rest.startswith(m + "_")), rest)
        files.setdefault(os.path.basename(os.path.dirname(path)), {})[key] = path
    details = {"patient": patient, "fcsf": fcsf, "snr": snr, "fwhm": fwhm, "crlb": crlb,
               "metabolites": METABOLITES, "runtime": toolbox.runtime()}
    return Maps(folder=copy, files=files,
                included=dict(zip(METABOLITES, (int(n) for n in included))),
                provenance=[provenance.entry(
                    "Oryx-MRSI", toolbox.version(),
                    "Coregistration, Segmentation, ExclusionCriteria, Conc_Map_Generation, "
                    "Conc_to_Ratio_Maps", details)])
