####################################################################################################
#                                        digital_phantom.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 27/10/26                                                                                #
#                                                                                                  #
# Purpose: Simulating with MRS-Digital-Phantom (van de Sande et al.): a voxel of its digital brain #
#          as a spectrum, its metabolites from tissue tables, macromolecules, lipids, water and    #
#          noise, through its own code in a process of its own (digital_phantom_driver.py).        #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from nifti_mrs.create_nmrs import gen_nifti_mrs
from nifti_mrs_plus import NIfTI_MRS_Plus

from mrsuite import externals
from mrsuite.core import provenance
from mrsuite.octave import scratch

DRIVER = Path(__file__).with_name("digital_phantom_driver.py")
REPOSITORY = "https://github.com/dennisvds/MRS-Digital-Phantom"
# the ppm its spectra are centred on
CENTRE = 4.65


#**********#
#   scan   #
#**********#
def scan(sequence: str = "PRESS", te: int = 35, size: float = 20.0,
         centre: Optional[Sequence[int]] = None, seed: int = 0,
         **params: Any) -> Tuple[NIfTI_MRS_Plus, List[Dict[str, float]]]:
    """
    One voxel of MRS-Digital-Phantom's digital brain (its MRiLab skeleton) as a spectrum:
    its metabolites from the tissue tables, macromolecules, lipids, water and noise, on one
    of the Philips basis sets it ships, with the simulation settings its config.json gives.

    Args:
        sequence (str): 'PRESS' or 'sLASER', with ``te`` naming a basis set it ships:
            PRESS at 30, 35, 68 or 120 ms, sLASER at 10, 30, 35 or 120 ms.
        te (int): The echo time in ms.
        size (float): The voxel's edge in mm.
        centre (Optional[Sequence[int]]): The voxel's centre on the skeleton's grid; the
            brain's centre of mass when None.
        seed (int): Seeds PyTorch and NumPy, which it draws from.
        **params: Any of its config.json's simulation_params, e.g. ``noise_level``; where
            it lists several values, the first is taken, as its own runs split them.

    Returns:
        Tuple[NIfTI_MRS_Plus, List[Dict[str, float]]]: The spectrum, how it was made in
        its history, and the voxel's concentrations (mM, weighted by its tissues).
    """
    fsl = "conda install -c conda-forge -c https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/"
    for module, how in (("torch", "pip install torch"), ("torchio", "pip install torchio"),
                        ("fsl_mrs", f"{fsl} fsl_mrs")):
        if importlib.util.find_spec(module) is None:
            raise ImportError(f"MRS-Digital-Phantom needs {module}: {how}")
    root = externals.ensure("mrs-digital-phantom")
    config = json.loads((root / "config.json").read_text())
    basis = f"LCModel_Universal_Philips_UnEdited_{sequence}_TE{int(te)}.BASIS"
    if not (root / "data" / "basissets" / basis).exists():
        raise ValueError(f"MRS-Digital-Phantom ships no basis set {basis}")
    first = {k: v[0] if isinstance(v, list) else v for k, v in config["simulation_params"].items()}
    settings = {**first, **params, "vendor": "Universal_Philips", "localization": sequence,
                "TE": int(te)}

    folder = scratch("mrsuite_digital_phantom_")
    try:
        job, result = os.path.join(folder, "job.json"), os.path.join(folder, "result.npz")
        with open(job, "w") as fh:
            json.dump({"skeleton": "MRiLab", "path2metabs": config["path2metabs"],
                       "path2basis": f"data/basissets/{basis}", "params": settings,
                       "centre": None if centre is None else [int(c) for c in centre],
                       "size": float(size), "seed": int(seed), "result": result}, fh)
        proc = subprocess.run([sys.executable, str(DRIVER), job], cwd=root, capture_output=True,
                              text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        if proc.returncode != 0 or not os.path.exists(result):
            raise RuntimeError(f"MRS-Digital-Phantom failed:\n{proc.stderr[-3000:]}")
        with np.load(result) as r:
            out = {k: r[k] for k in r.files}
    finally:
        shutil.rmtree(folder, ignore_errors=True)

    # its spectra are fftshifted frequency domain, the NIfTI-MRS way round
    fid = np.fft.ifft(np.fft.ifftshift(out["spectrum"]))
    item = gen_nifti_mrs(fid[None, None, None], float(out["dwell"]), float(out["frequency"]))
    item.add_hdr_field("SpecFreqChemShift", CENTRE)
    item.add_hdr_field("EchoTime", float(te) / 1e3)
    provenance.append(item, provenance.entry(
        "MRS-Digital-Phantom", externals.REGISTRY["mrs-digital-phantom"][1][:12],
        "simulation.mrs_phantom.DigitalPhantom.simulate_data",
        {"repository": REPOSITORY, "skeleton": "MRiLab", "basis": basis,
         "voxel": out["coords"].tolist(), "seed": seed, "simulation_params": settings,
         "tissue_fractions": json.loads(str(out["fractions"]))}))
    truth = dict(zip((str(n) for n in out["names"]), (float(c) for c in out["concentrations"])))
    return NIfTI_MRS_Plus([item]), [truth]
