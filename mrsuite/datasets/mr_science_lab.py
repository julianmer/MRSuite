####################################################################################################
#                                         mr_science_lab.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 26/10/26                                                                                #
#                                                                                                  #
# Purpose: The MR SCIENCE Lab's (Columbia; INSPECTOR, MARSS) public data: its MARSS-simulated     #
#          basis sets for the vendors' PRESS, sLASER and STEAM, and its INSPECTOR synthetic brain  #
#          spectra, fetched per set into MRSuite's data folder and read from INSPECTOR's .mat.     #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from nifti_mrs.create_nmrs import gen_nifti_mrs
from nifti_mrs_plus import NIfTI_MRS_Plus

from mrsuite import externals
from mrsuite.core import provenance
from mrsuite.core.basis import Basis

BASIS_SETS = "https://raw.githubusercontent.com/mr-science-lab/mrs-basis-sets/main"
SPECTRA = "https://raw.githubusercontent.com/mr-science-lab/mrs-synth-spectra/main"
DATA = externals.local("data") / "mr_science_lab"
LICENCE = "BSD-2-Clause, (c) 2023 MR SCIENCE Lab"
METABOLITES = ("2HG", "Ace", "Ala", "Asc", "Asp", "BHB", "Cho", "Cr", "Cr391", "CrNo391", "Cys",
               "Eth", "GABA", "GPC", "GSH", "Glc", "Gln", "Glu", "Gly", "Homocarnosine", "Lac",
               "NAA", "NAAG", "PA", "PCho", "PCr", "PE", "Tau", "Thr", "Try", "Tyr", "Val", "mI",
               "sI")
SEQUENCES = {"press": "PRESS", "slaser": "sLASER", "steam": "STEAM"}
VENDORS = {"siemens": "Siemens", "ge": "GE", "philips": "Philips", "bruker": "Bruker"}
# INSPECTOR's ppm reference, as its synthetic spectra state it (ppmCalib)
CENTRE = 4.6616


#***********#
#   basis   #
#***********#
def basis(sequence: str = "press", vendor: str = "siemens", te: float = 30, bandwidth: float = 4000,
          points: int = 2048, tm: Optional[float] = None,
          metabolites: Sequence[str] = METABOLITES) -> Basis:
    """
    One of the MR SCIENCE Lab's MARSS basis sets (Landheer, Swanberg & Juchem, NMR Biomed
    2019): the vendor's sequence with its exact timings and shaped pulses, 128^3 spatial
    points, read from the INSPECTOR .mat files and fetched on first use.

    Args:
        sequence (str): 'press', 'slaser' or 'steam'.
        vendor (str): 'siemens', 'ge', 'philips' or 'bruker' (Bruker: PRESS at 7 T only).
        te (float): Echo time in ms, as the repository names it (30, 35, 20, 16.5).
        bandwidth (float): 2500, 4000 or 10000 Hz, as named.
        points (int): 1024, 2048 or 4096, as named.
        tm (Optional[float]): STEAM's mixing time in ms (10).
        metabolites (Sequence[str]): Of METABOLITES; Bruker's set adds 'Water'.

    Returns:
        Basis: The functions, the water at INSPECTOR's reference; the repository, folder and
            licence in ``source``.

    Raises:
        FileNotFoundError: A set the repository does not hold, naming the URL tried.
    """
    from scipy.io import loadmat

    name = (f"SummedSpins_for_{SEQUENCES[sequence]}{VENDORS[vendor]}_TE_{_plain(te)}"
            f"{'_TM_' + _plain(tm) if tm is not None else ''}_BW_{_plain(bandwidth)}"
            f"_NPts_{int(points)}")
    folder = DATA / "basis_sets" / sequence / vendor / name
    url = f"{BASIS_SETS}/{sequence}/inspector/{vendor}/{name}"
    files = _fetched(f"the MR SCIENCE Lab {SEQUENCES[sequence]} {VENDORS[vendor]} TE {_plain(te)} "
                     f"basis set", folder, url, [f"{m}.mat" for m in metabolites])
    fids, frequency, width = [], None, None
    for file in files:
        expt = loadmat(file, squeeze_me=True, struct_as_record=False)["exptDat"]
        fids.append(np.asarray(expt.fid, dtype=complex).ravel())    # NIfTI-MRS's way round already
        frequency, width = float(expt.sf), float(expt.sw_h)
    return Basis(list(metabolites), np.array(fids), width, frequency, centre=CENTRE,
                 te=float(te), seq=SEQUENCES[sequence],
                 source={"repository": f"{BASIS_SETS}/{sequence}/inspector/{vendor}/{name}",
                         "simulator": "MARSS (Landheer, Swanberg & Juchem, NMR Biomed 2019)",
                         "format": "INSPECTOR .mat (exptDat)", "licence": LICENCE})


#**************#
#   spectrum   #
#**************#
def spectrum(model: str = "perfect", broadening: int = 8) -> NIfTI_MRS_Plus:
    """
    One of the MR SCIENCE Lab's synthetic brain spectra, simulated in INSPECTOR for its
    Monte-Carlo fit-error analysis: the perfect model, or the imperfect one with its
    macromolecules changed, at 3, 8 or 20 Hz^2 of Gaussian broadening; Siemens 3 T, 2048
    points at 4 kHz.

    Args:
        model (str): 'perfect' or 'imperfect'.
        broadening (int): 3, 8 or 20.

    Returns:
        NIfTI_MRS_Plus: The spectrum, the source in its history.
    """
    from scipy.io import loadmat

    folder = {"perfect": "PerfectModelSpectra", "imperfect": "ImperfectModelSpectra"}[model]
    name = f"GB{int(broadening)}Hz2.mat"
    file, = _fetched(f"the MR SCIENCE Lab {model} synthetic spectrum ({name})",
                     DATA / "synthetic" / folder, f"{SPECTRA}/{folder}", [name])
    mat = loadmat(file, squeeze_me=True, struct_as_record=False)
    expt = mat["exptDat"] if "exptDat" in mat else mat["lcm"].expt      # the one full worksheet
    fid = np.asarray(expt.fid, dtype=complex).ravel()
    item = gen_nifti_mrs(fid[None, None, None], 1.0 / float(expt.sw_h), float(expt.sf))
    item.add_hdr_field("SpecFreqChemShift", CENTRE)
    provenance.append(item, provenance.entry(
        "MRSuite", provenance.mrsuite_version(), "mrsuite.datasets.mr_science_lab.spectrum",
        {"repository": f"{SPECTRA}/{folder}/{name}", "simulator": "INSPECTOR", "licence": LICENCE}))
    return NIfTI_MRS_Plus([item])


#*************#
#   fetched   #
#*************#
def _fetched(what: str, folder: Path, url: str, names: Iterable[str]) -> List[Path]:
    """The files of a set, downloaded into ``folder`` the first time."""
    names = list(names)
    missing = [n for n in names if not (folder / n).is_file()]
    if missing:
        folder.mkdir(parents=True, exist_ok=True)
        with externals.fetching(what, folder):
            for n in missing:
                try:
                    externals.download(f"{url}/{n}", folder / n)
                except Exception as err:               # noqa: BLE001 - said with the URL
                    raise FileNotFoundError(f"{url}/{n}: {err}") from err
    return [folder / n for n in names]


#***********#
#   plain   #
#***********#
def _plain(value: float) -> str:
    """A number as the repository spells it: 30, 16.5, 4000."""
    return str(int(value)) if float(value).is_integer() else str(value)
