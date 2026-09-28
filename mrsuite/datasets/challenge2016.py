####################################################################################################
#                                         challenge2016.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-27                                                                              #
#                                                                                                  #
# Purpose: Loads the ISMRM 2016 fitting challenge into NIfTI-MRS+: the water-suppressed            #
#          spectra, their water references, and the ground-truth concentrations, paired            #
#          dataset by dataset. The spectra and basis come from the challenge submodule; the        #
#          ground truth is kept in-repo, as that repository does not carry it.                     #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fsl_mrs.utils import mrs_io

from nifti_mrs_plus import NIfTI_MRS_Plus
from mrsuite.core import provenance

DATA = Path(__file__).resolve().parents[2] / "data"
ROOT = DATA / "2016_fitting_challenge"
TRUTH = DATA / "2016_fitting_challenge_gts"
BASIS = ROOT / "basisset_LCModel" / "press3T_30ms.BASIS"


#*************#
#   listing   #
#*************#
def _listing(path: Path, suffix: Optional[str] = None) -> List[str]:
    """Sorted, suffix-filtered listing of a data directory.

    The challenge repository keeps water-suppressed and unsuppressed spectra in one
    folder (`dataset1_WS.txt` alongside `dataset1_nWS.txt`), so the two sets have to
    be separated by suffix. Sorting is by the dataset number rather than by name:
    spectra, water references and ground truth are paired positionally, and both
    `os.listdir` order and plain lexicographic order would misalign them
    (`dataset10` sorts before `dataset2`).
    """
    names = [f for f in os.listdir(path) if not f.startswith('.')]
    if suffix:
        names = [f for f in names if f.endswith(suffix)]
    return sorted(names, key=lambda f: (int(re.search(r'\d+', f).group())
                                        if re.search(r'\d+', f) else 0, f))


#***********#
#   truth   #
#***********#
def _truth(path: Path) -> Dict[str, float]:
    """Ground-truth concentrations of one dataset, from its spreadsheet."""
    import pandas as pd

    truth = {'Ace': 0.0}  # initialize, Ace is only partially present

    file = pd.read_excel(path, header=17)
    for i, met in enumerate(file['Metabolites']):
        if not isinstance(met, str): break
        truth[met] = file['concentration'][i]

    truth['Mac'] = truth.pop('MMBL')  # rename key MMBL to Mac, as the basis names it
    return dict(sorted(truth.items()))


#**********#
#   load   #
#**********#
def load(n: Optional[int] = None, root: Path = ROOT,
         truth: Path = TRUTH) -> Tuple[NIfTI_MRS_Plus, NIfTI_MRS_Plus, List[Dict[str, float]]]:
    """
    Load the challenge datasets.

    Args:
        n (Optional[int]): How many datasets, in dataset order. All 28 when None.
        root (Path): The challenge submodule.
        truth (Path): The folder of ground-truth spreadsheets.

    Returns:
        data (NIfTI_MRS_Plus): Water-suppressed spectra.
        water (NIfTI_MRS_Plus): Their unsuppressed references, one per spectrum.
        truth (List[Dict[str, float]]): Ground-truth concentrations per spectrum, keyed
            by the basis set's metabolite names where the basis has them.

        Each spectrum's processing history starts with the file it was read from.
    """
    import fsl_mrs

    spectra = Path(root) / 'datasets_JMRUI'

    def read(suffix):
        out = []
        for f in _listing(spectra, suffix)[:n]:
            spectrum = mrs_io.read_FID(str(spectra / f))
            provenance.append(spectrum, provenance.entry(
                'MRSuite', provenance.mrsuite_version(), 'mrsuite.datasets.challenge2016.load',
                {'file': spectra / f,
                 'reader': f'FSL-MRS {fsl_mrs.__version__} mrs_io.read_FID'}))
            out.append(spectrum)
        return NIfTI_MRS_Plus(out)

    concs = [_truth(Path(truth) / f) for f in _listing(truth)[:n]]
    return read('_WS.txt'), read('_nWS.txt'), concs
