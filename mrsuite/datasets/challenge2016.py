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


#********************#
#   mrs-sim config   #
#********************#
def mrs_sim_config(folder: Path = DATA / 'basis_sets') -> Dict:
    """
    MRS-Sim on this challenge's own basis and grid, to simulate training data from.

    The challenge's .BASIS is compiled onto the data's grid - 2048 points at 4000 Hz,
    which cuts the basis' 4096 points to the data's - into ``folder`` once, and the
    config simulates on it (see ``mrsuite.simulators.mrs_sim``). Concentrations are
    sampled relative to Cr + PCr, fixed at the ground truth's mean split; each range
    runs from 0 to 1.25 times the largest concentration in the challenge's ground
    truth, so the training data covers every dataset, the pathological ones included.
    The remaining ranges and switches follow MRS-Sim's published 30 ms config and are
    the recipe's to tune.

    Args:
        folder (Path): Where the compiled basis lives.

    Returns:
        Dict: The config, as ``mrs_sim.simulate`` and ``mrs_sim.Source`` take it.
    """
    from lcmodel_wrapper import read_basis

    from mrsuite.simulators import mrs_sim

    folder.mkdir(parents=True, exist_ok=True)
    mat = folder / 'challenge_2016.mat'
    if not mat.exists():
        mrs_sim.compile_basis(BASIS, mat, te=30.0, centre=4.65, vendor='Siemens',
                              pulse_sequence='PRESS', points=2048)
    names = [str(name) for name in read_basis(str(BASIS)).names]

    truths = [_truth(Path(TRUTH) / f) for f in _listing(TRUTH)]
    creatine = [t['Cr'] + t['PCr'] for t in truths]
    split = sum(t['Cr'] / c for t, c in zip(truths, creatine)) / len(truths)
    ranges = {m.lower(): [0.0, round(1.25 * max(t[m] / c for t, c in zip(truths, creatine)), 3)]
              for m in names if m not in ('Cr', 'PCr')}

    config = mrs_sim.config()
    config.update({
        'PM_basis_set': str(mat), 'metabolites': names,
        'wrt_metab': 'Cr,PCr', 'wrt_metab_ratio': [round(split, 3), round(1 - split, 3)],
        'snr_metab': 'Cr,PCr',
        'B0': 3.0, 'TE': 30, 'vendor': 'Siemens', 'ppm_ref': 4.65,
        'spectralwidth': 4000, 'spectrum_length': 2048, 'basis_fcn_length': 2048,
        'lineshape': 'voigt', 'num_coils': 1, 'drop_prob': 0.0,
        'b0': False, 'eddy': False, 'fshift_g': True, 'fshift_i': True,
        'phi0': True, 'phi1': True, 'noise': True, 'apodize': False, 'mm_lip': False,
        'snr': [5, 30],
        'notes': 'ISMRM 2016 fitting challenge: basis compiled from '
                 'basisset_LCModel/press3T_30ms.BASIS, cut to the data grid; '
                 'concentration ranges are 0 to 1.25x the ground truth maxima relative '
                 'to Cr+PCr, the reference fixed at the mean split.'})
    config['parameters'] = {'d': [3, 10], 'g': [5, 20], 'dmm': [10, 100], 'gmm': [5, 20],
                            'f_shift': [-0.1, 0.1], 'phi0': [-45, 45], 'phi1': [-20, 20],
                            'snr': [5, 30], **ranges}
    return config
