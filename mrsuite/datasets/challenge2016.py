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

import numpy as np

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


#*********#
#   snr   #
#*********#
def snr(data: NIfTI_MRS_Plus) -> np.ndarray:
    """
    Each spectrum's SNR as the challenge defines it (Marjanska et al., MRM 2022): the
    largest value of the real spectrum over 0-4 ppm divided by the RMS noise over
    -10.6 to -7.5 ppm. On the released data it gives the paper's own values (86 for
    dataset 1, 20, 30, 43 and 158 for datasets 12, 15, 18 and 21).

    Args:
        data (NIfTI_MRS_Plus): Spectra on the challenge's grid.

    Returns:
        np.ndarray: (n,) SNRs.
    """
    item = data.list()[0]
    fids = data.numpy().reshape(len(data), -1)
    spectra = np.fft.fftshift(np.fft.fft(fids, axis=-1), axes=-1)
    ppm = item.SpecFreqChemShift + np.fft.fftshift(
        np.fft.fftfreq(fids.shape[-1], item.dwelltime)) / item.spectrometer_frequency[0]
    peak = spectra[:, (ppm > 0) & (ppm < 4)].real.max(-1)
    noise = spectra[:, (ppm > -10.6) & (ppm < -7.5)].real
    return peak / np.sqrt(np.mean((noise - noise.mean(-1, keepdims=True)) ** 2, -1))


#********************#
#   mrs-sim config   #
#********************#
def mrs_sim_config(folder: Path = DATA / 'basis_sets') -> Dict:
    """
    MRS-Sim on this challenge's own basis and grid, matched to how the challenge was made.

    The challenge's .BASIS is compiled onto the data's grid - 2048 points at 4000 Hz,
    which cuts the basis' 4096 points to the data's - into ``folder`` once. The ranges
    follow the challenge's own design (README; Marjanska et al., MRM 2022) for its 24
    brain-like datasets; the three with artifacts and the four tumour-like ones (lipids,
    acetate) are not what this config simulates:

    - concentrations relative to Cr + PCr (fixed at the datasets' mean split) over the
      ground truth's range across datasets 1-24;
    - linewidths of 6-12 Hz from a basis already 2 Hz wide: Lorentzian datasets add
      pi*(4..10) per s (MRS-Sim's d, less its own fixed 1 per s), Voigt ones a Gaussian
      of up to 10.9 Hz FWHM (g up to 422 per s^2), so d spans 0-30.4 and g 0-422;
    - SNR 20-161 as the challenge defines it (see ``snr``), converted to MRS-Sim's
      definition by simulating this config at a known SNR and measuring it the
      challenge's way;
    - no phase errors, shifts, eddy currents, baseline or residual water, as the
      brain-like datasets have none.

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

    truths = [_truth(Path(TRUTH) / f) for f in _listing(TRUTH)][:24]     # the brain-like ones
    creatine = [t['Cr'] + t['PCr'] for t in truths]
    split = sum(t['Cr'] / c for t, c in zip(truths, creatine)) / len(truths)
    ranges = {m.lower(): [round(min(t[m] / c for t, c in zip(truths, creatine)), 4),
                          round(max(t[m] / c for t, c in zip(truths, creatine)), 4)]
              for m in names if m not in ('Cr', 'PCr')}

    config = mrs_sim.config()
    for nuisance in ('baseline_cfg', 'resWater_cfg'):
        config.pop(nuisance, None)
    config.update({
        'PM_basis_set': str(mat), 'metabolites': names,
        'wrt_metab': 'Cr,PCr', 'wrt_metab_ratio': [round(split, 3), round(1 - split, 3)],
        'snr_metab': 'Cr,PCr',
        'B0': 3.0, 'TE': 30, 'vendor': 'Siemens', 'ppm_ref': 4.65,
        'spectralwidth': 4000, 'spectrum_length': 2048, 'basis_fcn_length': 2048,
        'lineshape': 'voigt', 'num_coils': 1, 'drop_prob': 0.0,
        'b0': False, 'eddy': False, 'fshift_g': False, 'fshift_i': False,
        'phi0': False, 'phi1': False, 'noise': True, 'apodize': False, 'mm_lip': False,
        'notes': 'ISMRM 2016 fitting challenge, brain-like datasets 1-24: basis compiled '
                 'from basisset_LCModel/press3T_30ms.BASIS, cut to the data grid; '
                 'concentrations over the ground truth range relative to Cr+PCr, '
                 'linewidths 6-12 Hz Lorentzian or Voigt, SNR 20-161 as the challenge '
                 'defines it; no phase, shift, eddy currents, baseline or water.'})
    config['parameters'] = {'d': [0.0, round(np.pi * 10 - 1, 2)], 'g': [0.0, 422.0],
                            'dmm': [10, 100], 'gmm': [5, 20], **ranges}

    # MRS-Sim's SNR per unit of the challenge's, measured on this config itself
    probe = {**config, 'parameters': {**config['parameters'], 'snr': [1000, 1000]}}
    measured = snr(mrs_sim.simulate(probe, n=32, seed=0, components=False)[0])
    factor = float(np.median(1000 / measured))
    config['snr'] = [round(20 * factor, 1), round(161 * factor, 1)]
    config['parameters']['snr'] = config['snr']
    return config
