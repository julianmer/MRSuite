####################################################################################################
#                                        mrsi_challenge.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-27                                                                              #
#                                                                                                  #
# Purpose: Loads the 2024 MRSI challenge into NIfTI-MRS+ through Augmentrum, one single-voxel      #
#          spectrum per brain voxel with its ground-truth amplitudes, and writes the basis set     #
#          the release ships (VtMeta, VtMM) as an LCModel .BASIS, which LCModel, Osprey and        #
#          FSL-MRS all read.                                                                       #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from nifti_mrs_plus import NIfTI_MRS_Plus
from mrsuite.core import provenance

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "mrsi_challenge"

# The order of metaMap's metabolite axis and of the VtMeta columns. Not alphabetical;
# verified against the named ground-truth maps and the descriptor paper's Table 1.
METABOLITES = ('asp', 'gaba', 'gln', 'glu', 'lac', 'lac41', 'naa', 'naag',
               'pe', 'pe40', 'tau', 'mins', 'sino', 'tcho', 'tcr', 'tcr39')

# What each is called in the .BASIS and in the ground truth: LCModel's own names where
# one means the same thing. LCModel's preliminary fit uses only NAA, Cr, GPC, Glu and
# Ins (or their synonyms, Cho for GPC among them) and stops if none is there, so the
# dataset's lowercase names would leave it nothing to start from.
NAMES = dict(zip(METABOLITES, ('Asp', 'GABA', 'Gln', 'Glu', 'Lac', 'Lac41', 'NAA', 'NAAG',
                               'PE', 'PE40', 'Tau', 'Ins', 'Scyllo', 'Cho', 'Cr', 'Cr39')))

# The first nine VtMM columns, by the dataset's names: LCModel's own MM names are what
# PyLCModel omits by default. The tenth column is a damped creatine reference that the
# simulation used to scale the macromolecules and never put into the signal.
MACROMOLECULES = ('mm_0_9', 'mm_1_21', 'mm_1_43', 'mm_1_67', 'mm_2_04',
                  'mm_2_26', 'mm_2_99', 'mm_3_21', 'mm_3_77')


#************#
#   module   #
#************#
def _module(data_dir: Path):
    """Augmentrum's loader, reading from the .mat.

    The .mat rather than the NIfTI files: TestSub10's spectroscopy NIfTI is mirrored in
    x against its own brain mask and B0 map, while the .mat is consistent throughout.
    """
    from augmentrum.dataset.mrsi_challenge import MRSIChallengeDataModule
    return MRSIChallengeDataModule(data_dir=str(data_dir), source='mat')


#**********#
#   load   #
#**********#
def load(subject: str, signal: str = 'all', voxels: Optional[int] = None,
         data_dir: Path = DATA_DIR) -> Tuple[NIfTI_MRS_Plus, List[Dict[str, float]], np.ndarray]:
    """
    Load one subject as single-voxel spectra, one per voxel in the brain mask.

    A subject missing from ``data_dir`` is fetched from Zenodo first.

    Args:
        subject (str): 'Sub1' ... 'Sub24', 'TestSub1' ... 'TestSub5' (track 1, with
            water and lipid) or 'TestSub10' ... 'TestSub12' (track 2, without).
        signal (str): What to load, as Augmentrum names it: 'all' is the contaminated
            data a method is given; 'meta' the noiseless metabolites alone.
        voxels (Optional[int]): How many brain voxels, spread evenly through the mask.
            All of them when None.
        data_dir (Path): Where the release lives.

    Returns:
        data (NIfTI_MRS_Plus): One spectrum per voxel.
        truth (List[Dict[str, float]]): Ground-truth amplitudes per voxel, by the basis
            names in NAMES, on the scale of the VtMeta basis.
        index (np.ndarray): (n, 3) voxel indices on the NIfTI grid, to map results back.

        Each spectrum's processing history starts with its subject, voxel and signal,
        and the file it came from.
    """
    import augmentrum

    module = _module(data_dir)
    volume = module.load([subject], signal=signal)[0][0][:]         # (x, y, z, t)
    aux = module.read_aux(subject, signal='meta')                   # test truth lives apart
    acquisition = module.read_acquisition(subject)

    index = np.argwhere(aux['brainMask'] > 0)
    if voxels is not None:
        index = index[np.linspace(0, len(index) - 1, voxels).astype(int)]

    # anything beyond the composite comes from a test subject's separate truth file
    source = module.mat_path(subject, need_truth=module.canonical_signal(signal) != 'all')
    origin = {'subject': subject, 'signal': signal, 'file': provenance.file_checksum(source),
              'reader': f'Augmentrum {augmentrum.__version__} (git '
                        f'{provenance.git_version(augmentrum.__path__[0])})'}

    spectra = []
    for x, y, z in index:
        spectrum = module.to_nifti(volume[x, y, z][None, None, None, :], acquisition)
        # the release's ppm reference, which NIfTI-MRS would otherwise take as 4.65
        spectrum.add_hdr_field('SpecFreqChemShift', acquisition['ppm_offset'])
        provenance.append(spectrum, provenance.entry(
            'MRSuite', provenance.mrsuite_version(), 'mrsuite.datasets.mrsi_challenge.load',
            {**origin, 'voxel': [int(x), int(y), int(z)]}))
        spectra.append(spectrum)
    data = NIfTI_MRS_Plus(spectra)
    truth = [dict(zip(NAMES.values(), map(float, aux['metaMap'][x, y, z]))) for x, y, z in index]
    return data, truth, index


#***********#
#   basis   #
#***********#
def basis(path: str, subject: str = 'Sub1', data_dir: Path = DATA_DIR) -> str:
    """
    Write the release's own basis set as an LCModel .BASIS file.

    The basis is VtMeta and the first nine VtMM columns from the subject's .mat, which
    every subject shares. Their rows start at t = 0 and the data at TE = 1.66 ms, two
    dwell times later, so rows 3 onwards are taken: the basis then sits on exactly the
    time points of the data, and no fitter has to absorb the delay as a phase.

    Args:
        path (str): Where to write the .BASIS.
        subject (str): Whose .mat to read the basis from.
        data_dir (Path): Where the release lives.

    Returns:
        str: The path written.
    """
    import h5py
    from lcmodel_wrapper.convert import write_basis

    module = _module(data_dir)
    acquisition = module.read_acquisition(subject)
    skip = int(round(acquisition['echo_time_s'] / acquisition['dwell_time_s']))

    with h5py.File(module.mat_path(subject, need_truth=False), 'r') as f:
        def columns(name):
            raw = f[name][()]
            return (raw['real'] + 1j * raw['imag'])[skip:skip + acquisition['n_points']]
        fids = np.concatenate([columns('VtMeta'), columns('VtMM')[:, :len(MACROMOLECULES)]],
                              axis=1)

    # write_basis takes LCModel's .RAW orientation, the conjugate of NIfTI-MRS'
    return write_basis(path, list(NAMES.values()) + list(MACROMOLECULES), list(np.conj(fids.T)),
                       dwell=acquisition['dwell_time_s'],
                       central_freq=acquisition['spectrometer_frequency_mhz'],
                       centre_ppm=acquisition['ppm_offset'])
