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
    origin = {'subject': subject, 'signal': signal, 'file': source,
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


#********************#
#   mrs-sim config   #
#********************#
def mrs_sim_config(folder: Optional[Path] = None,
                   subjects: Tuple[str, ...] = ('Sub1', 'Sub2'),
                   data_dir: Path = DATA_DIR) -> Dict:
    """
    MRS-Sim on this challenge's own basis and grid, to simulate training data from.

    The release's basis (see ``basis``) is written and compiled into ``folder`` once,
    and the config simulates on the data's grid: 384 points at the release's bandwidth.
    Concentrations are sampled relative to Cr + Cr39, the two moieties of the shared
    creatine map, both fixed at 1; each range runs from 0 to 1.25 times the largest
    ground-truth amplitude relative to creatine over ``subjects``' brain voxels. The
    challenge ties each moiety pair (Lac/Lac41, PE/PE40) to one amplitude, which
    MRS-Sim cannot, so the pairs are sampled independently over the same range; the
    macromolecules, whose amplitudes the release keeps fixed, get 0 to 2 at the basis'
    own scale; lactate, absent in the training subjects, gets 0 to 0.5. The remaining
    ranges and switches follow MRS-Sim's published config and are the recipe's to tune.

    Args:
        folder (Optional[Path]): Where the compiled basis lives; ``data/basis_sets``
            by default.
        subjects (Tuple[str, ...]): Whose ground truth sets the concentration ranges.
        data_dir (Path): Where the release lives.

    Returns:
        Dict: The config, as ``mrs_sim.simulate`` and ``mrs_sim.Source`` take it.
    """
    from mrsuite.simulators import mrs_sim

    module = _module(data_dir)
    acquisition = module.read_acquisition(subjects[0])

    folder = Path(folder) if folder else data_dir.parent / 'basis_sets'
    folder.mkdir(parents=True, exist_ok=True)
    mat = folder / 'mrsi_challenge_2024.mat'
    if not mat.exists():
        path = folder / 'mrsi_challenge_2024.BASIS'
        if not path.exists():
            basis(str(path), data_dir=data_dir)
        # the .BASIS is zero-filled to twice the data's points, as LCModel stores it
        mrs_sim.compile_basis(path, mat, te=1.66, centre=4.7, pulse_sequence='FID-MRSI',
                              points=acquisition['n_points'])
    ratios = []
    for subject in subjects:
        aux = module.read_aux(subject, signal='meta')
        values = aux['metaMap'][aux['brainMask'] > 0]
        creatine = values[:, METABOLITES.index('tcr')]
        keep = creatine > 0.5 * np.median(creatine[creatine > 0])
        ratios.append(values[keep] / creatine[keep, None])
    ratio = np.concatenate(ratios)

    names = list(NAMES.values()) + list(MACROMOLECULES)
    ranges = {}
    for i, met in enumerate(METABOLITES):
        if met in ('tcr', 'tcr39'):
            continue
        high = round(1.25 * float(np.percentile(ratio[:, i], 99)), 3)
        ranges[NAMES[met].lower()] = [0.0, high if high > 0 else 0.5]
    ranges.update({mm: [0.0, 2.0] for mm in MACROMOLECULES})

    config = mrs_sim.config()
    config.update({
        'PM_basis_set': str(mat), 'metabolites': names,
        'wrt_metab': 'Cr,Cr39', 'wrt_metab_ratio': [1.0, 1.0], 'snr_metab': 'Cr,Cr39',
        'B0': 3.0, 'TE': 1.66, 'vendor': '', 'ppm_ref': 4.7,
        'spectralwidth': 1 / acquisition['dwell_time_s'],
        'spectrum_length': acquisition['n_points'],
        'basis_fcn_length': acquisition['n_points'],
        'lineshape': 'voigt', 'num_coils': 1, 'drop_prob': 0.0,
        'b0': False, 'eddy': False, 'fshift_g': True, 'fshift_i': False,
        'phi0': False, 'phi1': False, 'noise': True, 'apodize': False, 'mm_lip': False,
        'snr': [5, 30],
        'notes': '2024 MRSI challenge: the release\'s basis (VtMeta and VtMM) on the '
                 'data grid; concentration ranges are 0 to 1.25x the ground truth\'s '
                 '99th percentile relative to Cr+Cr39 over the training subjects\' '
                 'brain voxels; moiety pairs sampled independently.'})
    config['parameters'] = {'d': [3, 10], 'g': [5, 20], 'dmm': [10, 100], 'gmm': [5, 20],
                            'f_shift': [-0.1, 0.1], 'phi0': [-45, 45], 'phi1': [-20, 20],
                            'snr': [5, 30], **ranges}
    return config
