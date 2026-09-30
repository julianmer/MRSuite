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

# The fixed simulation constants of the descriptor paper's Table 1 (arXiv 2609.02637):
# grey- and white-matter concentration (mM) and T2 (ms) of each basis function.
TISSUE = {'asp': (3.70, 2.07, 200, 200), 'gaba': (1.86, 0.88, 200, 200),
          'gln': (3.90, 2.20, 200, 200), 'glu': (8.60, 6.00, 200, 200),
          'lac': (0.0, 0.0, 200, 200), 'lac41': (0.0, 0.0, 200, 200),
          'naa': (10.98, 7.49, 261, 351), 'naag': (1.40, 2.60, 261, 351),
          'pe': (2.12, 2.12, 200, 200), 'pe40': (2.12, 2.12, 200, 200),
          'tau': (1.93, 1.74, 200, 200), 'mins': (4.30, 3.10, 200, 200),
          'sino': (0.50, 0.57, 200, 200), 'tcho': (1.38, 1.78, 235, 239),
          'tcr': (6.90, 5.50, 160, 177), 'tcr39': (6.90, 5.50, 160, 177)}

# The training subjects' noise SD per channel (the paper's Table 6).
NOISE_SD = 1.0e-3


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


#**********************#
#   truth parameters   #
#**********************#
def truth_parameters(subjects: Tuple[str, ...] = ('Sub1', 'Sub2'), voxels: int = 2000,
                     seed: int = 0, data_dir: Path = DATA_DIR) -> Dict[str, np.ndarray]:
    """
    Each brain voxel's simulation parameters, read off the release's own ground truth.

    The release is built as the descriptor paper (arXiv 2609.02637, Section 2) states:
    each metabolite decays with its tissue's T2 (``TISSUE``), and each voxel adds one
    decay exp(-t/T2' - t^2/TG^2) and its B0 shift. So the noiseless metabolite signal,
    divided by the metabolites with their decays and shift, leaves exactly that voxel
    factor, real to machine precision; a line through its logarithm gives T2' and TG.
    The tissue weights come from the concentrations themselves (the paper's Eq. 6).

    Only voxels with at least half the median creatine are used: at the slab's edges the
    signal vanishes and the division with it.

    Args:
        subjects (Tuple[str, ...]): The training subjects to read.
        voxels (int): Voxels to draw, evenly over the subjects' brain voxels.
        seed (int): Seeds the noise level each voxel is given (see 'snr').
        data_dir (Path): Where the release lives.

    Returns:
        Dict[str, np.ndarray]: One value per voxel (``voxels`` rounded up to a multiple
        of the subjects) in MRS-Sim's terms: each metabolite's amplitude relative to
        creatine (Lac and Lac41, zero throughout the release, and the reference left
        out), each metabolite's Lorentzian width '<name>.d' in 1/s less MRS-Sim's own
        1/s, 'snr' (the peak of Cr + Cr39's real spectrum over the time-domain noise
        SD, at the training noise level times 0.6-1.4, which the testing subjects span),
        'f_shift' in ppm, and 'TG' in s. All but 'TG' are as ``mrs_sim.distributions``
        takes them; MRS-Sim's Gaussian width is g = 1/TG^2.
    """
    import h5py

    module = _module(data_dir)
    acquisition = module.read_acquisition(subjects[0])
    frequency = acquisition['spectrometer_frequency_mhz']
    table = np.array([TISSUE[m] for m in METABOLITES])      # c_GM, c_WM, T2_GM, T2_WM
    concentrations, t2 = table[:, :2], table[:, 2:] / 1e3
    creatine = [METABOLITES.index('tcr'), METABOLITES.index('tcr39')]
    per_subject = -(-voxels // len(subjects))
    rows = []
    for subject in subjects:
        with h5py.File(module.mat_path(subject, need_truth=False), 'r') as f:
            def complex_(a):
                return a['real'] + 1j * a['imag'] if a.dtype.names else a
            skip = int(round(acquisition['echo_time_s'] / acquisition['dwell_time_s']))
            vt = complex_(f['VtMeta'][()])[skip:skip + acquisition['n_points']]
            t = np.asarray(f['t'][()]).ravel()
            maps, b0 = f['metaMap'][()], f['B0map'][()]     # the .mat as h5py sees it: (.., z, y, x)
            mask = f['brainMask'][()] > 0
            level = 0.5 * np.median(maps[creatine[0]][mask & (maps[creatine[0]] > 0)])
            where = np.argwhere(mask & (maps[creatine[0]] > level))
            for z, y, x in where[np.linspace(0, len(where) - 1, per_subject).astype(int)]:
                conc = maps[:, z, y, x]
                weights, *_ = np.linalg.lstsq(concentrations, conc, rcond=None)
                decay = (weights * concentrations)[:, None, :] * np.exp(-t[None, :, None] / t2[:, None, :])
                lines = vt * decay.sum(-1).T                # (points, metabolites)
                shift = np.exp(2j * np.pi * b0[z, y, x] * t)
                factor = (complex_(f['xtMeta'][:, z, y, x]) / (lines.sum(1) * shift)).real
                fit = slice(0, acquisition['n_points'] // 3)   # well above rounding
                (r1, r2), *_ = np.linalg.lstsq(np.stack([t[fit], t[fit] ** 2], 1),
                                               -np.log(factor[fit]), rcond=None)
                gm = weights[0] * concentrations[:, 0]
                share = np.where(conc > 0, gm / np.maximum(weights @ concentrations.T, 1e-30),
                                 weights[0] / weights.sum())
                rates = share / t2[:, 0] + (1 - share) / t2[:, 1] + r1
                peak = np.fft.fft(lines[:, creatine].sum(1) * factor).real.max()
                rows.append((conc / conc[creatine[0]], rates, peak, b0[z, y, x], r2))
    ratio, rates, peak, b0, r2 = (np.array(v) for v in zip(*rows))
    level = np.random.default_rng(seed).uniform(0.6, 1.4, len(peak))
    samples = {NAMES[m]: ratio[:, i] for i, m in enumerate(METABOLITES)
               if m not in ('tcr', 'tcr39', 'lac', 'lac41')}
    samples.update({f'{NAMES[m]}.d': rates[:, i] - 1 for i, m in enumerate(METABOLITES)})
    samples.update({'snr': peak / (NOISE_SD * level), 'f_shift': b0 / frequency,
                    'TG': 1 / np.sqrt(r2)})
    return samples


#********************#
#   mrs-sim config   #
#********************#
def mrs_sim_config(folder: Optional[Path] = None,
                   subjects: Tuple[str, ...] = ('Sub1', 'Sub2'), voxels: int = 2000,
                   seed: int = 0, data_dir: Path = DATA_DIR) -> Dict:
    """
    MRS-Sim on this challenge's own basis and grid, matched to its training data.

    The release's basis (see ``basis``) is written and compiled into ``folder`` once,
    and the config simulates on the data's grid: 384 points at the release's bandwidth.
    Every parameter is drawn uniformly, and independently, over the 1st-99th percentile
    of the training subjects' own ground truth (``truth_parameters``): each metabolite's
    amplitude relative to Cr + Cr39 (both fixed at 1, the two moieties of the one
    creatine map), the Lorentzian width (one per spectrum, shared by its lines, since the
    voxel's T2' dominates; the metabolites' own T2 differences are left out),
    the Gaussian width 1/TG^2, SNR and shift. Independent amplitudes also keep a model
    from learning the release's two fixed tissue compositions (the paper's own warning).
    MRS-Sim's fitted distributions (``mrs_sim.distributions``) are not used here: at
    MRS-Sim dffb5ed they give these bounded values heavy tails, a third of the Asp, GABA
    and Gln amplitudes far outside the release's. What the training truth does not give
    keeps a range: Lac and Lac41 (zero in the release) 0 to 0.5, the macromolecules 0 to
    2 at the basis' own scale, with the metabolites' width ranges; baseline and residual water follow MRS-Sim's published
    config. The release has no phase errors or eddy currents, so neither is simulated.

    Args:
        folder (Optional[Path]): Where the compiled basis lives; ``data/basis_sets`` by
            default. It is made once.
        subjects (Tuple[str, ...]): Whose ground truth the parameters follow.
        voxels (int): Voxels the distributions are fitted to.
        seed (int): Seeds the noise levels drawn for the SNR.
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

    samples = truth_parameters(subjects, voxels, seed, data_dir)
    span = lambda v: [round(float(np.percentile(v, 1)), 4), round(float(np.percentile(v, 99)), 4)]
    names = list(NAMES.values()) + list(MACROMOLECULES)
    ranges = {name.lower(): span(v) for name, v in samples.items()
              if '.' not in name and name not in ('snr', 'f_shift', 'TG')}
    ranges.update({'lac': [0.0, 0.5], 'lac41': [0.0, 0.5]})
    ranges.update({mm: [0.0, 2.0] for mm in MACROMOLECULES})
    widths = np.concatenate([v for k, v in samples.items() if k.endswith('.d')])

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
        'snr': span(samples['snr']), 'share_lorentzian': True,
        'notes': '2024 MRSI challenge: the release\'s basis (VtMeta and VtMM) on the '
                 'data grid; amplitudes, Lorentzian and Gaussian widths, SNR and shift '
                 'sampled uniformly over the 1st-99th percentile of the ground truth of '
                 f'{", ".join(subjects)} ({voxels} brain voxels); SNR covers 0.6-1.4x '
                 'the training noise level.'})
    config['parameters'] = {'d': span(widths), 'g': span(1 / samples['TG'] ** 2),
                            'f_shift': span(samples['f_shift']),
                            'snr': span(samples['snr']), **ranges}
    return config
