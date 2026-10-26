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
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from nifti_mrs_plus import NIfTI_MRS_Plus
from mrsuite import externals
from mrsuite.core import provenance
from mrsuite.training import Source, Truth

DATA_DIR = externals.local("data") / "mrsi_challenge"

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

# The release's 24 training subjects, as Augmentrum splits them: 19 to learn from, 5 to
# validate on; the test subjects of both tracks stay apart.
TRAIN = tuple(f'Sub{i}' for i in range(1, 20))
VALIDATION = tuple(f'Sub{i}' for i in range(20, 25))


#************#
#   module   #
#************#
def _module(data_dir: Path, *subjects: str):
    """Augmentrum's loader, reading from the .mat, with ``subjects`` fetched from Zenodo
    first where they are missing, said while they download (Augmentrum says their size).

    The .mat rather than the NIfTI files: TestSub10's spectroscopy NIfTI is mirrored in
    x against its own brain mask and B0 map, while the .mat is consistent throughout.
    """
    try:
        from augmentrum.dataset.mrsi_challenge import MRSIChallengeDataModule
    except ModuleNotFoundError as err:
        if (err.name or "").partition(".")[0] != "augmentrum":   # installed, but broken
            raise
        raise ImportError("Augmentrum is not installed: with FSL-MRS installed, pip install "
                          "git+https://github.com/julianmer/Augmentrum.git") from err
    module = MRSIChallengeDataModule(data_dir=str(data_dir), source='mat')
    missing = [s for s in subjects if not module.available(s)]
    if missing:
        with externals.fetching(f"the MRSI challenge's {', '.join(missing)} from Zenodo",
                                Path(data_dir), say=False):
            module.fetch(missing, str(data_dir))
    return module


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
    module = _module(data_dir, subject)
    import augmentrum

    volume = module.load([subject], signal=signal)[0][0][:]         # (x, y, z, t)
    aux = module.read_aux(subject, signal='meta')                   # test truth lives apart
    acquisition = module.read_acquisition(subject)

    index = _spread(np.argwhere(aux['brainMask'] > 0), voxels)

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


#************#
#   volume   #
#************#
def volume(subject: str, signal: str = 'all', z: Optional[int] = None,
           data_dir: Path = DATA_DIR) -> Tuple[NIfTI_MRS_Plus, Dict[str, np.ndarray], np.ndarray]:
    """
    Load one subject as it is: one spatial NIfTI-MRS of the whole volume, or of one axial
    slice, for the MRSI steps; with the truth as maps and the brain mask.

    Args:
        subject (str): As ``load`` takes it.
        signal (str): As ``load`` takes it; 'all-nuisance' leaves the water and lipids out.
        z (Optional[int]): One slice (0 to 31) rather than the whole volume.
        data_dir (Path): Where the release lives.

    Returns:
        data (NIfTI_MRS_Plus): The volume or slice, x by y by z voxels of one spectrum each.
        truth (Dict[str, np.ndarray]): The ground-truth amplitude map per basis name, on
            the same grid, on the scale of the VtMeta basis.
        mask (np.ndarray): The brain mask on the same grid.
    """
    module = _module(data_dir, subject)
    import augmentrum

    whole = module.load([subject], signal=signal)[0][0]
    aux = module.read_aux(subject, signal='meta')
    acquisition = module.read_acquisition(subject)
    sl = slice(None) if z is None else slice(z, z + 1)
    item = module.to_nifti(np.asarray(whole[:])[:, :, sl], acquisition) if z is not None \
        else whole
    item.add_hdr_field('SpecFreqChemShift', acquisition['ppm_offset'])
    source = module.mat_path(subject, need_truth=module.canonical_signal(signal) != 'all')
    provenance.append(item, provenance.entry(
        'MRSuite', provenance.mrsuite_version(), 'mrsuite.datasets.mrsi_challenge.volume',
        {'subject': subject, 'signal': signal, 'z': z, 'file': source,
         'reader': f'Augmentrum {augmentrum.__version__} (git '
                   f'{provenance.git_version(augmentrum.__path__[0])})'}))
    truth = {name: aux['metaMap'][:, :, sl, k] for k, name in enumerate(NAMES.values())}
    return NIfTI_MRS_Plus([item]), truth, aux['brainMask'][:, :, sl] > 0


#*****************#
#   voxel index   #
#*****************#
def voxel_index(subject: str, voxels: Optional[int] = None,
                data_dir: Path = DATA_DIR) -> np.ndarray:
    """(n, 3) indices of a subject's brain voxels on the NIfTI grid, all of them or an
    evenly spread few: the voxels load() gives, without loading them."""
    aux = _module(data_dir, subject).read_aux(subject, signal='meta')
    return _spread(np.argwhere(aux['brainMask'] > 0), voxels)


#************#
#   spread   #
#************#
def _spread(index: np.ndarray, voxels: Optional[int]) -> np.ndarray:
    """All of the index, or an evenly spread few."""
    if voxels is None:
        return index
    return index[np.linspace(0, len(index) - 1, voxels).astype(int)]


#**************************************************************************************************#
#                                          Class Release                                           #
#**************************************************************************************************#
#                                                                                                  #
# The release's own training subjects as a training source: brain voxels drawn at random.          #
#                                                                                                  #
#**************************************************************************************************#
class Release(Source):
    """The release's own training subjects as a training source: brain voxels drawn at random.

    Each spectrum is one brain voxel of one of ``subjects``, drawn uniformly over all their
    brain voxels, as the release simulated it: ``signal`` 'all' keeps its water and lipids
    (as track 1), 'all-nuisance' takes them out (as track 2). The truth is the voxel's
    metaMap, the amplitudes of the release's basis; with ``components`` also the voxel's
    metabolites and, where the signal holds them, its water and lipids on their own
    ('metabolites', 'nuisance'). The training subjects ship nothing else apart: their
    macromolecules, baseline and noise are only in the sum.

    Each subject's brain voxels are kept once per signal as a (voxels, points) array in
    ``data_dir/_mrsuite_cache`` and read memory-mapped, so a batch reads its own voxels
    alone and runs learning side by side share them.

        source = mrsi_challenge.Release(mrsi_challenge.TRAIN, 'all-nuisance')
        mlp.learn(source, steps=10_000, objective='scaled')

    Args:
        subjects (Sequence[str]): Training subjects to draw from; TRAIN by default.
        signal (str): 'all' or 'all-nuisance'.
        components (bool): Give the metabolites and nuisance on their own, as WAND learns
            them.
        seed (int): Seeds which voxels are drawn.
        data_dir (Path): Where the release lives.
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, subjects: Sequence[str] = TRAIN, signal: str = 'all',
                 components: bool = False, seed: int = 0, data_dir: Path = DATA_DIR) -> None:
        if signal not in ('all', 'all-nuisance'):
            raise ValueError(f"signal is 'all' or 'all-nuisance', not {signal!r}")
        self.subjects, self.signal, self.components = tuple(subjects), signal, components
        self.seed, self.data_dir = seed, Path(data_dir)
        self.rng = np.random.default_rng(seed)
        module = _module(data_dir, *self.subjects)
        parts = [signal] + (['meta'] + (['nuisance'] if signal == 'all' else [])
                            if components else [])
        self._voxels = {part: [_brain(module, s, part, self.data_dir) for s in self.subjects]
                        for part in parts}
        aux = [module.read_aux(s, signal='meta') for s in self.subjects]
        self._truth = [a['metaMap'][a['brainMask'] > 0] for a in aux]
        self._doi = module.DOI
        self._starts = np.cumsum([0] + [len(t) for t in self._truth])
        acquisition = module.read_acquisition(self.subjects[0])
        self.acquisition = {'points': acquisition['n_points'],
                            'bandwidth': 1 / acquisition['dwell_time_s'],
                            'frequency': acquisition['spectrometer_frequency_mhz'],
                            'centre': acquisition['ppm_offset']}
        self.dim_tags = []

    #**********#
    #   draw   #
    #**********#
    def draw(self, n: int) -> Tuple[np.ndarray, Truth]:
        drawn = self.rng.integers(self._starts[-1], size=n)
        subject = np.searchsorted(self._starts, drawn, side='right') - 1
        voxel = drawn - self._starts[subject]
        pick = lambda arrays: np.stack([arrays[s][v] for s, v in zip(subject, voxel)])
        maps = pick(self._truth)
        names = {'meta': 'metabolites', 'nuisance': 'nuisance'}
        return pick(self._voxels[self.signal]), Truth(
            concentrations={name: maps[:, i] for i, name in enumerate(NAMES.values())},
            components={names[part]: pick(arrays) for part, arrays in self._voxels.items()
                        if part in names})

    #***********#
    #   basis   #
    #***********#
    @property
    def basis(self) -> Dict[str, np.ndarray]:
        """The release's basis on the data's time points (see ``basis_fids``)."""
        return basis_fids(self.subjects[0], self.data_dir)

    #************#
    #   ranges   #
    #************#
    @property
    def ranges(self) -> Dict[str, Tuple[float, float]]:
        """Each metabolite's lowest and highest amplitude over the subjects' brain voxels."""
        maps = np.concatenate(self._truth)
        return {name: (float(maps[:, i].min()), float(maps[:, i].max()))
                for i, name in enumerate(NAMES.values())}

    #************#
    #   recipe   #
    #************#
    def recipe(self) -> Dict[str, Any]:
        """What the spectra are: which subjects' brain voxels, of which signal."""
        return {'generation': {'release': f'2024 MRSI challenge (doi:{self._doi})',
                               'subjects': list(self.subjects), 'signal': self.signal,
                               'voxels': int(self._starts[-1]), 'seed': self.seed}}


#***********#
#   brain   #
#***********#
def _brain(module: Any, subject: str, signal: str, data_dir: Path) -> np.ndarray:
    """A subject's brain voxels of one signal as a (voxels, points) array, in the brain
    mask's own order (as ``load`` indexes them), written once and read memory-mapped."""
    path = data_dir / '_mrsuite_cache' / f'{subject}_{module.canonical_signal(signal)}.npy'
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        mask = module.read_aux(subject, signal='meta')['brainMask'] > 0
        writing = path.with_name(f'{path.stem}.{os.getpid()}.partial.npy')   # one per run
        np.save(writing, module.load([subject], signal=signal)[0][0][:][mask])
        os.replace(writing, path)
    return np.load(path, mmap_mode='r')


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
    from lcmodel_wrapper.convert import write_basis

    acquisition = _module(data_dir, subject).read_acquisition(subject)
    fids = basis_fids(subject, data_dir)

    # write_basis takes LCModel's .RAW orientation, the conjugate of NIfTI-MRS'
    return write_basis(path, list(fids), [np.conj(fid) for fid in fids.values()],
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

    module = _module(data_dir, *subjects)
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

    module = _module(data_dir, *subjects)
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


#****************#
#   basis fids   #
#****************#
def basis_fids(subject: str = 'Sub1', data_dir: Path = DATA_DIR) -> Dict[str, np.ndarray]:
    """
    The release's basis functions as FIDs on the data's time points, in NIfTI-MRS
    orientation and on the ground truth's scale: VtMeta and the first nine VtMM columns,
    from TE on (see ``basis``).

    Args:
        subject (str): Whose .mat to read the basis from; every subject shares it.
        data_dir (Path): Where the release lives.

    Returns:
        Dict[str, np.ndarray]: (384,) FIDs by the names in NAMES, then MACROMOLECULES.
    """
    import h5py

    module = _module(data_dir, subject)
    acquisition = module.read_acquisition(subject)
    skip = int(round(acquisition['echo_time_s'] / acquisition['dwell_time_s']))
    with h5py.File(module.mat_path(subject, need_truth=False), 'r') as f:
        def columns(name):
            raw = f[name][()]
            return (raw['real'] + 1j * raw['imag'])[skip:skip + acquisition['n_points']]
        fids = np.concatenate([columns('VtMeta'), columns('VtMM')[:, :len(MACROMOLECULES)]],
                              axis=1)
    return dict(zip(list(NAMES.values()) + list(MACROMOLECULES), fids.T))


#*****************#
#   vespa prior   #
#*****************#
def vespa_prior(subject: str = 'Sub1',
                data_dir: Path = DATA_DIR) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Vespa's prior for the release: its basis functions, sampled (see
    ``mrsuite.bridges.vespa.fit``), the macromolecules among them. Vespa's own
    macromolecule model takes one acquired function with one area, while the release has
    nine, whose amplitudes vary apart from voxel to voxel; and line lists cannot hold
    the broad ones, nor the J-coupled metabolites' lines closer than the 0.32 s FID
    resolves.

    Args:
        subject (str): Whose .mat to read the basis from.
        data_dir (Path): Where the release lives.

    Returns:
        Dict[str, Dict[str, np.ndarray]]: ``{name: {"fid": fid}}`` for every function.
    """
    return {name: {'fid': fid} for name, fid in basis_fids(subject, data_dir).items()}


#********************#
#   pyamares prior   #
#********************#
def pyamares_prior(folder: Optional[Path] = None, subject: str = 'Sub1',
                   data_dir: Path = DATA_DIR) -> str:
    """
    pyAMARES' prior knowledge for the release, written from its basis functions into
    ``folder`` (see ``mrsuite.bridges.pyamares.prior``), afresh each time: seconds, and
    never a file from older code or another subject.

    Args:
        folder (Optional[Path]): Where the prior lives; ``data/basis_sets`` by default.
        subject (str): Whose .mat to read the basis from.
        data_dir (Path): Where the release lives.

    Returns:
        str: The prior-knowledge file.
    """
    from mrsuite.bridges import pyamares

    folder = Path(folder) if folder else data_dir.parent / 'basis_sets'
    path = folder / 'mrsi_challenge_2024_pyamares.csv'
    folder.mkdir(parents=True, exist_ok=True)
    acquisition = _module(data_dir, subject).read_acquisition(subject)
    return pyamares.prior(str(path), basis_fids(subject, data_dir),
                          dwell=acquisition['dwell_time_s'],
                          frequency=acquisition['spectrometer_frequency_mhz'],
                          centre=acquisition['ppm_offset'])
