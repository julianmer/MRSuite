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
#          ground truth is kept in-repo, as that repository does not carry it. An installed        #
#          MRSuite downloads the release as published, which holds both, on first use.             #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import os
import re
import shutil
import tempfile
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from nifti_mrs_plus import NIfTI_MRS_Plus
from mrsuite import externals
from mrsuite.core import provenance

DATA = externals.local("data")
ROOT = DATA / "2016_fitting_challenge"
TRUTH = DATA / "2016_fitting_challenge_gts"
BASIS = ROOT / "basisset_LCModel" / "press3T_30ms.BASIS"

# The release as published (University of Minnesota Data Repository, doi:10.13020/kw61-3j13):
# url, bytes and MD5. Its spectra, basis sets and ground truth are byte for byte the
# submodule's and the in-repo truth, the truth named dataset1_answer.xlsx for dataset1.xlsx.
RELEASE = ("https://conservancy.umn.edu/server/api/core/bitstreams/"
           "6aa26a1e-0a04-4af7-b49b-40bd6d9ad8e9/content", 5969351,
           "9da21a50739c1b9844157704cab885ff")
# its zips that hold no folder of their own, by the submodule's folder for them
FLAT = {"basisset_text.zip": "basisset_text", "basisset_JMRUI.zip": "basisset_JMRUI",
        "basisset_text_noTMS_20160929.zip": "basisset_text_noTMS"}

# The challenge's water scaling (its README, "Water scaling information"): water at 1 mol
# per 18.015 g in 60% GM (0.78 g/ml, T2 110 ms) and 40% WM (0.65 g/ml, T2 80 ms) at TE 30 ms,
# 29697 mM, and metabolites corrected for their T2 of 160 ms, x 1.206; TR >> T1, so no T1 term.
TE = 0.030
WATER_MM = 1e6 / 18.015 * (0.6 * 0.78 * np.exp(-30 / 110) + 0.4 * 0.65 * np.exp(-30 / 80))
METAB_T2 = np.exp(30 / 160)

# The datasets by what they test, as the WAND paper's challenge figure groups them
# (externals/wand-for-mrs visualizations/visChallenge.py, its loader's order mapped to the
# file numbers); datasets 23 and 24 carry a water peak 4.5 and 2.8 times NAA's, the others
# at most 0.4 times.
SCENARIOS = {"lorentzian": [1, 2, 3], "gaussian": [4, 5, 6], "gaba-gsh": [7, 8, 9, 10],
             "no-mm": [11], "snr-20": [12, 13, 14], "snr-30": [15, 16, 17],
             "snr-40": [18, 19, 20], "snr-160": [21], "eddy-currents": [22],
             "residual-water": [23, 24], "tumour-like": [25, 26, 27, 28]}

# Each water-scaling fitter's own options that put it on that convention, for its fit. On
# its defaults a tool measures its own constants, not its fit: LCModel and TARQUIN come
# out at 0.70x the truth, FSL-MRS at 1.24x.
WATER = {
    "lcmodel": {"params": {"wconc": WATER_MM * METAB_T2, "atth2o": 1.0}},
    "tarquin": {"opts": {"w_conc": WATER_MM * METAB_T2, "w_att": 1.0}},
    "fsl_mrs": {"te": TE, "tr": 10.0, "quant": {      # the spectra's files carry no TE or TR
        "t1": dict.fromkeys(("H2O_GM", "H2O_WM", "H2O_CSF", "METAB"), 1e-9),   # TR >> T1
        "t2": {"H2O_GM": 0.110, "H2O_WM": 0.080, "H2O_CSF": 2.55, "METAB": 0.160},
        "fractions": {"GM": 0.6, "WM": 0.4, "CSF": 0.0},
        "densities": {"GM": 0.78, "WM": 0.65, "CSF": 0.97},
        "add_corr": 1e6 / 18.015 / 55010}},    # FSL-MRS takes water as 55.01 M
}


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


#***********#
#   fetch   #
#***********#
def fetch() -> None:
    """
    The release, downloaded into ROOT and its truth into TRUTH on first use, in the
    submodule's layout; the truth first, so the challenge folder, moved in last, marks it
    whole. Two processes fetching at once both finish; the first one's stands.

    Raises:
        RuntimeError: In a checkout whose submodule is not initialised, or if the download
            is not the release published.
    """
    if ROOT.is_dir() and any(ROOT.iterdir()):
        return
    if externals.checkout():
        raise RuntimeError(f"{ROOT} is empty: git -C {externals.CHECKOUT} submodule update "
                           f"--init data/{ROOT.name}")
    url, size, md5 = RELEASE
    DATA.mkdir(parents=True, exist_ok=True)
    part = Path(tempfile.mkdtemp(prefix=".part", dir=DATA))
    try:
        with externals.fetching(f"the 2016 fitting challenge ({size / 1e6:.0f} MB)", DATA):
            externals.download(url, part / "release.zip", size, md5)
        with zipfile.ZipFile(part / "release.zip") as release:
            for name in release.namelist():
                if name.startswith("ground_truth/") and name.endswith(".xlsx"):
                    (part / "truth").mkdir(exist_ok=True)
                    (part / "truth" / Path(name).name).write_bytes(release.read(name))
                elif name.endswith(".zip"):
                    release.extract(name, part)
                    inner = Path(name).name
                    with zipfile.ZipFile(part / name) as files:
                        files.extractall(part / "challenge" / FLAT.get(inner, ""))
        externals.move_in(part / "truth", TRUTH)
        externals.move_in(part / "challenge", ROOT)
    finally:
        shutil.rmtree(part, ignore_errors=True)


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
    import mrsuite.bridges.fsl_mrs        # where FSL-MRS is missing, says how to install it
    import fsl_mrs
    from fsl_mrs.utils import mrs_io

    if Path(root) == ROOT or (Path(truth) == TRUTH and not externals.checkout()):
        fetch()
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
    -10.6 to -7.5 ppm. On the released data it gives the paper's own values within a few
    percent (86 for dataset 1; 20, 30, 43 and 158 for datasets 12, 15, 18 and 21, which
    the paper states as 161).

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
      of up to 10.9 Hz FWHM (g up to 422 per s^2), so d spans 0-30.4 and g 0-422, one
      of each per spectrum ('share_lorentzian'), as each dataset had one linewidth; the
      MMBL takes the metabolites' ranges, since MRS-Sim has no separate ones;
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

    fetch()
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
        'share_lorentzian': True,
        'notes': 'ISMRM 2016 fitting challenge, brain-like datasets 1-24: basis compiled '
                 'from basisset_LCModel/press3T_30ms.BASIS, cut to the data grid; '
                 'concentrations over the ground truth range relative to Cr+PCr, '
                 'linewidths 6-12 Hz Lorentzian or Voigt, SNR 20-161 as the challenge '
                 'defines it; no phase, shift, eddy currents, baseline or water.'})
    config['parameters'] = {'d': [0.0, round(np.pi * 10 - 1, 2)], 'g': [0.0, 422.0],
                            **ranges}

    # MRS-Sim's SNR per unit of the challenge's, measured on this config itself
    probe = {**config, 'parameters': {**config['parameters'], 'snr': [1000, 1000]}}
    measured = snr(mrs_sim.simulate(probe, n=32, seed=0, components=False)[0])
    factor = float(np.median(1000 / measured))
    config['snr'] = [round(20 * factor, 1), round(161 * factor, 1)]
    config['parameters']['snr'] = config['snr']
    return config


#****************#
#   basis fids   #
#****************#
def basis_fids() -> Dict[str, np.ndarray]:
    """
    The challenge's basis functions as FIDs on the data's 2048 points, in NIfTI-MRS
    orientation and on the ground truth's scale.

    The text release (``basisset_text``, the conjugate orientation) holds the same
    functions as the .BASIS, which is them at 1/64 and zero-filled, Ace included, which
    the truth has and the .BASIS lacks. Every function there carries a reference peak at
    0 ppm that the data do not; the release without it (``basisset_text_noTMS``) differs
    from the other by exactly that peak, except for Mac, whose version there is another
    function altogether, one the data fit worse. So the reference is taken as that
    difference for Cr and removed from every function.

    Returns:
        Dict[str, np.ndarray]: (2048,) FIDs by the truth's names.
    """
    fetch()

    def read(folder, name):
        columns = np.loadtxt(ROOT / folder / f'{name}.txt')
        return np.conj(columns[:, 0] + 1j * columns[:, 1])

    reference = read('basisset_text', 'Cr') - read('basisset_text_noTMS', 'Cr')
    return {f.stem: read('basisset_text', f.stem) - reference
            for f in sorted((ROOT / 'basisset_text').glob('*.txt'))}


#*****************#
#   vespa prior   #
#*****************#
def vespa_prior() -> Dict[str, Dict[str, np.ndarray]]:
    """
    Vespa's prior for the challenge: its basis functions, sampled (see
    ``mrsuite.bridges.vespa.fit``). Line lists cannot hold them: the functions carry the
    challenge's 2 Hz Lorentzian, and lines of J-coupled metabolites closer than the
    0.512 s FID resolves leave HSVD only components of other widths, which a line list,
    one width for all, misses by 2-30% of the function (Glu, Gln, Glc, GSH, GABA, Ins).

    Returns:
        Dict[str, Dict[str, np.ndarray]]: ``{name: {"fid": fid}}`` for every function.
    """
    return {name: {'fid': fid} for name, fid in basis_fids().items()}


#********************#
#   pyamares prior   #
#********************#
def pyamares_prior(folder: Path = DATA / 'basis_sets') -> str:
    """
    pyAMARES' prior knowledge for the challenge, written from its basis functions into
    ``folder`` (see ``mrsuite.bridges.pyamares.prior``), afresh each time: seconds, and
    never a file from older code.

    Args:
        folder (Path): Where the prior lives.

    Returns:
        str: The prior-knowledge file.
    """
    from mrsuite.bridges import pyamares

    path = Path(folder) / 'challenge_2016_pyamares.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    spectrum = load(1)[0].list()[0]          # the prior in ppm as the fit will read it
    return pyamares.prior(str(path), basis_fids(), dwell=spectrum.dwelltime,
                          frequency=spectrum.spectrometer_frequency[0],
                          centre=spectrum.SpecFreqChemShift)


#***********#
#   suite   #
#***********#
# The benchmark suite on simulated spectra (paper/ismrm2027/SUITE.md): mrs_sim_config's
# spectra, the 2016 training distribution, then one effect at a time moved out of it, by
# level. MRS-Sim draws each range uniformly per spectrum, and a switched-on effect takes
# MRS-Sim's own wide default unless given one, as the base config sets none. A Lorentzian
# line is (d + 1) / pi Hz wide on the basis' own 2 Hz (MRS-Sim adds 1 per s, see
# mrs_sim_config); a Gaussian 2 sqrt(g ln 2) / pi Hz. Baseline and residual water are scaled
# to the Cr + PCr peak. Pure Lorentzian lines take g = 0 under 'voigt': MRS-Sim's own
# 'lorentzian' line shape refers to a g it does not have.
def _lorentzian(width: float) -> float:
    return round(np.pi * (width - 2) - 1, 2)


SUITE = {
    "snr": {"10": {"snr": 10}, "5": {"snr": 5}, "300": {"snr": 300}},   # as the challenge's
    "linewidth": {f"{w}Hz": {"parameters": {"d": [_lorentzian(w - 1), _lorentzian(w + 1)],
                                            "g": [0.0, 0.0]}} for w in (15, 20, 30)},
    "lineshape": {"gaussian": {"parameters": {"d": [0.0, 0.0], "g": [128.0, 513.0]}}},  # 6-12 Hz
    "concentrations": {"above-1-1.5x": {"beyond": (1.0, 1.5)},
                       "above-1.5-2x": {"beyond": (1.5, 2.0)}, "absent-30pc": {"drop_prob": 0.3}},
    "frequency-shift": {f"{x}ppm": {"fshift_g": True, "parameters": {"f_shift": [-x, x]}}
                        for x in (0.02, 0.05, 0.1)},
    "zero-order-phase": {f"{x}deg": {"phi0": True, "parameters": {"phi0": [-x, x]}}
                         for x in (15, 45, 90)},
    "first-order-phase": {f"{x}deg-per-ppm": {"phi1": True, "parameters": {"phi1": [-x, x]}}
                          for x in (10, 20)},
    "baseline": {f"{s}x-tcr": {"baseline_cfg": {"scale": [s, s]}} for s in (0.2, 0.5, 1.0)},
    # MRS-Sim's water is a broad walk: scaled 2, 8 and 20 its peak measures 0.5, 1.4 and 3.5
    # times NAA's (medians of 32), the last as datasets 23 and 24 carry it (4.5 and 2.8)
    "residual-water": {f"{s}x-tcr": {"resWater_cfg": {"scale": [s, s]}} for s in (2.0, 8.0, 20.0)},
    "macromolecules": {"x0": {"scale": {"mac": 0.0}}, "x2": {"scale": {"mac": 2.0}}},
}
# far from the seeds training draws, so the suite's spectra are new to every model
SEED = 10 ** 9


#****************#
#   suite base   #
#****************#
@lru_cache(maxsize=None)
def _suite_base() -> Dict:
    """mrs_sim_config, once: it simulates to calibrate the SNR."""
    return mrs_sim_config()


#******************#
#   suite config   #
#******************#
def suite_config(axis: Optional[str] = None, level: Optional[str] = None) -> Dict:
    """
    MRS-Sim's config for one level of the suite, or the base (no axis) as it is.

    Args:
        axis (Optional[str]): One of SUITE; None for the training distribution itself.
        level (Optional[str]): One of the axis' levels.

    Returns:
        Dict: The config, as ``mrs_sim.simulate`` takes it.
    """
    import copy

    from mrsuite.simulators import mrs_sim

    config = copy.deepcopy(_suite_base())
    if axis is None:
        return config
    change = SUITE[axis][level]
    parameters = config['parameters']
    if 'snr' in change:                          # the challenge's SNR, in MRS-Sim's units
        value = round(change['snr'] * config['snr'][0] / 20, 1)
        parameters['snr'] = [value, value]
    parameters.update(change.get('parameters', {}))
    for flag in ('fshift_g', 'phi0', 'phi1'):
        if flag in change:
            config[flag] = change[flag]
    for nuisance in ('baseline_cfg', 'resWater_cfg'):
        if nuisance in change:
            config[nuisance] = {**mrs_sim.config()[nuisance], **change[nuisance]}
    if 'drop_prob' in change:
        config['drop_prob'] = change['drop_prob']
    if 'beyond' in change:                        # each metabolite above its trained range
        low, high = change['beyond']
        for name in config['metabolites']:
            key = name.lower()
            if key in parameters and name not in ('Cr', 'PCr', 'Mac'):
                parameters[key] = [round(parameters[key][1] * low, 4),
                                   round(parameters[key][1] * high, 4)]
    for key, factor in change.get('scale', {}).items():
        parameters[key] = [round(v * factor, 4) for v in parameters[key]]
    return config


#***********#
#   suite   #
#***********#
def suite(axis: Optional[str] = None, level: Optional[str] = None,
          n: int = 100) -> Tuple[NIfTI_MRS_Plus, List[Dict[str, float]]]:
    """
    One level of the suite as spectra and their truth, seeded, so the same every time;
    the base (no axis) is the training distribution itself.

    Args:
        axis (Optional[str]): One of SUITE; None for the training distribution.
        level (Optional[str]): One of the axis' levels.
        n (int): Spectra.

    Returns:
        Tuple[NIfTI_MRS_Plus, List[Dict[str, float]]]: The spectra, and per spectrum the
            amplitude of each basis function in the spectra's own units.
    """
    from mrsuite.simulators import mrs_sim

    levels = [(None, None)] + [(a, lv) for a in SUITE for lv in SUITE[a]]
    data, truth = mrs_sim.simulate(suite_config(axis, level), n=n,
                                   seed=SEED + levels.index((axis, level)), components=False)
    names = list(truth.concentrations)
    return data, [{name: float(truth.concentrations[name][i]) for name in names}
                  for i in range(n)]
