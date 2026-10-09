####################################################################################################
#                                            config.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-02                                                                              #
#                                                                                                  #
# Purpose: What the benchmark runs, in one place: the datasets, the methods, which method runs     #
#          on which dataset in each stage, and how the results are scored. A new method is one     #
#          more entry in METHODS and in a stage; run.py, score.py and compare.py read it from      #
#          here.                                                                                   #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from dataclasses import dataclass, field, replace
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from mrsvalidate.analysis import agreement

from nifti_mrs_plus import NIfTI_MRS_Plus
from mrsuite.benchmark import bias, mae
from mrsuite.bridges import fsl_mrs, lcmodel, osprey, pyamares, spant, tarquin, vespa
from mrsuite.datasets import challenge2016, mrsi_challenge

MRSI_BASIS = Path(mrsi_challenge.DATA_DIR).parent / "basis_sets" / "mrsi_challenge_2024.BASIS"
WAND = "MRSuite/wand-ismrm2016"


#**************************************************************************************************#
#                                          Class Dataset                                           #
#**************************************************************************************************#
#                                                                                                  #
# Spectra with their ground truth and what a fit needs besides them.                               #
#                                                                                                  #
#**************************************************************************************************#
@dataclass
class Dataset:
    """Spectra with their ground truth and what a fit needs besides them.

    Attributes:
        data (NIfTI_MRS_Plus): The spectra.
        truth (List[Dict[str, float]]): Ground truth per spectrum.
        ids (List[str]): One ID per spectrum, unique, by which results are kept.
        basis (str): The basis set every fitter takes.
        water (Optional[NIfTI_MRS_Plus]): Water references, one per spectrum.
    """

    data: NIfTI_MRS_Plus
    truth: List[Dict[str, float]]
    ids: List[str]
    basis: str
    water: Optional[NIfTI_MRS_Plus] = None


#**************************************************************************************************#
#                                            Class Fit                                             #
#**************************************************************************************************#
#                                                                                                  #
# One method of the benchmark: steps on the whole dataset, then a fit, chunk by chunk.             #
#                                                                                                  #
#**************************************************************************************************#
@dataclass
class Fit:
    """One method of the benchmark: steps on the whole dataset, then a fit, chunk by chunk.

    Attributes:
        fit (Callable): A method returning one FitResult per spectrum, called as
            ``fit(spectra, *args(dataset), **options)``, with ``workers=`` where it offers it.
        args (Callable[[Dataset], tuple]): What it takes after the spectra; the basis by
            default, a prior or nothing for others.
        options (Dict[str, Any]): Further keyword arguments.
        water (Optional[Dict[str, Any]]): Fit with the dataset's water references, and
            these keyword arguments with them; None for no water.
        before (Sequence[Callable]): Steps run on the whole dataset first, as one batch.
        chunk (int): Spectra per worker in one chunk, saved as it finishes: a crash loses
            at most one chunk, and starting a chunk's workers costs about 12 s.
    """

    fit: Callable
    args: Callable[[Dataset], tuple] = lambda dataset: (dataset.basis,)
    options: Dict[str, Any] = field(default_factory=dict)
    water: Optional[Dict[str, Any]] = None
    before: Sequence[Callable] = ()
    chunk: int = 100


#**************************************************************************************************#
#                                            Class Job                                             #
#**************************************************************************************************#
@dataclass
class Job:
    dataset: str
    method: str
    spectra: Optional[int] = None      # an evenly spread subsample, as mrsi_challenge.load's voxels


#***************#
#   challenge   #
#***************#
def challenge(n: Optional[int] = None) -> Dataset:
    """The 2016 fitting challenge, its water references with it; the first n datasets."""
    data, water, truth = challenge2016.load(n)
    return Dataset(data, truth, [f"dataset{i}" for i in range(1, len(truth) + 1)],
                   str(challenge2016.BASIS), water)


#**********#
#   mrsi   #
#**********#
def mrsi(subject: str, voxels: Optional[int] = None) -> Dataset:
    """One MRSI challenge subject, every brain voxel or an evenly spread few, on the
    release's own basis (written once); voxels by their x-y-z index."""
    if not MRSI_BASIS.exists():
        mrsi_challenge.basis(str(MRSI_BASIS))
    data, truth, index = mrsi_challenge.load(subject, voxels=voxels)
    return Dataset(data, truth, ["-".join(map(str, voxel)) for voxel in index], str(MRSI_BASIS))


#****************#
#   wand model   #
#****************#
@lru_cache(maxsize=None)
def _wand() -> Any:
    """The model, loaded once, when a method first needs it."""
    from mrsuite import models
    return models.load(WAND)


#**********#
#   wand   #
#**********#
def wand(data: NIfTI_MRS_Plus) -> NIfTI_MRS_Plus:
    """WAND removing its artifact component, the whole dataset in one batch, as the paper's
    batch statistics were."""
    return _wand()(data, remove=["artifacts"])


#*************#
#   methods   #
#*************#
# spectra per worker per chunk: about ten minutes of fitting at the measured speeds on MRSI
# (TARQUIN 0.2 s, FSL-MRS 0.55 s, spant 1.4 s, LCModel 1.7 s, Osprey ~127 s per spectrum)
CHUNK = {"fsl_mrs": 1000, "lcmodel": 350, "tarquin": 3000, "spant": 400, "osprey": 5}
FITS = {"fsl_mrs": fsl_mrs.fit, "lcmodel": lcmodel.fit, "tarquin": tarquin.fit,
        "spant": spant.fit, "osprey": osprey.fit}

METHODS = {name: Fit(fit, chunk=CHUNK[name]) for name, fit in FITS.items()}
METHODS.update({f"{name}-water": Fit(FITS[name], water=options, chunk=CHUNK[name])
                for name, options in challenge2016.WATER.items()})
# Vespa and pyAMARES take the challenge's priors in place of a basis, ~450 s and ~400 s per
# spectrum; pyAMARES with its own tolerances (1e-6: the FID's maximum, 1, under a root) but
# lmfit's evaluation limit, 2000 (free parameters + 1), in place of its 1000, which the
# prior's 67 parameters exhaust
METHODS["vespa"] = Fit(vespa.fit, args=lambda dataset: (challenge2016.vespa_prior(),), chunk=1)
METHODS["pyamares"] = Fit(pyamares.fit, args=lambda dataset: (challenge2016.pyamares_prior(),),
                          options={"opts": {"fit_kws": {"max_nfev": None, "xtol": 1e-6,
                                                        "ftol": 1e-6}}}, chunk=1)
METHODS.update({f"wand-{name}": replace(entry, before=(wand,))
                for name, entry in list(METHODS.items())})


#************#
#   stages   #
#************#
SUBJECTS = ("TestSub10", "TestSub11", "TestSub12")
OSPREY_VOXELS = 100

DATASETS = {"challenge2016": challenge, **{s: partial(mrsi, s) for s in SUBJECTS}}

STAGES = {
    "2016": [Job("challenge2016", name) for name in METHODS],
    "mrsi": [Job(s, name) for s in SUBJECTS for name in ("fsl_mrs", "lcmodel", "tarquin", "spant")]
            + [Job(s, "osprey", spectra=OSPREY_VOXELS) for s in SUBJECTS],
}


#*************#
#   scoring   #
#*************#
# No default reference: water where the truth is in mM and the fits are water-scaled; on
# MRSI, whose truth is on the basis' own scale, tCr, the optimal scale and none at all.
REFERENCES = {"challenge2016": ("water", "tCr", "optimal"),
              **{s: ("tCr", "optimal", "none") for s in SUBJECTS}}
METRICS = {"mae": mae, "bias": bias, "lin_ccc": agreement.lin_ccc}
