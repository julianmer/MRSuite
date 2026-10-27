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
import json
from dataclasses import dataclass, field, replace
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from mrsvalidate.analysis import agreement, descriptive, errors, method_comparison, regression

from nifti_mrs_plus import NIfTI_MRS_Plus
from mrsuite.benchmark import bias, mae
from mrsuite.bridges import fsl_mrs, lcmodel, osprey, pyamares, spant, tarquin, vespa
from mrsuite.core.method import method
from mrsuite.core.provenance import mrsuite_version
from mrsuite.datasets import challenge2016, mrsi_challenge
from mrsuite.training.loop import last_checkpoint

MRSI_BASIS = Path(mrsi_challenge.DATA_DIR).parent / "basis_sets" / "mrsi_challenge_2024.BASIS"
WAND = "MRSuite/wand-ismrm2016"
RUNS = Path(__file__).resolve().parents[2] / "data" / "runs"      # scripts/train writes them


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
        timeout (Optional[float]): Seconds a worker's batch may take, where the fitter
            offers it; a batch running over is interrupted and its spectra are failures. A
            limit on the run, not on the method, so a job does not record it.
    """

    fit: Callable
    args: Callable[[Dataset], tuple] = lambda dataset: (dataset.basis,)
    options: Dict[str, Any] = field(default_factory=dict)
    water: Optional[Dict[str, Any]] = None
    before: Sequence[Callable] = ()
    chunk: int = 100
    timeout: Optional[float] = None


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
    return Dataset(data, truth, _voxel_ids(index), str(MRSI_BASIS))


#***************#
#   voxel ids   #
#***************#
def _voxel_ids(index) -> List[str]:
    """Spectrum IDs of MRSI voxels: their x-y-z index."""
    return ["-".join(map(str, voxel)) for voxel in index]


#***********#
#   suite   #
#***********#
def suite(axis: Optional[str] = None, level: Optional[str] = None, n: int = 100) -> Dataset:
    """One level of the simulated suite (challenge2016.SUITE) on the challenge's basis; no
    axis for the training distribution itself."""
    data, truth = challenge2016.suite(axis, level, n)
    name = f"{axis}-{level}" if axis else "in-distribution"
    return Dataset(data, truth, [f"{name}-{i}" for i in range(n)], str(challenge2016.BASIS))


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


#***********#
#   model   #
#***********#
@lru_cache(maxsize=None)
def _model(checkpoint: str) -> Any:
    """A model trained here, loaded once per checkpoint."""
    from mrsuite import models
    return models.load(checkpoint)


#*************#
#   weights   #
#*************#
@lru_cache(maxsize=None)
def weights(name: str) -> str:
    """The weights of a training run's best checkpoint where scripts/train/best_checkpoint.py
    chose one (best.json), else of its last when first asked, so one session scores one
    checkpoint while training writes on; a job records them by SHA-256, so results of other
    weights are never resumed into."""
    best = RUNS / name / "best.json"
    folder = (RUNS / name / json.loads(best.read_text())["checkpoint"] if best.exists()
              else last_checkpoint(RUNS / name))
    return str(folder / "model.safetensors")


#*************#
#   trained   #
#*************#
@method("MRSuite", version=mrsuite_version, batched=True)
def trained(data: NIfTI_MRS_Plus, weights: str) -> List[Any]:
    """A model trained here (scripts/train), by its checkpoint's weights, on the batch."""
    return _model(str(Path(weights).parent)).run(data)


#***************#
#   wand mrsi   #
#***************#
def wand_mrsi(data: NIfTI_MRS_Plus) -> NIfTI_MRS_Plus:
    """WAND trained here on the MRSI challenge removing its water and lipids."""
    return _model(str(Path(weights("wand-mrsi2024")).parent))(data, remove=["nuisance"])


#**************#
#   wand run   #
#**************#
def wand_run(name: str, remove: Sequence[str]) -> Callable[[NIfTI_MRS_Plus], NIfTI_MRS_Plus]:
    """A WAND trained here (scripts/train), by its run's last checkpoint, removing ``remove``
    before a fit; the step is named after the run, as a job records it."""
    def step(data: NIfTI_MRS_Plus) -> NIfTI_MRS_Plus:
        return _model(str(Path(weights(name)).parent))(data, remove=list(remove))
    step.__name__ = name.replace("-", "_")
    return step


#*************#
#   methods   #
#*************#
# spectra per worker per chunk: about ten minutes of fitting at the measured speeds on MRSI
# (TARQUIN 0.2 s, FSL-MRS 0.55 s, spant 1.4 s, LCModel 1.7 s, Osprey ~127 s per spectrum)
CHUNK = {"fsl_mrs": 1000, "lcmodel": 350, "tarquin": 3000, "spant": 400, "osprey": 5}
FITS = {"fsl_mrs": fsl_mrs.fit, "lcmodel": lcmodel.fit, "tarquin": tarquin.fit,
        "spant": spant.fit, "osprey": osprey.fit}

METHODS = {name: Fit(fit, chunk=CHUNK[name]) for name, fit in FITS.items()}
# Osprey's fit can crawl for hours on one spectrum (its LM at a bound: a WAND-cleaned 2016
# spectrum stalled the stage for days), and a chunk is written only once every fit returns;
# half an hour for a worker's five is ten times the usual, and the spectrum is a failure
METHODS["osprey"] = replace(METHODS["osprey"], timeout=1800)
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
# the models trained here (scripts/train), each by its run's last checkpoint, and fitters
# after the WAND trained on the MRSI challenge
TRAINED = {name: Fit(trained, args=lambda dataset, name=name: (weights(name),), chunk=1000)
           for name in ("mlp-supervised-ismrm2016", "mlp-selfsupervised-ismrm2016",
                        "mlp-supervised-mrsi2024", "mlp-selfsupervised-mrsi2024")}
TRAINED.update({f"wand-mrsi2024-{name}": replace(METHODS[name], before=(wand_mrsi,))
                for name in ("lcmodel", "fsl_mrs")})
# the MRSI MLPs trained on the release's own subjects, and the WANDs trained on the MRSI
# challenge afterwards, each before LCModel and FSL-MRS: taking out the water and lipids on
# track 1, the artifacts on track 2 (as on the 2016 challenge)
TRAINED.update({name: Fit(trained, args=lambda dataset, name=name: (weights(name),), chunk=1000)
                for name in ("mlp-supervised-mrsi2024-release-track1",
                             "mlp-supervised-mrsi2024-release-track2",
                             "mlp-selfsupervised-mrsi2024-release-track2")})
WANDS = {"wand-mrsi2024-full-track1": ["nuisance"], "wand-mrsi2024-full-track2": ["artifacts"],
         "wand-mrsi2024-release-track1": ["nuisance"],
         "wand-mrsi2024-release-track1-weighted": ["nuisance"]}
TRAINED.update({f"{wand}-{name}": replace(METHODS[name], before=(wand_run(wand, remove),))
                for wand, remove in WANDS.items() for name in ("lcmodel", "fsl_mrs")})
METHODS.update(TRAINED)


#************#
#   stages   #
#************#
SUBJECTS = ("TestSub10", "TestSub11", "TestSub12")              # track 2: no water, no lipids
TRACK1 = ("TestSub1", "TestSub2", "TestSub3", "TestSub4", "TestSub5")    # with water and lipids
OSPREY_VOXELS = 100
MODEL_VOXELS = 1000

# each subject as an evenly spread few voxels, for the trained models: WAND on CPU takes
# ~0.2 s per voxel, and takes them all before a job's own subsample is drawn
SAMPLED = {s: f"{s}-{MODEL_VOXELS}" for s in (*SUBJECTS, *TRACK1)}

# the simulated suite: 500 spectra in distribution, 100 per level of each axis
SUITE = {"suite-in-distribution": partial(suite, n=500),
         **{f"suite-{axis}-{level}": partial(suite, axis, level)
            for axis, levels in challenge2016.SUITE.items() for level in levels}}

DATASETS = {"challenge2016": challenge, **{s: partial(mrsi, s) for s in SUBJECTS},
            **{name: partial(mrsi, s, MODEL_VOXELS) for s, name in SAMPLED.items()}, **SUITE}

STAGES = {
    "2016": [Job("challenge2016", name) for name in METHODS if name not in TRAINED],
    "mrsi": [Job(s, name) for s in SUBJECTS for name in ("fsl_mrs", "lcmodel", "tarquin", "spant")]
            + [Job(s, "osprey", spectra=OSPREY_VOXELS) for s in SUBJECTS],
    # the trained models against the fitters on the same spectra
    "models-2016": [Job("challenge2016", name) for name in
                    ("lcmodel", "fsl_mrs", "mlp-supervised-ismrm2016",
                     "mlp-selfsupervised-ismrm2016")],
    "models-mrsi": [Job(SAMPLED[s], name) for s in SUBJECTS for name in
                    ("lcmodel", "fsl_mrs", "mlp-supervised-mrsi2024", "mlp-selfsupervised-mrsi2024",
                     "wand-mrsi2024-lcmodel", "wand-mrsi2024-fsl_mrs")],
    # WAND where there is water and lipid to take out
    "models-mrsi-track1": [Job(SAMPLED[s], name) for s in TRACK1 for name in
                           ("lcmodel", "fsl_mrs", "wand-mrsi2024-lcmodel",
                            "wand-mrsi2024-fsl_mrs")],
    # the models trained on the release, and the later WANDs, on each track; on track 1 the
    # MRS-Sim MLPs too, which never saw lipids
    "models-mrsi-release": [Job(SAMPLED[s], name) for s in SUBJECTS for name in
                            ("lcmodel", "fsl_mrs", "mlp-supervised-mrsi2024-release-track2",
                             "mlp-selfsupervised-mrsi2024-release-track2",
                             "wand-mrsi2024-full-track2-lcmodel",
                             "wand-mrsi2024-full-track2-fsl_mrs")],
    # every method on the simulated suite but Vespa and pyAMARES (held back for now)
    "suite": [Job(dataset, name) for dataset in SUITE for name in
              (*(f"{wand}{fit}" for fit in ("lcmodel", "fsl_mrs", "spant", "tarquin", "osprey")
                 for wand in ("", "wand-")),
               "mlp-supervised-ismrm2016", "mlp-selfsupervised-ismrm2016")],
    "models-mrsi-release-track1": [Job(SAMPLED[s], name) for s in TRACK1 for name in
                                   ("lcmodel", "fsl_mrs", "mlp-supervised-mrsi2024",
                                    "mlp-selfsupervised-mrsi2024",
                                    "mlp-supervised-mrsi2024-release-track1",
                                    *(f"{wand}-{fit}" for wand in WANDS if "track1" in wand
                                      for fit in ("lcmodel", "fsl_mrs")))],
}


#*************#
#   scoring   #
#*************#
# No default reference: water where the truth is in mM and the fits are water-scaled; on
# MRSI, whose truth is on the basis' own scale, tCr, the optimal scale and none at all.
REFERENCES = {"challenge2016": ("water", "tCr", "optimal"),
              **dict.fromkeys(SUITE, ("tCr", "optimal")),
              **{s: ("tCr", "optimal", "none") for s in (*SUBJECTS, *SAMPLED.values())}}
# the challenge's 24 brain-like datasets apart from its four tumour-like ones (lipids, lactate,
# acetate), which no brain-like simulation covers; and each scenario on its own
SUBSETS = {"challenge2016": {"brain-like": [f"dataset{i}" for i in range(1, 25)],
                             **{name: [f"dataset{i}" for i in numbers]
                                for name, numbers in challenge2016.SCENARIOS.items()}}}
# and every MRSI subject on the models' evenly spread voxels (read when scored: the release
# is fetched on first use), where the whole subject takes MRSValidate hours per metabolite
SUBSETS.update({s: {str(MODEL_VOXELS): partial(lambda s: _voxel_ids(
    mrsi_challenge.voxel_index(s, MODEL_VOXELS)), s)} for s in SUBJECTS})
# a measure or two per aspect of MRSValidate, so the radar has a corner for each: errors,
# agreement, regression, method comparison, and the errors' spread
METRICS = {"mae": mae, "bias": bias, "rmse": errors.rmse, "lin_ccc": agreement.lin_ccc,
           "robust_ccc": agreement.robust_ccc, "pearson_r": regression.pearson_r,
           "r_squared_direct": regression.r_squared_direct,
           "bland_altman": method_comparison.bland_altman, "robust_sd": descriptive.robust_sd,
           "iqr": descriptive.iqr}
