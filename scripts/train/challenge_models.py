####################################################################################################
#                                       challenge_models.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-02                                                                              #
#                                                                                                  #
# Purpose: Trains the benchmark's models on the two challenges' own acquisitions, with MRS-Sim:    #
#          the supervised and self-supervised MLPs of the out-of-distribution paper for the 2016   #
#          fitting challenge and the 2024 MRSI challenge, and WAND for the MRSI challenge with     #
#          its water and lipids as a component of their own, to take out before fitting; and the   #
#          MRSI MLPs on the release's own training subjects, with and without nuisance. Settings   #
#          as the published runs (batch 16, Adam 1e-4, the steps of their checkpoints, their       #
#          losses as far as MRS-Sim's truth carries them; see MODELS); each run writes loadable    #
#          checkpoints with their recipe.json and Adam's state to data/runs/<name>; --resume       #
#          learns on from the run's last checkpoint.                                               #
#                                                                                                  #
#          Usage: python scripts/train/challenge_models.py <name> [--steps N] [--ahead K]          #
#                 [--resume]                                                                       #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import argparse
import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from mrsuite import models
from mrsuite.datasets import challenge2016, mrsi_challenge
from mrsuite.models import MLP, WAND
from mrsuite.simulators import mrs_sim
from mrsuite.training import Source
from mrsuite.training.loop import OPTIMIZER, last_checkpoint

RUNS = Path(__file__).resolve().parents[2] / "data" / "runs"

# name -> (challenge, model, objective, steps). The steps are those of the published
# checkpoints, as Lightning counts them: one optimiser step on one batch of 16 (no
# gradient accumulation), which is one step of Model.learn. Supervised MLP: wyoo5p40,
# step 7537664; self-supervised: 48ke5d6u, step 3572224; WAND: t1axv47a, step 465664
# (wand-mrsi2024; the paper's own WAND is sskt2pvt, step 1195264, which the later
# WANDs follow).
# The losses as their runs: self-supervised mse_specs, WAND mse, and the supervised MLPs
# wyoo5p40's mae_all_scale as far as MRS-Sim's truth carries it ('scaled': the
# concentrations, each over its sampling range; the original's signal parameters have no
# counterpart in MRS-Sim's).
MODELS = {
    "mlp-supervised-ismrm2016": ("2016", "mlp", "scaled", 7_537_664),
    "mlp-selfsupervised-ismrm2016": ("2016", "mlp", "spectra", 3_572_224),
    "mlp-supervised-mrsi2024": ("mrsi", "mlp", "scaled", 7_537_664),
    "mlp-selfsupervised-mrsi2024": ("mrsi", "mlp", "spectra", 3_572_224),
    "wand-mrsi2024": ("mrsi", "wand", "components", 465_664),
    # the MRSI MLPs again, on the release's own training subjects instead of MRS-Sim: with
    # their water and lipids for track 1, without for track 2 (see mrsi_challenge.Release)
    "mlp-supervised-mrsi2024-release-track1": ("release-track1", "mlp", "scaled", 7_537_664),
    "mlp-supervised-mrsi2024-release-track2": ("release-track2", "mlp", "scaled", 7_537_664),
    "mlp-selfsupervised-mrsi2024-release-track2": ("release-track2", "mlp", "spectra",
                                                   3_572_224),
    # WAND on the MRSI challenge again, as long as the published WAND (sskt2pvt, step
    # 1195264): the full decomposition for each track - MRS-Sim's metabolites one by one,
    # with the release's water and lipids for track 1 ('mrsi'), without for track 2
    # ('mrsi-clean') - and for track 1 the release's own metabolites, summed, against its
    # water and lipids, plain and with the nuisance's error weighed down to their size
    "wand-mrsi2024-full-track1": ("mrsi", "wand", "components", 1_195_264),
    "wand-mrsi2024-full-track2": ("mrsi-clean", "wand", "components", 1_195_264),
    "wand-mrsi2024-release-track1": ("release-track1", "wand", "components", 1_195_264),
    "wand-mrsi2024-release-track1-weighted": ("release-track1", "wand", "components",
                                              1_195_264),
    # and with two masks only: the metabolites' and the artifacts' (1 - it), which then
    # take the water and lipids too
    "wand-mrsi2024-release-track1-two-masks": ("release-track1", "wand", "components",
                                               1_195_264),
}

# the release's signal each release-trained model learns from
RELEASE = {"release-track1": "all", "release-track2": "all-nuisance"}

# the WANDs weighing the nuisance's error (see nuisance_weight)
WEIGHTED = {"wand-mrsi2024-release-track1-weighted"}
TWO_MASKS = {"wand-mrsi2024-release-track1-two-masks"}

# as the published runs: batch 16, Adam at 1e-4
BATCH, LR = 16, 1e-4

# each model's window in ppm: the MLPs' as the originals' (ppmlim 0.5-4.0); WAND's from
# 0.2 ppm, the fitters' usual lower limit, so that what it takes out covers their fit
# ranges, up to 4.5 ppm: the plain FFT's order wraps at the water's 4.7 ppm, and a
# window cannot cross it
WINDOW = {"mlp": (0.5, 4.0), "wand": (0.2, 4.5)}

# spectra per MRS-Sim start: its start costs ~10 s, and its memory grows with the chunk
# (~5 GB for 4096 spectra of 2048 points)
CHUNK = {"2016": 2048, "mrsi": 8192}

# the training subjects whose water and lipids WAND learns to take out
SUBJECTS = ("Sub1", "Sub2")


#************#
#   window   #
#************#
def window(acquisition: Dict[str, float], low: float, high: float) -> Tuple[int, int]:
    """The points of the plain FFT (as mrsuite.models.base.spectra gives it) within
    [low, high] ppm, as first and last."""
    f = np.fft.fftfreq(int(acquisition["points"]), 1 / acquisition["bandwidth"])
    ppm = acquisition["centre"] + f / acquisition["frequency"]
    points = np.flatnonzero((ppm >= low) & (ppm <= high))
    if np.any(np.diff(points) != 1):
        raise ValueError(f"{low}-{high} ppm wraps around the plain FFT's order")
    return int(points[0]), int(points[-1]) + 1


#**************************************************************************************************#
#                                          Class Nuisance                                          #
#**************************************************************************************************#
#                                                                                                  #
# Simulated spectra with the MRSI challenge's own water and lipids added, as 'nuisance'.           #
#                                                                                                  #
#**************************************************************************************************#
class Nuisance(Source):
    """Simulated spectra with the MRSI challenge's own water and lipids added, as 'nuisance'.

    The release simulates the nuisance in ways a simulator does not: residual water
    from a Bloch simulation of WET, field dependent, and lipids measured in vivo,
    registered to each subject and blurred by k-space truncation, ~1000 times the
    metabolites. It ships them, noiseless and summed (xtNuisance), for its training
    subjects. Each spectrum here gets one brain voxel's nuisance, scaled so that it
    stands to the spectrum's metabolites as it stood to that voxel's (xtMeta); voxels
    with less than half the median metabolite signal are left out, as their ratio is
    unreliable. Water and lipids stay one component, as the release gives them.

    Args:
        source (Source): The spectra, giving each metabolite's signal on its own.
        subjects (Sequence[str]): The training subjects to take the nuisance from.
        seed (int): Seeds which voxel each spectrum gets.
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, source: Source, subjects: Sequence[str] = SUBJECTS,
                 seed: int = 0) -> None:
        from augmentrum.dataset.mrsi_challenge import MRSIChallengeDataModule

        self.source, self.subjects, self.seed = source, tuple(subjects), seed
        self.acquisition, self.dim_tags = source.acquisition, source.dim_tags
        self.metabolites = list(mrsi_challenge.NAMES.values())
        self.rng = np.random.default_rng(seed)

        module = MRSIChallengeDataModule(data_dir=str(mrsi_challenge.DATA_DIR), source="mat")
        signals, norms = [], []
        for subject in self.subjects:
            mask = module.read_aux(subject, signal="meta")["brainMask"] > 0
            nuisance = module.load([subject], signal="nuisance")[0][0][:][mask]
            norm = np.linalg.norm(module.load([subject], signal="meta")[0][0][:][mask], axis=1)
            kept = norm >= 0.5 * np.median(norm)
            signals.append(nuisance[kept])
            norms.append(norm[kept])
        self.signals, self.norms = np.concatenate(signals), np.concatenate(norms)

    #**********#
    #   draw   #
    #**********#
    def draw(self, n: int) -> Tuple[np.ndarray, Any]:
        fids, truth = self.source.draw(n)
        voxel = self.rng.integers(len(self.signals), size=n)
        metabolites = sum(truth.components[m] for m in self.metabolites)
        scale = np.linalg.norm(metabolites, axis=1) / self.norms[voxel]
        nuisance = self.signals[voxel] * scale[:, None]
        truth.components["nuisance"] = nuisance
        return fids + nuisance, truth

    #************#
    #   recipe   #
    #************#
    def recipe(self) -> Dict[str, Any]:
        """The source's recipe, with the nuisance as it is added."""
        recipe = copy.deepcopy(self.source.recipe())
        model = ("the 2024 MRSI challenge's own (LaMaster et al., doi:10.5281/zenodo.21890222): "
                 "Bloch-simulated residual water after WET and in vivo lipids registered to "
                 "each subject, summed in its xtNuisance; one brain voxel of "
                 f"{', '.join(self.subjects)} per spectrum, scaled so that it stands to the "
                 "spectrum's metabolites as to that voxel's")
        artifacts = recipe["MRSsynMRS"]["Artifacts"]
        artifacts["Residual Water"].update({
            "Model Used (Equation, Description, or DOI)": model,
            "If Analytical, Parameter Ranges": None,
            "Available in Isolation": "with the lipids, as the component 'nuisance'",
            "Applied": True, "Chemical Shift [ppm]": "4.0-5.4 (the release's water band)",
            "Amplitude Scaling [a.u.]": "as in the voxel, relative to its metabolites"})
        artifacts["Lipid"].update({
            "Model Used (Equation, Description, or DOI)": model,
            "Available in Isolation": "with the water, as the component 'nuisance'",
            "Applied": True})
        recipe["generation"]["nuisance"] = {"subjects": list(self.subjects),
                                            "voxels": len(self.signals), "seed": self.seed}
        return recipe


#*********************#
#   nuisance weight   #
#*********************#
def nuisance_weight(source: Source, first: int, last: int, n: int = 1024) -> float:
    """
    The fixed weight that brings the nuisance's squared error to the metabolites' size: one
    over the median ratio of their power in the window, over ``n`` of the source's spectra.

    Args:
        source (Source): Gives the metabolites and nuisance apart ('metabolites', 'nuisance').
        first (int): First point of the window.
        last (int): The point after it.
        n (int): Spectra to take the median over.

    Returns:
        float: The weight.
    """
    _, truth = source.draw(n)

    def power(fids):
        window = np.fft.fft(fids, axis=1)[:, first:last]
        return (np.abs(window) ** 2).sum(axis=1)
    return float(1 / np.median(power(truth.components["nuisance"])
                               / power(truth.components["metabolites"])))


#************#
#   config   #
#************#
def config(challenge: str, model: str, voxels: int = 2000) -> Dict[str, Any]:
    """
    MRS-Sim's config for a challenge, as its dataset module matches it to the challenge.

    Args:
        challenge (str): '2016' or 'mrsi'.
        model (str): 'mlp' or 'wand'. WAND on the MRSI challenge learns the release's own
            water (see Nuisance) instead of MRS-Sim's residual water: one water, in one
            component.
        voxels (int): Brain voxels the MRSI config's ranges are read from.

    Returns:
        Dict[str, Any]: The config.
    """
    if challenge == "2016":
        return challenge2016.mrs_sim_config()
    config = mrsi_challenge.mrs_sim_config(voxels=voxels)
    if model == "wand":
        config.pop("resWater_cfg", None)
    return config


#***********#
#   train   #
#***********#
def train(name: str, steps: Optional[int] = None, every: Optional[int] = None,
          to: Path = RUNS, batch: int = BATCH, seed: int = 0, chunk: Optional[int] = None,
          ahead: int = 0, voxels: int = 2000, subjects: Sequence[str] = SUBJECTS,
          release: Sequence[str] = mrsi_challenge.TRAIN, resume: bool = False) -> Any:
    """
    Train one of MODELS from scratch, or on from its run's last checkpoint, saving to
    ``to / name`` every ``every`` steps.

    Args:
        name (str): One of MODELS.
        steps (Optional[int]): Steps to learn; the published run's by default.
        every (Optional[int]): Steps between checkpoints; a sixteenth of the published run
            by default, or all the steps where that does not divide them.
        to (Path): Where the run's folder goes.
        batch (int): Spectra per step.
        seed (int): Seeds the training data and the initial weights.
        chunk (Optional[int]): Spectra per MRS-Sim start; CHUNK by default.
        ahead (int): MRS-Sim chunks simulated in the background (see mrs_sim.Source).
        voxels (int): Brain voxels the MRSI config's ranges are read from.
        subjects (Sequence[str]): Whose water and lipids WAND learns, on the MRSI challenge.
        release (Sequence[str]): The training subjects the release-trained models draw
            their voxels from.
        resume (bool): Learn on from the run's last checkpoint, with Adam's state where it
            was saved with one (runs before it was saved start Adam afresh), on spectra
            seeded past ``seed`` by the steps already taken, so none are seen again.

    Returns:
        The trained model, as also saved in ``to / name / step-<steps taken>``.
    """
    challenge, kind, objective, published = MODELS[name]
    steps = steps or published
    sixteenth = max(published // 16, 1)
    every = every or (sixteenth if steps % sixteenth == 0 else steps)
    checkpoint = last_checkpoint(Path(to) / name) if resume else None
    start = int(checkpoint.name[len("step-"):]) if checkpoint else 0
    if challenge in RELEASE:
        simulator = None
        data = mrsi_challenge.Release(release, RELEASE[challenge], components=kind == "wand",
                                      seed=seed + start)
        # WAND learns the release's metabolites summed, as it gives them; the self-supervised
        # fit rebuilds the whole signal; the supervised one learns what the release varies,
        # as it holds Lac and Lac41 at zero
        metabolites = (["metabolites"] if kind == "wand" else
                       list(data.basis) if objective == "spectra" else
                       [m for m, (low, high) in data.ranges.items() if high > low])
    else:
        simulated = challenge.removesuffix("-clean")
        simulation = config(simulated, kind, voxels)
        simulator = data = mrs_sim.Source(simulation, seed=seed + start,
                                          chunk=chunk or CHUNK[simulated],
                                          components=kind == "wand", ahead=ahead)
        metabolites = simulation["metabolites"]
    try:
        first, last = window(data.acquisition, *WINDOW[kind])
        torch.manual_seed(seed)                 # the initial weights, so a run replays
        if kind == "wand" and challenge == "mrsi":
            data = Nuisance(data, subjects, seed + start)
        if checkpoint:
            model = models.load(str(checkpoint))
        elif kind == "mlp":
            model = MLP(metabolites, first, last, data.acquisition)
        else:
            weights = ({"nuisance": nuisance_weight(
                mrsi_challenge.Release(release, "all", components=True, seed=seed),
                first, last)} if name in WEIGHTED else None)
            model = WAND(metabolites, first, last, data.acquisition,
                         nuisance=["nuisance"] if challenge in ("mrsi", "release-track1")
                         and name not in TWO_MASKS else [],
                         baseline=challenge not in RELEASE, weights=weights)
        state = checkpoint if checkpoint and (checkpoint / OPTIMIZER).exists() else None
        model.learn(data, steps=steps, objective=objective, batch=batch, lr=LR, every=every,
                    to=str(Path(to) / name), resume=str(state) if state else None)
    finally:
        if simulator:
            simulator.close()                   # no chunks simulated for nobody
    return model


#**********#
#   main   #
#**********#
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a challenge model with MRS-Sim.")
    parser.add_argument("name", choices=list(MODELS))
    parser.add_argument("--steps", type=int, help="the published run's by default")
    parser.add_argument("--every", type=int, help="steps between checkpoints")
    parser.add_argument("--to", type=Path, default=RUNS, help="where the run's folder goes")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ahead", type=int, default=0,
                        help="MRS-Sim chunks simulated in the background, in parallel")
    parser.add_argument("--resume", action="store_true",
                        help="learn on from the run's last checkpoint")
    args = parser.parse_args()
    # deterministic kernels, so a run on a GPU replays bit for bit (about 5% slower)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    train(args.name, steps=args.steps, every=args.every, to=args.to, seed=args.seed,
          ahead=args.ahead, resume=args.resume)
