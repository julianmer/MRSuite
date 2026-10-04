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
#          its water and lipids as a component of their own, to take out before fitting. Settings  #
#          as the published runs (batch 16, Adam 1e-4, the steps of their checkpoints), but the    #
#          supervised loss is MRSuite's (see MODELS); each run writes loadable checkpoints with    #
#          their recipe.json to data/runs/<name>.                                                  #
#                                                                                                  #
#          Usage: python scripts/train/challenge_models.py <name> [--steps N] [--ahead K]          #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import argparse
import copy
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from mrsuite.datasets import challenge2016, mrsi_challenge
from mrsuite.models import MLP, WAND
from mrsuite.simulators import mrs_sim
from mrsuite.training import Source

RUNS = Path(__file__).resolve().parents[2] / "data" / "runs"

# name -> (challenge, model, objective, steps). The steps are those of the published
# checkpoints, as Lightning counts them: one optimiser step on one batch of 16 (no
# gradient accumulation), which is one step of Model.learn. Supervised MLP: wyoo5p40,
# step 7537664; self-supervised: 48ke5d6u, step 3572224; WAND: t1axv47a, step 465664.
# The losses: self-supervised and WAND as their runs (mse_specs, mse); the supervised
# MLPs MRSuite's 'concentrations' (MAE of the concentrations, macromolecules left out),
# not wyoo5p40's 'mae_all_scale' (all outputs min-max scaled).
MODELS = {
    "mlp-supervised-ismrm2016": ("2016", "mlp", "concentrations", 7_537_664),
    "mlp-selfsupervised-ismrm2016": ("2016", "mlp", "spectra", 3_572_224),
    "mlp-supervised-mrsi2024": ("mrsi", "mlp", "concentrations", 7_537_664),
    "mlp-selfsupervised-mrsi2024": ("mrsi", "mlp", "spectra", 3_572_224),
    "wand-mrsi2024": ("mrsi", "wand", "components", 465_664),
}

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
          ahead: int = 0, voxels: int = 2000, subjects: Sequence[str] = SUBJECTS) -> Any:
    """
    Train one of MODELS from scratch, saving to ``to / name`` every ``every`` steps.

    Args:
        name (str): One of MODELS.
        steps (Optional[int]): The published run's by default.
        every (Optional[int]): Steps between checkpoints; a sixteenth of the run by default,
            or the whole run where a sixteenth does not divide it.
        to (Path): Where the run's folder goes.
        batch (int): Spectra per step.
        seed (int): Seeds the training data.
        chunk (Optional[int]): Spectra per MRS-Sim start; CHUNK by default.
        ahead (int): MRS-Sim chunks simulated in the background (see mrs_sim.Source).
        voxels (int): Brain voxels the MRSI config's ranges are read from.
        subjects (Sequence[str]): Whose water and lipids WAND learns, on the MRSI challenge.

    Returns:
        The trained model, as also saved in ``to / name / step-<steps>``.
    """
    challenge, kind, objective, published = MODELS[name]
    steps = steps or published
    simulation = config(challenge, kind, voxels)
    sixteenth = max(steps // 16, 1)
    every = every or (sixteenth if steps % sixteenth == 0 else steps)
    simulator = mrs_sim.Source(simulation, seed=seed, chunk=chunk or CHUNK[challenge],
                               components=kind == "wand", ahead=ahead)
    try:
        data = simulator
        first, last = window(data.acquisition, *WINDOW[kind])
        if kind == "mlp":
            model = MLP(simulation["metabolites"], first, last, data.acquisition)
        else:
            data = Nuisance(data, subjects, seed)
            model = WAND(simulation["metabolites"], first, last, data.acquisition,
                         nuisance=["nuisance"])
        model.learn(data, steps=steps, objective=objective, batch=batch, lr=LR, every=every,
                    to=str(Path(to) / name))
    finally:
        simulator.close()                       # no chunks simulated for nobody
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
    args = parser.parse_args()
    train(args.name, steps=args.steps, every=args.every, to=args.to, seed=args.seed,
          ahead=args.ahead)
