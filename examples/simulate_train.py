####################################################################################################
#                                         simulate_train.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-11-04                                                                              #
#                                                                                                  #
# Purpose: Simulate, train, score, report: MRS-Sim on the 2016 challenge's own basis and grid as   #
#          a training source, a QuantMLP learning from it for a few hundred steps, its checkpoint  #
#          and recipe saved, loaded back, run on the challenge's spectra as a step and scored      #
#          against the truth; its recipe printed as the MRSsynMRS table. Nothing is pushed to the  #
#          Hub: see docs/models.md for that.                                                       #
#                                                                                                  #
#          Needs MRSValidate; MRS-Sim, the QuantMLP code and the challenge data are fetched on     #
#          first use.                                                                              #
#                                                                                                  #
#          python examples/simulate_train.py --steps 50 --n 4 --out out/simulate_train             #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from mrsuite import models, mrsvalidate
from mrsuite.core.pipeline import Pipeline
from mrsuite.datasets import challenge2016
from mrsuite.models.mlp import MLP
from mrsuite.simulators import mrs_sim
from mrsuite.training import recipe


#************#
#   window   #
#************#
def window(acquisition: dict, low: float, high: float) -> tuple:
    """The spectrum's points between ``low`` and ``high`` ppm, as first and last."""
    f = np.fft.fftfreq(int(acquisition["points"]), 1 / acquisition["bandwidth"])
    ppm = acquisition["centre"] + f / acquisition["frequency"]
    points = np.flatnonzero((ppm >= low) & (ppm <= high))
    return int(points[0]), int(points[-1]) + 1


#**********#
#   main   #
#**********#
def main(steps: int, n: Optional[int], out: Path) -> None:
    """Learn ``steps`` batches from MRS-Sim, score on ``n`` challenge spectra."""
    config = challenge2016.mrs_sim_config()                    # the challenge as MRS-Sim makes it
    source = mrs_sim.Source(config, seed=0, chunk=256)
    first, last = window(source.acquisition, 0.5, 4.0)        # the model reads 0.5 to 4 ppm
    model = MLP(config["metabolites"], first, last, source.acquisition)
    losses = model.learn(source, steps=steps, to=str(out / "run"))
    print(f"mean loss {losses[-1]:.4g} over {steps} steps")

    checkpoint = out / "run" / f"step-{steps}"
    trained = models.load(str(checkpoint))                     # weights, recipe and all
    spectra, _, truth = challenge2016.load(n)
    fits = Pipeline([trained]).run(spectra)                    # a model is a step
    scores = mrsvalidate.score({"QuantMLP": fits}, truth, references=["optimal"],
                               metrics={"mosae": mrsvalidate.mae})
    print(scores.to_string())
    print(recipe.table(trained.recipe))                       # how it was trained, as MRSsynMRS
    print(f"the checkpoint and its recipe.json are in {checkpoint}")


#**********#
#   args   #
#**********#
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300, help="batches to learn from")
    parser.add_argument("--n", type=int, default=None, help="challenge spectra to score on")
    parser.add_argument("--out", type=Path, default=Path("out/simulate_train"))
    args = parser.parse_args()
    main(args.steps, args.n, args.out)
