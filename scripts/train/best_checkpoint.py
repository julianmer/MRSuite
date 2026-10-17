####################################################################################################
#                                        best_checkpoint.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-16                                                                              #
#                                                                                                  #
# Purpose: Chooses a training run's best checkpoint, as the published runs kept theirs (a          #
#          ModelCheckpoint on validation): every saved checkpoint of an MLP run scored by MOSAE    #
#          on held-out spectra - fresh MRS-Sim spectra seeded apart from training, or the          #
#          release's validation subjects for a release-trained run - and the best written to       #
#          data/runs/<name>/best.json, which the benchmark takes in place of the last.             #
#                                                                                                  #
#          Usage: python scripts/train/best_checkpoint.py <name> [--n 1024]                        #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from challenge_models import MODELS, RELEASE, RUNS, config
from mrsuite import models
from mrsuite.benchmark import scoring
from mrsuite.datasets import mrsi_challenge
from mrsuite.simulators import mrs_sim

# validation spectra seeded far from every seed training and the suite draw
SEED = 2 * 10 ** 9


#***********#
#   score   #
#***********#
def score(name: str, n: int = 1024, runs: Path = RUNS) -> Dict[str, float]:
    """
    MOSAE of every saved checkpoint of an MLP run on ``n`` held-out spectra.

    Args:
        name (str): One of challenge_models.MODELS, an MLP's.
        n (int): Validation spectra.
        runs (Path): Where the runs are.

    Returns:
        Dict[str, float]: Checkpoint folder name to its MOSAE, in step order.
    """
    challenge, kind, _, _ = MODELS[name]
    if kind != "mlp":
        raise ValueError(f"{name} is not an MLP: WAND has no validation score here yet, so the "
                         "benchmark takes its last checkpoint")
    if challenge in RELEASE:
        source = mrsi_challenge.Release(mrsi_challenge.VALIDATION, RELEASE[challenge], seed=SEED)
    else:
        source = mrs_sim.Source(config(challenge.removesuffix("-clean"), kind), seed=SEED,
                                chunk=n, ahead=0)
    try:
        data, truth = source.batch(n)
    finally:
        getattr(source, "close", lambda: None)()
    truths = [{m: float(v[i]) for m, v in truth.concentrations.items()} for i in range(n)]
    checkpoints = sorted((p for p in (Path(runs) / name).glob("step-*")
                          if p.name[len("step-"):].isdigit()),
                         key=lambda p: int(p.name[len("step-"):]))
    found = {}
    for folder in checkpoints:
        fits = models.load(str(folder)).run(data)
        table = scoring.score({name: fits}, truths, ["optimal"], {"mosae": scoring.mae})
        found[folder.name] = float(table.query("metabolite == 'all'")["value"].iloc[0])
        print(f"{folder.name}: MOSAE {found[folder.name]:.4f}", flush=True)
    return found


#**********#
#   main   #
#**********#
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="an MLP run of challenge_models.MODELS")
    parser.add_argument("--n", type=int, default=1024, help="validation spectra")
    args = parser.parse_args(argv)

    found = score(args.name, args.n)
    finite = [k for k, v in found.items() if math.isfinite(v)]   # a diverged one is NaN
    if not finite:
        raise SystemExit(f"no checkpoint of {args.name} with a finite MOSAE: {found}")
    best = min(finite, key=found.get)
    record = {"checkpoint": best, "mosae": found[best],
              "measure": f"MOSAE (optimal scale, pooled) on {args.n} held-out spectra, "
                         f"seed {SEED}", "scores": found}
    (RUNS / args.name / "best.json").write_text(json.dumps(record, indent=2))
    print(f"best: {best} (MOSAE {found[best]:.4f}) of {len(found)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
