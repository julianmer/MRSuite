####################################################################################################
#                                           benchmark.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-11-04                                                                              #
#                                                                                                  #
# Purpose: Interchangeable fitters on one benchmark: the ISMRM 2016 fitting challenge into         #
#          LCModel, FSL-MRS and spant with the challenge's basis, and WAND from the zoo (artifacts #
#          removed) before LCModel; every fit scored against the truth on MRSValidate's five       #
#          aspects, the scores on one radar. Any method wired in is scored the same way.           #
#                                                                                                  #
#          Needs LCModel (fetched on first use), R with spant, and MRSValidate; the challenge      #
#          data and the WAND checkpoint are fetched on first use.                                  #
#                                                                                                  #
#          python examples/benchmark.py --n 4 --out out/benchmark                                  #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import argparse
from pathlib import Path
from typing import Optional

from mrsvalidate.analysis import agreement, descriptive, method_comparison, regression

from mrsuite import io, models
from mrsuite.benchmark import scoring
from mrsuite.benchmark.radar import plot_scores
from mrsuite.bridges import fsl_mrs, lcmodel, spant
from mrsuite.core.pipeline import Pipeline
from mrsuite.datasets import challenge2016

# one metric per aspect of MRSValidate, each a corner of the radar
ASPECTS = {"mosae": scoring.mae, "lin_ccc": agreement.lin_ccc, "pearson_r": regression.pearson_r,
           "bland_altman": method_comparison.bland_altman, "mean": descriptive.mean}


#**********#
#   main   #
#**********#
def main(n: Optional[int], out: Path) -> None:
    """The first ``n`` challenge spectra through four methods, scored; the radar into ``out``."""
    spectra, _, truth = challenge2016.load(n)                   # fetched on first use
    basis = io.basis(challenge2016.BASIS)

    wand = models.load("MRSuite/wand-ismrm2016").step(remove=["artifacts"])
    fits = {"LCModel": Pipeline([lcmodel.step.fit(basis=basis)]).run(spectra),
            "FSL-MRS": Pipeline([fsl_mrs.step.fit(basis=basis)]).run(spectra),
            "spant": Pipeline([spant.step.fit(basis=basis)]).run(spectra),
            "WAND then LCModel": Pipeline([wand, lcmodel.step.fit(basis=basis)]).run(spectra)}

    scores = scoring.score(fits, truth, references=["optimal"], metrics=ASPECTS)
    out.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out / "scores.csv", index=False)
    plot_scores(scores).figure.savefig(out / "radar.svg")
    print(scores.query("metric == 'mosae' and metabolite == 'all'").to_string())
    print(f"the score table and the radar are in {out}")


#**********#
#   args   #
#**********#
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None, help="challenge spectra to take (28 in all)")
    parser.add_argument("--out", type=Path, default=Path("out/benchmark"))
    args = parser.parse_args()
    main(args.n, args.out)
