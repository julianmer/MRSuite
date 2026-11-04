####################################################################################################
#                                            fitters.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-11-04                                                                              #
#                                                                                                  #
# Purpose: One spectrum, every fitter: COWS (OpenNeuro ds006812) sub-01 processed once, then       #
#          fitted with the same basis by LCModel, FSL-MRS, spant, TARQUIN and Osprey; their ratios #
#          to creatine side by side, in a table and a figure. The fit is the step where the        #
#          fitters differ; everything before it is shared.                                         #
#                                                                                                  #
#          Needs Octave, R, LCModel and TARQUIN (fetched on first use); the subject's data (1 GB)  #
#          and its TE 26 ms sLASER basis set are fetched on first use.                             #
#                                                                                                  #
#          python examples/fitters.py --n 1 --out out/fitters                                      #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import argparse
from pathlib import Path

import pandas as pd

from mrsuite.benchmark import scoring
from mrsuite.bridges import fid_a, fsl_mrs, lcmodel, osprey, spant, tarquin
from mrsuite.core.pipeline import Pipeline
from mrsuite.datasets import openneuro

SHOWN = ["NAA", "NAAG", "Glu", "Gln", "mI", "GPC", "GSH", "Lac"]


#**********#
#   main   #
#**********#
def main(n: int, out: Path) -> None:
    """The first ``n`` spectra of sub-01 through one processing and five fitters."""
    spectra, water, *_ = openneuro.load("ds006812", "sub-01")
    spectra, water = spectra[:n], water[:n]
    basis = openneuro.basis("ds006812")                         # TE 26 ms sLASER

    # the Twix is sampled twice as fast as the basis: both go onto the basis' grid, since
    # TARQUIN takes the data as it is
    reference = Pipeline([fid_a.step.op_addrcvrs(), spant.step.downsample_mrs_td()]).run(water)
    processed = Pipeline([fsl_mrs.step.coilcombine(reference=water),
                          spant.step.downsample_mrs_td(),
                          fid_a.step.op_alignAverages(0.2, 'n'),
                          fid_a.step.op_rmbadaverages(3.0, 't'),
                          spant.step.mean_dyns(),
                          osprey.step.op_eccKlose(reference),
                          spant.step.hsvd_filt(),
                          fid_a.step.op_ppmref(2.9, 3.1, 3.027),
                          fid_a.step.op_autophase(2.9, 3.1)]).run(spectra)

    fitters = {"LCModel": lcmodel.step.fit(basis=basis, water=reference),
               "FSL-MRS": fsl_mrs.step.fit(basis=basis, water=reference),
               "spant": spant.step.fit(basis=basis),
               "TARQUIN": tarquin.step.fit(basis=basis, water=reference,
                                           opts={"rescale_lcm_basis": False}),
               "Osprey": osprey.step.fit(basis=basis)}
    ratios = {name: scoring.ratios(Pipeline([fit]).run(processed)) for name, fit in fitters.items()}

    # the first spectrum's ratios per fitter, the metabolites every fitter gives
    table = pd.DataFrame({name: fits[0].conc("tCr") for name, fits in ratios.items()}).dropna()
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "ratios.csv")
    ax = table.reindex([m for m in SHOWN if m in table.index]).plot.bar(figsize=(8, 3.5),
                                                                        ylabel="ratio to tCr")
    ax.figure.tight_layout()
    ax.figure.savefig(out / "ratios.svg")
    print(table.round(3).to_string())
    print(f"the table and the figure are in {out}")


#**********#
#   args   #
#**********#
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1, help="spectra of sub-01 to take (8 in all)")
    parser.add_argument("--out", type=Path, default=Path("out/fitters"))
    args = parser.parse_args()
    main(args.n, args.out)
