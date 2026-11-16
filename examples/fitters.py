####################################################################################################
#                                            fitters.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-11-04                                                                              #
#                                                                                                  #
# Purpose: One spectrum, every fitter: COWS (OpenNeuro ds006812) sub-01 processed once, then       #
#          fitted with the same basis by LCModel, FSL-MRS, spant, TARQUIN, Osprey, Vespa and       #
#          pyAMARES; their ratios to creatine side by side, in a table and a figure. The fit is    #
#          the step where the fitters differ; everything before it is shared.                      #
#                                                                                                  #
#          Needs Octave, R, LCModel and TARQUIN (fetched on first use), Vespa and pyAMARES (some   #
#          7 minutes a spectrum each); the subject's data (1 GB) and its TE 26 ms sLASER basis     #
#          set are fetched on first use.                                                           #
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
from mrsuite.bridges import fid_a, fsl_mrs, lcmodel, osprey, pyamares, spant, tarquin, vespa
from mrsuite.core.basis import CENTRE
from mrsuite.core.pipeline import Pipeline
from mrsuite.datasets import openneuro

SHOWN = ["NAA", "NAAG", "Glu", "Gln", "mI", "GPC", "GSH", "Lac"]


#**********#
#   main   #
#**********#
def main(n: int, out: Path) -> None:
    """The first ``n`` spectra of sub-01 through one processing and seven fitters."""
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

    # Vespa and pyAMARES take the basis functions as priors, on the data's own points: the
    # basis is as finely sampled and twice as long
    one = processed[0]
    functions = dict(zip(basis.names, basis.fids[:, :one.shape[3]]))
    out.mkdir(parents=True, exist_ok=True)
    prior = pyamares.prior(str(out / "pyamares_prior.csv"), functions, one.dwelltime,
                           one.spectrometer_frequency[0], CENTRE)

    fitters = {"LCModel": lcmodel.step.fit(basis=basis, water=reference),
               "FSL-MRS": fsl_mrs.step.fit(basis=basis, water=reference),
               "spant": spant.step.fit(basis=basis),
               "TARQUIN": tarquin.step.fit(basis=basis, water=reference,
                                           opts={"rescale_lcm_basis": False}),
               "Osprey": osprey.step.fit(basis=basis),
               "Vespa": vespa.step.fit({name: {"fid": fid} for name, fid in functions.items()}),
               # pyAMARES fits the whole sweep unless told the range the others take, starts
               # best from its Levenberg-Marquardt pass, and a prior of this many peaks
               # needs more evaluations than lmfit's 1000
               "pyAMARES": pyamares.step.fit(prior, opts={"fit_range": (0.2, 4.2),
                                                          "initialize_with_lm": True,
                                                          "fit_kws": {"max_nfev": None,
                                                                      "xtol": 1e-6,
                                                                      "ftol": 1e-6}})}
    ratios = {name: scoring.ratios(Pipeline([fit]).run(processed)) for name, fit in fitters.items()}

    # the first spectrum's ratios per fitter, the metabolites every fitter gives
    table = pd.DataFrame({name: fits[0].conc("tCr") for name, fits in ratios.items()}).dropna()
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
