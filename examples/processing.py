####################################################################################################
#                                           processing.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-11-04                                                                              #
#                                                                                                  #
# Purpose: One in vivo chain across four languages: COWS (OpenNeuro ds006812) sub-01 through       #
#          every step Near et al. (NMR Biomed 2021) recommend, each by a different toolbox where   #
#          one offers it - Python (FSL-MRS, MRSpecLAB), MATLAB under Octave (FID-A, Osprey), R     #
#          (spant), a binary (LCModel) - then the report. Swap any step for another toolbox's:     #
#          the rest of the chain stands.                                                           #
#                                                                                                  #
#          Needs Octave, R and LCModel (fetched on first use); the subject's data (1 GB) is        #
#          fetched on first use. The basis set is the TE 26 ms sLASER set (--basis).               #
#                                                                                                  #
#          python examples/processing.py --n 2 --out out/processing                                #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import argparse
from pathlib import Path

import pandas as pd

from mrsuite import io, report
from mrsuite.benchmark import scoring
from mrsuite.bridges import fid_a, fsl_mrs, lcmodel, mrspeclab, osprey, spant
from mrsuite.core.pipeline import Pipeline
from mrsuite.datasets import openneuro

BASIS = Path("data/basis_sets/TE26_basis_summed")


#**********#
#   main   #
#**********#
def main(n: int, out: Path, basis: Path) -> None:
    """The chain on the first ``n`` spectra of sub-01, the fits and the report into ``out``."""
    spectra, water, *_ = openneuro.load("ds006812", "sub-01")     # metabolite and water
    spectra, water = spectra[:n], water[:n]
    basis = io.basis(sorted(str(f) for f in basis.glob("*.json")))

    # the water reference weights the coil combination as acquired; combined once, it feeds
    # the eddy-current correction and the fit
    reference = Pipeline([fid_a.step.op_addrcvrs()]).run(water)
    processed = Pipeline([fsl_mrs.step.coilcombine(reference=water),
                          fid_a.step.op_alignAverages(0.2, 'n'),
                          mrspeclab.step.RemoveBadAverages(),
                          spant.step.mean_dyns(),
                          osprey.step.op_eccKlose(reference),
                          spant.step.hsvd_filt(),
                          fid_a.step.op_ppmref(2.9, 3.1, 3.027),
                          fid_a.step.op_autophase(2.9, 3.1)]).run(spectra)
    fits = Pipeline([lcmodel.step.fit(basis=basis, water=reference)]).run(processed)

    out.mkdir(parents=True, exist_ok=True)
    processed.save_nifti(str(out), prefix="processed")           # the history inside
    ratios = scoring.ratios(fits)                                 # to creatine, as Near et al.
    print(pd.DataFrame([r.conc("tCr") for r in ratios]).round(3).to_string())
    print(report.report(processed, fits, path=out / "report.md")[:1500])
    print(f"... the whole report, the spectra and their history are in {out}")


#**********#
#   args   #
#**********#
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8, help="spectra of sub-01 to take (8 in all)")
    parser.add_argument("--out", type=Path, default=Path("out/processing"))
    parser.add_argument("--basis", type=Path, default=BASIS, help="a folder of FSL-MRS .json")
    args = parser.parse_args()
    main(args.n, args.out, args.basis)
