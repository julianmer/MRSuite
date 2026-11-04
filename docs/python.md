# Python

The README's example fits the 2016 fitting challenge with two fitters and scores them:

```python
from mrsuite import io, lcmodel, tarquin
from mrsuite.benchmark import scoring
from mrsuite.core.pipeline import Pipeline
from mrsuite.datasets import challenge2016

spectra, water, truth = challenge2016.load()            # fetched on first use
basis = io.basis(challenge2016.BASIS)
fits = {"LCModel": Pipeline([lcmodel.step.fit(basis=basis)]).run(spectra),
        "TARQUIN": Pipeline([tarquin.step.fit(basis=basis)]).run(spectra)}
scores = scoring.score(fits, truth, references=["optimal"], metrics={"mosae": scoring.mae})
```

## Steps

A step is anything callable that takes the data and returns it: a toolbox's function, your
own function, or `bridge.step.f(args)`, which binds the arguments and takes the data later.

```python
from mrsuite import fid_a, fsl_mrs

water = Pipeline([fsl_mrs.step.coilcombine()]).run(water)

process = Pipeline([fsl_mrs.step.coilcombine(),
                    fid_a.step.op_rmbadaverages(4, 't'),
                    fsl_mrs.step.ecc(water)])
analyse = Pipeline([fsl_mrs.step.fit(basis, water)])     # spectra -> results

results = Pipeline([process, analyse]).run(data)         # a pipeline is a step
```

Every function of a bridged toolbox is a step, not only the ones MRSuite wraps:
`fid_a.step.op_alignAverages(0.2, 'n')`, `spant.step.hsvd_filt()`,
`osprey.step.op_eccKlose(water)`. Swap one for another and the rest of the chain stands.
What a step returns flows on, so a pipeline carries spectra, a basis set or fit results
alike; a fit is the step where spectra become results.

The data is NIfTI-MRS: one spectrum, or a batch as NIfTI-MRS+. A method written for one
spectrum takes a batch too, and gives one output per spectrum in order; one that takes the
whole batch says so (`batched=True`, see [Contributing](contributing.md)).

## Provenance

Every step adds one entry to the data's own processing history (NIfTI-MRS
`ProcessingApplied`): the toolbox, its exact version and every argument. A fit result carries
the history of its spectrum plus the fit's entry. External toolboxes are recorded by the
commit MRSuite pins (`mrsuite.externals.version`).

## The pipeline as Python

`Pipeline.script()` writes a pipeline as the Python that builds it, a function `pipeline`
taking the spectra its steps were given by name:

```python
pipeline = Pipeline([fsl_mrs.step.coilcombine(), fsl_mrs.step.ecc(water)])
print(pipeline.script(inputs={"water": water}))
```

The GUI writes its graphs out the same way.

## The report

```python
from mrsuite.report import report

print(report(spectra, fits))                # or report(spectra, fits, path="report.md")
```

`report` writes the MRSinMRS checklist (Lin et al., NMR Biomed 2021) item by item from what the
spectra and fits record: their NIfTI-MRS header and processing history. What they do not record
reads "not recorded". Then every step with its toolbox, version and arguments, the MRSsynMRS table
of each trained model in the chain, and the references of every tool used. A batch is one report;
an item that differs across it lists each spectrum's value. In the GUI, a block's report button
shows the same.

## Examples

One script per claim in `examples/`, each taking `--n` for a small run and saying what it
needs at the top; the two on COWS take the TE 26 ms basis set (`--basis`), the rest fetch
everything on first use.

| Script | Shows |
|:---|:---|
| `processing.py` | one in vivo chain across four languages (COWS sub-01), then the report |
| `fitters.py` | one processed spectrum fitted by LCModel, FSL-MRS, spant, TARQUIN and Osprey |
| `benchmark.py` | the 2016 challenge into three fitters and WAND then LCModel, scored on a radar |
| `mrsi.py` | the 2024 MRSI challenge slice: augmentation, two fits, maps over the truth |
| `simulate_train.py` | MRS-Sim into a QuantMLP, saved with its recipe, scored on the challenge |

```bash
python examples/benchmark.py --n 4 --out out/benchmark
```

## Your own method

`@mrsuite.method` makes a function of one spectrum a method that batches, records its
provenance and runs as a step; `mrsuite.testing.check` checks one against the contract. See
[Contributing](contributing.md).

See [Core](reference/core.md) for `Pipeline`, `Method` and `FitResult`.
