# Benchmark

Fits are scored against the ground truth with
[MRSValidate](https://github.com/JohnLaMaster/MRSValidate) (see [Install](install.md#scoring)).

```python
from mrsuite.benchmark import scoring

scores = scoring.score(fits, truth, references=["optimal", "tCr"],
                       metrics={"mosae": scoring.mae, "ccc": scoring.analyses()["agreement.lin_ccc"]})
```

`score` takes the fits by method, one per spectrum, and gives one table: every metric for
every method, reference and metabolite, and pooled over the metabolites. Every method is
scored on the same metabolites, so pooled scores compare like with like.

## References

Fits report amplitudes on a scale of their own, so they meet the truth on a reference. There
is no default, as the right one depends on the question (`scoring.REFERENCES`):

| Reference | The fits against the truth |
|---|---|
| `water` | water-scaled concentrations (mM, by each tool's own water constants) against the truth in mM, the 2016 fitting challenge's absolute comparison (Marjańska et al., MRM 2022) |
| `tCr` | both sides as ratios to total creatine, Cr + PCr, the challenge's other comparison (Near et al., NMR Biomed 2021) |
| `optimal` | each fit times its optimal scale against its truth, the comparison behind MOSAE (Merkofer et al., arXiv:2511.23135) |
| `none` | both as given, for fits already on the truth's scale |

Only single metabolites are scored; sums, a tool's own or the truth's, are not.

## Metrics

`scoring.mae` and `scoring.bias` are MRSuite's own; `mae` pooled on `optimal` is the MOSAE.
`scoring.analyses()` lists every MRSValidate analysis that compares fits with the truth:
errors, Lin's CCC, regression, Bland-Altman, normality. `scoring.figure` draws MRSValidate's
plots.

## The radar

```python
from mrsuite.benchmark import plot_scores

plot_scores(scores)                         # a corner per aspect, a polygon per method
```

Every measure is scaled from the worst method to the best, and a corner is the mean of its
aspect's measures.

## Benchmark runs

`scripts/benchmark/` runs fitters over a dataset (`run.py`), scores them (`score.py`) and
compares runs (`compare.py`), recording the environment of each run. See
[Benchmark](reference/benchmark.md).
