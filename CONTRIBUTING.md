# Contributing to MRSuite

MRSuite runs every method the same way, whether it is a toolbox function, a fit, a trained
model or a function you wrote this morning. Anything that keeps one contract can be swapped
into a pipeline, fitted on both challenge datasets and benchmarked beside everything else.
This page shows how to add yours and how to check it.

## The contract

A method is called as `method(data, *args, **kwargs)` and:

1. takes one spectrum as NIfTI-MRS, or a batch as NIfTI-MRS+;
2. returns what it was given: one output for one spectrum; for a batch, a NIfTI-MRS+ of
   spectra or a list of `FitResult`, as long as the batch and in its order;
3. keeps the header of each spectrum it returns (echo time, reference, history) and adds
   one ProcessingApplied entry: program, exact version, method, arguments. A `FitResult`
   carries its spectrum's history plus that entry;
4. leaves its input as it was;
5. takes further spectra, such as a water reference, one per spectrum.

You write the work. MRSuite handles rules 1, 2, 3 and 5; rule 4 is up to you.

## Adding a method

**A function.** Decorate a function that takes one spectrum:

```python
import mrsuite

@mrsuite.method(program="MyLab denoiser", version="1.2")
def denoise(spectrum, strength=1.0):
    out = spectrum.copy()
    out[:] = my_denoiser(spectrum[:], strength)
    return out

denoise(batch)                                            # a batch works too
Pipeline([denoise.step(strength=2), mrsuite.fsl_mrs.step.fit(basis)])
```

Return a `FitResult` instead of a spectrum and it is an analysis method. Options:
`batched=True` if your function takes the whole batch (NIfTI-MRS+) and returns a list;
`parallel=True` to offer `workers=`; `version` and `runtime` may be functions, asked
after the work is done. The fitters in `mrsuite/bridges/` (FSL-MRS, LCModel, Osprey,
spant) are written this way.

**A class.** Subclass `mrsuite.Method` when the method holds state: set `program`,
implement `run`, and `version` and `runtime` where you can.

**A trained model.** Subclass `mrsuite.models.Model` for any framework, or
`mrsuite.models.TorchModel` for PyTorch. You implement `prepare` (spectra to input),
`forward` or `predict`, and `interpret` (output to one `FitResult` or spectrum per
input), and set `acquisition` to what the model was trained on. To make it learnable,
name its `objectives` and give `target` (what the output should be for a batch's truth)
and `loss` (in `keras.ops`); then `model.learn(source, steps=...)` trains it in one Keras
loop, whatever the framework. `MLP` and `WAND` in `mrsuite/models/` are complete examples.

**A whole toolbox.** Subclass `mrsuite.bridges.Bridge` and implement `_resolve`,
`_to_toolbox` and `_from_toolbox`. Every function of the toolbox then becomes a method.
`mrsuite/bridges/fid_a.py` (Octave) and `mrsuite/bridges/spant.py` (R) show how.

## Checking it

```python
from mrsuite.testing import check

check(denoise, batch, strength=2.0)
check(my_fit, batch, "basis.BASIS", water=references)
```

`check` runs your method on data you choose and raises an `AssertionError` that names
the rule broken. It checks that:

- one spectrum gives one output, and a batch gives one per spectrum, in order;
- each result in a batch matches the result for that spectrum alone;
- the method runs as a pipeline step;
- headers and history are kept, and one entry with a known version is added;
- the input is unchanged, and mismatched further spectra are refused.

Put the call in a test under `tests/`, as `tests/test_conformance.py` does for MRSuite's
own methods.

## Sharing a trained model

A trained model goes on the Hugging Face Hub as a repo of its own: its config, its
weights, the recipe it was trained with (`recipe.json`) and a model card generated from
them, whose MRSsynMRS table reports how the training spectra were made.

```python
model.learn(source, steps=100_000)          # or MLP.from_pretrained(...) first, to fine-tune
model.push_to_hub("your-name/my-model")     # after `hf auth login`
```

The weights' licence goes with the push, as any Hub licence id:
`model.push_to_hub("your-name/my-model", model_card_kwargs={"license": "apache-2.0"})`.

A model whose class is part of MRSuite then loads by name alone, and runs as a step in
any pipeline:

```python
model = mrsuite.models.load("your-name/my-model")
```

`load` picks the class from the repo's config; a class of your own loads with its own
`from_pretrained`, and the card says which. With its `acquisition` set, the model
refuses spectra acquired unlike its training data. A source that describes itself fills
the MRSsynMRS table (`mrsuite.simulators.mrs_sim` does); for your own source, return a
filled `mrsuite.training.recipe.report()` from its `recipe`, as `{"MRSsynMRS": report}`.

## Style and tests

Match the code around you: the file banner, class box and function box at the top of
every file, docstrings in Google style, and tests named as sentences. Run `pytest`.
Suites that need Octave, MATLAB, R or LCModel skip cleanly on a machine without them.
