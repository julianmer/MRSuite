# Models and training

A trained model is a method like any other: it runs on NIfTI-MRS, batches, records its
provenance and goes into a pipeline as a step. MRSuite carries two kinds, both wrapped from
their authors' code, not ported: WAND (`mrsuite.models.WAND`), a processing step that gives
spectra, and the MLPs (`mrsuite.models.MLP`), which fit.

## The zoo

The trained models live on the Hugging Face Hub, in the
[MRSuite organisation](https://huggingface.co/MRSuite).

```python
import mrsuite.models

mrsuite.models.zoo()                                    # repo id -> the class it loads as
model = mrsuite.models.load("MRSuite/wand-ismrm2016")   # any Hub repo or local folder
```

`load` picks the class from the repo's config. A model whose `acquisition` is set refuses
spectra acquired unlike its training data.

## Training

```python
model.learn(source, steps=100_000)          # or load(...) first, to fine-tune
```

`learn` runs one Keras 3 loop whatever the model's framework, on the CPU, CUDA or Apple's MPS
(`KERAS_TORCH_DEVICE`). A source is a `mrsuite.training.Source`: a simulator, a benchmark's
training split or your own data. With `to=`, it saves a checkpoint every `every` steps, each
loadable with `load`, with its recipe and the optimiser's state; `resume=` goes on from one.

## Recipes and sharing

A trained model carries the recipe it was trained with (`recipe.json`): how its training
spectra were made, reported as the MRSsynMRS minimum standard for synthetic data
(`mrsuite.training.recipe`). Pushing a model writes its config, weights, recipe and a model
card generated from them:

```python
model.push_to_hub("your-name/my-model")     # after `hf auth login`
```

How to add a model of your own, and its licence on the Hub, is under
[Contributing](contributing.md). See [Models](reference/models.md) and
[Training](reference/training.md).
