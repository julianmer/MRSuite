<div align="center">
  <img src="https://raw.githubusercontent.com/julianmer/MRSuite/main/assets/logo/logo.png" alt="MRSuite Logo" width="130" style="margin-bottom: -5px;"/>
  <h1 style="margin-top: 5px; margin-bottom: 5px;">MRSuite</h1>
  <p style="margin-top: 0px;"><em>MRS Processing, Analysis, AI Model Integration, and Benchmarking</em></p>

  [![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
</div>

---

The MRSuite project is an open-source, modular framework for magnetic resonance spectroscopy (MRS) that brings together tools for data handling, signal processing, machine learning, and reproducible reporting in a modular and extensible ecosystem.

---

## Overview

MRSuite is designed to support the full MRS workflow, from raw data processing to advanced analysis. It is built around the following core principles:
- **Modularity**: Each processing and analysis step is encapsulated in reusable, interchangeable modules.
- **Flexibility**: Custom algorithms, models, and external-language tools can be easily integrated.
- **Scalability**: Capable of handling large datasets and computationally intensive workflows.
- **Extensibility**: New modules and features can be added by following a consistent interface.
- **Reproducibility**: Standardized interfaces and reporting tools ensure transparent, consistent documentation.
- **Benchmarking**: Built-in tools allow for systematic evaluation across datasets and methods.
- **Community-Driven**: Open-source development encourages contributions and collaboration.


## Installation

MRSuite needs Python 3.11 or newer:

```bash
pip install mrsuite
```

The toolboxes MRSuite drives come in as you use them: what pip cannot install (Octave or
MATLAB, R, FSL-MRS) is named when a step first needs it, and everything else is fetched on
first use, each under its own licence (FSL-MRS, for one, is for non-commercial use only).


## Python

```python
from mrsuite import fsl_mrs, io, lcmodel
from mrsuite.benchmark import scoring
from mrsuite.core.pipeline import Pipeline
from mrsuite.datasets import challenge2016

spectra, water, truth = challenge2016.load()            # fetched on first use
basis = io.basis(challenge2016.BASIS)
fits = {"LCModel": Pipeline([lcmodel.step.fit(basis=basis)]).run(spectra),
        "FSL-MRS": Pipeline([fsl_mrs.step.fit(basis=basis)]).run(spectra)}
scores = scoring.score(fits, truth, references=["optimal"], metrics={"mosae": scoring.mae})
```

A step is any toolbox's function: `fid_a.step.op_alignAverages(0.2, 'n')`,
`spant.step.hsvd_filt()`, `osprey.step.op_eccKlose(water)`. Swap one for another and the
rest of the chain stands. Scoring runs on
[MRSValidate](https://github.com/JohnLaMaster/MRSValidate), which is not on PyPI yet:
`pip install git+https://github.com/JohnLaMaster/MRSValidate.git`.


## Key Components

### Processing Pipeline
A modular pipeline for preprocessing MRS data, including:
- **Data Import**: Load data from various formats.
- **Processing**: Choose from a collection of processing steps or add your own.
- **Provenance**: Every step is recorded in the data's own processing history (NIfTI-MRS): the
  toolbox, its exact version and every argument.

### AI Model Zoo
A collection of (pre-trained) AI models for MRS analysis, including:
- **Interface**: A standardized interface for integrating AI models into the processing pipeline.
- **Training**: Train or retrain a model in MRSuite (`model.learn`, one Keras 3 loop, on the CPU,
  CUDA or Apple's MPS), its recipe recorded with it.
- **Sharing**: Push a model to the Hugging Face Hub and load anyone's by its name.

### Benchmarking Suite
A suite of tools for benchmarking MRS processing and analysis methods, including:
- **Accuracy and agreement**: MRSuite's own errors (MAE, MOSAE) and every analysis of
  [MRSValidate](https://github.com/JohnLaMaster/MRSValidate) that compares fits with the truth:
  errors, Lin's CCC, regression, Bland-Altman, normality; per metabolite and pooled.
- **Comparative Analysis**: Compare different methods and configurations on the same data.
- **Score tables**: One table per measure, by method, reference and metabolite.
- **Visualization**: A radar of the methods, a corner per aspect, and MRSValidate's plots.

### Graphical Interface

```bash
mrsuite-gui
```

Wire blocks on a canvas. One block per step, each offering every toolbox that does it, and
every function of every toolbox besides. Each block shows what it made, exports it, and
reads out its NIfTI-MRS header and history. The graph is written out as the Python that
makes it again.

![In vivo processing: an OpenNeuro dataset through FSL-MRS, FID-A, spant and Osprey into LCModel](https://raw.githubusercontent.com/julianmer/MRSuite/main/assets/gui/processing.png)

*In vivo processing: coils, alignment, outliers, averaging, eddy currents, water, referencing
and phasing by four toolboxes, then LCModel with the water reference.*

![A benchmark: the 2016 fitting challenge into LCModel, FSL-MRS, spant and WAND, scored on five aspects](https://raw.githubusercontent.com/julianmer/MRSuite/main/assets/gui/benchmark.png)

*A benchmark: the 2016 fitting challenge into four fitters, scored on five aspects against
the truth and drawn on a radar.*

![The same processing as the Python it writes](https://raw.githubusercontent.com/julianmer/MRSuite/main/assets/gui/script.png)

*The same processing as the Python it writes.*

### Reporting Standards
A set of standards for generating reproducible reports, including:
- **Documentation**: Each step of the processing pipeline is documented automatically, in the
  data's history.
- **Training recipes**: A trained model carries how its training spectra were made, reported as
  the MRSsynMRS minimum standard for synthetic data.
- **Contribution Guidelines**:
  [CONTRIBUTING.md](https://github.com/julianmer/MRSuite/blob/main/CONTRIBUTING.md) says how to
  add a method, check it, and share a trained model.

