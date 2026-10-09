<div align="center">
  <img src="assets/logo/logo.png" alt="MRSuite Logo" width="130" style="margin-bottom: -5px;"/>
  <h1 style="margin-top: 5px; margin-bottom: 5px;">MRSuite</h1>
  <p style="margin-top: 0px;"><em>MRS Processing, Analysis, AI Model Integration, and Benchmarking</em></p>

  [![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
  [![Status](https://img.shields.io/badge/status-alpha-orange.svg)](#project-status)
</div>

---

The MRSuite project is an open-source, modular framework for magnetic resonance spectroscopy (MRS) that brings together tools for data handling, signal processing, machine learning, and reproducible reporting in a modular and extensible ecosystem.

---

## Project Status
> **Alpha Stage**  
MRSuite is currently in active development. Core architecture and modular interfaces are being defined and tested. Contributions and feedback are welcome as we build toward a stable beta release.


## Overview

MRSuite is designed to support the full MRS workflow, from raw data processing to advanced analysis. It is built around the following core principles:
- **Modularity**: Each processing and analysis step is encapsulated in reusable, interchangeable modules.
- **Flexibility**: Custom algorithms, models, and external-language tools can be easily integrated.
- **Scalability**: Capable of handling large datasets and computationally intensive workflows.
- **Extensibility**: New modules and features can be added by following a consistent interface.
- **Reproducibility**: Standardized interfaces and reporting tools ensure transparent, consistent documentation.
- **Benchmarking**: Built-in tools allow for systematic evaluation across datasets and methods.
- **Community-Driven**: Open-source development encourages contributions and collaboration.


## Key Components

### Processing Pipeline
A modular pipeline for preprocessing MRS data, including:
- **Data Import**: Load data from various formats.
- **Processing**: Choose from a collection of processing steps or add your own.
- **Reporting**: Generate reproducible reports with detailed documentation of each step.

### AI Model Zoo
A collection of (pre-trained) AI models for MRS analysis, including:
- **Interface**: A standardized interface for integrating AI models into the processing pipeline.
- **Model Management**: Tools for managing model versions, training, and retraining.
- **GPU Acceleration**: Leverage GPU capabilities for faster processing.

### Benchmarking Suite
A suite of tools for benchmarking MRS processing and analysis methods, including:
- **Performance Metrics**: Evaluate speed, accuracy, and resource usage.
- **Comparative Analysis**: Compare different methods and configurations.
- **Reporting**: Generate detailed reports on benchmarking results.
- **Visualization**: Visualize performance metrics and results.

### Graphical Interface
Build an analysis from blocks wired on a canvas: read data in, a file or a whole study folder,
preprocess, fit, check quality and
score against ground truth, with each block offering every toolbox that does its step (grouped
after Near et al. and Kreis et al., NMR Biomed 2021) and every function of every toolbox besides.
Each block shows what it made and exports it, and the whole graph is written out as the Python
that makes it again:

```bash
uvx --from "mrsuite[gui] @ git+https://github.com/julianmer/MRSuite" mrsuite-gui
```

It opens a window of its own, or the browser where none can open (`--browser` asks for it);
`mrsuite-gui --check` says which toolboxes this machine has.

### Reporting Standards
A set of standards for generating reproducible reports, including:
- **Documentation**: Automatically document each step of the processing pipeline.
- **Contribution List**: Construct a list of contributors for the utilized modules.
- **Version Control**: Track changes to modules versions and hardware specifications.
- **Contribution Guidelines**: Clear guidelines for contributing new modules and improvements.


## External Tools and Their Licences
MRSuite does not ship the external tools it drives: each is fetched separately and keeps its
own licence. A checkout has them as submodules; an installed MRSuite clones each the first time
it is used, at the commit MRSuite pins, into `~/.mrsuite` (or `$MRSUITE_HOME`). MRS-Sim, for example, is licensed under AGPL-3.0; MRSuite runs it in a process of
its own and never bundles it.
