# Datasets and simulators

## Datasets

Each dataset loads as NIfTI-MRS+, with its ground truth where it has one, and is fetched on
first use (`mrsuite.datasets`).

| Module | Dataset |
|---|---|
| `challenge2016` | the ISMRM 2016 fitting challenge: the water-suppressed spectra, their water references and the ground-truth concentrations, with the challenge's basis set |
| `mrsi_challenge` | the 2024 MRSI challenge, through Augmentrum: one single-voxel spectrum per brain voxel with its ground-truth amplitudes, and the challenge's basis set |
| `agnostic` | AGNOSTIC (Gudmundson et al.), a synthetic PRESS dataset across fields of 1.4 to 3.1 T and echo times of 10 to 80 ms |
| `openneuro` | the OpenNeuro datasets ds005371 (HERMES and HERCULES) and ds006812, one subject at a time |
| `mr_science_lab` | the MR SCIENCE Lab's public data (Columbia): MARSS-simulated basis sets for the vendors' PRESS, sLASER and STEAM, and INSPECTOR synthetic spectra |

```python
from mrsuite.datasets import challenge2016

spectra, water, truth = challenge2016.load()
```

## Simulators

`mrsuite.simulators` makes spectra and basis sets with known contents:

| Module | What it simulates |
|---|---|
| `bases` | basis sets by the toolboxes: FID-A's density-matrix simulation of ideal sequences, spant's, FSL-MRS's from its sequence description, and Vespa's and PASTIS's on PyGAMMA (PASTIS: sLASER with shaped adiabatic pulses, 1H and 31P), each recording how it was simulated |
| `spinach` | Spinach (Hogben, Kuprov et al.): 1H spin systems with their chemical shifts and J-couplings |
| `synthetic` | spectra synthesised from a basis set or a brain model: FSL-MRS's synthetic module, spant's `sim_brain_1h` |
| `mrs_sim` | MRS-Sim (LaMaster et al., NMR Biomed 2025), in a process of its own, with the whole truth |
| `digital_phantom` | MRS-Digital-Phantom (van de Sande et al.): a voxel of its digital brain |
| `mrsandbox` | MRSandbox: pulse sequences from its library or a Pulseq file, acquisitions of its virtual scanner on its virtual brain |
| `pulses` | RF pulses by FID-A's rfPulseTools |

A simulator that describes itself fills a trained model's MRSsynMRS recipe (see
[Models and training](models.md)). See [Datasets](reference/datasets.md) and
[Simulators](reference/simulators.md).
