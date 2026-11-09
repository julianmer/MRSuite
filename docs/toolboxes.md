# Toolboxes

MRSuite runs other people's toolboxes as published, each through a bridge in
`mrsuite/bridges/`. A bridge makes every function of its toolbox a method and a pipeline
step; the fitters are written as methods of their own. The code a bridge runs comes from the
toolbox's own repository at the commit MRSuite pins (`mrsuite.externals.REGISTRY`), or from
its package.

| Toolbox | Module | Runs in | What MRSuite takes from it |
|---|---|---|---|
| [FID-A](https://github.com/CIC-methods/FID-A) | `mrsuite.fid_a` | Octave, MATLAB or Docker | every function as a step; basis simulation; RF pulses |
| [Osprey](https://github.com/schorschinho/osprey) | `mrsuite.osprey` | Octave, MATLAB or Docker | every function as a step; its fit |
| [Gannet](https://github.com/markmikkelsen/Gannet) | `mrsuite.bridges.gannet` | Octave, MATLAB or Docker | edited MRS (MEGA-PRESS, HERMES) through GannetLoad and GannetFit, with a T1 through co-registration and segmentation (SPM12) |
| [MRS4Brain](https://github.com/MRS4Brain/MRS4Brain-toolbox) | `mrsuite.bridges.mrs4brain` | Octave, MATLAB or Docker | a Bruker single-voxel study through its preprocessing and LCModel quantification |
| [Oryx-MRSI](https://github.com/Computational-Imaging-LAB/Oryx-MRSI) | `mrsuite.bridges.oryx` | Octave, MATLAB or Docker | metabolite maps from an LCModel-fitted MRSI exam |
| [spant](https://github.com/martin3141/spant) | `mrsuite.spant` | R or Docker | every function as a step; its fit; basis simulation; synthetic spectra |
| [FSL-MRS](https://git.fmrib.ox.ac.uk/fsl/fsl_mrs) | `mrsuite.fsl_mrs` | Python (conda) | every processing function as a step; its fit and MRSI fit; basis simulation; synthetic spectra |
| [MRSpecLAB](https://github.com/MRSEPFL/MRSpecLAB) | `mrsuite.mrspeclab` | an environment of its own (uv) | its processing nodes as steps |
| [suspect](https://github.com/openmrslab/suspect) | `mrsuite.suspect` | Python | its processing functions as steps |
| [nifti_mrs tools](https://github.com/wtclarke/nifti_mrs_tools) | `mrsuite.nifti_mrs` | Python | its dimension tools as steps |
| [Augmentrum](https://pypi.org/project/augmentrum/) | `mrsuite.augmentrum` | Python | its augmentations as steps |
| [LCModel](https://github.com/julianmer/PyLCModel) | `mrsuite.lcmodel` | a binary, through PyLCModel | its fit |
| [TARQUIN](https://github.com/julianmer/PyTARQUIN) | `mrsuite.tarquin` | a binary, through PyTARQUIN | its fit |
| [pyAMARES](https://pypi.org/project/pyAMARES/) | `mrsuite.pyamares` | Python | AMARES peak fitting from a prior-knowledge file |
| [Vespa](https://github.com/vespa-mrs/vespa) | `mrsuite.vespa` | Python | Vespa-Analysis's fit; basis simulation through PyGAMMA |
| [PASTIS](https://github.com/tngrssl/pastis) | `mrsuite.pastis` | Python, PyGAMMA built on first use | its time-domain linear-combination fit; basis simulation of its sequences (sLASER with shaped adiabatic pulses among them, 1H and 31P); transient rejection, realignment, phasing and coil combination as steps |

Each toolbox keeps its own licence; see its repository. The simulators MRSuite drives are
listed under [Datasets and simulators](datasets.md).

## Asking a toolbox

```python
from mrsuite import fid_a

fid_a.available()                       # every function it can dispatch
fid_a.about("op_alignAverages")         # its arguments and help
```

A function MRSuite never wrapped is a step all the same:
`fid_a.step.op_alignAverages(0.2, 'n')`. See [Bridges](reference/bridges.md).
