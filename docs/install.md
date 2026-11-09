# Install

MRSuite needs Python 3.11 or newer:

```bash
pip install mrsuite
```

That brings MRSuite, its GUI and the Python toolboxes it drives. The rest come in as a step
first needs them, each under its own licence.

## What comes on first use

Other people's code MRSuite runs, such as FID-A, Osprey, BasiC, MRS-Sim and the published
models' repositories, is not in the wheel. Each is fetched the first time it is used: GitHub's
archive of the exact commit MRSuite pins, into `$MRSUITE_HOME` (else `~/.mrsuite`), or into
`externals/` in a source checkout. The pins are `mrsuite.externals.REGISTRY`; see
[Externals](reference/externals.md).

The LCModel and TARQUIN binaries download on first use too, through `lcmodel-wrapper` and
`tarquin-wrapper`.

## Runtimes

| Toolboxes | Runtime | How MRSuite finds it |
|---|---|---|
| FID-A, Osprey, Gannet, MRS4Brain, Oryx-MRSI, OXSA, Spinach, SimnTG | Octave, MATLAB or Docker | `MRSUITE_OCTAVE_RUNTIME`: `auto` (default: `octave-cli` on `PATH`, else Docker), `local`, `docker` or `matlab` |
| spant | R, or Docker | `MRSUITE_R_RUNTIME`: `auto` (default: `Rscript` on `PATH`, else Docker), `local` or `docker` |
| FSL-MRS | its conda package | imported when a step first needs it |
| MRSpecLAB | an environment of its own | built with `uv` on first use, or `MRSUITE_MRSPECLAB_PYTHON` |

Each says what is missing, and how to get it, when a step first needs it, for instance:

- FSL-MRS: `conda install -c conda-forge -c https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/ fsl_mrs`
- MATLAB: `pip install <matlabroot>/extern/engines/python` (the MATLAB Engine for Python)
- Octave: "no Octave available: install one, or start Docker for the containerised runtime"
- pyAMARES on Apple Silicon: `pip install git+https://github.com/hawkMRS/pyAMARES` (its PyPI
  release needs hlsvdpro, which has no wheel for it, so `pip install mrsuite` leaves it out there)

The rules are `mrsuite.octave.resolve_runtime` and `mrsuite.r`; the Docker images are built
from `mrsuite/octave/dockerfile` and `mrsuite/r/dockerfile`.

## What this machine has

```bash
mrsuite-gui --check
```

says what the GUI and the toolboxes find here, and starts nothing.

## The GUI as an app

```bash
mrsuite-gui --install
```

makes the GUI an app with MRSuite's icon, for the Python it was installed from: in
`/Applications` on macOS (else `~/Applications`), as a menu entry on Linux, and as a
Start-menu and desktop shortcut on Windows. `mrsuite-gui --uninstall` takes it away. See
`mrsuite/gui/launcher.py`.

## Scoring

Scoring runs on [MRSValidate](https://github.com/JohnLaMaster/MRSValidate):

```bash
pip install git+https://github.com/JohnLaMaster/MRSValidate.git
```
