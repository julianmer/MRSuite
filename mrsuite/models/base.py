####################################################################################################
#                                             base.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-27                                                                              #
#                                                                                                  #
# Purpose: What every trained model in MRSuite is: a pipeline step and a Hugging Face Hub model,   #
#          whatever framework it is written in. A new model says how spectra become its input,     #
#          what it computes, and what its output means; batching, the acquisition check and        #
#          provenance come with it. mrsuite.models.pytorch adapts this to PyTorch; another         #
#          framework needs only the same few lines.                                                #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional

import numpy as np

from huggingface_hub import ModelHubMixin
from nifti_mrs.nifti_mrs import NIFTI_MRS

from nifti_mrs_plus import NIfTI_MRS_Plus
from mrsuite.core import provenance
from mrsuite.core.result import FitResult

HUB = {"library_name": "mrsuite", "repo_url": "https://github.com/julianmer/MRSuite",
       "tags": ["mrs", "mrsuite"]}


#***************#
#   load file   #
#***************#
def load_file(path: Path) -> ModuleType:
    """A Python file from a model's own repository, imported by its path.

    By path rather than by package: the original repositories are not installable, and
    share top-level names such as ``utils``, so importing them as packages would clash.
    """
    name = f"mrsuite_external_{abs(hash(str(path)))}"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        sys.modules[name] = module = importlib.util.module_from_spec(spec)
        dont_write, sys.dont_write_bytecode = sys.dont_write_bytecode, True  # keep it clean
        try:
            spec.loader.exec_module(module)
        finally:
            sys.dont_write_bytecode = dont_write
    return sys.modules[name]


#*************#
#   spectra   #
#*************#
def spectra(items: List[NIFTI_MRS]) -> np.ndarray:
    """A batch as (batch, 2, points): the plain FFT of each FID, real and imaginary.

    No shift and no conjugation - the spectrum as NumPy's FFT gives it from the samples
    as NIfTI-MRS stores them.
    """
    specs = np.fft.fft(np.stack([np.asarray(i[:]).reshape(i.shape[3]) for i in items]), axis=-1)
    return np.stack([specs.real, specs.imag], axis=1)


#**************************************************************************************************#
#                                           Class Model                                            #
#**************************************************************************************************#
#                                                                                                  #
# A trained model as a pipeline step: spectra in, spectra or fit results out, in any framework.    #
#                                                                                                  #
#**************************************************************************************************#
class Model(ModelHubMixin, **HUB):
    """A trained model as a pipeline step: spectra in, spectra or fit results out.

    Nothing here depends on a framework. A subclass states its constructor arguments -
    they are its config on the Hub, so keep them to JSON values - and implements:

        prepare(items)         the spectra of a batch -> the model's input
        predict(x)             the model's output for that input, without training
        interpret(out, items)  the output -> one FitResult or NIfTI-MRS per spectrum
        fingerprint()          a checksum of the weights, for the provenance
        runtime()              the framework's versions, for the provenance

    plus the Hub's ``_save_pretrained`` and ``_from_pretrained`` to write and read its
    weights. TorchModel does all but prepare and interpret for PyTorch, where the model
    is a torch module with a ``forward``; a JAX or TensorFlow model is adapted the same
    way.

    Calling the model on spectra runs the three steps batch by batch and records what
    was run on every result:

        model = MyModel.from_pretrained("MRSuite/my-model")
        results = Pipeline([fsl_mrs.step.coilcombine(), model]).run(data)

    Attributes:
        acquisition (Dict[str, float]): What the model was trained on - 'points',
            'bandwidth' (Hz), 'frequency' (MHz) and 'centre' (ppm); spectra that differ
            are refused rather than silently misread. Empty to check nothing.
        batch_size (int): Spectra per prediction.
    """

    acquisition: Dict[str, float] = {}
    batch_size = 64

    #*************#
    #   prepare   #
    #*************#
    def prepare(self, items: List[NIFTI_MRS]) -> Any:
        """The model's input for a batch of spectra."""
        raise NotImplementedError

    #*************#
    #   predict   #
    #*************#
    def predict(self, x: Any) -> Any:
        """The model's output for one prepared batch, in inference mode."""
        raise NotImplementedError

    #***************#
    #   interpret   #
    #***************#
    def interpret(self, out: Any, items: List[NIFTI_MRS]) -> List[Any]:
        """The model's output as one FitResult or NIfTI-MRS per spectrum."""
        raise NotImplementedError

    #*****************#
    #   fingerprint   #
    #*****************#
    def fingerprint(self) -> str:
        """A SHA-256 of the weights, so the provenance pins exactly what ran."""
        raise NotImplementedError

    #*************#
    #   runtime   #
    #*************#
    def runtime(self) -> Dict[str, str]:
        """The framework's exact versions, for the provenance."""
        return {}

    #**********#
    #   call   #
    #**********#
    def __call__(self, data: Any, *args, **kwargs) -> Any:
        """Spectra run as a pipeline step; anything else goes to the framework's own call."""
        if not isinstance(data, (NIFTI_MRS, NIfTI_MRS_Plus)):
            return super().__call__(data, *args, **kwargs)

        items = data.list() if isinstance(data, NIfTI_MRS_Plus) else [data]
        for item in items:
            self.check(item)
        outs = []
        for start in range(0, len(items), self.batch_size):
            chunk = items[start:start + self.batch_size]
            outs += self.interpret(self.predict(self.prepare(chunk)), chunk)

        item = self.record()
        for out, source in zip(outs, items):
            if isinstance(out, FitResult):
                out.provenance = provenance.history(source) + [item]
            else:
                for key, value in source.hdr_ext.to_dict().items():
                    if not key.startswith("dim_") and key not in out.hdr_ext:
                        out.add_hdr_field(key, value)      # echo time, reference, history
                provenance.append(out, item)

        if not isinstance(data, NIfTI_MRS_Plus):
            return outs[0]
        return outs if isinstance(outs[0], FitResult) else NIfTI_MRS_Plus(outs)

    #***********#
    #   check   #
    #***********#
    def check(self, item: NIFTI_MRS) -> None:
        """Refuse a spectrum acquired unlike the model's training data."""
        if np.prod(item.shape) != item.shape[3]:
            raise ValueError(f"{type(self).__name__} takes single spectra, not shape "
                             f"{item.shape}; combine coils and averages first")
        found = {"points": item.shape[3], "bandwidth": 1 / item.dwelltime,
                 "frequency": item.spectrometer_frequency[0], "centre": item.SpecFreqChemShift}
        tolerance = {"points": 0, "bandwidth": 1e-3, "frequency": 1e-2, "centre": 1e-3}
        for key, expected in self.acquisition.items():
            if abs(found[key] - expected) > tolerance[key] * abs(expected):
                raise ValueError(f"{type(self).__name__} expects {key} {expected}, the data has "
                                 f"{found[key]}; resample or reference the data first")

    #************#
    #   record   #
    #************#
    def record(self) -> Dict[str, str]:
        """The provenance entry: the model, its config, its exact weights and where from."""
        details = {"model": f"{type(self).__module__}.{type(self).__qualname__}",
                   "config": getattr(self, "_hub_mixin_config", {}),
                   "weights": self.fingerprint(), "source": getattr(self, "source", None),
                   **self.runtime()}
        return provenance.entry("MRSuite", provenance.mrsuite_version(), type(self).__name__,
                                details)

    #*********************#
    #   from pretrained   #
    #*********************#
    @classmethod
    def from_pretrained(cls, name: str, *args, revision: Optional[str] = None,
                        **kwargs) -> "Model":
        """Load from the Hub or a folder, remembering which, for the provenance."""
        model = super().from_pretrained(name, *args, revision=revision, **kwargs)
        model.source = {"name": str(name), "revision": revision}
        return model
