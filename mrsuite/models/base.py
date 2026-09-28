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
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from huggingface_hub import ModelHubMixin
from nifti_mrs.nifti_mrs import NIFTI_MRS

from nifti_mrs_plus import NIfTI_MRS_Plus
from mrsuite.core import provenance
from mrsuite.core.method import Method

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
def spectra(items: Union[List[NIFTI_MRS], NIfTI_MRS_Plus]) -> np.ndarray:
    """A batch as (batch, 2, points): the plain FFT of each FID, real and imaginary.

    No shift and no conjugation - the spectrum as NumPy's FFT gives it from the samples
    as NIfTI-MRS stores them. A NIfTI-MRS+ is read as the one array it holds, which is
    how a training batch arrives: no NIfTI-MRS object per spectrum.
    """
    if isinstance(items, NIfTI_MRS_Plus):
        fids = items.numpy()
    else:
        fids = np.stack([np.asarray(i[:]) for i in items])
    specs = np.fft.fft(fids.reshape(len(fids), -1), axis=-1)
    return np.stack([specs.real, specs.imag], axis=1)


#**************************************************************************************************#
#                                           Class Model                                            #
#**************************************************************************************************#
#                                                                                                  #
# A trained model as a pipeline step: spectra in, spectra or fit results out, in any framework.    #
#                                                                                                  #
#**************************************************************************************************#
class Model(Method, ModelHubMixin, **HUB):
    """A trained model as a pipeline step: spectra in, spectra or fit results out.

    A method like any other (see ``mrsuite.core.method``), and nothing here depends on a
    framework. A subclass states its constructor arguments - they are its config on the
    Hub, so keep them to JSON values - and implements:

        prepare(items)         the spectra of a batch -> the model's input
        predict(x)             the model's output for that input, without training
        interpret(out, items)  the output -> one FitResult or NIfTI-MRS per spectrum
        fingerprint()          a checksum of the weights, for the provenance
        runtime()              adds the framework's versions to the provenance

    plus the Hub's ``_save_pretrained`` and ``_from_pretrained`` to write and read its
    weights. TorchModel does all but prepare and interpret for PyTorch, where the model
    is a torch module with a ``forward``; a JAX or TensorFlow model is adapted the same
    way.

    Calling the model on spectra runs the three steps batch by batch and records what
    was run on every result:

        model = MyModel.from_pretrained("MRSuite/my-model")
        results = Pipeline([fsl_mrs.step.coilcombine(), model]).run(data)

    To learn, a model names its ``objectives`` and implements, beside prepare:

        target(truth, data, objective)  what the output should be for a batch's truth
        loss(target, out, objective)    how far the output is from it, in keras.ops

    and its framework adapter gives ``as_keras`` and ``keras_backend``, so every model
    learns in the one Keras loop (see ``learn``).

    Attributes:
        acquisition (Dict[str, float]): What the model was trained on - 'points',
            'bandwidth' (Hz), 'frequency' (MHz) and 'centre' (ppm); spectra that differ
            are refused rather than silently misread. Empty to check nothing.
        batch_size (int): Spectra per prediction.
        objectives (Tuple[str, ...]): What the model can learn from, the first by default.
        keras_backend (Optional[str]): The Keras backend its framework learns on.
    """

    program = "MRSuite"
    batched = True
    acquisition: Dict[str, float] = {}
    batch_size = 64
    objectives: Tuple[str, ...] = ()
    keras_backend: Optional[str] = None

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
    def interpret(self, out: Any, items: List[NIFTI_MRS], **options) -> List[Any]:
        """The model's output as one FitResult or NIfTI-MRS per spectrum; ``options`` are
        the model's own, given when it is called (e.g. WAND's keep and remove)."""
        raise NotImplementedError

    #*****************#
    #   fingerprint   #
    #*****************#
    def fingerprint(self) -> str:
        """A SHA-256 of the weights, so the provenance pins exactly what ran."""
        raise NotImplementedError

    #************#
    #   target   #
    #************#
    def target(self, truth: Any, data: NIfTI_MRS_Plus, objective: str) -> Any:
        """What the model's output should be for a training batch, from its truth."""
        raise NotImplementedError

    #**********#
    #   loss   #
    #**********#
    def loss(self, target: Any, out: Any, objective: str) -> Any:
        """How far the output is from the target, as one number, in keras.ops."""
        raise NotImplementedError

    #**************#
    #   as keras   #
    #**************#
    def as_keras(self, keras: Any) -> Any:
        """The model as Keras trains it; the framework adapter's to give."""
        raise NotImplementedError

    #***********#
    #   learn   #
    #***********#
    def learn(self, source: Any, steps: int, objective: Optional[str] = None, batch: int = 16,
              lr: float = 1e-4, every: Optional[int] = None, to: Optional[str] = None
              ) -> List[float]:
        """
        Learn from a source of spectra and their truth, in one Keras loop for every framework.

        Keras settles its backend and device once per process: the first model to learn
        sets KERAS_BACKEND to its framework's, and KERAS_TORCH_DEVICE picks the device
        ('cpu', 'cuda', 'mps'; Keras' own choice when unset). The model learns in place:
        afterwards it predicts, saves and pushes to the Hub as it did before.

        Args:
            source: A ``mrsuite.training.Source``: a simulator, a benchmark's training
                split or your own data.
            steps (int): Batches to learn from.
            objective (Optional[str]): One of ``objectives``; the first by default.
            batch (int): Spectra per batch.
            lr (float): Adam's learning rate.
            every (Optional[int]): Steps between checkpoints and loss reports; must divide
                ``steps``. All of them by default.
            to (Optional[str]): Folder to save a checkpoint to every ``every`` steps, as
                ``step-<n>``, each loadable with ``from_pretrained``.

        Returns:
            List[float]: The mean loss over each stretch of ``every`` steps.

        Raises:
            ValueError: If the objective is not one the model has, or the source's
                spectra are shaped or acquired unlike the ones the model takes.
        """
        from mrsuite.training.loop import learn

        objective = objective or (self.objectives[0] if self.objectives else None)
        if objective not in self.objectives:
            raise ValueError(f"{type(self).__name__} learns from {list(self.objectives)}, "
                             f"not {objective!r}")
        if any(source.dim_tags):
            raise ValueError(f"{type(self).__name__} takes single spectra; the source gives "
                             f"{[t for t in source.dim_tags if t]} too")
        self._compare(source.acquisition)
        return learn(self, source, steps, objective, batch, lr, every, to)

    #*************#
    #   version   #
    #*************#
    def version(self) -> str:
        return provenance.mrsuite_version()

    #*************#
    #   runtime   #
    #*************#
    def runtime(self) -> Dict[str, Any]:
        """What ran, for the provenance: the model, its config, its exact weights and
        where they came from. A framework adds its versions: ``{**super().runtime(), ...}``.
        """
        return {"model": f"{type(self).__module__}.{type(self).__qualname__}",
                "config": getattr(self, "_hub_mixin_config", {}),
                "weights": self.fingerprint(), "source": getattr(self, "source", None)}

    #*********#
    #   run   #
    #*********#
    def run(self, data: NIfTI_MRS_Plus, **options) -> List[Any]:
        """Check every spectrum, then prepare, predict and interpret batch by batch."""
        items = data.list()
        for item in items:
            self.check(item)
        outs = []
        for start in range(0, len(items), self.batch_size):
            chunk = items[start:start + self.batch_size]
            outs += self.interpret(self.predict(self.prepare(chunk)), chunk, **options)
        return outs

    #**********#
    #   call   #
    #**********#
    def __call__(self, *args, **kwargs) -> Any:
        """Spectra run as a method; anything else goes to the framework's own call."""
        if args and isinstance(args[0], (NIFTI_MRS, NIfTI_MRS_Plus)):
            return Method.__call__(self, *args, **kwargs)
        return super(Method, self).__call__(*args, **kwargs)

    #***********#
    #   check   #
    #***********#
    def check(self, item: NIFTI_MRS) -> None:
        """Refuse a spectrum acquired unlike the model's training data."""
        if np.prod(item.shape) != item.shape[3]:
            raise ValueError(f"{type(self).__name__} takes single spectra, not shape "
                             f"{item.shape}; combine coils and averages first")
        self._compare({"points": item.shape[3], "bandwidth": 1 / item.dwelltime,
                       "frequency": item.spectrometer_frequency[0],
                       "centre": item.SpecFreqChemShift})

    #*************#
    #   compare   #
    #*************#
    def _compare(self, found: Dict[str, float]) -> None:
        """Refuse an acquisition - a spectrum's or a training source's - unlike the model's."""
        tolerance = {"points": 0, "bandwidth": 1e-3, "frequency": 1e-2, "centre": 1e-3}
        for key, expected in self.acquisition.items():
            if key not in found:
                raise ValueError(f"{type(self).__name__} expects {key} {expected}; the data "
                                 f"does not state its {key}")
            if abs(found[key] - expected) > tolerance[key] * abs(expected):
                raise ValueError(f"{type(self).__name__} expects {key} {expected}, the data has "
                                 f"{found[key]}; resample or reference the data first")

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
