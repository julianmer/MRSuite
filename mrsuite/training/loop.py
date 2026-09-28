####################################################################################################
#                                             loop.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-28                                                                              #
#                                                                                                  #
# Purpose: Learning from a source, as one Keras loop for every framework. Keras' fit runs on JAX,  #
#          TensorFlow and PyTorch, and a model joins it through its framework adapter's            #
#          as_keras. Keras settles its backend and device once per process, from KERAS_BACKEND     #
#          and KERAS_TORCH_DEVICE, so learning does too.                                           #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import os
import sys
from pathlib import Path
from typing import Any, List, Optional


#**************#
#   keras on   #
#**************#
def keras_on(backend: str) -> Any:
    """
    Keras on ``backend``, imported now if nothing imported it yet.

    Args:
        backend (str): The Keras backend the model's framework needs, e.g. 'torch'.

    Returns:
        The keras module.

    Raises:
        RuntimeError: If Keras already runs on another backend in this process, which
            Keras cannot change once imported.
    """
    if "keras" not in sys.modules:
        os.environ.setdefault("KERAS_BACKEND", backend)
    import keras
    if keras.backend.backend() != backend:
        raise RuntimeError(f"Keras runs on {keras.backend.backend()} in this process; a {backend} "
                           f"model learns in one started with KERAS_BACKEND={backend}")
    return keras


#***********#
#   learn   #
#***********#
def learn(model: Any, source: Any, steps: int, objective: str, batch: int, lr: float,
          every: Optional[int], to: Optional[str]) -> List[float]:
    """
    Train ``model`` on batches from ``source``; see Model.learn.

    Returns:
        List[float]: The mean loss over each stretch of ``every`` steps.
    """
    keras = keras_on(model.keras_backend)
    every = every or steps
    if steps % every:
        raise ValueError(f"every={every} does not divide steps={steps}")

    class Stream(keras.utils.PyDataset):
        """Endless batches, prepared as the model prepares real data."""

        @property
        def num_batches(self):
            return None

        def __getitem__(self, index):
            data, truth = source.batch(batch)
            return model.prepare(data), model.target(truth, data, objective)

    class Keep(keras.callbacks.Callback):
        """The model as it stands after every ``every`` steps, loadable by from_pretrained."""

        def on_epoch_end(self, epoch, logs=None):
            model.save_pretrained(Path(to) / f"step-{(epoch + 1) * every}")

    trainable = model.as_keras(keras)
    trainable.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                      loss=lambda target, out: model.loss(target, out, objective))
    history = trainable.fit(Stream(), steps_per_epoch=every, epochs=steps // every,
                            shuffle=False, callbacks=[Keep()] if to else [])
    return history.history["loss"]
