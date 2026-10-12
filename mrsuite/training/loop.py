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
import copy
import os
import queue
import shutil
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from safetensors.numpy import load_file, save_file

from mrsuite.models.base import classpath

# batches prepared ahead of the one being learned from
AHEAD = 16

# Adam's state beside each checkpoint's weights, so learning resumes where it stopped
OPTIMIZER = "optimizer.safetensors"


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
          every: Optional[int], to: Optional[str], resume: Optional[str] = None) -> List[float]:
    """
    Train ``model`` on batches from ``source``; see Model.learn.

    Returns:
        List[float]: The mean loss over each stretch of ``every`` steps.
    """
    keras = keras_on(model.keras_backend)
    every = every or steps
    if steps % every:
        raise ValueError(f"every={every} does not divide steps={steps}")
    made = copy.deepcopy(source.recipe() or {})
    # where the weights start from: the model loaded, and how it was trained so far
    before = {"source": getattr(model, "source", None),
              "training": (model.recipe or {}).get("training")}
    training = {"model": classpath(type(model)), "objective": objective, "steps": 0,
                "batch": batch, "learning_rate": lr,
                "from": before if any(before.values()) else None}
    start = learned(before["training"])         # checkpoints count every step the weights took

    class Stream(keras.utils.PyDataset):
        """Endless batches, prepared as the model prepares real data, by one thread of
        their own while the model learns from the last: one thread, so they are drawn in
        the order they would be in turn. Keras asks for one batch more than it learns
        from (the first, for the data's shape), so that many are made, and the source
        stands where it would without the thread."""

        def __init__(self):
            super().__init__()
            self.ready = queue.Queue(maxsize=AHEAD)
            threading.Thread(target=self.make, daemon=True).start()

        def make(self):
            for _ in range(steps + 1):
                try:
                    data, truth = source.batch(batch, recycle=True)
                    made = model.prepare(data), model.target(truth, data, objective)
                except BaseException as err:        # raised where Keras asked for it
                    made = err
                self.ready.put(made)
                if isinstance(made, BaseException):
                    return

        @property
        def num_batches(self):
            return None

        def __getitem__(self, index):
            made = self.ready.get()
            if isinstance(made, BaseException):
                raise made
            return made

    class Keep(keras.callbacks.Callback):
        """The recipe kept up with every step learned, so it describes the weights even
        when learning stops early, and with ``to`` the model as it stands after every
        ``every`` steps, loadable by from_pretrained, recipe and all."""

        def on_train_batch_end(self, index, logs=None):
            if not training["steps"]:   # the weights are no longer the ones loaded
                model.recipe, model.source = {**made, "training": training}, None
            training["steps"] += 1

        def on_epoch_end(self, epoch, logs=None):
            if to:
                folder = Path(to) / f"step-{start + training['steps']}"
                writing = folder.with_name(folder.name + ".partial")    # step-<n> only whole
                model.save_pretrained(writing)
                state = {}
                trainable.optimizer.save_own_variables(state)
                save_file(state, str(writing / OPTIMIZER))
                shutil.rmtree(folder, ignore_errors=True)
                os.replace(writing, folder)

    trainable = model.as_keras(keras)
    trainable.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                      loss=lambda target, out: model.loss(target, out, objective))
    if resume:
        state = load_file(str(Path(resume) / OPTIMIZER))
        trainable.optimizer.build(trainable.trainable_variables)
        if len(state) != len(trainable.optimizer.variables):
            raise ValueError(f"{resume} holds {len(state)} optimizer variables, Adam on "
                             f"{type(model).__name__} has {len(trainable.optimizer.variables)}")
        trainable.optimizer.load_own_variables(state)
        trainable.optimizer.learning_rate = lr          # the rate given, not the one saved
    history = trainable.fit(Stream(), steps_per_epoch=every, epochs=steps // every,
                            shuffle=False, callbacks=[Keep()])
    return history.history["loss"]


#*************#
#   learned   #
#*************#
def learned(training: Optional[Dict[str, Any]]) -> int:
    """Steps the weights took in total: a training's own and those of what it started from;
    a converted checkpoint's training states none (see mrsuite.models.convert)."""
    if not training:
        return 0
    return training.get("steps", 0) + learned((training.get("from") or {}).get("training"))


#*********************#
#   last checkpoint   #
#*********************#
def last_checkpoint(folder: Any) -> Path:
    """The checkpoint learning wrote last into ``folder``, by its step count."""
    steps = [path for path in Path(folder).glob("step-*") if path.name[len("step-"):].isdigit()]
    return max(steps, key=lambda path: int(path.name[len("step-"):]))
