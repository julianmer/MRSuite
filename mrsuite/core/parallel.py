####################################################################################################
#                                           parallel.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-27                                                                              #
#                                                                                                  #
# Purpose: Fitting a batch across processes. A fit is independent per spectrum, and the            #
#          interpreters behind the toolboxes are single-threaded, so the cores of a machine are    #
#          used by giving each process its own session and a share of the spectra.                 #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import importlib
import itertools
import math
import multiprocessing
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List

from nifti_mrs.nifti_mrs import NIFTI_MRS

from nifti_mrs_plus import NIfTI_MRS_Plus


#************#
#   spread   #
#************#
def spread(method: Any, items: List[NIFTI_MRS], args: tuple, kwargs: dict,
           workers: int) -> List[Any]:
    """
    Run a method on a batch over ``workers`` processes, a chunk at a time.

    The batch is cut into chunks a few times smaller than a worker's share, so that slow
    and quick spectra even out, and the results come back in the order of the batch.
    Each process keeps its own interpreter session for all the chunks it is given, and
    runs the whole call on each, provenance included.

    Spectra travel as NIfTI-MRS files rather than as objects: they cannot be pickled,
    and a file carries the header, processing history included, exactly. Further spectra,
    a water reference say, are cut into the same chunks and travel the same way.

    Scripts that use workers need the usual ``if __name__ == "__main__":`` guard, since
    each worker starts by importing the calling module.

    Args:
        method: A method importable by its ``__module__`` and ``__name__``.
        items (List[NIFTI_MRS]): The batch's spectra.
        args (tuple): The further arguments, as the method was given them.
        kwargs (dict): Likewise, by name.
        workers (int): Processes to start.

    Returns:
        List[Any]: One output per spectrum, in the order of the batch.

    Raises:
        PartlyFailed: If some chunks failed, with the outputs of the others; if all of
            them failed, the first chunk's error itself.
    """
    from mrsuite.octave import scratch

    folder = scratch("mrsuite_parallel_")
    names = itertools.count()

    def save(spectra: List[NIFTI_MRS]) -> _Files:
        paths = [os.path.join(folder, f"{next(names)}.nii.gz") for _ in spectra]
        for path, spectrum in zip(paths, spectra):
            spectrum.copy().save(path)           # nibabel repoints a saved image at its file
        return _Files(paths)

    def cut(value: Any, part: slice) -> Any:
        return save(value.list()[part]) if isinstance(value, NIfTI_MRS_Plus) else value

    try:
        size = max(1, math.ceil(len(items) / (4 * workers)))
        chunks = []
        for start in range(0, len(items), size):
            part = slice(start, start + size)
            chunks.append((method.__module__, method.__name__, save(items[part]),
                           tuple(cut(a, part) for a in args),
                           {k: cut(v, part) for k, v in kwargs.items()}))

        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            futures = [pool.submit(_run_chunk, chunk) for chunk in chunks]
        outputs, errors = {}, {}
        for start, future in zip(range(0, len(items), size), futures):
            try:
                outputs.update(enumerate(future.result(), start))
            except Exception as err:
                errors[start] = err
        if errors and not outputs:
            raise next(iter(errors.values()))
        if errors:
            raise PartlyFailed(outputs, errors) from next(iter(errors.values()))
        return [outputs[i] for i in range(len(items))]
    finally:
        shutil.rmtree(folder, ignore_errors=True)


#**************************************************************************************************#
#                                        Class PartlyFailed                                        #
#**************************************************************************************************#
#                                                                                                  #
# Some chunks of a spread batch failed; what the others gave is kept.                              #
#                                                                                                  #
#**************************************************************************************************#
class PartlyFailed(RuntimeError):
    """Some chunks of a spread batch failed; what the others gave is kept.

    Attributes:
        outputs (Dict[int, Any]): The outputs of the chunks that ran, by position in the
            batch; a position missing here belongs to a chunk that failed.
        errors (Dict[int, Exception]): Each failed chunk's error, by its first position.
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, outputs: Dict[int, Any], errors: Dict[int, Exception]) -> None:
        first = next(iter(errors.values()))
        super().__init__(f"{len(errors)} chunk(s) of the batch failed, the first with "
                         f"{type(first).__name__}: {first}")
        self.outputs, self.errors = outputs, errors


#**************************************************************************************************#
#                                           Class _Files                                           #
#**************************************************************************************************#
#                                                                                                  #
# Spectra saved for a worker, by path; read back there as one NIfTI-MRS+.                          #
#                                                                                                  #
#**************************************************************************************************#
class _Files(list):
    """Spectra saved for a worker, by path; read back there as one NIfTI-MRS+."""


#***************#
#   run chunk   #
#***************#
def _run_chunk(task) -> List[Any]:
    """In a worker: read one chunk back and run the method on it, in this process."""
    module, name, data, args, kwargs = task

    def load(value: Any) -> Any:
        if isinstance(value, _Files):
            return NIfTI_MRS_Plus([NIFTI_MRS(path, validate_on_creation=False) for path in value])
        return value

    method = getattr(importlib.import_module(module), name)
    return method(load(data), *map(load, args), **{k: load(v) for k, v in kwargs.items()})
