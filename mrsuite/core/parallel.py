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
import functools
import importlib
import math
import multiprocessing
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, List

from nifti_mrs_plus import NIfTI_MRS_Plus


#********************#
#   over processes   #
#********************#
def over_processes(fit: Callable) -> Callable:
    """
    Give a fit a ``workers`` option that spreads a batch over that many processes.

    The batch is cut into chunks a few times smaller than a worker's share, so that slow
    and quick spectra even out, and the results come back in the order of the batch.
    Each process keeps its own interpreter session for all the chunks it is given.

    Spectra travel as NIfTI-MRS files rather than as objects: they cannot be pickled,
    and a file carries the header, processing history included, exactly.

    Scripts that use workers need the usual ``if __name__ == "__main__":`` guard, since
    each worker starts by importing the calling module.
    """
    @functools.wraps(fit)
    def wrapped(data: Any, *args, workers: int = 1, **kwargs) -> Any:
        items = data.list() if isinstance(data, NIfTI_MRS_Plus) else None
        if workers <= 1 or items is None or len(items) < 2:
            return fit(data, *args, **kwargs)

        from mrsuite.octave import scratch

        folder = scratch("mrsuite_parallel_")
        try:
            size = max(1, math.ceil(len(items) / (4 * workers)))
            chunks = []
            for start in range(0, len(items), size):
                paths = [os.path.join(folder, f"{i}.nii.gz")
                         for i in range(start, min(start + size, len(items)))]
                for path, item in zip(paths, items[start:start + size]):
                    item.copy().save(path)      # nibabel repoints a saved image at its file
                chunks.append((fit.__module__, fit.__name__, paths, args, kwargs))

            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
                return [r for part in pool.map(_fit_chunk, chunks) for r in part]
        finally:
            shutil.rmtree(folder, ignore_errors=True)

    return wrapped


#***************#
#   fit chunk   #
#***************#
def _fit_chunk(task) -> List[Any]:
    """In a worker: read one chunk back and fit it with the undecorated fit."""
    from fsl_mrs.utils import mrs_io

    module, name, paths, args, kwargs = task
    fit = getattr(importlib.import_module(module), name).__wrapped__
    return fit(NIfTI_MRS_Plus([mrs_io.read_FID(p) for p in paths]), *args, **kwargs)
