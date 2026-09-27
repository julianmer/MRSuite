####################################################################################################
#                                            matlab.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-09-27                                                                              #
#                                                                                                  #
# Purpose: MATLAB as a runtime for the bridges, for anyone with a licence: the MATLAB Engine for   #
#          Python behind the few oct2py calls the bridges make (eval, feval, push, pull), with     #
#          values converted the way oct2py converts them, so a bridge does not know which          #
#          interpreter answered. The toolboxes then run in MATLAB itself, at its speed and with    #
#          its numerics; Octave's shims and MEX builds are not needed there.                       #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from collections.abc import Mapping
from typing import Any

import numpy as np


#******************#
#   is available   #
#******************#
def available() -> bool:
    """Whether the MATLAB Engine for Python is installed in this environment."""
    try:
        import matlab.engine                                    # noqa: F401
        return True
    except ImportError:
        return False


#***************#
#   to matlab   #
#***************#
def to_matlab(value: Any) -> Any:
    """
    A Python value as the engine should pass it, following oct2py's conventions.

    Arrays become doubles, a 1-D one as a row, as oct2py sends it; tuples become cells;
    mappings become structs, including objects that only behave like one, such as a
    NIfTI header; lists of numbers become rows, lists of anything else cells.

    Args:
        value: What a bridge hands MATLAB.

    Returns:
        Something ``matlab.engine`` accepts.
    """
    import matlab

    if value is None:
        return matlab.double([])
    if isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, float, complex, np.number)):
        return complex(value) if np.iscomplexobj(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return [to_matlab(v) for v in value.ravel()]
        if value.dtype.kind in "US":
            return [str(v) for v in value.ravel()]
        if value.size == 0:
            return matlab.double([])
        array = value if value.ndim > 1 else value.reshape(1, -1)
        complex_ = np.iscomplexobj(array)
        return matlab.double(array.astype(complex if complex_ else float).tolist(),
                             is_complex=complex_)
    if isinstance(value, tuple):
        return [to_matlab(v) for v in value]
    if isinstance(value, list):
        if value and all(isinstance(v, (int, float, np.number)) and not isinstance(v, bool)
                         for v in value):
            return to_matlab(np.asarray(value))
        return [to_matlab(v) for v in value]
    if isinstance(value, Mapping) or (hasattr(value, "keys") and hasattr(value, "__getitem__")):
        return {str(k): to_matlab(value[k]) for k in value.keys()}
    if callable(getattr(value, "to_dict", None)):              # NIfTI-MRS' Hdr_Ext
        return to_matlab(value.to_dict())
    raise TypeError(f"cannot pass a {type(value).__name__} to MATLAB")


#*****************#
#   from matlab   #
#*****************#
def from_matlab(value: Any) -> Any:
    """
    A value MATLAB returned, as oct2py would have returned it.

    Scalars become floats, arrays NumPy arrays of MATLAB's shape, structs dicts and cells
    lists.
    """
    import matlab

    if isinstance(value, dict):
        return {k: from_matlab(v) for k, v in value.items()}
    if isinstance(value, list):
        return [from_matlab(v) for v in value]
    if isinstance(value, (matlab.double, matlab.single, matlab.int8, matlab.int16,
                          matlab.int32, matlab.int64, matlab.uint8, matlab.uint16,
                          matlab.uint32, matlab.uint64, matlab.logical)):
        array = np.asarray(value)
        return array.item() if array.size == 1 else array
    return value


#**************************************************************************************************#
#                                       Class MatlabSession                                        #
#**************************************************************************************************#
#                                                                                                  #
# The part of an oct2py session the bridges use, answered by MATLAB.                               #
#                                                                                                  #
#**************************************************************************************************#
class MatlabSession:
    """The part of an oct2py session the bridges use, answered by MATLAB.

    Args:
        root (str): Toolbox directory to add recursively to MATLAB's path.
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, root: str):
        import matlab.engine
        self._engine = matlab.engine.start_matlab("-nodesktop -nosplash")
        self._engine.addpath(self._engine.genpath(root), nargout=0)

    #**********#
    #   eval   #
    #**********#
    def eval(self, code: str, nout: int = 0) -> Any:
        """Run MATLAB code; with ``nout=1``, return the value of an expression."""
        return from_matlab(self._engine.eval(code, nargout=nout))

    #***********#
    #   feval   #
    #***********#
    def feval(self, func: str, *args, nout: int = 1) -> Any:
        """Call a MATLAB function; several outputs come back as a tuple, as in oct2py."""
        out = self._engine.feval(func, *(to_matlab(a) for a in args), nargout=nout)
        return tuple(from_matlab(o) for o in out) if nout > 1 else from_matlab(out)

    #**********#
    #   push   #
    #**********#
    def push(self, name: str, value: Any) -> None:
        """Set a variable in MATLAB's workspace."""
        self._engine.workspace[name] = to_matlab(value)

    #**********#
    #   pull   #
    #**********#
    def pull(self, name: str) -> Any:
        """Read a variable from MATLAB's workspace."""
        return from_matlab(self._engine.workspace[name])

    #**********#
    #   exit   #
    #**********#
    def exit(self) -> None:
        """End the MATLAB session."""
        self._engine.quit()
