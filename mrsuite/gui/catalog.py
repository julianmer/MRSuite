####################################################################################################
#                                            catalog.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-07                                                                              #
#                                                                                                  #
# Purpose: What MRSuite offers, for the GUI to browse: every bridge's functions, the steps among   #
#          them first, each with its arguments and help, and the arguments as the fields of a      #
#          form. Nothing here knows NiceGUI.                                                       #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import ast
import importlib
import inspect
import re
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Dict, List, Literal, Optional, Tuple, get_args, get_origin

import mrsuite
from mrsuite.bridges.base import Bridge

BRIDGES = sorted(mrsuite._BRIDGES)
MORE = "**"                    # further arguments by name, for what a form cannot name


#************#
#   module   #
#************#
def module(bridge: str) -> ModuleType:
    """A bridge's module, or a module by its path; ImportError, with how to install it, for a
    toolbox not installed."""
    return importlib.import_module(bridge if "." in bridge else f"mrsuite.bridges.{bridge}")


#***************#
#   functions   #
#***************#
def functions(bridge: str, everything: bool = False) -> List[str]:
    """
    A bridge's functions: its hand-written ones (the fits) first, then its steps, or every
    function it can dispatch.

    Args:
        bridge (str): A name of ``BRIDGES``.
        everything (bool): Every function, not only the steps.

    Returns:
        List[str]: The names.
    """
    found = module(bridge)
    own = [name for name in ("fit",) if callable(vars(found).get(name))]
    bridged = getattr(found, "bridge", None)
    names = [] if not isinstance(bridged, Bridge) else \
        bridged.available() if everything else bridged.steps()
    return own + [name for name in names if name not in own]


#*************#
#   folders   #
#*************#
def folders(bridge: str) -> Dict[str, str]:
    """The toolbox's folder each function lives in, for the bridges that know (FID-A and
    Osprey, from their files); empty for the rest."""
    bridged = getattr(module(bridge), "bridge", None)
    return bridged.folders() if hasattr(bridged, "folders") else {}


#***********#
#   about   #
#***********#
_ABOUT: Dict[Tuple[str, str], Any] = {}         # what each function said, or why it could not


def about(bridge: str, func: str) -> Tuple[Optional[inspect.Signature], str]:
    """
    A function's arguments and help: a hand-written one's own, else the bridge's account.
    Read once, a failure too: asking a toolbox can take seconds (R, MRSpecLAB), and one
    not installed fails every time it is asked.

    Raises:
        Exception: Why the toolbox could not tell, as it said the first time.
    """
    key = (bridge, func)
    if key not in _ABOUT:
        try:
            found = module(bridge)
            own = vars(found).get(func)
            _ABOUT[key] = (inspect.signature(own), inspect.getdoc(own) or "") if callable(own) \
                else found.bridge.about(func)
        except Exception as err:                       # noqa: BLE001 - kept, raised again
            _ABOUT[key] = err
    if isinstance(_ABOUT[key], Exception):
        raise _ABOUT[key]
    return _ABOUT[key]


#**********#
#   read   #
#**********#
def read(bridge: str, func: str) -> None:
    """Ask a function's arguments now, so ``about`` answers at once later; a failure is kept
    for when it is used."""
    try:
        about(bridge, func)
    except Exception:                                  # noqa: BLE001 - kept in about
        pass


#***********#
#   asked   #
#***********#
def asked(bridge: str, func: str) -> bool:
    """Whether a function's arguments have been read, so ``about`` answers at once."""
    return (bridge, func) in _ABOUT


#**************************************************************************************************#
#                                           Class Field                                            #
#**************************************************************************************************#
#                                                                                                  #
# One argument as a form shows it.                                                                 #
#                                                                                                  #
#**************************************************************************************************#
@dataclass
class Field:
    """One argument as a form shows it.

    Attributes:
        name (str): The argument's name.
        kind (str): 'choice' for one of a few values, 'pick' for one of them or one typed
            (a name), else 'value', typed as Python.
        choices (List[Any]): The values to choose from, for a choice.
        default (str): The default as text, shown while the field is left empty; ''
            where there is none.
        positional (bool): Given by position, as Octave and R take their first ones.
    """

    name: str
    kind: str
    choices: List[Any]
    default: str
    positional: bool


#************#
#   fields   #
#************#
def fields(signature: Optional[inspect.Signature]) -> List[Field]:
    """
    The arguments a form asks for: all but the data, which the pipeline hands on, and but
    the catch-alls, which the form's further arguments (``MORE``) stand for.

    Args:
        signature: The function's, None where it cannot be told.

    Returns:
        List[Field]: In the function's order.
    """
    out = []
    for p in list(signature.parameters.values())[1:] if signature else []:
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) or p.name == "nout":
            continue                  # nout: how many outputs to ask Octave for; a step gives one
        has = p.default is not p.empty
        choices = list(get_args(p.annotation)) if get_origin(p.annotation) is Literal else \
            [True, False] if isinstance(p.default, bool) else []
        out.append(Field(p.name, "choice" if choices else "value", choices,
                         repr(p.default) if has else "", p.kind == p.POSITIONAL_ONLY))
    return out


#***********#
#   paths   #
#***********#
def paths(help_: str, names: List[str]) -> List[str]:
    """The arguments a function's help says are a file, path or folder, or gives a file's
    extension, from the line that names each (``basis (str): Path to ...``)."""
    found = []
    for name in names:
        line = next((t for t in help_.splitlines()
                     if re.match(rf"\s*{re.escape(name)}\s*[(:=]", t)), "")
        if re.search(r"\b(file|files|path|folder|directory)\b|``\.\w+``", line, re.IGNORECASE):
            found.append(name)
    return found


#***********#
#   parse   #
#***********#
def parse(text: str) -> Any:
    """What is typed in a field, as Python: 5, [1, 2], 'text', None; text that is not
    Python is taken as text, so a word needs no quotes."""
    try:
        return ast.literal_eval(text.strip())
    except (ValueError, SyntaxError):
        return text.strip()
