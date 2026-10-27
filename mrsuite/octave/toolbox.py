####################################################################################################
#                                            toolbox.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. M.                                                                                   #
#                                                                                                  #
# Created: 27/10/26                                                                                #
#                                                                                                  #
# Purpose: A MATLAB toolbox MRSuite runs whole, through its own entry points, rather than          #
#          function by function: Gannet, Oryx-MRSI, MRS4Brain and Spinach. Each gets an Octave     #
#          session of its own, with its code as published on the path, the toolboxes it needs     #
#          beside it, and the shims only it needs in front.                                        #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import os
from typing import Any, Dict, Optional, Sequence

from mrsuite import externals
from mrsuite.core.provenance import git_version
from mrsuite.octave import resolve_runtime, session, shim_path

HERE = os.path.dirname(os.path.abspath(__file__))


#**************************************************************************************************#
#                                          Class Toolbox                                           #
#**************************************************************************************************#
#                                                                                                  #
# A MATLAB toolbox in an Octave session of its own.                                                #
#                                                                                                  #
#**************************************************************************************************#
class Toolbox:
    """A MATLAB toolbox in an Octave session of its own.

    The session is the toolbox's alone, so the shims scoped to it (``shims/<name>``) are on
    that session's path and no other: they stand in front of Octave's own functions for this
    toolbox's calls only. The toolbox's folder goes on recursively, as its installation
    says; other externals it needs go in front of it (``front``, their top folder only, as
    SPM12 asks) or behind it (``behind``, recursively). MRSuite's own MATLAB code for a
    toolbox (``drivers/<name>``) goes on under every runtime; the shims only under Octave,
    since MATLAB has what they stand in for. Figures are drawn off screen, by gnuplot where
    Octave has it, which prints without a display.

    Args:
        external (str): The toolbox, a key of ``externals.REGISTRY``.
        program (str): Its own name, for provenance.
        shims (Optional[str]): Its folder of scoped shims, if it has one.
        drivers (Optional[str]): Its folder of MRSuite's MATLAB code, if it has one.
        front (Sequence[str]): Externals to put in front of it, their top folder only.
        behind (Sequence[str]): Externals to put behind it, recursively.
        within (str): The folder of its repository that is the toolbox, where the
            repository holds more (MRS4Brain's holds its demo data beside it).
        runtime (Optional[str]): 'local', 'docker', 'matlab' or 'auto'.
        verbose (bool): Leave the interpreter's warnings on.
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, external: str, program: str, shims: Optional[str] = None,
                 drivers: Optional[str] = None, front: Sequence[str] = (),
                 behind: Sequence[str] = (), within: str = "", runtime: Optional[str] = None,
                 verbose: bool = False):
        self.external, self.program, self.within = external, program, within
        self.shims, self.drivers = shims, drivers
        self.front, self.behind = tuple(front), tuple(behind)
        self._runtime, self._verbose = runtime, verbose
        self._octave = None
        self._interpreter: Optional[str] = None

    #**********#
    #   root   #
    #**********#
    @property
    def root(self) -> str:
        """The toolbox's folder, fetched on first use when MRSuite is installed."""
        return str(externals.ensure(self.external) / self.within)

    #*************#
    #   octave   #
    #*************#
    @property
    def octave(self) -> Any:
        """The toolbox's session, started on first use."""
        if self._octave is None:
            octave = session(self.root, runtime=self._runtime, verbose=self._verbose)
            for name in self.behind:
                octave.eval(f"addpath(genpath('{externals.ensure(name)}'), '-end');")
            for name in self.front:
                octave.eval(f"addpath('{externals.ensure(name)}');")
            if self.drivers:
                octave.eval(f"addpath('{os.path.join(HERE, 'drivers', self.drivers)}');")
            if self.shims and not self.matlab:
                octave.eval(f"addpath('{os.path.join(shim_path(), self.shims)}');")
            octave.eval("set(0, 'defaultfigurevisible', 'off');")
            if not self.matlab:
                octave.eval("if any(strcmp(available_graphics_toolkits(), 'gnuplot')), "
                            "graphics_toolkit('gnuplot'); end")
            self._octave = octave
        return self._octave

    #************#
    #   matlab   #
    #************#
    @property
    def matlab(self) -> bool:
        """Whether the toolbox runs in MATLAB rather than Octave."""
        return resolve_runtime(self._runtime) == "matlab"

    #*************#
    #   version   #
    #*************#
    def version(self) -> Optional[str]:
        """The toolbox checkout's exact state, e.g. 'v3.5.4-0-gf437bb7'."""
        return git_version(self.root)

    #*************#
    #   runtime   #
    #*************#
    def runtime(self) -> Dict[str, Any]:
        """Which interpreter ran the toolbox, and the exact state of each external beside it."""
        if self._interpreter is None:
            name = "MATLAB" if self.matlab else "Octave"
            self._interpreter = f"{name} {str(self.octave.eval('version()', nout=1)).strip()}"
        return {"runtime": resolve_runtime(self._runtime), "interpreter": self._interpreter,
                **{name: git_version(str(externals.ensure(name)))
                   for name in self.front + self.behind}}

    #*********#
    #   run   #
    #*********#
    def run(self, code: str, folder: str) -> None:
        """
        Code in the toolbox's session, run from ``folder``: the toolboxes write their
        output into the working folder, and some change it (MRS4Brain's LCModel call), so
        the session's own is put back afterwards, whatever happened.

        Args:
            code (str): Octave code.
            folder (str): The folder to run it from.
        """
        here = str(self.octave.eval("pwd()", nout=1))
        self.octave.eval(f"cd('{folder}');")
        try:
            self.octave.eval(code)
        finally:
            self.octave.eval(f"cd('{here}');")
