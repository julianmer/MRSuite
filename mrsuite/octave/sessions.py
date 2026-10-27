####################################################################################################
#                                           sessions.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-27                                                                              #
#                                                                                                  #
# Purpose: The Oct2Py sessions MRSuite starts, apart from the package: importing oct2py starts     #
#          an Octave, or complains on a machine without one, so only a session imports it.         #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import os
import shutil
import tempfile

from octave_kernel.kernel import STDIN_PROMPT
from oct2py import Oct2Py


#**************************************************************************************************#
#                                         Class Unprompted                                         #
#**************************************************************************************************#
#                                                                                                  #
# An Oct2Py session in which a toolbox asking for input fails at once.                             #
#                                                                                                  #
#**************************************************************************************************#
class Unprompted(Oct2Py):
    """An Oct2Py session in which a toolbox asking for input fails at once.

    oct2py answers Octave's ``input(...)`` with Python's ``input()``, which reads the
    terminal: a call MRSuite makes, from the GUI or a script, would wait there for good.
    FID-A asks so where an argument is left out (op_ecc, op_autophase on edited data).
    The answer is an Octave error, so the function stops inside Octave and the session
    is free for the next call (raising in Python instead left Octave waiting, and the
    next call was read as the answer).

    Overrides oct2py's private ``_handle_stdin``, the one path such a prompt takes.
    """

    #******************#
    #   handle stdin   #
    #******************#
    def _handle_stdin(self, line: str) -> str:
        asked = line.replace(STDIN_PROMPT, "").strip().replace("'", "''")
        return (f"error('MRSuite:prompt', 'Octave asked for input (%s): MRSuite cannot "
                f"answer it; give the argument the function asks about', '{asked}')")


#**************************************************************************************************#
#                                     Class FreshExchange                                          #
#**************************************************************************************************#
#                                                                                                  #
# An Oct2Py session that passes each call through files Octave has not seen before.                #
#                                                                                                  #
#**************************************************************************************************#
class FreshExchange(Unprompted):
    """An Oct2Py session that passes each call through files Octave has not seen before.

    oct2py hands every call over in writer.mat, rewriting one open file in place. A local
    Octave shares the host's page cache and always reads the current bytes. A container
    on a macOS host reads the bind mount through the VM's file sharing, which caches a
    file's size and contents for about a second; a call arriving within it reads the
    truncated or half-rewritten file and fails with "unable to determine file format"
    or "invalid array name subelement" - half the sessions, measured, and none once
    every call waited 1.2 s. A folder of its own per call leaves the VM no cached view.

    Overrides oct2py's private ``_feval``, the one path every call takes: oct2py derives
    both exchange files from ``temp_dir`` on each call, but opens the writer only once.
    """

    #***********#
    #   feval   #
    #***********#
    def _feval(self, *args, **kwargs):
        root = self._settings.temp_dir
        folder = tempfile.mkdtemp(dir=root)
        self._out_fh.close()
        self._out_fh = open(os.path.join(folder, "writer.mat"), "w+b")
        self._settings.temp_dir = folder
        try:
            return super()._feval(*args, **kwargs)
        finally:
            self._settings.temp_dir = root
            self._out_fh.close()
            shutil.rmtree(folder, ignore_errors=True)
