####################################################################################################
#                                           launcher.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-30                                                                              #
#                                                                                                  #
# Purpose: The GUI as an app with MRSuite's icon, `mrsuite-gui --install`: an app bundle in        #
#          /Applications on macOS, a menu entry on Linux, a Start-menu and desktop shortcut on     #
#          Windows. Each starts the GUI with the Python it was installed from; on macOS and Linux  #
#          with the PATH it was installed with too, which an app started from the Dock or a menu   #
#          would not have, so it finds the toolboxes the shell found.                              #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import os
import plistlib
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

NAME = "MRSuite"
STATIC = Path(__file__).resolve().parent / "static"
APPLICATIONS = Path("/Applications")    # macOS's own, which an admin may write to

# the shortcut's folders, as Windows names them, and the shortcut made in each, with this
# Python's GUI and icon passed in the environment rather than quoted into the script
_WINDOWS = """
foreach ($folder in 'Programs', 'Desktop') {
    $link = Join-Path ([Environment]::GetFolderPath($folder)) 'MRSuite.lnk'
    if ($env:MRSUITE_REMOVE) { if (Test-Path $link) { Remove-Item $link; $link }; continue }
    $shortcut = (New-Object -ComObject WScript.Shell).CreateShortcut($link)
    $shortcut.TargetPath = $env:MRSUITE_PYTHON
    $shortcut.Arguments = '-m mrsuite.gui'
    $shortcut.IconLocation = $env:MRSUITE_ICON
    $shortcut.Save()
    $link
}
"""


#*************#
#   command   #
#*************#
def command() -> List[str]:
    """The GUI from this very Python, so the app runs the environment it was installed from;
    on Windows its windowless twin, so no console opens beside the GUI."""
    python = Path(sys.executable)
    if sys.platform == "win32" and python.with_name("pythonw.exe").exists():
        python = python.with_name("pythonw.exe")
    return [str(python), "-m", "mrsuite.gui"]


#*************#
#   install   #
#*************#
def install() -> List[Path]:
    """The app on this system, made anew; what was made."""
    if sys.platform == "darwin":
        return [_mac()]
    if sys.platform == "win32":
        return _windows()
    return [_linux()]


#***************#
#   uninstall   #
#***************#
def uninstall() -> List[Path]:
    """The app taken off this system; what was removed."""
    if sys.platform == "win32":
        return _windows(remove=True)
    if sys.platform == "darwin":
        return _remove(_apps())
    entry = _entry()
    if not entry.exists():
        return []
    entry.unlink()
    return [entry]


#**********#
#   apps   #
#**********#
def _apps() -> List[Path]:
    """Where the app bundle may lie on macOS, first choice first: the Applications folder every
    user sees, else the user's own."""
    return [APPLICATIONS / f"{NAME}.app", Path.home() / "Applications" / f"{NAME}.app"]


#************#
#   remove   #
#************#
def _remove(apps: List[Path]) -> List[Path]:
    """The app bundles of these that are there, removed."""
    there = [app for app in apps if app.exists()]
    for app in there:
        shutil.rmtree(app)
    return there


#*********#
#   mac   #
#*********#
def _mac() -> Path:
    """An app bundle whose executable is a script that starts the GUI, in the first folder of
    _apps() this user may write to; any made before, there or in the other, removed."""
    from mrsuite import __version__
    _remove(_apps())
    app = next((a for a in _apps() if os.access(a.parent, os.W_OK)), _apps()[-1])
    (app / "Contents" / "MacOS").mkdir(parents=True)
    (app / "Contents" / "Resources").mkdir()
    shutil.copyfile(STATIC / "icon.icns", app / "Contents" / "Resources" / f"{NAME}.icns")
    with open(app / "Contents" / "Info.plist", "wb") as fh:
        plistlib.dump({"CFBundleName": NAME, "CFBundleDisplayName": NAME,
                       "CFBundleIdentifier": "org.mrsuite.gui", "CFBundleExecutable": NAME,
                       "CFBundleIconFile": f"{NAME}.icns", "CFBundlePackageType": "APPL",
                       "CFBundleShortVersionString": __version__,
                       "NSHighResolutionCapable": True}, fh)
    script = app / "Contents" / "MacOS" / NAME
    script.write_text(f"#!/bin/sh\nexport PATH={shlex.quote(os.environ.get('PATH', ''))}\n"
                      f"exec {shlex.join(command())}\n")
    script.chmod(0o755)
    return app


#***********#
#   entry   #
#***********#
def _entry() -> Path:
    data = os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share"
    return Path(data) / "applications" / "mrsuite.desktop"


#***********#
#   linux   #
#***********#
def _linux() -> Path:
    """A desktop entry, which the menus of GNOME, KDE and the like list."""
    def quoted(arg: str) -> str:        # the Desktop Entry Specification's quoting for Exec
        return '"' + "".join("\\" + c if c in '"`$\\' else c for c in arg) + '"'

    run = ["env", f"PATH={os.environ.get('PATH', '')}", *command()]
    entry = _entry()
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text("[Desktop Entry]\nType=Application\n"
                     f"Name={NAME}\n"
                     "Comment=MRS Processing, Analysis, AI Model Integration, and Benchmarking\n"
                     f"Exec={' '.join(quoted(arg) for arg in run)}\n"
                     f"Icon={STATIC / 'icon.png'}\nTerminal=false\n"
                     "Categories=Science;MedicalSoftware;\n")
    return entry


#*************#
#   windows   #
#*************#
def _windows(remove: bool = False) -> List[Path]:
    """Shortcuts in the Start menu and on the desktop, made by Windows' own shell object."""
    env = {**os.environ, "MRSUITE_PYTHON": command()[0], "MRSUITE_ICON": str(STATIC / "icon.ico")}
    if remove:
        env["MRSUITE_REMOVE"] = "1"
    run = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", _WINDOWS],
                         env=env, capture_output=True, text=True)
    if run.returncode != 0:
        raise RuntimeError(f"making MRSuite's shortcuts failed:\n{run.stderr.strip()}")
    return [Path(line) for line in run.stdout.splitlines() if line.strip()]
