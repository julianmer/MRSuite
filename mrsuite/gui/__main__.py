####################################################################################################
#                                           __main__.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-07                                                                              #
#                                                                                                  #
# Purpose: Starts the GUI: `mrsuite-gui`, or `uvx --from "mrsuite @ git+<repo>" mrsuite-gui`.       #
#          A window of its own where one can open, else the browser; `--check` says what this      #
#          machine has for MRSuite's toolboxes, and starts nothing; `--install` makes it an app    #
#          with MRSuite's icon (see launcher.py), `--uninstall` takes that away.                   #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import argparse
import gc
import importlib.util
import os
import shutil
import sys
from typing import Dict, List, Optional


#************#
#   native   #
#************#
def native() -> bool:
    """Whether a window of its own can open here: pywebview, and on Linux its GTK or Qt and
    a display; NiceGUI would exit rather than fall back to the browser."""
    if importlib.util.find_spec("webview") is None:
        return False
    if sys.platform.startswith("linux"):
        toolkit = any(importlib.util.find_spec(m) for m in ("gi", "qtpy", "PyQt6", "PyQt5"))
        return toolkit and bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


#***********#
#   check   #
#***********#
def check() -> Dict[str, str]:
    """What this machine has for the GUI and MRSuite's toolboxes."""
    def module(name: str) -> str:
        spec = importlib.util.find_spec(name)
        if spec is None:
            return "not installed"
        from importlib.metadata import packages_distributions, version
        try:                                   # by its distribution, e.g. webview's pywebview
            return version(packages_distributions().get(name, [name])[0])
        except Exception:                              # noqa: BLE001 - a version is a nicety
            return "installed"

    from mrsuite.core.provenance import mrsuite_version
    found = {"Python": sys.version.split()[0], "MRSuite": mrsuite_version(),
             "NiceGUI": module("nicegui"), "pywebview": module("webview"),
             "window": "own" if native() else "browser"}
    for name, tool in (("Octave", "octave-cli"), ("R", "Rscript"), ("Docker", "docker"),
                       ("MATLAB", "matlab")):
        found[name] = shutil.which(tool) or "not found"
    for name, package in (("FSL-MRS", "fsl_mrs"), ("pyAMARES", "pyAMARES"), ("Vespa", "vespa"),
                          ("Augmentrum", "augmentrum"), ("MRSValidate", "mrsvalidate")):
        found[name] = module(package)
    return found


#**********#
#   main   #
#**********#
def main(argv: Optional[List[str]] = None) -> None:
    """Start the GUI."""
    parser = argparse.ArgumentParser(prog="mrsuite-gui", description="MRSuite's GUI.")
    parser.add_argument("--browser", action="store_true", help="open in the browser")
    parser.add_argument("--port", type=int, default=None, help="the port (default: a free one)")
    parser.add_argument("--check", action="store_true",
                        help="say what this machine has, and start nothing")
    parser.add_argument("--install", action="store_true",
                        help="make the GUI an app with MRSuite's icon, for this Python")
    parser.add_argument("--uninstall", action="store_true", help="take that app away")
    args = parser.parse_args(argv)
    if args.check:
        for name, value in check().items():
            print(f"{name:12} {value}")
        return
    if args.install or args.uninstall:
        from mrsuite.gui import launcher
        done = launcher.install() if args.install else launcher.uninstall()
        print("\n".join(f"{'Made' if args.install else 'Removed'} {path}" for path in done)
              or "Nothing to remove")
        return

    try:
        from nicegui import app, ui
    except ModuleNotFoundError:
        sys.exit("The GUI needs nicegui, which comes with MRSuite: pip install mrsuite")
    from mrsuite.gui.app import drop, page
    # what is loaded by now lives as long as the GUI: kept out of the collector's full sweeps,
    # which over the whole of it stall the page for a third of a second each
    gc.collect()
    gc.freeze()

    window = native() and not args.browser
    if window:
        app.native.on("drop", drop)
        app.native.settings["ALLOW_DOWNLOADS"] = True
        app.native.window_args["background_color"] = "#0e1217"   # no white flash at the start
    # a free port unless one is asked for: NiceGUI finds one only for a window of its own
    from nicegui.native.native_mode import find_open_port
    port = args.port or (None if window else find_open_port())
    # reload=False: a console script cannot be re-run by NiceGUI's reloader
    # dark from the first frame, as the page starts; it turns light if the system is
    # reconnect_timeout: the page and the GUI's process share one machine, so a process busy
    # for a few seconds is no lost connection (NiceGUI's 3 s, which also sets its heartbeat)
    ui.run(page, title="MRSuite", native=window, reload=False, show=True, dark=True,
           host=None if window else "127.0.0.1", port=port, reconnect_timeout=30,
           window_size=(1400, 900) if window else None)


if __name__ == "__main__":
    main()
