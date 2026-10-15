####################################################################################################
#                                             files.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-07                                                                              #
#                                                                                                  #
# Purpose: Files and folders on this machine, chosen where they lie. Raw data comes as a folder    #
#          (DICOM) or as files that go together (Philips .SDAT and .SPAR), which a browser's       #
#          upload cannot hand over, so the GUI reads paths in place.                               #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from pathlib import Path
from typing import Any, Callable, List

from nicegui import ui


#**************************************************************************************************#
#                                           Class Picker                                           #
#**************************************************************************************************#
#                                                                                                  #
# A dialog that chooses files and folders on this machine, any number of each at once.             #
#                                                                                                  #
#**************************************************************************************************#
class Picker(ui.dialog):
    """A dialog that chooses files and folders on this machine, any number of each at once:
    the system's own dialogs choose one or the other. It also chooses one folder, or a
    file to save, named in a box, where the system's dialogs cannot be used (a browser).

    Args:
        start (Path): The folder it opens in.
        on_choose (Callable): Called with the paths, and awaited.
        title (str): What it is for.
        mode (str): 'open' for files and folders, 'folder' for one folder, 'save' for a
            file to write, named in a box.
        name (str): The file name the box starts with, for 'save'.
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, start: Path, on_choose: Callable[[List[Path]], Any],
                 title: str = "Open data", mode: str = "open", name: str = "") -> None:
        super().__init__()
        self.path, self.chosen, self.on_choose, self.mode = Path(start), set(), on_choose, mode
        with self, ui.card().classes("w-[44rem] max-w-full"):
            ui.label(title).classes("text-base font-bold")
            with ui.row().classes("w-full no-wrap items-center"):
                ui.button(icon="arrow_upward", on_click=lambda: self.show(self.path.parent)) \
                    .props("flat dense")
                self.where = ui.input("Folder", value=str(self.path)).classes("grow") \
                    .props("dense")
                self.where.on("keydown.enter", lambda: self.show(Path(self.where.value)))
            self.listing = ui.column().classes("w-full h-80 overflow-auto gap-0")
            if mode == "save":
                self.name = ui.input("File name", value=name).classes("w-full") \
                    .props("dense outlined").mark("picker-name")
                self.name.on("keydown.enter", lambda: self.choose([self.path / self.name.value]))
            with ui.row().classes("w-full items-center"):
                ui.label({"open": "Tick files and folders; open a folder by its name.",
                          "folder": "Open folders by their names, then take this one.",
                          "save": "Open folders by their names; a file clicked gives its name."}
                         [mode]).classes("text-xs mr-muted")
                ui.space()
                ui.button("Cancel", on_click=self.close).props("flat")
                if mode == "save":
                    ui.button("Save", icon="save",
                              on_click=lambda: self.choose([self.path / self.name.value])) \
                        .props("unelevated").mark("picker-save")
                else:
                    ui.button("This folder", on_click=lambda: self.choose([self.path])) \
                        .props("outline" if mode == "open" else "unelevated") \
                        .mark("picker-folder")
                if mode == "open":
                    ui.button("Open selected",
                              on_click=lambda: self.choose(sorted(self.chosen))) \
                        .mark("picker-open")
        self.show(self.path)

    #**********#
    #   show   #
    #**********#
    def show(self, path: Path) -> None:
        """Show a folder: its folders first, then its files, each to tick; a folder opened
        by its name."""
        if not path.is_dir():
            ui.notify(f"{path} is not a folder", type="warning")
            return
        self.path, self.where.value = path, str(path)
        self.chosen.clear()
        self.listing.clear()
        try:
            entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError as err:
            ui.notify(str(err), type="negative")
            return
        with self.listing:
            for entry in entries:
                if entry.name.startswith("."):
                    continue
                with ui.row().classes("w-full items-center no-wrap gap-0"):
                    if self.mode == "open":
                        ui.checkbox(on_change=lambda ev, e=entry: (
                            self.chosen.add(e) if ev.value else self.chosen.discard(e))) \
                            .props("dense").mark(f"file-{entry.name}")
                    if entry.is_dir():
                        ui.button(entry.name, icon="folder",
                                  on_click=lambda _, e=entry: self.show(e)) \
                            .props("flat dense no-caps align=left").classes("grow") \
                            .mark(f"open-{entry.name}")
                    elif self.mode == "save":
                        ui.button(entry.name, on_click=lambda _, e=entry:
                                  self.name.set_value(e.name)) \
                            .props("flat dense no-caps align=left").classes("grow") \
                            .mark(f"name-{entry.name}")
                    else:
                        ui.label(entry.name).classes("px-2")

    #************#
    #   choose   #
    #************#
    async def choose(self, paths: List[Path]) -> None:
        if not paths or (self.mode == "save" and not self.name.value.strip()):
            ui.notify("tick a file or a folder, or take this folder" if self.mode == "open"
                      else "give the file a name", type="warning")
            return
        await self.on_choose(paths)                         # before closing: closing says
        self.close()                                        # nothing was chosen
