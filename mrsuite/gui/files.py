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

from mrsuite.gui.blocks import FORMATS


#**************************************************************************************************#
#                                           Class Picker                                           #
#**************************************************************************************************#
#                                                                                                  #
# A dialog that chooses files, or a folder, on this machine, and the format to read them as.       #
#                                                                                                  #
#**************************************************************************************************#
class Picker(ui.dialog):
    """A dialog that chooses files, or a folder, on this machine, and the format to read
    them as.

    Args:
        start (Path): The folder it opens in.
        on_choose (Callable): Called with the paths and the format, and awaited.
        title (str): What it is for.
        formats (bool): Whether to ask the format to read them as ('auto' if not).
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, start: Path, on_choose: Callable[[List[Path], str], Any],
                 title: str = "Open data", formats: bool = True) -> None:
        super().__init__()
        self.path, self.chosen, self.on_choose = Path(start), set(), on_choose
        with self, ui.card().classes("w-[44rem] max-w-full"):
            ui.label(title).classes("text-base font-bold")
            with ui.row().classes("w-full no-wrap items-center"):
                ui.button(icon="arrow_upward", on_click=lambda: self.show(self.path.parent)) \
                    .props("flat dense")
                self.where = ui.input("Folder", value=str(self.path)).classes("grow") \
                    .props("dense")
                self.where.on("keydown.enter", lambda: self.show(Path(self.where.value)))
            self.listing = ui.column().classes("w-full h-80 overflow-auto gap-0")
            with ui.row().classes("w-full items-center"):
                self.format = ui.select(FORMATS, value="auto", label="Format") \
                    .classes("w-32").set_visibility(formats)
                ui.space()
                ui.button("Cancel", on_click=self.close).props("flat")
                ui.button("This folder", on_click=lambda: self.choose([self.path])) \
                    .props("outline")
                ui.button("Open selected", on_click=lambda: self.choose(sorted(self.chosen))) \
                    .mark("picker-open")
        self.show(self.path)

    #**********#
    #   show   #
    #**********#
    def show(self, path: Path) -> None:
        """Show a folder: its folders first, to go into, then its files, to tick."""
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
                if entry.is_dir():
                    ui.button(entry.name, icon="folder",
                              on_click=lambda _, e=entry: self.show(e)) \
                        .props("flat dense no-caps align=left").classes("w-full")
                else:
                    ui.checkbox(entry.name, on_change=lambda ev, e=entry: (
                        self.chosen.add(e) if ev.value else self.chosen.discard(e))) \
                        .mark(f"file-{entry.name}")

    #************#
    #   choose   #
    #************#
    async def choose(self, paths: List[Path]) -> None:
        if not paths:
            ui.notify("tick a file, or open this folder", type="warning")
            return
        await self.on_choose(paths, self.format.value)      # before closing: closing says
        self.close()                                        # nothing was chosen
