####################################################################################################
#                                              app.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-07                                                                              #
#                                                                                                  #
# Purpose: The GUI's page: the blocks on the left, the canvas they are wired on in the middle, the #
#          selected block's settings on the right and below the canvas what a block made. The      #
#          graph's work lives in mrsuite.gui.graph; this only shows it and hands it the clicks.    #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import ast
import asyncio
import base64
import io
import json
import logging
import numbers
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from nicegui import app as nicegui, background_tasks, run, ui

logger = logging.getLogger(__name__)

from nifti_mrs_plus import NIfTI_MRS_Plus
from mrsuite import collect, externals
from mrsuite.core.basis import Basis
from mrsuite.gui import blocks, catalog
from mrsuite.gui.blocks import ANY, BLOCKS, GROUPS
from mrsuite.gui.canvas import Canvas, view
from mrsuite.gui.files import Picker
from mrsuite.gui.graph import (Graph, brief, drawn, export, exports, meets, results, table,
                               traces)

HERE = Path(__file__).parent
STATIC = "/mrsuite-static"
nicegui.add_static_files(STATIC, HERE / "static")

# the page last opened, which files dropped on the native window go to
LATEST: List["App"] = []


#**************************************************************************************************#
#                                            Class Tab                                             #
#**************************************************************************************************#
#                                                                                                  #
# One graph open on the page: its file, how it was last saved, and its page state while hidden.    #
#                                                                                                  #
#**************************************************************************************************#
@dataclass
class Tab:
    """One graph open on the page: its file, how it was last saved, and its page state while
    hidden.

    Attributes:
        graph (Graph): The graph.
        number (int): Its number among the untitled ones, for its name until saved.
        path (Optional[Path]): The file it was opened from or saved to.
        saved (str): The graph as it was then (``_snapshot``), so an edit since shows.
        state (Dict[str, Any]): The selection, preview and thumbnails while another is shown.
    """

    graph: Graph
    number: int = 0
    path: Optional[Path] = None
    saved: str = ""
    state: Dict[str, Any] = field(default_factory=dict)

    #**********#
    #   name   #
    #**********#
    @property
    def name(self) -> str:
        return self.path.name.split(".")[0] if self.path else f"Untitled {self.number}"

    #***********#
    #   dirty   #
    #***********#
    @property
    def dirty(self) -> bool:
        return _snapshot(self.graph) != self.saved


#**************#
#   snapshot   #
#**************#
def _snapshot(graph: Graph) -> str:
    """A graph as text, to tell an edit since it was saved."""
    return json.dumps(graph.to_dict(), sort_keys=True)


#**************************************************************************************************#
#                                            Class App                                             #
#**************************************************************************************************#
#                                                                                                  #
# One page of the GUI over one graph.                                                              #
#                                                                                                  #
#**************************************************************************************************#
class App:
    """One page of the GUI over one graph.

    Args:
        graph (Optional[Graph]): The graph to show; a new one by default.
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self, graph: Optional[Graph] = None) -> None:
        self.graph = graph or Graph()
        self.selected: Optional[str] = None          # the block in the inspector
        self.previewed: Optional[str] = None         # the block in the preview
        self.port: Optional[str] = None              # which of its outputs
        self.index, self.scaling = 0, None           # which spectrum, which concentrations
        self.spread, self.stack = None, 0.0           # which dimension drawn apart, how far
        self.folded = 70                              # the canvas' share while the preview shows
        self.ppm_range = None                         # the viewer's ppm window; None shows all
        self.running: Optional[str] = None           # the block under way
        self.runs: set = set()                       # the graphs runs are under way on
        self.theme = "system"
        self.folder = Path.home()                    # where files are looked for and written
        self.thumbs: Dict[str, Tuple[int, Any, str]] = {}
        self.functions: Dict[str, Any] = {}          # every toolbox function by its tab, once listed
        self.zoo: Dict[str, List[str]] = {}          # the Hub's checkpoints by block, once listed
        self.count = 1                               # untitled tabs so far
        self.tabs: List[Tab] = [Tab(self.graph, 1, saved=_snapshot(self.graph))]
        self.tab = self.tabs[0]                      # the one shown
        self.opened: set = set()                     # the palette's folds left open
        self.about: Dict[Tuple[str, str], Any] = {}  # a function's form and help, once read
        self.client = ui.context.client
        LATEST[:] = [self]

        for css in (HERE / "vendor" / "vue-flow-core" / "style.css",
                    HERE / "vendor" / "vue-flow-core" / "theme-default.css",
                    HERE / "static" / "mrsuite.css"):
            ui.add_css(css)
        ui.colors(primary="#5274f0", secondary="#16bdc6", accent="#8ea4ff", positive="#16bdc6",
                  negative="#ef5d6c", dark="#141920", dark_page="#0e1217")
        self.dark = ui.dark_mode(True)               # dark until the system says otherwise

        with ui.element("div").classes("mr-splash"):
            ui.element("img").props(f"src={STATIC}/logo.png alt=MRSuite")
            ui.label("MRSuite")
        # where windows opened from the inspector live, so its redraw does not delete them
        self.windows = ui.element("div")
        self._header()
        # the blocks, the canvas over the preview, and the inspector, each edge dragged to size
        with ui.splitter(value=264, limits=(180, 420)).props("unit=px") \
                .classes("w-full").style("height: calc(100vh - 52px)") as left:
            with left.before:
                with ui.element("div").classes("mr-drawer w-full h-full"):
                    self._blocks()
            with left.after:
                with ui.splitter(value=320, limits=(220, 480)).props("unit=px reverse") \
                        .classes("w-full h-full") as right:
                    with right.before:
                        self.split = ui.splitter(horizontal=True, value=self.folded,
                                                 limits=(25, 100)).classes("w-full h-full")
                        with self.split as split:
                            with split.before:
                                with ui.column().classes("w-full h-full gap-0 no-wrap relative"):
                                    self.tabbar()
                                    self.canvas = Canvas().classes("w-full grow")
                                    # brings the preview back once folded away
                                    self.peek = ui.button(icon="expand_less", color=None,
                                                          on_click=self.unfold_preview) \
                                        .props("flat round dense").classes("mr-peek") \
                                        .tooltip("Show the preview").mark("show-preview")
                                    self.peek.set_visibility(False)
                            with split.after:
                                with ui.column().classes("mr-preview w-full h-full p-3 gap-2 "
                                                         "no-wrap overflow-auto"):
                                    self.preview()
                    with right.after:
                        with ui.scroll_area().classes("mr-drawer w-full h-full"):
                            with ui.column().classes("w-full p-4 gap-3"):
                                self.inspector()
        self.canvas.on("select", lambda e: self.select(e.args["id"]))
        self.canvas.on("act", lambda e: self.act(e.args["id"], e.args["what"]))
        self.canvas.on("connect", lambda e: self.connect(e.args["source"], e.args["sourceHandle"],
                                                        e.args["target"], e.args["targetHandle"]))
        self.canvas.on("loose", lambda e: self.loose(e.args))
        self.canvas.on("removed", lambda e: self.removed(e.args["kind"], e.args["ids"]))
        self.canvas.on("moved", lambda e: ([self.graph.move(m["id"], (m["x"], m["y"]))
                                            for m in e.args if m["id"] in self.graph.nodes],
                                           self.tabbar.refresh()))     # moved: edited
        self.canvas.on("search", lambda e: self.search((e.args["x"], e.args["y"])))
        self.canvas.on("dropped", lambda e: self.add(e.args["key"], (e.args["x"], e.args["y"]),
                                                     join=False))
        # anything a handler did not see coming is said on the page, not only in the console
        ui.on_exception(lambda err: _said(err, logged=True))      # NiceGUI's own log has it
        # leaving the page with a tab edited since its save is asked about (a browser's way)
        ui.add_body_html('''<script>window.addEventListener("beforeunload", e => {
            if (document.querySelector(".mr-tab[data-dirty='1']")) { e.preventDefault();
                e.returnValue = ""; } });</script>''')
        self.draw()
        ui.timer(0.3, self._tick)
        ui.timer(0, self._system, once=True)

    #************#
    #   header   #
    #************#
    def _header(self) -> None:
        def button(text: str, icon: str, on_click: Callable, tip: str) -> ui.button:
            return ui.button(text, icon=icon, on_click=on_click, color=None) \
                .props("flat dense no-caps") \
                .classes("px-2").tooltip(tip)

        with ui.header().classes("mr-header items-center gap-1 px-3 py-1"):
            ui.element("img").props(f"src={STATIC}/logo.png alt=").style("height: 30px")
            ui.label("MRSuite").classes("mr-brand")
            ui.label(_version()).classes("mr-version")
            ui.element("div").classes("mr-divider")
            button("New", "add_box", self.new, "An empty graph in a new tab").mark("new")
            button("Open", "folder_open", self.open, "Open a saved graph in a new tab")
            button("Save", "save", self.save, "Save the graph to its file").mark("save")
            button("Save as", "save_as", self.save_as, "Save the graph to a new file")
            ui.element("div").classes("mr-divider")
            ui.button("Run", icon="play_arrow", on_click=lambda: self.run()) \
                .props("unelevated dense no-caps").classes("mr-run px-3") \
                .tooltip("Run every block that has not run").mark("run")
            button("Stop", "stop", self.graph_stop, "Stop after the block under way").mark("stop")
            ui.element("div").classes("mr-divider")
            button("Script", "code", self.script, "The graph as Python").mark("script")
            button("", "fit_screen", lambda: self.canvas.fit(), "Show every block")
            ui.space()
            self.spinner = ui.spinner(size="1.1em", color="primary").mark("spinner")
            self.spinner.set_visibility(False)
            self.status = ui.label("").classes("mr-muted text-sm")
            with ui.button(icon="settings", color=None).props("flat round dense") \
                    .tooltip("Settings"):
                with ui.menu().classes("p-3"):
                    ui.label("Theme").classes("mr-section")
                    ui.radio({"system": "As the system", "dark": "Dark", "light": "Light"},
                             value=self.theme, on_change=lambda e: self._system(e.value))

    #****************#
    #   graph stop   #
    #****************#
    def graph_stop(self) -> None:
        self.graph.stop()

    #************#
    #   system   #
    #************#
    async def _system(self, theme: Optional[str] = None) -> None:
        """Dark or light as chosen; as the system has it, dark where it says nothing."""
        self.theme = theme or self.theme
        if self.theme != "system":
            self.dark.set_value(self.theme == "dark")
        else:
            try:
                await self.client.connected(timeout=10)
                light = await ui.run_javascript(
                    "window.matchMedia('(prefers-color-scheme: light)').matches")
            except Exception:                          # noqa: BLE001 - not told: dark
                light = False
            self.dark.set_value(not light)
        self.preview.refresh()                         # the plot's colours

    #************#
    #   blocks   #
    #************#
    def _blocks(self) -> None:
        with ui.column().classes("w-full h-full p-3 gap-2 no-wrap"):
            self.find = ui.input(placeholder="Search blocks…").props("dense outlined clearable") \
                .classes("w-full").mark("find")
            self.find.on_value_change(lambda: self._listing())
            with ui.scroll_area().classes("w-full grow"):
                self.palette()
            ui.label("Drag a block onto the canvas, or click it to add it after the selected "
                     "one. Double-click the canvas to search.").classes("mr-muted text-xs")

    #*************#
    #   palette   #
    #*************#
    @ui.refreshable
    def palette(self) -> None:
        text = (self.find.value or "").strip().lower()
        for group in GROUPS.values():
            if group.key == "toolbox":
                continue
            shown = [b for b in BLOCKS.values() if b.group == group.key and _matches(b, text)]
            if not shown:
                continue
            with ui.expansion(value=True).classes("mr-group w-full").props("dense") as part:
                with part.add_slot("header"):
                    with ui.row().classes("items-center gap-2 w-full no-wrap"):
                        ui.element("span").classes("mr-swatch") \
                            .style(f"background: {group.colour}")
                        ui.label(group.title).classes("text-sm font-semibold")
                        ui.icon("info_outline").classes("mr-muted ml-auto text-sm") \
                            .tooltip(group.help)
                for b in shown:
                    if not b.folder:
                        self._item(b.key, b.title, group.colour, b.help)
                for folder in dict.fromkeys(b.folder for b in shown if b.folder):
                    key = f"{group.key}/{folder}"
                    with ui.expansion(folder, value=bool(text) or key in self.opened,
                                      on_value_change=lambda e, k=key: self._fold(k, e.value)) \
                            .classes("w-full mr-folder").props("dense"):
                        for b in shown:
                            if b.folder == folder:
                                self._item(b.key, b.title, group.colour, b.help)
        toolbox = GROUPS["toolbox"]
        with ui.expansion(value=bool(text) or "all" in self.opened,
                          on_value_change=lambda e: self._fold("all", e.value)) \
                .classes("mr-group w-full").props("dense") as part:
            with part.add_slot("header"):
                with ui.row().classes("items-center gap-2 w-full no-wrap"):
                    ui.element("span").classes("mr-swatch").style(f"background: {toolbox.colour}")
                    ui.label(toolbox.title).classes("text-sm font-semibold")
            for bridge in catalog.BRIDGES:
                names = self.functions.get(bridge)
                if isinstance(names, dict):
                    names = {n: g for n, g in names.items()
                             if not text or text in n.lower() or text in bridge}
                    if text and not names:
                        continue
                with ui.expansion(blocks.label((bridge, "")).rstrip(" ·"),
                                  value=bool(text) or bridge in self.opened,
                                  on_value_change=lambda e, k=bridge: self._fold(k, e.value)) \
                        .classes("w-full").props("dense"):
                    if names is None:
                        ui.label("listing…").classes("mr-muted text-xs px-2")
                    elif isinstance(names, str):
                        ui.label(names).classes("mr-muted text-xs px-2")
                    else:
                        self._folders(bridge, names, text)

    #*************#
    #   folders   #
    #*************#
    def _folders(self, bridge: str, names: Dict[str, str], text: str) -> None:
        """A toolbox's functions by the tab each belongs to, in the tabs' order."""
        colour = GROUPS["toolbox"].colour
        for group in GROUPS.values():
            found = [n for n, g in names.items() if g == group.key]
            if not found:
                continue
            key = f"{bridge}/{group.key}"
            with ui.expansion(group.title, value=bool(text) or key in self.opened,
                              on_value_change=lambda e, k=key: self._fold(k, e.value)) \
                    .classes("w-full mr-folder").props("dense").mark(f"folder-{key}"):
                for name in found[:200]:
                    self._item(f"{bridge}:{name}", name, colour, "")

    #**********#
    #   item   #
    #**********#
    def _item(self, key: str, title: str, colour: str, tip: str) -> None:
        """A block to drag onto the canvas, or click to add."""
        item = ui.element("div").classes("mr-item").props("draggable").mark(f"block-{key}")
        with item:
            ui.element("span").classes("mr-swatch").style(f"background: {colour}")
            ui.label(title)
        item.on("dragstart", js_handler="(e) => e.dataTransfer.setData('text/mrsuite-block', "
                                        f"{json.dumps(key)})")
        item.on("click", lambda k=key: self.add(k))
        if tip:
            item.tooltip(tip)

    #**********#
    #   fold   #
    #**********#
    def _fold(self, part: str, opened: bool) -> None:
        """A fold of the palette opened or closed, kept so a refresh leaves it so: a group's
        folder ('group/folder'), or a toolbox, whose functions are listed on opening."""
        if opened:
            self.opened.add(part)
            if "/" not in part:
                self._listing()
        else:
            self.opened.discard(part)

    #*************#
    #   listing   #
    #*************#
    def _listing(self) -> None:
        """Every toolbox's functions, listed in the background the first time they are
        searched or opened: R and MRSpecLAB take seconds."""
        self.palette.refresh()
        for bridge in catalog.BRIDGES:
            if bridge not in self.functions:
                self.functions[bridge] = None
                background_tasks.create(self._list(bridge))
        if not self.zoo:
            self.zoo = {k: [] for k, b in BLOCKS.items() if b.model}
            background_tasks.create(self._zoo())

    #*********#
    #   zoo   #
    #*********#
    async def _zoo(self) -> None:
        """The checkpoints MRSuite keeps on the Hub, by the block that loads them; the pinned
        ones stay where the Hub is out of reach."""
        from mrsuite import models
        try:
            listed = await run.io_bound(models.zoo)
        except Exception as err:                       # noqa: BLE001 - said, not hidden
            logger.info("the model zoo could not be listed: %s", brief(err))
            return
        self.zoo = {k: sorted(r for r, spec in listed.items() if spec == b.model)
                    for k, b in BLOCKS.items() if b.model}
        self.inspector.refresh()

    #**********#
    #   list   #
    #**********#
    async def _list(self, bridge: str) -> None:
        try:
            self.functions[bridge] = await run.io_bound(blocks.categories, bridge)
        except Exception as err:                       # noqa: BLE001 - said, not hidden
            self.functions[bridge] = brief(err)
        self.palette.refresh()

    #*********#
    #   add   #
    #*********#
    def add(self, key: str, position: Optional[Tuple[float, float]] = None,
            join: bool = True) -> str:
        """
        A block, where it was dropped, else after the selected one and wired to it.

        Returns:
            str: Its id.
        """
        before = self.selected if join and self.selected in self.graph.nodes and \
            not _chooses(blocks.block(key)) else None          # its files are chosen
        if position is None:
            if before:
                x, y = self.graph.nodes[before].position
                position = (x + 280, y)
            else:
                xs = [n.position[0] for n in self.graph.nodes.values()]
                position = (max(xs) + 280 if xs else 80, 80)
        id = self.graph.add(key, position)
        background_tasks.create(self._asked(id))    # its arguments' inputs, read in a thread
        if before:
            self._join(before, id)
        self.select(id)
        if _chooses(blocks.block(key)):
            ui.timer(0, lambda: self.collect(id) if key == "load" else self.choose_files(id),
                     once=True)
        return id

    #***********#
    #   asked   #
    #***********#
    async def _asked(self, id: str) -> None:
        """A new block's arguments read in a thread, then drawn as its inputs - unless the
        block, or its whole graph, went meanwhile."""
        graph = self.graph
        if id in graph.nodes:
            await run.io_bound(graph.ask, id)
            self.draw()

    #**********#
    #   join   #
    #**********#
    def _join(self, source: str, target: str) -> bool:
        """Wire the first output of one block that fits the first input of another."""
        for out in self.graph.ports(source)[1]:
            for into in self.graph.ports(target)[0]:
                if meets(out.kind, into.kind):
                    self.graph.connect(source, out.name, target, into.name)
                    return True
        return False

    #************#
    #   search   #
    #************#
    async def search(self, position: Tuple[float, float],
                     fits: Optional[Callable[[blocks.Block], bool]] = None) -> Optional[str]:
        """A block chosen by name, added where asked; its id."""
        if not self.functions:
            self._listing()
        with ui.dialog() as dialog, ui.card().classes("w-[26rem] p-2 gap-1"):
            box = ui.input(placeholder="Add a block…").props("autofocus dense outlined") \
                .classes("w-full")
            listing = ui.column().classes("w-full gap-0 max-h-80 overflow-auto no-wrap")

            def fill() -> None:
                listing.clear()
                with listing:
                    for key, title, colour, where in self._candidates(box.value, fits)[:60]:
                        with ui.element("div").classes("mr-item") \
                                .on("click", lambda k=key: dialog.submit(k)):
                            ui.element("span").classes("mr-swatch").style(f"background: {colour}")
                            ui.label(title)
                            ui.label(where).classes("mr-muted text-xs ml-auto")

            def first() -> None:
                found = self._candidates(box.value, fits)
                if (box.value or "").strip() and found:
                    dialog.submit(found[0][0])

            box.on_value_change(fill)
            box.on("keydown.enter", first)
            fill()
        key = await dialog
        dialog.delete()
        return self.add(key, position, join=False) if key else None

    #****************#
    #   candidates   #
    #****************#
    def _candidates(self, text: Optional[str], fits: Optional[Callable[[blocks.Block], bool]]) \
            -> List[Tuple[str, str, str, str]]:
        """Blocks whose name has the text, and that fit, as (key, title, colour, group)."""
        text = (text or "").strip().lower()
        found = [(b.key, b.title, GROUPS[b.group].colour, GROUPS[b.group].title)
                 for b in BLOCKS.values() if _matches(b, text) and (fits is None or fits(b))]
        # names that start so first, then names that have it, then the help that does
        found.sort(key=lambda f: (not f[1].lower().startswith(text), text not in f[1].lower()))
        for bridge, names in self.functions.items():
            if not isinstance(names, dict) or not text:
                continue
            for name in names:
                key = f"{bridge}:{name}"
                if (text in name.lower() or text in bridge) and \
                        (fits is None or fits(blocks.block(key))):
                    found.append((key, name, GROUPS["toolbox"].colour,
                                  blocks.label((bridge, "")).rstrip(" ·")))
        return found

    #************#
    #   canvas   #
    #************#
    def draw(self) -> None:
        nodes, edges = view(self.graph, self.running, self.thumbs)
        for n in nodes:
            n["selected"] = n["id"] == self.selected
        self.canvas.show(nodes, edges)
        self.tabbar.refresh()                          # an edit shows on the tab

    #************#
    #   select   #
    #************#
    def select(self, id: Optional[str]) -> None:
        """A block into the inspector, and what it made into the preview."""
        self.selected = id
        if id is not None and self.graph.nodes[id].made is not None and id != self.previewed:
            self.previewed, self.port, self.index = id, None, 0
            self.preview.refresh()
        self.inspector.refresh()
        self.draw()

    #*************#
    #   connect   #
    #*************#
    def connect(self, source: str, output: str, target: str, input: str) -> None:
        try:
            self.graph.connect(source, output, target, input)
        except ValueError as err:
            ui.notify(str(err), type="warning")
        self._changed()

    #***********#
    #   loose   #
    #***********#
    async def loose(self, wire: Dict[str, Any]) -> None:
        """A wire let go on the canvas: a block that takes it, or gives it, added there."""
        node, port, side = wire["nodeId"], wire["handleId"], wire["handleType"]
        ports = self.graph.ports(node)[1 if side == "source" else 0]
        kind = next((p.kind for p in ports if p.name == port), ANY)

        def fits(b: blocks.Block) -> bool:
            ins, outs = blocks.ports(b, [])
            return any(meets(kind, p.kind) if side == "source" else meets(p.kind, kind)
                       for p in (ins if side == "source" else outs))

        id = await self.search((wire["x"], wire["y"]), fits)
        if id is None or node not in self.graph.nodes:
            return
        for p in self.graph.ports(id)[0 if side == "source" else 1]:
            if meets(kind, p.kind) if side == "source" else meets(p.kind, kind):
                if side == "source":
                    self.connect(node, port, id, p.name)
                else:
                    self.connect(id, p.name, node, port)
                break

    #*************#
    #   removed   #
    #*************#
    def removed(self, kind: str, ids: List[str]) -> None:
        """Blocks or wires deleted on the canvas."""
        for k in ids:
            if kind == "node" and k in self.graph.nodes:
                self.graph.remove(k)
            for e in [e for e in self.graph.edges if e.id == k]:
                self.graph.disconnect(e)
        self.selected = self.selected if self.selected in self.graph.nodes else None
        self.previewed = self.previewed if self.previewed in self.graph.nodes else None
        self._changed()

    #*********#
    #   act   #
    #*********#
    async def act(self, id: str, what: str) -> None:
        """A block's own buttons."""
        if what == "remove":
            self.removed("node", [id])
        elif what == "run":
            await self.run(id)
        elif what == "preview":
            if id == self.previewed:                   # the eye again: closed
                self.previewed = None
                self.preview.refresh()
                return
            self.previewed, self.port, self.index = id, None, 0
            if self.split.value >= 100:
                self.unfold_preview()
            if self.graph.nodes[id].made is None:
                await self.run(id)
            self.select(id)
            self.preview.refresh()
        elif what == "export":
            await self.export(id)
        elif what == "header":
            await self.header(id)

    #********************#
    #   fold the preview   #
    #********************#
    def fold_preview(self) -> None:
        """The preview panel away: the canvas takes its height; a chevron at the bottom,
        or the eye on a block, brings it back where it was."""
        if self.split.value < 100:
            self.folded = self.split.value
        self.split.value = 100
        self.peek.set_visibility(True)

    #**********************#
    #   unfold the preview   #
    #**********************#
    def unfold_preview(self) -> None:
        self.split.value = self.folded
        self.peek.set_visibility(False)

    #*************#
    #   changed   #
    #*************#
    def _changed(self) -> None:
        self.draw()
        self.inspector.refresh()
        self.preview.refresh()

    #*********#
    #   run   #
    #*********#
    async def run(self, upto: Optional[str] = None) -> None:
        """Run what has not run, all or as far as one block, in a thread, so the page
        stays live."""
        if self.busy:
            ui.notify("Already running; stop it first", type="warning")
            return
        if not self.graph.nodes:
            ui.notify("Add blocks first: drag one from the left", type="info")
            return
        graph = self.graph
        self.runs.add(graph)
        try:
            done = await run.io_bound(graph.run, upto, lambda k: self._started(graph, k))
            if not done and graph is self.graph:
                ui.notify("Stopped before the end: press Run to go on", type="info")
        except Exception as err:                       # noqa: BLE001 - said, not hidden
            if graph is self.graph:
                _said(err, logged=True)                # the graph logged it
        finally:
            self.runs.discard(graph)
            if graph is self.graph:
                self.running = None
                self.status.text = ""
            self.spinner.set_visibility(bool(self.runs))
        self._changed()

    #**********#
    #   busy   #
    #**********#
    @property
    def busy(self) -> bool:
        """Whether the graph shown is running; another tab's may be too."""
        return self.graph in self.runs

    #*************#
    #   started   #
    #*************#
    def _started(self, graph: Graph, id: str) -> None:
        """From the running thread: the block under way, drawn by the timer."""
        if graph is self.graph:
            self.running = id

    #**********#
    #   idle   #
    #**********#
    async def _idle(self) -> None:
        """The run under way stopped, after the block it is in."""
        self.graph.stop()
        while self.busy:
            await asyncio.sleep(0.1)

    #**********#
    #   tick   #
    #**********#
    def _tick(self) -> None:
        self.spinner.set_visibility(bool(self.runs))   # any tab's run
        if self.busy:
            self.draw()
            node = self.graph.nodes.get(self.running or "")
            self.status.text = externals.now() or (f"Running {node.name}…" if node else
                                                   "Running…")

    #***************#
    #   inspector   #
    #***************#
    @ui.refreshable
    def inspector(self) -> None:
        node = self.graph.nodes.get(self.selected or "")
        if node is None:
            ui.label("Blocks").classes("mr-section")
            ui.label("Drag blocks from the left onto the canvas and wire an output (right) to "
                     "an input (left). Select a block to set it up here; ▶ runs up to it, the "
                     "eye shows what it made, and Script writes the whole graph as Python.") \
                .classes("text-sm mr-muted")
            return
        b = blocks.block(node.block)
        group = GROUPS[b.group]
        with ui.row().classes("items-center gap-2 no-wrap w-full"):
            ui.element("span").classes("mr-swatch").style(f"background: {group.colour}")
            ui.label(b.title).classes("text-base font-semibold")
        box = ui.input("name", value=node.name).props("dense outlined") \
            .classes("w-full font-mono").mark("name")
        box.on("blur", lambda _: self._rename(node.id, box.value))
        box.on("keydown.enter", lambda _: self._rename(node.id, box.value))
        if b.help:
            ui.label(b.help).classes("text-xs mr-muted")
        if len(b.offers) > 1:
            ui.select({i: blocks.label(o) for i, o in enumerate(b.offers)}, value=node.option,
                      label="Toolbox" if b.step else "Analysis",
                      on_change=lambda e: self._choose(node.id, e.value)) \
                .classes("w-full").props("dense outlined").mark("toolbox")
        if node.block == "load":
            entries = node.values.get("entries") or []
            counts: Dict[str, int] = {}
            for _, name, _, _ in entries:
                counts[name] = counts.get(name, 0) + 1
            for name in collect.order(counts):
                ui.label(f"{name}: {counts[name]} file{'s' * (counts[name] != 1)}") \
                    .classes("text-xs font-mono")
            ui.button("Add or name files…", icon="folder_open",
                      on_click=lambda: self.collect(node.id)) \
                .props("outline dense no-caps").classes("w-full").mark("collect") \
                .tooltip("Files and folders to load, each named, e.g. data, water, mm, truth")
        elif _chooses(b) and not self._wired(node.id, "files"):
            for f in node.values.get("files", []):
                ui.label(str(f)).classes("text-xs font-mono break-all")
            ui.button("Choose files…", icon="folder_open",
                      on_click=lambda: self.choose_files(node.id)) \
                .props("outline dense no-caps").classes("w-full")
        if node.block == "basis-add":
            ui.button("Choose functions…", icon="checklist",
                      on_click=lambda: self.functions_kept(node.id)) \
                .props("outline dense no-caps").classes("w-full").mark("keep") \
                .tooltip("Which functions the merged basis keeps; one per name")
        if node.error:
            ui.label(node.error).classes("text-xs text-negative break-words")
        ui.label("Settings").classes("mr-section pt-2")
        self._form(node)

    #**********#
    #   form   #
    #**********#
    def _form(self, node: Any) -> None:
        b = blocks.block(node.block)
        if b.step:
            option = b.options[node.option]
            known = self.about.get(option)
            if known is None:
                self.about[option] = "reading"
                background_tasks.create(self._read_about(option))
            if not isinstance(known, tuple):
                ui.label("Reading its arguments…" if known in (None, "reading") else known) \
                    .classes("text-xs mr-muted")
                return
            fields, signature, help_ = known
            source = blocks.PIPELINE.get(option, ({}, ""))[1]
            if source:
                ui.label(f"Set as {source} runs it.").classes("text-xs mr-muted")
        elif b.analyses:
            option = b.analyses[node.option]
            try:
                fields = self.graph.fields(node.id)
                found, help_ = catalog.about(*option)
            except Exception as err:                   # noqa: BLE001 - said, not hidden
                ui.label(brief(err)).classes("text-xs mr-muted")
                return
            signature = f"{option[1]}{found}"
        else:
            fields, signature, help_ = list(b.fields), "", ""
        paths = catalog.paths(help_, [f.name for f in fields])
        for f in fields:
            if (_chooses(b) and f.name == "files" and not self._wired(node.id, "files")) or \
                    (node.block == "basis-add" and f.name == "keep"):
                continue
            self._field(node, f, f.name in paths)
        if b.analyses:
            with ui.expansion("Help", icon="help_outline").classes("w-full").props("dense"):
                ui.label(signature).classes("font-mono text-xs break-all")
                ui.code(help_ or "No help.", language="text").classes("w-full text-xs")
        if b.step:
            more = node.values.get(catalog.MORE)
            box = ui.input("further arguments", placeholder="{'name': value}",
                           value=repr(more) if more else "").classes("w-full") \
                .props("dense outlined")
            box.on("blur", lambda _: self._typed(node.id, catalog.MORE, box.value))
            box.on("keydown.enter", lambda _: self._typed(node.id, catalog.MORE, box.value))
            with ui.expansion("Help", icon="help_outline").classes("w-full").props("dense"):
                if signature:
                    ui.label(signature).classes("font-mono text-xs break-all")
                ui.code(help_ or "No help.", language="text").classes("w-full text-xs")

    #***********#
    #   field   #
    #***********#
    def _field(self, node: Any, f: catalog.Field, path: bool = False) -> None:
        """One argument: set here, or, wired in, taken from the wire; a file's with a button
        to choose it, and one with a default with a button to put it on the block."""
        wired = self._wired(node.id, f.name)
        with ui.row().classes("w-full no-wrap items-center gap-1"):
            if wired:
                ui.label(f"{f.name} ← by wire").classes("grow text-sm mr-muted")
            elif f.kind == "choice":
                options = {None: f"default {f.default}".strip(), **{c: repr(c) for c in f.choices}}
                ui.select(options, label=f.name, value=node.values.get(f.name),
                          on_change=lambda e, n=f.name: self._set(node.id, n, e.value)) \
                    .classes("grow").props("dense outlined")
            elif f.kind == "pick":                     # the Hub's, else the pinned; or typed
                names = self.zoo.get(node.block) or f.choices
                ui.select(sorted({*names, *filter(None, [node.values.get(f.name)])}),
                          label=f.name, value=node.values.get(f.name), with_input=True,
                          new_value_mode="add-unique",
                          on_change=lambda e, n=f.name: self._set(node.id, n, e.value)) \
                    .classes("grow").props("dense outlined").mark(f"field-{f.name}")
            else:
                shown = repr(node.values[f.name]) if f.name in node.values else ""
                hint = f.default or ("in order" if f.positional else "needed")
                box = ui.input(f.name, value=shown, placeholder=hint).classes("grow") \
                    .props("dense outlined").mark(f"field-{f.name}")
                box.on("blur", lambda _, b=box, n=f.name: self._typed(node.id, n, b.value))
                box.on("keydown.enter", lambda _, b=box, n=f.name: self._typed(node.id, n, b.value))
                if path:
                    ui.button(icon="folder_open",
                              on_click=lambda _, n=f.name: self._choose_path(node.id, n)) \
                        .props("flat dense round size=sm").tooltip("Choose a file") \
                        .mark(f"path-{f.name}")
            if f.default:
                on = f.name in node.shown
                ui.button(icon="input", on_click=lambda _, n=f.name, o=on:
                          self._put(node.id, n, not o)) \
                    .props(f"flat dense round size=sm {'color=primary' if on else 'color=grey'}") \
                    .tooltip("Take it off the block" if on else "Put it on the block, to wire in") \
                    .mark(f"show-{f.name}")

    #***********#
    #   wired   #
    #***********#
    def _wired(self, id: str, name: str) -> bool:
        return any((e.target, e.input) == (id, name) for e in self.graph.edges)

    #****************#
    #   read about   #
    #****************#
    async def _read_about(self, option: Tuple[str, str]) -> None:
        """A function's arguments and help, read in a thread: R and MRSpecLAB take seconds."""
        try:
            signature, help_ = await run.io_bound(catalog.about, *option)
            self.about[option] = (catalog.fields(signature),
                                  f"{option[1]}{signature}" if signature else "", help_)
        except Exception as err:                       # noqa: BLE001 - said, not hidden
            self.about[option] = brief(err)
        self.inspector.refresh()
        self.draw()                                  # its arguments' inputs too

    #***********#
    #   typed   #
    #***********#
    def _typed(self, id: str, name: str, text: str) -> None:
        value = catalog.parse(text) if text.strip() else None
        if name == catalog.MORE and value is not None and not isinstance(value, dict):
            ui.notify("further arguments are a dictionary: {'name': value}", type="warning")
            return
        if value != self.graph.nodes[id].values.get(name):
            self._set(id, name, value)

    #*********#
    #   set   #
    #*********#
    def _set(self, id: str, name: str, value: Any) -> None:
        try:
            self.graph.set(id, name, value)
        except ValueError as err:
            ui.notify(str(err), type="warning")
        self._changed()

    #************#
    #   rename   #
    #************#
    def _rename(self, id: str, name: str) -> None:
        name = (name or "").strip()
        if name == self.graph.nodes[id].name:          # left as it was: nothing to redraw
            return
        try:
            self.graph.rename(id, name)
        except ValueError as err:
            ui.notify(str(err), type="warning")
            return
        self._changed()

    #*********#
    #   put   #
    #*********#
    def _put(self, id: str, name: str, on: bool) -> None:
        self.graph.show(id, name, on)
        self._changed()

    #************#
    #   choose   #
    #************#
    async def _choose(self, id: str, option: int) -> None:
        b = blocks.block(self.graph.nodes[id].block)
        await run.io_bound(catalog.read, *b.offers[option])      # a toolbox can take seconds
        if id in self.graph.nodes:
            self.graph.choose(id, option)
            self._changed()

    #*****************#
    #   choose path   #
    #*****************#
    async def _choose_path(self, id: str, name: str) -> None:
        """A file for an argument, from the same dialog files are read with."""
        chosen = await self._pick("Choose a file")
        if chosen and id in self.graph.nodes:
            self.folder = _folder(chosen[0])
            self._set(id, name, str(chosen[0]))

    #***********#
    #   files   #
    #***********#
    async def choose_files(self, id: str) -> None:
        """Files for a block that takes them, e.g. a basis set's, from the system's own
        dialog where there is one."""
        chosen = await self._pick("Choose files")
        if not chosen or id not in self.graph.nodes:
            return
        self.folder = _folder(chosen[0])
        self.graph.set(id, "files", [str(p) for p in chosen])
        self._changed()

    #*************#
    #   collect   #
    #*************#
    async def collect(self, id: str) -> None:
        """The window over what a load loads: files and folders added from anywhere, a folder
        searched; rows selected get a name, a group or a format at once, or go; kept in
        order by group and name, then loaded."""
        rows: List[Dict[str, str]] = []

        def add(found: List[collect.Entry]) -> None:
            known = {r["path"] for r in rows}
            rows.extend({"path": str(e.path), "file": e.path.name, "name": e.name,
                         "group": e.group, "format": e.format}
                        for e in found if str(e.path) not in known)
            show()

        def show() -> None:
            order = collect.sort([collect.Entry(Path(r["path"]), r["name"], r["group"],
                                                r["format"]) for r in rows])
            place = {str(e.path): k for k, e in enumerate(order)}
            rows.sort(key=lambda r: place[r["path"]])
            table.rows = list(rows)                    # the table keeps a list of its own
            name.set_autocomplete(collect.order([*collect.NAMES, *(r["name"] for r in rows)]))

        async def add_from() -> None:
            chosen = await self._pick("Add files and folders", both=True)
            if chosen:
                self.folder = _folder(chosen[0])
                add(_found(chosen))

        def chosen_rows() -> List[Dict[str, str]]:
            picked = {r["path"] for r in table.selected}
            return [r for r in rows if r["path"] in picked]

        def apply() -> None:
            for r in chosen_rows():
                r["name"] = (name.value or "").strip() or r["name"]
                r["group"] = (group.value or "").strip() or r["group"]
                r["format"] = format.value or r["format"]
            show()

        def remove() -> None:
            gone = {r["path"] for r in chosen_rows()}
            rows[:] = [r for r in rows if r["path"] not in gone]
            table.selected = []
            show()

        def load() -> None:
            unread = [r["file"] for r in rows if not r["format"]]
            if unread:
                ui.notify(f"Give a format to {', '.join(unread[:5])}", type="warning")
                return
            dialog.submit(True)

        with self.windows, ui.dialog() as dialog, \
                ui.card().classes("w-[64rem] max-w-full gap-2"):
            with ui.row().classes("w-full items-center gap-2"):
                ui.label("Load data").classes("text-base font-semibold")
                ui.button("Add…", icon="add", on_click=add_from) \
                    .props("outline dense no-caps").mark("collect-add") \
                    .tooltip("Files and folders, any number of each; a folder is searched")
                ui.space()
                search = ui.input(placeholder="search").props("dense outlined clearable") \
                    .classes("w-48").mark("collect-search")
            columns = [{"name": c, "label": c, "field": c, "sortable": True, "align": "left"}
                       for c in ("file", "group", "name", "format")]
            table = ui.table(columns=columns, rows=rows, row_key="path", selection="multiple",
                             pagination=0).classes("w-full").props("dense flat") \
                .style("max-height: 55vh").mark("collect-table")
            search.bind_value_to(table, "filter")
            with ui.row().classes("w-full items-center gap-2"):
                name = ui.input(label="name", autocomplete=list(collect.NAMES)) \
                    .props("dense outlined").classes("w-36").mark("collect-name")
                group = ui.input(label="group").props("dense outlined").classes("w-44") \
                    .mark("collect-group")
                format = ui.select({None: "as found", **{f: f for f in blocks.FORMATS}},
                                   value=None, label="format") \
                    .props("dense outlined").classes("w-32").mark("collect-format")
                ui.button("Apply to selected", on_click=apply).props("outline dense no-caps") \
                    .mark("collect-apply")
                ui.button("Remove selected", on_click=remove).props("outline dense no-caps") \
                    .mark("collect-remove")
                ui.space()
                ui.button("Cancel", on_click=lambda: dialog.submit(False)).props("flat")
                ui.button("Load", on_click=load).props("unelevated no-caps").mark("collect-load")
            add([collect.Entry(Path(p), n, g, f)
                 for p, n, g, f in self.graph.nodes[id].values.get("entries") or []])
        if not rows:
            ui.timer(0, add_from, once=True)            # nothing yet: straight to the files
        loaded = await dialog
        kept = [(r["path"], r["name"], r["group"], r["format"]) for r in rows]
        dialog.delete()
        if not loaded or id not in self.graph.nodes:
            return
        if not kept:
            ui.notify("Nothing to load: add files or folders", type="info")
            return
        await self._idle()                           # so they are loaded straight away
        if id not in self.graph.nodes:
            return
        self.graph.set(id, "entries", kept)
        self._changed()
        await self.run(id)

    #********************#
    #   functions kept   #
    #********************#
    async def functions_kept(self, id: str) -> None:
        """Which functions a merged basis keeps: every wired basis set's and spectrum's,
        ticked; a name in two places can be ticked in one only."""
        wired = [("basis", e) for e in self.graph.edges if (e.target, e.input) == (id, "basis")]
        wired += self.graph.labels(id, "more")
        if not wired:
            ui.notify("Wire a basis set in first", type="info")
            return
        for _, e in wired:                              # what they hold, made if need be
            if e.source in self.graph.nodes and self.graph.nodes[e.source].made is None:
                await self.run(e.source)
            if id not in self.graph.nodes or e.source not in self.graph.nodes or \
                    self.graph.nodes[e.source].made is None:
                return
        parts = {}
        for label, e in wired:
            value = self.graph.output(e.source, e.output)
            parts[label] = list(value.names) if isinstance(value, Basis) else [label]
        kept = self.graph.nodes[id].values.get("keep")
        if kept is None:                                # all, a shared name from its first
            keep: Dict[str, str] = {}
            for where, names in parts.items():
                for n in names:
                    keep.setdefault(n, where)
        else:                                           # what is still wired in
            keep = {n: w for n, w in kept.items() if n in parts.get(w, [])}
        boxes: Dict[Tuple[str, str], Any] = {}

        def tick(where: str, n: str, on: bool) -> None:
            if on:
                keep[n] = where
                for (w, m), box in boxes.items():       # elsewhere off: one per name
                    if m == n and w != where:
                        box.set_value(False)
            elif keep.get(n) == where:
                del keep[n]

        with self.windows, ui.dialog() as dialog, \
                ui.card().classes("w-[56rem] max-w-full gap-2"):
            ui.label("Functions the basis keeps").classes("text-base font-semibold")
            ui.label("A name two inputs share is kept from one of them.") \
                .classes("text-xs mr-muted")
            with ui.row().classes("w-full gap-6 items-start no-wrap overflow-auto"):
                for where, names in parts.items():
                    with ui.column().classes("gap-0 max-h-[55vh] overflow-auto no-wrap"):
                        ui.label(where).classes("mr-section")
                        for n in names:
                            boxes[where, n] = ui.checkbox(
                                n, value=keep.get(n) == where,
                                on_change=lambda ev, w=where, m=n: tick(w, m, ev.value)) \
                                .props("dense").mark(f"keep-{where}-{n}")
            with ui.row().classes("w-full"):
                ui.space()
                ui.button("Cancel", on_click=lambda: dialog.submit(False)).props("flat")
                ui.button("Keep", on_click=lambda: dialog.submit(True)) \
                    .props("unelevated no-caps").mark("keep-done")
        done = await dialog
        dialog.delete()
        if not done or id not in self.graph.nodes:
            return
        every = [(n, w) for w, names in parts.items() for n in names]
        whole = len({n for n, _ in every}) == len(every) and keep == dict(every)
        self._set(id, "keep", None if whole else keep)

    #**********#
    #   pick   #
    #**********#
    async def _pick(self, title: str = "Open data", both: bool = False, mode: str = "open",
                    name: str = "") -> Optional[List[Path]]:
        """Files, one folder or a file to save ('open', 'folder', 'save') from the system's
        own dialogs where there is a window; files and folders together, and everything in
        a browser, from the page's own picker."""
        from webview import FileDialog                  # picklable, as the window's queue needs

        window = nicegui.native.main_window
        if window is not None and not both:
            kind = {"open": FileDialog.OPEN, "folder": FileDialog.FOLDER,
                    "save": FileDialog.SAVE}[mode]
            found = await window.create_file_dialog(kind, allow_multiple=mode == "open",
                                                    directory=str(self.folder),
                                                    save_filename=name)
            found = [found] if isinstance(found, str) else found
            return [Path(p) for p in found] if found else None
        done: asyncio.Future = asyncio.get_running_loop().create_future()

        async def chosen(paths: List[Path]) -> None:
            if not done.done():
                done.set_result(paths)

        with self.windows:
            picker = Picker(self.folder, chosen, title, mode, name)
        picker.on_value_change(lambda e: e.value or done.done() or done.set_result(None))
        picker.open()
        out = await done
        picker.delete()
        return out

    #*************#
    #   preview   #
    #*************#
    @ui.refreshable
    def preview(self) -> None:
        node = self.graph.nodes.get(self.previewed or "")
        if node is None:
            with ui.row().classes("w-full items-center no-wrap"):
                ui.label("Press the eye on a block to see what it made.") \
                    .classes("mr-muted text-sm")
                ui.space()
                self._fold_button()
            return
        b = blocks.block(node.block)
        outs = self.graph.ports(node.id)[1]
        value = self.graph.output(node.id, self.port)
        count = len(value) if isinstance(value, NIfTI_MRS_Plus) or results(value) else 1
        self.index = min(self.index, max(count - 1, 0))
        with ui.row().classes("w-full items-center gap-3 no-wrap"):
            ui.element("span").classes("mr-swatch").style(
                f"background: {GROUPS[b.group].colour}")
            ui.label(b.title).classes("font-semibold")
            ui.label(node.name).classes("font-mono text-xs mr-muted")
            ui.space()
            if len(outs) > 1:
                ui.select({p.name: p.name for p in outs}, value=self.port or outs[0].name,
                          label="output", on_change=lambda e: self._show(port=e.value)) \
                    .props("dense outlined").classes("w-32")
            if count > 1:
                ui.number("spectrum", value=self.index, min=0, max=count - 1, step=1,
                          on_change=lambda e: self._show(index=int(e.value or 0))) \
                    .props("dense outlined").classes("w-28")
            tags = [t for t in getattr(value.list()[0], "dim_tags", [])[:value.list()[0].ndim - 4]
                    if t] if isinstance(value, NIfTI_MRS_Plus) else []
            if tags:                                   # coils, averages, ...: drawn apart
                ui.select({t: t.removeprefix("DIM_").lower() for t in tags},
                          value=self.spread if self.spread in tags else tags[0], label="spread",
                          on_change=lambda e: self._show(spread=e.value)) \
                    .props("dense outlined").classes("w-28").mark("spread")
                ui.slider(min=0, max=1, step=0.1, value=self.stack,
                          on_change=lambda e: self._show(stack=float(e.value or 0))) \
                    .props("dense label").classes("w-28 self-center").tooltip("stack") \
                    .mark("stack")
            if results(value):
                scalings = sorted(value[self.index].concentrations)
                self.scaling = self.scaling if self.scaling in scalings else \
                    ("raw" if "raw" in scalings else scalings[0] if scalings else None)
            self._fold_button()
        if node.error:
            ui.label(node.error).classes("text-negative text-sm")
        if node.made is None:
            ui.label("Not run yet: press ▶ on the block.").classes("mr-muted text-sm")
            return
        self.shown(value, count)

    #*****************#
    #   fold button   #
    #*****************#
    def _fold_button(self) -> None:
        ui.button(icon="expand_more", color=None, on_click=self.fold_preview) \
            .props("flat round dense").tooltip("Hide the preview").mark("fold-preview")

    #***********#
    #   shown   #
    #***********#
    @ui.refreshable
    def shown(self, value: Any, count: int) -> None:
        """The spectrum or fit chosen, refreshed alone so the spectrum box keeps its focus."""
        self.index = min(self.index, max(count - 1, 0))      # typed past the last
        if isinstance(value, NIfTI_MRS_Plus):
            ui.plotly(self._figure(value.list()[self.index])) \
                .classes("w-full grow basis-0 min-h-[12rem]")
        elif isinstance(value, Basis):
            ui.plotly(self._figure(value)).classes("w-full grow basis-0 min-h-[12rem]")
        elif drawn(value) is not None:
            picture = io.StringIO()
            drawn(value).savefig(picture, format="svg")
            data = base64.b64encode(picture.getvalue().encode()).decode()
            ui.image(f"data:image/svg+xml;base64,{data}").props("fit=contain") \
                .classes("w-full grow basis-0 min-h-[12rem]").mark("figure")
        elif results(value):
            columns, rows = table(value[self.index])
            ui.table(columns=[{"name": c, "label": c, "field": c, "align": "left"}
                              for c in columns],
                     rows=[dict(zip(columns, _round(r))) for r in rows],
                     pagination=0).classes("w-full").props("dense flat")
        elif hasattr(value, "to_dict") and hasattr(value, "columns"):
            ui.table(columns=[{"name": str(c), "label": str(c), "field": str(c), "align": "left"}
                              for c in value.columns],
                     rows=[{str(k): _round([v])[0] for k, v in r.items()}
                           for r in value.head(500).to_dict("records")],
                     pagination=0).classes("w-full").props("dense flat")
        else:
            ui.code(repr(value)[:20000], language="text").classes("w-full text-xs")

    #**********#
    #   show   #
    #**********#
    def _show(self, port: Optional[str] = None, index: Optional[int] = None,
              spread: Optional[str] = None, stack: Optional[float] = None) -> None:
        if index is None and spread is None and stack is None:
            self.port = port or self.port
            self.preview.refresh()
        else:
            self.index = self.index if index is None else index
            self.spread = spread or self.spread
            self.stack = self.stack if stack is None else stack
            self.shown.refresh()

    #************#
    #   figure   #
    #************#
    def _figure(self, spectrum: Any) -> dict:
        """A spectrum for plotly: the real part, a trace per coil or average, drawn apart
        and stacked as chosen (per metabolite for a basis set), ppm falling."""
        ppm, ys, labels = traces(spectrum, spread=self.spread, stack=self.stack)
        ink, grid = ("#e5e9ef", "#29313c") if self.dark.value else ("#18202b", "#dbe1ea")
        single = len(ys) == 1
        return {"data": [{"type": "scatter", "mode": "lines", "x": ppm.tolist(), "y": y.tolist(),
                          "name": label, "line": {"width": 1.2, **({"color": "#5274f0"}
                                                                    if single else {})}}
                         for y, label in zip(ys, labels)],
                "layout": {"xaxis": {"title": {"text": "ppm"}, "gridcolor": grid, "zeroline": False,
                                     **({"range": list(self.ppm_range)} if self.ppm_range
                                        else {"autorange": "reversed"})},
                           "yaxis": {"showticklabels": False, "gridcolor": grid,
                                     "zeroline": False},
                           "showlegend": not single, "font": {"color": ink},
                           "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)",
                           "margin": {"l": 10, "r": 10, "t": 10, "b": 40}}}

    #************#
    #   header   #
    #************#
    async def header(self, id: str) -> None:
        """What a block made, as NIfTI-MRS states it, in a window of its own: the data's
        shape and axes, every field of the header extension, and the processing history
        step by step with every argument; a fit's method and provenance; a basis set's
        source. Read off the live object, so it is always what an export would carry."""
        if self.graph.nodes[id].made is None:
            await self.run(id)
            if id not in self.graph.nodes or self.graph.nodes[id].made is None:
                return                                 # it failed, and said why
        node = self.graph.nodes[id]
        value = self.graph.output(id, self.port if id == self.previewed else None)
        index = self.index if id == self.previewed else 0
        with ui.dialog() as dialog, ui.card().classes("w-[84rem] max-w-full").mark("header"):
            with ui.row().classes("w-full items-center gap-3 no-wrap"):
                ui.label(blocks.block(node.block).title).classes("text-base font-semibold")
                ui.label(node.name).classes("font-mono text-xs mr-muted")
                ui.space()
                ui.label("As NIfTI-MRS records it; every export carries it.") \
                    .classes("mr-muted text-xs")
            described(value, index)
            with ui.row().classes("w-full"):
                ui.space()
                ui.button("Close", on_click=lambda: dialog.submit(None)).props("flat") \
                    .mark("header-close")
        await dialog
        dialog.delete()

    #************#
    #   export   #
    #************#
    async def export(self, id: str) -> None:
        """What a block made, written where asked."""
        if self.graph.nodes[id].made is None:
            await self.run(id)
            if id not in self.graph.nodes or self.graph.nodes[id].made is None:
                return                                 # it failed, and said why
        value = self.graph.output(id, self.port if id == self.previewed else None)
        kinds = exports(value)
        if not kinds:
            ui.notify("Nothing to write out here; Script writes the graph as Python",
                      type="info")
            return
        name = self.graph.nodes[id].name
        with ui.dialog() as dialog, ui.card().classes("w-[32rem] max-w-full"):
            ui.label(f"Export {name}").classes("text-base font-semibold")
            with ui.row().classes("w-full no-wrap items-center gap-1"):
                where = ui.input("Folder", value=str(self.folder)).classes("grow") \
                    .props("dense outlined").mark("export-folder")
                ui.button(icon="folder_open", on_click=lambda: self._browse(where, "folder")) \
                    .props("flat dense").tooltip("Choose a folder").mark("export-browse")
            with ui.row().classes("w-full"):
                ui.button("Cancel", on_click=lambda: dialog.submit(None)).props("flat")
                ui.space()
                for kind in kinds:
                    ui.button(kind, on_click=lambda _, k=kind: dialog.submit(k)) \
                        .props("unelevated no-caps").mark(f"export-{kind}")
        kind = await dialog
        folder = where.value
        dialog.delete()
        if not kind:
            return
        try:
            index = self.index if id == self.previewed else 0      # the spectrum shown
            written = await run.io_bound(export, value, folder, name, kind, index)
        except Exception as err:                       # noqa: BLE001 - said, not hidden
            _said(err)
            return
        self.folder = Path(folder)
        ui.notify(f"Wrote {len(written)} file{'s' * (len(written) != 1)} to {folder}",
                  type="positive")

    #************#
    #   script   #
    #************#
    async def script(self) -> None:
        """The graph as Python, to read, copy or save."""
        try:
            code = await run.io_bound(self.graph.script)
        except Exception as err:                       # noqa: BLE001 - said, not hidden
            _said(err)
            return
        with ui.dialog() as dialog, ui.card().classes("w-[60rem] max-w-full"):
            ui.label("The graph as Python").classes("text-base font-semibold")
            ui.label("Running it makes what the canvas made.").classes("mr-muted text-xs")
            ui.code(code, language="python").classes("w-full mr-code")
            with ui.row().classes("w-full no-wrap items-center gap-1"):
                path = ui.input("Save to", value=str(self.folder / "graph.py")).classes("grow") \
                    .props("dense outlined").mark("script-path")
                ui.button(icon="folder_open",
                          on_click=lambda: self._browse(path, "save", "graph.py")) \
                    .props("flat dense").tooltip("Choose where").mark("script-browse")
            with ui.row().classes("w-full"):
                ui.space()
                ui.button("Close", on_click=lambda: dialog.submit(None)).props("flat")
                ui.button("Save", icon="save", on_click=lambda: dialog.submit(path.value)) \
                    .props("unelevated no-caps").mark("script-save")
        target = await dialog
        dialog.delete()
        if target:
            try:
                Path(target).expanduser().write_text(code)
            except OSError as err:
                _said(err)
                return
            ui.notify(f"Wrote {target}", type="positive")

    #************#
    #   browse   #
    #************#
    async def _browse(self, box: Any, mode: str, name: str = "") -> None:
        """A path box filled from a browser: a folder, or a file to save."""
        chosen = await self._pick("Choose a folder" if mode == "folder" else "Save as",
                                  mode=mode, name=name)
        if chosen:
            box.set_value(str(chosen[0]))

    #**********#
    #   file   #
    #**********#
    def new(self) -> None:
        """An empty graph in a new tab, shown."""
        self.count += 1
        graph = Graph()
        self.tabs.append(Tab(graph, self.count, saved=_snapshot(graph)))
        self._switch(self.tabs[-1])

    #*************#
    #   tab bar   #
    #*************#
    @ui.refreshable
    def tabbar(self) -> None:
        """The graphs open, a tab each: its name, a dot once edited since its save, a cross
        to close it; and a plus for a new one."""
        with ui.row().classes("mr-tabs w-full no-wrap items-center gap-0"):
            for tab in self.tabs:
                shown = tab is self.tab
                with ui.element("div").classes("mr-tab" + (" mr-tab-active" if shown else "")) \
                        .props(f'data-dirty="{int(tab.dirty)}"') \
                        .on("click", lambda _, t=tab: self._switch(t)).mark(f"tab-{tab.number}"):
                    ui.label(tab.name + (" •" if tab.dirty else "")).classes("mr-tab-name")
                    ui.icon("close").classes("mr-tab-close") \
                        .on("click.stop", lambda _, t=tab: self.close_tab(t)) \
                        .mark(f"close-{tab.number}")
            ui.button(icon="add", on_click=self.new, color=None).props("flat dense round") \
                .classes("mr-tab-add").tooltip("An empty graph in a new tab")

    #************#
    #   switch   #
    #************#
    def _switch(self, tab: Tab) -> None:
        """Another tab in front: the one shown keeps its selection, preview and thumbnails;
        the block its run is under way in shows from the next one started."""
        if tab is self.tab:
            return
        keys = ("selected", "previewed", "port", "index", "scaling", "spread", "stack",
                "thumbs")
        if self.tab is not None:                       # none after the last tab closed
            self.tab.state = {k: getattr(self, k) for k in keys}
        self.tab, self.graph = tab, tab.graph
        fresh = {"selected": None, "previewed": None, "port": None, "index": 0, "scaling": None,
                 "spread": None, "stack": 0.0, "thumbs": {}}
        for k in keys:
            setattr(self, k, tab.state.get(k, fresh[k]))
        self.running = None
        self.status.text = ""
        self._changed()

    #**************#
    #   show tab   #
    #**************#
    def _show_tab(self, tab: Tab) -> None:
        """A tab opened and shown; an empty tab never edited makes way for it."""
        if not self.tab.graph.nodes and not self.tab.dirty and self.tab.path is None:
            self.tabs[self.tabs.index(self.tab)] = tab
        else:
            self.tabs.append(tab)
        self._switch(tab)

    #***************#
    #   close tab   #
    #***************#
    async def close_tab(self, tab: Tab) -> None:
        """A tab closed; one edited since its save asks first. The last one closed leaves an
        empty one."""
        if tab.dirty:
            with ui.dialog() as dialog, ui.card().classes("w-[26rem] max-w-full"):
                ui.label(f"Save the changes to {tab.name}?").classes("text-base font-semibold")
                ui.label("They are lost if you don't.").classes("text-sm mr-muted")
                with ui.row().classes("w-full"):
                    ui.button("Don't save", on_click=lambda: dialog.submit("discard")) \
                        .props("flat no-caps").mark("close-discard")
                    ui.space()
                    ui.button("Cancel", on_click=lambda: dialog.submit(None)).props("flat") \
                        .mark("close-cancel")
                    ui.button("Save", icon="save", on_click=lambda: dialog.submit("save")) \
                        .props("unelevated").mark("close-save")
            choice = await dialog
            dialog.delete()
            if choice is None:
                return
            if choice == "save":
                self._switch(tab)
                await self.save()
                if tab.dirty:                          # not saved after all
                    return
        if tab.graph in self.runs:                     # its run ends after the block under way
            tab.graph.stop()
        if tab is self.tab and len(self.tabs) > 1:
            self._switch(self.tabs[max(self.tabs.index(tab) - 1, 0)]
                         if self.tabs.index(tab) else self.tabs[1])
        self.tabs.remove(tab)
        if not self.tabs:
            self.count += 1
            graph = Graph()
            self.tabs.append(Tab(graph, self.count, saved=_snapshot(graph)))
            self.tab = None                            # so the switch takes the fresh one
            self._switch(self.tabs[0])
        self.tabbar.refresh()

    #**********#
    #   open   #
    #**********#
    async def open(self) -> None:
        """A saved graph in a new tab, named after its file."""
        chosen = await self._pick("Open a graph")
        if not chosen:
            return
        try:
            graph = await run.io_bound(Graph.load, chosen[0])
        except Exception as err:                       # noqa: BLE001 - said, not hidden
            _said(err)
            return
        self.folder = chosen[0].parent
        self._show_tab(Tab(graph, path=chosen[0], saved=_snapshot(graph)))
        self.canvas.fit()

    #**********#
    #   save   #
    #**********#
    async def save(self) -> None:
        """The graph to its file, or to one chosen where it has none yet."""
        if self.tab.path is None:
            await self.save_as()
            return
        self._save(self.tab.path)

    #*************#
    #   save as   #
    #*************#
    async def save_as(self) -> None:
        chosen = await self._pick("Save the graph", mode="save",
                                  name=self.tab.path.name if self.tab.path
                                  else "graph.mrsuite.json")
        if chosen:
            self._save(chosen[0])

    #**********#
    #   save   #
    #**********#
    def _save(self, path: Path) -> None:
        try:
            self.graph.save(path)
        except OSError as err:
            _said(err)
            return
        self.tab.path, self.tab.saved = path, _snapshot(self.graph)
        self.folder = path.parent
        self.tabbar.refresh()
        ui.notify(f"Saved {path}", type="positive")



#*************#
#   chooses   #
#*************#
def _chooses(b: blocks.Block) -> bool:
    """Whether a block's files are chosen with a button rather than typed: a load's, or a
    call's that takes files."""
    return not b.step and (b.key == "load" or any(f.name == "files" for f in b.fields))


#***********#
#   found   #
#***********#
def _found(paths: List[Any]) -> List[collect.Entry]:
    """Files and folders chosen, as entries, each once: a folder searched (one a format
    reads as a whole, a DICOM series or a Varian .fid, taken as one file), a file as it
    is."""
    found: Dict[Path, collect.Entry] = {}
    for path in map(Path, paths):
        for e in collect.discover(path) if path.is_dir() and not collect._format(path) \
                else collect.entries([path]):
            found.setdefault(e.path, e)                 # a file in a folder chosen too: once
    return list(found.values())


#************#
#   folder   #
#************#
def _folder(path: Any) -> Path:
    """Where to look next: a folder chosen, or the folder of a file chosen."""
    path = Path(path)
    return path if path.is_dir() else path.parent


#*************#
#   matches   #
#*************#
def _matches(b: blocks.Block, text: str) -> bool:
    """Whether a block answers a search: its name, its help, or a toolbox it offers."""
    words = [b.title, b.help, b.folder, *[f"{blocks.label(o)} {o[1]}" for o in b.offers]]
    return not text or any(text in w.lower() for w in words)


#**********#
#   said   #
#**********#
def _said(err: BaseException, logged: bool = False) -> None:
    """An error said on the page in one line, and, unless logged already, in the terminal in
    full, traceback and all, so a session run from a shell shows what went wrong while the
    GUI is young."""
    if not logged:
        logger.error("%s", brief(err), exc_info=err)
    ui.notify(brief(err), type="negative", multi_line=True)


#***************#
#   described   #
#***************#
def described(value: Any, index: int = 0) -> None:
    """A value's account of itself on the page: spectra by their NIfTI-MRS header and
    history, fits by their method and provenance, a basis set by its source."""
    def tree(value: Any, notes: Dict[str, str] = {}) -> None:
        """A structure read as a record: keys in front of their values, plain lists on one
        line, dicts and lists of structures as indented branches, JSON kept as text (a
        step's Details, and what it quotes) parsed into branches too; a note after a value
        where there is one (a unit, a field's description)."""
        items = value.items() if isinstance(value, dict) else enumerate(value, 1)
        for key, v in items:
            v = _parsed(v)
            if _branches(v):
                with ui.element("div").classes("mr-branch"):
                    ui.label(str(key)).classes("mr-key")
                    tree(v)
            else:
                with ui.row().classes("mr-leaf no-wrap items-baseline"):
                    ui.label(str(key)).classes("mr-key")
                    ui.label(_plain(v)).classes("mr-value")
                    if notes.get(key):
                        ui.label(notes[key]).classes("mr-muted text-xs")

    def record(title: str, items: List[Tuple[str, Any, str]]) -> None:
        ui.label(title).classes("mr-section mt-2")
        with ui.element("div").classes("mr-record"):
            tree({k: v for k, v, _ in items}, {k: about for k, _, about in items if about})

    def history(title: str, entries: List[Dict[str, Any]]) -> None:
        """The steps in order, each with its method, program, version and time, and its
        details underneath: as a branch where they are JSON, as a line where they are text."""
        ui.label(f"{title} · {len(entries)} step{'s' * (len(entries) != 1)}") \
            .classes("mr-section mt-2")
        if not entries:
            ui.label("none recorded").classes("mr-muted text-sm")
        for k, entry in enumerate(entries, 1):
            with ui.element("div").classes("mr-step"):
                with ui.row().classes("w-full no-wrap items-baseline gap-2"):
                    ui.label(str(k)).classes("mr-step-number")
                    ui.label(str(entry.get("Method", ""))).classes("mr-step-method").mark("step")
                    ui.label(f"{entry.get('Program', '')} {entry.get('Version', '')}") \
                        .classes("mr-muted text-sm").mark("program")
                    ui.space()
                    ui.label(_when(entry.get("Time", ""))).classes("mr-muted text-xs")
                details = _parsed(entry.get("Details", ""))
                if isinstance(details, dict):            # nothing passed is nothing to read
                    details = {k: v for k, v in details.items() if v not in ({}, [], "", None)}
                with ui.element("div").classes("mr-step-body mr-record").mark("details"):
                    if _branches(details):
                        tree(details)
                    elif details not in ("", None):
                        ui.label(_plain(details)).classes("mr-value")

    # what it is on the left, how it came about on the right
    with ui.row().classes("w-full no-wrap gap-4 h-[78vh]"):
        left = ui.scroll_area().classes("w-1/2 h-full")
        right = ui.scroll_area().classes("w-1/2 h-full")
    if isinstance(value, NIfTI_MRS_Plus):
        if not len(value):
            with left:
                ui.label("no spectra").classes("mr-muted text-sm")
            return
        item = value.list()[min(index, len(value) - 1)]
        fields = item.hdr_ext.to_dict()
        tags = [t for t in list(item.dim_tags)[:item.ndim - 4]]
        with left:
            record("Data", [("spectra in the batch", len(value), ""),
                            ("shape", list(item.shape), ""),
                            ("dimensions", ", ".join(f"{t}: {n}"
                                                     for t, n in zip(tags, item.shape[4:]))
                             or "one spectrum", ""),
                            ("points", item.shape[3], ""),
                            ("dwell time", item.dwelltime, "s"),
                            ("spectral width", 1.0 / item.dwelltime, "Hz"),
                            ("spectrometer frequency", item.spectrometer_frequency[0], "MHz"),
                            ("nucleus", item.nucleus[0], ""),
                            ("centre", item.SpecFreqChemShift, "ppm")])
            record("Header extension", [                  # user-defined fields come described
                (k, v.get("Value", v) if isinstance(v, dict) and "Description" in v else v,
                 v.get("Description", "") if isinstance(v, dict) else "")
                for k, v in fields.items() if k != "ProcessingApplied"])
        with right:
            history("Processing history", fields.get("ProcessingApplied", []))
    elif results(value):
        item = value[min(index, len(value) - 1)]
        with left:
            record("Fit", [("fits in the batch", len(value), ""), ("method", item.method, ""),
                           ("metabolites", ", ".join(item.metabolites), ""),
                           ("concentrations", ", ".join(sorted(item.concentrations)), "scalings"),
                           ("uncertainties", ", ".join(sorted(item.uncertainties)), ""),
                           *[(k, v, "quality") for k, v in item.qc.items()]])
        with right:
            history("Provenance", item.provenance)
    elif isinstance(value, Basis):
        with left:
            record("Basis set", [("functions", ", ".join(value.names), ""),
                                 ("points", value.fids.shape[1], ""),
                                 ("bandwidth", value.bandwidth, "Hz"),
                                 ("frequency", value.frequency, "MHz"),
                                 ("centre", value.centre, "ppm"), ("echo time", value.te, "ms"),
                                 ("sequence", value.seq, "")])
        with right:
            record("Source", [(k, v, "") for k, v in value.source.items()])
    else:
        with left:
            ui.code(repr(value)[:20000], language="text").classes("w-full text-xs")


#************#
#   parsed   #
#************#
def _parsed(value: Any) -> Any:
    """A string holding JSON, or a Python literal (Augmentrum writes its details so), as the
    structure it holds; anything else as it is."""
    if isinstance(value, str) and value[:1] in "{[":
        for read in (json.loads, ast.literal_eval):
            try:
                return read(value)
            except (ValueError, SyntaxError):
                pass
    return value


#**************#
#   branches   #
#**************#
def _branches(value: Any) -> bool:
    """Whether a value is drawn as a branch of the tree: a dict with something in it, or a
    list holding structures rather than plain values."""
    return (isinstance(value, dict) and bool(value)) or \
        (isinstance(value, list) and any(isinstance(v, (dict, list)) for v in value))


#***********#
#   plain   #
#***********#
def _plain(value: Any) -> str:
    """A value as a person reads it: words and numbers as they are, nothing quoted, a plain
    list joined by commas, an empty value as a dash."""
    if value is None or (isinstance(value, (str, dict, list, tuple)) and not value):
        return "—"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, numbers.Real) and not isinstance(value, numbers.Integral):
        return f"{value:.10g}"                       # numpy's floats too
    if isinstance(value, (list, tuple)):
        return ", ".join(_plain(v) for v in value)
    return str(value)


#**********#
#   when   #
#**********#
def _when(time: Any) -> str:
    """An ISO time as a person reads it; anything else as it came."""
    try:
        return datetime.fromisoformat(str(time)).strftime("%d %b %Y, %H:%M:%S")
    except ValueError:
        return str(time)


#***********#
#   round   #
#***********#
def _round(row: List[Any]) -> List[Any]:
    """A table row's numbers to four significant figures, for reading; the export keeps all."""
    return [f"{v:.4g}" if isinstance(v, float) else v for v in row]


#*************#
#   version   #
#*************#
def _version() -> str:
    from mrsuite.core.provenance import mrsuite_version
    return mrsuite_version()


#**********#
#   page   #
#**********#
def page() -> None:
    """The page, one graph per visitor."""
    ui.page_title("MRSuite")
    App()


#**********#
#   drop   #
#**********#
def drop(event: Any) -> None:
    """Files and folders dropped on the native window, loaded by a block of the page last
    opened, each named by its file name."""
    paths = [str(p) for p in event.args.get("files", []) if p]
    if paths and LATEST:
        app = LATEST[0]

        async def read() -> None:
            xs = [n.position[0] for n in app.graph.nodes.values()]
            id = app.graph.add("load", (min(xs) - 280 if xs else 80, 80))
            app.graph.set(id, "entries", collect.as_tuples(_found(paths)))
            app.selected = id
            app._changed()
            await app.run(id)

        with app.client:
            ui.timer(0, read, once=True)
