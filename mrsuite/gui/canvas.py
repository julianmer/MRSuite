####################################################################################################
#                                             canvas.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-07                                                                              #
#                                                                                                  #
# Purpose: The canvas the graph is drawn on: Vue Flow (MIT, vendored, so it runs offline) in a     #
#          NiceGUI element, and the graph as the canvas draws it.                                  #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
from typing import Any, Dict, List, Optional, Tuple

from nicegui import ui

from mrsuite.gui import blocks
from mrsuite.gui.graph import Graph, brief, thumb


#**************************************************************************************************#
#                                          Class Canvas                                            #
#**************************************************************************************************#
#                                                                                                  #
# Blocks wired together, as Vue Flow draws them.                                                   #
#                                                                                                  #
#**************************************************************************************************#
class Canvas(ui.element, component="canvas.js",
             esm={"@vue-flow/core": "vendor/vue-flow-core",
                  "@vue-flow/background": "vendor/vue-flow-background"}):
    """Blocks wired together, as Vue Flow draws them.

    Its events, each with its arguments: ``select`` (id, None for the canvas), ``act``
    (id, what: run, preview, export, remove), ``connect`` (Vue Flow's connection),
    ``loose`` (a wire let go on the canvas: where it came from, x, y), ``removed`` (kind,
    ids), ``moved`` (id, x, y each), ``search`` (x, y of a double click) and ``dropped``
    (a block's key from the palette, x, y).
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self) -> None:
        super().__init__()
        self._props["nodes"], self._props["edges"] = [], []

    #**********#
    #   show   #
    #**********#
    def show(self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> None:
        """Draw these, if they differ from what is drawn."""
        if (nodes, edges) != (self._props["nodes"], self._props["edges"]):
            self._props["nodes"], self._props["edges"] = nodes, edges
            self.update()

    #*********#
    #   fit   #
    #*********#
    def fit(self) -> None:
        """Zoom to show every block."""
        self.run_method("fit")


#**********#
#   view   #
#**********#
def view(graph: Graph, running: Optional[str], thumbs: Dict[str, Tuple[int, Any]]) \
        -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    The graph as Vue Flow takes it.

    Args:
        graph (Graph): The graph.
        running (Optional[str]): The block running now.
        thumbs (Dict[str, Tuple[int, Any]]): Pictures drawn so far, by block, with the id of
            what they show, so each is drawn once.

    Returns:
        The nodes and the edges.
    """
    nodes = []
    for node in graph.nodes.values():
        b = blocks.block(node.block)
        ins, outs = graph.ports(node.id)
        picture = None
        if node.made is not None and outs:
            value = graph.output(node.id)
            if thumbs.get(node.id, (None,))[0] != id(value):
                thumbs[node.id] = (id(value), thumb(value))
            picture = thumbs[node.id][1]
        status = "running" if node.id == running else "error" if node.error else \
            "done" if node.made is not None else "idle"
        option = b.offers[node.option] if b.offers else None
        entries = node.values.get("entries") or []
        files = [f"{len(entries)} file{'s' * (len(entries) != 1)}"] if entries else \
            [str(f).rstrip("/").rsplit("/", 1)[-1] for f in node.values.get("files") or []]
        wired = {e.input for e in graph.edges if e.target == node.id}
        set_ = {f.name: repr(node.values[f.name]) if f.name in node.values else f.default
                for f in graph.arguments(node.id) if f.name not in wired}  # what one unwired is
        nodes.append({
            "id": node.id, "type": "block",
            "position": {"x": node.position[0], "y": node.position[1]},
            "data": {"title": b.title, "name": node.name, "help": b.help,
                     "colour": blocks.GROUPS[b.group].colour,
                     "tool": blocks.label(option) if option else ", ".join(files),
                     "inputs": [{"name": p.name, "kind": p.kind, "many": p.many,
                                 "value": set_.get(p.name, "")} for p in ins],
                     "outputs": [{"name": p.name, "kind": p.kind} for p in outs],
                     "status": status, "thumb": picture,
                     "error": node.error and brief(node.error), "detail": node.error}})
    edges = [{"id": e.id, "source": e.source, "sourceHandle": e.output, "target": e.target,
              "targetHandle": e.input, "animated": e.target == running,
              "class": f"mr-kind-{_kind(graph, e)}"} for e in graph.edges]
    return nodes, edges


#**********#
#   kind   #
#**********#
def _kind(graph: Graph, edge: Any) -> str:
    """What flows along a wire, for its colour."""
    return next((p.kind for p in graph.ports(edge.source)[1] if p.name == edge.output), "any")
