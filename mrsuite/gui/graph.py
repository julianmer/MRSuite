####################################################################################################
#                                             graph.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-07                                                                              #
#                                                                                                  #
# Purpose: What the GUI builds, without the GUI: blocks wired into a graph, each block one line of #
#          Python. Running a block runs exactly its line, so the script the graph writes makes     #
#          what the GUI made. What a block made is kept until it or a block before it changes.     #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import ast
import json
import keyword
import re
import shutil
import threading
from dataclasses import dataclass, field
from inspect import Parameter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from nifti_mrs_plus import NIfTI_MRS_Plus
from mrsuite import collect
from mrsuite.core.basis import Basis
from mrsuite.core.result import FitResult
from mrsuite.gui import blocks, catalog
from mrsuite.gui.blocks import Block, Port

PIPELINE = "from mrsuite.core.pipeline import Pipeline"
PARTIAL = "from functools import partial"
KEYWORD_ONLY = Parameter.KEYWORD_ONLY

# names the script's imports bind, which a block's name must not take
TAKEN = {"Pipeline", "partial", *catalog.BRIDGES,
         *(line.split()[-1] for b in blocks.BLOCKS.values() for line in b.imports),
         *(where.rsplit(".", 1)[-1] for b in blocks.BLOCKS.values() for where, _ in b.analyses)}


#**************************************************************************************************#
#                                           Class Node                                             #
#**************************************************************************************************#
#                                                                                                  #
# One block placed in the graph.                                                                   #
#                                                                                                  #
#**************************************************************************************************#
@dataclass
class Node:
    """One block placed in the graph.

    Attributes:
        id (str): Its id in the graph.
        block (str): Its block's key (see ``blocks.block``).
        name (str): Its Python name in the script.
        option (int): The toolbox chosen, for a step.
        values (Dict[str, Any]): Arguments set in its form, by name; one wired in takes
            its value from the wire instead.
        shown (List[str]): Arguments with a default put on the block, to wire in (see
            ``Graph.show``); those without one are always on it.
        position (Tuple[float, float]): Where it sits on the canvas.
        found (List[str]): The outputs a load found when it last ran, beyond the names
            it was given (a GE P-file's own water reference).
        made (Optional[Dict[str, Any]]): The names its line assigned, with their values;
            None until it has run since it or a block before it changed.
        error (Optional[str]): Why it failed, when it last ran.
        changes (int): How often it or a block before it changed, so a run under way
            keeps what it made only if neither did meanwhile.
    """

    id: str
    block: str
    name: str
    option: int = 0
    values: Dict[str, Any] = field(default_factory=dict)
    shown: List[str] = field(default_factory=list)
    position: Tuple[float, float] = (0.0, 0.0)
    found: List[str] = field(default_factory=list)
    made: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    changes: int = 0


#**************************************************************************************************#
#                                           Class Edge                                             #
#**************************************************************************************************#
#                                                                                                  #
# A wire from one block's output to another's input.                                               #
#                                                                                                  #
#**************************************************************************************************#
@dataclass(frozen=True)
class Edge:
    """A wire from one block's output to another's input."""

    source: str
    output: str
    target: str
    input: str

    #********#
    #   id   #
    #********#
    @property
    def id(self) -> str:
        return f"{self.source}.{self.output}-{self.target}.{self.input}"


#**************************************************************************************************#
#                                           Class Graph                                            #
#**************************************************************************************************#
#                                                                                                  #
# Blocks wired together, each one line of Python, run as far as asked.                             #
#                                                                                                  #
#**************************************************************************************************#
class Graph:
    """Blocks wired together, each one line of Python, run as far as asked.

    Attributes:
        nodes (Dict[str, Node]): The blocks, by id.
        edges (List[Edge]): The wires.
    """

    #**********#
    #   init   #
    #**********#
    def __init__(self) -> None:
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self._count = 0
        self._changes = 0                    # counted, so a run under way sees an edit
        self._stop = threading.Event()
        self._lock = threading.Lock()        # an edit and a run storing what it made

    #***********#
    #   build   #
    #***********#
    def add(self, key: str, position: Tuple[float, float] = (0.0, 0.0)) -> str:
        """A block, by its key; its id."""
        b = blocks.block(key)
        self._count += 1
        node = Node(f"n{self._count}", key, self._name(b), position=tuple(position))
        node.values = self._pipeline(node)
        self.nodes[node.id] = node
        return node.id

    #************#
    #   remove   #
    #************#
    def remove(self, id: str) -> None:
        """A block and its wires."""
        self.invalidate(id)
        self.edges = [e for e in self.edges if id not in (e.source, e.target)]
        del self.nodes[id]

    #**********#
    #   move   #
    #**********#
    def move(self, id: str, position: Tuple[float, float]) -> None:
        self.nodes[id].position = tuple(position)

    #*************#
    #   connect   #
    #*************#
    def connect(self, source: str, output: str, target: str, input: str) -> Edge:
        """
        A wire; an input that takes one wire gives up the one it had.

        Raises:
            ValueError: For ports that are not there, kinds that do not meet, or a loop.
        """
        self.ask(target)
        outs = {p.name: p for p in self.ports(source)[1]}
        ins = {p.name: p for p in self.ports(target)[0]}
        if output not in outs or input not in ins:
            raise ValueError(f"no such port: {source}.{output} or {target}.{input}")
        if not meets(outs[output].kind, ins[input].kind):
            raise ValueError(f"{outs[output].kind} cannot go into {ins[input].kind}")
        if source == target or source in self.descendants(target):
            raise ValueError("that wire would make a loop")
        edge = Edge(source, output, target, input)
        self.edges = [e for e in self.edges if e != edge and not (
            not ins[input].many and (e.target, e.input) == (target, input))]
        self.edges.append(edge)
        self.invalidate(target)
        return edge

    #****************#
    #   disconnect   #
    #****************#
    def disconnect(self, edge: Edge) -> None:
        if edge in self.edges:
            self.edges.remove(edge)
            self.invalidate(edge.target)

    #*********#
    #   set   #
    #*********#
    def set(self, id: str, name: str, value: Any = None) -> None:
        """
        An argument in a block's form; None for its default. The same again changes
        nothing, so what the block made, and found, stays.

        Raises:
            ValueError: For a value Python text cannot give back, e.g. inf or nan, which
                the script and a saved graph could not hold.
        """
        node = self.nodes[id]
        if value is not None and not _written(value):
            raise ValueError(f"{name}: {value!r} cannot be written as Python and read back")
        if node.values.get(name) == value:
            return
        if value is None:
            node.values.pop(name, None)
        else:
            node.values[name] = value
        self.invalidate(id)                  # before wires go, so the blocks after it hear
        if node.block == "load":
            node.found = []
            self._drop_wires(node)

    #************#
    #   choose   #
    #************#
    def choose(self, id: str, option: int) -> None:
        """Another toolbox for a step, or analysis for a score: its own form, so its
        function's arguments start over, and the wires into arguments it does not have go."""
        node = self.nodes[id]
        b = blocks.block(node.block)
        self.invalidate(id)
        node.option = option
        own = {f.name for f in b.fields}                # a score's, kept
        node.values = self._pipeline(node) if b.step else \
            {k: v for k, v in node.values.items() if k in own}
        self.ask(id)                         # before the wires its arguments take are kept
        named = {f.name for f in self.arguments(id)}
        node.shown = [n for n in node.shown if n in named]
        self._drop_wires(node)

    #**********#
    #   show   #
    #**********#
    def show(self, id: str, name: str, on: bool = True) -> None:
        """An argument with a default on the block, to wire in, or off it again with its
        wire."""
        node = self.nodes[id]
        node.shown = [n for n in node.shown if n != name] + [name] * on
        wired = len(self.edges)
        self._drop_wires(node)
        if len(self.edges) < wired:
            self.invalidate(id)

    #****************#
    #   invalidate   #
    #****************#
    def invalidate(self, id: str) -> None:
        """Drop what a block and every block after it made; a run under way stops."""
        with self._lock:
            self._changes += 1
            for k in [id, *self.descendants(id)]:
                node = self.nodes[k]
                node.made, node.error, node.changes = None, None, node.changes + 1

    #***********#
    #   ports   #
    #***********#
    def ports(self, id: str) -> Tuple[List[Port], List[Port]]:
        """A block's inputs - the arguments it needs, and those with a default put on it
        (see ``show``) - and its outputs."""
        node = self.nodes[id]
        if node.block == "load":
            entries = node.values.get("entries") or []
            truths = {name for _, name, _, format in entries if format == "truth"}
            names = collect.order([*(name for _, name, _, _ in entries), *node.found])
            return [], [Port(n, blocks.TRUTH if n in truths else blocks.SPECTRA) for n in names]
        named = [f.name for f in self.arguments(id) if not f.default or f.name in node.shown]
        return blocks.ports(blocks.block(node.block), named)

    #***************#
    #   arguments   #
    #***************#
    def arguments(self, id: str) -> List[catalog.Field]:
        """The arguments that can come in by wire as well as from the form: a step's and a
        call's, once its function has been asked (see ``ask``), so drawing never waits on a
        toolbox; none where it cannot be asked (not installed), which its run says."""
        node = self.nodes[id]
        b = blocks.block(node.block)
        if not (b.step or b.call) or node.block == "load" or \
                (b.offers and not catalog.asked(*b.offers[node.option])):
            return []
        try:
            return self.fields(id)
        except Exception:                              # noqa: BLE001 - said when it runs
            return []

    #*********#
    #   ask   #
    #*********#
    def ask(self, id: str) -> None:
        """Read a step's arguments from its toolbox, which can take seconds the first time,
        or a score's from its analysis: its argument inputs exist from then on. One that
        cannot tell is said when the block runs."""
        b = blocks.block(self.nodes[id].block)
        if b.offers:
            catalog.read(*b.offers[self.nodes[id].option])

    #************#
    #   fields   #
    #************#
    def fields(self, id: str) -> List[catalog.Field]:
        """The arguments a block's form asks for: a step's, its function's (which may ask a
        toolbox, so it can take seconds the first time); a score's, its own and those its
        analysis takes by name."""
        node = self.nodes[id]
        b = blocks.block(node.block)
        if b.analyses:
            signature = catalog.about(*b.analyses[node.option])[0]
            return list(b.fields) + [f for f in catalog.fields(signature) if
                                     signature.parameters[f.name].kind == KEYWORD_ONLY]
        if not b.step:
            return list(b.fields)
        return catalog.fields(catalog.about(*b.options[node.option])[0])

    #**********#
    #   walk   #
    #**********#
    def parents(self, id: str) -> List[str]:
        return [e.source for e in self.edges if e.target == id]

    #*****************#
    #   descendants   #
    #*****************#
    def descendants(self, id: str) -> Set[str]:
        found, todo = set(), [id]
        while todo:
            k = todo.pop()
            for e in self.edges:
                if e.source == k and e.target not in found:
                    found.add(e.target)
                    todo.append(e.target)
        return found

    #***************#
    #   ancestors   #
    #***************#
    def ancestors(self, id: str) -> Set[str]:
        found, todo = set(), [id]
        while todo:
            for k in self.parents(todo.pop()):
                if k not in found:
                    found.add(k)
                    todo.append(k)
        return found

    #***********#
    #   order   #
    #***********#
    def order(self, ids: Optional[Set[str]] = None) -> List[str]:
        """Blocks so that each comes after the ones it takes from; all by default."""
        ids = set(self.nodes) if ids is None else ids
        out: List[str] = []
        while len(out) < len(ids):
            ready = [k for k in self.nodes if k in ids and k not in out
                     and all(p in out for p in self.parents(k))]
            if not ready:
                raise ValueError("the graph has a loop")      # connect refuses one
            out.extend(ready)
        return out

    #**********#
    #   code   #
    #**********#
    def code(self, id: str) -> Tuple[str, Set[str], List[str]]:
        """
        A block as its line of Python.

        Returns:
            The line, the imports it needs, and the names it assigns.

        Raises:
            ValueError: If an argument it needs is not set, or an input not wired.
        """
        node, b = self.nodes[id], blocks.block(self.nodes[id].block)
        wires = self._wires(node)
        needed = ["data"] if b.step else [p.name for p in b.inputs]
        missing = [name for name in needed if name not in wires]
        if missing:
            raise ValueError(f"{node.name}: wire something into {', '.join(missing)}")
        if b.step:
            where, function = b.options[node.option]
            args = self._args(node, self.fields(id), wires, b.wrap)
            if where in catalog.BRIDGES:
                line = f"Pipeline([{where}.step.{function}({args})]).run({wires['data']})"
                imports = {f"from mrsuite.bridges import {where}", PIPELINE}
            else:
                package, alias = where.rsplit(".", 1)
                line = f"{alias}.{function}({', '.join(filter(None, [wires['data'], args]))})"
                imports = {f"from {package} import {alias}"}
            return f"{node.name} = {line}", imports, [node.name]
        if b.model:
            checkpoint = node.values.get("checkpoint")
            if not checkpoint:
                raise ValueError(f"{node.name}: choose a checkpoint")
            options = self._args(node, [f for f in b.fields if f.name != "checkpoint"], wires,
                                 b.wrap)
            line = f"Pipeline([models.load({checkpoint!r}).step({options})]).run({wires['data']})"
            return f"{node.name} = {line}", {*b.imports, PIPELINE}, [node.name]
        if node.block == "load":
            chosen = node.values.get("entries")
            if not chosen:
                raise ValueError(f"{node.name}: add files or folders to load")
            listed = ",\n    ".join(repr(tuple(e)) for e in chosen)
            return f"{node.name} = collect.load([\n    {listed}])", set(b.imports), [node.name]
        unset = [f.name for f in b.fields if not f.default and f.name not in node.values
                 and f.name not in wires]
        if unset:
            raise ValueError(f"{node.name}: set {', '.join(unset)}")
        inputs = [catalog.Field(p.name, "value", [], "", True) for p in b.inputs]
        args = self._args(node, inputs + list(b.fields), wires, b.wrap)
        fixed = dict(b.fixed)
        imports = set(b.imports)
        if b.analyses:
            where, function = b.analyses[node.option]
            package, alias = where.rsplit(".", 1)
            own = self.fields(id)[len(b.fields):]
            unset = [f.name for f in own if not f.default and f.name not in node.values
                     and f.name not in wires]
            if unset:
                raise ValueError(f"{node.name}: set {', '.join(unset)}")
            given = self._args(node, own, wires, {})
            metric = f"partial({alias}.{function}, {given})" if given else f"{alias}.{function}"
            fixed["metrics"] = f"{{{function!r}: {metric}}}"
            imports |= {f"from {package} import {alias}", *[PARTIAL] * bool(given)}
        args = ", ".join(filter(None, [args, *(f"{k}={v}" for k, v in fixed.items())]))
        outputs = self.ports(id)[1]
        names = [node.name] if len(outputs) == 1 else [f"{node.name}_{p.name}" for p in outputs]
        return f"{', '.join(names)} = {b.call}({args})", imports, names

    #***********#
    #   wires   #
    #***********#
    def _wires(self, node: Node) -> Dict[str, str]:
        """What comes in by wire, as code: an input taking many, a dictionary by name (see
        ``labels``)."""
        ins = {p.name: p for p in self.ports(node.id)[0]}
        out: Dict[str, Any] = {}
        for e in self.edges:
            if e.target == node.id and not ins[e.input].many:
                out[e.input] = self.ref(e.source, e.output)
        for name in {p.name for p in ins.values() if p.many}:
            wired = self.labels(node.id, name)
            if wired:
                out[name] = "{" + ", ".join(f"{label!r}: {self.ref(e.source, e.output)}"
                                            for label, e in wired) + "}"
        return out

    #************#
    #   labels   #
    #************#
    def labels(self, id: str, input: str) -> List[Tuple[str, Edge]]:
        """
        The wires into an input taking many, each with its name there: its block's, or its
        output's where the block has several or is a load (data, mm, ...), with the block's
        before it, and a number after, where one would be taken - by another wire, or by
        another input of the block (a merge's 'basis').
        """
        out = []
        taken = {p.name for p in self.ports(id)[0]} - {input}
        for e in self.edges:
            if (e.target, e.input) != (id, input):
                continue
            source = self.nodes[e.source]
            single = len(self.ports(e.source)[1]) == 1 and not self._indexed(source)
            label = source.name if single else e.output
            label = f"{source.name}_{e.output}" if label in taken else label
            unique, k = label, 2
            while unique in taken:
                unique, k = f"{label}_{k}", k + 1
            taken.add(unique)
            out.append((unique, e))
        return out

    #*********#
    #   ref   #
    #*********#
    def ref(self, id: str, output: str) -> str:
        """The code that reaches one output of a block."""
        node = self.nodes[id]
        if self._indexed(node):
            return f"{node.name}[{output!r}]"
        return node.name if len(self.ports(id)[1]) == 1 else f"{node.name}_{output}"

    #**********#
    #   args   #
    #**********#
    @staticmethod
    def _args(node: Node, fields: List[catalog.Field], wires: Dict[str, str],
              wrap: Dict[str, str]) -> str:
        """
        The arguments as code: those taken in order by position up to the last one given,
        one left out before it as its default where that is a plain value, the rest by
        name; what is left unset keeps its default.

        Raises:
            ValueError: If one taken in order is left out before one that is given, and its
                default is worked out by the toolbox or there is none.
        """
        given = {f.name: wires[f.name] if f.name in wires else
                 wrap.get(f.name, "{}").format(repr(node.values[f.name]))
                 for f in fields if f.name in wires or f.name in node.values}
        more = node.values.get(catalog.MORE) or {}
        twice = [k for k in more if k in given]
        if twice:
            raise ValueError(f"{node.name}: {', '.join(twice)} is set in its own field or by "
                             "wire, and again in the further arguments")
        ordered = [f.name for f in fields if f.positional]
        last = max((k for k, name in enumerate(ordered) if name in given), default=-1)
        args = []
        defaults = {f.name: f.default for f in fields}
        for name in ordered[:last + 1]:
            if name not in given:
                try:
                    given[name] = repr(ast.literal_eval(defaults[name]))
                except (ValueError, SyntaxError):
                    raise ValueError(f"{node.name} takes its arguments in order: "
                                     f"set {name} too") from None
            args.append(given.pop(name))
        args += [f"{k}={v}" for k, v in given.items()]
        named = {k: v for k, v in more.items() if k.isidentifier() and not keyword.iskeyword(k)}
        args += [f"{k}={v!r}" for k, v in named.items()]
        spelt = {k: v for k, v in more.items() if k not in named}     # e.g. na.rm, lambda
        if spelt:
            args.append(f"**{spelt!r}")
        return ", ".join(args)

    #************#
    #   script   #
    #************#
    def script(self, upto: Optional[str] = None) -> str:
        """The graph, or what one block needs, as the Python that makes it."""
        from mrsuite.core.provenance import mrsuite_version

        ids = self.order(None if upto is None else self.ancestors(upto) | {upto})
        lines, imports = [], set()
        for k in ids:
            line, needs, _ = self.code(k)
            lines.append(line)
            imports |= needs
        return "\n".join([f"# Written by MRSuite {mrsuite_version()}: the graph as code.",
                          *sorted(imports), "", *lines, ""])

    #*********#
    #   run   #
    #*********#
    def run(self, upto: Optional[str] = None,
            progress: Optional[Callable[[str], None]] = None) -> bool:
        """
        Run the blocks that have not run since they or a block before them changed: all, or
        those one block needs. ``stop`` ends it after the block under way.

        Args:
            upto (Optional[str]): The block to run up to.
            progress: Told each block's id as it starts.

        Returns:
            bool: Whether it got to the end; not when stopped or the graph was edited, which
                keeps what the blocks before the edit made.

        Raises:
            Exception: What a block raised, which is kept on it as its error.
        """
        self._stop.clear()
        changes = self._changes
        for k in self.order(None if upto is None else self.ancestors(upto) | {upto}):
            node = self.nodes.get(k)
            if node is not None and node.made is not None:
                continue
            if progress:
                progress(k)
            if self._stop.is_set() or self._changes != changes or node is None:  # stopped,
                return False                                                    # or edited
            space: Dict[str, Any] = {}
            mine = node.changes
            try:
                line, imports, names = self.code(k)
                exec("\n".join(sorted(imports)), space)                    # noqa: S102
                for p in self.parents(k):
                    space.update(self.nodes[p].made)
                exec(line, space)                                          # noqa: S102
            except Exception as err:
                with self._lock:
                    if node.changes != mine:     # edited meanwhile: its failure is stale too
                        return False
                    node.error = f"{type(err).__name__}: {err}"
                raise
            with self._lock:
                if node.changes != mine:         # edited while it ran: what it made is stale
                    return False
                node.made = {n: space[n] for n in names}
            if self._indexed(node):
                node.found = list(node.made[node.name])
                self._drop_wires(node)
        return True

    #**********#
    #   stop   #
    #**********#
    def stop(self) -> None:
        self._stop.set()

    #************#
    #   output   #
    #************#
    def output(self, id: str, port: Optional[str] = None) -> Any:
        """What a block made at one output, its first by default; None until it has run."""
        node = self.nodes[id]
        if node.made is None:
            return None
        outputs = self.ports(id)[1]
        port = port or (outputs[0].name if outputs else None)
        if self._indexed(node):
            return node.made[node.name].get(port)
        return node.made[self.ref(id, port)] if port else None

    #**********#
    #   save   #
    #**********#
    def to_dict(self) -> Dict[str, Any]:
        """The graph as JSON, its arguments as Python text so they read back exactly."""
        from mrsuite.core.provenance import mrsuite_version

        return {"format": "mrsuite.graph", "mrsuite": mrsuite_version(),
                "nodes": [{"id": n.id, "block": n.block, "name": n.name, "option": n.option,
                           "values": {k: repr(v) for k, v in n.values.items()},
                           "shown": n.shown,
                           "position": list(n.position), "found": n.found}
                          for n in self.nodes.values()],
                "edges": [[e.source, e.output, e.target, e.input] for e in self.edges]}

    #***************#
    #   from dict   #
    #***************#
    @classmethod
    def from_dict(cls, saved: Dict[str, Any]) -> "Graph":
        """
        A graph again from ``to_dict``.

        Raises:
            ValueError: If it is not a saved graph.
        """
        if saved.get("format") != "mrsuite.graph":
            raise ValueError("not a saved graph: its format is not 'mrsuite.graph'")
        graph, where = cls(), "the graph"
        try:
            if len({n["name"] for n in saved["nodes"]}) != len(saved["nodes"]):
                raise ValueError("two blocks share a name")
            for n in saved["nodes"]:
                where = f"block {n.get('id')!r}"
                b = blocks.block(n["block"])
                shown = list(n.get("shown", []))              # none in graphs saved before
                names = [n["name"], *n["values"], *shown]         # found: only ever quoted
                if not re.fullmatch(r"n\d+", n["id"]) or n["name"] in TAKEN or not all(
                        _identifier(k) for k in names if k != catalog.MORE):
                    raise ValueError("its id or a name is not one MRSuite writes")
                if not 0 <= n["option"] < max(len(b.offers), 1):
                    raise ValueError(f"no toolbox {n['option']}")
                values = {k: ast.literal_eval(v) for k, v in n["values"].items()}
                graph.nodes[n["id"]] = Node(n["id"], n["block"], n["name"], n["option"], values,
                                            shown, tuple(n["position"]), list(n["found"]))
            graph._count = max((int(k[1:]) for k in graph.nodes), default=0)
            for e in saved["edges"]:
                where = f"wire {e!r}"
                graph.connect(*e)
        except (KeyError, TypeError, ValueError, SyntaxError) as err:
            hint = " (a toolbox it uses may not be installed here)" \
                if "no such port" in str(err) else ""
            raise ValueError(f"not a graph MRSuite can read, at {where}: "
                             f"{type(err).__name__}: {err}{hint}") from None
        return graph

    #**********#
    #   save   #
    #**********#
    def save(self, path: Any) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    #**********#
    #   load   #
    #**********#
    @classmethod
    def load(cls, path: Any) -> "Graph":
        return cls.from_dict(json.loads(Path(path).read_text()))

    #*************#
    #   helpers   #
    #*************#
    def _name(self, b: Block) -> str:
        """A Python name for a new block, not yet taken."""
        name = re.sub(r"\W", "_", b.key.split(":")[-1]).lower()
        name = "_" + name if name[0].isdigit() else name
        if name in TAKEN or keyword.iskeyword(name):
            name += "_"
        taken = {n.name for n in self.nodes.values()}
        unique, k = name, 2
        while unique in taken:
            unique, k = f"{name}_{k}", k + 1
        return unique

    #**************#
    #   pipeline   #
    #**************#
    @staticmethod
    def _pipeline(node: Node) -> Dict[str, Any]:
        """A step's form as its toolbox's own pipeline starts it (see blocks.PIPELINE)."""
        b = blocks.block(node.block)
        return dict(blocks.PIPELINE.get(b.options[node.option], ({}, ""))[0]) if b.step else {}

    #*************#
    #   indexed   #
    #*************#
    @staticmethod
    def _indexed(node: Node) -> bool:
        """Whether a block's outputs are the keys of what it made: a load's names."""
        return node.block == "load"

    #****************#
    #   drop wires   #
    #****************#
    def _drop_wires(self, node: Node) -> None:
        """Wires from outputs and into inputs a block no longer has."""
        ins, outs = ({p.name for p in side} for side in self.ports(node.id))
        self.edges = [e for e in self.edges if (e.source != node.id or e.output in outs)
                      and (e.target != node.id or e.input in ins)]


#***********#
#   brief   #
#***********#
def brief(error: Any, limit: int = 240) -> str:
    """
    An error as one line to read: its kind and first line, with the next while a line
    ends in a colon (Octave's "evaluation error:"), and where its message carries a
    traceback (a toolbox's own, e.g. spec2nii's), the traceback's last line; cut short.
    """
    text = error if isinstance(error, str) else f"{type(error).__name__}: {error}"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if any(t.startswith("Traceback") for t in lines):
        lines = lines[:1] + lines[-1:]
    head = next((k for k, t in enumerate(lines) if not t.endswith(":")), len(lines) - 1)
    line = " ".join(lines[:head + 1])
    return line if len(line) <= limit else line[:limit - 1] + "…"


#****************#
#   identifier   #
#****************#
def _identifier(name: Any) -> bool:
    """Whether a name can stand in Python code as written."""
    return isinstance(name, str) and name.isidentifier() and not keyword.iskeyword(name)


#*************#
#   written   #
#*************#
def _written(value: Any) -> bool:
    """Whether a value's Python text reads back as the value."""
    try:
        return ast.literal_eval(repr(value)) == value
    except (ValueError, SyntaxError, TypeError):
        return False


#***********#
#   meets   #
#***********#
def meets(output: str, input: str) -> bool:
    """Whether what an output makes can go into an input."""
    return blocks.ANY in (output, input) or output == input


#*************#
#   results   #
#*************#
def results(value: Any) -> bool:
    """Whether a value is fit results, one per spectrum."""
    return isinstance(value, list) and bool(value) and all(isinstance(r, FitResult) for r in value)


#************#
#   traces   #
#************#
def traces(spectrum: Any, limit: int = 64, spread: Optional[str] = None,
           stack: float = 0.0) -> Tuple[np.ndarray, List[np.ndarray], List[str]]:
    """
    A spectrum's real part on its ppm axis, one trace per element of one higher dimension
    (its first by default), at most ``limit``; a basis set's, one per metabolite, by name.

    Args:
        spectrum: NIfTI-MRS, or a Basis.
        limit (int): Traces at most.
        spread (Optional[str]): The dimension to draw element by element, by its tag, e.g.
            'DIM_COIL' or 'DIM_DYN'. The others are averaged where they are averages
            (DIM_DYN) and taken at their first element otherwise.
        stack (float): How far each trace sits above the one before, as a fraction of
            the largest amplitude; 0 draws them over each other.

    Returns:
        The ppm axis, the traces and a label for each, e.g. 'coil 3'.
    """
    if isinstance(spectrum, Basis):
        return (spectrum.ppm(), list(spectrum.spectra()[:limit].real),
                list(spectrum.names[:limit]))
    data = np.asarray(spectrum[:])
    tags = list(spectrum.dim_tags)[:data.ndim - 4]
    axis = 4 + tags.index(spread) if spread in tags else 4
    for d in reversed(range(4, data.ndim)):
        if d != axis:
            data = data.mean(axis=d, keepdims=True) if tags[d - 4] == "DIM_DYN" \
                else data.take([0], axis=d)
    fids = np.moveaxis(data, 3, 0).reshape(data.shape[3], -1)
    specs = np.fft.fftshift(np.fft.fft(fids, axis=0), axes=0)
    n, dwell = fids.shape[0], spectrum.dwelltime
    hz = np.fft.fftshift(np.fft.fftfreq(n, dwell))
    ppm = hz / spectrum.spectrometer_frequency[0] + spectrum.SpecFreqChemShift   # by nucleus
    count = min(specs.shape[1], limit)
    ys = [specs[:, j].real for j in range(count)]
    if stack and count > 1:
        step = stack * max(float(np.abs(y).max()) for y in ys)
        ys = [y + j * step for j, y in enumerate(ys)]
    header = spectrum.hdr_ext.to_dict().get(f"dim_{axis + 1}_header", {})   # NIfTI counts from 1
    named = next(iter(header.values()), None)        # one list per element, or {'Value': ...}
    named = named.get("Value") if isinstance(named, dict) else named
    if isinstance(named, list) and len(named) >= count:     # e.g. WAND's components by name
        return ppm, ys, [str(v) for v in named[:count]]
    name = (tags[axis - 4] or "DIM").removeprefix("DIM_").lower() + " " if tags else ""
    return ppm, ys, [f"{name}{j}" for j in range(count)]


#***********#
#   table   #
#***********#
def table(result: FitResult) -> Tuple[List[str], List[List[Any]]]:
    """A fit's concentrations, a column per scaling and per uncertainty."""
    columns = sorted(result.concentrations) + [f"{k} (sd)" for k in sorted(result.uncertainties)]
    rows = [[m] + [result.concentrations[k].get(m) for k in sorted(result.concentrations)]
            + [result.uncertainties[k].get(m) for k in sorted(result.uncertainties)]
            for m in result.metabolites]
    return ["metabolite"] + columns, rows


#***********#
#   thumb   #
#***********#
def thumb(value: Any, points: int = 120) -> Dict[str, Any]:
    """
    A small picture of a value, for its block: a spectrum's mean real part over 0.5-4.2 ppm
    where it covers them, ppm falling, scaled to 0-1; a basis set's spectra the same way,
    stacked one above the other as a basis set is shown; a fit's largest concentrations as
    bars; anything else in a few words.
    """
    def picture(ppm, ys):
        """Each trace over 0.5-4.2 ppm, ppm falling, in ``points`` points keeping the peaks."""
        ppm, ys = ppm[::-1], [y[::-1] for y in ys]
        keep = (ppm > 0.5) & (ppm < 4.2)
        ys = [y[keep] if keep.sum() > points / 4 else y for y in ys]
        return [np.array([c[np.argmax(np.abs(c))] for c in np.array_split(y, min(points, len(y)))])
                for y in ys]

    if isinstance(value, NIfTI_MRS_Plus) and len(value):
        ppm, ys, _ = traces(value.list()[0])
        y, = picture(ppm, [np.mean(ys, axis=0)])
        span = np.ptp(y) or 1.0
        return {"kind": "line", "y": [round(float(v), 3) for v in (y - y.min()) / span]}
    if isinstance(value, Basis):
        ppm, ys, _ = traces(value)
        ys = picture(ppm, ys)
        step = 0.7 / max(len(ys), 1)           # each spectrum a step up, each at its own height
        return {"kind": "lines", "ys": [[round(float(v), 3) for v in
                                         np.clip(i * step + 0.3 * y / (np.abs(y).max() or 1.0),
                                                 0, 1)]
                                        for i, y in enumerate(ys)]}
    if results(value):
        scaling = "raw" if "raw" in value[0].concentrations else \
            next(iter(value[0].concentrations), None)
        concs = value[0].concentrations.get(scaling, {})
        top = sorted(((m, c) for m, c in concs.items() if np.isfinite(c or np.nan)),
                     key=lambda mc: -abs(mc[1]))[:8]
        peak = max((abs(c) for _, c in top), default=1.0) or 1.0
        return {"kind": "bars", "y": [round(abs(c) / peak, 3) for _, c in top],
                "labels": [m for m, _ in top], "text": f"{len(value)} fits"}
    if isinstance(value, np.ndarray) and value.size <= 4 or isinstance(value, (int, float)):
        return {"kind": "text", "text": np.array2string(np.asarray(value), precision=4)}
    if hasattr(value, "shape"):
        return {"kind": "text", "text": f"{type(value).__name__} {tuple(value.shape)}"}
    if isinstance(value, (list, tuple, dict)):
        return {"kind": "text", "text": f"{type(value).__name__} of {len(value)}"}
    if isinstance(value, Path):
        return {"kind": "text", "text": value.name}
    if drawn(value) is not None:
        return {"kind": "text", "text": "figure"}
    return {"kind": "text", "text": type(value).__name__}


#***********#
#   drawn   #
#***********#
def drawn(value: Any) -> Any:
    """The matplotlib figure a value is, or is drawn on (an Axes); None for anything else."""
    from matplotlib.figure import Figure

    figure = value if isinstance(value, Figure) else getattr(value, "figure", None)
    return figure if isinstance(figure, Figure) else None


#************#
#   export   #
#************#
def exports(value: Any) -> List[str]:
    """What a value can be written out as."""
    if isinstance(value, NIfTI_MRS_Plus):
        return ["spectra", "figure"]
    if isinstance(value, Basis):
        return ["basis"]
    if results(value):
        return ["results"]
    if hasattr(value, "to_csv"):
        return ["table"]
    if drawn(value) is not None:
        return ["figure"]
    return []


#************#
#   export   #
#************#
def export(value: Any, folder: Any, name: str, what: str, index: int = 0) -> List[Path]:
    """
    Write a value out.

    Args:
        value: What a block made.
        folder: Where.
        name (str): The files' name, the block's.
        what (str): 'spectra' (NIfTI-MRS), 'figure' (SVG, of a spectrum or a plot made),
            'basis' (LCModel .BASIS), 'results' (JSON) or 'table' (CSV); see ``exports``.
        index (int): Which spectrum the figure shows.

    Returns:
        List[Path]: The files written.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if what not in exports(value):
        raise ValueError(f"this cannot be written out as {what}")
    if what == "spectra":
        return [Path(p) for p in value.save_nifti(str(folder), prefix=name)]
    if what == "figure" and drawn(value) is not None:
        drawn(value).savefig(folder / f"{name}.svg")
        return [folder / f"{name}.svg"]
    if what == "figure":
        return [figure(value, folder / f"{name}.svg", index)]
    if what == "basis":
        return [Path(shutil.copy(value.lcmodel(), folder / f"{name}.BASIS"))]
    if what == "results":
        from mrsuite.core import result
        path = folder / f"{name}.json"
        result.save(value, str(path), raw=False)
        return [path]
    path = folder / f"{name}.csv"
    value.to_csv(path, index=False)
    return [path]


#************#
#   figure   #
#************#
def figure(value: Any, path: Path, index: int = 0) -> Path:
    """A spectrum as an SVG for a paper: the real part, ppm falling."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    ppm, ys, _ = traces(value.list()[index])
    fig, ax = plt.subplots(figsize=(6, 3))
    for y in ys:
        ax.plot(ppm, y, linewidth=0.8)
    ax.set_xlim(ppm.max(), ppm.min())
    ax.set_xlabel("ppm")
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
