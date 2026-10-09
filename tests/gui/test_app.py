####################################################################################################
#                                           test_app.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-07                                                                              #
#                                                                                                  #
# Purpose: The GUI's page, driven as a user would, without a browser: what it shows, adding and    #
#          wiring blocks, setting one up, running the graph and writing it out.                    #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import asyncio
import pickle

import pytest

pytest.importorskip("nicegui")

import nicegui                                                          # noqa: E402
from nicegui import ui                                                  # noqa: E402
from nicegui.testing.user_simulation import user_simulation            # noqa: E402

from mrsuite.datasets import challenge2016                              # noqa: E402
from mrsuite.gui.app import App                                         # noqa: E402
from mrsuite.gui.canvas import Canvas                                   # noqa: E402
from mrsuite.gui.graph import Graph                                     # noqa: E402

needs_challenge = pytest.mark.skipif(not challenge2016.BASIS.exists(),
                                     reason="challenge submodule not checked out")


#**************#
#   simulate   #
#**************#
def simulate(graph, script):
    """Open the page over a graph and play a user's part on it, the page at hand."""
    apps = []

    async def play():
        async with user_simulation(root=lambda: apps.append(App(graph))) as user:
            await user.open("/")
            await asyncio.wait_for(script(user, apps[-1]), 300)   # a hang fails, with where
    asyncio.run(play())


#***********#
#   until   #
#***********#
async def until(condition, seconds: float = 120.0) -> None:
    """Wait for work in a thread, as the page does."""
    for _ in range(int(seconds / 0.1)):
        if condition():
            return
        await asyncio.sleep(0.1)
    raise AssertionError("it did not happen in time")


#***********#
#   drawn   #
#***********#
def drawn(user):
    """What the canvas is drawing."""
    canvas = next(iter(user.find(Canvas).elements))
    return canvas.props["nodes"], canvas.props["edges"]


#***********#
#   empty   #
#***********#
def test_an_empty_page_says_where_to_start():
    async def script(user, app):
        await user.should_see("Drag blocks from the left onto the canvas", kind=ui.label)
        await user.should_see("Press the eye on a block to see what it made.")
        await user.should_see("Coil combination")
        assert drawn(user) == ([], [])

    simulate(Graph(), script)


#*********#
#   add   #
#*********#
def test_a_block_clicked_is_added_after_the_selected_one_and_wired_to_it():
    graph = Graph()
    data = graph.add("ismrm2016")

    async def script(user, app):
        app.select(data)
        user.find(marker="block-averaging").click()
        await until(lambda: len(graph.nodes) == 2)
        nodes, edges = drawn(user)
        assert [n["data"]["title"] for n in nodes] == ["2016 fitting challenge", "Averaging"]
        assert [(e["source"], e["sourceHandle"], e["targetHandle"]) for e in edges] == \
            [(data, "spectra", "data")]
        await user.should_see("Toolbox")               # the new block, selected

    simulate(graph, script)


def test_a_search_finds_blocks_by_toolbox():
    async def script(user, app):
        user.find(marker="find").type("osprey")
        await user.should_see("Linear-combination fit")
        await user.should_not_see("Averaging ")
        await user.should_not_see("Read files")

    simulate(Graph(), script)


#**********#
#   form   #
#**********#
def test_a_setting_typed_reaches_the_graph():
    graph = Graph()
    data, apodize = graph.add("ismrm2016"), graph.add("apodization")
    graph.connect(data, "spectra", apodize, "data")

    async def script(user, app):
        app.select(apodize)
        await asyncio.sleep(0.1)                       # the page redrawn
        await until(lambda: isinstance(app.about.get(("fsl_mrs", "apodize")), tuple))
        await asyncio.sleep(0.1)                       # its arguments read, the form drawn
        user.find(marker="field-amount").type("[5.0]").trigger("keydown.enter")
        await until(lambda: graph.nodes[apodize].values == {"amount": [5.0]})

    simulate(graph, script)


def test_another_toolbox_is_chosen_from_the_inspector():
    graph = Graph()
    step = graph.add("averaging")

    async def script(user, app):
        app.select(step)
        await asyncio.sleep(0.1)                       # the page redrawn
        box = next(iter(user.find(marker="toolbox").elements))
        box.value = 1
        await until(lambda: graph.nodes[step].option == 1)
        assert drawn(user)[0][0]["data"]["tool"] == "FID-A · op_averaging"

    simulate(graph, script)


#*********#
#   run   #
#*********#
@needs_challenge
def test_the_graph_runs_from_the_page_and_is_written_out(tmp_path):
    graph = Graph()
    data, apodize = graph.add("ismrm2016"), graph.add("apodization")
    graph.set(data, "n", 1)
    graph.set(apodize, "amount", [5.0])
    graph.connect(data, "spectra", apodize, "data")

    async def script(user, app):
        user.find(marker="run").click()
        await until(lambda: graph.nodes[apodize].made is not None and not app.busy)
        nodes, _ = drawn(user)
        assert [n["data"]["status"] for n in nodes] == ["done", "done"]
        assert nodes[1]["data"]["thumb"]["kind"] == "line"
        await app.act(apodize, "preview")
        await asyncio.sleep(0.1)                       # the page redrawn
        assert user.find(ui.plotly).elements, "the spectrum is drawn"
        user.find(marker="script").click()
        await user.should_see("Running it makes what the canvas made.")
        user.find(marker="script-path").clear().type(str(tmp_path / "graph.py"))
        user.find(marker="script-save").click()
        await until(lambda: (tmp_path / "graph.py").exists())

    simulate(graph, script)
    assert (tmp_path / "graph.py").read_text() == graph.script()


@needs_challenge
def test_a_failure_shows_on_its_block():
    graph = Graph()
    data, fit = graph.add("ismrm2016"), graph.add("lcm-fit")
    graph.set(data, "n", 1)
    graph.connect(data, "spectra", fit, "data")       # no basis set

    async def script(user, app):
        await app.run()
        await asyncio.sleep(0.1)                       # the page redrawn
        nodes, _ = drawn(user)
        assert nodes[1]["data"]["status"] == "error"
        assert "basis" in nodes[1]["data"]["error"]

    simulate(graph, script)


#***********#
#   wires   #
#***********#
def test_a_wire_that_does_not_fit_is_said():
    graph = Graph()
    data, fit = graph.add("ismrm2016"), graph.add("lcm-fit")

    async def script(user, app):
        app.connect(data, "truth", fit, "data")
        await user.should_see("truth cannot go into spectra")
        assert graph.edges == []

    simulate(graph, script)


def test_a_block_gone_before_its_toolbox_answered_is_let_go(caplog):
    """Its arguments are read in a thread; removed, or the graph replaced, meanwhile."""
    graph = Graph()

    async def script(user, app):
        gone = app.add("averaging", (0, 0), join=False)
        app.removed("node", [gone])
        await asyncio.sleep(0.5)
        app.add("averaging", (0, 0), join=False)
        app.new()
        await asyncio.sleep(0.5)
        assert not app.graph.nodes

    simulate(graph, script)
    assert "KeyError" not in caplog.text


def test_blocks_and_wires_deleted_on_the_canvas_leave_the_graph():
    graph = Graph()
    data, one, two = graph.add("ismrm2016"), graph.add("averaging"), graph.add("averaging")
    graph.connect(data, "spectra", one, "data")
    graph.connect(data, "water", two, "data")

    async def script(user, app):
        app.removed("edge", [graph.edges[0].id])
        app.removed("node", [two])
        assert list(graph.nodes) == [data, one] and graph.edges == []
        assert len(drawn(user)[0]) == 2

    simulate(graph, script)


#************#
#   export   #
#************#
@needs_challenge
def test_what_a_block_made_is_exported_where_asked(tmp_path):
    graph = Graph()
    data = graph.add("ismrm2016")
    graph.set(data, "n", 1)

    async def script(user, app):
        await app.run()
        async def click():                             # as the block's button does, on the page
            with app.client:
                await app.export(data)
        task = asyncio.create_task(click())
        await user.should_see("Export ismrm2016")
        user.find(marker="export-folder").clear().type(str(tmp_path))
        user.find(marker="export-spectra").click()
        await task

    simulate(graph, script)
    assert len(list(tmp_path.glob("ismrm2016*.nii.gz"))) == 1


@needs_challenge
def test_the_figure_exported_is_the_spectrum_shown(tmp_path):
    import re
    from mrsuite.gui.graph import export

    graph = Graph()
    data = graph.add("ismrm2016")
    graph.set(data, "n", 2)

    async def script(user, app):
        await app.run()
        app.previewed, app.index = data, 1
        async def click():
            with app.client:
                await app.export(data)
        task = asyncio.create_task(click())
        await user.should_see("Export ismrm2016")
        user.find(marker="export-folder").clear().type(str(tmp_path))
        user.find(marker="export-figure").click()
        await task

    simulate(graph, script)
    shown, = export(graph.output(data), tmp_path / "ref", "shown", "figure", index=1)
    first, = export(graph.output(data), tmp_path / "ref", "first", "figure")
    line = lambda p: max(re.findall(r' d="([^"]*)"', p.read_text()), key=len)   # the spectrum
    assert line(tmp_path / "ismrm2016.svg") == line(shown) != line(first)


#**********#
#   save   #
#**********#
def test_a_new_graph_starts_empty():
    graph = Graph()
    graph.add("ismrm2016")

    async def script(user, app):
        user.find(marker="new").click()
        await until(lambda: not app.graph.nodes)
        assert drawn(user) == ([], [])

    simulate(graph, script)


#**********#
#   open   #
#**********#
def test_the_native_open_dialog_gets_what_can_reach_the_window(monkeypatch, tmp_path):
    asked = []

    class Window:                                      # as NiceGUI's: arguments go by a queue
        async def create_file_dialog(self, *args, **kwargs):
            asked.append(pickle.dumps((args, kwargs)))
            return (str(tmp_path / "a.nii.gz"),)

    async def script(user, app):
        with monkeypatch.context() as patch:               # the window only while picking
            patch.setattr(nicegui.app.native, "main_window", Window())
            assert await app._pick() == ([tmp_path / "a.nii.gz"], "auto")

    simulate(Graph(), script)
    assert len(asked) == 1


def test_the_native_window_can_choose_a_folder_too(monkeypatch, tmp_path):
    """A DICOM series is a folder; pywebview's dialog chooses files or a folder, not both."""
    import webview
    asked = []

    class Window:
        async def create_file_dialog(self, *args, **kwargs):
            asked.append(kwargs)
            return (str(tmp_path),)

    async def script(user, app):
        with monkeypatch.context() as patch:
            patch.setattr(nicegui.app.native, "main_window", Window())
            assert await app._pick(folder=True) == ([tmp_path], "auto")

    simulate(Graph(), script)
    assert asked[0]["dialog_type"] == webview.FileDialog.FOLDER


#*************#
#   preview   #
#*************#
@needs_challenge
def test_the_spectrum_box_stays_while_its_spectra_are_stepped_through():
    graph = Graph()
    data = graph.add("ismrm2016")
    graph.set(data, "n", 3)

    async def script(user, app):
        await app.run()
        with app.client:                                # as the block's eye does, on the page
            await app.act(data, "preview")
        await user.should_see(kind=ui.number)
        box, = user.find(kind=ui.number).elements      # rebuilt, it would lose its focus
        for index in (1, 2):
            box.value = index
            await asyncio.sleep(0.2)                   # a refresh lands on the loop's next turn
            assert user.find(kind=ui.number).elements == {box}
            assert app.index == index

    simulate(graph, script)


#**********#
#   slow   #
#**********#
def slow(monkeypatch, graph, id):
    """A block that takes a second, so the page can act while it runs."""
    code = graph.code
    monkeypatch.setattr(graph, "code", lambda k: (lambda line, imports, names: (
        "time.sleep(1)\n" + line if k == id else line, imports | {"import time"}, names))(
        *code(k)))


#*************#
#   spinner   #
#*************#
@needs_challenge
def test_a_run_under_way_spins_and_its_block_shows_it(monkeypatch):
    graph = Graph()
    data = graph.add("ismrm2016")
    graph.set(data, "n", 1)
    slow(monkeypatch, graph, data)

    async def script(user, app):
        assert not app.spinner.visible
        task = asyncio.create_task(app.run())
        await until(lambda: app.spinner.visible)
        assert [n["data"]["status"] for n in drawn(user)[0]] == ["running"]
        await task
        assert not app.spinner.visible
        assert [n["data"]["status"] for n in drawn(user)[0]] == ["done"]

    simulate(graph, script)


#*************#
#   mid-run   #
#*************#
@needs_challenge
def test_a_new_graph_during_a_run_is_its_own(monkeypatch):
    graph = Graph()
    data = graph.add("ismrm2016")
    graph.set(data, "n", 1)
    slow(monkeypatch, graph, data)

    async def script(user, app):
        old = asyncio.create_task(app.run())
        await until(lambda: app.running == data)
        app.new()
        new = app.add("ismrm2016", (0, 0), join=False)
        app.graph.set(new, "n", 1)
        assert [n["data"]["status"] for n in drawn(user)[0]] == ["idle"], "not the old run's"
        await app.run()                                # not refused for the old run
        assert app.graph.nodes[new].made is not None
        await old

    simulate(graph, script)


@needs_challenge
def test_files_chosen_during_a_run_are_read(monkeypatch, tmp_path):
    graph = Graph()
    data, read = graph.add("ismrm2016"), graph.add("files")
    graph.set(data, "n", 1)
    slow(monkeypatch, graph, data)
    challenge2016.load(1)[0].list()[0].save(str(tmp_path / "one.nii.gz"))

    async def script(user, app):
        async def pick(**_):
            return [tmp_path / "one.nii.gz"], "auto"
        app._pick = pick
        task = asyncio.create_task(app.run())
        await until(lambda: app.running == data)
        await app.choose_files(read)
        assert graph.nodes[read].made is not None
        await task

    simulate(graph, script)


def test_files_chosen_for_a_basis_set_are_set_and_not_read(tmp_path):
    """Its files are chosen as a read block's, opened on adding it; it runs when asked."""
    graph = Graph()
    chosen = [tmp_path / "NAA.txt", tmp_path / "Cr.txt"]

    async def script(user, app):
        asked, runs = [], []

        async def pick(title="Open data", formats=True, folder=False):
            asked.append(formats)
            return chosen, "auto"

        async def run(*args):
            runs.append(args)
        app._pick, app.run = pick, run
        with app.client:
            id = app.add("basis")
        await until(lambda: graph.nodes[id].values.get("files"))
        assert graph.nodes[id].values == {"files": [str(p) for p in chosen]}
        assert asked == [False] and runs == []
        await user.should_see("Choose files…")
        await user.should_see(str(chosen[0]))
        await user.should_not_see(marker="field-files")

    simulate(graph, script)


def test_a_block_whose_files_are_chosen_is_not_wired_in_after_the_one_selected():
    """Spectra are no basis files: the files chosen are what it reads."""
    graph = Graph()
    data = graph.add("ismrm2016")

    async def script(user, app):
        async def pick(**_):
            return None
        app._pick = pick
        with app.client:
            app.select(data)
            app.add("basis")
        assert graph.edges == []

    simulate(graph, script)


def test_a_basis_set_is_wired_into_the_fit_added_after_it_by_its_basis():
    graph = Graph()
    basis, fit = graph.add("basis"), graph.add("lcm-fit")
    graph.ask(fit)                                     # its arguments read, as once drawn

    async def script(user, app):
        app._join(basis, fit)
        assert [(e.output, e.input) for e in graph.edges] == [("basis", "basis")]

    simulate(graph, script)


def test_files_wired_in_are_said_so_and_not_chosen():
    graph = Graph()
    convert, basis = graph.add("basis-convert"), graph.add("basis")
    graph.set(basis, "files", ["/old/NAA.txt"])
    graph.connect(convert, "files", basis, "files")

    async def script(user, app):
        with app.client:
            app.select(basis)
        await user.should_see("files ← by wire")
        await user.should_not_see("Choose files…")
        await user.should_not_see("/old/NAA.txt")

    simulate(graph, script)


def test_files_made_are_shown_whole_without_a_spectrum_to_step_through(tmp_path):
    graph = Graph()
    convert = graph.add("basis-convert")
    graph.nodes[convert].made = {"basis_convert": [tmp_path / "NAA.json", tmp_path / "Cr.json"]}

    async def script(user, app):
        with app.client:
            await app.act(convert, "preview")
        await user.should_see("NAA.json")
        await user.should_not_see(kind=ui.number)

    simulate(graph, script)



def test_a_basis_set_is_shown_as_its_spectra():
    import numpy as np
    from mrsuite.core.basis import Basis

    graph = Graph()
    basis = graph.add("basis")
    fids = np.exp(-np.arange(256)[None] / 30.0) * np.ones((2, 1))
    graph.nodes[basis].made = {"basis": Basis(["NAA", "Cr"], fids, 4000.0, 123.2)}

    async def script(user, app):
        with app.client:
            await app.act(basis, "preview")
        await user.should_see(kind=ui.plotly)
        await user.should_not_see(kind=ui.number)

    simulate(graph, script)

def test_a_graph_or_script_saved_where_it_cannot_be_is_said(tmp_path):
    graph = Graph()
    graph.add("ismrm2016")
    nowhere = str(tmp_path / "no" / "such" / "folder" / "g.json")

    async def script(user, app):
        async def click():                             # as the header's button does
            with app.client:
                await app.save()
        task = asyncio.create_task(click())
        await user.should_see(marker="answer")
        user.find(marker="answer").clear().type(nowhere)
        user.find("OK").click()
        await task                                      # said, not raised
        await user.should_see("FileNotFoundError")

    simulate(graph, script)


def test_a_failed_run_before_an_export_says_only_why():
    graph = Graph()
    graph.add("lcm-fit")                               # nothing wired into it

    async def script(user, app):
        with app.client:
            await app.export("n1")
        await user.should_see("wire something into data")
        await user.should_not_see("Nothing to write out")

    simulate(graph, script)


#****************#
#   unforeseen   #
#****************#
def test_an_error_no_handler_foresaw_is_said_on_the_page(monkeypatch):
    graph = Graph()
    data = graph.add("ismrm2016")

    def boom(*args):
        raise RuntimeError("something nobody foresaw")

    async def script(user, app):
        monkeypatch.setattr(app.graph, "move", boom)
        user.find(Canvas).trigger("moved", [{"id": data, "x": 1, "y": 2}])
        await user.should_see("RuntimeError: something nobody foresaw")

    simulate(graph, script)


#************#
#   picker   #
#************#
def test_a_file_chosen_in_the_pages_own_picker_is_taken(tmp_path):
    """Browser mode, without the system's dialog: the choice must reach the block before
    the picker closes, which would otherwise say nothing was chosen."""
    from mrsuite.gui.files import Picker

    (tmp_path / "set.BASIS").write_text("")
    graph = Graph()

    async def script(user, app):
        app.folder = tmp_path
        async def pick():                              # as the folder button does
            with app.client:
                return await app._pick("Choose a file", formats=False)
        task = asyncio.create_task(pick())
        await until(lambda: any(isinstance(e, Picker) for e in app.client.elements.values()))
        picker = next(e for e in app.client.elements.values() if isinstance(e, Picker))
        with app.client:
            await picker.choose([tmp_path / "set.BASIS"])
        assert await task == ([tmp_path / "set.BASIS"], "auto")

    simulate(graph, script)


#*************#
#   buttons   #
#*************#
def test_a_file_argument_has_a_button_to_choose_it_and_a_run_waits_its_turn(monkeypatch):
    graph = Graph()
    data = graph.add("ismrm2016")
    graph.set(data, "n", 1)
    fit = graph.add("lcm-fit")
    graph.choose(fit, 1)                               # LCModel: basis and control are files
    slow(monkeypatch, graph, data)

    async def script(user, app):
        app.select(fit)
        await user.should_see(marker="path-basis")
        await user.should_see(marker="path-control")
        await user.should_not_see(marker="path-ppmlim")
        run = asyncio.create_task(app.run(data))
        await until(lambda: app.running == data)
        with app.client:
            await app.run()
        await user.should_see("Already running; stop it first")
        await run

    simulate(graph, script)


def test_an_argument_with_a_default_goes_on_the_block_by_its_button():
    graph = Graph()
    fit = graph.add("lcm-fit")
    graph.ask(fit)

    async def script(user, app):
        inputs = lambda: [p["name"] for p in app.canvas._props["nodes"][0]["data"]["inputs"]]
        colour = lambda: [e.props["color"] for e in user.find(marker="show-water").elements]
        app.select(fit)
        await user.should_see(marker="show-water")                # its arguments read
        await user.should_not_see(marker="show-basis")            # needed: always on it
        assert "water" not in inputs()
        user.find(marker="show-water").click()
        await until(lambda: colour() == ["primary"])              # the form redrawn
        assert "water" in inputs()
        user.find(marker="show-water").click()
        await until(lambda: colour() == ["grey"])
        assert "water" not in inputs()

    simulate(graph, script)


def test_a_search_puts_blocks_named_so_first():
    async def script(user, app):
        assert app._candidates("phas", None)[0][0] == "phasing"

    simulate(Graph(), script)
