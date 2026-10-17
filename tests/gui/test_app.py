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
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("nicegui")

import nicegui                                                          # noqa: E402
from nicegui import ui                                                  # noqa: E402
from nicegui.testing.user_simulation import user_simulation            # noqa: E402

from mrsuite import collect                                             # noqa: E402
from mrsuite.datasets import challenge2016                              # noqa: E402
from mrsuite.gui import blocks, catalog                                 # noqa: E402
from mrsuite.gui.app import App, _found                                 # noqa: E402
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


def test_a_load_block_added_asks_for_files_and_folders_at_once_and_loads_them(study):
    """Its window opens on the page's own picker, a folder and a file chosen together;
    the folder is searched and every file named; Load loads them."""
    graph = Graph()

    async def script(user, app):
        asked, runs = [], []

        async def pick(title="Open data", both=False):
            asked.append(both)
            return [study / "sub-02", study / "extra/press_mm_PFL.nii.gz"]

        async def run(*args):
            runs.append(args)
        app._pick, app.run = pick, run
        with app.client:
            id = app.add("load")
        await user.should_see(marker="collect-table")
        table = user.find(marker="collect-table").elements.pop()
        await until(lambda: len(table.rows) == 3)
        assert [(r["file"], r["name"]) for r in table.rows] == [
            ("press_mm_PFL.nii.gz", "mm"), ("sub-02_acq-press_svs.nii.gz", "data"),
            ("sub-02_acq-press_mrsref.nii.gz", "water")]
        user.find(marker="collect-load").click()
        await until(lambda: runs)
        assert asked == [True] and runs == [(id,)]
        assert [e[1] for e in graph.nodes[id].values["entries"]] == ["mm", "data", "water"]
        await user.should_see("Add or name files…")
        await user.should_see("water: 1 file")

    simulate(graph, script)


def test_the_load_window_names_many_files_at_once_in_order(study):
    """Files added from elsewhere, rows chosen given a name and a group together, others
    removed; a file nothing reads waits for its format; what is left is the block's own
    list, kept in order by group and name, and loads."""
    graph = Graph()
    load = graph.add("load")
    graph.set(load, "entries", collect.as_tuples(collect.discover(study / "sub-02")))
    (study / "extra" / "notes.txt").write_text("")
    elsewhere = [study / "extra/press_metab_PFL.nii.gz", study / "extra/press_mm_PFL.nii.gz",
                 study / "extra/notes.txt"]

    async def script(user, app):
        runs = []

        async def run(*args):
            runs.append(args)

        async def pick(title="Open data", both=False):
            return elsewhere
        app.run, app._pick = run, pick

        async def opened():
            with app.client:
                await app.collect(load)
        task = asyncio.create_task(opened())
        await user.should_see(marker="collect-table")
        table = user.find(marker="collect-table").elements.pop()
        assert [r["file"] for r in table.rows] == ["sub-02_acq-press_svs.nii.gz",
                                                   "sub-02_acq-press_mrsref.nii.gz"]
        user.find(marker="collect-add").click()
        await until(lambda: len(table.rows) == 5)
        assert [(r["file"], r["name"], r["format"]) for r in table.rows][:3] == [
            ("notes.txt", "data", ""), ("press_metab_PFL.nii.gz", "data", "nifti"),
            ("press_mm_PFL.nii.gz", "mm", "nifti")]
        user.find(marker="collect-load").click()
        await user.should_see("Give a format to notes.txt")
        table.selected = [r for r in table.rows if r["file"] == "press_metab_PFL.nii.gz"]
        user.find(marker="collect-name").elements.pop().set_value("water")
        user.find(marker="collect-group").elements.pop().set_value("water-run")
        user.find(marker="collect-apply").click()
        table.selected = [r for r in table.rows if r["file"] in ("press_mm_PFL.nii.gz",
                                                                   "notes.txt")]
        user.find(marker="collect-remove").click()
        user.find(marker="collect-load").click()
        await task

        assert graph.nodes[load].values["entries"] == [
            (str(study / "sub-02/mrs/sub-02_acq-press_svs.nii.gz"), "data", "sub-02_acq-press",
             "nifti"),
            (str(study / "sub-02/mrs/sub-02_acq-press_mrsref.nii.gz"), "water",
             "sub-02_acq-press", "nifti"),
            (str(elsewhere[0]), "water", "water-run", "nifti")]
        assert runs == [(load,)]
        assert "collect.load([\n" in graph.script()

    simulate(graph, script)


#*****************#
#   basis merge   #
#*****************#
def test_functions_kept_are_ticked_one_per_name():
    """Each input's functions, a name both have kept from where it was ticked last; all
    of them, one per name, is no choice at all."""
    from mrsuite.core.basis import Basis

    graph = Graph()
    basis, other, merge = graph.add("basis"), graph.add("basis"), graph.add("basis-add")
    graph.connect(basis, "basis", merge, "basis")
    graph.connect(other, "basis", merge, "more")
    fid = np.ones((1, 8), complex)
    graph.nodes[basis].made = {"basis": Basis(["NAA", "Cr"], np.vstack([fid, fid]), 1.0, 1.0)}
    graph.nodes[other].made = {"basis_2": Basis(["Cr"], fid, 1.0, 1.0)}

    async def script(user, app):
        async def chosen(answer):
            async def kept():
                with app.client:
                    await app.functions_kept(merge)
            task = asyncio.create_task(kept())
            await user.should_see(marker="keep-done")
            await answer()
            await task

        async def other_cr():
            assert user.find(marker="keep-basis-Cr").elements.pop().value is True
            user.find(marker="keep-basis_2-Cr").elements.pop().set_value(True)
            assert user.find(marker="keep-basis-Cr").elements.pop().value is False
            user.find(marker="keep-done").click()
        await chosen(other_cr)
        assert graph.nodes[merge].values["keep"] == {"NAA": "basis", "Cr": "basis_2"}

        async def own_cr():
            user.find(marker="keep-basis-Cr").elements.pop().set_value(True)
            user.find(marker="keep-done").click()
        await chosen(own_cr)
        assert graph.nodes[merge].values["keep"] == {"NAA": "basis", "Cr": "basis"}

    simulate(graph, script)


def test_the_load_window_outlives_a_run_ending_while_it_is_open(study, monkeypatch):
    """Opened from the inspector, which a run's end redraws: the window stays."""
    graph = Graph()
    data, load = graph.add("ismrm2016"), graph.add("load")
    graph.set(data, "n", 1)
    graph.set(load, "entries", collect.as_tuples(collect.discover(study / "sub-02")))
    slow(monkeypatch, graph, data)

    async def script(user, app):
        with app.client:
            app.select(load)
        await user.should_see(marker="collect")
        running = asyncio.create_task(app.run(data))
        await until(lambda: app.running == data)
        user.find(marker="collect").click()              # the inspector's button
        await user.should_see(marker="collect-table")
        await running
        await asyncio.sleep(0.5)
        await user.should_see(marker="collect-table")
        user.find(marker="collect-load").click()

    simulate(graph, script)


def test_a_folder_and_a_file_in_it_chosen_together_are_one_row_each(study):
    graph = Graph()
    load = graph.add("load")
    chosen = [study / "sub-02", study / "sub-02/mrs/sub-02_acq-press_svs.nii.gz"]

    async def script(user, app):
        async def pick(title="Open data", both=False):
            return chosen
        app._pick = pick

        async def opened():
            with app.client:
                await app.collect(load)
        task = asyncio.create_task(opened())
        await user.should_see(marker="collect-table")
        table = user.find(marker="collect-table").elements.pop()
        await until(lambda: len(table.rows) >= 2)
        await asyncio.sleep(0.3)
        user.find(marker="collect-load").click()
        await task
        assert len(table.rows) == 2
        assert len(_found(chosen)) == 2, "dropped, too"

    simulate(graph, script)


def test_functions_kept_are_those_still_wired_in_and_none_ticked_stays_so():
    """A kept function whose input went is let go; keeping nothing reopens as nothing."""
    from mrsuite.core.basis import Basis

    graph = Graph()
    basis, other, merge, third = (graph.add(k) for k in ("basis", "basis", "basis-add",
                                                          "basis"))
    graph.connect(basis, "basis", merge, "basis")
    graph.connect(other, "basis", merge, "more")
    fid = np.ones((1, 8), complex)
    graph.nodes[basis].made = {"basis": Basis(["NAA", "Cr"], np.vstack([fid, fid]), 1.0, 1.0)}
    graph.nodes[other].made = {"basis_2": Basis(["GABA"], fid, 1.0, 1.0)}
    graph.nodes[third].made = {"basis_3": Basis(["Lac"], fid, 1.0, 1.0)}
    graph.set(merge, "keep", {"NAA": "basis", "GABA": "basis_2"})

    async def script(user, app):
        app.removed("node", [other])
        with app.client:
            app.connect(third, "basis", merge, "more")

        async def kept(answer):
            async def opened():
                with app.client:
                    await app.functions_kept(merge)
            task = asyncio.create_task(opened())
            await user.should_see(marker="keep-done")
            answer()
            user.find(marker="keep-done").click()
            await task

        await kept(lambda: user.find(marker="keep-basis_3-Lac").elements.pop().set_value(True))
        assert graph.nodes[merge].values["keep"] == {"NAA": "basis", "Lac": "basis_3"}
        graph.set(merge, "keep", {})
        ticked = []
        await kept(lambda: ticked.extend(user.find(marker=m).elements.pop().value
                                         for m in ("keep-basis-NAA", "keep-basis_3-Lac")))
        assert ticked == [False, False] and graph.nodes[merge].values["keep"] == {}

    simulate(graph, script)


def test_a_basis_set_named_basis_wired_into_more_is_a_column_of_its_own():
    from mrsuite.core.basis import Basis

    graph = Graph()
    first, second, merge = graph.add("basis"), graph.add("basis"), graph.add("basis-add")
    graph.connect(second, "basis", merge, "basis")
    graph.connect(first, "basis", merge, "more")
    fid = np.ones((1, 8), complex)
    graph.nodes[first].made = {"basis": Basis(["Lac"], fid, 1.0, 1.0)}
    graph.nodes[second].made = {"basis_2": Basis(["NAA"], fid, 1.0, 1.0)}

    async def script(user, app):
        async def opened():
            with app.client:
                await app.functions_kept(merge)
        task = asyncio.create_task(opened())
        await user.should_see(marker="keep-done")
        await user.should_see(marker="keep-basis-NAA")
        await user.should_see(marker="keep-basis_basis-Lac")
        user.find(marker="keep-done").click()
        await task

    simulate(graph, script)


def test_a_block_gone_while_its_functions_are_made_is_let_go(study, monkeypatch):
    from mrsuite.core.basis import Basis

    graph = Graph()
    basis, load, merge = graph.add("basis"), graph.add("load"), graph.add("basis-add")
    graph.set(load, "entries", [(str(study / "extra/press_mm_PFL.nii.gz"), "mm", "x", "nifti")])
    graph.connect(basis, "basis", merge, "basis")
    graph.connect(load, "mm", merge, "more")
    graph.nodes[basis].made = {"basis": Basis(["NAA"], np.ones((1, 8), complex), 1.0, 1.0)}
    slow(monkeypatch, graph, load)

    async def script(user, app):
        async def opened():
            with app.client:
                await app.functions_kept(merge)
        task = asyncio.create_task(opened())
        await until(lambda: app.running == load)
        app.removed("node", [load])
        await task                                       # no KeyError

    simulate(graph, script)


#**************#
#   analyses   #
#**************#
def test_a_score_shows_its_analysis_and_its_own_arguments():
    graph = Graph()
    score = graph.add("method-comparison")

    async def script(user, app):
        with app.client:
            app.select(score)
        await user.should_see("Analysis")
        await user.should_see("bland_altman(reference")            # its help
        await user.should_not_see(marker="field-reference_error_variance")
        with app.client:
            await app._choose(score, 2)                   # deming
        await user.should_see(marker="field-reference_error_variance")

    simulate(graph, script)


def test_a_figure_made_is_shown_as_a_picture():
    from matplotlib.figure import Figure

    graph = Graph()
    radar = graph.add("radar")
    graph.nodes[radar].made = {"radar": Figure().add_subplot()}

    async def script(user, app):
        with app.client:
            await app.act(radar, "preview")
        await user.should_see(marker="figure")

    simulate(graph, script)


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


#*********#
#   eye   #
#*********#
@needs_challenge
def test_the_eye_opens_the_preview_and_closes_it_again():
    graph = Graph()
    data = graph.add("ismrm2016")
    graph.set(data, "n", 1)

    async def script(user, app):
        await app.run()
        with app.client:
            await app.act(data, "preview")
            assert app.previewed == data
            await app.act(data, "preview")
            assert app.previewed is None
            await user.should_see("Press the eye on a block")
            await app.act(data, "preview")
            assert app.previewed == data

    simulate(graph, script)


#*************#
#   spread   #
#*************#
def test_a_spectrum_with_coils_offers_to_draw_them_apart_and_stacked(tmp_path):
    from nifti_mrs.create_nmrs import gen_nifti_mrs
    from nifti_mrs_plus import NIfTI_MRS_Plus

    t = np.arange(256) * 2.5e-4
    fid = np.exp(2j * np.pi * (2.0 - 4.65) * 123.2 * t - t / 0.05)
    fids = np.stack([fid, 2 * fid], -1).reshape(1, 1, 1, -1, 2)
    path, = NIfTI_MRS_Plus([gen_nifti_mrs(fids, 2.5e-4, 123.2, dim_tags=["DIM_COIL", None, None])]) \
        .save_nifti(str(tmp_path), prefix="coils")
    graph = Graph()
    data = graph.add("load")
    graph.set(data, "entries", [(path, "data", "coils", "nifti")])

    async def script(user, app):
        await app.run()
        with app.client:
            await app.act(data, "preview")
        await user.should_see(marker="spread")
        app._show(stack=1.0)
        figure = app._figure(app.graph.output(data, "data").list()[0])
        assert [d["name"] for d in figure["data"]] == ["coil 0", "coil 1"]
        assert max(figure["data"][1]["y"]) > 2 * max(figure["data"][0]["y"]), "stacked above"

    simulate(graph, script)


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
            assert await app._pick() == [tmp_path / "a.nii.gz"]
            assert await app._pick(mode="save", name="g.json") == [tmp_path / "a.nii.gz"]
            assert await app._pick(mode="folder") == [tmp_path / "a.nii.gz"]

    simulate(Graph(), script)
    kinds = [pickle.loads(a)[0][0] for a in asked]
    assert [int(k) for k in kinds] == [10, 30, 20], "open, save, folder dialogs"
    assert pickle.loads(asked[1])[1]["save_filename"] == "g.json"


def test_files_and_folders_together_come_from_the_pages_own_picker_in_the_window_too(
        monkeypatch, tmp_path):
    """pywebview's dialog chooses files or a folder, not both."""
    from mrsuite.gui.files import Picker
    asked = []

    class Window:
        async def create_file_dialog(self, *args, **kwargs):
            asked.append(kwargs)

    async def script(user, app):
        with monkeypatch.context() as patch:
            patch.setattr(nicegui.app.native, "main_window", Window())
            app.folder = tmp_path

            async def pick():
                with app.client:
                    return await app._pick(both=True)
            task = asyncio.create_task(pick())
            await until(lambda: any(isinstance(e, Picker) for e in app.client.elements.values()))
            picker = next(e for e in app.client.elements.values() if isinstance(e, Picker))
            with app.client:
                await picker.choose([tmp_path, tmp_path / "a.nii.gz"])
            assert await task == [tmp_path, tmp_path / "a.nii.gz"]

    simulate(Graph(), script)
    assert asked == []


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
def test_files_loaded_during_a_run_are_read(monkeypatch, tmp_path):
    graph = Graph()
    data, read = graph.add("ismrm2016"), graph.add("load")
    graph.set(data, "n", 1)
    graph.set(read, "entries", [(str(tmp_path / "one.nii.gz"), "data", "one", "nifti")])
    slow(monkeypatch, graph, data)
    challenge2016.load(1)[0].list()[0].save(str(tmp_path / "one.nii.gz"))

    async def script(user, app):
        task = asyncio.create_task(app.run())
        await until(lambda: app.running == data)

        async def opened():
            with app.client:
                await app.collect(read)
        loading = asyncio.create_task(opened())
        await user.should_see(marker="collect-load")
        user.find(marker="collect-load").click()
        await loading
        assert graph.nodes[read].made is not None
        await task

    simulate(graph, script)


def test_files_chosen_for_a_basis_set_are_set_and_not_read(tmp_path):
    """Its files are chosen as a read block's, opened on adding it; it runs when asked."""
    graph = Graph()
    chosen = [tmp_path / "NAA.txt", tmp_path / "Cr.txt"]

    async def script(user, app):
        asked, runs = [], []

        async def pick(title="Open data", both=False):
            asked.append(both)
            return chosen

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
        async def pick(*_, **__):
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
        app.folder = tmp_path / "no" / "such" / "folder"       # gone by the time it saves
        async def click():                             # as the header's button does
            with app.client:
                await app.save()
        task = asyncio.create_task(click())
        await user.should_see(marker="picker-name")
        user.find(marker="picker-name").clear().type("g.json")
        user.find(marker="picker-save").click()
        await task                                      # said, not raised
        await user.should_see("FileNotFoundError")
        assert not Path(nowhere).exists()

    simulate(graph, script)


#*************#
#   browser   #
#*************#
def test_a_graph_is_saved_and_opened_through_the_pages_own_browser(tmp_path):
    """Save names the file in the picker's box; Open ticks it; the folder is remembered."""
    graph = Graph()
    graph.add("ismrm2016")

    async def script(user, app):
        app.folder = tmp_path

        async def save():
            with app.client:
                await app.save()
        task = asyncio.create_task(save())
        await user.should_see(marker="picker-name")
        assert user.find(marker="picker-name").elements.pop().value == "graph.mrsuite.json"
        user.find(marker="picker-name").clear().type("mine.json")
        user.find(marker="picker-save").click()
        await task
        assert (tmp_path / "mine.json").exists() and app.folder == tmp_path

        app.new()
        assert not app.graph.nodes
        async def open_():
            with app.client:
                await app.open()
        task = asyncio.create_task(open_())
        await user.should_see(marker="file-mine.json")
        user.find(marker="file-mine.json").elements.pop().set_value(True)
        user.find(marker="picker-open").click()
        await task
        assert [n.block for n in app.graph.nodes.values()] == ["ismrm2016"]

    simulate(graph, script)


def test_the_export_folder_and_the_script_path_come_from_a_browser(tmp_path):
    import pandas as pd

    graph = Graph()
    data = graph.add("mosae")
    graph.nodes[data].made = {"mosae": pd.DataFrame({"score": [1.0]})}
    (tmp_path / "out").mkdir()

    async def script(user, app):
        app.folder = tmp_path
        async def export():
            with app.client:
                await app.export(data)
        task = asyncio.create_task(export())
        await user.should_see(marker="export-browse")
        user.find(marker="export-browse").click()
        await user.should_see(marker="open-out")
        user.find(marker="open-out").click()
        await user.should_see(marker="picker-folder")
        user.find(marker="picker-folder").click()
        await until(lambda: user.find(marker="export-folder").elements.pop().value
                    == str(tmp_path / "out"))
        user.find("Cancel").click()
        await task

        graph.remove(data)                             # unwired, it has no script
        graph.add("ismrm2016")

        async def script_():
            with app.client:
                await app.script()
        task = asyncio.create_task(script_())
        await user.should_see(marker="script-browse")
        user.find(marker="script-browse").click()
        await user.should_see(marker="picker-name")
        user.find(marker="picker-name").clear().type("made.py")
        user.find(marker="picker-save").click()
        await until(lambda: user.find(marker="script-path").elements.pop().value
                    == str(tmp_path / "made.py"))
        user.find(marker="script-save").click()
        await task
        assert (tmp_path / "made.py").read_text().startswith("#")

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
                return await app._pick("Choose a file")
        task = asyncio.create_task(pick())
        await until(lambda: any(isinstance(e, Picker) for e in app.client.elements.values()))
        picker = next(e for e in app.client.elements.values() if isinstance(e, Picker))
        with app.client:
            await picker.choose([tmp_path / "set.BASIS"])
        assert await task == [tmp_path / "set.BASIS"]

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


#*********#
#   zoo   #
#*********#
def test_a_model_block_offers_the_pinned_checkpoints_until_the_hub_is_listed(monkeypatch):
    """The dropdown holds the pinned checkpoints; once the Hub's listing is in, those of the
    block's class from there, and a name typed stays offered."""
    from mrsuite import models
    listed = {"MRSuite/mlp-supervised-7t-slaser": "mrsuite.models.mlp.MLP",
              "MRSuite/wand-ismrm2016": "mrsuite.models.wand.WAND",
              "MRSuite/wand-fmrs-in-pain": "mrsuite.models.wand.WAND"}
    monkeypatch.setattr(models, "zoo", lambda: listed)
    graph = Graph()
    wand = graph.add("wand")

    async def script(user, app):
        app.select(wand)
        await asyncio.sleep(0.1)                       # the page redrawn
        box, = user.find(marker="field-checkpoint").elements
        assert box.options == ["MRSuite/wand-ismrm2016"] == list(blocks.ZOO["wand"])
        await app._zoo()
        assert app.zoo["wand"] == ["MRSuite/wand-fmrs-in-pain", "MRSuite/wand-ismrm2016"]
        await asyncio.sleep(0.1)                       # the inspector redrawn
        box, = user.find(marker="field-checkpoint").elements
        assert box.options == ["MRSuite/wand-fmrs-in-pain", "MRSuite/wand-ismrm2016"]
        assert app.zoo["mlp"] == ["MRSuite/mlp-supervised-7t-slaser"]

        graph.set(wand, "checkpoint", "me/my-wand")
        app.inspector.refresh()
        await asyncio.sleep(0.1)
        box, = user.find(marker="field-checkpoint").elements
        assert box.value == "me/my-wand" and "me/my-wand" in box.options

    simulate(graph, script)


#*************#
#   folders   #
#*************#
def test_the_data_blocks_sit_in_folders_closed_until_opened_and_kept_so():
    async def script(user, app):
        for folder in ("Files", "Simulated datasets", "In vivo datasets", "Basis sets"):
            fold, = [e for e in user.find(folder).elements if isinstance(e, ui.expansion)]
            assert fold.value is False
        app._fold("data/In vivo datasets", True)
        app.palette.refresh()
        await asyncio.sleep(0.1)
        fold, = [e for e in user.find("In vivo datasets").elements if isinstance(e, ui.expansion)]
        assert fold.value is True
        await user.should_see(marker="block-openneuro-ds005371")
        assert "data/In vivo datasets" in app.opened and app.functions == {}, "no toolbox listed"

    simulate(Graph(), script)


#*************#
#   by tab   #
#*************#
def test_a_toolboxs_functions_are_listed_under_the_tabs_they_belong_to():
    async def script(user, app):
        app.functions = {b: {} for b in catalog.BRIDGES}
        app.functions["fsl_mrs"] = {"fit": "fit", "add": "other", "align": "preprocess"}
        app._fold("all", True)
        app._fold("fsl_mrs", True)
        app.palette.refresh()
        await asyncio.sleep(0.1)
        folds = {k: [e for e in user.find(marker=f"folder-fsl_mrs/{k}").elements][0]
                 for k in ("preprocess", "fit", "other")}
        assert [f.text for f in folds.values()] == ["Preprocess", "Fit", "Other"]
        assert not any(f.value for f in folds.values()), "folded until opened"
        await user.should_see(marker="block-fsl_mrs:align")
        app._fold("fsl_mrs/preprocess", True)
        app.palette.refresh()
        await asyncio.sleep(0.1)
        fold, = user.find(marker="folder-fsl_mrs/preprocess").elements
        assert fold.value is True
        await user.should_not_see(marker="folder-fsl_mrs/data")      # nothing of that tab

    simulate(Graph(), script)


#**************#
#   terminal   #
#**************#
def test_an_error_said_on_the_page_is_logged_in_full_for_the_terminal(caplog):
    """A block's failure, and any error a page says, goes to the log with its traceback."""
    graph = Graph()
    data = graph.add("ismrm2016")
    graph.set(data, "n", "two")                        # not a number: the load fails

    async def script(user, app):
        with app.client:
            await app.run()
        await until(lambda: any("Traceback" in r.exc_text for r in caplog.records if r.exc_text))
        failed = [r for r in caplog.records if r.levelname == "ERROR"]
        assert failed and "ismrm2016 failed" in failed[0].message and failed[0].exc_text

    with caplog.at_level("ERROR", logger="mrsuite.gui"):
        simulate(graph, script)


#**********#
#   tabs   #
#**********#
def test_new_opens_a_tab_and_the_first_keeps_its_graph():
    graph = Graph()
    graph.add("ismrm2016")

    async def script(user, app):
        user.find(marker="new").click()
        await until(lambda: not app.graph.nodes)
        assert [t.name for t in app.tabs] == ["Untitled 1", "Untitled 2"]
        assert app.tab is app.tabs[1] and drawn(user) == ([], [])

        user.find(marker="tab-1").click()
        await until(lambda: app.graph is graph)
        assert len(drawn(user)[0]) == 1 and app.tab.name == "Untitled 1"

    simulate(graph, script)


def test_a_tab_edited_since_its_save_shows_a_dot_and_asks_before_closing():
    async def script(user, app):
        assert not app.tab.dirty
        app.add("ismrm2016", (0, 0), join=False)
        await user.should_see("Untitled 1 •")

        async def close():
            with app.client:
                await app.close_tab(app.tab)
        task = asyncio.create_task(close())
        await user.should_see(marker="close-cancel")
        user.find(marker="close-cancel").click()
        await task
        assert len(app.tabs) == 1 and app.graph.nodes, "kept"

        task = asyncio.create_task(close())
        await user.should_see(marker="close-discard")
        user.find(marker="close-discard").click()
        await task
        assert len(app.tabs) == 1 and not app.graph.nodes, "closed; an empty tab is left"
        assert app.tab.name == "Untitled 2" and not app.tab.dirty

    simulate(Graph(), script)


def test_a_graph_opened_has_its_own_tab_named_after_its_file_and_saves_back_to_it(tmp_path):
    saved = Graph()
    saved.add("ismrm2016")
    saved.save(tmp_path / "mine.mrsuite.json")

    async def script(user, app):
        app.folder = tmp_path

        async def open_():
            with app.client:
                await app.open()
        task = asyncio.create_task(open_())
        await user.should_see(marker="file-mine.mrsuite.json")
        user.find(marker="file-mine.mrsuite.json").elements.pop().set_value(True)
        user.find(marker="picker-open").click()
        await task
        assert [t.name for t in app.tabs] == ["mine"], "the empty untouched tab made way"
        assert not app.tab.dirty and app.tab.path == tmp_path / "mine.mrsuite.json"

        app.add("averaging", (0, 0), join=False)
        await user.should_see("mine •")
        with app.client:
            await app.save()                           # to its file, nothing asked
        await until(lambda: not app.tab.dirty)
        assert len(Graph.load(tmp_path / "mine.mrsuite.json").nodes) == 2

        user.find(marker="new").click()
        await until(lambda: len(app.tabs) == 2)
        assert [t.name for t in app.tabs] == ["mine", "Untitled 2"]

    simulate(Graph(), script)


def test_new_opens_a_tab_even_over_an_untouched_empty_one():
    async def script(user, app):
        user.find(marker="new").click()
        await until(lambda: len(app.tabs) == 2)
        assert [t.name for t in app.tabs] == ["Untitled 1", "Untitled 2"]
        assert app.tab is app.tabs[1]

    simulate(Graph(), script)


@needs_challenge
def test_a_tabs_run_is_its_own_and_ends_when_its_tab_closes(monkeypatch):
    """A tab shown again refuses a second run while its first is under way, though another
    tab's started since; a tab closed stops its run after the block under way."""
    graph = Graph()
    data = graph.add("ismrm2016")
    graph.set(data, "n", 1)
    for _ in range(2):                                 # two seconds
        slow(monkeypatch, graph, data)

    async def script(user, app):
        first = asyncio.create_task(app.run())
        await until(lambda: app.running == data)
        app.new()
        other = app.graph
        one, two = other.add("ismrm2016"), other.add("ismrm2016")
        for k in (one, two):
            other.set(k, "n", 1)
        slow(monkeypatch, other, one)
        second = asyncio.create_task(app.run())
        await until(lambda: app.running == one)

        async def close():
            with app.client:
                await app.close_tab(app.tabs[1])
        closing = asyncio.create_task(close())
        await user.should_see(marker="close-discard")
        user.find(marker="close-discard").click()
        await closing
        assert app.graph is graph and app.busy and app.spinner.visible
        with app.client:
            await app.run()
        await user.should_see("Already running; stop it first")

        await second
        assert other.nodes[one].made is not None and other.nodes[two].made is None, "stopped"
        await first
        assert not app.runs and not app.spinner.visible

    simulate(graph, script)


@needs_challenge
def test_a_tab_whose_run_ended_while_hidden_shows_no_block_running(monkeypatch):
    graph = Graph()
    data = graph.add("ismrm2016")
    graph.set(data, "n", 1)
    slow(monkeypatch, graph, data)

    async def script(user, app):
        task = asyncio.create_task(app.run())
        await until(lambda: app.running == data)
        app.new()
        await task
        user.find(marker="tab-1").click()
        await until(lambda: app.graph is graph)
        assert app.running is None
        assert [n["data"]["status"] for n in drawn(user)[0]] == ["done"]

    simulate(graph, script)


def test_a_name_left_as_it_was_redraws_nothing(monkeypatch):
    graph = Graph()
    data = graph.add("ismrm2016")

    async def script(user, app):
        app.select(data)
        await asyncio.sleep(0.1)                       # the page redrawn
        changed = []
        monkeypatch.setattr(app, "_changed", lambda: changed.append(True))
        user.find(marker="name").trigger("blur")
        await asyncio.sleep(0.1)
        assert not changed
        user.find(marker="name").elements.pop().set_value("challenge")
        user.find(marker="name").trigger("blur")
        await until(lambda: changed)
        assert graph.nodes[data].name == "challenge"

    simulate(graph, script)
