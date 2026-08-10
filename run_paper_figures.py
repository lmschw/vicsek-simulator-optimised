import argparse
import json
import os
import time

from enums.EnumMetrics import Metrics

import paper_figures_specs as specs
import services.ServiceGeneral as ServiceGeneral
import services.ServicePaperFigureRunner as Runner
import services.ServicePaperFigureGrid as Grid

"""
Orchestrates the whole paper-figure pipeline: for every figure defined in paper_figures_specs.py,
generates whatever simulation data it needs (skipping anything already generated, seeded and
reproducible - see ServicePaperFigureRunner), evaluates it (reusing
services/ServiceNeighbourSelectionEval.py via ServicePaperFigureRunner.evaluateCell), renders the
figure (pdf/svg/png via ServicePaperFigureGrid), and deletes the raw data once every figure that
needs it has been rendered (tracked via a reference count, since several figures can share the
exact same underlying combination - see paper_figures_specs.ComboRegistry).

Resumable: figures already recorded in the manifest are skipped; within a not-yet-finished figure,
individual simulation runs already on disk are skipped too (see ServicePaperFigureRunner.runBatch).
"""

MANIFEST_PATH_NAME = "_manifest.json"


def loadManifest(outputRoot):
    path = os.path.join(outputRoot, MANIFEST_PATH_NAME)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def saveManifest(outputRoot, manifest):
    os.makedirs(outputRoot, exist_ok=True)
    path = os.path.join(outputRoot, MANIFEST_PATH_NAME)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


def iterCells(figureSpec):
    """Returns [(key, cellSpec), ...] uniformly regardless of figure kind."""
    if figureSpec["kind"] == "grid":
        return list(figureSpec["cells"].items())
    return list(enumerate(figureSpec["cells"]))


def cellPaths(cellSpec):
    return [cellSpec["orderedPath"], cellSpec["randomPath"]]


def computePathToFigures(figures):
    """Which figures reference each path - a path is safe to delete once all of them are done."""
    pathToFigures = {}
    for name, spec in figures.items():
        paths = set()
        for _key, cell in iterCells(spec):
            paths.update(cellPaths(cell))
        for path in paths:
            pathToFigures.setdefault(path, []).append(name)
    return pathToFigures


def evalIntervalForPath(path, runSpecs):
    spec = runSpecs.get(path)
    if spec is None:
        return 1
    return 50 if spec["logInterval"] == 1 else 1


def invertResult(result):
    if result is None:
        return None
    t, mean, std = result
    return t, 100 - mean, std


def buildCellSeries(cellSpec, metric, numReps, evalInterval):
    switchType = cellSpec.get("switchType")
    switchTypeOptions = cellSpec.get("switchTypeOptions")
    results = Runner.evaluateCell(cellSpec["orderedPath"], cellSpec["randomPath"], numReps, metric, evalInterval,
                                   switchType=switchType, switchTypeOptions=switchTypeOptions)
    orderedResult, disorderedResult = results["ordered"], results["disordered"]
    if metric == Metrics.ORDER_VALUE_PERCENTAGE and cellSpec.get("invertPercentage"):
        orderedResult = invertResult(orderedResult)
        disorderedResult = invertResult(disorderedResult)

    if "colourOverride" in cellSpec:
        oc, dc = cellSpec["colourOverride"]
        ol, dl = cellSpec.get("linestyleOverride", ("-", "-"))
        suffix = cellSpec.get("seriesLabelSuffix", "")
        series = []
        if orderedResult is not None:
            t, mean, std = orderedResult
            series.append(Grid.seriesFor(t, mean, std, f"ordered start{suffix}", oc, ol))
        if disorderedResult is not None:
            t, mean, std = disorderedResult
            series.append(Grid.seriesFor(t, mean, std, f"disordered start{suffix}", dc, dl))
        return series

    return Grid.orderedDisorderedSeries(orderedResult, disorderedResult)


def renderFigure(name, figureSpec, numReps, runSpecs, outputRoot):
    metric = figureSpec["metric"]
    kind = figureSpec["kind"]
    outputPathBase = os.path.join(outputRoot, name)

    if kind == "grid":
        cellsOut = {}
        for (r, c), cellSpec in figureSpec["cells"].items():
            evalInterval = evalIntervalForPath(cellSpec["orderedPath"], runSpecs)
            cellsOut[(r, c)] = buildCellSeries(cellSpec, metric, numReps, evalInterval)
        legendEntries = figureSpec["legendEntries"] or Grid.STANDARD_LEGEND
        Grid.renderGrid(cellsOut, figureSpec["nRows"], figureSpec["nCols"], figureSpec["rowHeaders"],
                         figureSpec["colHeaders"], figureSpec["yLabel"], "timesteps", legendEntries,
                         outputPathBase, ylim=figureSpec.get("ylim"), backgroundSpan=figureSpec.get("backgroundSpan"),
                         cellTitles=figureSpec.get("cellTitles"))
    elif kind == "single":
        allSeries = []
        for cellSpec in figureSpec["cells"]:
            evalInterval = evalIntervalForPath(cellSpec["orderedPath"], runSpecs)
            allSeries.extend(buildCellSeries(cellSpec, metric, numReps, evalInterval))
        legendEntries = [(s["label"], s["colour"], s["linestyle"]) for s in allSeries]
        Grid.renderSinglePanel(allSeries, figureSpec["yLabel"], "timesteps", legendEntries, outputPathBase,
                                ylim=figureSpec.get("ylim"), backgroundSpan=figureSpec.get("backgroundSpan"))
    elif kind == "stack":
        panels = []
        for cellSpec in figureSpec["cells"]:
            evalInterval = evalIntervalForPath(cellSpec["orderedPath"], runSpecs)
            panels.append(buildCellSeries(cellSpec, metric, numReps, evalInterval))
        legendEntries = figureSpec["legendEntries"] or Grid.STANDARD_LEGEND
        Grid.renderStack(panels, figureSpec["yLabel"], "timesteps", legendEntries, outputPathBase,
                          ylim=figureSpec.get("ylim"), backgroundSpan=figureSpec.get("backgroundSpan"),
                          panelHeaders=figureSpec.get("panelHeaders"))
    else:
        raise ValueError(f"unknown figure kind {kind}")


def run(dataRoot, outputRoot, numReps, workers, figureFilter, listOnly):
    ServiceGeneral.logWithTime("building figure specs...")
    runSpecs, figures = specs.buildAll(dataRoot, numReps)
    pathToFigures = computePathToFigures(figures)
    manifest = loadManifest(outputRoot)
    seedLogPath = os.path.join(outputRoot, "seeds.csv")

    names = sorted(figures.keys()) if figureFilter is None else [n for n in figureFilter if n in figures]
    unknown = [] if figureFilter is None else [n for n in figureFilter if n not in figures]
    if unknown:
        ServiceGeneral.logWithTime(f"WARNING: unknown figure names ignored: {unknown}")

    if listOnly:
        for name in sorted(figures.keys()):
            status = "done" if manifest.get(name) == "done" else "pending"
            ServiceGeneral.logWithTime(f"{name}: {status}")
        return

    os.makedirs(dataRoot, exist_ok=True)
    os.makedirs(outputRoot, exist_ok=True)

    ServiceGeneral.logWithTime(f"{len(figures)} figures total ({len(runSpecs)} unique combinations, "
                                f"{len(runSpecs) * numReps} individual runs at {numReps} reps each)")

    for name in names:
        if manifest.get(name) == "done":
            ServiceGeneral.logWithTime(f"skipping {name} (already done)")
            continue
        figureSpec = figures[name]
        figureStart = time.time()
        ServiceGeneral.logWithTime(f"--- {name} ---")

        neededPaths = set()
        for _key, cell in iterCells(figureSpec):
            neededPaths.update(cellPaths(cell))
        neededSpecs = [runSpecs[p] for p in neededPaths if p in runSpecs]

        completed = Runner.runBatch(neededSpecs, seedLogPath, numReps, workers)
        ServiceGeneral.logWithTime(f"  ran {completed} simulations ({ServiceGeneral.formatTime(time.time() - figureStart)})")

        renderFigure(name, figureSpec, numReps, runSpecs, outputRoot)
        ServiceGeneral.logWithTime(f"  rendered {name} ({ServiceGeneral.formatTime(time.time() - figureStart)})")

        manifest[name] = "done"
        saveManifest(outputRoot, manifest)

        # A path is safe to delete once every figure that references it is done - re-checked against
        # the persisted manifest (not an in-memory counter) so this is correct regardless of whether
        # those figures were processed in this invocation or an earlier one.
        toClean = []
        for path in neededPaths:
            referencingFigures = pathToFigures.get(path, [])
            if all(manifest.get(fig) == "done" for fig in referencingFigures):
                toClean.append(runSpecs[path])
        if toClean:
            removed = Runner.cleanupBatch(toClean, numReps)
            ServiceGeneral.logWithTime(f"  cleaned up {len(toClean)} combinations ({removed} files)")

        ServiceGeneral.logWithTime(f"--- {name} done: {ServiceGeneral.formatTime(time.time() - figureStart)} ---")


def main():
    parser = argparse.ArgumentParser(description="Generate the paper figures end to end.")
    parser.add_argument("--data-root", default=os.path.expanduser("~/paper_figures_tmp"),
                         help="scratch directory for raw simulation data (deleted per-figure as it's used)")
    parser.add_argument("--output-root", default="plots/paper_figures", help="where the rendered figures/manifest/seed log go")
    parser.add_argument("--num-reps", type=int, default=20)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    parser.add_argument("--figures", nargs="+", default=None, help="only process these figure names (default: all)")
    parser.add_argument("--list", action="store_true", help="list all figure names and their manifest status, then exit")
    args = parser.parse_args()

    run(args.data_root, args.output_root, args.num_reps, args.workers, args.figures, args.list)


if __name__ == "__main__":
    main()
