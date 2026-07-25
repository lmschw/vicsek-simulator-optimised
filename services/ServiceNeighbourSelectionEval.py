import argparse
import os
import re
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import evaluators.EvaluatorMultiComp as EvaluatorMultiComp
import services.ServiceGeneral as ServiceGeneral
from enums.EnumMetrics import Metrics

"""
Shared evaluation/plotting helpers for the neighbour_selection_data*.py family of experiments.
Used by neighbour_selection_data_plots.py and its variants (neighbour_selection_data_no_neighbour_switch_plots.py,
neighbour_selection_data_random_subset_plots.py) so the loading, aggregation and plotting logic - and any
future bugfixes to it - only needs to exist in one place.
"""

COLOURS = EvaluatorMultiComp.COLOURS
CLUSTER_THRESHOLD = 0.01


# ------------------------------------------------------------------ run-file discovery ------

def buildRunIndex(baseDataLocation):
    """
    Scans the data directory once and returns {basePathPrefix: sorted [run indices]}, so that
    looking up which values of i exist for a given base path is an O(1) dict lookup instead of
    a fresh directory scan per parameter combination.
    """
    index = {}
    for name in os.listdir(baseDataLocation):
        if not name.endswith(".csv"):
            continue
        stem = name[:-4]
        if stem.endswith("_modelParams") or stem.endswith("_globalOrder"):
            continue
        match = re.match(r"^(.*)_(\d+)$", stem)
        if not match:
            continue
        prefix, i = match.group(1), int(match.group(2))
        index.setdefault(prefix, []).append(i)
    for key in index:
        index[key].sort()
    return index


def getIndices(runIndex, basePath):
    return runIndex.get(os.path.basename(basePath), [])


def toContiguousRange(indices):
    """
    EvaluatorMultiAvgComp expects a contiguous (start, stop) run range. Run indices are expected
    to be contiguous in practice (jobs are generated and executed in contiguous i-blocks), so this
    just derives that range and warns if that assumption doesn't hold for the runs found on disk.
    """
    if not indices:
        return None
    if indices != list(range(indices[0], indices[-1] + 1)):
        ServiceGeneral.logWithTime(f"WARNING: run indices are not contiguous ({indices}) - evaluating the full span anyway")
    return (indices[0], indices[-1] + 1)


# ------------------------------------------------------------------ plotting helpers ---------

def saveFigure(fig, outputPathBase):
    os.makedirs(os.path.dirname(outputPathBase), exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(f"{outputPathBase}.{ext}")


def plotSeries(series, xLabel, yLabel, title, outputPathBase, ylim=None, backgroundSpan=None):
    """
    series: list of {"label": str, "t": array, "mean": array, "std": array}
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, s in enumerate(series):
        colour = COLOURS[i % len(COLOURS)]
        ax.plot(s["t"], s["mean"], label=s["label"], color=colour, linewidth=1.5)
        ax.fill_between(s["t"], s["mean"] - s["std"], s["mean"] + s["std"], color=colour, alpha=0.2)
    if backgroundSpan is not None:
        ax.axvspan(backgroundSpan[0], backgroundSpan[1], color="green", alpha=0.15, label="event")
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(left=0)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(fontsize=7)
    fig.tight_layout()
    saveFigure(fig, outputPathBase)
    plt.close(fig)


# ------------------------------------------------------------------ data evaluation ----------

def loadGlobalOrderSeries(basePath, indices):
    runs = []
    tvals = None
    for i in indices:
        path = f"{basePath}_{i}_globalOrder.csv"
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, usecols=["t", "globalOrder"]).sort_values("t")
        if tvals is None:
            tvals = df["t"].to_numpy()
        elif len(df) != len(tvals):
            ServiceGeneral.logWithTime(f"WARNING: skipping incomplete run {path}")
            continue
        runs.append(df["globalOrder"].to_numpy())
    if not runs:
        return None
    arr = np.array(runs)
    return tvals, arr.mean(axis=0), arr.std(axis=0)


def evaluateMetricSeries(basePath, indices, metric, evalInterval, switchType=None, switchTypeOptions=None):
    runRange = toContiguousRange(indices)
    if runRange is None:
        return None
    switchTypeValues = [True] if switchType is not None else None
    evaluator = EvaluatorMultiComp.EvaluatorMultiAvgComp(
        metric=metric,
        basePaths=[basePath],
        runRange=runRange,
        from_csv=True,
        evaluationTimestepInterval=evalInterval,
        threshold=CLUSTER_THRESHOLD,
        switchTypeValues=switchTypeValues,
        switchType=switchType,
        switchTypeOptions=switchTypeOptions,
        use_median=False,
    )
    dd, varianceData = evaluator.evaluate()
    if not dd:
        return None
    times = np.array(sorted(dd.keys()))
    means = np.array([dd[t][0] for t in times])
    bounds = np.array(varianceData[0])
    stds = (bounds[:, 1] - bounds[:, 0]) / 2
    return times, means, stds


# ------------------------------------------------------------------ per-combination plots ----

def processCombo(runIndex, outputRoot, comboType, basePathOrdered, basePathRandom, paramSuffix, evalInterval,
                  switchType=None, switchTypeOptions=None, valueLabels=None, eventShading=None):
    """
    Produces (and saves as png/svg/pdf) up to three plots for one parameter combination: order over
    time (from the precomputed globalOrder.csv files), switch value percentage over time (only if
    switchType is given) and number of clusters over time. All three show mean +/- standard deviation
    across every run sharing the same base path (i.e. differing only in run index i).

    Returns True if at least one plot was produced (i.e. any data was found for this combination).
    """
    indicesOrdered = getIndices(runIndex, basePathOrdered)
    indicesRandom = getIndices(runIndex, basePathRandom)
    if not indicesOrdered and not indicesRandom:
        return False

    runs = [("ordered start", basePathOrdered, indicesOrdered),
            ("random start", basePathRandom, indicesRandom)]
    producedAny = False

    # 1) order over time, straight from the precomputed globalOrder.csv files
    seriesOrder = []
    for label, basePath, indices in runs:
        if not indices:
            continue
        result = loadGlobalOrderSeries(basePath, indices)
        if result is None:
            continue
        t, mean, std = result
        seriesOrder.append({"label": label, "t": t, "mean": mean, "std": std})
    if seriesOrder:
        plotSeries(seriesOrder, "timestep", "order",
                   f"order over time\n{comboType}: {paramSuffix}",
                   f"{outputRoot}/{comboType}/order_{paramSuffix}",
                   ylim=(0, 1.1), backgroundSpan=eventShading)
        producedAny = True

    # 2) switch value percentage over time (only for combinations that actually switch)
    if switchType is not None:
        seriesSwitch = []
        for label, basePath, indices in runs:
            if not indices:
                continue
            result = evaluateMetricSeries(basePath, indices, Metrics.ORDER_VALUE_PERCENTAGE, evalInterval,
                                           switchType=switchType, switchTypeOptions=switchTypeOptions)
            if result is None:
                continue
            t, mean, std = result
            seriesSwitch.append({"label": f"{label} - {valueLabels[0]}", "t": t, "mean": mean, "std": std})
            seriesSwitch.append({"label": f"{label} - {valueLabels[1]}", "t": t, "mean": 100 - mean, "std": std})
        if seriesSwitch:
            plotSeries(seriesSwitch, "timestep", "% of swarm",
                       f"switch value percentage over time\n{comboType}: {paramSuffix}",
                       f"{outputRoot}/{comboType}/switch_percentage_{paramSuffix}",
                       ylim=(0, 100.1), backgroundSpan=eventShading)
            producedAny = True

    # 3) number of (spatial + orientation) clusters over time
    seriesCluster = []
    for label, basePath, indices in runs:
        if not indices:
            continue
        result = evaluateMetricSeries(basePath, indices, Metrics.CLUSTER_NUMBER_WITH_RADIUS, evalInterval)
        if result is None:
            continue
        t, mean, std = result
        seriesCluster.append({"label": label, "t": t, "mean": mean, "std": std})
    if seriesCluster:
        plotSeries(seriesCluster, "timestep", "number of clusters",
                   f"number of clusters over time\n{comboType}: {paramSuffix}",
                   f"{outputRoot}/{comboType}/cluster_count_{paramSuffix}",
                   backgroundSpan=eventShading)
        producedAny = True

    return producedAny


# ------------------------------------------------------------------ CLI driver ---------------

def buildArgParser(sectionNames):
    parser = argparse.ArgumentParser(description="Evaluate and plot neighbour_selection_data*.py output.")
    parser.add_argument("--sections", nargs="+", choices=sectionNames, default=sectionNames,
                         help="which combination sections to evaluate (default: all)")
    parser.add_argument("--limit", type=int, default=None,
                         help="only process the first N combinations with data per section (for a quick test run)")
    return parser


def runSections(baseDataLocation, outputRoot, sectionGenerators, args):
    """
    sectionGenerators: {sectionName: generator function yielding dicts of processCombo kwargs
    (comboType, basePathOrdered, basePathRandom, paramSuffix, evalInterval, switchType,
    switchTypeOptions, valueLabels, eventShading)}
    """
    ServiceGeneral.logWithTime(f"indexing run files in {baseDataLocation}")
    runIndex = buildRunIndex(baseDataLocation)
    ServiceGeneral.logWithTime(f"found data for {len(runIndex)} individual base paths")

    for sectionName in args.sections:
        ServiceGeneral.logWithTime(f"--- section {sectionName} ---")
        sectionStart = time.time()
        processed = 0
        considered = 0
        for combo in sectionGenerators[sectionName]():
            considered += 1
            try:
                comboStart = time.time()
                didPlot = processCombo(
                    runIndex,
                    outputRoot,
                    combo["comboType"],
                    combo["basePathOrdered"],
                    combo["basePathRandom"],
                    combo["paramSuffix"],
                    combo["evalInterval"],
                    switchType=combo.get("switchType"),
                    switchTypeOptions=combo.get("switchTypeOptions"),
                    valueLabels=combo.get("valueLabels"),
                    eventShading=combo.get("eventShading"),
                )
                if didPlot:
                    processed += 1
                    ServiceGeneral.logWithTime(f"  {combo['paramSuffix']} ({ServiceGeneral.formatTime(time.time() - comboStart)})")
            except Exception as e:
                ServiceGeneral.logWithTime(f"  ERROR on {combo['paramSuffix']}: {e}")
            if args.limit is not None and processed >= args.limit:
                break
        ServiceGeneral.logWithTime(f"--- section {sectionName} done: {processed}/{considered} combinations had data, "
                                    f"{ServiceGeneral.formatTime(time.time() - sectionStart)} ---")

