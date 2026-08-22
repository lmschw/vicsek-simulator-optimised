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
# The orientation-similarity cutoff used by ServiceClusters.find_clusters_with_radius (in Euclidean
# distance between unit orientation vectors). 0.01 was too tight to ever merge a genuinely ordered
# swarm into one cluster: at 1% noise (the pipeline default), independent per-step noise alone gives
# neighbouring particles' instantaneous orientations a typical pairwise distance well above 0.01, so a
# fully-converged, order~0.998 swarm was still reported as ~10 separate clusters. Empirically (15
# reps, order~0.998, 1% noise), cluster counts stop decreasing anywhere past ~0.1 - beyond that point
# the only clusters left are genuine spatial fragments (particles that drifted outside the perception
# radius of the rest of the flock), not an orientation-threshold artifact. 0.1 is set at that plateau:
# large enough to absorb all the noise-driven scatter, not so large it starts merging particles that
# are only coincidentally similar in orientation.
CLUSTER_THRESHOLD = 0.1


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
    """
    Loads the precomputed *_globalOrder.csv files (one globalOrder value per saved timestep, written
    incrementally while a run is in progress) for the given run indices. A run still being generated
    - possibly by a simulation sweep running concurrently with this evaluation - will have fewer
    completed timesteps than a finished one, and its file may be mid-write when read (yielding a torn
    last row): both are expected, not something to warn about, so runs are truncated to whatever
    length of data they have in common rather than being dropped.
    """
    runs = []
    minLength = None
    for i in indices:
        path = f"{basePath}_{i}_globalOrder.csv"
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path, usecols=["t", "globalOrder"])
        except (pd.errors.ParserError, pd.errors.EmptyDataError):
            continue  # file is mid-write; its data will be picked up once this run has progressed further
        df = df.dropna(subset=["globalOrder"]).sort_values("t")
        if df.empty:
            continue
        runs.append(df)
        minLength = len(df) if minLength is None else min(minLength, len(df))
    if not runs:
        return None
    tvals = runs[0]["t"].to_numpy()[:minLength]
    arr = np.array([df["globalOrder"].to_numpy()[:minLength] for df in runs])
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
                  switchType=None, switchTypeOptions=None, valueLabels=None, eventShading=None,
                  minI=None, maxI=None, clusterMode="exclude"):
    """
    Produces (and saves as png/svg/pdf) up to three plots for one parameter combination: order over
    time (from the precomputed globalOrder.csv files), switch value percentage over time (only if
    switchType is given) and number of clusters over time. All three show mean +/- standard deviation
    across every run sharing the same base path (i.e. differing only in run index i, optionally
    restricted to the [minI, maxI] range).

    clusterMode controls which plots are produced, since the cluster plot is by far the slowest of
    the three (it requires loading full position/orientation data rather than the small precomputed
    files the other two use):
        - "exclude" (default): order + switch percentage, no cluster plot
        - "only": cluster plot only, skipping order and switch percentage
        - "include": all three plots

    Returns True if at least one plot was produced (i.e. any data was found for this combination).
    """
    indicesOrdered = [i for i in getIndices(runIndex, basePathOrdered) if (minI is None or i >= minI) and (maxI is None or i <= maxI)]
    indicesRandom = [i for i in getIndices(runIndex, basePathRandom) if (minI is None or i >= minI) and (maxI is None or i <= maxI)]
    if not indicesOrdered and not indicesRandom:
        return False

    runs = [("ordered start", basePathOrdered, indicesOrdered),
            ("random start", basePathRandom, indicesRandom)]
    producedAny = False

    if clusterMode != "only":
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

        # 2) switch value percentage over time (only for combinations that actually switch). Mirrors
        # the order plot exactly: one line per starting condition, no more - the two switch values
        # are complements of each other (100 - x), so plotting both per starting condition would
        # just be redundant clutter, not extra information.
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
                seriesSwitch.append({"label": label, "t": t, "mean": mean, "std": std})
            if seriesSwitch:
                plotSeries(seriesSwitch, "timestep", f"% of swarm using {valueLabels[0]}",
                           f"switch value percentage over time\n{comboType}: {paramSuffix}",
                           f"{outputRoot}/{comboType}/switch_percentage_{paramSuffix}",
                           ylim=(0, 100.1), backgroundSpan=eventShading)
                producedAny = True

    # 3) number of (spatial + orientation) clusters over time
    if clusterMode in ("only", "include"):
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
    parser.add_argument("--min-i", type=int, default=None,
                         help="only use runs with index i >= this value (default: no lower bound)")
    parser.add_argument("--max-i", type=int, default=None,
                         help="only use runs with index i <= this value (default: no upper bound)")
    parser.add_argument("--clusters", choices=["exclude", "only", "include"], default="exclude",
                         help="whether to produce the number-of-clusters plot, which is by far the slowest of the three: "
                              "\"exclude\" (default) produces order + switch percentage only, \"only\" produces just the "
                              "cluster plot, \"include\" produces all three")
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
                    minI=args.min_i,
                    maxI=args.max_i,
                    clusterMode=args.clusters,
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

