import numpy as np

from enums.EnumNeighbourSelectionMechanism import NeighbourSelectionMechanism
from enums.EnumSwitchType import SwitchType
from enums.EnumEventEffect import EventEffect
from enums.EnumMetrics import Metrics
from enums.EnumDistributionType import DistributionType
from events.ExternalStimulusEvent import ExternalStimulusOrientationChangeEvent
from model.SwitchInformation import SwitchInformation
from model.SwitchSummary import SwitchSummary

import services.ServicePreparation as ServicePreparation

"""
Data-driven definitions of every paper figure: what grid of parameter combinations each one needs,
how the grid is labelled, and which metric it shows. Kept separate from the execution engine
(services/ServicePaperFigureRunner.py) and the rendering engine (services/ServicePaperFigureGrid.py)
so the figure-specific content here can be reviewed/corrected independently of that machinery.

buildAll(dataRoot, numReps, updateIfNoNeighbours=True) is the entry point: it returns (runSpecs, figureSpecs) where runSpecs is
{path: runSpec} for every simulation run needed across every figure, already deduplicated (the same underlying
combination is only generated once even if several figures use it, e.g. fig_4/switching_percentage/
switching_clustering, or fig_4's default point coinciding with a duration=1000/threshold=0.1 point
in the sweep figures), and figureSpecs is {figureName: FigureSpec} describing how to render each one.

FigureSpec (a dict) has:
    - "kind": "grid" | "single" | "stack"
    - "nRows", "nCols" (grid only), "rowHeaders", "colHeaders" (grid/stack, or None)
    - "yLabel", "ylim"
    - "metric": Metrics.ORDER | Metrics.CLUSTER_NUMBER_WITH_RADIUS | Metrics.ORDER_VALUE_PERCENTAGE
    - "legendEntries": list of (label, colour, linestyle) - None means use the standard
      ordered/disordered legend
    - "cells": {(row, col): CellSpec} (grid) or [CellSpec, ...] (single/stack)
    - "backgroundSpan": (start, end) tuple, {(row, col): (start, end)} dict, or None

CellSpec (a dict):
    - "orderedPath", "randomPath": the two base save paths (already including run index will be
      appended as "_i") to evaluate and average across reps
    - for percentage metric only: "switchType", "switchTypeOptions", "invertPercentage" (whether to
      show 100-x instead of x, since "percentage of swarm choosing the second item in the
      combination" doesn't always mean the same switchTypeOptions slot - see comboKSwitch/
      comboNsmSwitch docstrings)
    - "seriesLabelSuffix": optional string appended to "ordered start"/"disordered start" labels,
      used only by switching_no_ev_order, which needs its own 10-entry legend instead of the
      standard 2-entry one
    - "colourOverride": optional (colour, colour) pair for (ordered, disordered), used by
      switching_no_ev_order for its per-combo colours
"""

DOMAIN_SIZE = (50, 50)
TMAX_GLOBAL = 3000
TMAX_LOCAL = 15000
EVENT_START = 5000
DEFAULT_EVENT_DURATION = 1000
DEFAULT_WINDOW = 100
DEFAULT_THRESHOLD = 0.1
DEFAULT_NOISE_PCT = 1
DEFAULT_DENSITY = 0.06
DEFAULT_RADIUS = 10
SPEED = 1
DEGREES_OF_VISION = 2 * np.pi

K_COMBO = (5, 1)
# index 0 is always the order value, index 1 the disorder value (matches K_COMBO's convention and
# SwitchInformation's values=(orderSwitchValue, disorderSwitchValue)). FARTHEST/HOD are the
# order-associated mechanisms at k=1, not NEAREST/LOD - confirmed against s_d/s_o labelling.
NSM_COMBO_NEAREST_FARTHEST = [NeighbourSelectionMechanism.FARTHEST, NeighbourSelectionMechanism.NEAREST]
NSM_COMBO_LOD_HOD = [NeighbourSelectionMechanism.HIGHEST_ORIENTATION_DIFFERENCE, NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE]

EVENT_EFFECTS = [
    (EventEffect.ALIGN_TO_FIXED_ANGLE, "distant"),
    (EventEffect.AWAY_FROM_ORIGIN, "predator"),
    (EventEffect.RANDOM, "random"),
]

REDUCED_NSM = [NeighbourSelectionMechanism.NEAREST, NeighbourSelectionMechanism.FARTHEST,
               NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE, NeighbourSelectionMechanism.HIGHEST_ORIENTATION_DIFFERENCE]


def _n(density):
    return ServicePreparation.getNumberOfParticlesForConstantDensity(density=density, domainSize=DOMAIN_SIZE)


def _noise(noisePct):
    return ServicePreparation.getNoiseAmplitudeValueForPercentage(noisePct)


def _event(eventEffect, radius, duration):
    areas = [DOMAIN_SIZE[0] / 2, DOMAIN_SIZE[1] / 2, radius]
    return ExternalStimulusOrientationChangeEvent(startTimestep=EVENT_START, duration=duration, domainSize=DOMAIN_SIZE,
                                                   eventEffect=eventEffect, distributionType=DistributionType.LOCAL_SINGLE_SITE,
                                                   areas=[areas], radius=radius, angle=np.pi)


class ComboRegistry:
    """
    Deduplicates simulation runs across the whole pipeline: two figures (or two cells) that need the
    exact same combination of parameters share the one underlying set of runs, keyed by the
    combination's save path (which is itself built deterministically from its parameters).
    """
    def __init__(self, dataRoot, numReps, updateIfNoNeighbours=True):
        self.dataRoot = dataRoot
        self.numReps = numReps
        # whether an isolated individual (no other neighbours in radius) is still allowed to switch
        # strategy - passed straight through to VicsekWithNeighbourSelection. Only affects combos
        # that actually switch (comboKSwitch/comboNsmSwitch); irrelevant to comboNoSwitch.
        self.updateIfNoNeighbours = updateIfNoNeighbours
        self.runSpecs = {}  # savePathBase (with _ordered/_random suffix) -> run spec

    def register(self, savePathBase, tmax, n, logInterval, kwargsFnForStart):
        paths = {}
        for sc in ("ordered", "random"):
            path = f"{self.dataRoot}/{savePathBase}_{sc}"
            paths[sc] = path
            if path not in self.runSpecs:
                self.runSpecs[path] = dict(
                    savePathBase=path,
                    simulatorKwargsFn=(lambda sc=sc: kwargsFnForStart(sc)),
                    tmax=tmax, startingCondition=sc, domainSize=DOMAIN_SIZE, n=n, logInterval=logInterval,
                )
        return paths["ordered"], paths["random"]

    def comboNoSwitch(self, density, radius, noisePct, nsm, k, eventEffect=None, duration=DEFAULT_EVENT_DURATION):
        """No switching, optionally with one event. tmax/logInterval depend on whether there's an event."""
        n = _n(density)
        hasEvent = eventEffect is not None
        evTag = f"_ee={eventEffect.val}_dur={duration}" if hasEvent else "_noev"
        base = f"nosw_d={density}_r={radius}_noise={noisePct}_nsm={nsm.value}_k={k}{evTag}"

        def kwargsFn(sc):
            kwargs = dict(domainSize=DOMAIN_SIZE, radius=radius, noise=_noise(noisePct), numberOfParticles=n,
                          k=k, neighbourSelectionMechanism=nsm, speed=SPEED, degreesOfVision=DEGREES_OF_VISION)
            if hasEvent:
                kwargs["events"] = [_event(eventEffect, radius, duration)]
            return kwargs

        tmax, logInterval = (TMAX_LOCAL, 100) if hasEvent else (TMAX_GLOBAL, 1)
        return self.register(base, tmax, n, logInterval, kwargsFn)

    def comboKSwitch(self, density, radius, noisePct, nsm, eventEffect=None, duration=DEFAULT_EVENT_DURATION,
                      window=DEFAULT_WINDOW, threshold=DEFAULT_THRESHOLD):
        """
        k-switching (kCombo=(5,1)) on a fixed neighbour selection mechanism. Ordered start = k=5,
        disordered start = k=1 - so "percentage choosing switchTypeOptions[0]" already equals
        "percentage choosing k=5", i.e. the *second* item in labels like "<nearest(1), nearest(5)>".
        """
        n = _n(density)
        hasEvent = eventEffect is not None
        evTag = f"_ee={eventEffect.val}_dur={duration}" if hasEvent else "_noev"
        noNeighbourTag = "" if self.updateIfNoNeighbours else "_blockiso"
        base = f"ksw_d={density}_r={radius}_noise={noisePct}_nsm={nsm.value}_th={threshold}_w={window}{evTag}{noNeighbourTag}"

        def kwargsFn(sc):
            k = K_COMBO[0] if sc == "ordered" else K_COMBO[1]
            switchSummary = SwitchSummary([SwitchInformation(switchType=SwitchType.K, values=K_COMBO,
                                                               thresholds=[threshold], numberPreviousStepsForThreshold=window)])
            kwargs = dict(domainSize=DOMAIN_SIZE, radius=radius, noise=_noise(noisePct), numberOfParticles=n,
                          k=k, neighbourSelectionMechanism=nsm, speed=SPEED, degreesOfVision=DEGREES_OF_VISION,
                          switchSummary=switchSummary, updateIfNoNeighbours=self.updateIfNoNeighbours)
            if hasEvent:
                kwargs["events"] = [_event(eventEffect, radius, duration)]
            return kwargs

        ordered, random = self.register(base, TMAX_LOCAL, n, 100, kwargsFn)
        return ordered, random, SwitchType.K, K_COMBO, False  # invertPercentage=False

    def comboNsmSwitch(self, density, radius, noisePct, k, nsmCombo, eventEffect=None, duration=DEFAULT_EVENT_DURATION,
                        window=DEFAULT_WINDOW, threshold=DEFAULT_THRESHOLD):
        """
        nsm-switching on a fixed k. Ordered start = nsmCombo[0] (the "order" mechanism - by
        NSM_COMBO_NEAREST_FARTHEST/NSM_COMBO_LOD_HOD's construction this is FARTHEST/HOD, since those
        are empirically the order-associated mechanisms at k=1, not NEAREST/LOD), disordered start =
        nsmCombo[1] - so "percentage choosing switchTypeOptions[0]" already equals "percentage
        choosing the *second* item" in labels like "<nearest(1), farthest(1)>" (farthest(1)), no
        inversion needed.
        """
        n = _n(density)
        hasEvent = eventEffect is not None
        evTag = f"_ee={eventEffect.val}_dur={duration}" if hasEvent else "_noev"
        noNeighbourTag = "" if self.updateIfNoNeighbours else "_blockiso"
        base = f"nsmsw_d={density}_r={radius}_noise={noisePct}_nsmCombo={nsmCombo[0].value}-{nsmCombo[1].value}_k={k}_th={threshold}_w={window}{evTag}{noNeighbourTag}"

        def kwargsFn(sc):
            nsm = nsmCombo[0] if sc == "ordered" else nsmCombo[1]
            switchSummary = SwitchSummary([SwitchInformation(switchType=SwitchType.NEIGHBOUR_SELECTION_MECHANISM, values=nsmCombo,
                                                               thresholds=[threshold], numberPreviousStepsForThreshold=window)])
            kwargs = dict(domainSize=DOMAIN_SIZE, radius=radius, noise=_noise(noisePct), numberOfParticles=n,
                          k=k, neighbourSelectionMechanism=nsm, speed=SPEED, degreesOfVision=DEGREES_OF_VISION,
                          switchSummary=switchSummary, updateIfNoNeighbours=self.updateIfNoNeighbours)
            if hasEvent:
                kwargs["events"] = [_event(eventEffect, radius, duration)]
            return kwargs

        ordered, random = self.register(base, TMAX_LOCAL, n, 100, kwargsFn)
        return ordered, random, SwitchType.NEIGHBOUR_SELECTION_MECHANISM, tuple(nsmCombo), False  # invertPercentage=False


# The five row-combinations shared by fig_4_switching_order/switching_percentage/switching_clustering,
# switching_no_ev_order, event_duration_*, window_size_* and thresholds_*.
# Each entry: (key, rowLabel, kind, kwargs-for-comboKSwitch/comboNsmSwitch, percentageLabel)
SWITCHING_COMBOS = [
    dict(key="a", rowLabel="(a) <nearest(1), nearest(5)>", kind="k", nsm=NeighbourSelectionMechanism.NEAREST,
         percentageLabel="nearest(5)"),
    dict(key="b", rowLabel="(b) <lod(1), lod(5)>", kind="k", nsm=NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE,
         percentageLabel="lod(5)"),
    dict(key="c", rowLabel="(c) <hod(1), hod(5)>", kind="k", nsm=NeighbourSelectionMechanism.HIGHEST_ORIENTATION_DIFFERENCE,
         percentageLabel="hod(5)"),
    dict(key="d", rowLabel="(d) <nearest(1), farthest(1)>", kind="nsm", nsmCombo=NSM_COMBO_NEAREST_FARTHEST, k=1,
         percentageLabel="farthest(1)"),
    dict(key="e", rowLabel="(e) <lod(1), hod(1)>", kind="nsm", nsmCombo=NSM_COMBO_LOD_HOD, k=1,
         percentageLabel="hod(1)"),
]

# The (mechanism, k) row-combinations for fig_3_nosw_1ev_order / nosw_1ev_clusters. hod(1), hod(5)
# and farthest(5) are intentionally excluded.
NOSW_ROWS = [
    (NeighbourSelectionMechanism.NEAREST, 1, "nearest(1)"),
    (NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE, 1, "lod(1)"),
    (NeighbourSelectionMechanism.FARTHEST, 1, "farthest(1)"),
    (NeighbourSelectionMechanism.NEAREST, 5, "nearest(5)"),
    (NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE, 5, "lod(5)"),
]

MECHANISM_LABELS = {
    NeighbourSelectionMechanism.ALL: "Standard Vicsek (SV)",
    NeighbourSelectionMechanism.RANDOM: "Random",
    NeighbourSelectionMechanism.NEAREST: "Nearest",
    NeighbourSelectionMechanism.FARTHEST: "Farthest",
    NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE: "Least Orientation Difference",
    NeighbourSelectionMechanism.HIGHEST_ORIENTATION_DIFFERENCE: "Highest Orientation Difference",
}


def _switchingCombo(reg, combo, density, radius, noisePct, eventEffect=None, duration=DEFAULT_EVENT_DURATION,
                     window=DEFAULT_WINDOW, threshold=DEFAULT_THRESHOLD):
    if combo["kind"] == "k":
        return reg.comboKSwitch(density, radius, noisePct, combo["nsm"], eventEffect=eventEffect, duration=duration,
                                 window=window, threshold=threshold)
    return reg.comboNsmSwitch(density, radius, noisePct, combo["k"], combo["nsmCombo"], eventEffect=eventEffect,
                               duration=duration, window=window, threshold=threshold)


def _cell(orderedPath, randomPath, switchInfo=None):
    cell = {"orderedPath": orderedPath, "randomPath": randomPath}
    if switchInfo is not None:
        switchType, switchTypeOptions, invert = switchInfo
        cell["switchType"] = switchType
        cell["switchTypeOptions"] = switchTypeOptions
        cell["invertPercentage"] = invert
    return cell


def _standardFigure(kind, nRows, nCols, rowHeaders, colHeaders, yLabel, metric, cells, ylim=None, backgroundSpan=None):
    return dict(kind=kind, nRows=nRows, nCols=nCols, rowHeaders=rowHeaders, colHeaders=colHeaders,
                yLabel=yLabel, metric=metric, cells=cells, legendEntries=None, ylim=ylim, backgroundSpan=backgroundSpan)


def buildAll(dataRoot, numReps, updateIfNoNeighbours=True):
    reg = ComboRegistry(dataRoot, numReps, updateIfNoNeighbours=updateIfNoNeighbours)
    figures = {}

    # ---------------------------------------------------------------- fig_2_global_order / app_global_clusters
    grid6 = [
        [(NeighbourSelectionMechanism.ALL, None), (NeighbourSelectionMechanism.RANDOM, None)],
        [(NeighbourSelectionMechanism.NEAREST, None), (NeighbourSelectionMechanism.FARTHEST, None)],
        [(NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE, None), (NeighbourSelectionMechanism.HIGHEST_ORIENTATION_DIFFERENCE, None)],
    ]
    for k in (1, 5):
        cellsOrder, cellsClusters = {}, {}
        for r in range(3):
            for c in range(2):
                nsm, _ = grid6[r][c]
                ordered, random = reg.comboNoSwitch(DEFAULT_DENSITY, DEFAULT_RADIUS, DEFAULT_NOISE_PCT, nsm, k)
                cellsOrder[(r, c)] = _cell(ordered, random)
                cellsClusters[(r, c)] = _cell(ordered, random)
        rowHeaders = None
        colHeaders = None
        titles = {(r, c): MECHANISM_LABELS[grid6[r][c][0]] for r in range(3) for c in range(2)}
        figures[f"fig_2_global_order_k{k}"] = _standardFigure("grid", 3, 2, None, None, "global order",
                                                                Metrics.ORDER, cellsOrder, ylim=(0, 1.1))
        figures[f"fig_2_global_order_k{k}"]["cellTitles"] = titles
        figures[f"app_global_clusters_k{k}"] = _standardFigure("grid", 3, 2, None, None, "number of clusters",
                                                                 Metrics.CLUSTER_NUMBER_WITH_RADIUS, cellsClusters)
        figures[f"app_global_clusters_k{k}"]["cellTitles"] = titles

    # ---------------------------------------------------------------- fig_3_nosw_1ev_order / nosw_1ev_clusters
    cellsOrder, cellsClusters = {}, {}
    rowHeaders = [label for (_, _, label) in NOSW_ROWS]
    colHeaders = [label for (_, label) in EVENT_EFFECTS]
    for r, (nsm, k, _label) in enumerate(NOSW_ROWS):
        for c, (eventEffect, _colLabel) in enumerate(EVENT_EFFECTS):
            ordered, random = reg.comboNoSwitch(DEFAULT_DENSITY, DEFAULT_RADIUS, DEFAULT_NOISE_PCT, nsm, k, eventEffect=eventEffect)
            cellsOrder[(r, c)] = _cell(ordered, random)
            cellsClusters[(r, c)] = _cell(ordered, random)
    span = (EVENT_START, EVENT_START + DEFAULT_EVENT_DURATION)
    figures["fig_3_nosw_1ev_order"] = _standardFigure("grid", len(NOSW_ROWS), 3, rowHeaders, colHeaders, "global order",
                                                        Metrics.ORDER, cellsOrder, ylim=(0, 1.1), backgroundSpan=span)
    figures["nosw_1ev_clusters"] = _standardFigure("grid", len(NOSW_ROWS), 3, rowHeaders, colHeaders, "number of clusters",
                                                     Metrics.CLUSTER_NUMBER_WITH_RADIUS, cellsClusters, backgroundSpan=span)

    # ---------------------------------------------------------------- fig_4_switching_order / _percentage / _clustering
    cellsOrder, cellsClusters, cellsPct = {}, {}, {}
    rowHeaders = [c["rowLabel"] for c in SWITCHING_COMBOS]
    colHeaders = [label for (_, label) in EVENT_EFFECTS]
    for r, combo in enumerate(SWITCHING_COMBOS):
        for c, (eventEffect, _colLabel) in enumerate(EVENT_EFFECTS):
            ordered, random, switchType, switchTypeOptions, invert = _switchingCombo(reg, combo, DEFAULT_DENSITY, DEFAULT_RADIUS,
                                                                                       DEFAULT_NOISE_PCT, eventEffect=eventEffect)
            cellsOrder[(r, c)] = _cell(ordered, random)
            # clusters and percentage share the exact same underlying run data for these combos - attaching
            # switchInfo to the clusters cell too (even though it doesn't need it) lets ServicePaperFigureRunner's
            # data cache load the switch columns once and reuse them for both metrics instead of loading twice.
            cellsClusters[(r, c)] = _cell(ordered, random, switchInfo=(switchType, switchTypeOptions, invert))
            cellsPct[(r, c)] = _cell(ordered, random, switchInfo=(switchType, switchTypeOptions, invert))

    # fig_4_switching_order alone shows a reduced set of rows - <nearest(1), nearest(5)> [a],
    # <nearest(1), farthest(1)> [d] and <lod(1), lod(5)> [b], in that order - while
    # switching_clustering/switching_percentage (built from the same cellsClusters/cellsPct below)
    # keep all five rows.
    FIG_4_ROW_KEYS = ["a", "d", "b"]
    fig4RowIndices = sorted((i for i, c in enumerate(SWITCHING_COMBOS) if c["key"] in FIG_4_ROW_KEYS),
                            key=lambda i: FIG_4_ROW_KEYS.index(SWITCHING_COMBOS[i]["key"]))
    fig4RowHeaders = [SWITCHING_COMBOS[i]["rowLabel"] for i in fig4RowIndices]
    fig4CellsOrder = {(newR, c): cellsOrder[(origR, c)] for newR, origR in enumerate(fig4RowIndices) for c in range(len(EVENT_EFFECTS))}
    figures["fig_4_switching_order"] = _standardFigure("grid", len(fig4RowIndices), 3, fig4RowHeaders, colHeaders, "global order",
                                                         Metrics.ORDER, fig4CellsOrder, ylim=(0, 1.1), backgroundSpan=span)
    figures["switching_clustering"] = _standardFigure("grid", 5, 3, rowHeaders, colHeaders, "number of clusters",
                                                        Metrics.CLUSTER_NUMBER_WITH_RADIUS, cellsClusters, backgroundSpan=span)
    pctFig = _standardFigure("grid", 5, 3, rowHeaders, colHeaders, "% order-inducing", Metrics.ORDER_VALUE_PERCENTAGE,
                              cellsPct, ylim=(0, 100.1), backgroundSpan=span)
    pctFig["cellPercentageLabels"] = {(r, 0): combo["percentageLabel"] for r, combo in enumerate(SWITCHING_COMBOS)}
    figures["switching_percentage"] = pctFig

    # ---------------------------------------------------------------- switching_no_ev_order (single panel, 10 lines)
    from services.ServicePaperFigureGrid import CATEGORICAL_COLOURS
    singleCells = []
    for i, combo in enumerate(SWITCHING_COMBOS):
        ordered, random, _switchType, _opts, _invert = _switchingCombo(reg, combo, DEFAULT_DENSITY, DEFAULT_RADIUS, DEFAULT_NOISE_PCT)
        colour = CATEGORICAL_COLOURS[i % len(CATEGORICAL_COLOURS)]
        singleCells.append(_cell(ordered, random))
        singleCells[-1]["seriesLabelSuffix"] = f" {combo['rowLabel']}"
        singleCells[-1]["colourOverride"] = (colour, colour)
        singleCells[-1]["linestyleOverride"] = ("-", "--")
    figures["switching_no_ev_order"] = dict(kind="single", yLabel="global order", ylim=(0, 1.1), metric=Metrics.ORDER,
                                             cells=singleCells, legendEntries="perCell", backgroundSpan=None)

    # ---------------------------------------------------------------- event_duration_* / window_size_* / thresholds_*
    durations = [1, 10, 50, 100, 200, 500, 1000]
    windows = [1, 25, 50, 75, 200]
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

    def _sweepGrid(sweepName, sweepValues, sweepKwargName, metricSuffix, metric, yLabel, ylim=None):
        for combo in SWITCHING_COMBOS:
            cells = {}
            backgroundSpans = {}
            colHeaders = [label for (_, label) in EVENT_EFFECTS]
            rowHeaders = [str(v) for v in sweepValues]
            for r, sweepVal in enumerate(sweepValues):
                for c, (eventEffect, _colLabel) in enumerate(EVENT_EFFECTS):
                    kwargs = {sweepKwargName: sweepVal}
                    duration = sweepVal if sweepKwargName == "duration" else DEFAULT_EVENT_DURATION
                    ordered, random, switchType, switchTypeOptions, invert = _switchingCombo(
                        reg, combo, DEFAULT_DENSITY, DEFAULT_RADIUS, DEFAULT_NOISE_PCT, eventEffect=eventEffect, **kwargs)
                    if metric in (Metrics.ORDER_VALUE_PERCENTAGE, Metrics.CLUSTER_NUMBER_WITH_RADIUS):
                        # clusters and percentage share the same underlying run data for these combos -
                        # attaching switchInfo to both (even though clusters doesn't need it) lets the
                        # data cache load the switch columns once and reuse them for both metrics.
                        cells[(r, c)] = _cell(ordered, random, switchInfo=(switchType, switchTypeOptions, invert))
                    else:
                        cells[(r, c)] = _cell(ordered, random)
                    backgroundSpans[(r, c)] = (EVENT_START, EVENT_START + duration)
            spec = _standardFigure("grid", len(sweepValues), 3, rowHeaders, colHeaders, yLabel, metric, cells,
                                    ylim=ylim, backgroundSpan=backgroundSpans)
            if metric == Metrics.ORDER_VALUE_PERCENTAGE:
                spec["cellPercentageLabels"] = {(0, 0): combo["percentageLabel"]}
            figures[f"{sweepName}_{metricSuffix}_{combo['key']}"] = spec

    _sweepGrid("event_duration", durations, "duration", "order", Metrics.ORDER, "global order", ylim=(0, 1.1))
    _sweepGrid("event_duration", durations, "duration", "clusters", Metrics.CLUSTER_NUMBER_WITH_RADIUS, "number of clusters")
    _sweepGrid("event_duration", durations, "duration", "percentage", Metrics.ORDER_VALUE_PERCENTAGE, "% order-inducing", ylim=(0, 100.1))

    _sweepGrid("window_size", windows, "window", "order", Metrics.ORDER, "global order", ylim=(0, 1.1))
    _sweepGrid("window_size", windows, "window", "clusters", Metrics.CLUSTER_NUMBER_WITH_RADIUS, "number of clusters")
    _sweepGrid("window_size", windows, "window", "percentage", Metrics.ORDER_VALUE_PERCENTAGE, "% order-inducing", ylim=(0, 100.1))

    _sweepGrid("thresholds", thresholds, "threshold", "order", Metrics.ORDER, "global order", ylim=(0, 1.1))
    _sweepGrid("thresholds", thresholds, "threshold", "clusters", Metrics.CLUSTER_NUMBER_WITH_RADIUS, "number of clusters")
    _sweepGrid("thresholds", thresholds, "threshold", "percentage", Metrics.ORDER_VALUE_PERCENTAGE, "% order-inducing", ylim=(0, 100.1))

    # ---------------------------------------------------------------- noise_elevation_all (stack, mechanism=ALL only)
    stackCells = []
    for noisePct in (1, 4):
        ordered, random = reg.comboNoSwitch(DEFAULT_DENSITY, DEFAULT_RADIUS, noisePct, NeighbourSelectionMechanism.ALL, 1)
        stackCells.append(_cell(ordered, random))
    figures["noise_elevation_all"] = dict(kind="stack", yLabel="global order", xLabel="timesteps", ylim=(0, 1.1),
                                           metric=Metrics.ORDER, cells=stackCells, legendEntries=None,
                                           panelHeaders=["1% noise", "4% noise"], backgroundSpan=None)

    # ---------------------------------------------------------------- noise_elevation_random/nearest/farthest/lod/hod
    kValues = [0, 1, 2, 3, 4, 5]
    noiseMechanisms = [
        (NeighbourSelectionMechanism.RANDOM, "random"),
        (NeighbourSelectionMechanism.NEAREST, "nearest"),
        (NeighbourSelectionMechanism.FARTHEST, "farthest"),
        (NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE, "lod"),
        (NeighbourSelectionMechanism.HIGHEST_ORIENTATION_DIFFERENCE, "hod"),
    ]
    for nsm, slug in noiseMechanisms:
        cells = {}
        for r, noisePct in enumerate((1, 4)):
            for c, k in enumerate(kValues):
                ordered, random = reg.comboNoSwitch(DEFAULT_DENSITY, DEFAULT_RADIUS, noisePct, nsm, k)
                cells[(r, c)] = _cell(ordered, random)
        rowHeaders = ["1% noise", "4% noise"]
        colHeaders = [f"k={k}" for k in kValues]
        figures[f"noise_elevation_{slug}"] = _standardFigure("grid", 2, 6, rowHeaders, colHeaders, "global order",
                                                               Metrics.ORDER, cells, ylim=(0, 1.1))

    # ---------------------------------------------------------------- density_radius_*
    densities = [0.01, 0.06, 0.09, 0.12]
    radii = [5, 10, 20]
    densityRadiusMechanisms = [
        (NeighbourSelectionMechanism.ALL, "all", [None]),
        (NeighbourSelectionMechanism.RANDOM, "random", [1, 5]),
        (NeighbourSelectionMechanism.NEAREST, "nearest", [1, 5]),
        (NeighbourSelectionMechanism.FARTHEST, "farthest", [1, 5]),
        (NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE, "lod", [1, 5]),
        (NeighbourSelectionMechanism.HIGHEST_ORIENTATION_DIFFERENCE, "hod", [1, 5]),
    ]
    rowHeaders = [f"d={d}" for d in densities]
    colHeaders = [f"r={r}" for r in radii]
    for nsm, slug, ks in densityRadiusMechanisms:
        for k in ks:
            cells = {}
            for r, density in enumerate(densities):
                for c, radius in enumerate(radii):
                    ordered, random = reg.comboNoSwitch(density, radius, DEFAULT_NOISE_PCT, nsm, k if k is not None else 1)
                    cells[(r, c)] = _cell(ordered, random)
            name = f"density_radius_{slug}_order" if k is None else f"density_radius_{slug}_order_k{k}"
            figures[name] = _standardFigure("grid", 4, 3, rowHeaders, colHeaders, "global order", Metrics.ORDER,
                                             cells, ylim=(0, 1.1))

    return reg.runSpecs, figures


# Figures that involve switching at all (i.e. built from comboKSwitch/comboNsmSwitch) - the only ones
# affected by ComboRegistry's updateIfNoNeighbours flag, since it's meaningless without a switchSummary.
# Used by run_paper_figures_no_neighbour_switch.py to only (re)generate this subset.
SWITCHING_FIGURE_PREFIXES = ("fig_4_switching", "switching_percentage", "switching_clustering",
                              "switching_no_ev_order", "event_duration_", "window_size_", "thresholds_")


def isSwitchingFigure(name):
    return name.startswith(SWITCHING_FIGURE_PREFIXES)
