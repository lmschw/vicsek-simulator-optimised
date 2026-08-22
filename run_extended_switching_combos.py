import argparse
import os
import time

import paper_figures_specs as specs
import services.ServicePaperFigureRunner as Runner
import services.ServicePaperFigureGrid as Grid
import services.ServiceGeneral as ServiceGeneral
import run_paper_figures as MainPipeline
from enums.EnumMetrics import Metrics
from enums.EnumSwitchType import SwitchType
from enums.EnumNeighbourSelectionMechanism import NeighbourSelectionMechanism
from model.SwitchInformation import SwitchInformation
from model.SwitchSummary import SwitchSummary

"""
An extended set of switching combinations beyond paper_figures_specs.SWITCHING_COMBOS: every
disorder-inducing strategy paired with every order-inducing strategy (full cross product), each pair
always at k=0 for disorder:

    disorder (always k=0): random(0), nearest(0), farthest(0), lod(0), hod(0)
    order:                 random(1), random(5), nearest(5), farthest(1), farthest(5), lod(5)

5 disorder x 6 order = 30 switching combinations x 3 events (distant/predator/random) = 90
combinations, each with an ordered-start and a disordered-start run.

Rendered as 6 separate figures (one per order strategy - "separate plots by order-inducing
strategy"), each in fig_4_switching_order's layout: 5 rows (disorder strategies) x 3 columns (event
effects), with the standard ordered/disordered-start legend. Three metrics per order strategy: global
order, cluster count (ServiceNeighbourSelectionEval.CLUSTER_THRESHOLD, tuned so a fully-ordered swarm
reads as 1 cluster - see that constant's comment), and cluster count normalised by particle count
(clusters / n) - 18 figures total.

Implementation note - joint (mechanism, k) switching: the existing SwitchType machinery only switches
one property at a time (either NEIGHBOUR_SELECTION_MECHANISM with k fixed, or K with the mechanism
fixed) - see paper_figures_specs.comboKSwitch/comboNsmSwitch. To get a disorder state and an order
state that can each specify BOTH a mechanism AND a k, this registers two SwitchInformation objects (one
per SwitchType) in the same SwitchSummary, with IDENTICAL thresholds/window. Both switches read the
exact same thresholdEvaluationChoiceValues every timestep (see
VicsekWithNeighbourSelection.simulate/getDecisions), so with matching parameters they always flip in
lockstep for a given particle - functionally a single joint switch. k=0 needs no special-casing in the
model: with 0 candidates picked by the mechanism, computeNewOrientations's later
np.fill_diagonal(pickedNeighbours, True) still applies, so it correctly reduces to "no real neighbours,
self only" for every disorder-inducing mechanism regardless of which one is nominally selected.

Uses its own scratch data directory (not the main pipeline's ~/paper_figures_tmp/) so it can run
safely alongside an in-progress run of run_paper_figures.py without touching its files.

WARNING: this is by far the most expensive script in this project so far. 90 combinations x 2 starting
conditions x --num-reps (50 by default) repetitions x tmax=15000 - at the per-step costs measured
elsewhere in this project (roughly 2-8ms/step depending on mechanism), that's on the order of a full
day or more of CPU time, only some of which is parallelised away by --workers. Consider a smaller
--num-reps for a first look before committing to the default.
"""

DISORDER_STRATEGIES = [
    (NeighbourSelectionMechanism.RANDOM, 0, "random(0)"),
    (NeighbourSelectionMechanism.NEAREST, 0, "nearest(0)"),
    (NeighbourSelectionMechanism.FARTHEST, 0, "farthest(0)"),
    (NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE, 0, "lod(0)"),
    (NeighbourSelectionMechanism.HIGHEST_ORIENTATION_DIFFERENCE, 0, "hod(0)"),
]
ORDER_STRATEGIES = [
    (NeighbourSelectionMechanism.RANDOM, 1, "random(1)"),
    (NeighbourSelectionMechanism.RANDOM, 5, "random(5)"),
    (NeighbourSelectionMechanism.NEAREST, 5, "nearest(5)"),
    (NeighbourSelectionMechanism.FARTHEST, 1, "farthest(1)"),
    (NeighbourSelectionMechanism.FARTHEST, 5, "farthest(5)"),
    (NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE, 5, "lod(5)"),
]

DEFAULT_DATA_ROOT = os.path.expanduser("~/paper_figures_extended_switching_tmp")
DEFAULT_OUTPUT_ROOT = "plots/paper_figures_extended_switching"


def registerCombo(reg, disorderNsm, disorderK, orderNsm, orderK, eventEffect,
                   density=specs.DEFAULT_DENSITY, radius=specs.DEFAULT_RADIUS, noisePct=specs.DEFAULT_NOISE_PCT,
                   threshold=specs.DEFAULT_THRESHOLD, window=specs.DEFAULT_WINDOW, duration=specs.DEFAULT_EVENT_DURATION):
    n = specs._n(density)
    base = (f"extsw_d={density}_r={radius}_noise={noisePct}_"
            f"dis={disorderNsm.value}k{disorderK}_ord={orderNsm.value}k{orderK}_"
            f"th={threshold}_w={window}_ee={eventEffect.val}_dur={duration}")

    def kwargsFn(sc):
        nsm = orderNsm if sc == "ordered" else disorderNsm
        switchSummary = SwitchSummary([
            SwitchInformation(switchType=SwitchType.NEIGHBOUR_SELECTION_MECHANISM, values=(orderNsm, disorderNsm),
                               thresholds=[threshold], numberPreviousStepsForThreshold=window),
            SwitchInformation(switchType=SwitchType.K, values=(orderK, disorderK),
                               thresholds=[threshold], numberPreviousStepsForThreshold=window),
        ])
        k = orderK if sc == "ordered" else disorderK
        return dict(domainSize=specs.DOMAIN_SIZE, radius=radius, noise=specs._noise(noisePct), numberOfParticles=n,
                    k=k, neighbourSelectionMechanism=nsm, speed=specs.SPEED, degreesOfVision=specs.DEGREES_OF_VISION,
                    switchSummary=switchSummary,
                    events=[specs._event(eventEffect, radius, duration)])

    ordered, random = reg.register(base, specs.TMAX_LOCAL, n, 100, kwargsFn)
    return ordered, random


def buildFigureSpecs(dataRoot):
    reg = specs.ComboRegistry(dataRoot, 1)
    colHeaders = [label for (_, label) in specs.EVENT_EFFECTS]

    # figures[metricSuffix][orderLabel] -> FigureSpec
    figures = {"order": {}, "clusters": {}, "clusters_per_agent": {}}
    for orderNsm, orderK, orderLabel in ORDER_STRATEGIES:
        rowHeaders = [disorderLabel for (_, _, disorderLabel) in DISORDER_STRATEGIES]
        cellsOrder, cellsClusters = {}, {}
        for r, (disorderNsm, disorderK, disorderLabel) in enumerate(DISORDER_STRATEGIES):
            for c, (eventEffect, _colLabel) in enumerate(specs.EVENT_EFFECTS):
                ordered, random = registerCombo(reg, disorderNsm, disorderK, orderNsm, orderK, eventEffect)
                cellsOrder[(r, c)] = dict(orderedPath=ordered, randomPath=random)
                cellsClusters[(r, c)] = dict(orderedPath=ordered, randomPath=random)
        span = (specs.EVENT_START, specs.EVENT_START + specs.DEFAULT_EVENT_DURATION)
        figures["order"][orderLabel] = dict(kind="grid", nRows=len(DISORDER_STRATEGIES), nCols=3,
                                             rowHeaders=rowHeaders, colHeaders=colHeaders, yLabel="global order",
                                             metric=Metrics.ORDER, cells=cellsOrder, legendEntries=None,
                                             ylim=(0, 1.1), backgroundSpan=span)
        figures["clusters"][orderLabel] = dict(kind="grid", nRows=len(DISORDER_STRATEGIES), nCols=3,
                                                rowHeaders=rowHeaders, colHeaders=colHeaders, yLabel="number of clusters",
                                                metric=Metrics.CLUSTER_NUMBER_WITH_RADIUS, cells=cellsClusters,
                                                legendEntries=None, ylim=None, backgroundSpan=span)
        # clusters_per_agent reuses the exact same underlying cells/data as clusters - only the
        # rendered series get rescaled (see renderClustersPerAgent), so no separate registration.
        figures["clusters_per_agent"][orderLabel] = figures["clusters"][orderLabel]

    return reg, figures


def renderOrderOrClusters(name, figureSpec, numReps, runSpecs, outputRoot):
    MainPipeline.renderFigure(name, figureSpec, numReps, runSpecs, os.path.join(outputRoot, "order_and_clusters"))


def renderClustersPerAgent(name, figureSpec, numReps, n, runSpecs, outputRoot):
    cellsOut = {}
    for (r, c), cellSpec in figureSpec["cells"].items():
        evalInterval = MainPipeline.evalIntervalForPath(cellSpec["orderedPath"], runSpecs)
        results = Runner.evaluateCell(cellSpec["orderedPath"], cellSpec["randomPath"], numReps,
                                       Metrics.CLUSTER_NUMBER_WITH_RADIUS, evalInterval)
        scaled = {}
        for label in ("ordered", "disordered"):
            result = results[label]
            scaled[label] = None if result is None else (result[0], result[1] / n, result[2] / n)
        cellsOut[(r, c)] = Grid.orderedDisorderedSeries(scaled["ordered"], scaled["disordered"])
    outputPathBase = os.path.join(outputRoot, "clusters_per_agent", name)
    Grid.renderGrid(cellsOut, figureSpec["nRows"], figureSpec["nCols"], figureSpec["rowHeaders"],
                     figureSpec["colHeaders"], "clusters / agents", "timesteps", Grid.STANDARD_LEGEND,
                     outputPathBase, ylim=(0, 1.05), backgroundSpan=figureSpec.get("backgroundSpan"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--num-reps", type=int, default=50)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    parser.add_argument("--skip-data", action="store_true", help="skip simulation, just (re-)render from existing data")
    parser.add_argument("--list", action="store_true", help="list the targeted combinations and exit, without running anything")
    args = parser.parse_args()

    reg, figures = buildFigureSpecs(args.data_root)
    n = specs._n(specs.DEFAULT_DENSITY)

    if args.list:
        print(f"{len(reg.runSpecs)} unique combinations x 2 starting conditions x {args.num_reps} reps each")
        print(f"{len(ORDER_STRATEGIES)} order strategies (separate figures), {len(DISORDER_STRATEGIES)} disorder "
              f"strategies (rows) x 3 events (columns) each:")
        for orderNsm, orderK, orderLabel in ORDER_STRATEGIES:
            print(f"  {orderLabel}")
        return

    os.makedirs(args.data_root, exist_ok=True)
    os.makedirs(args.output_root, exist_ok=True)

    if not args.skip_data:
        seedLog = os.path.join(args.data_root, "seeds.csv")
        ServiceGeneral.logWithTime(f"generating {len(reg.runSpecs)} combinations x {args.num_reps} reps...")
        t0 = time.time()
        Runner.runBatch(list(reg.runSpecs.values()), seedLog, args.num_reps, args.workers)
        ServiceGeneral.logWithTime(f"  done ({ServiceGeneral.formatTime(time.time() - t0)})")

    ServiceGeneral.logWithTime("rendering order + cluster-count figures...")
    for orderLabel, figureSpec in figures["order"].items():
        renderOrderOrClusters(f"order_{orderLabel}", figureSpec, args.num_reps, reg.runSpecs, args.output_root)
    for orderLabel, figureSpec in figures["clusters"].items():
        renderOrderOrClusters(f"clusters_{orderLabel}", figureSpec, args.num_reps, reg.runSpecs, args.output_root)
    ServiceGeneral.logWithTime("rendering clusters/agent figures...")
    for orderLabel, figureSpec in figures["clusters_per_agent"].items():
        renderClustersPerAgent(f"clusters_per_agent_{orderLabel}", figureSpec, args.num_reps, n, reg.runSpecs, args.output_root)

    ServiceGeneral.logWithTime("done")


if __name__ == "__main__":
    main()
