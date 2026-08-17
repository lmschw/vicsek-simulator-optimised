import csv
import hashlib
import os
import random
import time

from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from model.VicsekIndividualsMultiSwitch import VicsekWithNeighbourSelection
from enums.EnumMetrics import Metrics

import services.ServicePreparation as ServicePreparation
import services.ServiceGeneral as ServiceGeneral
import services.ServiceSavedModel as ServiceSavedModel
import services.ServiceNeighbourSelectionEval as ServiceNeighbourSelectionEval
import evaluators.EvaluatorMultiComp as EvaluatorMultiComp

"""
Seeded, resumable execution of the runs needed for the paper figures (see paper_figures_specs.py),
plus thin wrappers reusing the existing evaluation code in ServiceNeighbourSelectionEval.py.

Each individual simulation run is fully reproducible: its seed is derived deterministically from a
single master seed and the run's own save path (see deriveSeed), independent of execution order, and
recorded to a seed log before the run happens.
"""

MASTER_SEED = 20260807


def deriveSeed(masterSeed, uniqueKey):
    """
    Deterministically derives a reproducible 32-bit seed from a master seed and a run's unique
    identity (its save path). Hash-based rather than positional (e.g. a simple counter), so a run's
    seed never depends on what order runs happen to be enumerated/submitted in - only on its own
    identity and the master seed, both of which are recorded.
    """
    digest = hashlib.sha256(f"{masterSeed}:{uniqueKey}".encode()).digest()
    keyInt = int.from_bytes(digest[:8], "little")
    return int(np.random.SeedSequence([masterSeed, keyInt]).generate_state(1)[0])


def runSimulationJob(simulator, initialState, tmax, seed, label):
    """
    Runs a single simulation to completion with a fixed, reproducible seed. Executed in a worker
    process (see runBatch).
    """
    np.random.seed(seed)
    jobStart = time.time()
    if initialState is not None:
        simulator.simulate(initialState=initialState, tmax=tmax)
    else:
        simulator.simulate(tmax=tmax)
    return label, time.time() - jobStart


def isRunComplete(savePath, tmax):
    """
    Cheap completion check used for resumability: a run is considered done if its globalOrder log
    reaches tmax. Avoids reloading/re-simulating runs that already finished in an earlier, interrupted
    invocation of the pipeline.
    """
    path = f"{savePath}_globalOrder.csv"
    if not os.path.exists(path):
        return False
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 200))
            tail = f.read().decode(errors="ignore")
        lastLine = [line for line in tail.strip().split("\n") if line][-1]
        lastT = int(lastLine.split(",")[0])
        return lastT >= tmax
    except Exception:
        return False


def appendSeedLog(seedLogPath, entries):
    """
    entries: list of (savePath, seed). Writes a header (with the master seed) the first time the
    log is created, then appends - so the log is always a complete, correct record of every run's
    seed even if the pipeline is interrupted mid-batch.
    """
    if not entries:
        return
    os.makedirs(os.path.dirname(seedLogPath), exist_ok=True)
    isNew = not os.path.exists(seedLogPath)
    with open(seedLogPath, "a", newline="") as f:
        w = csv.writer(f)
        if isNew:
            w.writerow([f"# master_seed={MASTER_SEED}"])
            w.writerow(["save_path", "seed"])
        for savePath, seed in entries:
            w.writerow([savePath, seed])


def buildInitialState(startingCondition, domainSize, n, seed):
    if startingCondition == "ordered":
        # createOrderedInitialDistributionEquidistancedIndividual draws the shared initial
        # orientation angle from Python's stdlib random module (not numpy), and does so here in the
        # main process rather than inside the seeded worker - so it needs its own explicit seeding
        # to be reproducible; np.random.seed(seed) in the worker has no effect on it.
        random.seed(seed)
        return ServicePreparation.createOrderedInitialDistributionEquidistancedIndividual(None, domainSize, n)
    return None


def runBatch(runSpecs, seedLogPath, numReps, workers):
    """
    runSpecs: list of dicts, each describing one *combination* (not one repetition):
        - savePathBase (str): path prefix, without the trailing "_i"
        - simulatorKwargsFn (callable() -> dict): builds the VicsekWithNeighbourSelection kwargs
          (everything except logPath/logInterval/returnHistories) - a callable rather than a plain
          dict so that mutable objects it references (SwitchSummary, events) are freshly constructed
          per call rather than shared/mutated across reps.
        - tmax (int)
        - startingCondition ("ordered" | "random")
        - domainSize (tuple), n (int)
        - logInterval (int)

    For each combination, runs `numReps` repetitions (i = 0..numReps-1), skipping any already
    complete (see isRunComplete). Seeds are derived and logged before any run happens.
    """
    jobs = []
    seedEntries = []
    for spec in runSpecs:
        for i in range(numReps):
            savePath = f"{spec['savePathBase']}_{i}"
            seed = deriveSeed(MASTER_SEED, savePath)
            seedEntries.append((savePath, seed))
            if isRunComplete(savePath, spec["tmax"]):
                continue
            simulator = VicsekWithNeighbourSelection(
                **spec["simulatorKwargsFn"](),
                logPath=savePath,
                logInterval=spec["logInterval"],
                returnHistories=False,
            )
            initialState = buildInitialState(spec["startingCondition"], spec["domainSize"], spec["n"], seed)
            jobs.append((simulator, initialState, spec["tmax"], seed, savePath))

    appendSeedLog(seedLogPath, seedEntries)

    if not jobs:
        return 0

    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(runSimulationJob, *job) for job in jobs]
        for future in as_completed(futures):
            future.result()
            completed += 1
    return completed


# Cache of loaded per-run CSV data, keyed by (basePath, i): (modelParams, simulationData, switchValuesDict
# or None). Several figures can need the CLUSTER_NUMBER_WITH_RADIUS and/or ORDER_VALUE_PERCENTAGE metric
# for the exact same underlying combination (e.g. switching_clustering/switching_percentage, or fig_4's
# default point coinciding with a sweep figure's default-value column) - both metrics need the full
# per-timestep position/orientation/switch-value data, which is the expensive part to load (not the metric
# computation itself), so loading it once per run and reusing it across metrics/figures avoids re-reading
# and re-parsing the same CSV file repeatedly. Cleared per-combination via evictCache() once every figure
# referencing it is done (see run_paper_figures.py), so it stays bounded to whatever's currently in flight.
_rawDataCache = {}


def _loadRunDataCached(basePath, i, switchTypes):
    """
    switchTypes: list of SwitchType to load switch values for (empty if the metric doesn't need them).
    If an earlier call cached this run without switch data but a later call needs it, reloads (with
    switch data) rather than returning a stale, incomplete cache entry.
    """
    key = (basePath, i)
    cached = _rawDataCache.get(key)
    if cached is not None and (not switchTypes or cached[2] is not None):
        return cached
    try:
        if switchTypes:
            params, simData, switchVals = ServiceSavedModel.loadModelFromCsv(
                filepathData=f"{basePath}_{i}.csv", filePathModelParams=f"{basePath}_{i}_modelParams.csv",
                switchTypes=switchTypes)
        else:
            params, simData = ServiceSavedModel.loadModelFromCsv(
                filepathData=f"{basePath}_{i}.csv", filePathModelParams=f"{basePath}_{i}_modelParams.csv")
            switchVals = None
    except Exception:
        return None
    result = (params, simData, switchVals)
    _rawDataCache[key] = result
    return result


def evictCache(basePath, numReps):
    """Drops cached loaded data for a combination once nothing still needs it (see run_paper_figures.py)."""
    for i in range(numReps):
        _rawDataCache.pop((basePath, i), None)


def _evaluateMetricSeriesCached(basePath, indices, metric, evalInterval, switchType=None, switchTypeOptions=None):
    """
    Same contract as ServiceNeighbourSelectionEval.evaluateMetricSeries, but reads through
    _loadRunDataCached instead of always hitting disk, and feeds the already-loaded data into
    EvaluatorMultiAvgComp directly (its simulationData= path) instead of its from_csv=True path.
    """
    switchTypes = [switchType] if switchType is not None else []
    modelParamsList, simDataList, switchValsForKey = [], [], []
    for i in indices:
        loaded = _loadRunDataCached(basePath, i, switchTypes)
        if loaded is None:
            continue
        params, simData, switchVals = loaded
        modelParamsList.append(params)
        simDataList.append(simData)
        if switchType is not None:
            switchValsForKey.append(switchVals[switchType.switchTypeValueKey])
    if not modelParamsList:
        return None

    switchTypeValuesArg = [{switchType.switchTypeValueKey: switchValsForKey}] if switchType is not None else None
    evaluator = EvaluatorMultiComp.EvaluatorMultiAvgComp(
        metric=metric, modelParams=[modelParamsList], simulationData=[simDataList],
        evaluationTimestepInterval=evalInterval, threshold=ServiceNeighbourSelectionEval.CLUSTER_THRESHOLD,
        switchTypeValues=switchTypeValuesArg, switchType=switchType, switchTypeOptions=switchTypeOptions,
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


def evaluateCell(basePathOrdered, basePathRandom, numReps, metric, evalInterval, switchType=None, switchTypeOptions=None):
    """
    Evaluates one grid cell's two series (ordered start, disordered start) for the given metric.
    Returns a dict {"ordered": (t, mean, std) or None, "disordered": (t, mean, std) or None}. The
    ORDER metric reads the small precomputed globalOrder.csv directly (already cheap, no caching
    needed); other metrics go through the shared run-data cache (see _loadRunDataCached).
    """
    indices = list(range(numReps))
    results = {}
    for label, basePath in (("ordered", basePathOrdered), ("disordered", basePathRandom)):
        if metric == Metrics.ORDER:
            results[label] = ServiceNeighbourSelectionEval.loadGlobalOrderSeries(basePath, indices)
        else:
            results[label] = _evaluateMetricSeriesCached(
                basePath, indices, metric, evalInterval, switchType=switchType, switchTypeOptions=switchTypeOptions
            )
    return results


def cleanupBatch(runSpecs, numReps):
    """
    Deletes the raw per-run data files (main csv, globalOrder csv, modelParams csv) for a batch of
    combinations, once their figure(s) have been rendered successfully.
    """
    removed = 0
    for spec in runSpecs:
        for i in range(numReps):
            savePath = f"{spec['savePathBase']}_{i}"
            for suffix in ("", "_globalOrder", "_modelParams"):
                path = f"{savePath}{suffix}.csv"
                if os.path.exists(path):
                    os.remove(path)
                    removed += 1
    return removed
