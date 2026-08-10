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
import services.ServiceNeighbourSelectionEval as ServiceNeighbourSelectionEval

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


def evaluateCell(basePathOrdered, basePathRandom, numReps, metric, evalInterval, switchType=None, switchTypeOptions=None):
    """
    Evaluates one grid cell's two series (ordered start, disordered start) for the given metric,
    reusing the existing evaluation code in ServiceNeighbourSelectionEval.py. Returns a dict
    {"ordered": (t, mean, std) or None, "disordered": (t, mean, std) or None}.
    """
    indices = list(range(numReps))
    results = {}
    for label, basePath in (("ordered", basePathOrdered), ("disordered", basePathRandom)):
        if metric == Metrics.ORDER:
            results[label] = ServiceNeighbourSelectionEval.loadGlobalOrderSeries(basePath, indices)
        else:
            results[label] = ServiceNeighbourSelectionEval.evaluateMetricSeries(
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
