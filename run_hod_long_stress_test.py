import argparse
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from model.VicsekIndividualsMultiSwitch import VicsekWithNeighbourSelection
from enums.EnumNeighbourSelectionMechanism import NeighbourSelectionMechanism

import services.ServicePreparation as ServicePreparation
import services.ServiceGeneral as ServiceGeneral

import paper_figures_specs as specs

"""
Stress-tests NeighbourSelectionMechanism.HIGHEST_ORIENTATION_DIFFERENCE at k=1 and k=5 for
10,000,000 timesteps by default - about 700x longer than anything else in the pipeline (the longest
regular runs are 15,000 steps) - to check the fixed model doesn't show the kind of rare, long-horizon
lock-in event found (and fixed) for nearest(1)/farthest(1) NSM-switching (a broadcasting bug in
model/VicsekIndividualsMultiSwitch.computeNewOrientations that only showed up after ~9000 steps in 1
of 6 replicates). HOD itself never switches here (no SwitchSummary involved), so that specific bug
doesn't apply, but this checks for any OTHER slow-onset issue at a much longer timescale.

4 runs total: k in {1, 5} x starting condition in {ordered, random}, 1 repetition each - no
switching/no event, so there's nothing to average over repetitions for; a single long run is what
actually tests the long-horizon question. Domain/density/radius/noise match the rest of the paper
figures (paper_figures_specs.DEFAULT_*).

Logs only every --log-interval steps (10,000 points per run by default) via the model's built-in
logPath/logInterval mechanism, and does NOT use returnHistories - holding 10,000,000 timesteps of
positions/orientations in memory would need tens of GB. Reads back the small *_globalOrder.csv log
files afterward to plot.

Runs the 4 combinations in parallel (one process each) since they're fully independent. At the default
tmax, each one is expected to take on the order of hours - this is a long background job, not
something to babysit. Use --tmax/--log-interval for a much shorter smoke test first (e.g.
--tmax 20000 --log-interval 200).
"""

DATA_ROOT = os.path.expanduser("~/paper_figures_hod_long_tmp")
OUTPUT_ROOT = "plots/paper_figures_hod_long"


def buildJob(k, startingCondition, tmax, logInterval, dataRoot):
    n = ServicePreparation.getNumberOfParticlesForConstantDensity(specs.DEFAULT_DENSITY, specs.DOMAIN_SIZE)
    noise = ServicePreparation.getNoiseAmplitudeValueForPercentage(specs.DEFAULT_NOISE_PCT)
    savePath = f"{dataRoot}/hod_long_k={k}_{startingCondition}"
    return dict(k=k, startingCondition=startingCondition, n=n, noise=noise, savePath=savePath,
                tmax=tmax, logInterval=logInterval)


def runJob(job):
    # re-seed both RNGs from OS entropy - worker processes are forked and would otherwise inherit
    # identical RNG state from the parent, correlating the 4 "independent" runs.
    np.random.seed(None)
    random.seed(None)

    n = job["n"]
    if job["startingCondition"] == "ordered":
        initialState = ServicePreparation.createOrderedInitialDistributionEquidistancedIndividual(
            None, specs.DOMAIN_SIZE, n)
    else:
        initialState = None

    simulator = VicsekWithNeighbourSelection(
        domainSize=specs.DOMAIN_SIZE, radius=specs.DEFAULT_RADIUS, noise=job["noise"], numberOfParticles=n,
        k=job["k"], neighbourSelectionMechanism=NeighbourSelectionMechanism.HIGHEST_ORIENTATION_DIFFERENCE,
        speed=specs.SPEED, degreesOfVision=specs.DEGREES_OF_VISION,
        logPath=job["savePath"], logInterval=job["logInterval"], returnHistories=False,
    )
    t0 = time.time()
    if initialState is not None:
        simulator.simulate(initialState=initialState, tmax=job["tmax"])
    else:
        simulator.simulate(tmax=job["tmax"])
    return job["savePath"], time.time() - t0


def plotResults(jobs, outputRoot):
    import services.ServicePaperFigureGrid as Grid
    import matplotlib.pyplot as plt
    import pandas as pd

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, k in zip(axes, (1, 5)):
        for startingCondition, colour in (("ordered", Grid.ORDERED_COLOUR), ("random", Grid.DISORDERED_COLOUR)):
            job = next(j for j in jobs if j["k"] == k and j["startingCondition"] == startingCondition)
            df = pd.read_csv(f"{job['savePath']}_globalOrder.csv")
            ax.plot(df["t"], df["globalOrder"], color=colour, label=f"{startingCondition} start", linewidth=0.8)
        ax.set_title(f"k={k}")
        ax.set_xlabel("timesteps")
        ax.set_ylim(0, 1.1)
    axes[0].set_ylabel("global order")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    outputPath = os.path.join(outputRoot, "hod_long_stress_test.png")
    fig.savefig(outputPath, dpi=150)
    plt.close(fig)
    ServiceGeneral.logWithTime(f"wrote {outputPath}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tmax", type=int, default=10_000_000)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    args = parser.parse_args()

    os.makedirs(args.data_root, exist_ok=True)
    os.makedirs(args.output_root, exist_ok=True)

    jobs = [buildJob(k, sc, args.tmax, args.log_interval, args.data_root) for k in (1, 5) for sc in ("ordered", "random")]

    ServiceGeneral.logWithTime(f"starting {len(jobs)} runs at tmax={args.tmax}, log_interval={args.log_interval} "
                                f"in {args.data_root} - this will take a long time at the default tmax")
    with ProcessPoolExecutor(max_workers=len(jobs)) as executor:
        futures = [executor.submit(runJob, job) for job in jobs]
        for future in as_completed(futures):
            savePath, elapsed = future.result()
            ServiceGeneral.logWithTime(f"completed {savePath} in {ServiceGeneral.formatTime(elapsed)}")

    ServiceGeneral.logWithTime("all runs complete, plotting...")
    plotResults(jobs, args.output_root)


if __name__ == "__main__":
    main()
