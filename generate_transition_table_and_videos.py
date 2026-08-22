import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio_ffmpeg
plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

import numpy as np

import paper_figures_specs as specs
import services.ServicePaperFigureRunner as Runner
import services.ServiceSavedModel as ServiceSavedModel
import services.ServiceSwitchAnalysis as ServiceSwitchAnalysis
import services.ServicePaperFigureGrid as Grid
import services.ServiceGeneral as ServiceGeneral
import animator.AnimatorMatplotlib as AnimatorMatplotlib
from animator.Animator2D import Animator2D

"""
Generates two extra deliverables beyond the main paper_figures pipeline, reusing its combination
definitions (paper_figures_specs.py) and execution engine (services/ServicePaperFigureRunner.py):

1) A LaTeX table of min/avg/max transition duration (event exposure -> switch), measured in whichever
   direction is actually physically meaningful for each event - see EVENT_TRANSITION_DIRECTION and
   fig_4_switching_order: "distant" pulls a disordered swarm into order (s_d -> s_o, measured on the
   disordered-start runs), while "predator" and "random" push an ordered swarm into disorder (s_o ->
   s_d, measured on the ordered-start runs) - the other starting condition barely responds to that
   event type at all, so measuring it there would mostly capture noise rather than the event's actual
   effect. Pooled across TABLE_REPS repetitions per combination.
2) mp4 videos (ordered start + disordered start) for:
   - the 4 reduced neighbour selection mechanisms x k=1,5 x 3 events, no switching (matches
     fig_3_nosw_1ev_order's combinations)
   - the same 15 switching combinations x 3 events used for the table, coloured by each particle's
     current switch value (order-value colour vs disorder-value colour, matching the paper figures'
     palette)

Uses its own scratch data directory (not the main pipeline's ~/paper_figures_tmp/) so it can run
safely alongside an in-progress run of run_paper_figures.py without touching its files.
"""

DATA_ROOT = os.path.expanduser("~/paper_figures_extra_tmp")
OUTPUT_ROOT = "plots/paper_figures_extra"
VIDEO_DIR = os.path.join(OUTPUT_ROOT, "videos")
TABLE_REPS = 20
WORKERS = max(1, os.cpu_count() - 2)

# Animator.prepareAnimation()'s frameInterval (ms between frames) doubles as the saved video's frame
# pacing: Animation.save() derives fps as 1000/interval when no explicit fps is given (and
# Animator.saveAnimation()'s own fpsVar parameter is unfortunately never actually passed through to
# save()). Left at the default 10ms interval, our 151-frame (tmax=15000, logged every 100 steps)
# videos come out as 100fps, 1.5s long. 100ms (10fps, ~15s per video) is a lot more watchable.
VIDEO_FRAME_INTERVAL_MS = 100

EVENT_ORIGIN = (specs.DOMAIN_SIZE[0] / 2, specs.DOMAIN_SIZE[1] / 2)
LOG_INTERVAL = 100  # matches comboNoSwitch/comboKSwitch/comboNsmSwitch's logInterval for event runs
EVENT_START_INDEX = specs.EVENT_START // LOG_INTERVAL
SENTINEL_DURATION = specs.TMAX_LOCAL  # for particles that switch without ever being exposed

# Which starting condition to read the transition from, and which switchTypeOptions slot ([0] = order
# value, [1] = disorder value) counts as "switched", per event - the direction that's actually
# physically meaningful for that event type (see the module docstring).
EVENT_TRANSITION_DIRECTION = {
    "distant": dict(startKey="randomPath", targetIndex=0, directionLabel=r"$s_d \to s_o$"),
    "predator": dict(startKey="orderedPath", targetIndex=1, directionLabel=r"$s_o \to s_d$"),
    "random": dict(startKey="orderedPath", targetIndex=1, directionLabel=r"$s_o \to s_d$"),
}


# ------------------------------------------------------------------ combination definitions --

def buildCombos():
    reg = specs.ComboRegistry(DATA_ROOT, 1)

    switchingEntries = []
    for combo in specs.SWITCHING_COMBOS:
        for eventEffect, eventLabel in specs.EVENT_EFFECTS:
            ordered, random, switchType, switchTypeOptions, _invert = specs._switchingCombo(
                reg, combo, specs.DEFAULT_DENSITY, specs.DEFAULT_RADIUS, specs.DEFAULT_NOISE_PCT, eventEffect=eventEffect)
            switchingEntries.append(dict(
                comboKey=combo["key"], rowLabel=combo["rowLabel"], eventLabel=eventLabel,
                orderedPath=ordered, randomPath=random, switchType=switchType, switchTypeOptions=switchTypeOptions,
            ))

    noswEntries = []
    for nsm, k, mechLabel in specs.NOSW_ROWS:
        for eventEffect, eventLabel in specs.EVENT_EFFECTS:
            ordered, random = reg.comboNoSwitch(specs.DEFAULT_DENSITY, specs.DEFAULT_RADIUS, specs.DEFAULT_NOISE_PCT,
                                                 nsm, k, eventEffect=eventEffect)
            noswEntries.append(dict(mechLabel=mechLabel, eventLabel=eventLabel, orderedPath=ordered, randomPath=random))

    return reg, switchingEntries, noswEntries


def collectSpecs(reg, entries):
    paths = set()
    for e in entries:
        paths.add(e["orderedPath"])
        paths.add(e["randomPath"])
    return [reg.runSpecs[p] for p in paths]


def generateData(reg, switchingEntries, noswEntries, workers):
    seedLog = os.path.join(DATA_ROOT, "seeds.csv")
    switchingSpecs = collectSpecs(reg, switchingEntries)
    noswSpecs = collectSpecs(reg, noswEntries)

    ServiceGeneral.logWithTime(f"generating {len(switchingSpecs)} switching combinations x {TABLE_REPS} reps "
                                f"(needed for the table + part-2 videos)...")
    t0 = time.time()
    Runner.runBatch(switchingSpecs, seedLog, TABLE_REPS, workers)
    ServiceGeneral.logWithTime(f"  done ({ServiceGeneral.formatTime(time.time() - t0)})")

    ServiceGeneral.logWithTime(f"generating {len(noswSpecs)} no-switching combinations x 1 rep (part-1 videos only)...")
    t0 = time.time()
    Runner.runBatch(noswSpecs, seedLog, 1, workers)
    ServiceGeneral.logWithTime(f"  done ({ServiceGeneral.formatTime(time.time() - t0)})")


# ------------------------------------------------------------------ transition duration table

def computeTransitionDurations(positions, switchValues, targetValue):
    """
    For every particle that switches into targetValue at some point after the event starts, records
    how long that took since it was first exposed to the event (within event radius of the event's
    origin point) - or SENTINEL_DURATION if it switched without ever being exposed. Operates on the
    saved (subsampled, every LOG_INTERVAL steps) arrays, so durations are scaled back to real
    timesteps.
    """
    exposedAt = {}
    durations = []
    for t in range(EVENT_START_INDEX, len(positions)):
        affected = ServiceSwitchAnalysis.get_affected_individuals(
            positions=positions[t], event_origin_point=EVENT_ORIGIN, event_radius=specs.DEFAULT_RADIUS,
            domain_size=specs.DOMAIN_SIZE)
        for i in affected:
            exposedAt[i] = t
        if t > EVENT_START_INDEX:
            switched = np.argwhere((switchValues[t - 1] != targetValue) & (switchValues[t] == targetValue)).flatten()
            for i in switched:
                if i in exposedAt:
                    durations.append((t - exposedAt[i]) * LOG_INTERVAL)
                else:
                    durations.append(SENTINEL_DURATION)
    return durations


def buildTable(switchingEntries):
    rows = []
    for entry in switchingEntries:
        direction = EVENT_TRANSITION_DIRECTION[entry["eventLabel"]]
        allDurations = []
        for i in range(TABLE_REPS):
            path = entry[direction["startKey"]]
            try:
                _params, simData, switchVals = ServiceSavedModel.loadModelFromCsv(
                    filepathData=f"{path}_{i}.csv", filePathModelParams=f"{path}_{i}_modelParams.csv",
                    switchTypes=[entry["switchType"]])
            except Exception as e:
                ServiceGeneral.logWithTime(f"  WARNING: skipping {path}_{i}: {e}")
                continue
            _times, positions, _orientations = simData
            switchArr = np.array(switchVals[entry["switchType"].switchTypeValueKey])
            targetValue = entry["switchTypeOptions"][direction["targetIndex"]]
            allDurations.extend(computeTransitionDurations(positions, switchArr, targetValue))

        ServiceGeneral.logWithTime(f"  {entry['rowLabel']} / {entry['eventLabel']} ({direction['directionLabel']}): "
                                    f"{len(allDurations)} transitions")
        if allDurations:
            arr = np.array(allDurations)
            rows.append(dict(rowLabel=entry["rowLabel"], eventLabel=entry["eventLabel"],
                              directionLabel=direction["directionLabel"],
                              minD=arr.min(), avgD=arr.mean(), maxD=arr.max(), n=len(arr)))
        else:
            rows.append(dict(rowLabel=entry["rowLabel"], eventLabel=entry["eventLabel"],
                              directionLabel=direction["directionLabel"],
                              minD=None, avgD=None, maxD=None, n=0))
    return rows


def writeLatexTable(rows, outputPath):
    lines = [
        "% requires \\usepackage{booktabs} in the preamble",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\begin{tabular}{lllrrr}",
        r"\toprule",
        r"Combination & Event & Direction & Min & Avg & Max \\",
        r"\midrule",
    ]
    for row in rows:
        label = row["rowLabel"].replace("_", r"\_")
        if row["n"] == 0:
            lines.append(f"{label} & {row['eventLabel']} & {row['directionLabel']} & -- & -- & -- \\\\")
        else:
            lines.append(f"{label} & {row['eventLabel']} & {row['directionLabel']} & "
                          f"{row['minD']:.0f} & {row['avgD']:.1f} & {row['maxD']:.0f} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Transition duration (in timesteps) from event exposure to switching, measured in "
        r"the direction physically driven by that event: $s_d \to s_o$ for distant (measured on "
        r"disordered-start runs), $s_o \to s_d$ for predator and random (measured on ordered-start "
        f"runs) - see EVENT_TRANSITION_DIRECTION. Pooled across {TABLE_REPS} repetitions per "
        f"combination. Particles that switched without ever being exposed to the event are counted "
        f"with a duration of {SENTINEL_DURATION} (the full simulation length).}}",
        r"\label{tab:transition-durations}",
        r"\end{table}",
    ]
    os.makedirs(os.path.dirname(outputPath), exist_ok=True)
    with open(outputPath, "w") as f:
        f.write("\n".join(lines) + "\n")


# ------------------------------------------------------------------ videos ---------------------

def renderVideo(basePath, repIndex, outputPath, switchType=None, switchTypeOptions=None):
    if switchType is not None:
        params, simData, switchVals = ServiceSavedModel.loadModelFromCsv(
            filepathData=f"{basePath}_{repIndex}.csv", filePathModelParams=f"{basePath}_{repIndex}_modelParams.csv",
            switchTypes=[switchType])
        switchArr = np.array(switchVals[switchType.switchTypeValueKey])
        orderValue = switchTypeOptions[0]
        colours = np.where(switchArr == orderValue, Grid.ORDERED_COLOUR, Grid.DISORDERED_COLOUR).tolist()
    else:
        params, simData = ServiceSavedModel.loadModelFromCsv(
            filepathData=f"{basePath}_{repIndex}.csv", filePathModelParams=f"{basePath}_{repIndex}_modelParams.csv")
        colours = None

    times, _positions, _orientations = simData
    domainSize = tuple(params["domainSize"]) + (100,)
    animator = AnimatorMatplotlib.MatplotlibAnimator(simData, domainSize, colours=colours, showRadiusForExample=False)
    preparedAnimator = animator.prepare(Animator2D(params), frames=len(times), frameInterval=VIDEO_FRAME_INTERVAL_MS)
    os.makedirs(os.path.dirname(outputPath), exist_ok=True)
    preparedAnimator.saveAnimation(outputPath)
    plt.close("all")


def renderVideosForEntry(entry, outputSubdir, switchType=None, switchTypeOptions=None):
    for label, path in (("ordered", entry["orderedPath"]), ("random", entry["randomPath"])):
        outputPath = os.path.join(outputSubdir, f"{label}.mp4")
        renderVideo(path, 0, outputPath, switchType=switchType, switchTypeOptions=switchTypeOptions)


def renderAllVideos(switchingEntries, noswEntries):
    for entry in noswEntries:
        subdir = os.path.join(VIDEO_DIR, "no_switching", f"{entry['mechLabel']}_{entry['eventLabel']}")
        t0 = time.time()
        renderVideosForEntry(entry, subdir)
        ServiceGeneral.logWithTime(f"  video {entry['mechLabel']} / {entry['eventLabel']} "
                                    f"({ServiceGeneral.formatTime(time.time() - t0)})")

    for entry in switchingEntries:
        subdir = os.path.join(VIDEO_DIR, "switching", f"{entry['comboKey']}_{entry['eventLabel']}")
        t0 = time.time()
        renderVideosForEntry(entry, subdir, switchType=entry["switchType"], switchTypeOptions=entry["switchTypeOptions"])
        ServiceGeneral.logWithTime(f"  video {entry['comboKey']} / {entry['eventLabel']} "
                                    f"({ServiceGeneral.formatTime(time.time() - t0)})")


# ------------------------------------------------------------------ CLI ------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate the transition-duration table and the requested videos.")
    parser.add_argument("--skip-table", action="store_true")
    parser.add_argument("--skip-videos", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N entries per group (for testing)")
    parser.add_argument("--workers", type=int, default=WORKERS, help=f"parallel worker processes (default {WORKERS})")
    args = parser.parse_args()

    reg, switchingEntries, noswEntries = buildCombos()
    if args.limit is not None:
        switchingEntries = switchingEntries[:args.limit]
        noswEntries = noswEntries[:args.limit]

    generateData(reg, switchingEntries, noswEntries, args.workers)

    if not args.skip_table:
        ServiceGeneral.logWithTime("computing transition durations...")
        rows = buildTable(switchingEntries)
        tablePath = os.path.join(OUTPUT_ROOT, "transition_durations.tex")
        writeLatexTable(rows, tablePath)
        ServiceGeneral.logWithTime(f"wrote {tablePath}")

    if not args.skip_videos:
        ServiceGeneral.logWithTime("rendering videos...")
        renderAllVideos(switchingEntries, noswEntries)

    ServiceGeneral.logWithTime("done")


if __name__ == "__main__":
    main()
