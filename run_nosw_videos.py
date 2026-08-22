import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio_ffmpeg
plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

from enums.EnumNeighbourSelectionMechanism import NeighbourSelectionMechanism

import paper_figures_specs as specs
import services.ServicePaperFigureRunner as Runner
import services.ServiceGeneral as ServiceGeneral
import generate_transition_table_and_videos as TransitionTable

"""
Renders one ordered-start + one disordered-start video for every neighbour selection mechanism (all
six: ALL, RANDOM, NEAREST, FARTHEST, LEAST_ORIENTATION_DIFFERENCE, HIGHEST_ORIENTATION_DIFFERENCE) at
k=1 and k=5 - no switching in either case, but two event conditions:
    - no event: the plain baseline behaviour (tmax=paper_figures_specs.TMAX_GLOBAL=3000)
    - one event (all three effects: distant/predator/random, each starting at
      paper_figures_specs.EVENT_START): tmax=paper_figures_specs.TMAX_LOCAL=15000
Density/radius/noise match paper_figures_specs.DEFAULT_*. 6 mechanisms x 2 k x (1 no-event + 3 event
conditions) = 48 combinations, 96 videos total.

Reuses generate_transition_table_and_videos.py's video-rendering machinery (renderVideosForEntry, same
ffmpeg/animator setup, same VIDEO_FRAME_INTERVAL_MS pacing) but not its combination definitions, since
those (paper_figures_specs.NOSW_ROWS) only cover 5 of the 6 mechanisms.

Runs its own dedicated logging (coarser than the main pipeline's for the no-event case; matching it for
the with-event case - see NOEVENT_LOG_INTERVAL/EVENT_LOG_INTERVAL) under its own save-path prefix
("video_nosw_..."), deliberately distinct from the main pipeline's "nosw_..." paths. The main pipeline
logs every single step for its own no-event combinations (logInterval=1), and reusing that path with a
different logInterval here would either wrongly appear "already complete" to this script (skipping
video generation) or wrongly appear "already complete" to the main pipeline later (skipping
full-resolution data it still needs for its own plots) - see
services/ServicePaperFigureRunner.isRunComplete, which only checks the final logged timestep, not the
logging interval used to get there.

Uses its own scratch data directory (not the main pipeline's ~/paper_figures_tmp/) so it can run
safely alongside an in-progress run of run_paper_figures.py without touching its files.
"""

DATA_ROOT = os.path.expanduser("~/paper_figures_nosw_videos_tmp")
OUTPUT_ROOT = "plots/paper_figures_nosw_videos"
NOEVENT_LOG_INTERVAL = 20  # 3000 / 20 = 150 frames, matching the ~15s/150-frame pacing used elsewhere
EVENT_LOG_INTERVAL = 100  # 15000 / 100 = 150 frames, matches the main pipeline's own event runs
WORKERS = max(1, os.cpu_count() - 2)

ALL_MECHANISMS = [
    (NeighbourSelectionMechanism.ALL, "all"),
    (NeighbourSelectionMechanism.RANDOM, "random"),
    (NeighbourSelectionMechanism.NEAREST, "nearest"),
    (NeighbourSelectionMechanism.FARTHEST, "farthest"),
    (NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE, "lod"),
    (NeighbourSelectionMechanism.HIGHEST_ORIENTATION_DIFFERENCE, "hod"),
]


def buildCombos(dataRoot, noEventLogInterval, eventLogInterval):
    reg = specs.ComboRegistry(dataRoot, 1)
    n = specs._n(specs.DEFAULT_DENSITY)
    noise = specs._noise(specs.DEFAULT_NOISE_PCT)

    entries = []
    for nsm, mechLabel in ALL_MECHANISMS:
        for k in (1, 5):
            baseTag = (f"video_nosw_d={specs.DEFAULT_DENSITY}_r={specs.DEFAULT_RADIUS}_"
                       f"noise={specs.DEFAULT_NOISE_PCT}_nsm={nsm.value}_k={k}")

            # no event
            def noEventKwargsFn(sc, nsm=nsm, k=k):
                return dict(domainSize=specs.DOMAIN_SIZE, radius=specs.DEFAULT_RADIUS, noise=noise,
                            numberOfParticles=n, k=k, neighbourSelectionMechanism=nsm,
                            speed=specs.SPEED, degreesOfVision=specs.DEGREES_OF_VISION)

            ordered, random = reg.register(f"{baseTag}_noev", specs.TMAX_GLOBAL, n, noEventLogInterval, noEventKwargsFn)
            entries.append(dict(mechLabel=mechLabel, k=k, eventLabel=None, orderedPath=ordered, randomPath=random))

            # one event, each of the three effects
            for eventEffect, eventLabel in specs.EVENT_EFFECTS:
                def eventKwargsFn(sc, nsm=nsm, k=k, eventEffect=eventEffect):
                    kwargs = dict(domainSize=specs.DOMAIN_SIZE, radius=specs.DEFAULT_RADIUS, noise=noise,
                                  numberOfParticles=n, k=k, neighbourSelectionMechanism=nsm,
                                  speed=specs.SPEED, degreesOfVision=specs.DEGREES_OF_VISION,
                                  events=[specs._event(eventEffect, specs.DEFAULT_RADIUS, specs.DEFAULT_EVENT_DURATION)])
                    return kwargs

                base = f"{baseTag}_ee={eventEffect.val}_dur={specs.DEFAULT_EVENT_DURATION}"
                ordered, random = reg.register(base, specs.TMAX_LOCAL, n, eventLogInterval, eventKwargsFn)
                entries.append(dict(mechLabel=mechLabel, k=k, eventLabel=eventLabel, orderedPath=ordered, randomPath=random))

    return reg, entries


def generateData(reg, entries, dataRoot, workers):
    seedLog = os.path.join(dataRoot, "seeds.csv")
    specsList = TransitionTable.collectSpecs(reg, entries)
    ServiceGeneral.logWithTime(f"generating {len(specsList)} combinations x 1 rep...")
    t0 = time.time()
    Runner.runBatch(specsList, seedLog, 1, workers)
    ServiceGeneral.logWithTime(f"  done ({ServiceGeneral.formatTime(time.time() - t0)})")


def entryTag(entry):
    tag = f"{entry['mechLabel']}_k{entry['k']}"
    return tag if entry["eventLabel"] is None else f"{tag}_{entry['eventLabel']}"


def renderAllVideos(entries, outputRoot):
    for entry in entries:
        subdir = os.path.join(outputRoot, entryTag(entry))
        t0 = time.time()
        TransitionTable.renderVideosForEntry(entry, subdir)
        ServiceGeneral.logWithTime(f"  video {entryTag(entry)} ({ServiceGeneral.formatTime(time.time() - t0)})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    parser.add_argument("--noevent-log-interval", type=int, default=NOEVENT_LOG_INTERVAL)
    parser.add_argument("--event-log-interval", type=int, default=EVENT_LOG_INTERVAL)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--skip-data", action="store_true", help="skip simulation, just (re-)render videos from existing data")
    parser.add_argument("--list", action="store_true", help="list the targeted combinations and exit, without running anything")
    args = parser.parse_args()

    reg, entries = buildCombos(args.data_root, args.noevent_log_interval, args.event_log_interval)

    if args.list:
        print(f"{len(entries)} combinations (mechanism x k x event condition), each with an ordered + disordered video:")
        for e in entries:
            print(f"  {entryTag(e)}")
        return

    os.makedirs(args.data_root, exist_ok=True)
    os.makedirs(args.output_root, exist_ok=True)

    if not args.skip_data:
        generateData(reg, entries, args.data_root, args.workers)

    ServiceGeneral.logWithTime("rendering videos...")
    renderAllVideos(entries, args.output_root)
    ServiceGeneral.logWithTime("done")


if __name__ == "__main__":
    main()
