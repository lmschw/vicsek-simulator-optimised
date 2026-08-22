import argparse
import os

import paper_figures_specs as specs
import run_paper_figures as MainPipeline
import generate_transition_table_and_videos as TransitionTable
import services.ServicePaperFigureRunner as Runner
import services.ServiceGeneral as ServiceGeneral

"""
Single entry point for the three deliverables that got a presentation-only fix (no underlying model
logic changed, so nothing else in the pipeline needs to be touched):
  1) density_radius_lod_order_k1 / density_radius_lod_order_k5
  2) switching_no_ev_order_a..e (split from the old single 10-line switching_no_ev_order panel)
  3) the transition-duration table (transition_durations.tex), now measuring the direction each event
     actually drives (see generate_transition_table_and_videos.EVENT_TRANSITION_DIRECTION)

Does not render the transition-duration videos - those are unaffected by any of the above and don't
need regenerating.

Parts 1-2 go through the standard run_paper_figures.py figure pipeline, in their own fresh
data-root/output-root (so nothing stale from other runs gets silently reused). Part 3 reuses
generate_transition_table_and_videos.py's own canonical data-root/output-root directly - it was
already cleared of stale pre-fix data earlier in this session, so no separate copy is needed; the
underlying simulation combinations (with events) only exist there anyway.
"""

FIGURE_PIPELINE_FIGURES = ["density_radius_lod_order_k1", "density_radius_lod_order_k5"] + \
    [f"switching_no_ev_order_{key}" for key in "abcde"]

DEFAULT_DATA_ROOT = os.path.expanduser("~/paper_figures_selected_tmp")
DEFAULT_OUTPUT_ROOT = "plots/paper_figures_selected"


def regenerateFigures(dataRoot, outputRoot, numReps, workers):
    ServiceGeneral.logWithTime(f"regenerating {len(FIGURE_PIPELINE_FIGURES)} figures: {FIGURE_PIPELINE_FIGURES}")
    MainPipeline.run(dataRoot, outputRoot, numReps, workers, FIGURE_PIPELINE_FIGURES, listOnly=False)


def regenerateTransitionTable(workers):
    ServiceGeneral.logWithTime("regenerating transition-duration table (no videos)...")
    reg, switchingEntries, _noswEntries = TransitionTable.buildCombos()
    seedLog = os.path.join(TransitionTable.DATA_ROOT, "seeds.csv")
    # only the switching combinations are needed for the table - the no-switching combinations
    # (noswEntries) exist purely for the part-1 videos, which this script doesn't render.
    switchingSpecs = TransitionTable.collectSpecs(reg, switchingEntries)
    Runner.runBatch(switchingSpecs, seedLog, TransitionTable.TABLE_REPS, workers)
    rows = TransitionTable.buildTable(switchingEntries)
    tablePath = os.path.join(TransitionTable.OUTPUT_ROOT, "transition_durations.tex")
    TransitionTable.writeLatexTable(rows, tablePath)
    ServiceGeneral.logWithTime(f"wrote {tablePath}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT,
                         help="for density_radius_lod / switching_no_ev_order (parts 1-2)")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT,
                         help="for density_radius_lod / switching_no_ev_order (parts 1-2)")
    parser.add_argument("--num-reps", type=int, default=50,
                         help="should match the run you're comparing against (default 50, matching "
                              "plots/paper_figures_d0.06_50reps)")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    parser.add_argument("--skip-figures", action="store_true",
                         help="skip parts 1-2 (density_radius_lod / switching_no_ev_order)")
    parser.add_argument("--skip-table", action="store_true", help="skip part 3 (transition-duration table)")
    parser.add_argument("--list", action="store_true",
                         help="list what would be regenerated and exit, without running anything")
    args = parser.parse_args()

    if args.list:
        print(f"figures (parts 1-2, {len(FIGURE_PIPELINE_FIGURES)}):")
        for name in FIGURE_PIPELINE_FIGURES:
            print(f"  {name}")
        print("plus: transition-duration table (part 3)")
        return

    if not args.skip_figures:
        regenerateFigures(args.data_root, args.output_root, args.num_reps, args.workers)
    if not args.skip_table:
        regenerateTransitionTable(args.workers)

    ServiceGeneral.logWithTime("done")


if __name__ == "__main__":
    main()
