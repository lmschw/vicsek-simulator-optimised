import argparse
import os

import paper_figures_specs as specs
import run_paper_figures as MainPipeline
import run_extended_switching_combos as ExtendedSwitching
import services.ServiceGeneral as ServiceGeneral

"""
Regenerates fig_3_nosw_1ev_order's and fig_4_switching_order's cluster-count companions
(nosw_1ev_clusters, switching_clustering) with the corrected cluster threshold (see
services/ServiceNeighbourSelectionEval.CLUSTER_THRESHOLD - was 0.01, too tight to ever merge a
genuinely ordered swarm into one cluster; now 0.1), and adds a "clusters / agents" companion plot for
each (nosw_1ev_clusters_per_agent, switching_clustering_per_agent) - same underlying data, normalised
by particle count.

Also needs a full resimulation regardless of the threshold fix: fig_3/fig_4's combinations changed
since the original pipeline run in more fundamental ways too (the minimum-image-convention fix, the
switching-hysteresis-formula fix, and the NSM-switch broadcasting-bug fix all predate this), and
NOSW_ROWS/SWITCHING_COMBOS themselves were edited (hod/farthest(5) dropped from NOSW_ROWS) since that
run - so reusing any old data would be wrong on several counts, not just the threshold.

Writes to its own data-root/output-root, deliberately not the main pipeline's - reusing that would
either silently skip regeneration (keeping stale results) or overwrite files other in-progress figures
still reference.
"""

DEFAULT_DATA_ROOT = os.path.expanduser("~/paper_figures_fig3_fig4_clustering_tmp")
DEFAULT_OUTPUT_ROOT = "plots/paper_figures_fig3_fig4_clustering"

TARGET_FIGURES = ["nosw_1ev_clusters", "switching_clustering"]


def renderClustersPerAgentCompanions(dataRoot, outputRoot, numReps):
    runSpecs, figures = specs.buildAll(dataRoot, numReps)
    n = specs._n(specs.DEFAULT_DENSITY)
    for name in TARGET_FIGURES:
        figureSpec = figures[name]
        ExtendedSwitching.renderClustersPerAgent(f"{name}_per_agent", figureSpec, numReps, n, runSpecs, outputRoot)
        ServiceGeneral.logWithTime(f"  wrote {name}_per_agent")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--num-reps", type=int, default=50,
                         help="should match the run you're comparing against (default 50, matching "
                              "plots/paper_figures_d0.06_50reps)")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    parser.add_argument("--list", action="store_true",
                         help="list what would be regenerated and exit, without running anything")
    args = parser.parse_args()

    if args.list:
        print("figures (corrected cluster threshold):")
        for name in TARGET_FIGURES:
            print(f"  {name}")
        print("plus their clusters/agents companions:")
        for name in TARGET_FIGURES:
            print(f"  {name}_per_agent")
        return

    ServiceGeneral.logWithTime(f"regenerating {TARGET_FIGURES} with the corrected cluster threshold...")
    MainPipeline.run(args.data_root, args.output_root, args.num_reps, args.workers, TARGET_FIGURES, listOnly=False)

    ServiceGeneral.logWithTime("rendering clusters/agents companions...")
    renderClustersPerAgentCompanions(args.data_root, args.output_root, args.num_reps)

    ServiceGeneral.logWithTime("done")


if __name__ == "__main__":
    main()
