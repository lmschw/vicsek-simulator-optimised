import argparse
import os

import paper_figures_specs as specs
import run_paper_figures as MainPipeline

"""
Variant of run_paper_figures.py where an individual with no other neighbours in radius is BLOCKED
from switching strategy (updateIfNoNeighbours=False), instead of the default (updateIfNoNeighbours=
True, where an isolated individual's local order is computed from itself alone - always exactly 1.0 -
so it's always nudged towards the order-inducing switch value regardless of the swarm's actual state).

Only generates the subset of figures that involve switching at all (see
paper_figures_specs.isSwitchingFigure) - the other 23 figures (fig_2, fig_3, density_radius_*,
noise_elevation_*, etc.) never call getDecisions()/use a SwitchSummary, so this flag can't change
their output; regenerating them would just waste compute reproducing pixel-identical plots.

Uses its own scratch/output directories (not the main pipeline's), and its own path tag
("_blockiso") on every switching combination, so the two variants' data and manifests never collide
even if pointed at the same physical drive - safe to run alongside the main pipeline. Reuses the
exact same combination-registration code (paper_figures_specs.ComboRegistry) and orchestration
(run_paper_figures.run, including its cross-figure/cross-metric data cache - see
services/ServicePaperFigureRunner.py) as the main pipeline, just with updateIfNoNeighbours=False and
a figure filter restricted to the switching subset.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=os.path.expanduser("~/paper_figures_no_neighbour_switch_tmp"))
    parser.add_argument("--output-root", default="plots/paper_figures_no_neighbour_switch")
    parser.add_argument("--num-reps", type=int, default=50)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    parser.add_argument("--figures", nargs="+", default=None,
                         help="only process these figure names (default: all 49 switching figures)")
    parser.add_argument("--list", action="store_true", help="list figure names and their manifest status, then exit")
    args = parser.parse_args()

    figureFilter = args.figures if args.figures is not None else specs.isSwitchingFigure

    MainPipeline.run(args.data_root, args.output_root, args.num_reps, args.workers, figureFilter, args.list,
                      updateIfNoNeighbours=False)


if __name__ == "__main__":
    main()
