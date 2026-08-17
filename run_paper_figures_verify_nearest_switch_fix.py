import argparse
import os

import paper_figures_specs as specs
import run_paper_figures as MainPipeline

"""
Verification-only variant of run_paper_figures.py: regenerates just the figures that (a) involve
switching (built from comboKSwitch/comboNsmSwitch, i.e. paper_figures_specs.isSwitchingFigure) and
(b) include NeighbourSelectionMechanism.NEAREST at k=1 as one of the two switch values - i.e. row
combos "a" (<nearest(1), nearest(5)>, K-switch) and "d" (<nearest(1), farthest(1)>, NSM-switch) from
paper_figures_specs.SWITCHING_COMBOS.

Exists to isolate and verify two recent fixes to the underlying model before rerunning the whole
pipeline:
    - restoring the minimum-image convention in services/ServiceVicsekHelper.getDifferences
    - correcting the switching hysteresis formula in
      model/VicsekIndividualsMultiSwitch.getDecisions to match the reference simulator
Both bugs made nearest(1)-in-a-switching-context lock into permanent order instead of sustaining
disorder; this script's target figures are exactly the ones that would show that regression.

Note: the four figures that stack all five row-combos into one image (fig_4_switching_order,
switching_clustering, switching_percentage, switching_no_ev_order) necessarily regenerate combos b/c/
e (lod/hod switching) too, since a grid image can't mix data from two different code versions across
its rows - but the hysteresis fix affects those combos as well, so this is correct, not wasted work.
The per-combo sweep figures (event_duration_*_b/c/e, window_size_*_b/c/e, thresholds_*_b/c/e) are
untouched, since they don't reference nearest(1) at all.

Also includes fig_3_nosw_1ev_order and nosw_1ev_clusters even though neither uses a SwitchSummary (so
the hysteresis fix doesn't apply to them): both reference nearest(1) directly via NOSW_ROWS and had
their row set edited in the same session as these fixes, so they're in scope for this verification
pass too, on the MIC fix and the row-set change alike.

Writes to its own data-root/output-root by default, deliberately NOT the ones used by the original
(pre-fix) run - reusing those could silently skip regeneration and keep the stale, buggy results,
since run_paper_figures.py resumes from whatever raw data/figures are already on disk. Seeds are still
derived deterministically from each run's save path (see services/ServicePaperFigureRunner.deriveSeed),
so runs here use the exact same seeds as the equivalent combos in the original run - the two are
directly comparable, isolating the code change as the only difference.
"""

EXTRA_NEAREST_FIGURES = {"fig_3_nosw_1ev_order", "nosw_1ev_clusters"}


def isNearestSwitchingFigure(name, figures):
    if name in EXTRA_NEAREST_FIGURES:
        return True
    if not specs.isSwitchingFigure(name):
        return False
    figureSpec = figures[name]
    cells = figureSpec["cells"].values() if figureSpec["kind"] == "grid" else figureSpec["cells"]
    for cell in cells:
        for path in (cell["orderedPath"], cell["randomPath"]):
            # "_nsm=N_" -> comboKSwitch with neighbourSelectionMechanism=NEAREST (combo "a").
            # "_nsmCombo=F-N_" -> comboNsmSwitch with NSM_COMBO_NEAREST_FARTHEST (combo "d"), where
            # NEAREST is always the disorder value, i.e. always used at k=1 there.
            if "_nsm=N_" in path or "_nsmCombo=F-N_" in path:
                return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default=os.path.expanduser("~/paper_figures_verify_nearest_switch_tmp"),
                         help="scratch directory for raw simulation data - kept separate from the main "
                              "pipeline's so stale pre-fix data can never be silently reused")
    parser.add_argument("--output-root", default="plots/paper_figures_verify_nearest_switch",
                         help="where the rendered figures/manifest/seed log go - kept separate from the "
                              "main pipeline's output so you can compare old vs. new side by side")
    parser.add_argument("--num-reps", type=int, default=50,
                         help="should match the run you're verifying against (default 50, matching "
                              "plots/paper_figures_d0.06_50reps)")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    parser.add_argument("--list", action="store_true",
                         help="list the targeted figure names and exit, without running anything")
    args = parser.parse_args()

    _runSpecs, figures = specs.buildAll(args.data_root, args.num_reps)
    targetNames = sorted(n for n in figures if isNearestSwitchingFigure(n, figures))

    print(f"{len(targetNames)} figures involve switching + nearest(1):")
    for name in targetNames:
        print(f"  {name}")

    if args.list:
        return

    MainPipeline.run(args.data_root, args.output_root, args.num_reps, args.workers, targetNames, listOnly=False)


if __name__ == "__main__":
    main()
