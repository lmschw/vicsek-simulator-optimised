import argparse
import os

import paper_figures_specs as specs
import run_paper_figures as MainPipeline

"""
Variant of run_paper_figures.py that regenerates every figure referencing
NeighbourSelectionMechanism.HIGHEST_ORIENTATION_DIFFERENCE (HOD) in any cell - unlike
run_paper_figures_verify_nearest_switch_fix.py, this is not restricted to switching figures, since
HOD also appears in plain comboNoSwitch figures (fig_2_global_order_k1/k5, noise_elevation_hod,
density_radius_hod_order_k1/k5). Found by inspecting each figure's actual run paths for HOD's
mechanism tag rather than hardcoding figure names, so it stays correct if the spec file changes.

fig_3_nosw_1ev_order/nosw_1ev_clusters and fig_4_switching_order no longer reference HOD at all (hod
rows were dropped from both in a recent edit), so they're correctly excluded here.

Also relevant to the NSM-switch broadcasting bug fixed in
model/VicsekIndividualsMultiSwitch.computeNewOrientations (the order/disorder mask merge broadcast
per-column instead of per-row): combo "e" (<lod(1), hod(1)>) uses HOD as its order value and carried
that bug too, even though it didn't visibly misbehave the way nearest(1)/farthest(1) did - see
switching_clustering, switching_percentage, switching_no_ev_order and the event_duration_*_e/
window_size_*_e/thresholds_*_e sweep figures below.

Writes to its own data-root/output-root by default, deliberately NOT the ones used by earlier runs -
reusing those could silently skip regeneration and keep stale results, since run_paper_figures.py
resumes from whatever raw data/figures are already on disk. Seeds are still derived deterministically
from each run's save path (see services/ServicePaperFigureRunner.deriveSeed), so runs here use the
exact same seeds as the equivalent combos elsewhere - directly comparable, isolating code changes as
the only difference.
"""


def isHodFigure(name, figures):
    figureSpec = figures[name]
    cells = figureSpec["cells"].values() if figureSpec["kind"] == "grid" else figureSpec["cells"]
    for cell in cells:
        for path in (cell["orderedPath"], cell["randomPath"]):
            # "_nsm=HOD_" -> comboNoSwitch/comboKSwitch with neighbourSelectionMechanism=HOD.
            # "_nsmCombo=HOD-LOD_" -> comboNsmSwitch with NSM_COMBO_LOD_HOD, where HOD is always the
            # order value (see paper_figures_specs.NSM_COMBO_LOD_HOD).
            if "_nsm=HOD_" in path or "_nsmCombo=HOD-LOD_" in path:
                return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default=os.path.expanduser("~/paper_figures_hod_tmp"),
                         help="scratch directory for raw simulation data - kept separate from other "
                              "pipeline runs so stale data can never be silently reused")
    parser.add_argument("--output-root", default="plots/paper_figures_hod",
                         help="where the rendered figures/manifest/seed log go - kept separate from "
                              "other pipeline output so you can compare side by side")
    parser.add_argument("--num-reps", type=int, default=50,
                         help="should match the run you're comparing against (default 50, matching "
                              "plots/paper_figures_d0.06_50reps)")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    parser.add_argument("--list", action="store_true",
                         help="list the targeted figure names and exit, without running anything")
    args = parser.parse_args()

    _runSpecs, figures = specs.buildAll(args.data_root, args.num_reps)
    targetNames = sorted(n for n in figures if isHodFigure(n, figures))

    print(f"{len(targetNames)} figures involve highest_orientation_difference:")
    for name in targetNames:
        print(f"  {name}")

    if args.list:
        return

    MainPipeline.run(args.data_root, args.output_root, args.num_reps, args.workers, targetNames, listOnly=False)


if __name__ == "__main__":
    main()
