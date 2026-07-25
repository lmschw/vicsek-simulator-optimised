from enums.EnumNeighbourSelectionMechanism import NeighbourSelectionMechanism
from enums.EnumSwitchType import SwitchType
from enums.EnumEventEffect import EventEffect

import services.ServicePreparation as ServicePreparation
import services.ServiceNeighbourSelectionEval as ServiceNeighbourSelectionEval

# This script evaluates the data produced by neighbour_selection_data.py. For every parameter
# combination it produces three plots (mean +/- standard deviation across all runs sharing the
# same base path, i.e. differing only in the run index i):
#   1) local order over time (read directly from the precomputed *_globalOrder.csv files)
#   2) for combinations that switch (between k values or neighbour selection mechanisms), the
#      percentage of the swarm that has chosen each of the two switch values over time
#   3) the number of (spatial + orientation) clusters in the swarm over time
# Every plot is saved as png, svg and pdf. The loading/aggregation/plotting logic lives in
# services/ServiceNeighbourSelectionEval.py, shared with this script's variants.

BASE_DATA_LOCATION = "/media/lilly/WD Elements/data_july26/"
OUTPUT_ROOT = "plots/neighbour_selection_data"

# data saved every single timestep (GLOBAL sections) needs to be subsampled to keep the cluster/
# switch-percentage evaluation fast. LOCAL sections are already only logged every 100 steps, so
# no further subsampling is applied.
GLOBAL_EVAL_INTERVAL = 50
LOCAL_EVAL_INTERVAL = 1

domainSize = (50, 50)
densities = [0.01, 0.06, 0.09]
radii = [5, 10, 20]
ks = [1, 5]
threshold_options = [0.1, 0.2, 0.3, 0.4, 0.5]
switchThresholdOptions = [[t] for t in threshold_options]
eventEffects = [EventEffect.ALIGN_TO_FIXED_ANGLE, EventEffect.AWAY_FROM_ORIGIN, EventEffect.RANDOM]

neighbourSelectionMechanisms = [NeighbourSelectionMechanism.ALL,
                                 NeighbourSelectionMechanism.RANDOM,
                                 NeighbourSelectionMechanism.NEAREST,
                                 NeighbourSelectionMechanism.FARTHEST,
                                 NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE,
                                 NeighbourSelectionMechanism.HIGHEST_ORIENTATION_DIFFERENCE]
reducedNeighbourSelectionMechanisms = [NeighbourSelectionMechanism.NEAREST,
                                        NeighbourSelectionMechanism.FARTHEST,
                                        NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE,
                                        NeighbourSelectionMechanism.HIGHEST_ORIENTATION_DIFFERENCE]
nsmCombos = [[NeighbourSelectionMechanism.NEAREST, NeighbourSelectionMechanism.FARTHEST],
             [NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE, NeighbourSelectionMechanism.HIGHEST_ORIENTATION_DIFFERENCE]]
kCombo = (5, 1)

tmaxGlobal = 3000
tmaxLocal = 15000
eventStartTimestep = 5000
eventDuration = 1000
eventWindow = (eventStartTimestep, eventStartTimestep + eventDuration)


# ------------------------------------------------------------------ combination generators ---

def genGlobalNoswNoev():
    for density in densities:
        n = ServicePreparation.getNumberOfParticlesForConstantDensity(density=density, domainSize=domainSize)
        for radius in radii:
            for nsm in neighbourSelectionMechanisms:
                for k in ks:
                    base = f"global_nosw_noev_{{}}_d={density}_n={n}_r={radius}_nsm={nsm.value}_k={k}"
                    yield dict(
                        comboType="global_nosw_noev",
                        basePathOrdered=f"{BASE_DATA_LOCATION}{base.format('ordered')}",
                        basePathRandom=f"{BASE_DATA_LOCATION}{base.format('random')}",
                        paramSuffix=f"d={density}_n={n}_r={radius}_nsm={nsm.value}_k={k}",
                        evalInterval=GLOBAL_EVAL_INTERVAL,
                        eventShading=None,
                    )


def genLocalNoswOneEvent():
    for density in densities:
        n = ServicePreparation.getNumberOfParticlesForConstantDensity(density=density, domainSize=domainSize)
        for radius in radii:
            for nsm in reducedNeighbourSelectionMechanisms:
                for k in ks:
                    for thresholds in threshold_options:
                        for eventEffect in eventEffects:
                            base = f"local_nosw_1ev_{{}}_d={density}_n={n}_r={radius}_nsm={nsm.value}_k={k}_th={thresholds}_ee={eventEffect.val}"
                            yield dict(
                                comboType="local_nosw_1ev",
                                basePathOrdered=f"{BASE_DATA_LOCATION}{base.format('ordered')}",
                                basePathRandom=f"{BASE_DATA_LOCATION}{base.format('random')}",
                                paramSuffix=f"d={density}_n={n}_r={radius}_nsm={nsm.value}_k={k}_th={thresholds}_ee={eventEffect.val}",
                                evalInterval=LOCAL_EVAL_INTERVAL,
                                eventShading=eventWindow,
                            )


def genLocalKswOneEvent():
    for density in densities:
        n = ServicePreparation.getNumberOfParticlesForConstantDensity(density=density, domainSize=domainSize)
        for radius in radii:
            for nsm in reducedNeighbourSelectionMechanisms:
                for thresholds in switchThresholdOptions:
                    for eventEffect in eventEffects:
                        base = f"local_ksw_1ev_{{}}_st={{}}_d={density}_n={n}_r={radius}_nsm={nsm.value}_kCombo={kCombo[0]}-{kCombo[1]}_th={thresholds}_ee={eventEffect.val}"
                        yield dict(
                            comboType="local_ksw_1ev",
                            basePathOrdered=f"{BASE_DATA_LOCATION}{base.format('ordered', kCombo[0])}",
                            basePathRandom=f"{BASE_DATA_LOCATION}{base.format('random', kCombo[1])}",
                            paramSuffix=f"d={density}_n={n}_r={radius}_nsm={nsm.value}_kCombo={kCombo[0]}-{kCombo[1]}_th={thresholds}_ee={eventEffect.val}",
                            evalInterval=LOCAL_EVAL_INTERVAL,
                            eventShading=eventWindow,
                            switchType=SwitchType.K,
                            switchTypeOptions=(kCombo[0], kCombo[1]),
                            valueLabels=(f"k={kCombo[0]}", f"k={kCombo[1]}"),
                        )


def genLocalNsmswOneEvent():
    for density in densities:
        n = ServicePreparation.getNumberOfParticlesForConstantDensity(density=density, domainSize=domainSize)
        for radius in radii:
            for nsmCombo in nsmCombos:
                for k in ks:
                    for thresholds in switchThresholdOptions:
                        for eventEffect in eventEffects:
                            base = (f"local_nsmsw_1ev_{{}}_st={{}}_d={density}_n={n}_r={radius}"
                                    f"_nsmCombo={nsmCombo[0].value}-{nsmCombo[1].value}_k={k}_th={thresholds}_ee={eventEffect.val}")
                            yield dict(
                                comboType="local_nsmsw_1ev",
                                basePathOrdered=f"{BASE_DATA_LOCATION}{base.format('ordered', nsmCombo[0].value)}",
                                basePathRandom=f"{BASE_DATA_LOCATION}{base.format('random', nsmCombo[1].value)}",
                                paramSuffix=f"d={density}_n={n}_r={radius}_nsmCombo={nsmCombo[0].value}-{nsmCombo[1].value}_k={k}_th={thresholds}_ee={eventEffect.val}",
                                evalInterval=LOCAL_EVAL_INTERVAL,
                                eventShading=eventWindow,
                                switchType=SwitchType.NEIGHBOUR_SELECTION_MECHANISM,
                                switchTypeOptions=(nsmCombo[0], nsmCombo[1]),
                                valueLabels=(nsmCombo[0].name, nsmCombo[1].name),
                            )


def genLocalKswNoEvent():
    for density in densities:
        n = ServicePreparation.getNumberOfParticlesForConstantDensity(density=density, domainSize=domainSize)
        for radius in radii:
            for nsm in reducedNeighbourSelectionMechanisms:
                for thresholds in switchThresholdOptions:
                    base = f"local_ksw_noev_{{}}_st={{}}_d={density}_n={n}_r={radius}_nsm={nsm.value}_kCombo={kCombo[0]}-{kCombo[1]}_th={thresholds}"
                    yield dict(
                        comboType="local_ksw_noev",
                        basePathOrdered=f"{BASE_DATA_LOCATION}{base.format('ordered', kCombo[0])}",
                        basePathRandom=f"{BASE_DATA_LOCATION}{base.format('random', kCombo[1])}",
                        paramSuffix=f"d={density}_n={n}_r={radius}_nsm={nsm.value}_kCombo={kCombo[0]}-{kCombo[1]}_th={thresholds}",
                        evalInterval=LOCAL_EVAL_INTERVAL,
                        eventShading=None,
                        switchType=SwitchType.K,
                        switchTypeOptions=(kCombo[0], kCombo[1]),
                        valueLabels=(f"k={kCombo[0]}", f"k={kCombo[1]}"),
                    )


def genLocalNsmswNoEvent():
    for density in densities:
        n = ServicePreparation.getNumberOfParticlesForConstantDensity(density=density, domainSize=domainSize)
        for radius in radii:
            for nsmCombo in nsmCombos:
                for k in ks:
                    for thresholds in switchThresholdOptions:
                        base = (f"local_nsmsw_noev_{{}}_st={{}}_d={density}_n={n}_r={radius}"
                                f"_nsmCombo={nsmCombo[0].value}-{nsmCombo[1].value}_k={k}_th={thresholds}")
                        yield dict(
                            comboType="local_nsmsw_noev",
                            basePathOrdered=f"{BASE_DATA_LOCATION}{base.format('ordered', nsmCombo[0].value)}",
                            basePathRandom=f"{BASE_DATA_LOCATION}{base.format('random', nsmCombo[1].value)}",
                            paramSuffix=f"d={density}_n={n}_r={radius}_nsmCombo={nsmCombo[0].value}-{nsmCombo[1].value}_k={k}_th={thresholds}",
                            evalInterval=LOCAL_EVAL_INTERVAL,
                            eventShading=None,
                            switchType=SwitchType.NEIGHBOUR_SELECTION_MECHANISM,
                            switchTypeOptions=(nsmCombo[0], nsmCombo[1]),
                            valueLabels=(nsmCombo[0].name, nsmCombo[1].name),
                        )


SECTION_GENERATORS = {
    "global_nosw_noev": genGlobalNoswNoev,
    "local_nosw_1ev": genLocalNoswOneEvent,
    "local_ksw_1ev": genLocalKswOneEvent,
    "local_nsmsw_1ev": genLocalNsmswOneEvent,
    "local_ksw_noev": genLocalKswNoEvent,
    "local_nsmsw_noev": genLocalNsmswNoEvent,
}


def main():
    args = ServiceNeighbourSelectionEval.buildArgParser(list(SECTION_GENERATORS.keys())).parse_args()
    ServiceNeighbourSelectionEval.runSections(BASE_DATA_LOCATION, OUTPUT_ROOT, SECTION_GENERATORS, args)


if __name__ == "__main__":
    main()
