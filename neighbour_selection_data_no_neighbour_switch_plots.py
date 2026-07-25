from enums.EnumNeighbourSelectionMechanism import NeighbourSelectionMechanism
from enums.EnumSwitchType import SwitchType
from enums.EnumEventEffect import EventEffect

import services.ServicePreparation as ServicePreparation
import services.ServiceNeighbourSelectionEval as ServiceNeighbourSelectionEval

# Evaluates the data produced by neighbour_selection_data_no_neighbour_switch.py (the
# updateIfNoNeighbours=False variant: individuals with no other neighbours keep their current switch
# value instead of updating it). Produces the same three plots as neighbour_selection_data_plots.py
# (order over time, switch value percentage over time, number of clusters over time), covering only
# the four switching combinations this variant exists for.

BASE_DATA_LOCATION = "/media/lilly/WD Elements/data_july26/"
OUTPUT_ROOT = "plots/neighbour_selection_data_no_neighbour_switch"

# data is logged every 100 steps for all these (LOCAL) combinations, so no further subsampling
# is applied when evaluating cluster count / switch percentage.
LOCAL_EVAL_INTERVAL = 1

domainSize = (50, 50)
densities = [0.01, 0.06, 0.09]
radii = [5, 10, 20]
ks = [1, 5]
threshold_options = [0.1, 0.2, 0.3, 0.4, 0.5]
switchThresholdOptions = [[t] for t in threshold_options]
eventEffects = [EventEffect.ALIGN_TO_FIXED_ANGLE, EventEffect.AWAY_FROM_ORIGIN, EventEffect.RANDOM]

reducedNeighbourSelectionMechanisms = [NeighbourSelectionMechanism.NEAREST,
                                        NeighbourSelectionMechanism.FARTHEST,
                                        NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE,
                                        NeighbourSelectionMechanism.HIGHEST_ORIENTATION_DIFFERENCE]
nsmCombos = [[NeighbourSelectionMechanism.NEAREST, NeighbourSelectionMechanism.FARTHEST],
             [NeighbourSelectionMechanism.LEAST_ORIENTATION_DIFFERENCE, NeighbourSelectionMechanism.HIGHEST_ORIENTATION_DIFFERENCE]]
kCombo = (5, 1)

tmaxLocal = 15000
eventStartTimestep = 5000
eventDuration = 1000
eventWindow = (eventStartTimestep, eventStartTimestep + eventDuration)


# ------------------------------------------------------------------ combination generators ---

def genLocalKswOneEvent():
    for density in densities:
        n = ServicePreparation.getNumberOfParticlesForConstantDensity(density=density, domainSize=domainSize)
        for radius in radii:
            for nsm in reducedNeighbourSelectionMechanisms:
                for thresholds in switchThresholdOptions:
                    for eventEffect in eventEffects:
                        base = f"local_ksw_1ev_blockiso_{{}}_st={{}}_d={density}_n={n}_r={radius}_nsm={nsm.value}_kCombo={kCombo[0]}-{kCombo[1]}_th={thresholds}_ee={eventEffect.val}"
                        yield dict(
                            comboType="local_ksw_1ev_blockiso",
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
                            base = (f"local_nsmsw_1ev_blockiso_{{}}_st={{}}_d={density}_n={n}_r={radius}"
                                    f"_nsmCombo={nsmCombo[0].value}-{nsmCombo[1].value}_k={k}_th={thresholds}_ee={eventEffect.val}")
                            yield dict(
                                comboType="local_nsmsw_1ev_blockiso",
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
                    base = f"local_ksw_noev_blockiso_{{}}_st={{}}_d={density}_n={n}_r={radius}_nsm={nsm.value}_kCombo={kCombo[0]}-{kCombo[1]}_th={thresholds}"
                    yield dict(
                        comboType="local_ksw_noev_blockiso",
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
                        base = (f"local_nsmsw_noev_blockiso_{{}}_st={{}}_d={density}_n={n}_r={radius}"
                                f"_nsmCombo={nsmCombo[0].value}-{nsmCombo[1].value}_k={k}_th={thresholds}")
                        yield dict(
                            comboType="local_nsmsw_noev_blockiso",
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
