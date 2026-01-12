import rabbit
import rabbit.inputdata

filename = (
    "/scratch/shared/mw/combine_studies/mw_unblinding/WMass_eta_pt_charge/WMass.hdf5"
)
# filename = "/scratch/shared/mw/combine_studies/mz_wlike_unblinding/ZMassWLike_eta_pt_charge/ZMassWLike.hdf5"
# filename = "/scratch/shared/mw/combine_studies/mz_wlike_unblinding/ZMassWLike_eta_pt_charge_fitMassDiff_charge/ZMassWLike.hdf5"

indata = rabbit.inputdata.FitInputData(filename)

# debug = narf.combineutils.FitDebugData(indata)

# print(indata.procs)
# print(indata.systgroups)
# print(indata.systgroupidxs)
print(f"Total number of nuisance parameters: {len(indata.systs)}")

# groupsToPrint = [b'pTModeling', b'resum', b'angularCoeffs', b'theory_ew']
groupsToPrint = [b"angularCoeffs"]

print("")
print("")
for ig, group in enumerate(indata.systgroups):
    # if group not in groupsToPrint:
    #    continue
    systs = [indata.systs[i] for i in indata.systgroupidxs[ig]]
    print(f"{group} = {len(systs)} items")  # : {systs}")
    if group in groupsToPrint:
        print(f"{group}: {systs}")
    # print("")
    print("")


# test = debug.nonzeroSysts(procs = ["Diboson"], channels = ["ch0"])
# test2 = debug.channelsForNonzeroSysts(procs = ["Zmumu"])
# test3 = debug.procsForNonzeroSysts(systs = ["effStat_idip_eta12pt1q1"])

# print(test)
# print(test2)
# print(test3)
print(f"Total number of nuisance parameters: {len(indata.systs)}")
