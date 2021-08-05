import glob
import uproot
import awkward as ak
import numpy as np
import time
from coffea import processor,util,hist
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
from coffea.nanoevents.methods import candidate
#Coffea analyzer: This loads the entire NanoAOD files and processes the data in a columnar way.
#Fast and simple to make a simple analysis.  More advanced stuff may prove quite difficult however -- seems very black boxy.
#Also, the group is trying to reimplement many parts of root, which is a distinct disadvantage.

ak.behavior.update(candidate.behavior)




fnames={}
#fnames["TT"] = glob.glob('/eos/uscms/store/user/knash/ZprimeToTT_M2500_W25_TuneCP2_PSweights_13TeV-madgraph-pythiaMLM-pythia8/RunIISummer20UL16MiniAOD-106X_mcRun2_asymptotic_v13-v2_NanoB2GNano2016mc_v0/210731_181605/0000/*.root')
#fnames["QCD_HT1000to1500"] = glob.glob('/eos/uscms/store/user/knash/QCD_HT1000to1500_TuneCP5_13TeV-madgraph-pythia8/RunIIFall17MiniAODv2-PU2017_12Apr2018_new_pmx_94X_mc2017_realistic_v14-v1_NanoSlimNtuples2017mc_v8/190802_180606/0000/*.root')
fnames["QCD_HT1500to2000"]= sorted(glob.glob('/eos/uscms/store/user/knash/QCD_HT1500to2000_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8/RunIISummer20UL16MiniAOD-106X_mcRun2_asymptotic_v13-v1_NanoB2GNano2016mc_v1/210804_233703/0000/*.root'))

fileset={}
for ffi in fnames:
    fileset[ffi]=[ffj.replace("/eos/uscms/","root://cmsxrootd.fnal.gov///") for ffj in fnames[ffi]]
    fileset[ffi]=fileset[ffi][:23]

class MyProcessor(processor.ProcessorABC):
    def __init__(self):
        self._accumulator = processor.dict_accumulator({
            "sumw": processor.defaultdict_accumulator(float),
            "invm": hist.Hist(
                "Events",
                hist.Cat("dataset", "Dataset"),
                hist.Bin("invm", " [GeV]",100,0,5000),
            ),
        })

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        output = self.accumulator.identity()

        dataset = events.metadata['dataset']
        AK8 = ak.zip({
            "pt": events.FatJet_pt,
            "eta": events.FatJet_eta,
            "phi": events.FatJet_phi,
            "mass": events.FatJet_mass,
            "charge": events.FatJet_phi,
        }, with_name="PtEtaPhiMCandidate")

        cut = (ak.num(AK8) >= 2)
        # add first and second muon in every event together
        AK8inv = AK8[cut][:, 0] + AK8[cut][:, 1]

        output["sumw"][dataset] += len(events)
        output["invm"].fill(
            dataset=dataset,
            invm=AK8inv.mass,
        )

        return output

    def postprocess(self, accumulator):
        return accumulator
print("Starting Uproot Job")
sttime = time.time()
out = processor.run_uproot_job(
    fileset,
    treename="Events",
    processor_instance=MyProcessor(),
    executor=processor.iterative_executor,
    executor_args={
        "schema": BaseSchema,
    },
    maxchunks=999,
)
DeltaT=time.time()-sttime

print("Done")

pltplot=hist.plot1d(out['invm'])
pltplot.figure.savefig("FromCoffea.pdf")
util.save(out, "coffeapod.coff")

print ("Execution time",DeltaT)


