from glob import glob
import uproot3
import numpy as np
import pandas as pd
import tqdm
from RooPandasFunctions import PNanotoDataFrame
from collections import OrderedDict
import pyspark
fnames={}
fnames["TT"] = sorted(glob('/eos/uscms/store/user/knash/ZprimeToTT_M2500_W25_TuneCP2_PSweights_13TeV-madgraph-pythiaMLM-pythia8/RunIISummer20UL16MiniAOD-106X_mcRun2_asymptotic_v13-v2_NanoB2GNano2016mc_v1/210804_233356/0000/*.root'))
fnames["QCD_HT1500to2000"]= sorted(glob('/eos/uscms/store/user/knash/QCD_HT1500to2000_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8/RunIISummer20UL16MiniAOD-106X_mcRun2_asymptotic_v13-v1_NanoB2GNano2016mc_v1/210804_233703/0000/*.root'))

fileset={}
for ffi in fnames:
    fileset[ffi]=[ffj.replace("/eos/uscms/","root://cmsxrootd.fnal.gov///") for ffj in fnames[ffi]]
    fileset[ffi]=fileset[ffi]
    #fileset[ffi]=fileset[ffi][:1]
batchesperfile={"TT":5,"QCD_HT1500to2000":15}
branchestokeep=OrderedDict([("Jet",["pt","eta","phi","mass"]),("FatJet",["pt","eta","phi","mass","msoftdrop","iAEMSE"]),("HLT",["PFHT900"]),("",["run","luminosityBlock","event"])])
mind={"FatJet":5,"Jet":5,"Muon":1,"":None,"HLT":None}
PNanotoDataFrame(fileset,branchestokeep,filesperchunk=batchesperfile,nproc=6,atype="flat",dirname="RooFlatFull",maxind=mind).Run()


