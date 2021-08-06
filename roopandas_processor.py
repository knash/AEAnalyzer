from glob import glob
import uproot3
import numpy as np
import pandas as pd
import tqdm
from roopandas_functions import PNanotoDataFrame

fnames={}
fnames["TT"] = sorted(glob('/eos/uscms/store/user/knash/ZprimeToTT_M2500_W25_TuneCP2_PSweights_13TeV-madgraph-pythiaMLM-pythia8/RunIISummer20UL16MiniAOD-106X_mcRun2_asymptotic_v13-v2_NanoB2GNano2016mc_v1/210804_233356/0000/*.root'))
fnames["QCD_HT1500to2000"]= sorted(glob('/eos/uscms/store/user/knash/QCD_HT1500to2000_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8/RunIISummer20UL16MiniAOD-106X_mcRun2_asymptotic_v13-v1_NanoB2GNano2016mc_v1/210804_233703/0000/*.root'))

fileset={}
for ffi in fnames:
    fileset[ffi]=[ffj.replace("/eos/uscms/","root://cmsxrootd.fnal.gov///") for ffj in fnames[ffi]]
    #fileset[ffi]=fileset[ffi][:60]
branchestokeep=["nFatJet","FatJet_pt","FatJet_eta","FatJet_phi","FatJet_mass","FatJet_msoftdrop","FatJet_iAE*","nJet","Jet_pt","Jet_eta","Jet_phi","Jet_mass"]
batchesperfile=10

PNanotoDataFrame(fileset,branchestokeep,filesperchunk=batchesperfile,nproc=4).Run()



