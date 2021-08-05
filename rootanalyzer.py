import glob
import uproot3
import awkward as ak
import numpy as np
import awkward0
import ROOT
import h5py
import pandas as pd
import itertools
import time
from coffea import processor,util,hist
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
from coffea.nanoevents.methods import candidate
#the standard rowwise analysis.  Very slow compared to columnar data wrangling, but logical and simpler


fnames={}
#fnames["TT"] = glob.glob('/eos/uscms/store/user/knash/ZprimeToTT_M2500_W25_TuneCP2_PSweights_13TeV-madgraph-pythiaMLM-pythia8/RunIISummer20UL16MiniAOD-106X_mcRun2_asymptotic_v13-v2_NanoB2GNano2016mc_v0/210731_181605/0000/*.root')
fnames["QCD_HT1500to2000"]= sorted(glob.glob('/eos/uscms/store/user/knash/QCD_HT1500to2000_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8/RunIISummer20UL16MiniAOD-106X_mcRun2_asymptotic_v13-v1_NanoB2GNano2016mc_v1/210804_233703/0000/*.root'))

fileset={}
for ffi in fnames:
    fileset[ffi]=[ffj.replace("/eos/uscms/","root://cmsxrootd.fnal.gov///") for ffj in fnames[ffi]]
    fileset[ffi]=fileset[ffi][:23]
histos={}
histos["invm"]=ROOT.TH1F("invm","invm",100,0,5000)

sttime = time.time()
totev=0
for ffi in fileset:
    for ffj in fileset[ffi]:
            curfile = ROOT.TFile.Open(ffj)
            curttree = curfile.Get("Events")

            nent = curttree.GetEntries()

            itertree = iter(curttree)


            for iev in range(nent):

                if (totev%10000==0 and  totev>0):
                    print (totev)



                ev=next(itertree)
                evdict= {		
			        "FatJet_pt":[],
			        "FatJet_eta":[],
			        "FatJet_phi":[],
			        "FatJet_mass":[],	
			        "FatJet_lv":[]	
                        }
                nFatJet=int(getattr(ev, "nFatJet"))

                for ijet in range(int(getattr(ev, "nFatJet"))):
                    evdict["FatJet_pt"].append(float(getattr(ev, "FatJet_pt")[ijet]))
                    evdict["FatJet_eta"].append(float(getattr(ev, "FatJet_eta")[ijet]))
                    evdict["FatJet_phi"].append(float(getattr(ev, "FatJet_phi")[ijet]))
                    evdict["FatJet_mass"].append(float(getattr(ev, "FatJet_mass")[ijet]))	
                    evdict["FatJet_lv"].append(ROOT.TLorentzVector())	
                    evdict["FatJet_lv"][-1].SetPtEtaPhiM(evdict["FatJet_pt"][-1],evdict["FatJet_eta"][-1],evdict["FatJet_phi"][-1],evdict["FatJet_mass"][-1])
                if (nFatJet>1):
                    histos["invm"].Fill((evdict["FatJet_lv"][0]+evdict["FatJet_lv"][1]).M())
                totev+=1

DeltaT=time.time()-sttime
print(histos["invm"].Integral())
output = ROOT.TFile("FromROOT.root","recreate")
output.cd()
histos["invm"].Write()
output.Close()
print ("Execution time",DeltaT)





