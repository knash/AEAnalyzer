
import glob
import uproot3
import awkward as ak
import numpy as np
import awkward0
import h5py
import ROOT
import pandas as pd
import time
from array import array
from coffea import processor,util,hist
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
from coffea.nanoevents.methods import candidate
import dask
import dask.dataframe as dd
from dask.multiprocessing import get
import root_numpy
from collections import OrderedDict
import swifter
from tqdm import tqdm

#Pandas analysis step2:  Open the Parquet chunks and run a columnwise function defining the entire analysis
#Can be sped up through multiprocessing, also through skimming during step1
#This method is faster than ROOT rowwise and more flexible then Coffea.  But, requires a one-time conversion from step 1.

chunklist =sorted(glob.glob("QCD_HT1500to2000__Chunk*.parquet"))

histos=OrderedDict()
histos["invm"]=ROOT.TH1F("invm","invm",100,0,5000)
histos["logMSE"]=ROOT.TH1F("logMSE","logMSE",40,-20,20)



histindices=OrderedDict()
for ih,hh in enumerate(histos):
    histos[hh].Sumw2()
    histindices[hh]=ih


#Performing apply on a DataFrame can be slow, but seems to be pretty quick for us
#Will be good to perform columnwise skimming first 
def Processor(DataFrame,Func):
    return list(zip(*df.progress_apply(lambda row: Func(row), axis = 1)))

def Analyze(row):
    rvals=[0.0]*len(histos)
    LV=[]

    if row["nFatJet"]>1:
        for ir in range(row["nFatJet"]):
                LV.append(ROOT.TLorentzVector())
                LV[-1].SetPtEtaPhiM(row.FatJet_pt[ir],row.FatJet_eta[ir],row.FatJet_phi[ir],row.FatJet_mass[ir]) 
        invm=(LV[0] + LV[1]).M()
        rvals[histindices["invm"]]=invm
        rvals[histindices["logMSE"]]=np.log(row.FatJet_iAEMSE[0])
    
    return rvals
branchestoread=["FatJet_pt","FatJet_eta","FatJet_phi","FatJet_mass","FatJet_iAEMSE","nFatJet"]
maxchunks=1
sttime = time.time()
for ichunk,chunk in enumerate(chunklist):

    #Bare Pandas implementation 
    print ("Load Pandas Chunk",ichunk,"/",len(chunklist))
    df = pd.read_parquet(chunk,columns=branchestoread)
    df = df.reset_index(drop=True)
    print ("\tDone -- Events:",df.shape[0],"Branches:",df.shape[1])
    print ("\tRun Pandas Analysis...")

    tqdm.pandas()
    output=Processor(df,Analyze)

    print ("\tFilling Histograms...")
    for ih,hh in enumerate(histos):
            nent=len(output[ih])
            histos[hh].FillN(nent,array("d",output[ih]),array("d",[1.0]*nent))


    #Dask implementation -- Same as Pandas but with parallel processing.  Never got it working though...
    #print ("Load Dask")
    #ddf = dd.read_parquet(chunk, engine='pyarrow')
    #ddf = ddf.repartition(8)
    #print ("Run Dask")
    #sttimeD = time.time()
    #with dask.config.set(scheduler='threads',workers=4):
    #        res = ddf.map_partitions(lambda df: df.apply((lambda row: filter(row)), axis = 1),meta=(None, 'object')).compute(scheduler='threads',workers=4)
    #print (time.time()-sttimeD)



    #Swifter implementation -- Same as Pandas but with automatic vectorization.  Seems to not improve performance...
    #print ("Run Swifter")
    #sttimeD = time.time()
    #df.swifter.apply(lambda row: filter(row), axis = 1)
    #print (time.time()-sttimeD)
    #print ("Done")



print ("Done!")
DeltaT=time.time()-sttime
print (histos["invm"].Integral())
output = ROOT.TFile("FromPandas.root","recreate")
output.cd()
histos["invm"].Write()
histos["logMSE"].Write()
output.Close()
#print(df)
print ("Execution time",DeltaT)



