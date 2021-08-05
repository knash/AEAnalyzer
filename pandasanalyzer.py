import glob
import uproot3
import awkward as ak
import numpy as np
import awkward0
import h5py
import ROOT
import pandas as pd
import time
from coffea import processor,util,hist
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
from coffea.nanoevents.methods import candidate
import dask
import dask.dataframe as dd
from dask.multiprocessing import get
import swifter
#Pandas analysis step2:  Open the Parquet chunks and run a columnwise function defining the entire analysis
#Can be sped up through multiprocessing, also through skimming during step1
#This method is faster than ROOT rowwise and more flexible then Coffea.  But, requires a one-time conversion from step 1.

chunklist =sorted(glob.glob("QCD_HT1500to2000__Chunk*.parquet"))

histos={}
histos["invm"]=ROOT.TH1F("invm","invm",100,0,5000)
for hh in histos:
    histos[hh].Sumw2()

#Performing apply on a DataFrame can be slow, but seems to be pretty quick for us
#Will be good to perform columnwise skimming first 
def Processor(DataFrame,Func):
    return zip(*df.apply(lambda row: Func(row), axis = 1))

def Analyze(row):
    LV=[]
    invm=None

    if row["nFatJet"]>1:
        for ir in range(row["nFatJet"]):
                LV.append(ROOT.TLorentzVector())
                LV[-1].SetPtEtaPhiM(row.FatJet_pt[ir],row.FatJet_eta[ir],row.FatJet_phi[ir],row.FatJet_mass[ir]) 
        invm=(LV[0] + LV[1]).M()
        histos["invm"].Fill(invm)
        
    return[invm,LV]
branchestoread=["FatJet_pt","FatJet_eta","FatJet_phi","FatJet_mass","nFatJet"]

sttime = time.time()
for ichunk,chunk in enumerate(chunklist):

    #Bare Pandas implementation 
    print ("Load Pandas Chunk",ichunk)
    df = pd.read_parquet(chunk,columns=branchestoread)
    df = df.reset_index(drop=True)
    print( df.shape)
    print ("Run Pandas Analysis")
    ProcOut=Processor(df,Analyze)
    #You can easily make new columns from a processor 
    #print (ProcOut)
    #(df["invm"],df["LV"])=ProcOut



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
output.Close()
#print(df)
print ("Execution time",DeltaT)



