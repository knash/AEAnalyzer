import glob
import uproot3
import awkward as ak
import numpy as np
import awkward0
import h5py
import pandas as pd
import tqdm
from coffea import processor,util,hist
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
from coffea.nanoevents.methods import candidate
#Pandas analysis step1:  Open the NanoAOD root files and convert them to a Pandas DataFrame, saving as a parquet file. 
#Only need to save required branches and can do a natural skim to save space and step2 processing time.
#The NanoAOD arrays are saved as lists per cell -- a more natural way to do this is use awkward arrays, but I ran into some trouble with the saving and loading

def NanotoDataFrame(fileset,filestr,branches,filesperchunk=5,filetype="parquet"):
    print ("Running set",filestr)
    nchunk=0
    nfiles=len(fileset)
    #with tqdm.tqdm(total=(nfiles-1)) as pbar:
    reset=True
    for ibatch,batch in enumerate(uproot3.pandas.iterate(path=fileset,flatten=False,treepath="Events", branches=branchestokeep,entrysteps=float("inf"))):
            batch=batch.reset_index()
            print(ibatch,"/",min(filesperchunk*(nchunk+1),nfiles-1))
            #pbar.update(ibatch)
            if reset:
                fullout = batch
                reset=False
            else:
                fullout = pd.concat((fullout,batch))
            
            if (ibatch%filesperchunk==0 and (ibatch>0)) or (ibatch==(nfiles-1)):
                print ("Saving Chunk",nchunk)
                print("With Size:",fullout.shape)

                if (filetype=="parquet"):
                    fullout.to_parquet(filestr+"__Chunk"+str(nchunk)+".parquet")
                    reset=True
                else:
                    raise(filetype+" not implemented")

                nchunk+=1


    
fnames={}
#fnames["TT"] = glob.glob('/eos/uscms/store/user/knash/ZprimeToTT_M2500_W25_TuneCP2_PSweights_13TeV-madgraph-pythiaMLM-pythia8/RunIISummer20UL16MiniAOD-106X_mcRun2_asymptotic_v13-v2_NanoB2GNano2016mc_v0/210731_181605/0000/*.root')
fnames["QCD_HT1500to2000"]= sorted(glob.glob('/eos/uscms/store/user/knash/QCD_HT1500to2000_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8/RunIISummer20UL16MiniAOD-106X_mcRun2_asymptotic_v13-v1_NanoB2GNano2016mc_v1/210804_233703/0000/*.root'))

fileset={}
for ffi in fnames:
    fileset[ffi]=[ffj.replace("/eos/uscms/","root://cmsxrootd.fnal.gov///") for ffj in fnames[ffi]]
    fileset[ffi]=fileset[ffi][:23]
branchestokeep=["nFatJet","FatJet_pt","FatJet_eta","FatJet_phi","FatJet_mass","FatJet_msoftdrop","FatJet_iAE*","nJet","Jet_pt","Jet_eta","Jet_phi","Jet_mass"]
batchesperfile=5
for ffi in fileset:
    NanotoDataFrame(fileset[ffi],ffi,branchestokeep,filesperchunk=batchesperfile)



