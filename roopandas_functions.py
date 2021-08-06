

import uproot3
import numpy as np
import ROOT
import pandas as pd
import time
from array import array
from collections import OrderedDict
from tqdm.auto import tqdm
from pathlib import Path
import os
from glob import glob
import multiprocessing as mp





class PSequential():
    def __init__(self,seq):
        self.seq=seq

    def __call__(self,df):

        for iseq,seq in enumerate(self.seq):
            df=seq(df)
          
        return(df)
class PFilter():
    def __init__(self,func):
        self.func=func
    def __call__(self,df):
        rdf=self.func(df)
        rdf=rdf.reset_index(drop=True)
        return rdf

class PProduce():
    def __init__(self,bname,func,inputs=None):
        self.func=func
        self.args=func.args
        self.inputs=inputs
        self.bname=bname
    def __call__(self,df):
        serarr=[]

        for arg in self.args:
            serarr.append(df[arg])
        if self.inputs==None:
            rser=pd.Series(np.vectorize(self.func)(*serarr))
            df=df.assign(**({self.bname:rser}))
  
        else:
            df[self.bname]=self.inputs.np.vectorize(self.self.func)(*serarr)

        return df
class PAnalyze():
    def __init__(self,func):
        self.func=func
    def __call__(self,df):
        tqdm(leave=False,disable=True).pandas()
        rval=list(zip(*df.progress_apply(lambda row: self.func(row), axis = 1)))
        rvaldict={}
        for om in self.func.outmap:
            df[om]=rval[self.func.outmap.index(om)]
        return df

class PProcessor():
    def __init__(self,files,hists,branches,ana,nproc=1):
        self.files=files
        self.hists=hists 
        self.branches=branches   
        self.ana=ana 
        self.nproc=nproc 
        self.cutflow={}
    def Run(self,crange=None):
        STtime=time.time()


        for ids,ds in enumerate(self.files):
            selfiles=self.files[ds]

            if crange!=None:
                selfiles=self.files[ds][crange]
            if self.nproc>1:
                    pool = mp.Pool(self.nproc)
                    print (selfiles)
                    splitfiles=np.array_split(np.array(selfiles),self.nproc)
                    print (splitfiles)
                    results = [pool.apply_async(self.RunChunks, args=(ds,spf,ispf)) for ispf,spf in enumerate(splitfiles)]


                    pool.close()    
                    print(results)
                    for result in results:
                        print (result.get())
                    resarr = [result.get() for result in results]
                    print (resarr)
            else:
                    self.RunChunks(ds,selfiles)

        print ("Execution time",time.time()-STtime)


    def RunChunks(self,ds,selfiles,proc=1):

            self.cutflow[ds]=[0,0]
            print ("-"*20)
            print ("Dataset:",ds,"Started")
            for ichunk,chunk in enumerate(selfiles):

                print ("\tLoad Pandas Chunk",ichunk,"/",len(selfiles))
                df = pd.read_parquet(chunk,columns=self.branches)
                df=df.reset_index(drop=True)
                self.cutflow[ds][0]+=df.shape[0]
                print ("\tDone -- Events:",df.shape[0],"Branches:",df.shape[1])
                print ("\tRun Pandas Analysis...")

                df = self.ana(df)
                #print("out",out)
                self.cutflow[ds][1]+=df.shape[0]
                print ("\tFilling Histograms...")
                for ih,hh in enumerate(self.hists[ds]):
                        nent=len(df[hh])
                        #print(nent)
                        self.hists[ds][hh].FillN(nent,array("d",df[hh]),array("d",[1.0]*nent))

            print ("Dataset:",ds,"Completed")
            print ("Events input:",self.cutflow[ds][0],"output:",self.cutflow[ds][1])
            return(self.hists)
def PInitDir(path):
    filedict=OrderedDict()
    folders = os.listdir(path)
    for ff in folders:
        filedict[ff] = sorted(glob(path+"/"+ff+"/*"))
    return(filedict)
class PNanotoDataFrame():
    def __init__(self,fileset,branches,filesperchunk=5,filetype="parquet",path="./",nproc=1):
        self.fileset=fileset
        self.branches=branches
        self.filesperchunk=filesperchunk
        self.filetype=filetype
        self.path=path
        self.nproc=nproc


    def Run(self):
        #os.system("taskset -p 0xfffff %d" % os.getpid())

        for ffi in self.fileset:
            curpath=self.path+"RooPandas/"+ffi
            Path(curpath).mkdir(parents=True, exist_ok=False)

            if self.nproc>1:

                pool = mp.Pool(self.nproc)
                splitfiles=np.array_split(np.array(self.fileset[ffi]),self.nproc)

                results = [pool.apply_async(self.Convert, args=(ffi,spf,ispf)) for ispf,spf in enumerate(splitfiles)]
                pool.close()    
                resarr = [result.get() for result in results]


            else:
                self.Convert(ffi,self.fileset[ffi])
            print ("Converting set",ffi)
    
            

    def Convert(self,setname,filearr,proc=1):

            totf=0
            curpath=self.path+"RooPandas/"+setname
            nchunk=0
            reset=True
            nfiles=len(filearr)
            nchunks=int((nfiles-1)/self.filesperchunk)+1
            with tqdm(total=(nfiles)) as pbar:

                with tqdm(total=(self.filesperchunk)) as pbar1:
                    for ibatch,batch in enumerate(uproot3.pandas.iterate(path=filearr,flatten=False,treepath="Events", branches=self.branches,entrysteps=float("inf"))):
                            batch=batch.reset_index()
                            #print(ibatch,"/",min(self.filesperchunk*(nchunk+1),nfiles-1))
                     
                            if reset:
                                fullout = batch
                                reset=False
                            else:
                                fullout = pd.concat((fullout,batch))
                            if (ibatch%self.filesperchunk==0 and (ibatch>0)) or (ibatch==(nfiles-1)):
                                #print ("Saving Chunk",nchunk)
                                #print("With Size:",fullout.shape)

                                if (self.filetype=="parquet"):
                                    fullout.to_parquet(curpath+"/"+setname+"_"+str(proc)+"_Chunk"+str(nchunk+1)+"of"+str(nchunks)+".parquet")
                                    reset=True
                                else:
                                    raise(self.filetype+" not implemented")
                                if not (ibatch==(nfiles-1)):
                                    pbar1.total=min(self.filesperchunk,(nfiles-ibatch))
                                    pbar1.refresh() 
                                    pbar1.reset() 
                                nchunk+=1
                            pbar.update(1)

                            pbar1.update(1)
                            pbar1.update(0)
                            totf+=1            

