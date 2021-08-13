
import awkward0
import awkward as ak
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
import pyarrow as pa
import pyarrow.parquet as pq
import itertools
import xarray as xa
import copy 
import functools


class PSequential():
    def __init__(self,Proc,seq):
        self.Proc=Proc
        self.seq=seq
    def __call__(self,df,cureeventinfo):

        for iseq,seq in enumerate(self.seq):
            inddict={}
            keys=df.keys()
            for branch in keys:
                inddict[branch]=(df[branch].index)

            df=seq(self.Proc,df,cureeventinfo)
            self.Proc.timing[str(iseq)+":"+str(type(seq))]=(time.time())

            if  isinstance(df,type(None)):
                return df
            if (not isinstance(seq,PFilter)):
                for branch in keys:
                    if (not df[branch].index.identical(inddict[branch])):
                         raise ValueError("Only PFilter should change indexing, not",type(seq))
          
        return(df)


#These two could easily be combined. Mainly seperate for indexing safety
class PFilter():
    def __init__(self,func):
        self.func=func
    def __call__(self,Proc,df,cureeventinfo):

        Cond=self.func(df,cureeventinfo)
        if ((not isinstance(Cond,pd.Series)) and (not isinstance(Cond,pd.DataFrame)) and (not isinstance(Cond,type(None)))):
            raise ValueError("PFilter output of type",type(Cond)," not supported" )
        if isinstance(Cond,type(None)):
            return Cond
        if (Cond.index.nlevels)>1:
            Cond= Cond.droplevel(level=1)
        for branch in df:
            df[branch]=df[branch][df[branch].index.get_level_values(0).isin(Cond[Cond].index,level=0)]
        return df

class PColumn():
    def __init__(self,func):
        self.func=func
    def __call__(self,Proc,df,cureeventinfo):
        self.func(df,cureeventinfo)
        return df

class PRow():
    def __init__(self,bname,func):
        self.func=func
        self.bname=bname

    def __call__(self,Proc,df,cureeventinfo):
        serarr=self.func.prepdf(df)
        Serdict={}
        indices=[]
        for iser in range(len(serarr)):
                serarr[iser]=serarr[iser].dropna()
                indices.append(serarr[iser].index)
                if (not indices[-1].identical(indices[0])):
                    raise ValueError("Indices must be identical when using PRow()")
        nrows=serarr[iser].shape[0]
        indices=(serarr[0].index)
        serarrl=list(zip(*serarr))
        funcapply=[self.func([*x],cureeventinfo) for x in (serarrl)]

        if (len(funcapply)!=nrows):
            raise ValueError("Output must have same number of elements as the input")

        cname=np.array(self.bname)[:,1]
        dictnames=np.array(self.bname)[:,0]

        Serdict=pd.DataFrame(funcapply,columns=cname,index=indices)

        for idname,dictname in enumerate(dictnames):

            if not dictname in df:
                df[dictname]=pd.DataFrame()  
            else:
                Serdict[cname[idname]].index=df[dictname].index
            
            df[dictname]=df[dictname].assign(**({cname[idname]:Serdict[cname[idname]]}))
        return df
class PEventInfo():
    def __init__(self,dataset,nevents,ichunk,nevchunk,eventcontainer):
                self.dataset=dataset
                self.nevents=nevents
                self.ichunk=ichunk
                self.nevchunk=nevchunk
                self.eventcontainer=eventcontainer

def RunProc(Proc):
    
    return Proc.Run()
def FillHist(df,hists):
        for ih,hh in enumerate(hists):
                #print(df,hh)
                nent=len(df[hh])
                hists[hh].FillN(nent,array("d",df[hh]),array("d",[1.0]*nent))

class PProcRunner():
    def __init__(self,Proc,nproc):
                self.nproc=nproc
                self.Proc=Proc
    def Run(self,crange=None):
            fulltime=time.time()
            if self.nproc>1:
                    pool = mp.Pool(self.nproc)

                    farrays={}
                    for iff,ff in enumerate(self.Proc.files):
                        farrays[ff]=np.array_split(np.array(self.Proc.files[ff]),self.nproc)

                    splitfiles=[]
                    for proc in range(self.nproc):
                        splitfiles.append({})
                        for iff,ff in enumerate(self.Proc.files):
                            splitfiles[proc][ff]=farrays[ff][proc]

                    allprocs=[]
                    allinstances=[]
                    for proc in range(self.nproc):
                        allprocs.append(copy.deepcopy(self.Proc))
                        allprocs[-1].files=splitfiles[proc]
                        allprocs[-1].verbose=False
                        allprocs[-1].proc=proc



                    #pool.map(allinstances)
                    print("Running process over ",self.nproc,"processors")
                    results = pool.map(RunProc ,[x for x in allprocs])
                    #results = [pool.apply_async(RunProc for CProc in allprocs)]
                    print("Done")

                    pool.close()    

                    resarr = results
                    #print (resarr)
                    for rr in resarr:
                        #print (rr)
                        for ds in rr:
                                FillHist(rr[ds][0],self.Proc.hists[ds])
                    #self.Proc.hists=
            else:
                    runout=self.Proc.Run(crange)
                    #print(runout)
                    #print(self.Proc.hists)
                    for ds in runout:
                        FillHist(runout[ds][0],self.Proc.hists[ds])
            print("Total time",time.time()-fulltime)


class PProcessor():
    def __init__(self,files,hists,branches,sequence,branchname="Dataset",proc=1,atype="flat",eventcontainer={},scalars=[],verbose=True):
        self.files=files
        self.hists=hists 
        self.scalars=scalars 
        self.branches=branches  
        self.branchname=branchname    
        self.proc=proc 
        self.cutflow={}
        self.atype=atype
        self.eventcontainer=eventcontainer
        self.ana=PSequential(self,sequence)
        self.verbose=verbose
        self.timing={}

    def Run(self,crange=None):
        STtime=time.time()

        returnval={}
        for ids,ds in enumerate(self.files):
            selfiles=self.files[ds]

            if crange!=None:
                selfiles=self.files[ds][crange]
            returnval[ds]=self.RunChunks(ds,selfiles)
        if self.verbose:
            print ("Execution time",time.time()-STtime)
        #print(hists)
        return (returnval)

    def RunChunks(self,ds,selfiles):

            self.cutflow[ds]=[0,0]
            if self.verbose:
                print ("-"*20)
                print ("Dataset:",ds,"Started")
            fillH=True
            timingagg=None
            histreturn=None
         
            with tqdm(total=(len(selfiles)),desc="Dataset:"+ds+" Process:"+str(self.proc)) as pbar:
                for ichunk,chunk in enumerate(selfiles):
                    #print(ichunk)
                    self.timing={}
                    self.timing["Start"]=[time.time()]
                    df={}

                    bdict={"":self.branches}
                    if isinstance(self.branches,dict):
                            bdict=self.branches
                    allcolumns=[]
                    for ibmaj,bmaj in enumerate(bdict):
                            delim="_"
                            if bmaj=="":
                                delim=""
                            allcolumns+=[bmaj+delim+x for x in bdict[bmaj]]
                            if not (bmaj in self.scalars):
                                    allcolumns.append("n"+bmaj)
                                        
                    self.timing["Start"]=(time.time())
                    dffull=pd.read_parquet(chunk,columns=allcolumns)
                    self.timing["File Read"]=(time.time())
                    for ibmaj,bmaj in enumerate(bdict):
                            delim="_"
                            if bmaj=="":
                                delim=""
                            brancharr=[bmaj+delim+x for x in bdict[bmaj]]
                            tempbr=copy.deepcopy(brancharr)
                         
                            if bmaj!="":
                                if not (bmaj in self.scalars):
                                    brancharr.append("n"+bmaj)
                                brancharr.append("event")
                            df[bmaj]=dffull[brancharr]
                            namemap = {x:x.replace(bmaj+"_","") for x in brancharr}

                            df[bmaj]=df[bmaj].rename(columns=namemap)
                            df[bmaj]=df[bmaj].dropna(how='all')

                    df["Hists"]=pd.DataFrame(df[""]["event"],index=df[""].index)
                   
                    df[""]=df[""].droplevel(level=1)
                    df["Hists"]=df["Hists"].droplevel(level=1)

             
                    prekeys=df.keys()
                    nevchunk=df[""].shape[0]
                   
                    self.cutflow[ds][0]+= nevchunk
                    
                    cureeventinfo=PEventInfo(ds,self.cutflow[ds][0],ichunk,nevchunk,self.eventcontainer)
                    self.timing["Parsed"]=(time.time())
                    df = self.ana(df,cureeventinfo)
                    if  isinstance(df,type(None)):
                            print("Skip",ichunk)
                            pbar.update(1)
                            continue
                    self.timing["Analyzed"]=(time.time())
                    self.cutflow[ds][1]+= df[""].shape[0]
                    eventlist=df[""]["event"].values
                    for branch in prekeys:
                            if branch!="":
                                br = df[branch]["event"].dropna().values==eventlist
                                if not (br).all():
                                    raise ValueError("Events are not 1-to-1 in collection",branch)
                    if ichunk==0:
                        histreturn=df["Hists"]
                    else:
                        #print(histreturn.shape,df["Hists"].shape)
                        histreturn=pd.concat((histreturn,df["Hists"]))
                    #print(ichunk,histreturn.shape,df["Hists"].shape)
                    #if fillH:
                     #   self.FillHist(df["Hists"],ds)
                    self.timing["Histograms Filled"]=(time.time())
            
                    pbar.update(1)
                    if isinstance(timingagg,type(None)):
                        timingagg=self.timing
                    else:
                        for benchmark in self.timing:
                            timingagg[benchmark]+=self.timing[benchmark]
                if self.verbose:
                    print("Timing...")
                    for benchmark in timingagg:
                        print ("\t",benchmark,timingagg[benchmark]-timingagg["Start"])
                    print ("Dataset:",ds,"Completed")
                    print ("Events input:",self.cutflow[ds][0],"output:",self.cutflow[ds][1])
            return(histreturn,timingagg)
def PInitDir(path):
    filedict=OrderedDict()
    folders = os.listdir(path)
    for ff in folders:
        filedict[ff] = sorted(glob(path+"/"+ff+"/*"))
    return(filedict)
class PNanotoDataFrame():
    def __init__(self,fileset,branches,maxind=None,filesperchunk=5,filetype="parquet",path="./",nproc=1,atype="pandas",dirname="RooPandas"):
        self.fileset=fileset
        self.branches=branches
        self.filesperchunk=filesperchunk
        self.filetype=filetype
        self.path=path
        self.nproc=nproc
        self.atype=atype
        self.dirname=dirname
        self.maxind=maxind
    def Run(self):


        for ffi in self.fileset:
            curpath=self.path+self.dirname+"/"+ffi
            Path(curpath).mkdir(parents=True, exist_ok=False)
            print ("Converting set",ffi)
            if self.nproc>1:

                pool = mp.Pool(self.nproc)
                splitfiles=np.array_split(np.array(self.fileset[ffi]),self.nproc)

                results = [pool.apply_async(self.Convert, args=(ffi,spf,ispf)) for ispf,spf in enumerate(splitfiles)]
                pool.close()    
                resarr = [result.get() for result in results]


            else:
                self.Convert(ffi,self.fileset[ffi])

    
            

    def Convert(self,setname,filearr,proc=1):

            totf=0
            curpath=self.path+self.dirname+"/"+setname

            reset=True
            nfiles=len(filearr)
            #print(proc,nfiles,filearr)

            if isinstance(self.filesperchunk,int):
                fpcproc=self.filesperchunk
            else:
                fpcproc=self.filesperchunk[setname]

            nchunks=int((nfiles-1)/fpcproc)+1
            flatten=False
            if self.atype=="flat":
                flatten=True
            disable=False
            if (self.nproc>1):
                disable=True
            with tqdm(total=(nfiles*len(self.branches))) as pbar:
                with tqdm(total=(fpcproc), disable=disable)  as pbar1:
                        bdict=OrderedDict([("",self.branches)])
                        if isinstance(self.branches,OrderedDict) or isinstance(self.branches,dict):
                            bdict=self.branches
                        
                        for ibmaj,bmaj in enumerate(bdict):
                            pbar.desc="Set:"+setname+" Collection:"+bmaj+" Process:"+str(proc)
                            scalar=False
                            if self.maxind!=None:   
                                if (self.maxind[bmaj]==None):
                                    scalar=True
                            nchunk=0
                        
                            delim="_"
                            if bmaj=="":
                                delim=""
                            brancharr=[bmaj+delim+x for x in bdict[bmaj]]
                            if not scalar:
                                if self.atype=="flat": 
                                    brancharr.append("n"+bmaj)
                            #if bmaj!="":
                             #   brancharr.append("event")
                            fullout=pd.DataFrame()
                            for ibatch,batch in enumerate(uproot3.pandas.iterate(path=filearr,flatten=flatten,treepath="Events", branches=brancharr,entrysteps=float("inf"))):
                                    if self.maxind!=None:
                                        idx = pd.IndexSlice
                                       
                                        if scalar:
                                            batch['subentry'] = batch.groupby(level=0).cumcount()
                                            batch=batch.set_index('subentry', append=True)
                                        else:
                                            batch=batch.loc[idx[:,:self.maxind[bmaj]],idx[:]]
                                        
                                    if self.atype=="pandas":
                                        batch=batch.reset_index()
                                  
                                    fullout = pd.concat((fullout,batch))

                                    if (ibatch%fpcproc==0 and (ibatch>0)) or (ibatch==(nfiles-1)):

                                        fname=curpath+"/"+setname+"_"+str(proc)+"_Chunk"+str(nchunk+1)+"of"+str(nchunks)+".parquet"


                                        if self.atype=="awk": 
                                            fullout=(pa.table(fullout))
                                        if (self.filetype=="parquet"):
                                            if self.atype=="awk":
                                                pq.write_table(fullout,fname)
                                            else:

                                                if self.atype=="flat": 
 
                                                    if ibmaj>0:
                                                        loaded=pd.read_parquet(fname)
                                                        fullout=pd.concat([fullout,loaded], axis=1,verify_integrity=True)
                                                       
                                                    for col in fullout:
                                                        if fullout[col].isnull().values.all():
                                                            raise ValueError("Empty DataFrame",col,fullout)
                                                    fullout.to_parquet(fname)
                                                    fullout=pd.DataFrame()
                                                if self.atype=="pandas": 
                                                    fullout.to_parquet(fname)
                                            reset=True
                                        else:
                                            raise(self.filetype+" not implemented")
                                        if (self.nproc==1):
                                            if not (ibatch==(nfiles-1)):
                                                pbar1.total=min(fpcproc,(nfiles-ibatch))
                                                pbar1.refresh() 
                                                pbar1.reset() 
                                        nchunk+=1
                                    pbar.update(1)

                                    pbar1.update(1)
                                    pbar1.update(0)
                                    totf+=1     
             

