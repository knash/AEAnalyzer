import awkward0
import awkward as ak
import uproot3
import numpy as np
import ROOT
import pandas as pd
import time
from array import array
from collections import OrderedDict
from tqdm import tqdm
from pathlib import Path
import os
from glob import glob
import multiprocessing as mp
import pyarrow as pa
import pyarrow.parquet as pq
import itertools
import copy 
import functools
import warnings
class PSequential():
    def __init__(self,seq):
        self.seq=seq
    def __call__(self,df,cureeventinfo):
        timesummary={}
        for iseq,seq in enumerate(self.seq):
            inddict={}
            prekeys=df.keys()
            for branch in prekeys:
                
                if isinstance(df[branch],pd.DataFrame):
                    #if branch=="FatJet":
                     #   print(df[branch].shape)
                    inddict[branch]=(df[branch].index)
            #print(type(seq))

            df=seq(df,cureeventinfo)

            if  isinstance(df,type(None)):
                return df

            postkeys=df.keys()
            if (not isinstance(seq,PFilter)):
                for branch in prekeys:
                    if branch in postkeys:
                        if isinstance(df[branch],pd.DataFrame):
                            #if branch=="FatJet":
                            #    print(df[branch].shape)
                            if (not df[branch].index.identical(inddict[branch])):
                                 raise ValueError("Only PFilter should change indexing, not",type(seq))
              
        return(df,timesummary)


class PFilter():
    def __init__(self,func):
        self.func=func
    def __call__(self,df,cureeventinfo):

        Cond=self.func(df,cureeventinfo)
        if ((not isinstance(Cond,pd.Series)) and (not isinstance(Cond,pd.DataFrame)) and (not isinstance(Cond,type(None)))):
            raise ValueError("PFilter output of type",type(Cond)," not supported" )
        if isinstance(Cond,type(None)):
            return Cond
        if (Cond.index.nlevels)>1:
            Cond= Cond.droplevel(level=1)
        for branch in df:
            if isinstance(df[branch],pd.DataFrame):
                #print("branch",branch)
                #if branch=="":
                 #   print(branch,len(df[branch]["event"].index.unique()))
                #else:
                #    print(branch,len(df[branch]["event"][:,0].index))
                df[branch]=df[branch][df[branch].index.get_level_values(0).isin(Cond[Cond].index,level=0)]
                #if branch=="":
                #    print(branch,len(df[branch]["event"].index.unique()))
                #else:
                #    print(branch,len(df[branch]["event"][:,0].index))
            if branch=="Hists":

                df[branch]["event"]=df[branch]["event"][df[branch]["event"].index.get_level_values(0).isin(Cond[Cond].index,level=0)]
                #print(df[branch]["event"])
                #print("post")
        return df

class PColumn():
    def __init__(self,func):
        self.func=func
    def __call__(self,df,cureeventinfo):
        df=self.func(df,cureeventinfo)
        return df

class PRow():
    def __init__(self,bname,func):
        self.func=func
        self.bname=bname

    def __call__(self,df,cureeventinfo):
        serarr=self.func.prepdf(df)
        if isinstance(serarr,type(None)):
            warnings.warn("None returned from Prow")
            return None
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
        #print(np.array(self.bname))
        cname=np.array(self.bname)[:,1]
        dictnames=np.array(self.bname)[:,0]
        #print(cname.shape)
        #print(cname)
        #print(indices.shape)
        #print(len(funcapply))

        Serdict=pd.DataFrame(funcapply,columns=cname,index=indices)

        for idname,dictname in enumerate(dictnames):

            if not dictname in df:
                df[dictname]=pd.DataFrame()  
            else:
                if dictname!="Hists":
                    Serdict[cname[idname]].index=df[dictname].index
                else:
                    #print (df[dictname]["event"].index)
                    #print (Serdict[cname[idname]].index.unique())
                    #print (df[dictname]["event"])
                    Serdict[cname[idname]].index=df[dictname]["event"].index.unique()
                    #print (Serdict[cname[idname]])
            if isinstance(df[dictname],dict):
                #print (Serdict)
                df[dictname][cname[idname]]=pd.DataFrame({cname[idname]:Serdict[cname[idname]]})

            else:
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
    rr=Proc.Run()
    #print (rr)
    return Proc.hists,rr

def FillHist(df,hists):
        #print (df)

        for ih,hh in enumerate(hists):

                if (isinstance(hists[hh],ROOT.TH2)):
                    axes=hh.split("__")
                    if len(axes)!=2:
                        raise ValueError("2D histograms need to be formatted as axis1__axis2")
                    
                else:
                    axes=[hh]
                #print(axes[0],df.keys())
                
                if not axes[0] in df:
                    #print(axes[0],"not found")
                    continue
                #else:
                 #   print(axes[0],df[axes[0]])
                nent=len(df[axes[0]])
                weights=array("d",[1.0]*nent)
                for caxis in axes:
                    if isinstance(df[caxis],pd.Series):
                        df[caxis]=pd.DataFrame({caxis:df[caxis]})
                        if (df[caxis].index.nlevels >1)  :
                            df[caxis]=df[caxis].droplevel(level=1)
                if not hh+"__weight" in df.keys():
                    df[hh+"__weight"]=pd.DataFrame({hh+"__weight":df["weight"]})
                else:
                    
                    if isinstance(df[hh+"__weight"],pd.DataFrame):
                        df[hh+"__weight"]=pd.Series(df[hh+"__weight"][hh+"__weight"])
                    df[hh+"__weight"]=pd.DataFrame({hh+"__weight":df[hh+"__weight"]})
                if (df[axes[0]].index.nlevels < df[hh+"__weight"].index.nlevels )  :
                            df[hh+"__weight"]=df[hh+"__weight"].droplevel(level=1)
                        
                if (len(df[hh+"__weight"])!=len(df[axes[0]])):
                    warnings.warn("Warning: Attempting to project weights for histogram, this should be done in the analyzer")
                    print(hh)
                    df[hh+"__weight"] = df[hh+"__weight"][df[hh+"__weight"].index.isin(df[axes[0]].index)]

                if (isinstance(hists[hh],ROOT.TH2)):
                    hists[hh].FillN(nent,array("d",df[axes[0]][axes[0]].values),array("d",df[axes[1]][axes[1]].values),array("d",df[hh+"__weight"][hh+"__weight"].values))
                else:
                    #print(df[hh][hh].values)
                    #print(df[hh+"__weight"][hh+"__weight"].values)
                    if((len(df[hh][hh].values)>0) and (len(df[hh+"__weight"][hh+"__weight"].values)==len(df[hh][hh].values))):
                        hists[hh].FillN(nent,array("d",df[hh][hh].values),array("d",df[hh+"__weight"][hh+"__weight"].values))
                    #else:
                        #print("Err")
                        #print(df[hh][hh].values,df[hh+"__weight"][hh+"__weight"].values)
                    
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
                    print("Running process over",self.nproc,"processors")
                    results = pool.map(RunProc ,[x for x in allprocs])
                    #results = [pool.apply_async(RunProc for CProc in allprocs)]
                    print("Done")

                    pool.close()    
                    timetot={}
                    cutflowtot={}
                    #resarr = results
                    for icproc,cproc in enumerate(results):

                        hists,histreturn=cproc
                        for ds in hists:
                            for histo in hists[ds]:
                                #print (self.Proc.hists[ds][histo].Integral(),hists[ds][histo].Integral())
                                self.Proc.hists[ds][histo].Add(hists[ds][histo])
                                #print (self.Proc.hists[ds][histo].Integral())
                                #print (ds,histo,self.Proc.hists[ds][histo].Integral())
                    #for ds in cutflowtot:
                     #       print("Timing...")

                      #      for benchmark in timetot[ds]:
                       #         print ("\t",benchmark,timetot[ds][benchmark])
                        #    print ("Dataset:",ds,"Completed")
                         #   print ("Events input:",cutflowtot[ds][0],"output:",cutflowtot[ds][1])
                    print("Done")
                    

            else:
                    histreturn=self.Proc.Run(crange)
                    timetot={}
                    cutflowtot={}
                 
                    #for ds in cutflowtot:
                     #       print("Timing...")

                      #      for benchmark in timetot[ds]:
                       #         print ("\t",benchmark,timetot[ds][benchmark])
                        #    print ("Dataset:",ds,"Completed")
                         #   print ("Events input:",cutflowtot[ds][0],"output:",cutflowtot[ds][1])


            print("Total time",time.time()-fulltime)
            return histreturn

class PProcessor():
    def __init__(self,files,hists,branches,sequence,proc=1,atype="flat",eventcontainer={},scalars=[],verbose=True,rhistlist=[]):
        self.files=files
        self.hists=hists 
        self.rhistlist=rhistlist 
        self.scalars=scalars 
        self.branches=branches  
        self.proc=proc 
        self.cutflow={}
        self.atype=atype
        self.eventcontainer=eventcontainer
        self.ana=PSequential(sequence)
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
        return (returnval)

    def RunChunks(self,ds,selfiles):
                
            self.cutflow=[0,0]
            if self.verbose:
                print ("-"*20)
                print ("Dataset:",ds,"Started")
            fillH=True
            timingagg=None
            histreturn={}
         
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

                            
                            df[bmaj]=dffull[brancharr]
                            namemap = {x:x.replace(bmaj+"_","") for x in brancharr}

                            df[bmaj]=df[bmaj].rename(columns=namemap)
                            #print(df[bmaj])
                            df[bmaj]=df[bmaj].dropna(how='all')
                            if bmaj=="":
                                df[bmaj]=df[bmaj].dropna(how='any')
                            if bmaj!="":
                                    df[bmaj]=pd.concat((df[bmaj],dffull["event"]),axis=1,join='inner')
                                    #print(df[bmaj])

                    df["Hists"]={}
                    df["Hists"]["event"]=pd.Series(df[""]["event"],index=df[""].index)
                    df["Hists"]["weight"]=pd.Series(array("d",[1.0]*len(df[""]["event"])),index=df[""].index)
                    


                    #print(df["Hists"])
                    df[""]=df[""].droplevel(level=1)
                    df["Hists"]["event"]=df["Hists"]["event"].droplevel(level=1)

                    df["Hists"]["weight"]=df["Hists"]["weight"].droplevel(level=1)
                    #print(df["Hists"])
                    prekeys=df.keys()
                    nevchunk=df[""].shape[0]
                   
                    self.cutflow[0]+= nevchunk
                    
                    cureeventinfo=PEventInfo(ds,self.cutflow[0],ichunk,nevchunk,self.eventcontainer)
                    self.timing["Parsed"]=(time.time())
                    retval=self.ana(df,cureeventinfo)

                    if  isinstance(retval,type(None)):
                            #print("Skip",ichunk)
                            pbar.update(1)
                            continue

                    df = retval[0]
                
                    timing = retval[1]
                    for tt in timing:
                        self.timing[tt]=timing[tt]
                    if  isinstance(df,type(None)):
                            #print("Skip",ichunk)
                            pbar.update(1)
                            continue
                    self.timing["Analyzed"]=(time.time())
                    self.cutflow[1]+= df[""].shape[0]
                    eventlist=df[""]["event"]
                    
                    for branch in prekeys:
                            #tofix
                            if branch!="" and  isinstance(df[branch],pd.DataFrame):
                                #print(len(df[branch]["event"][:,0]))
                                #sprint(len(eventlist.unique()))
                                br = df[branch]["event"].unique()==eventlist.unique()
                               
                                if not (br).all():
                                    raise ValueError("Events are not 1-to-1 in collection",branch)
                                    
                   
                    #FillHist(rr[ds][0],Proc.hists[ds])


                    for hh in df["Hists"]:
                            #print ("temp")
                            #print(self.hists[ds])
                            #print(hh , self.rhistlist)

                            if (not hh in self.rhistlist):
                                continue
                            #print("in!",hh)
                            #if (hh not in histreturn):
                             #   continue
                            if hh in histreturn:
                                histreturn[hh]=pd.concat((histreturn[hh],df["Hists"][hh]))
                            else:
                                histreturn[hh]=df["Hists"][hh]
                            #print ("--")
                            #print (hh,histreturn[hh])
                    #print("Fill")
                    FillHist(df["Hists"],self.hists[ds])

                    pbar.update(1)

                    if isinstance(timingagg,type(None)):
                        timingagg=self.timing
                    else:
                        for benchmark in self.timing:
                            timingagg[benchmark]+=self.timing[benchmark]
            return(histreturn,timingagg,self.cutflow)

def PInitDir(path):
    filedict=OrderedDict()
    folders = os.listdir(path)
    for ff in folders:
        filedict[ff] = sorted(glob(path+"/"+ff+"/*"))
    return(filedict)
class PNanotoDataFrame():
    def __init__(self,fileset,branches,maxind=None,filesperchunk=5,filetype="parquet",path="./",nproc=1,atype="pandas",dirname="RooPandas",seq=None):
        self.fileset=fileset
        self.branches=branches
        self.filesperchunk=filesperchunk
        self.filetype=filetype
        self.path=path
        self.nproc=nproc
        self.atype=atype
        self.dirname=dirname
        self.maxind=maxind
        self.seq=seq
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
                        loaded={}
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
                                        if not nchunk in loaded.keys():
                                            loaded[nchunk]=pd.DataFrame()

                                        if self.atype=="awk": 
                                            fullout=(pa.table(fullout))
                                        if (self.filetype=="parquet"):
                                            if self.atype=="awk":
                                                pq.write_table(fullout,fname)
                                            else:

                                                if self.atype=="flat": 
 
                                                    loaded[nchunk]=pd.concat([fullout,loaded[nchunk]], axis=1)
                                                    #print(loaded[nchunk].columns)
                                                    for col in loaded[nchunk]:
                                                        if loaded[nchunk][col].isnull().values.all():
                                                            raise ValueError("Empty DataFrame",col,loaded[nchunk])
                                                    if ibmaj==len(bdict)-1:
                                                        if self.seq!=None:

                                                            (loaded[nchunk],_)=PSequential(self.seq)(loaded[nchunk],None)
                                                            #print(loaded[nchunk])
                                                            loaded[nchunk].loc[:,'event']=loaded[nchunk].loc[:,'event'].fillna(method="ffill")
                                                            #print(loaded[nchunk])
                                                            loaded[nchunk].to_parquet(fname)
                                        
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
             

