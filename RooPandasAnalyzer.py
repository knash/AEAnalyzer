from RooPandasFunctions import PSequential,PColumn,PFilter,PRow,PProcessor,PProcRunner,PInitDir
import pandas as pd
from glob import glob
from ROOT import TH1F,TH2F,TLorentzVector,TFile
from collections import OrderedDict
import numpy as np
import copy
import pyarrow as pa

#PInitDir parses a directory created at nanoaod conversion time.  It creates dataset and file lists to be used by the processor
chunklist =PInitDir("RooFlatFull")

#Histograms are stored as a dict, and accessed through matching dict keys and DataFrame labels
histostemp=OrderedDict  ([
                        ("invm",TH1F("invm","invm",100,0,5000)),
                        ("pt",TH1F("pt","pt",100,0,2000)),
                        ("tight90pt1",TH1F("tight90pt1","tight90pt1",100,0,2000)),
                        ("loose90pt1",TH1F("loose90pt1","loose90pt1",100,0,2000)),
                        ("Et",TH1F("Et","Et",100,0,5000)),
                        #2D histogram.  parse variables with a "__" delimeter
                        ("invm__logMSE",TH2F("invm__logMSE","invm__logMSE",100,0,5000,80,-20,0)),
                        ("deta",TH1F("deta","deta",50,0,5.0)),
                        ("mindetaak4",TH1F("mindetaak4","mindetaak4",50,0,5.0)),
                        ("logMSE",TH1F("logMSE","logMSE",80,-20,0)),
                        ("logMSEall",TH1F("logMSEall","logMSEall",80,-20,0)),
                        ("logMSEshift",TH1F("logMSEshift","logMSEshift",80,-20,0))
                        ])
for hist in histostemp:
    histostemp[hist].Sumw2() 

#This takes the histograms and makes a copy for each dataset
histos= {}
for ds in chunklist:
    histos[ds]=copy.deepcopy(histostemp)

#The analysis uses PColumn,PFilter, and PRow actions which all take in a python function as an argument and use it 
#to efficiently analyze a DataFrame.

#Here are some functions that will later be used as input to these actions

#PFilter -- can be a function or class with a __call__ method
class KinematicSelection():
    def __init__(self,ptcut,msdcut):
        self.ptcut=ptcut
        self.msdcut=msdcut
    def __call__(self,df,EventInfo):
        #Kinematic filter operation.  For safety, the event-level filtering should only occur through the PFilter function.
        #This takes in a bool Series with one truth value per event.  This is very fast, and should be performed before any slower actions.
        #print(df["FatJet"])

        #njetcut is a elementwise selection used as  input to C1.  If you want to use this as an event filter, add [:,0] like in HLT selection below
        njetcut=df["FatJet"]["nFatJet"]>1


        #print(df["FatJet"])
        #passing None out of Filter will skip the entire current file (to avoid index errors)
        if (not njetcut.any()):
            return None
        #print(df["FatJet"]["pt"][njetcut])
        C1=((df["FatJet"]["pt"][njetcut][:,0]>self.ptcut) & (df["FatJet"]["pt"][njetcut][:,1]>self.ptcut) & (df["FatJet"]["msoftdrop"][njetcut][:,0]>self.msdcut) & (df["FatJet"]["msoftdrop"][njetcut][:,1]>self.msdcut))

        #Triggers are already stored as bools
        C2=df["HLT"]["PFHT900"][:,0]
        if (not (C1 & C2).any()):
            return None

        return ( C1 & C2 )


#PColumn function  
#creates an Et histogram before the filter (more entries)
class ColumnSelectionPre():
    def __call__(self,df,EventInfo):
        #Examples of columnwise actions. 
        #We can define a new collection variable (one item per entry).  Here we make fatjet et out of pt and mass
        df["FatJet"]["Et"]=np.sqrt(df["FatJet"]["pt"]*df["FatJet"]["pt"]+df["FatJet"]["mass"]*df["FatJet"]["mass"])
        df["Hists"]["Et"] = df["FatJet"]["Et"][:,0]

        #this creates the generic histrogram weights by taking the event weight and projecting to the event size
        #njetcut=df["FatJet"]["nFatJet"]==1
        ptcut=df["FatJet"]["pt"]>200.

        dfsel=df["FatJet"][ptcut]


        cut90,cut99,cut999=-11.3,-9.9,-9.2
        #print(cut90,EventInfo.dataset)
        logmse=np.log(dfsel["iAEMSE"])
        njettight=((logmse>cut90).groupby(level=0).sum())
        njetloose=((logmse<cut90).groupby(level=0).sum())
        #print(njetloose)
 
        #print(dfsel["pt"])
        #print(((msegroup<99999.).sum()))

        #this is printed at the end.  Should chain Mprocs for general solution

        #make sure there are any events left
        if len(njettight)>0:
            df["Hists"]["tight90pt1"] = dfsel["pt"][logmse>cut90][:,0]
            #print(dfsel["pt"][logmse>cut90])
            #print(dfsel["pt"][logmse>cut90].groupby(level=0).size())
        if len(njetloose)>0:
            df["Hists"]["loose90pt1"] = dfsel["pt"][logmse<cut90][:,0]

        EventInfo.eventcontainer["evweight"] = EventInfo.eventcontainer["lumi"]*EventInfo.eventcontainer["xsec"][EventInfo.dataset]/EventInfo.eventcontainer["nev"][EventInfo.dataset]



        #the  "weight" specific key will be used to weight all histograms unless there exists a histname__weight entry in the "Hists" dict
        #It is initialized as 1, so additional weights are multiplicative 
        df["Hists"]["weight"] *= EventInfo.eventcontainer["evweight"]
        df["Hists"]["Et"] = df["FatJet"]["Et"][:,0]
        df["Hists"]["logMSE_all"] = np.log(df["FatJet"]["iAEMSE"])

        return df

#one way to set the weights.  In general, each histogram needs a corresponding weights.
#Until we have all weights, we  just project the event weights to each histogram.
#You can skip this step and it will be done at histogram filling time automatically, but will be much slower and print a warning
#Probably need to find a better way to do this 
class ColumnWeights():
    def __call__(self,df,EventInfo):
        keys=list(df["Hists"].keys())
        for hh in keys:
            if hh in ["invm__logMSE","event","weight"]:
                continue
            df["Hists"][hh+"__weight"]=df["Hists"]["weight"]
            if (df["Hists"][hh].index.nlevels > df["Hists"]["weight"].index.nlevels )  :
                df["Hists"][hh]=df["Hists"][hh].droplevel(level=1)
            df["Hists"][hh+"__weight"] = df["Hists"][hh+"__weight"][df["Hists"][hh+"__weight"].index.isin(df["Hists"][hh].index)]
        df["Hists"]["invm__logMSE__weight"]=df["Hists"]["invm__weight"]
        return df

#PColumn function
class ColumnSelection():
    def __call__(self,df,EventInfo):
        #Examples of columnwise actions. 
        #We can define a new collection variable (one item per entry).  Here we make fatjet et out of pt and mass
        df["FatJet"]["Et"]=np.sqrt(df["FatJet"]["pt"]*df["FatJet"]["pt"]+df["FatJet"]["mass"]*df["FatJet"]["mass"])



        #We can store the leading two jet ht (one item per event) the "" collection is a key used for event-level info
        #print(df["FatJet"]["pt"])
        df[""]["dijetht"] = df["FatJet"]["pt"][:,0]+df["FatJet"]["pt"][:,1]    

        #We can also define variables for plotting.  Here, deta refers to the histogram above. 
        #The "Hists" collection is special, and holds all variables visible to the histogram filling
        df["Hists"]["deta"] = np.abs(df["FatJet"]["eta"][:,0]-df["FatJet"]["eta"][:,1])  
        df["Hists"]["pt"] = df["FatJet"]["pt"][:,0]  
        #Here is an example of a many-to-one operation where we loop through muons to find the closest (in eta) to the leading AK8 jet
        
        njets=df["Muon"].index.get_level_values(1).max()+1 #index+1 is number of objects 
        for ii in range(njets):
            curdiff=(np.abs(df["FatJet"]["eta"][:,0]-df["Muon"]["eta"][:,ii]))
            if ii==0:
                temp=curdiff
            else:
                temp=pd.concat([temp,curdiff],axis=1)
        df["Hists"]["mindetaak4"] = temp.min(axis=1)

        #df["Hists"]["weight"] *= EventInfo.eventcontainer["evweight"]
        #print(df["Hists"]["weight"])
        #You can also drop unused columns at any time -- here we drop the  pt,eta,phi,mass columns from the fatjet collection 
        #because we have stored the lorentzector already
        df["FatJet"] = df["FatJet"].drop(["pt","eta","phi","mass"],axis=1)
        return df

#PRow functions.  
#These are classes where the __call__ special function is performed in a rowwise loop
class MakePtEtaPhiMLV():
    #Example of a rowwise action.  These are slower than the columnwise selections, so best to do them last.
    #These are also much more general though, so likely they will be required.
    #These use the PRow() action which is a wrapper for the fastest method that I could find.
    #First, define your preprocessing steps so that you only consider the minimal number of columns
    def prepdf(self,df):
        args = [df["FatJet"]["pt"],df["FatJet"]["eta"],df["FatJet"]["phi"],df["FatJet"]["mass"]]
        return args

    #The args are then passed to the call function 
    def __call__(self,args,EventInfo):
        #Here, you can write general code as if you were acting on elements and not columns
        #This just takes pt,eta,phi, and mass and converts to a four vector,  This will be saved in the DataFrame and column passed below
        (pt,eta,phi,mass)=args
        LV=TLorentzVector()

        LV.SetPtEtaPhiM(pt,eta,phi,mass)
        return LV
class MyAnalyzerVec():
    #Another example of a rowwise action.  This will be likely replaced with the full analysis logic.
    #Here, we prepare the leading two jet invariant mass and MSE for plotting
    def prepdf(self,df):
        args = [df["FatJet"]["LV"][:,0],df["FatJet"]["LV"][:,1],df["FatJet"]["iAEMSE"][:,0]] 
        return args
    def __call__(self,args,EventInfo):
        (LV0,LV1,MSE)=args
        invm=(LV0 + LV1).M() 

        #This is where I can access the fake MSE shift object that was passed to the processor below
        msescale=EventInfo.eventcontainer["msescale"][EventInfo.dataset]

        return (invm,np.log(MSE),np.log(MSE*msescale))

#Dict of collections and variables to read in.
branchestoread={
                "Muon":["pt","eta","phi","mass"],
                "FatJet":["pt","eta","phi","mass","msoftdrop","iAEMSE"],
                "HLT":["PFHT900"],
                "":["run","luminosityBlock","event"]
                }
#Hacky, specifies which branches are single valued per event
scalars=["","HLT"]

#The analysis is defined here as a sequential list of actions
myana=  [

        PColumn(ColumnSelectionPre()),
        #PFilter just takes in a function that outputs a series of bools
        PFilter(KinematicSelection(200.,50.)),
        #PRow takes in two elements.  
        #The first describes the output collections and variables and the second is the function that will deliver them
        PRow([["FatJet","LV"]],MakePtEtaPhiMLV()),
        #PColumn just takes in a function that outputs a new dataframe
        PColumn(ColumnSelection()),
        #The collection here is "Hists" so we can plot these variables 
        PRow([["Hists","invm"],["Hists","logMSE"],["Hists","logMSEshift"]],MyAnalyzerVec()),
        PColumn(ColumnWeights()),
        ]


#Every action receives the "EventInfo" object.  This contains any state information (like current dataset or number of events processed)
#But also, you can pass anthing you want as well through the eventcontainer.  
#Here we pass some hypothetical dataset dependent MSE shift as an example.
#Also, objects can be passed through the class __init__ for each function  
evcont={"msescale":{"TT":1.10,"QCD_HT1500to2000":0.9},"lumi":(1000.0*137.65),"xsec":{"TT":0.047,"QCD_HT1500to2000":101.8},"nev":{"TT":305963.0,"QCD_HT1500to2000":10655313.0}}

#The processor just takes in all the peices

proc=PProcessor(chunklist,histos,branchestoread,myana,eventcontainer=evcont,atype="flat",scalars=scalars)
#Multiprocessor 
Mproc=PProcRunner(proc,6)
#Then runs them
returndf=Mproc.Run()
for rr in returndf:
    print  (rr ,"cut90",returndf[rr]["logMSE_all"].quantile(0.9))
    print  (rr ,"cut99",returndf[rr]["logMSE_all"].quantile(0.99))
    print  (rr ,"cut999",returndf[rr]["logMSE_all"].quantile(0.999))
#Finally, write out the histograms
output = TFile("FromFlatPandas.root","recreate")
output.cd()
for ds in histos:
    for var in histos[ds]:
            histos[ds][var].Write(ds+"__"+var)
output.Close()


