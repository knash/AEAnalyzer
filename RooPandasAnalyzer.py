from RooPandasFunctions import PSequential,PColumn,PFilter,PRow,PProcessor,PInitDir
import pandas as pd
from glob import glob
from ROOT import TH1F,TLorentzVector,TFile
from collections import OrderedDict
import numpy as np
import copy
import pyarrow as pa

#PInitDir parses a directory created at nanoaod conversion time.  It creates dataset and file lists to be used by the processor
chunklist =PInitDir("RooFlatFull")

#Histograms are stored as a dict, and accessed through matching dict keys and DataFrame labels
histostemp=OrderedDict  ([
                        ("invm",TH1F("invm","invm",100,0,5000)),
                        ("deta",TH1F("deta","deta",50,0,5.0)),
                        ("mindetaak4",TH1F("mindetaak4","mindetaak4",50,0,5.0)),
                        ("logMSE",TH1F("logMSE","logMSE",80,-20,0)),
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
    
        #passing None out of Filter will skip the entire current file (to avoid index errors)
        if (not njetcut.any()):
            return None
        #print(df["FatJet"]["pt"][njetcut])
        C1=((df["FatJet"]["pt"][njetcut][:,0]>self.ptcut) & (df["FatJet"]["pt"][njetcut][:,1]>self.ptcut) & (df["FatJet"]["msoftdrop"][njetcut][:,0]>self.msdcut) & (df["FatJet"]["msoftdrop"][njetcut][:,1]>self.msdcut))

        #Triggers are already stored as bools
        C2=df["HLT"]["PFHT900"][:,0]
        return ( C1 & C2 )

#PColumn function
class ColumnSelection():
    def __call__(self,df,EventInfo):
        #Examples of columnwise actions. The dataframe is modified inplace 
        #We can define a new collection variable (one item per entry).  Here we make fatjet et out of pt and mass
        df["FatJet"]["Et"]=np.sqrt(df["FatJet"]["pt"]*df["FatJet"]["pt"]+df["FatJet"]["mass"]*df["FatJet"]["mass"])



        #We can store the leading two jet ht (one item per event) the "" collection is a key used for event-level info
        #print(df["FatJet"]["pt"])
        df[""]["dijetht"] = df["FatJet"]["pt"][:,0]+df["FatJet"]["pt"][:,1]    

        #We can also define variables for plotting.  Here, deta refers to the histogram above. 
        #The "Hists" collection is special, and holds all variables visible to the histogram filling
        df["Hists"]["deta"] = np.abs(df["FatJet"]["eta"][:,0]-df["FatJet"]["eta"][:,1])  

        #Here is an example of a many-to-one operation where we loop through ak4 jets to find the closest (in eta) to the leading AK8 jet
        
        njets=df["Jet"].index.get_level_values(1).max()+1 #index+1 is number of objects 
        for ii in range(njets):
            curdiff=(np.abs(df["FatJet"]["eta"][:,0]-df["Jet"]["eta"][:,ii]))
            if ii==0:
                temp=curdiff
            else:
                temp=pd.concat([temp,curdiff],axis=1)
        df["Hists"]["mindetaak4"] = temp.min(axis=1)

        #You can also drop unused columns at any time -- here we drop the  pt,eta,phi,mass columns from the fatjet collection 
        #because we have stored the lorentzector already
        df["FatJet"] = df["FatJet"].drop(["pt","eta","phi","mass"],axis=1)


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
                "Jet":["pt","eta","phi","mass"],
                "FatJet":["pt","eta","phi","mass","msoftdrop","iAEMSE"],
                "HLT":["PFHT900"],
                "":["run","luminosityBlock","event"]
                }
#Hacky, specifies which branches are single valued per event
scalars=["","HLT"]

#The analysis is defined here as a sequential list of actions
myana=  [
        #PFilter just takes in a function that outputs a series of bools
        PFilter(KinematicSelection(200.,50.)),
        #PRow takes in two elements.  
        #The first describes the output collections and variables and the second is the function that will deliver them
        PRow([["FatJet","LV"]],MakePtEtaPhiMLV()),
        #PColumn just takes in a function that outputs a new dataframe
        PColumn(ColumnSelection()),
        #The collection here is "Hists" so we can plot these variables 
        PRow([["Hists","invm"],["Hists","logMSE"],["Hists","logMSEshift"]],MyAnalyzerVec())
        ]


#Every action receives the "EventInfo" object.  This contains any state information (like current dataset or number of events processed)
#But also, you can pass anthing you want as well through the eventcontainer.  
#Here we pass some hypothetical dataset dependent MSE shift as an example.
#Also, objects can be passed through the class __init__ for each function  
evcont={"msescale":{"TT":1.10,"QCD_HT1500to2000":0.9}}

#The processor just takes in all the peices
proc=PProcessor(chunklist,histos,branchestoread,myana,nproc=1,eventcontainer=evcont,atype="flat",scalars=scalars)
#Then runs them
proc.Run()

#Finally, write out the histograms
output = TFile("FromFlatPandas.root","recreate")
output.cd()
for ds in histos:
    for var in histos[ds]:
            histos[ds][var].Write(ds+"__"+var)
output.Close()

