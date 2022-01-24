#!/usr/bin/env python
# coding: utf-8

# In[1]:


from RooPandasFunctions import PSequential,PColumn,PFilter,PRow,PProcessor,PProcRunner,PInitDir
import pandas as pd
from glob import glob
import ROOT
from ROOT import TH1F,TH2F,TLorentzVector,TFile,TCanvas,TLegend,gPad
from collections import OrderedDict
import numpy as np
import copy
import pyarrow as pa
import array
from optparse import OptionParser
import subprocess,os,sys
import time


# In[2]:


parser = OptionParser()

parser.add_option('-p', '--nproc', metavar='F', type='string', action='store',
                  default	=	'6',
                  dest		=	'nproc',
                  help		=	'nproc')

parser.add_option('-n', '--njet', metavar='F', type='string', action='store',
                  default	=	'3',
                  dest		=	'njet',
                  help		=	'njet')

parser.add_option('-m', '--massrange', metavar='F', type='string', action='store',
                  default	=	'all',
                  dest		=	'massrange',
                  help		=	'0,1,2,3,all')

parser.add_option('-a', '--aeval', metavar='F', type='string', action='store',
                  default	=	'95',
                  dest		=	'aeval',
                  help		=	'90,95,99')

parser.add_option('-t', '--toys', metavar='F', type='string', action='store',
                  default	=	'2',
                  dest		=	'toys',
                  help		=	'toys')

parser.add_option('--quickrun', metavar='F', action='store_true',
		  default=False,
		  dest='quickrun',
		  help='quickrun')

parser.add_option('--qcdonly', metavar='F', action='store_true',
		  default=False,
		  dest='qcdonly',
		  help='qcdonly')


(options, args) = parser.parse_args()
op_nproc=int(options.nproc)
op_njet=int(options.njet)

op_massrange=options.massrange
op_aeval=options.aeval


# In[3]:



ntoys=int(options.toys)
quickrun=options.quickrun
qcdonly=options.qcdonly
if quickrun:
    op_nproc=1


# In[4]:


ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True


# In[5]:


#this creates histos and  weights before any selection
class PreColumn():
    def __call__(self,df,EventInfo):
        EventInfo.eventcontainer["evweight"] = EventInfo.eventcontainer["lumi"]*EventInfo.eventcontainer["xsec"][EventInfo.dataset]/EventInfo.eventcontainer["nev"][EventInfo.dataset]
        df["Hists"]["logMSE_all"] = np.log(df["FatJet"]["iAEMSE"])

        df["Hists"]["weight"] *= EventInfo.eventcontainer["evweight"]
        #meh, should be added to  columnweights -- todo
        df["Hists"]["logMSE_all__weight"] = pd.Series(EventInfo.eventcontainer["evweight"], df["Hists"]["logMSE_all"].index, name="logMSE_all__weight")
        return df


# In[6]:


#Select jetwise and eventwise. Exactly 4 jets with pt in region X, and all have sdmass in region Y
class KinematicSelection():
    def __init__(self,njet,ptcut,msdcut):
        self.ptcut=ptcut
        self.njet=njet
        self.msdcut=msdcut
    def __call__(self,df,EventInfo):
        
        fjcutpt=(df["FatJet"]["pt"]>self.ptcut[0]) &(df["FatJet"]["pt"]<self.ptcut[1]) 
        df["FatJet"]=df["FatJet"][fjcutpt]
        C1=(df["FatJet"]["event"].count(level=0))==self.njet

        fjcuteta=(df["FatJet"]["eta"].abs()>0.0) &(df["FatJet"]["eta"].abs()<2.4) 
        df["FatJet"]=df["FatJet"][fjcuteta]
        C3=(df["FatJet"]["event"].count(level=0))==self.njet

 

        fjcutmass=(df["FatJet"]["msoftdrop"]>self.msdcut[0])&(df["FatJet"]["msoftdrop"]<self.msdcut[1])
        df["FatJet"]=df["FatJet"][fjcutmass]

        C2=(df["FatJet"]["event"].count(level=0))==self.njet

        fjcut=fjcutpt&fjcutmass&fjcuteta
        C0=((fjcut).sum(level=0)>0)
   

        if (not ( C0 & C1 & C2 & C3).any()):
            return None
        return ( C0 & C1 & C2 & C3)


# In[7]:


#Select DeltaR cut to make sure AK8 jets are separated
class KinematicSelectionDR():
    def __init__(self,njet,drcut):
        self.drcut=drcut
        self.njet=njet
    def __call__(self,df,EventInfo):    
        alldiscs=[]

        for ijet in range(self.njet):
            #todo: find better way to check for nulls
            try:
                ijetphi=df["FatJet"]["phi"][:,ijet]
                ijeteta=df["FatJet"]["eta"][:,ijet]
            except:
                print ("ERR")
                print (df["FatJet"]["phi"])
                print (df["FatJet"]["eta"])
                return None

            drcutjet=None
            for jjet in range(self.njet):

                if ijet==jjet:
                    continue
            
                jjetphi=df["FatJet"]["phi"][:,jjet]
                jjeteta=df["FatJet"]["eta"][:,jjet]

                deta=(jjeteta-ijeteta).abs()
                dphi=(jjetphi-ijetphi).abs()

                dphi[dphi>3.1415]=2*3.1415-dphi

                dr=np.sqrt(dphi*dphi+deta*deta)
                curcond=dr>self.drcut
                #print(curcond)
                if isinstance(drcutjet,type(None)):
                    drcutjet=curcond
                else:
                    drcutjet=drcutjet&(curcond)

            alldiscs.append(drcutjet)

        for iad,ad in enumerate(alldiscs):
            if iad==0:
                evdisc=ad
            else:
                evdisc=evdisc&ad
        #print("evd",evdisc)

        return ( evdisc )


# In[8]:


#Create tight and loose jet tags
class MakeTags():
    def __init__(self,njet):
        self.njet=njet
    def __call__(self,df,EventInfo):


        cut90,cut95,cut99=-11.28,-10.74,-9.9
        logmse=np.log(df["FatJet"]["iAEMSE"])
        
        if op_aeval=="90":
            AEcut=cut90
        elif op_aeval=="95":
            AEcut=cut95
        elif op_aeval=="99":
            AEcut=cut99
        else:
            raise ValueError("Bad AE cut")

        njettight=((logmse>AEcut).sum(level=0))
        njetloose=((logmse<AEcut).sum(level=0))

        df["FatJet"]["tight"] = logmse>AEcut
        df["FatJet"]["loose"] = logmse<AEcut

        df["Hists"]["ht"]=df["FatJet"]["pt"].sum(level=0)
        df["Hists"]["njettight"] = njettight
        df["Hists"]["njetloose"] = njetloose
        df["Hists"]["abseta"]=df["FatJet"]["eta"].abs()


        return df


# In[9]:


#project weights
class ColumnWeights():
    def __call__(self,df,EventInfo):
        keys=list(df["Hists"].keys())
        for hh in keys:
            if hh in ["njettight__njetloose","event","weight"]:
                continue
            if hh+"__weight" in df["Hists"]:
                continue
            df["Hists"][hh+"__weight"]=df["Hists"]["weight"]
            if (df["Hists"][hh].index.nlevels > df["Hists"]["weight"].index.nlevels )  :
                df["Hists"][hh]=df["Hists"][hh].droplevel(level=1)

            df["Hists"][hh+"__weight"] = df["Hists"][hh+"__weight"][df["Hists"][hh+"__weight"].index.isin(df["Hists"][hh].index)]
         

        df["Hists"]["njettight__njetloose__weight"]=df["Hists"]["njettight__weight"]
        return df


# In[10]:


#make histograms to be used for creating the pass-to-fail ratio
class MakeHistsForBkg():
    def __init__(self,njet):
        self.njet=njet
    def __call__(self,df,EventInfo):
        bkgparam=EventInfo.eventcontainer["bkgparam"]
        for ijet in range(self.njet+1):
            for ebin in bkgparam["eta"]:
                #tightreq=(df["Hists"]["njettight"]==1) | (df["Hists"]["njettight"]==0)
                tightreq=True
                abseta=df["FatJet"]["eta"].abs()

                etacut=(bkgparam["eta"][ebin][0]<=abseta)&(abseta<bkgparam["eta"][ebin][1])

                try:
                    df["Hists"]["ptT"+str(ijet)+"_"+ebin]=df["FatJet"]["pt"][df["FatJet"]["tight"]][etacut][:,ijet]
                except:
                    pass
                try:
                    df["Hists"]["ptL"+str(ijet)+"_"+ebin]=df["FatJet"]["pt"][df["FatJet"]["loose"]][etacut][:,ijet]
                except:
                    pass
                try:
                    df["Hists"]["ptT21"+str(ijet)+"_"+ebin]=df["FatJet"]["pt"][df["FatJet"]["tight"]][etacut][:,ijet][df["Hists"]["njettight"]==1][df["Hists"]["njetloose"]==2]
                except:
                    pass
                try:
                    df["Hists"]["ptL21"+str(ijet)+"_"+ebin]=df["FatJet"]["pt"][df["FatJet"]["loose"]][etacut][:,ijet][df["Hists"]["njettight"]==1][df["Hists"]["njetloose"]==2]
                except:
                    pass
                try:
                    df["Hists"]["ptL30"+str(ijet)+"_"+ebin]=df["FatJet"]["pt"][df["FatJet"]["loose"]][etacut][:,ijet][df["Hists"]["njettight"]==0][df["Hists"]["njetloose"]==3]
                except:
                    pass

            
            regionstr="LT"+str(ijet)+str(njet-ijet)
            df["Hists"]["ht_"+regionstr]=df["Hists"]["ht"][df["Hists"]["njettight"]==(njet-ijet)][df["Hists"]["njetloose"]==(ijet)]
            df["Hists"]["abseta_"+regionstr]=df["Hists"]["abseta"][:,0][df["Hists"]["njettight"]==(njet-ijet)][df["Hists"]["njetloose"]==(ijet)]
            for jjet in range(njet):
                df["Hists"]["pt"+str(jjet)+"_"+regionstr]=df["FatJet"]["pt"][:,jjet][df["Hists"]["njettight"]==(njet-ijet)][df["Hists"]["njetloose"]==(ijet)]
                df["Hists"]["ptT"+str(jjet)+"_"+regionstr]=df["FatJet"]["pt"][df["FatJet"]["tight"]][:,jjet][df["Hists"]["njettight"]==(njet-ijet)][df["Hists"]["njetloose"]==(ijet)]
                df["Hists"]["ptL"+str(jjet)+"_"+regionstr]=df["FatJet"]["pt"][df["FatJet"]["loose"]][:,jjet][df["Hists"]["njettight"]==(njet-ijet)][df["Hists"]["njetloose"]==(ijet)]   
        return df


# In[11]:


#use pass-to-fail ratio created in step0 to predict background
#todo: Sometimes returns none -- look into
class BkgEst():
    
    def __init__(self,njet):
        self.njet=njet
    
    def prepdf(self,df):
        args=[df["Hists"]["ht"]]
        try:
            for ijet in range(self.njet):
                args.append(df["FatJet"]["pt"][:,ijet])
                args.append(df["FatJet"]["eta"][:,ijet].abs())
                args.append(df["FatJet"]["tight"][:,ijet])
                args.append(df["FatJet"]["loose"][:,ijet])
        except Exception as e:
            print (e)
            return None
        return args
    
    def __call__(self,args,EventInfo):
        
        bkgparam=EventInfo.eventcontainer["bkgparam"]
        RateHists=EventInfo.eventcontainer["RateHists"]
        RateHistsFULL=EventInfo.eventcontainer["RateHistsFULL"]
        #print(RateHistsFULL)
        ht=args[0]
        pt=[]
        eta=[]
        tight=[]
        loose=[]
        ptTIGHT=[]
        ptLOOSE=[]
        
        for ijet in range(self.njet):
            pt.append(args[ijet*4+1])
            eta.append(args[ijet*4+2])
            tight.append(args[ijet*4+3])
            loose.append(args[ijet*4+4])
            
            regionstr="LT"+str(ijet)+str(njet-ijet)
        
        nloosetr=0
        for ll in loose:
            nloosetr+=ll
        ntighttr=0
        for tt in tight:
            ntighttr+=tt

        #if ((nloose)==self.njet):
        
        maxbin=2**self.njet
        allregs=list(range(maxbin))
        allregs.reverse()
        Trate=[]
        Lrate=[]
        Trateup=[]
        Lrateup=[]   
        Tratedown=[]
        Lratedown=[] 
        jetebin=[] 
        jetptbin=[] 

        usefullrate=True

 
        Trateetajet=[]

        for ijet in range(self.njet):
            Trateetajet.append([])
            for iebin,ebin in enumerate(bkgparam["eta"]):
                Trateetajet[ijet].append(0.0)
                etacut=(bkgparam["eta"][ebin][0]<=eta[ijet]<bkgparam["eta"][ebin][1])
                
                if etacut:
                    

                    if usefullrate:

                        ptbin=RateHistsFULL["Rate"+ebin+"jet"+str(ijet)].FindBin(pt[ijet])
                        TRtemp=RateHistsFULL["Rate"+ebin+"jet"+str(ijet)].GetBinContent(ptbin)
                        TRLtemp=RateHistsFULL["RateL"+ebin+"jet"+str(ijet)].GetBinContent(ptbin)
                        TRtemperr=RateHistsFULL["Rate"+ebin+"jet"+str(ijet)].GetBinError(ptbin)
                    else:
                        ptbin=RateHists["Rate"+ebin].FindBin(pt[ijet])
                        TRtemp=RateHists["Rate"+ebin].GetBinContent(ptbin)
                        TRLtemp=RateHists["RateL"+ebin].GetBinContent(ptbin)
                        TRtemperr=RateHists["Rate"+ebin].GetBinError(ptbin)

                    Trateetajet[ijet][iebin]=TRtemp

                    if ntighttr==0:
                            Trate.append(TRtemp)
                            Lrate.append(TRLtemp)
                            #Lrate.append(1.0)
                    else:
                            Trate.append(0.0)
                            Lrate.append(0.0)
                    jetebin.append(iebin)
                    jetptbin.append(ptbin)                  
                    
                    
        weights=[0.0]*(self.njet+1)
        weightsT=[0.0]*(self.njet+1)
        weightsL=[0.0]*(self.njet+1)
        weights1=[0.0]*(self.njet+1)
        nweights=[0.0]*(self.njet+1)
        #print(Trate)
        for ar in allregs:
            ntight=0
            nloose=0
            weight=1.0
            weightT=1.0
            weightL=1.0
            weight1=1.0
            for ibit,bit in enumerate(range(self.njet)):
                curbit=(ar>>bit)&1
                ntight+=curbit
                nloose+=(curbit==0)
            for ibit,bit in enumerate(range(self.njet)):
            
                curbit=(ar>>bit)&1

                #really not sure what the right one is.  
                if curbit:
                    weight*=Trate[ibit] #is this nonsense?
                    #I use weight1 now.  Not sure though -- this is sum(Rloose)+sum(Rtight) but from LT21 so like the Rloose per jet is Rloose/2.  
                    #Based on the pt agreement, this is average of loose and tight for all jets.  I think there is a better way to do this  
                    weight1*=(float(ntight)*Trate[ibit]/1.0+float(nloose)*Lrate[ibit]/2.0)/float(nloose+ntight) #is this nonsense? 

                    weightT*=float(ntight)*Trate[ibit]/1.0 #is this nonsense?
                    weightL*= 1.0#is this nonsense?

                else: 
                    weight*=1.0-Trate[ibit] #is this nonsense?
                    weight1*=1.0 #is this nonsense?
                    weightT*=1.0 #is this nonsense?
                    weightL*=float(nloose)*Lrate[ibit]/2.0 #is this nonsense?
            #print
            weights1[self.njet-ntight]+=weight1
            weights[self.njet-ntight]+=weight
            weightsT[self.njet-ntight]+=weightT
            weightsL[self.njet-ntight]+=weightL
            nweights[self.njet-ntight]+=1.0
        weightLT=list((np.array(weightsL)+np.array(weightsT)))
        #print("----")
        #print(weights)
        #print(weights1)
        #print(weightsT)
        #print(weightsL)
        #print(weightLT)

        #print()
        allret=[]

        #This is the weighted average for ht.  Not sure if this makes sense, each jet has a probabilistic weight so maybe something like this
        htw = 0.0
        denom = 0.0
        if ntighttr==0:
                for ijet in range(njet):
                            htw+=pt[ijet]*(Trate[ijet]+Lrate[ijet])
                            denom+=(Trate[ijet]+Lrate[ijet])/3.0
                htw/=denom
        #print(ht,htw)
        
        for icweight,cweight in enumerate(weights1):
            allret.append(ht)
            #allret.append(htw)

            #print(icweight,cweight)
            allret.append(cweight*EventInfo.eventcontainer["evweight"])
            
            for jjet in range(njet):

                #pt of all events-- Events where jet==jjet is tight + Events where jet==jjet is loose
                #ht is something like sum(pt) with this weight averged over three jets?
                allret.append(pt[jjet])
                allret.append((Trate[jjet]+Lrate[jjet])*EventInfo.eventcontainer["evweight"])

                #pt of all tight events-- Events where jet==jjet is tight + 0
                allret.append(pt[jjet])
                allret.append(Trate[jjet]*EventInfo.eventcontainer["evweight"])

                #pt of all loose events--  0 + Events where jet==jjet is loose
                allret.append(pt[jjet])
                allret.append(Lrate[jjet]*EventInfo.eventcontainer["evweight"])
                
                
                allret.append(jetebin[jjet])
                allret.append(jetptbin[jjet])

        return (allret)

class MakeToys():
    def __init__(self,njet):
        self.njet=njet
    def __call__(self,df,EventInfo):
        bkgparam=EventInfo.eventcontainer["bkgparam"]
                
        maxbin=2**self.njet
        allregs=list(range(maxbin))
        allregs.reverse()

        for ijet in range(self.njet+1):

            regionstr="LT"+str(ijet)+str(njet-ijet) 
            
            allt=EventInfo.eventcontainer["toys"]

            allteval=[]
            for jjet in range(njet):
                allteval.append(allt[df["Hists"]["ebin"+str(jjet)+"_"+regionstr],:,df["Hists"]["ptbin"+str(jjet)+"_"+regionstr]])
                allteval[-1]=allteval[-1].squeeze()

            for tt in range(ntoys):

                sumem=np.zeros(df["Hists"]["ht"].shape)
                times=[]
                if True:

                    resampleT=[]
                    resampleL=[]

                    for jjet in range(njet):
               

                        resampleT.append(allteval[jjet][:,tt])              
                        resampleL.append(np.ones(resampleT[-1].shape))             
                        resampleL[-1]=(resampleL[-1]-resampleT[-1])
                                 
                
                    for ar in allregs:
                        ntight=0
                        for ibit,bit in enumerate(range(self.njet)):
                            curbit=(ar>>bit)&1
                            if curbit:
                                ntight+=1
                        if ntight!=(njet-ijet):
                            continue
                        weight=np.ones(resampleT[-1].shape)
                        for ibit,bit in enumerate(range(self.njet)):

                            curbit=(ar>>bit)&1
                            
                            if curbit:
                                weight*=resampleT[ibit]    
                            else:
                                weight*=resampleL[ibit]  
                        sumem+=weight*EventInfo.eventcontainer["evweight"]
                df["Hists"]["bkg_ht_toy"+str(tt)+"_"+regionstr+"__weight"]=pd.Series(sumem,index=df["Hists"]["ht"].index)
                df["Hists"]["bkg_ht_toy"+str(tt)+"_"+regionstr]=df["Hists"]["ht"]


        return df


# In[13]:


chunklist=PInitDir("RooFlatFull")
for ds in chunklist:
        if qcdonly:
            if (ds.split("_")[0]!="QCD"):
                del chunklist[ds]

#print (chunklist)
bkgparam={}

#three eta bins (probably overkill)
bkgparam["eta"]={"E0":[0.0,0.4],"E1":[0.4,0.9],"E2":[0.9,1.3],"E3":[1.3,float("inf")]}
#bkgparam["eta"]={"E0":[0.0,float("inf")]}

#todo: muon triggers a failure mode as sometimes events have no muons and no filter remo 
branchestoread={
                    #"Muon":["pt","eta","phi","mass"],
                    "FatJet":["pt","eta","phi","mass","msoftdrop","iAEMSE"],
                    "":["run","luminosityBlock","event"]
                    }

scalars=[""]

if op_massrange=="all":
    sdcut=[0.0,float("inf")]
else:
    #sdcuts=[[0.0,50.0],[50.0,100.0],[100.0,140.0],[140.0,200.0],[200.0,float("inf")]]
    sdcuts=[[0.0,50.0],[50.0,float("inf")]]
    sdcut=sdcuts[int(op_massrange)]


# In[14]:


#customize a multi-step processor
def MakeProc(njet,step,evcont):
    histostemp=OrderedDict  ([])
    if step==0:
        for ijet in range(njet+1):
            regionstr="LT"+str(ijet)+str(njet-ijet)
            histostemp["ht_"+regionstr]=TH1F("ht_"+regionstr,"ht_"+regionstr,700,0,7000)
            histostemp["abseta_"+regionstr]=TH1F("abseta_"+regionstr,"abseta_"+regionstr,120,0,3.0)

            for jjet in range(njet):
                histostemp["pt"+str(jjet)+"_"+regionstr]=TH1F("pt"+str(jjet)+"_"+regionstr,"pt"+str(jjet)+"_"+regionstr,1000,0,10000)
                histostemp["ptT"+str(jjet)+"_"+regionstr]=TH1F("ptT"+str(jjet)+"_"+regionstr,"ptT"+str(jjet)+"_"+regionstr,1000,0,10000)
                histostemp["ptL"+str(jjet)+"_"+regionstr]=TH1F("ptL"+str(jjet)+"_"+regionstr,"ptL"+str(jjet)+"_"+regionstr,1000,0,10000)

            for ebin in bkgparam["eta"]:
                    histostemp["ptL"+str(ijet)+"_"+ebin]=TH1F("ptL"+str(ijet)+"_"+ebin,"ptL"+str(ijet)+"_"+ebin,1000,0,10000)
                    histostemp["ptT"+str(ijet)+"_"+ebin]=TH1F("ptT"+str(ijet)+"_"+ebin,"ptT"+str(ijet)+"_"+ebin,1000,0,10000)
                    histostemp["ptL30"+str(ijet)+"_"+ebin]=TH1F("ptL30"+str(ijet)+"_"+ebin,"ptL30"+str(ijet)+"_"+ebin,1000,0,10000)
                    histostemp["ptT21"+str(ijet)+"_"+ebin]=TH1F("ptT21"+str(ijet)+"_"+ebin,"ptT21"+str(ijet)+"_"+ebin,1000,0,10000)
                    histostemp["ptL21"+str(ijet)+"_"+ebin]=TH1F("ptL21"+str(ijet)+"_"+ebin,"ptL21"+str(ijet)+"_"+ebin,1000,0,10000)
        histostemp["logMSE_all"]=TH1F("logMSE_all","logMSE_all",100,-20.,0.)
        myana=  [
                PColumn(PreColumn()),
                PFilter(KinematicSelection(njet,[200.0,float("inf")],sdcut)), 
                PFilter(KinematicSelectionDR(njet,1.4)),
                PColumn(MakeTags(njet)),
                PColumn(MakeHistsForBkg(njet)),
                PColumn(ColumnWeights()),
                ]

    if step==1:
        hpass=[]

        for ijet in range(njet+1):
            regionstr="LT"+str(ijet)+str(njet-ijet)
            
            histostemp["bkg_ht_"+regionstr]=TH1F("bkg_ht_"+regionstr,"bkg_ht_"+regionstr,700,0,7000)
            
            hpass.append(["Hists","bkg_ht_"+regionstr])
            hpass.append(["Hists","bkg_ht_"+regionstr+"__weight"])

          
            
            for jjet in range(njet):
                
                histostemp["bkg_pt"+str(jjet)+"_"+regionstr]=TH1F("bkg_pt"+str(jjet)+"_"+regionstr,"bkg_pt"+str(jjet)+"_"+regionstr,1000,0,10000)
                hpass.append(["Hists","bkg_pt"+str(jjet)+"_"+regionstr])
                hpass.append(["Hists","bkg_pt"+str(jjet)+"_"+regionstr+"__weight"])

                histostemp["bkg_ptT"+str(jjet)+"_"+regionstr]=TH1F("bkg_ptT"+str(jjet)+"_"+regionstr,"bkg_ptT"+str(jjet)+"_"+regionstr,1000,0,10000)
                hpass.append(["Hists","bkg_ptT"+str(jjet)+"_"+regionstr])
                hpass.append(["Hists","bkg_ptT"+str(jjet)+"_"+regionstr+"__weight"])

                histostemp["bkg_ptL"+str(jjet)+"_"+regionstr]=TH1F("bkg_ptL"+str(jjet)+"_"+regionstr,"bkg_ptL"+str(jjet)+"_"+regionstr,1000,0,10000)
                hpass.append(["Hists","bkg_ptL"+str(jjet)+"_"+regionstr])
                hpass.append(["Hists","bkg_ptL"+str(jjet)+"_"+regionstr+"__weight"])
                
    
                hpass.append(["Hists","ebin"+str(jjet)+"_"+regionstr])
                hpass.append(["Hists","ptbin"+str(jjet)+"_"+regionstr])

         
            for itoy in range(ntoys):
                histostemp["bkg_ht_toy"+str(itoy)+"_"+regionstr]=TH1F("bkg_ht_toy"+str(itoy)+"_"+regionstr,"bkg_ht_toy"+str(itoy)+"_"+regionstr,700,0,7000)         

        #print("len(hpass)",len(hpass))        
                    
        myana=  [
                PColumn(PreColumn()),
                PFilter(KinematicSelection(njet,[200.0,float("inf")],sdcut)),     
                PFilter(KinematicSelectionDR(njet,1.4)),
                PColumn(MakeTags(njet)),
                PRow(hpass,BkgEst(njet)),
                PColumn(MakeToys(njet)),
                PColumn(ColumnWeights()),
                ]
    for hist in histostemp:
        histostemp[hist].Sumw2() 


    histos= {}
    for ds in chunklist:
        #print ("ds",ds)

        if quickrun:
            chunklist[ds]=chunklist[ds][:1]

        #chunklist[ds]=chunklist[ds][:12]
        histos[ds]=copy.deepcopy(histostemp)

    return PProcessor(chunklist,histos,branchestoread,myana,eventcontainer=evcont,atype="flat",scalars=scalars)


# In[15]:


njet=op_njet
evcont={"lumi":(1000.0*137.65),"xsec":{"WgWg":1.0,"TT":1.0,"QCD_HT1500to2000":101.8,"QCD_HT1000to1500":1005.0,"QCD_HT2000toInf":20.54},"nev":{"WgWg":18000.0,"TT":305963.0,"QCD_HT1500to2000":10655313.0,"QCD_HT1000to1500":12660521.0,"QCD_HT2000toInf":4980828.0}}
evcont["bkgparam"]=bkgparam


# In[16]:


#Step0:make hists for pass-to-fail ratio
proc = MakeProc(njet,0,evcont)
nproc=op_nproc
Mproc=PProcRunner(proc,nproc)
returndf=Mproc.Run()


# In[17]:


#Print MSE quantilles
#for rr in returndf:
#    if  "logMSE_all" in returndf[rr]:
#        print  (rr ,"cut90",returndf[rr]["logMSE_all"].quantile(0.90))
#        print  (rr ,"cut95",returndf[rr]["logMSE_all"].quantile(0.95))
#        print  (rr ,"cut99",returndf[rr]["logMSE_all"].quantile(0.99))


# In[18]:



ratehistos=copy.deepcopy(proc.hists)

qcdnames = ["QCD_HT1000to1500","QCD_HT1500to2000","QCD_HT2000toInf"]


# In[ ]:


#Make pass-to-fail ratio TR(pt,eta)
THists={}
LTHists={}
LHists={}
ALLHists={}

THistsFULL={}
LTHistsFULL={}
LHistsFULL={}

#print("START")

bins=array.array('d',[0,200,220,240,280,320,340,380,400,440,480,520,580,650,700,800,900,1000,1200,1500,2000,10000])
for ijet in range(njet):
    #print(ijet)

    for qcdname in qcdnames:
        #print(qcdname)

        QCDhists=ratehistos[qcdname]
        for curhist in QCDhists:
            #print(curhist)
            if curhist[:6] =="ptL30"+str(ijet):
                Lstring=curhist
                Tstring=curhist.replace("ptL30"+str(ijet),"ptT21"+str(ijet))
                LTstring=curhist.replace("ptL30"+str(ijet),"ptL21"+str(ijet))

                
                paramstr=Lstring.split("_")[-1]
                paramstrwjet=Lstring.split("_")[-1]+"jet"+str(ijet)
                

                curhistL=QCDhists[Lstring]
                curhistT=QCDhists[Tstring]
                curhistLT=QCDhists[LTstring]
                #print("Tstring",Tstring,curhistT.Integral())

                #print (ijet,qcdname,paramstr)
                #print (curhistT.Integral(),curhistL.Integral())

                if not(paramstr in THists):
                    THists[paramstr]=copy.deepcopy(curhistT)
                    LTHists[paramstr]=copy.deepcopy(curhistLT)
                    LHists[paramstr]=copy.deepcopy(curhistL)


                else:
                    THists[paramstr].Add(curhistT)
                    LTHists[paramstr].Add(curhistLT)
                    LHists[paramstr].Add(curhistL)

  


                if not(paramstrwjet in THistsFULL):
                    THistsFULL[paramstrwjet]=copy.deepcopy(curhistT)
                    LTHistsFULL[paramstrwjet]=copy.deepcopy(curhistLT)
                    LHistsFULL[paramstrwjet]=copy.deepcopy(curhistL)
                else:
                    THistsFULL[paramstrwjet].Add(curhistT)
                    LTHistsFULL[paramstrwjet].Add(curhistLT)
                    LHistsFULL[paramstrwjet].Add(curhistL)
     
                            
for tth in THists:
    THists[tth]=THists[tth].Rebin(len(bins)-1,THists[tth].GetName()+"TEMP",bins)    
for tth in LTHists:
    LTHists[tth]=LTHists[tth].Rebin(len(bins)-1,LTHists[tth].GetName()+"TEMP",bins)                    
for tth in LHists:
    LHists[tth]=LHists[tth].Rebin(len(bins)-1,LHists[tth].GetName()+"TEMP",bins)                    

for tth in THistsFULL:
    THistsFULL[tth]=THistsFULL[tth].Rebin(len(bins)-1,THistsFULL[tth].GetName()+"TEMP",bins)   
for tth in LTHistsFULL:
    LTHistsFULL[tth]=LTHistsFULL[tth].Rebin(len(bins)-1,LTHistsFULL[tth].GetName()+"TEMP",bins)                   
for tth in LHistsFULL:
    LHistsFULL[tth]=LHistsFULL[tth].Rebin(len(bins)-1,LHistsFULL[tth].GetName()+"TEMP",bins)       
    


# In[ ]:




import matplotlib.pyplot as plt
import matplotlib.image as mpimg

RateHists=OrderedDict([])
canvrate=TCanvas("canvrate","canvrate",700,500)
color=1

alltoys=[]
#print("LHists",LHists)
for RH in LHists:
    #print(RH)
    #print(THists[RH].Integral())
    #print(LHists[RH].Integral())
    
    RateHists["Rate"+RH]=copy.deepcopy(THists[RH])
    RateHists["Rate"+RH].Divide(RateHists["Rate"+RH],LHists[RH])#,1.0,1.0,"B")

    RateHists["RateL"+RH]=copy.deepcopy(LTHists[RH])
    RateHists["RateL"+RH].Divide(RateHists["RateL"+RH],LHists[RH])#,1.0,1.0,"B")

    means = []
    errs = []
    toys = []
    for xbin in range(RateHists["Rate"+RH].GetXaxis().GetNbins()+1):
        means.append(RateHists["Rate"+RH].GetBinContent(xbin))
        errs.append(RateHists["Rate"+RH].GetBinError(xbin))

    curtoys=[]
    for tt in range(ntoys):
        curtoys.append(np.random.normal(means,errs))
    #print(curtoys[0].shape)
    alltoys.append(curtoys)
    #print (curtoys)
    #print("Rate"+RH)
    RateHists["Rate"+RH].SetLineColor(color)
    RateHists["Rate"+RH].SetMarkerColor(color)
    RateHists["Rate"+RH].Draw("same")
    color+=1
    
RateHistsFULL=OrderedDict([])
    
for RH in LHistsFULL:
 
    RateHistsFULL["Rate"+RH]=copy.deepcopy(THistsFULL[RH])
    RateHistsFULL["Rate"+RH].Divide(RateHistsFULL["Rate"+RH],LHistsFULL[RH])#,1.0,1.0,"B")

    RateHistsFULL["RateL"+RH]=copy.deepcopy(LTHistsFULL[RH])
    RateHistsFULL["RateL"+RH].Divide(RateHistsFULL["RateL"+RH],LHistsFULL[RH])#,1.0,1.0,"B")

    

canvrate.Print('plots/Trate.png', 'png')

evcont["RateHists"]=RateHists
evcont["RateHistsFULL"]=RateHistsFULL

evcont["toys"]=np.array(alltoys)
#print(evcont["toys"].shape)

# In[ ]:


#Step1:use pass-to-fail ratio to predict background
proc = MakeProc(njet,1,evcont)
Mproc=PProcRunner(proc,nproc)
returndf=Mproc.Run()


# In[ ]:





# In[ ]:


histos=copy.deepcopy(proc.hists)


htosum={}
htosum["QCD"]=["QCD_HT1500to2000","QCD_HT1000to1500","QCD_HT2000toInf"]

histdicts=[histos,ratehistos]
for hdict in histdicts:
        for curh in htosum:
            hdict[curh]={}
            for var in hdict[htosum[curh][0]]:
                for curhsum in htosum[curh]:
                        if  var in hdict[curh]:
                                hdict[curh][var].Add(hdict[curhsum][var])
                        else:
                                hdict[curh][var] = copy.deepcopy(hdict[curhsum][var])
                                hdict[curh][var].SetName(hdict[curhsum][var].GetName().replace(curhsum,curh))
                                hdict[curh][var].SetTitle(hdict[curhsum][var].GetName().replace(curhsum,curh))


# In[ ]:


#Plot ht
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
output = TFile("FromFlatPandas_AE"+op_aeval+"_M"+op_massrange+"_Njet"+str(op_njet)+".root","recreate")
output.cd()

for RHtext in RateHists:
    RateHists[RHtext].Write("TRate"+RHtext)
for RHtext in RateHistsFULL:
    RateHistsFULL[RHtext].Write("TRate"+RHtext)
for ds in ratehistos:
    for var in ratehistos[ds]:
            ratehistos[ds][var].Write(ds+"__"+var)

if ntoys>0:
    for ds in histos:

        canvtoys=TCanvas("httoys"+ds,"httoys"+ds,700,500)
        canvtoyspread=TCanvas("httoyspread"+ds,"httoyspread"+ds,700,500)

        histoiter=list(range(njet+1))
        histoiter.reverse()
        histoiter.pop(0)    
        for ijet in histoiter:
                
                regionstr="LT"+str(ijet)+str(njet-ijet)

                bkgname="bkg_ht_"+regionstr

                dataname="ht_"+regionstr
                canvtoys.cd()
                bindists=[]

                for itoy in  range(ntoys):
                    bindists.append([])
                    for xbin in range(histos[ds]["bkg_ht_toy"+str(itoy)+"_"+regionstr].GetXaxis().GetNbins()):
                        bindists[-1].append(histos[ds]["bkg_ht_toy"+str(itoy)+"_"+regionstr].GetBinContent(xbin))
                    if itoy==0:
                        histos[ds]["bkg_ht_toy"+str(itoy)+"_"+regionstr].Draw("hist")  
                    else:
                        histos[ds]["bkg_ht_toy"+str(itoy)+"_"+regionstr].Draw("samehist") 
                histos[ds]["bkg_ht_"+regionstr].Draw("same") 
                binarr=np.array(bindists)
                totoy=binarr.shape[0]
                totbins=binarr.shape[1]
                histos[ds]["bkg_ht_toyspread_"+regionstr]=copy.deepcopy(histos[ds][bkgname])
                histos[ds]["bkg_ht_toyspread_"+regionstr].SetName("bkg_ht_toyspread_"+regionstr)
                histos[ds]["bkg_ht_toyspread_"+regionstr].SetTitle("bkg_ht_toyspread_"+regionstr)
                for ibin in range(totbins):
                    #print(binarr[:,ibin].std())
                    histos[ds]["bkg_ht_toyspread_"+regionstr].SetBinContent(ibin,binarr[:,ibin].mean())
                    histos[ds]["bkg_ht_toyspread_"+regionstr].SetBinError(ibin,binarr[:,ibin].std())
                canvtoyspread.cd()
                histos[ds]["bkg_ht_toyspread_"+regionstr].Draw("same")
    canvtoys.Write()    
    canvtoys.Print('plots/httoys'+ds+'.png', 'png')  
    canvtoys.Print('plots/httoys'+ds+'.root', 'root')  
    image = mpimg.imread('plots/httoys'+ds+'.png')

    #print("toys")  

tocanv={"ht":[2,[0,5000]],"pt0":[2,[0,3000]],"pt1":[2,[0,3000]],"pt2":[2,[0,3000]],"ptT0":[2,[0,3000]],"ptT1":[2,[0,3000]],"ptT2":[2,[0,3000]],"ptL0":[2,[0,3000]],"ptL1":[2,[0,3000]],"ptL2":[2,[0,3000]]}

for tc in tocanv:
        for ds in histos:
            for var in histos[ds]:
                    histos[ds][var].Write(ds+"__"+var)
                    #print(ds,var,histos[ds][var].Integral())
            rebinval=tocanv[tc][0]
            xrangeval=tocanv[tc][1]
            #print(histos[ds])
            canv=TCanvas(tc+ds,tc+ds,700,500)
            main = ROOT.TPad("main", "main", 0, 0.3, 1, 1)
            sub = ROOT.TPad("sub", "sub", 0, 0, 1, 0.3)

            main.SetLeftMargin(0.16)
            main.SetRightMargin(0.05)
            main.SetTopMargin(0.11)
            main.SetBottomMargin(0.0)

            sub.SetLeftMargin(0.16)
            sub.SetRightMargin(0.05)
            sub.SetTopMargin(0)
            sub.SetBottomMargin(0.3)

            main.Draw()
            sub.Draw()
            #canvrat=TCanvas("htrat"+ds,"htrat"+ds,700,500)
            gPad.SetLeftMargin(0.12)
            leg = TLegend(0.65, 0.55, 0.84, 0.84)
            leg.SetFillColor(0)
            leg.SetBorderSize(0)
            histoiter=list(range(njet+1))
            histoiter.reverse()
            histoiter.pop(0)
            allrat=[]

            for ijet in histoiter:
                    regionstr="LT"+str(ijet)+str(njet-ijet)
                    if tc[0:2]=="pt":
  
                        if regionstr!="LT21":
                                continue

                    bkgname="bkg_"+tc+"_"+regionstr
                    dataname=tc+"_"+regionstr
                    color=ijet+1



                    ratehistos[ds][dataname].SetLineColor(color)
                    ratehistos[ds][dataname].SetTitle(";"+tc+"(GeV);events")
                    ratehistos[ds][dataname].SetStats(0) 
                    ratehistos[ds][dataname].Rebin(rebinval) 

                    ratehistos[ds][dataname].GetXaxis().SetRangeUser(xrangeval[0],xrangeval[1])

                    histos[ds][bkgname].SetLineColor(color)
                    histos[ds][bkgname].Rebin(rebinval) 

                    histos[ds][bkgname].GetXaxis().SetRangeUser(xrangeval[0],xrangeval[1])

                    main.cd()

                    ratehistos[ds][dataname].GetXaxis().SetTitleSize (0.06)
                    ratehistos[ds][dataname].GetXaxis().SetLabelSize (0.05)
                    ratehistos[ds][dataname].GetYaxis().SetTitleSize (0.06)
                    ratehistos[ds][dataname].GetYaxis().SetLabelSize (0.05)
                    ratehistos[ds][dataname].Draw("same")   
                    histos[ds][bkgname].Draw("histsame") 
                    
                    
                    leg.AddEntry(histos[ds][bkgname],ds+regionstr+"bkg","L")
                    leg.AddEntry(ratehistos[ds][dataname],ds+regionstr,"LE")

                    sub.cd()
                    allrat.append(copy.deepcopy(ratehistos[ds][dataname]) )
                    allrat[-1].Divide(histos[ds][bkgname])


                    allrat[-1].GetYaxis().SetRangeUser(0.5,1.5)
                    allrat[-1].SetTitle(";"+tc+"(GeV);")
                    allrat[-1].GetXaxis().SetTitleSize (0.12)
                    allrat[-1].GetXaxis().SetLabelSize (0.09)

                    allrat[-1].GetYaxis().SetTitleSize (0.12)
                    allrat[-1].GetYaxis().SetLabelSize (0.09)
                    allrat[-1].GetXaxis().SetRangeUser(xrangeval[0],xrangeval[1])


                    print("--Fit--",tc,regionstr)
                    if regionstr=="LT21":
                        allrat[-1].Fit("pol1")

                    line2=ROOT.TLine(xrangeval[0],1.0,xrangeval[1],1.0)
                    line2.SetLineColor(0)
                    line1=ROOT.TLine(xrangeval[0],1.0,xrangeval[1],1.0)
                    line1.SetLineStyle(2)

                    allrat[-1].Draw("histesame") 
                    line2.Draw()
                    line1.Draw()
            main.cd()
            leg.Draw()
            main.SetLogy()
            canv.Write()
            canv.Print('plots/'+tc+ds+'.png', 'png')
            canv.Print('plots/'+tc+ds+'.root', 'root')
            print(ds)
           

    
    
output.Close()




