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
import pandas as pd

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
                  default	=	'90',
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

parser.add_option('--dataonly', metavar='F', action='store_true',
		  default=False,
		  dest='dataonly',
		  help='dataonly')

(options, args) = parser.parse_args()
op_nproc=int(options.nproc)
op_njet=int(options.njet)

op_massrange=options.massrange
op_aeval=options.aeval

qcdonly=options.qcdonly
dataonly=options.dataonly
# In[3]:

exstr=""
if qcdonly:
        exstr+="_qcdonly"
if dataonly:
        exstr+="_dataonly"
ntoys=int(options.toys)
quickrun=options.quickrun
if quickrun:
    op_nproc=1


# In[4]:


ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True



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
        
        fjcutpt=(df["FatJet"]["pt"]>self.ptcut[0])&(df["FatJet"]["pt"]<self.ptcut[1])#&(df["FatJet"]["hadronFlavour"]>3.5) 
        #print(df["FatJet"]["hadronFlavour"])
        df["FatJet"]=(df["FatJet"][fjcutpt])
        C1=(df["FatJet"]["event"].count(level=0))==self.njet

        fjcutmass=(df["FatJet"]["msoftdrop"]>self.msdcut[0])&(df["FatJet"]["msoftdrop"]<self.msdcut[1])
        df["FatJet"]=df["FatJet"][fjcutmass]

        C2=(df["FatJet"]["event"].count(level=0))==self.njet

        fjcut=fjcutpt&fjcutmass
        C0=((fjcut).sum(level=0)>0)
   
        #print (df["FatJet"])
        #print (df["FatJet"]["hadronFlavour"]>3.5)

        if (not ( C0 & C1 & C2).any()):
            return None
        return ( C0 & C1 & C2)


# In[7]:
class KinematicSelectionDRAK4():
    def __init__(self,njet,njetAK4,maxht,drrange):
        self.njet=njet
        self.njetAK4=njetAK4
        self.maxht=maxht
        self.drrange=drrange
    def __call__(self,df,EventInfo):    
        alldiscs=[]

        Allphi=df["FatJet"]["phi"]
        Alleta=df["FatJet"]["eta"]

        AllphiAK4=df["Jet"]["phi"]
        AlletaAK4=df["Jet"]["eta"]

        for ijet in range(self.njet):
            #todo: find better way to check for nulls
            try:
                ijetphi=Allphi[:,ijet]
                ijeteta=Alleta[:,ijet]
            except:
                print ("ERR")
                print (Allphi)
                print (Alleta)
                return None

            closeht=df["Jet"]["pt"][:,0]*0.0

            for jjet in range(self.njetAK4):

                #if ijet==jjet:
                 #   continue
            
                jjetphi=AllphiAK4[:,jjet]
                jjeteta=AlletaAK4[:,jjet]


                deta=(jjeteta-ijeteta).abs()
                dphi=(jjetphi-ijetphi).abs()

                dphi[dphi>3.1415]=2*3.1415-dphi

                dr=np.sqrt(dphi*dphi+deta*deta)
                cond=(self.drrange[0]<dr)&(dr<self.drrange[1])
                #print("cond",cond)
                #print("df[Jet][pt][:,jjet]",df["Jet"]["pt"][:,jjet])
                #print("df[Jet][pt][:,jjet]*cond",df["Jet"]["pt"][:,jjet]*cond)
                #print("closehtPRE",ijet)
                #print(closeht)
                
                closeht+=df["Jet"]["pt"][:,jjet]*cond

                #print("closehtPOST",ijet)
                #print(closeht)
            
            alldiscs.append(closeht<self.maxht)

            #print("DONE")
            #print(ijet)
            #print(closeht)

        for iad,ad in enumerate(alldiscs):
            if iad==0:
                evdisc=ad
            else:
                evdisc=evdisc&ad
            #print("evd",iad,evdisc)
        #print("evd",evdisc)

        return ( evdisc )

#Select DeltaR cut to make sure AK8 jets are separated
class KinematicSelectionDR():
    def __init__(self,njet,drcut):
        self.drcut=drcut
        self.njet=njet
    def __call__(self,df,EventInfo):    
        alldiscs=[]

        Allphi=df["FatJet"]["phi"]
        Alleta=df["FatJet"]["eta"]
        for ijet in range(self.njet):
            #todo: find better way to check for nulls
            try:
                ijetphi=Allphi[:,ijet]
                ijeteta=Alleta[:,ijet]
            except:
                print ("ERR")
                print (Allphi)
                print (Alleta)
                return None

            drcutjet=None
            for jjet in range(self.njet):

                if ijet==jjet:
                    continue
            
                jjetphi=Allphi[:,jjet]
                jjeteta=Alleta[:,jjet]

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


        #loose and tight bias from QCD1500 mean meanR 1.0052935 meanL 0.9989067




        df["FatJet"]["tight"] = logmse>AEcut
        df["FatJet"]["loose"] = logmse<AEcut

        df["Hists"]["ht"]=df["FatJet"]["pt"].sum(level=0)
        if not EventInfo.eventcontainer["isdata"][EventInfo.dataset]:
                hfsum=(abs(df["FatJet"]["hadronFlavour"])==5).sum(level=0)
                df["Hists"]["hasb"]=(hfsum>0)
                df["Hists"]["nob"]=(hfsum==0)
        else:
                df["Hists"]["hasb"]=df["Hists"]["ht"]*0.
                df["Hists"]["nob"]=df["Hists"]["hasb"]


        df["Hists"]["njettight"] = njettight
        df["Hists"]["njetloose"] = njetloose
        df["Hists"]["logmse"] = logmse
        #df["FatJet"]["p"] = df["FatJet"]["pt"]*np.cosh(df["FatJet"]["eta"])
        #df["FatJet"]["E"] = np.sqrt(df["FatJet"]["p"]*df["FatJet"]["p"]+df["FatJet"]["msoftdrop"]*df["FatJet"]["msoftdrop"])
        #print()
        #print (df["FatJet"]["pt"])
        #print (df["FatJet"]["p"])
        for ijet in range(self.njet):
                df["Hists"]["logmse"+str(ijet)] = logmse[:,ijet]

                for jjet in range(self.njet):
                        if ijet!=jjet:
                                nthird=self.njet-(ijet+jjet)
                                prelogmse=logmse[:,ijet][df["FatJet"]["loose"][:,nthird]]
                                df["Hists"]["biasT"+str(ijet)+str(jjet)] = ((prelogmse[df["FatJet"]["tight"][:,jjet]]))
                                df["Hists"]["biasL"+str(ijet)+str(jjet)] = ((prelogmse[df["FatJet"]["loose"][:,jjet]]))

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
            #print(hh)
            df["Hists"][hh+"__weight"]=df["Hists"]["weight"]
            if (df["Hists"][hh].index.nlevels > df["Hists"]["weight"].index.nlevels )  :
                df["Hists"][hh]=df["Hists"][hh].droplevel(level=1)

            df["Hists"][hh+"__weight"] = df["Hists"][hh+"__weight"][df["Hists"][hh+"__weight"].index.isin(df["Hists"][hh].index)]
         

        df["Hists"]["njettight__njetloose__weight"]=df["Hists"]["njettight__weight"]
        return df


# In[10]:


#make histograms to be used for creating the pass-to-fail ratio
class MakeHistsForRate():
    def __init__(self,njet):
        self.njet=njet
    def __call__(self,df,EventInfo):
        bkgparam=EventInfo.eventcontainer["bkgparam"]
        Tpt=df["FatJet"]["pt"][df["FatJet"]["tight"]]
        Lpt=df["FatJet"]["pt"][df["FatJet"]["loose"]]
        for ijet in range(self.njet):
            for ebin in bkgparam["eta"]:
                    for mbin in bkgparam["mass"]:

                        abseta=df["FatJet"]["eta"].abs()

                        etacut=(bkgparam["eta"][ebin][0]<=abseta)&(abseta<bkgparam["eta"][ebin][1])
                        masscut=(bkgparam["mass"][mbin][0]<=df["FatJet"]["msoftdrop"])&(df["FatJet"]["msoftdrop"]<bkgparam["mass"][mbin][1])
                        tcond=True
           
                        try:
                            df["Hists"]["ptT"+str(ijet)+"_"+ebin+mbin]=Tpt[etacut][masscut][:,ijet]
                        except:
                            pass

                        try:
                            df["Hists"]["ptL"+str(ijet)+"_"+ebin+mbin]=Lpt[etacut][masscut][:,ijet]
                        except:
                            pass
        return df



#make histograms to be used for creating the pass-to-fail ratio
class MakeHistsForBkg():
    def __init__(self,njet):
        self.njet=njet
    def __call__(self,df,EventInfo):
        bkgparam=EventInfo.eventcontainer["bkgparam"]
        Lpt=df["FatJet"]["pt"][df["FatJet"]["loose"]]
        Tpt=df["FatJet"]["pt"][df["FatJet"]["tight"]]

        maxbin=2**self.njet      
        allregs=list(range(maxbin))
        allregs.reverse()

        for ar in allregs:
                condseries=None
                for ijet in range(self.njet):
                        condstr="tight" if ((ar>>ijet)&1) else "loose"
                        if isinstance(condseries,type(None)):
                                condseries=df["FatJet"][condstr][:,ijet]
                        else:
                                condseries&=df["FatJet"][condstr][:,ijet]
                print(ar)
                print(condseries)
                print(df["Hists"]["ht"][condseries])
                df["Hists"]["ht_"+str(bin(ar))]=df["Hists"]["ht"][condseries]



        for ijet in range(self.njet):




 
            regionstr="LT"+str(ijet)+str(njet-ijet)
            Tbool=df["Hists"]["njettight"]==(njet-ijet) 
            Lbool=df["Hists"]["njetloose"]==(ijet)

            T1bool=df["Hists"]["njettight"]>0

            htreg=df["Hists"]["ht"][Tbool][Lbool]
            df["Hists"]["ht_"+regionstr]=htreg
            if not EventInfo.eventcontainer["isdata"][EventInfo.dataset]:
                    df["Hists"]["htb_"+regionstr]=htreg[df["Hists"]["hasb"]]
                    df["Hists"]["htl_"+regionstr]=htreg[df["Hists"]["nob"]]

            for jjet in range(self.njet):

                df["Hists"]["pt_"+str(jjet)+regionstr]=df["FatJet"]["pt"][:,jjet][Tbool][Lbool]





                #print("pt_"+str(jjet)+regionstr,df["Hists"]["pt_"+str(jjet)+regionstr])
            #print(regionstr)
            #print(df["Hists"]["htb_"+regionstr])
            try:
                ind1=(ijet+1)%3
                ind2=(ijet+2)%3
                othertight=df["FatJet"]["tight"][:,ind1] | df["FatJet"]["tight"][:,ind2]
                #print(othertight)
                df["Hists"]["logmse"+str(ijet)]=df["FatJet"]["logmse"][:,ijet]
                df["Hists"]["logmseGT"+str(ijet)]=df["FatJet"]["logmse"][:,ijet][othertight]
            except:
                pass

            try:
                df["Hists"]["ptTIGHT"+str(ijet)]=Tpt[:,ijet]
            except:
                pass
            try:
                df["Hists"]["ptLOOSE"+str(ijet)]=Lpt[:,ijet]
                df["Hists"]["ptLOOSEGT"+str(ijet)]=Lpt[:,ijet][T1bool]
            except:
                pass
            #print(df["FatJet"]["pt"][df["FatJet"]["tight"]][:,ijet]    )          
        return df




# In[11]:


#use pass-to-fail ratio created in step0 to predict background
#todo: Sometimes returns none -- look into
class BkgEst():
    
    def __init__(self,njet):
        self.njet=njet
    
    def prepdf(self,df):
        args=[df["Hists"]["ht"]]
        args.append(df["Hists"]["hasb"])
        args.append(df["Hists"]["nob"])
        try:
            for ijet in range(self.njet):
                args.append(df["FatJet"]["pt"][:,ijet])
                #args.append(df["FatJet"]["p"][:,ijet])
                #args.append(df["FatJet"]["E"][:,ijet])
                args.append(df["FatJet"]["eta"][:,ijet].abs())
                args.append(df["FatJet"]["msoftdrop"][:,ijet])
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

        ht=args[0]
        hasb=args[1]
        nob=args[2]

        pt=[]
        eta=[]
        msd=[]
        tight=[]
        loose=[]
       
        disp=3 #hist-type inputs
        for ijet in range(self.njet):
            pt.append(args[ijet*5+disp+0])
            eta.append(args[ijet*5+disp+1])
            msd.append(args[ijet*5+disp+2])
            tight.append(args[ijet*5+disp+3])
            loose.append(args[ijet*5+disp+4])
            regionstr="LT"+str(ijet)+str(njet-ijet)
        
        nloose=0
        for ll in loose:
            nloose+=ll

        ntight=0
        for tt in tight:
            ntight+=tt
        
        maxbin=2**self.njet





        allregs=list(range(maxbin))
        allregs.reverse()

        Trate=[0.0] * self.njet
        Lrate=[0.0] * self.njet
    
        usefullrate=True

        for ijet in range(self.njet):
            for iebin,ebin in enumerate(bkgparam["eta"]):
                for mbin in bkgparam["mass"]:
                        etacut=(bkgparam["eta"][ebin][0]<=eta[ijet]<bkgparam["eta"][ebin][1])
                        masscut=(bkgparam["mass"][mbin][0]<=msd[ijet]<bkgparam["mass"][mbin][1])
                        if etacut and masscut:
                                    ptbin=RateHists["Rate"+ebin+mbin].FindBin(pt[ijet])
                                    TRtemp=RateHists["Rate"+ebin+mbin].GetBinContent(ptbin)
                                    TRtemperr=RateHists["Rate"+ebin+mbin].GetBinError(ptbin)

                                    Trate[ijet]=TRtemp
                                    Lrate[ijet]=1.0-TRtemp
    
        weights=[0.0]*(self.njet+1)
        nweights=[0.0]*(self.njet+1)

        LrateGtight=[0.0]*(self.njet+1)
        TrateGtight=[0.0]*(self.njet+1)


        rweight=[]
        for ar in allregs:

            ntight=0
            for ibit,bit in enumerate(range(self.njet)):
                ntight+=(ar>>bit)&1
            weight=1.0
            
            for ibit,bit in enumerate(range(self.njet)):
                curbit=(ar>>bit)&1
                #print(ibit,curbit,Trate[ibit],Lrate[ibit])
                if curbit:
                    weight*=Trate[ibit]   
             
                else:
                    weight*=Lrate[ibit]
            for ijet in range(self.njet):
                if ar!=0:
                        if (ar>>ijet)&1:
                                #print (bin(ar),ijet,weight)
                                TrateGtight[ijet]+=weight
                        else:
                                LrateGtight[ijet]+=weight
            rweight.append(weight)
            weights[self.njet-ntight]+=weight
            nweights[self.njet-ntight]+=1.0
        #print("Trate",Trate)
        #print("TrateGtight",TrateGtight)
        #print("Lrate",Lrate)
        #print("LrateGtight",LrateGtight)
        allret=[]

        for icweight,cweight in enumerate(weights):
            allret.append(ht)
            allret.append(cweight*EventInfo.eventcontainer["evweight"])

            allret.append(ht)
            allret.append(hasb*cweight*EventInfo.eventcontainer["evweight"])

            allret.append(ht)
            allret.append(nob*cweight*EventInfo.eventcontainer["evweight"])


            for ijet in range(njet):
                allret.append(pt[ijet])
                allret.append(cweight*EventInfo.eventcontainer["evweight"])

        for ijet in range(self.njet):
            allret.append(pt[ijet])
            allret.append(Trate[ijet]*EventInfo.eventcontainer["evweight"])

            allret.append(pt[ijet])
            allret.append((Lrate[ijet])*EventInfo.eventcontainer["evweight"])

            allret.append(pt[ijet])
            allret.append((LrateGtight[ijet])*EventInfo.eventcontainer["evweight"])

        for iar in range(len(allregs)):
            allret.append(ht)
            allret.append(rweight[iar]*EventInfo.eventcontainer["evweight"])


        return (allret)


# In[12]:



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
#print(chunklist)
todels=[]
for ds in chunklist:
        if qcdonly:
            if (ds.split("_")[0]!="QCD"):
                todels.append(ds)
        if dataonly:
            if (ds.split("_")[0]!="DATA"):
                todels.append(ds)
for todel in todels:
        del chunklist[todel]

bkgparam={}

bkgparam["eta"]={"E0":[0.0,0.5],"E1":[0.5,float("inf")]}
bkgparam["mass"]={"M0":[0.0,float("inf")]}

#todo: muon triggers a failure mode as sometimes events have no muons and no filter remo 
branchestoread={
                    #"Muon":["pt","eta","phi","mass"],
                    "Jet":["pt","eta","phi","mass"],
                    "FatJet":["pt","eta","phi","mass","msoftdrop","iAEMSE"],
                    "":["run","luminosityBlock","event"]
               }
if not dataonly:
        branchestoread["FatJet"].append("hadronFlavour")
scalars=[""]

if op_massrange=="all":
    sdcut=[0.0,float("inf")]
else:
    #sdcuts=[[0.0,50.0],[50.0,100.0],[100.0,140.0],[140.0,200.0],[200.0,float("inf")]]
    sdcuts=[[0.0,50.0],[50.0,float("inf")]]
    sdcut=sdcuts[int(op_massrange)]


# In[14]:
maxbin=2**op_njet      
allregs=list(range(maxbin))
allregs.reverse()


#customize a multi-step processor
def MakeProc(njet,step,evcont):
    histostemp=OrderedDict  ([])
    if step==0:
        rhistlist=[]
        for ijet in range(njet):
            rhistlist.append("logmse"+str(ijet))
            for jjet in range(njet):
                rhistlist.append("biasT"+str(ijet)+str(jjet))
                rhistlist.append("biasL"+str(ijet)+str(jjet))





        for ijet in range(njet+1):
        

            regionstr="LT"+str(ijet)+str(njet-ijet)

            histostemp["ht_"+regionstr]=TH1F("ht_"+regionstr,"ht_"+regionstr,700,0,7000)
            histostemp["htb_"+regionstr]=TH1F("htb_"+regionstr,"htb_"+regionstr,700,0,7000)
            histostemp["htl_"+regionstr]=TH1F("htl_"+regionstr,"htl_"+regionstr,700,0,7000)

            histostemp["logmse"+str(ijet)]=TH1F("logmse"+str(ijet),"logmse"+str(ijet),100,-20.,0.)
            histostemp["logmseGT"+str(ijet)]=TH1F("logmseGT"+str(ijet),"logmseGT"+str(ijet),100,-20.,0.)

            for jjet in range(njet):

                histostemp["logmse"+str(jjet)+"_"+regionstr]=TH1F("logmse"+str(jjet)+"_"+regionstr,"logmse"+str(jjet)+"_"+regionstr,100,-20.,0.)
                histostemp["pt_"+str(jjet)+regionstr]=TH1F("pt_"+str(jjet)+regionstr,"pt_"+str(jjet)+regionstr,1000,0,10000)
                
                histostemp["ptTIGHT"+str(jjet)+"_"+regionstr]=TH1F("ptTIGHT"+str(jjet)+"_"+regionstr,"ptTIGHT"+str(jjet)+"_"+regionstr,200,0,4000)
                histostemp["ptLOOSE"+str(jjet)+"_"+regionstr]=TH1F("ptLOOSE"+str(jjet)+"_"+regionstr,"ptLOOSE"+str(jjet)+"_"+regionstr,200,0,4000)
            
            for ebin in bkgparam["eta"]:
                for mbin in bkgparam["mass"]:
                    histostemp["ptL"+str(ijet)+"_"+ebin+mbin]=TH1F("ptL"+str(ijet)+"_"+ebin+mbin,"ptL"+str(ijet)+"_"+ebin+mbin,1000,0,10000)
                    histostemp["ptT"+str(ijet)+"_"+ebin+mbin]=TH1F("ptT"+str(ijet)+"_"+ebin+mbin,"ptT"+str(ijet)+"_"+ebin+mbin,1000,0,10000)
        histostemp["logMSE_all"]=TH1F("logMSE_all","logMSE_all",100,-20.,0.)



        myana=  [
                PColumn(PreColumn()),
                PFilter(KinematicSelection(njet,[400.0,float("inf")],sdcut)), 
                PFilter(KinematicSelectionDR(njet,1.4)),
                #PFilter(KinematicSelectionDRAK4(njet,5,200.0,[0.8,1.4])),
                PColumn(MakeTags(njet)),
                PColumn(MakeHistsForRate(njet)),
                PColumn(ColumnWeights()),
                ]

    if step==1:
        rhistlist=[]
        hpass=[]

        for ijet in range(njet+1):
            regionstr="LT"+str(ijet)+str(njet-ijet)
            


            histostemp["logmse"+str(ijet)]=TH1F("logmse"+str(ijet),"logmse"+str(ijet),100,-20.,0.)
            histostemp["logmseGT"+str(ijet)]=TH1F("logmseGT"+str(ijet),"logmseGT"+str(ijet),100,-20.,0.)


            histostemp["ht_"+regionstr]=TH1F("ht_"+regionstr,"ht_"+regionstr,700,0,7000)
            histostemp["htb_"+regionstr]=TH1F("htb_"+regionstr,"htb_"+regionstr,700,0,7000)
            histostemp["htl_"+regionstr]=TH1F("htl_"+regionstr,"htl_"+regionstr,700,0,7000)

            for jjet in range(njet):
                histostemp["pt_"+str(jjet)+regionstr]=TH1F("pt_"+str(jjet)+regionstr,"pt_"+str(jjet)+regionstr,200,0,4000)


            histostemp["bkg_ht_"+regionstr]=TH1F("bkg_ht_"+regionstr,"bkg_ht_"+regionstr,700,0,7000)
            
            hpass.append(["Hists","bkg_ht_"+regionstr])
            hpass.append(["Hists","bkg_ht_"+regionstr+"__weight"])

            histostemp["bkg_htb_"+regionstr]=TH1F("bkg_htb_"+regionstr,"bkg_htb_"+regionstr,700,0,7000)

            hpass.append(["Hists","bkg_htb_"+regionstr])
            hpass.append(["Hists","bkg_htb_"+regionstr+"__weight"])

            histostemp["bkg_htl_"+regionstr]=TH1F("bkg_htl_"+regionstr,"bkg_htl_"+regionstr,700,0,7000)

            hpass.append(["Hists","bkg_htl_"+regionstr])
            hpass.append(["Hists","bkg_htl_"+regionstr+"__weight"])
   
            for ijet in range(njet):

                    histostemp["bkg_pt_"+str(ijet)+regionstr]=TH1F("bkg_pt_"+str(ijet)+regionstr,"bkg_pt_"+str(ijet)+regionstr,200,0,4000)
                    
                    hpass.append(["Hists","bkg_pt_"+str(ijet)+regionstr])
                    hpass.append(["Hists","bkg_pt_"+str(ijet)+regionstr+"__weight"])
   

 
            for itoy in range(ntoys):
                histostemp["bkg_ht_toy"+str(itoy)+"_"+regionstr]=TH1F("bkg_ht_toy"+str(itoy)+"_"+regionstr,"bkg_ht_toy"+str(itoy)+"_"+regionstr,700,0,7000)         
        


        for ijet in range(njet):
        
                    histostemp["ptTIGHT"+str(ijet)]=TH1F("ptTIGHT"+str(ijet),"ptTIGHT"+str(ijet),200,0,4000)
                    histostemp["ptLOOSE"+str(ijet)]=TH1F("ptLOOSE"+str(ijet),"ptLOOSE"+str(ijet),200,0,4000)
                    histostemp["ptLOOSEGT"+str(ijet)]=TH1F("ptLOOSEGT"+str(ijet),"ptLOOSEGT"+str(ijet),200,0,4000)

                    histostemp["bkg_ptTIGHT"+str(ijet)]=TH1F("bkg_ptTIGHT"+str(ijet),"bkg_ptTIGHT"+str(ijet),200,0,4000)

                    hpass.append(["Hists","bkg_ptTIGHT"+str(ijet)])
                    hpass.append(["Hists","bkg_ptTIGHT"+str(ijet)+"__weight"])

                    histostemp["bkg_ptLOOSE"+str(ijet)]=TH1F("bkg_ptLOOSE"+str(ijet),"bkg_ptLOOSE"+str(ijet),200,0,4000)
                    hpass.append(["Hists","bkg_ptLOOSE"+str(ijet)])
                    hpass.append(["Hists","bkg_ptLOOSE"+str(ijet)+"__weight"])
         
                    histostemp["bkg_ptLOOSEGT"+str(ijet)]=TH1F("bkg_ptLOOSEGT"+str(ijet),"bkg_ptLOOSEGT"+str(ijet),200,0,4000)
                    hpass.append(["Hists","bkg_ptLOOSEGT"+str(ijet)])
                    hpass.append(["Hists","bkg_ptLOOSEGT"+str(ijet)+"__weight"])



        for ar in allregs:
                histostemp["ht_"+str(bin(ar))]=TH1F("ht_"+str(bin(ar)),"ht_"+str(bin(ar)),700,0,7000)
                histostemp["bkg_ht_"+str(bin(ar))]=TH1F("bkg_ht_"+str(bin(ar)),"bkg_ht_"+str(bin(ar)),700,0,7000)
                hpass.append(["Hists","bkg_ht_"+str(bin(ar))])
                hpass.append(["Hists","bkg_ht_"+str(bin(ar))+"__weight"])
        print("len(hpass)",len(hpass))        
                    
        myana=  [
                PColumn(PreColumn()),
                PFilter(KinematicSelection(njet,[400.0,float("inf")],sdcut)),     
                PFilter(KinematicSelectionDR(njet,1.4)),
                #PFilter(KinematicSelectionDRAK4(njet,5,200.0,[0.8,1.4])),
                PColumn(MakeTags(njet)),
                PColumn(MakeHistsForBkg(njet)),
                PRow(hpass,BkgEst(njet)),
                #PColumn(MakeToys(njet)),
                PColumn(ColumnWeights()),
                ]
    for hist in histostemp:
        histostemp[hist].Sumw2() 


    histos= {}
    for ds in chunklist:
        if quickrun:
            chunklist[ds]=chunklist[ds][:1]
        #chunklist[ds]=chunklist[ds][:12]
        histos[ds]=copy.deepcopy(histostemp)

    return PProcessor(chunklist,histos,branchestoread,myana,eventcontainer=evcont,atype="flat",scalars=scalars,rhistlist=rhistlist)


# In[15]:


njet=op_njet
evcont={"lumi":(1000.0*137.65),"isdata":{"WgWg":False,"TT":False,"QCD_HT1500to2000":False,"QCD_HT1000to1500":False,"QCD_HT2000toInf":False,"DATA_2017_B":True},"nev":{"WgWg":18000.0,"TT":305963.0,"QCD_HT1500to2000":10655313.0,"QCD_HT1000to1500":12660521.0,"QCD_HT2000toInf":4980828.0,"DATA_2017_B":1.0},"xsec":{"WgWg":1.0,"TT":1.0,"QCD_HT1500to2000":101.8,"QCD_HT1000to1500":1005.0,"QCD_HT2000toInf":20.54,"DATA_2017_B":1.0}}
evcont["bkgparam"]=bkgparam

# In[16]:


#Step0:make hists for pass-to-fail ratio
proc = MakeProc(njet,0,evcont)
nproc=op_nproc
Mproc=PProcRunner(proc,nproc)
returndf=Mproc.Run()
qcdnames = ["QCD_HT1000to1500","QCD_HT1500to2000","QCD_HT2000toInf"]

if dataonly:
        alldat = list(chunklist.keys())
        qcdnames = alldat
# In[18]:



ratehistos=copy.deepcopy(proc.hists)




# In[ ]:


#Make pass-to-fail ratio TR(pt,eta)
THists={}

LHists={}
ALLHists={}

THistsFULL={}
LHistsFULL={}

print("START")

bins=array.array('d',[0,200,210,220,230,240,250,260,280,300,320,340,360,380,420,500,600,700,800,900,1000,1200,1500,2000,10000])
for ijet in range(njet):
    #print(ijet)

    for qcdname in qcdnames:
        #print(qcdname)

        QCDhists=ratehistos[qcdname]
        for curhist in QCDhists:
            #print(curhist)
            if curhist[:4] =="ptL"+str(ijet):
                Lstring=curhist
                Tstring=curhist.replace("ptL"+str(ijet),"ptT"+str(ijet))


                paramstr=Lstring.split("_")[-1]
                paramstrwjet=Lstring.split("_")[-1]+"jet"+str(ijet)
                

                curhistL=QCDhists[Lstring]
                curhistT=QCDhists[Tstring]


                if not(paramstr in THists):
                    THists[paramstr]=copy.deepcopy(curhistT)
                    LHists[paramstr]=copy.deepcopy(curhistL)
                    LHists[paramstr].Add(curhistT)

                else:
                    THists[paramstr].Add(curhistT)
                    LHists[paramstr].Add(curhistL)
                    LHists[paramstr].Add(curhistT)
                    
                if not(paramstrwjet in THistsFULL):
                    THistsFULL[paramstrwjet]=copy.deepcopy(curhistT)
                    LHistsFULL[paramstrwjet]=copy.deepcopy(curhistL)
                    LHistsFULL[paramstrwjet].Add(curhistT)

                else:
                    THistsFULL[paramstrwjet].Add(curhistT)
                    LHistsFULL[paramstrwjet].Add(curhistL)
                    LHistsFULL[paramstrwjet].Add(curhistT)                    
                print("DONE",Tstring)
                    
print("DONE")          
for tth in THists:
    THists[tth]=THists[tth].Rebin(len(bins)-1,THists[tth].GetName()+"TEMP",bins)  

for llh in LHists:
    LHists[llh]=LHists[llh].Rebin(len(bins)-1,LHists[llh].GetName()+"TEMP",bins)                    

print("DONE1")          
  
for tth in THistsFULL:
    THistsFULL[tth]=THistsFULL[tth].Rebin(len(bins)-1,THistsFULL[tth].GetName()+"TEMP",bins)                    
for llh in LHistsFULL:
    LHistsFULL[llh]=LHistsFULL[llh].Rebin(len(bins)-1,LHistsFULL[llh].GetName()+"TEMP",bins)       
    


# In[ ]:




import matplotlib.pyplot as plt
import matplotlib.image as mpimg

RateHists=OrderedDict([])
canvrate=TCanvas("canvrate","canvrate",700,500)
color=1

alltoys=[]
for RH in LHists:
    print(RH)
    print("THists",THists[RH].Integral())
    print("LHists",LHists[RH].Integral())
    
    RateHists["Rate"+RH]=copy.deepcopy(THists[RH])
    RateHists["Rate"+RH].Divide(RateHists["Rate"+RH],LHists[RH],1.0,1.0,"B")

    means = []
    errs = []
    toys = []
    for xbin in range(RateHists["Rate"+RH].GetXaxis().GetNbins()+1):
        means.append(RateHists["Rate"+RH].GetBinContent(xbin))
        errs.append(RateHists["Rate"+RH].GetBinError(xbin))
    curtoys=[]
    for tt in range(ntoys):
        curtoys.append(np.random.normal(means,errs))
    alltoys.append(curtoys)
    #print (curtoys)
    RateHists["Rate"+RH].SetLineColor(color)
    RateHists["Rate"+RH].SetMarkerColor(color)
    RateHists["Rate"+RH].Draw("same")
    color+=1
    
RateHistsFULL=OrderedDict([])
    
for RH in LHistsFULL:
 
    RateHistsFULL["Rate"+RH]=copy.deepcopy(THistsFULL[RH])
    RateHistsFULL["Rate"+RH].Divide(RateHistsFULL["Rate"+RH],LHistsFULL[RH],1.0,1.0,"B")

   
canvrate.Print('plots/Trate.png', 'png')

evcont["RateHists"]=RateHists
evcont["RateHistsFULL"]=RateHistsFULL

evcont["toys"]=np.array(alltoys)


# In[ ]:


#Step1:use pass-to-fail ratio to predict background
proc = MakeProc(njet,1,evcont)
Mproc=PProcRunner(proc,nproc)
returndf=Mproc.Run()


# In[ ]:





# In[ ]:


histos=copy.deepcopy(proc.hists)
rebinval=20

htosum={}

if dataonly:
        qcdstr="DATA"
        htosum["DATA"]=alldat
else:
        qcdstr="QCD"
        htosum["QCD"]=["QCD_HT1500to2000","QCD_HT1000to1500","QCD_HT2000toInf"]
histdicts=[histos,ratehistos]
for hdict in histdicts:
        for curh in htosum:
            hdict[curh]={}
            print(curh)
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
paramstr=""

output = TFile("FromFlatPandas_AE"+op_aeval+"_M"+op_massrange+paramstr+"_Njet"+str(op_njet)+exstr+".root","recreate")
output.cd()

for RHtext in RateHists:
    RateHists[RHtext].Write("TRate"+RH)


for ijet in range(njet):
        regionstr="LT"+str(ijet)+str(njet-ijet)
        for jjet in range(njet):
 
                if ijet==0:
                     histos[qcdstr]["pt_"+str(jjet)]=histos[qcdstr]["pt_"+str(jjet)+regionstr]
                     histos[qcdstr]["bkg_pt_"+str(jjet)]=histos[qcdstr]["bkg_pt_"+str(jjet)+regionstr]
                else:
                     histos[qcdstr]["pt_"+str(jjet)].Add(histos[qcdstr]["pt_"+str(jjet)+regionstr])
                     histos[qcdstr]["bkg_pt_"+str(jjet)].Add(histos[qcdstr]["bkg_pt_"+str(jjet)+regionstr])

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

    print("toys")    
histos[qcdstr]["ptTIGHTsum"]=copy.deepcopy(histos[qcdstr]["ptTIGHT0"])
histos[qcdstr]["ptTIGHTsum"].Add(histos[qcdstr]["ptTIGHT1"])
histos[qcdstr]["ptTIGHTsum"].Add(histos[qcdstr]["ptTIGHT2"])

histos[qcdstr]["bkg_ptTIGHTsum"]=copy.deepcopy(histos[qcdstr]["bkg_ptTIGHT0"])
histos[qcdstr]["bkg_ptTIGHTsum"].Add(histos[qcdstr]["bkg_ptTIGHT1"])
histos[qcdstr]["bkg_ptTIGHTsum"].Add(histos[qcdstr]["bkg_ptTIGHT2"])

histos[qcdstr]["ptLOOSEsum"]=copy.deepcopy(histos[qcdstr]["ptLOOSE0"])
histos[qcdstr]["ptLOOSEsum"].Add(histos[qcdstr]["ptLOOSE1"])
histos[qcdstr]["ptLOOSEsum"].Add(histos[qcdstr]["ptLOOSE2"])

histos[qcdstr]["bkg_ptLOOSEsum"]=copy.deepcopy(histos[qcdstr]["bkg_ptLOOSE0"])
histos[qcdstr]["bkg_ptLOOSEsum"].Add(histos[qcdstr]["bkg_ptLOOSE1"])
histos[qcdstr]["bkg_ptLOOSEsum"].Add(histos[qcdstr]["bkg_ptLOOSE2"])


histos[qcdstr]["ptLOOSEGTsum"]=copy.deepcopy(histos[qcdstr]["ptLOOSEGT0"])
histos[qcdstr]["ptLOOSEGTsum"].Add(histos[qcdstr]["ptLOOSEGT1"])
histos[qcdstr]["ptLOOSEGTsum"].Add(histos[qcdstr]["ptLOOSEGT2"])

histos[qcdstr]["bkg_ptLOOSEGTsum"]=copy.deepcopy(histos[qcdstr]["bkg_ptLOOSEGT0"])
histos[qcdstr]["bkg_ptLOOSEGTsum"].Add(histos[qcdstr]["bkg_ptLOOSEGT1"])
histos[qcdstr]["bkg_ptLOOSEGTsum"].Add(histos[qcdstr]["bkg_ptLOOSEGT2"])

histos[qcdstr]["pt_sum"]=copy.deepcopy(histos[qcdstr]["pt_0"])
histos[qcdstr]["pt_sum"].Add(histos[qcdstr]["pt_1"])
histos[qcdstr]["pt_sum"].Add(histos[qcdstr]["pt_2"])

histos[qcdstr]["bkg_pt_sum"]=copy.deepcopy(histos[qcdstr]["bkg_pt_0"])
histos[qcdstr]["bkg_pt_sum"].Add(histos[qcdstr]["bkg_pt_1"])
histos[qcdstr]["bkg_pt_sum"].Add(histos[qcdstr]["bkg_pt_2"])


tocanv={"ht":[2,[0,5000]],"ptTIGHT":[2,[0,3000]],"ptTIGHTsum":[2,[0,3000]],"ptLOOSEGT":[2,[0,3000]],"ptLOOSEGTsum":[2,[0,3000]],"ptLOOSE":[2,[0,3000]],"ptLOOSEsum":[2,[0,3000]],"pt_":[2,[0,3000]],"pt_sum":[2,[0,3000]]}

for tc in tocanv:
        for ds in histos:
            if ds!=qcdstr:
                continue
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

            Sjet=False
            if (("pt" in tc)):
                Sjet=True

            if (("sum" in tc)):
                histoiter=[""]
                
            for iiter,ijet in enumerate(histoiter):
                    if ijet=="":
                        color=2
                    else:
                        color=ijet+1

                    if Sjet:
                            regionstr=""
                            bkgname="bkg_"+tc+str(ijet)
                            dataname=tc+str(ijet)
                            dathist=histos[ds][dataname]
                            print("ISSJET",ds,dataname,dathist.Integral())
                    else:
                            regionstr="LT"+str(ijet)+str(njet-ijet)
                            #if tc[0:2]=="pt":
                            #if len(tc)>2:

                             #   print(tc,len(tc))
                              #  print(tc[2])
                               # if tc[2]=="0" or tc[2]=="1" or tc[2]=="2":
                  
                                #        if regionstr!="LT21":
                                 #                continue

                            bkgname="bkg_"+tc+"_"+regionstr
                            dataname=tc+"_"+regionstr

                            dathist=histos[ds][dataname]
                            print("NOTSJET",ds,dataname,dathist.Integral())

                    dathist.SetLineColor(color)
                    dathist.SetTitle(";"+tc+"(GeV);events")
                    dathist.SetStats(0) 
                    dathist.Rebin(rebinval) 

                    dathist.GetXaxis().SetRangeUser(xrangeval[0],xrangeval[1])

                    bkghist = histos[ds][bkgname]
                    bkghist.SetLineColor(color)
                    bkghist.Rebin(rebinval) 

                    bkghist.GetXaxis().SetRangeUser(xrangeval[0],xrangeval[1])

                    main.cd()

                    dathist.GetXaxis().SetTitleSize (0.06)
                    dathist.GetXaxis().SetLabelSize (0.05)
                    dathist.GetYaxis().SetTitleSize (0.06)
                    dathist.GetYaxis().SetLabelSize (0.05)
                    dathist.Draw("same")   
                    bkghist.Draw("histsame") 
                    
                    
                    leg.AddEntry(bkghist,ds+regionstr+"bkg","L")
                    leg.AddEntry(dathist,ds+regionstr,"LE")

                    sub.cd()
                    allrat.append(copy.deepcopy(dathist) )
                    allrat[-1].Divide(bkghist)


                    allrat[-1].GetYaxis().SetRangeUser(0.5,1.5)
                    allrat[-1].SetTitle(";"+tc+"(GeV);")
                    allrat[-1].GetXaxis().SetTitleSize (0.12)
                    allrat[-1].GetXaxis().SetLabelSize (0.09)

                    allrat[-1].GetYaxis().SetTitleSize (0.12)
                    allrat[-1].GetYaxis().SetLabelSize (0.09)
                    allrat[-1].GetXaxis().SetRangeUser(xrangeval[0],xrangeval[1])


                    print("--Fit--",tc,regionstr)
                    #if regionstr=="LT21":
                    if iiter==0:
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




