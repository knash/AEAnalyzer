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
                  default	=	'0',
                  dest		=	'massrange',
                  help		=	'0,1,2,3,all')

parser.add_option('-a', '--aeval', metavar='F', type='string', action='store',
                  default	=	'95',
                  dest		=	'aeval',
                  help		=	'90,95,99')

(options, args) = parser.parse_args()
op_nproc=int(options.nproc)
op_njet=int(options.njet)
op_massrange=options.massrange
op_aeval=options.aeval

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

        fjcutmass=(df["FatJet"]["msoftdrop"]>self.msdcut[0])&(df["FatJet"]["msoftdrop"]<self.msdcut[1])
        df["FatJet"]=df["FatJet"][fjcutmass]

        C2=(df["FatJet"]["event"].count(level=0))==self.njet

        fjcut=fjcutpt&fjcutmass
        C0=((fjcut).sum(level=0)>0)
   
        C4=df["HLT"]["PFHT900"][:,0]

        if (not ( C0 & C1 & C2 & C4 ).any()):
            return None
        return ( C0 & C1 & C2  & C4 )

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

        return df


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

#make histograms to be used for creating the pass-to-fail ratio
class MakeHistsForBkg():
    def __init__(self,njet):
        self.njet=njet
    def __call__(self,df,EventInfo):
        bkgparam=EventInfo.eventcontainer["bkgparam"]
        for ijet in range(self.njet):
            for ebin in bkgparam["eta"]:
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
            regionstr="LT"+str(ijet)+str(njet-ijet)
            df["Hists"]["ht_"+regionstr]=df["Hists"]["ht"][df["Hists"]["njettight"]==(njet-ijet)][df["Hists"]["njetloose"]==(ijet)]

        return df

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
        except:
            print ("ERR")
            return None
        return args
    def __call__(self,args,EventInfo):
        bkgparam=EventInfo.eventcontainer["bkgparam"]
        RateHists=EventInfo.eventcontainer["RateHists"]
        ht=args[0]
        pt=[]
        eta=[]
        tight=[]
        loose=[]
        for ijet in range(self.njet):
            pt.append(args[ijet*4+1])
            eta.append(args[ijet*4+2])
            tight.append(args[ijet*4+3])
            loose.append(args[ijet*4+4])
        
            regionstr="LT"+str(ijet)+str(njet-ijet)
        nloose=0
        for ll in loose:
            nloose+=ll
        ntight=0
        for tt in tight:
            ntight+=tt
        if ((nloose)==self.njet):
            maxbin=2**self.njet
            allregs=list(range(maxbin))
            allregs.reverse()
            Trate=[]

            for ijet in range(self.njet):
                for ebin in bkgparam["eta"]:
                    etacut=(bkgparam["eta"][ebin][0]<=eta[ijet]<bkgparam["eta"][ebin][1])
                    if etacut:
                        ptbin=RateHists["Rate"+ebin].FindBin(pt[ijet])
                        Trate.append(RateHists["Rate"+ebin].GetBinContent(ptbin))
            weights=[0.0]*(self.njet+1)
            nweights=[0.0]*(self.njet+1)
            for ar in allregs:
                ntight=0
                weight=1.0
                for ibit,bit in enumerate(range(self.njet)):
                
                    curbit=(ar>>bit)&1
                    ntight+=curbit
                    
                    if curbit:
                        weight*=Trate[ibit]
                    else:
                        weight*=1.0

                weights[self.njet-ntight]+=weight
                nweights[self.njet-ntight]+=1.0
            allret=[]
            for icweight,cweight in enumerate(weights):
                allret.append(ht)
                allret.append(cweight*EventInfo.eventcontainer["evweight"])
        else:
            allret=[]
            for _ in range(self.njet+1):
                allret.append(ht)
                allret.append(0.0)
        return (allret)


chunklist =PInitDir("RooFlatFull")
bkgparam={}

#three eta bins (probably overkill)
bkgparam["eta"]={"E0":[0.0,0.4],"E1":[0.4,1.0],"E2":[1.0,float("inf")]}

#todo: muon triggers a failure mode as sometimes events have no muons and no filter remo 
branchestoread={
                    #"Muon":["pt","eta","phi","mass"],
                    "FatJet":["pt","eta","phi","mass","msoftdrop","iAEMSE"],
                    "HLT":["PFHT900"],
                    "":["run","luminosityBlock","event"]
                    }

scalars=["","HLT"]
                       

if op_massrange=="all":
    sdcut=[0.0,float("inf")]
else:
    #sdcuts=[[0.0,50.0],[50.0,100.0],[100.0,140.0],[140.0,200.0],[200.0,float("inf")]]
    sdcuts=[[0.0,50.0],[50.0,float("inf")]]
    sdcut=sdcuts[int(op_massrange)]


#customize a multi-step processor
def MakeProc(njet,step,evcont):
    histostemp=OrderedDict  ([])
    if step==0:
        for ijet in range(njet+1):
            regionstr="LT"+str(ijet)+str(njet-ijet)
            histostemp["ht_"+regionstr]=TH1F("ht_"+regionstr,"ht_"+regionstr,400,0,4000)

            for ebin in bkgparam["eta"]:
                    histostemp["ptL"+str(ijet)+"_"+ebin]=TH1F("ptL"+str(ijet)+"_"+ebin,"ptL"+str(ijet)+"_"+ebin,1000,0,10000)
                    histostemp["ptT"+str(ijet)+"_"+ebin]=TH1F("ptT"+str(ijet)+"_"+ebin,"ptT"+str(ijet)+"_"+ebin,1000,0,10000)
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
            histostemp["bkg_ht_"+regionstr]=TH1F("bkg_ht_"+regionstr,"bkg_ht_"+regionstr,400,0,4000)
            hpass.append(["Hists","bkg_ht_"+regionstr])
            hpass.append(["Hists","bkg_ht_"+regionstr+"__weight"])
        myana=  [
                PColumn(PreColumn()),
                PFilter(KinematicSelection(njet,[200.0,float("inf")],sdcut)),     
                PFilter(KinematicSelectionDR(njet,1.4)),
                PColumn(MakeTags(njet)),
                PRow(hpass,BkgEst(njet)),
                PColumn(ColumnWeights()),
                ]
    for hist in histostemp:
        histostemp[hist].Sumw2() 


    histos= {}
    for ds in chunklist:
        #chunklist[ds]=chunklist[ds][:10]
        histos[ds]=copy.deepcopy(histostemp)

    return PProcessor(chunklist,histos,branchestoread,myana,eventcontainer=evcont,atype="flat",scalars=scalars)
njet=op_njet

evcont={"msescale":{"TT":1.10,"QCD_HT1500to2000":0.9},"lumi":(1000.0*137.65),"xsec":{"TT":0.047,"QCD_HT1500to2000":101.8},"nev":{"TT":305963.0,"QCD_HT1500to2000":10655313.0}}
evcont["bkgparam"]=bkgparam

#Step0:make hists for pass-to-fail ratio
proc = MakeProc(njet,0,evcont)
nproc=op_nproc
Mproc=PProcRunner(proc,nproc)
returndf=Mproc.Run()


#Print MSE quantilles
for rr in returndf:
    if  "logMSE_all" in returndf[rr]:
        print  (rr ,"cut90",returndf[rr]["logMSE_all"].quantile(0.90))
        print  (rr ,"cut95",returndf[rr]["logMSE_all"].quantile(0.95))
        print  (rr ,"cut99",returndf[rr]["logMSE_all"].quantile(0.99))

output = TFile("FromFlatPandas_AE"+op_aeval+"_M"+op_massrange+"_Njet"+str(op_njet)+".root","recreate")
output.cd()
ratehistos=proc.hists
for ds in ratehistos:
    for var in ratehistos[ds]:
            ratehistos[ds][var].Write(ds+"__"+var)
QCDhists=ratehistos["QCD_HT1500to2000"]



#Make pass-to-fail ratio TR(pt,eta)
THists={}
LHists={}
bins=array.array('d',[0,300,400,600,800,1000,1500,10000])
for ijet in range(njet):
    for curhist in QCDhists:
        if curhist[:4] =="ptL"+str(ijet):
            Lstring=curhist
            Tstring=curhist.replace("ptL"+str(ijet),"ptT"+str(ijet))
            paramstr=Lstring.split("_")[-1]
            curhistL=QCDhists[Lstring]
            curhistT=QCDhists[Tstring]
            curhistL=curhistL.Rebin(len(bins)-1,curhistL.GetName()+"TEMP",bins)
            curhistT=curhistT.Rebin(len(bins)-1,curhistT.GetName()+"TEMP",bins)

            if ijet==0:
                THists[paramstr]=curhistT
                LHists[paramstr]=curhistL
            else:
                THists[paramstr].Add(curhistT)
                LHists[paramstr].Add(curhistL)

RateHists={}

for RH in LHists:
    print(RH)
    print(THists[RH].Integral())
    print(LHists[RH].Integral())
    RateHists["Rate"+RH]=copy.deepcopy(THists[RH])
    RateHists["Rate"+RH].Divide(LHists[RH])
    RateHists["Rate"+RH].Write("Rate"+RH)

evcont["RateHists"]=RateHists



#Step1:use pass-to-fail ratio to predict background
proc = MakeProc(njet,1,evcont)
Mproc=PProcRunner(proc,nproc)
returndf=Mproc.Run()

#Plot ht
histos=proc.hists
rebinval=4
for ds in histos:
    for var in histos[ds]:
            histos[ds][var].Write(ds+"__"+var)
    print(histos[ds])
    canv=TCanvas("ht"+ds,"ht"+ds,700,500)
    gPad.SetLeftMargin(0.12)
    leg = TLegend(0.65, 0.65, 0.84, 0.84)
    leg.SetFillColor(0)
    leg.SetBorderSize(0)
    histoiter=list(range(njet+1))
    histoiter.reverse()
    histoiter.pop(0)
    for ijet in histoiter:
            regionstr="LT"+str(ijet)+str(njet-ijet)

            bkgname="bkg_ht_"+regionstr
            dataname="ht_"+regionstr
            color=ijet+1

            ratehistos[ds][dataname].SetLineColor(color)
            ratehistos[ds][dataname].SetTitle(";ht(GeV);events")
            ratehistos[ds][dataname].SetStats(0) 
            ratehistos[ds][dataname].Rebin(rebinval) 
            ratehistos[ds][dataname].Draw("same")   


            histos[ds][bkgname].SetLineColor(color)
            histos[ds][bkgname].Rebin(rebinval)  
            histos[ds][bkgname].Draw("histsame") 
        

            leg.AddEntry(histos[ds][bkgname],ds+regionstr+"bkg","L")
            leg.AddEntry(ratehistos[ds][dataname],ds+regionstr,"P")


    leg.Draw()
    canv.SetLogy()
    canv.Write()


output.Close()



