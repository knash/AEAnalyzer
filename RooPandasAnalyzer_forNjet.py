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
class PreColumn():
    def __call__(self,df,EventInfo):
        EventInfo.eventcontainer["evweight"] = EventInfo.eventcontainer["lumi"]*EventInfo.eventcontainer["xsec"][EventInfo.dataset]/EventInfo.eventcontainer["nev"][EventInfo.dataset]
        df["Hists"]["logMSE_all"] = np.log(df["FatJet"]["iAEMSE"])
        df["Hists"]["weight"] *= EventInfo.eventcontainer["evweight"]
        #meh, should be added to  columnweights -- todo
        df["Hists"]["logMSE_all__weight"] = pd.Series(EventInfo.eventcontainer["evweight"], df["Hists"]["logMSE_all"].index, name="logMSE_all__weight")
        return df
class KinematicSelection():
    def __init__(self,njet,ptcut,msdcut):
        self.ptcut=ptcut
        self.njet=njet
        self.msdcut=msdcut
    def __call__(self,df,EventInfo):

        fjcut=(df["FatJet"]["pt"]>self.ptcut[0]) & (df["FatJet"]["msoftdrop"]>self.msdcut[0])&(df["FatJet"]["pt"]<self.ptcut[1]) & (df["FatJet"]["msoftdrop"]<self.msdcut[1])
        C0=((fjcut).sum(level=0)>0)

        df["FatJet"]=df["FatJet"][fjcut]

        df["FatJet"] = df["FatJet"].reset_index(level=1, drop=True)
        df["FatJet"] = df["FatJet"].set_index(df["FatJet"].groupby(level=0).cumcount().rename('subentry') , append=True)

        C1=((df["FatJet"]["pt"]>0.0).sum(level=0))==self.njet

        C2=df["HLT"]["PFHT900"][:,0]

        if (not ( C0 & C1 & C2 ).any()):
            return None
        return ( C0 & C1 & C2 )

class KinematicSelectionDR():
    def __init__(self,njet,drcut):
        self.drcut=drcut
        self.njet=njet
    def __call__(self,df,EventInfo):    
        alldiscs=[]

        for ijet in range(self.njet):
            #print("01")
            #print(ijet,df["FatJet"]["phi"])
            #print("02")
            idx = pd.IndexSlice

            #print("")

            ijetphi=df["FatJet"]["phi"][:,ijet]
            ijeteta=df["FatJet"]["eta"][:,ijet]
            drcut=None
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
                if isinstance(drcut,type(None)):
                    drcutjet=curcond
                else:
                    drcutjet=drcutjet&(curcond)

            alldiscs.append(drcutjet)

        for iad,ad in enumerate(alldiscs):
            if iad==0:
                evdisc=ad
            else:
                evdisc=evdisc&ad


        return ( evdisc )



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



class ColumnWeights():
    def __call__(self,df,EventInfo):
        keys=list(df["Hists"].keys())
        for hh in keys:
            #print(hh)
            if hh in ["njettight__njetloose","event","weight"]:
                continue
            if hh+"__weight" in df["Hists"]:
                continue
            df["Hists"][hh+"__weight"]=df["Hists"]["weight"]
            if (df["Hists"][hh].index.nlevels > df["Hists"]["weight"].index.nlevels )  :
                #print ("DROP")
                df["Hists"][hh]=df["Hists"][hh].droplevel(level=1)
            df["Hists"][hh+"__weight"] = df["Hists"][hh+"__weight"][df["Hists"][hh+"__weight"].index.isin(df["Hists"][hh].index)]
        df["Hists"]["njettight__njetloose__weight"]=df["Hists"]["njettight__weight"]
        return df

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
                    pass#print("Empty jet in Rate Numerator")
                try:
                    df["Hists"]["ptL"+str(ijet)+"_"+ebin]=df["FatJet"]["pt"][df["FatJet"]["loose"]][etacut][:,ijet]
                except:
                    pass
                    #print("Empty jet in Rate Denominator")
            regionstr="LT"+str(ijet)+str(njet-ijet)
            #print(df["Hists"]["njettight"])
            #print(df["Hists"]["njetloose"])
            df["Hists"]["ht_"+regionstr]=df["Hists"]["ht"][df["Hists"]["njettight"]==(njet-ijet)][df["Hists"]["njetloose"]==(ijet)]

            #df["FatJet"]["tight"][:,ijet] 
            #df["FatJet"]["loose"][:,ijet] 

        return df


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
            #print(regionstr)
        nloose=0
        for ll in loose:
            nloose+=ll
        ntight=0
        for tt in tight:
            ntight+=tt
        if ((nloose)==self.njet):
            #print(pt)
            #print(eta)
            maxbin=2**self.njet
            allregs=list(range(maxbin))
            allregs.reverse()
            Trate=[]

            for ijet in range(self.njet):
                for ebin in bkgparam["eta"]:
                    etacut=(bkgparam["eta"][ebin][0]<=eta[ijet]<bkgparam["eta"][ebin][1])
                    if etacut:
                        #print(pt,ijet,pt[ijet])
                        ptbin=RateHists["Rate"+ebin].FindBin(pt[ijet])
                        Trate.append(RateHists["Rate"+ebin].GetBinContent(ptbin))
            #print(Trate)
            #print(allregs)
            weights=[0.0]*(self.njet+1)
            nweights=[0.0]*(self.njet+1)
            for ar in allregs:
                #print(ar,bin(ar))
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
                #print(weight,ar,ntight)
            #print(weights,ar,ntight,weight)
            allret=[]
            for icweight,cweight in enumerate(weights):
                allret.append(ht)
                #print(nweights[icweight])
                allret.append(cweight*EventInfo.eventcontainer["evweight"])
        else:
            allret=[]
            for _ in range(self.njet+1):
                allret.append(ht)
                allret.append(0.0)
        #print (allret)
        return (allret)


chunklist =PInitDir("RooFlatFull")
bkgparam={}
bkgparam["eta"]={"E0":[0.0,0.4],"E1":[0.4,1.0],"E2":[1.0,float("inf")]}


branchestoread={
                    "Muon":["pt","eta","phi","mass"],
                    "FatJet":["pt","eta","phi","mass","msoftdrop","iAEMSE"],
                    "HLT":["PFHT900"],
                    "":["run","luminosityBlock","event"]
                    }

scalars=["","HLT"]
                       

if op_massrange=="all":
    sdcut=[0.0,float("inf")]
else:
    sdcuts=[[0.0,50.0],[50.0,100.0],[100.0,140.0],[140.0,200.0],[200.0,float("inf")]]
    sdcut=sdcuts[int(op_massrange)]

def MakeProc(njet,step,evcont):
    histostemp=OrderedDict  ([])
    if step==0:
        for ijet in range(njet+1):
            regionstr="LT"+str(ijet)+str(njet-ijet)
            histostemp["ht_"+regionstr]=TH1F("ht_"+regionstr,"ht_"+regionstr,400,0,4000)

            if ijet==njet:
                for jjet in range(njet):
                    histostemp["pt"+str(jjet)+"_"+regionstr]=TH1F("pt"+str(jjet)+"_"+regionstr,"pt"+str(jjet)+"_"+regionstr,100,0,2000)
                    histostemp["eta"+str(jjet)+"_"+regionstr]=TH1F("eta"+str(jjet)+"_"+regionstr,"eta"+str(jjet)+"_"+regionstr,48,0,2.4)
                    histostemp["phi"+str(jjet)+"_"+regionstr]=TH1F("phi"+str(jjet)+"_"+regionstr,"phi"+str(jjet)+"_"+regionstr,64,-3.2,3.2)
                    histostemp["mass"+str(jjet)+"_"+regionstr]=TH1F("mass"+str(jjet)+"_"+regionstr,"mass"+str(jjet)+"_"+regionstr,100,0.,300.)
                    histostemp["logMSE"+str(jjet)+"_"+regionstr]=TH1F("logMSE"+str(jjet)+"_"+regionstr,"logMSE"+str(jjet)+"_"+regionstr,100,-20.,0.)

            for ebin in bkgparam["eta"]:
                    #print("ptL"+str(ijet)+"_"+ebin)
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
proc = MakeProc(njet,0,evcont)
nproc=op_nproc
Mproc=PProcRunner(proc,nproc)
returndf=Mproc.Run()
for rr in returndf:
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

THists={}
LHists={}

bins=array.array('d',[0,300,400,600,800,1000,1500,10000])


for ijet in range(njet):
    for curhist in QCDhists:
        #print(curhist[:4],"ptL"+str(ijet))
        if curhist[:4] =="ptL"+str(ijet):
            Lstring=curhist
            Tstring=curhist.replace("ptL"+str(ijet),"ptT"+str(ijet))
            paramstr=Lstring.split("_")[-1]
            curhistL=QCDhists[Lstring]
            curhistT=QCDhists[Tstring]
            curhistL=curhistL.Rebin(len(bins)-1,curhistL.GetName()+"TEMP",bins)
            curhistT=curhistT.Rebin(len(bins)-1,curhistT.GetName()+"TEMP",bins)
            #print(ijet)
            if ijet==0:
                #print (paramstr,curhistT)
                THists[paramstr]=curhistT
                LHists[paramstr]=curhistL
            else:
                THists[paramstr].Add(curhistT)
                LHists[paramstr].Add(curhistL)

RateHists={}

for RH in LHists:
    #print(RH)
    #print(THists[RH].Integral())
    #print(LHists[RH].Integral())
    RateHists["Rate"+RH]=copy.deepcopy(THists[RH])
    RateHists["Rate"+RH].Divide(LHists[RH])
    RateHists["Rate"+RH].Write("Rate"+RH)



evcont["RateHists"]=RateHists

proc = MakeProc(njet,1,evcont)

Mproc=PProcRunner(proc,nproc)
returndf=Mproc.Run()
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



