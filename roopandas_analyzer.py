from roopandas_functions import PSequential,PFilter,PProduce,PAnalyze,PProcessor,PInitDir
from glob import glob
from ROOT import TH1F,TLorentzVector,TFile
from collections import OrderedDict
import numpy as np
import copy

chunklist =PInitDir("RooPandas")

histostemp=OrderedDict([("invm",TH1F("invm","invm",100,0,5000)),("logMSE",TH1F("logMSE","logMSE",80,-20,0))])
histos= {}

for hh in chunklist:
    histos[hh]=copy.deepcopy(histostemp)

def KinematicSelection(df):
    C1=(df["FatJet_pt"].str[0]>200.) & (df["FatJet_pt"].str[1]>200.) & (df["FatJet_msoftdrop"].str[0]>50.) & (df["FatJet_msoftdrop"].str[1]>50.)
    return df[C1]
class MakePtEtaPhiMLV():
    def __init__(self,args):
        self.args=args
    def __call__(self,pt,eta,phi,m):
        LV=[]
        for ir in range(len(pt)):
            LV.append(TLorentzVector())
            LV[-1].SetPtEtaPhiM(pt[ir],eta[ir],phi[ir],m[ir])

        return LV
class MyAnalyzer():
    def __init__(self,outmap):
        self.outmap=outmap
    
    def __call__(self,row):
        rvals=[0.0]*len(self.outmap)
        if row["nFatJet"]>1:

                #print (row)
                invm=(row["FatJet_LV"][0] + row["FatJet_LV"][1]).M()
                rvals[self.outmap.index("invm")]=invm
                rvals[self.outmap.index("logMSE")]=np.log(row.FatJet_iAEMSE[0])
            
        return rvals



myana = PSequential([PFilter(KinematicSelection),PProduce("FatJet_LV",MakePtEtaPhiMLV(["FatJet_pt","FatJet_eta","FatJet_phi","FatJet_mass"])),PAnalyze(MyAnalyzer(["invm","logMSE"]))])

branchestoread=["FatJet_pt","FatJet_eta","FatJet_phi","FatJet_mass","FatJet_iAEMSE","FatJet_msoftdrop","nFatJet"]
proc=PProcessor(chunklist,histos,branchestoread,myana,nproc=1)
proc.Run()
output = TFile("FromPandas.root","recreate")
output.cd()
for ds in histos:
    for var in histos[ds]:
            histos[ds][var].Write(ds+"__"+var)
output.Close()


