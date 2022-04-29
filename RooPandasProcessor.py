
from glob import glob
import uproot3
import numpy as np
import pandas as pd
import tqdm
from RooPandasFunctions import PNanotoDataFrame,PSequential,PColumn,PFilter,PRow,PProcessor,PProcRunner,PInitDir
from collections import OrderedDict

from optparse import OptionParser
parser = OptionParser()

parser.add_option('--data', metavar='F', action='store_true',
		  default=False,
		  dest='data',
		  help='data')

(options, args) = parser.parse_args()


#Define Datasets and corresponding file selections
fnames={}

if (options.data):
        #fnames["DATA_2017_B"]= sorted(glob('/cms/knash/EOS/JetHT/Run2017B-09Aug2019_UL2017-v1_NanoB2GNano2017data_v1/220225_203803/*/*.root'))

        fnames["DATA_2016_B"]= sorted(glob('/cms/knash/EOS/JetHT/Run2016B-21Feb2020_ver2_UL2016_HIPM-v1_NanoB2GNano2016data_v1/*/*/*.root'))
        fnames["DATA_2016_C"]= sorted(glob('/cms/knash/EOS/JetHT/Run2016C-21Feb2020_UL2016_HIPM-v1_NanoB2GNano2016data_v1/*/*/*.root'))
        fnames["DATA_2016_D"]= sorted(glob('/cms/knash/EOS/JetHT/Run2016D-21Feb2020_UL2016_HIPM-v1_NanoB2GNano2016data_v1/*/*/*.root'))
        fnames["DATA_2016_E"]= sorted(glob('/cms/knash/EOS/JetHT/Run2016E-21Feb2020_UL2016_HIPM-v1_NanoB2GNano2016data_v1/*/*/*.root'))
        fnames["DATA_2016_F"]= sorted(glob('/cms/knash/EOS/JetHT/Run2016F-21Feb2020_UL2016_HIPM-v1_NanoB2GNano2016data_v1/*/*/*.root'))
        fnames["DATA_2016_G"]= sorted(glob('/cms/knash/EOS/JetHT/Run2016G-21Feb2020_UL2016-v1_NanoB2GNano2016data_v1/*/*/*.root'))
        fnames["DATA_2016_H"]= sorted(glob('/cms/knash/EOS/JetHT/Run2016H-21Feb2020_UL2016-v1_NanoB2GNano2016data_v1/*/*/*.root'))

        #fnames["DATA_2017_B"]= sorted(glob('/cms/knash/EOS/JetHT/Run2017B-09Aug2019_UL2017-v1_NanoB2GNano2017data_v1/220225_203803/*/*.root'))
else:
        #fnames["QCD_HT1000to1500"] = sorted(glob('/cms/knash/EOS/QCD_HT1000to1500_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8/RunIISummer20UL16MiniAOD-106X_mcRun2_asymptotic_v13-v1_NanoB2GNano2016mc_v1/*/*/*.root'))
        #fnames["QCD_HT1500to2000"]= sorted(glob('/cms/knash/EOS/QCD_HT1500to2000_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8/RunIISummer20UL16MiniAOD-106X_mcRun2_asymptotic_v13-v1_NanoB2GNano2016mc_v1/*/*/*.root'))
        #fnames["QCD_HT2000toInf"]= sorted(glob('/cms/knash/EOS/QCD_HT2000toInf_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8/RunIISummer20UL16MiniAOD-106X_mcRun2_asymptotic_v13-v1_NanoB2GNano2016mc_v1/*/*/*.root'))
        #fnames["TT"] = sorted(glob('/cms/knash/EOS/ZprimeToTT_M2500_W25_TuneCP2_PSweights_13TeV-madgraph-pythiaMLM-pythia8/RunIISummer20UL16MiniAOD-106X_mcRun2_asymptotic_v13-v2_NanoB2GNano2016mc_v1/*/*/*.root'))
        #fnames["WgWg"] = sorted(glob('/cms/knash/EOS/SQSQtoqchiqchitoWs_M1500_M400_M200/knash-RunIISummer20UL16MiniAOD-106X_mcRun2_asymptotic_v13-76761e5076679f48cfaad96c1b8156aa_NanoB2GNano2016mc_v1/*/*/*.root'))
        fnames["HgHg_15001400"]=sorted(glob('/cms/knash/EOS/SQSQtoqchiqchitoHiggs_M1500_M1400_M200/knash-RunIISummer20UL16MiniAOD-106X_mcRun2_asymptotic_v13-76761e5076679f48cfaad96c1b8156aa_NanoB2GNano2016mc_v1/*/*/*.root'))
        fnames["PgPg_15001400"]=sorted(glob('/cms/knash/EOS/SQSQtoqchiqchitoPhotons_M1500_M1400_M200/knash-RunIISummer20UL16MiniAOD-106X_mcRun2_asymptotic_v13-76761e5076679f48cfaad96c1b8156aa_NanoB2GNano2016mc_v1/*/*/*.root'))
        fnames["PgPg_1500400"]=sorted(glob('/cms/knash/EOS/SQSQtoqchiqchitoPhotons_M1500_M400_M200/knash-RunIISummer20UL16MiniAOD-106X_mcRun2_asymptotic_v13-76761e5076679f48cfaad96c1b8156aa_NanoB2GNano2016mc_v1/*/*/*.root'))
        fnames["WgWg_15001400"]=sorted(glob('/cms/knash/EOS/SQSQtoqchiqchitoWs_M1500_M1400_M200/knash-RunIISummer20UL16MiniAOD-106X_mcRun2_asymptotic_v13-76761e5076679f48cfaad96c1b8156aa_NanoB2GNano2016mc_v1/*/*/*.root'))
        fnames["WgWg_1500400"]=sorted(glob('/cms/knash/EOS/SQSQtoqchiqchitoWs_M1500_M400_M200/knash-RunIISummer20UL16MiniAOD-106X_mcRun2_asymptotic_v13-76761e5076679f48cfaad96c1b8156aa_NanoB2GNano2016mc_v1/*/*/*.root'))

fileset={}
for ffi in fnames:
    fileset[ffi]=[ffj.replace("/eos/uscms/","root://cmsxrootd.fnal.gov///") for ffj in fnames[ffi]]
    fileset[ffi]=fileset[ffi]
    #fileset[ffi]=fileset[ffi][:10]

#This is the Nano->Parquet file reduction factor
batchesperfile={
                "TT":3,
                "HgHg_15001400":2,
                "PgPg_15001400":2,
                "PgPg_1500400":2,
                "WgWg_15001400":2,
                "WgWg_1500400":2,
                "QCD_HT1500to2000":5,
                "QCD_HT1000to1500":5,
                "QCD_HT2000toInf":5,
                "DATA_2016_B":40,
                "DATA_2016_C":40,
                "DATA_2016_D":40,
                "DATA_2016_E":40,
                "DATA_2016_F":40,
                "DATA_2016_G":40,
                "DATA_2016_H":40}


#Keep only the branches you want "Jet",["pt"] would be the branch Jet_pt in the NanoAOD
branchestokeep=OrderedDict([("Jet",["pt","eta","phi","mass"]),("FatJet",["pt","eta","phi","mass","msoftdrop","iAEMSE","iAEL0","iAEL1","iAEL2","iAEL3","iAEL4","iAEL5"]),("",["run","luminosityBlock","event"])])


#Trim out element indices you dont want (ie only keep top 5 jets etc)
mind={"FatJet":5,"Jet":10,"Muon":5,"":None,"HLT":None}
#mind={"FatJet":5,"LHEPart":10,"Jet":10,"Muon":5,"":None,"HLT":None}
if (not options.data):
        branchestokeep["FatJet"].append("hadronFlavour")
        #branchestokeep["LHEPart"]=["pt","eta","phi","mass","pdgId","status"]
        #mind["LHEPart"]=10
#It is possible to pass a column selection here similar to the analyzer.  
#Clearly the syntax is overly complicated compared to the analyzer -- to improve.  
#This is useful for skimming and calculating a value from collections that you dont want to save.
#ex/calculate ht from ak4 jets, then drop ak4s:
class ColumnSelection():
    def __call__(self,df,EventInfo):

            htdf=pd.DataFrame()
            htdf["ht"]=df["Jet_pt"].groupby(level=0).sum()
            htdf['subentry'] = 0
            htdf.set_index('subentry', append=True, inplace=True)
            df=pd.concat((df,htdf),axis=1)

            df=df.drop(["Jet_pt","Jet_eta","Jet_phi","Jet_mass"],axis=1)

            return df
         
skim=  [
        #PColumn(ColumnSelection()),
       ]

#Run it.  nproc is the number of processors.  >1 goes into multiprocessing model
PNanotoDataFrame(fileset,branchestokeep,filesperchunk=batchesperfile,nproc=6,atype="flat",dirname="RooFlatFull",maxind=mind,seq=skim).Run()


