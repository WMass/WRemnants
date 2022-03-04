import narf
import logging
import subprocess
import glob
import pathlib

lumicsv = f"{pathlib.Path(__file__).parent.parent}/data/bylsoutput.csv"
lumijson = f"{pathlib.Path(__file__).parent.parent}/data/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt"

#TODO add the rest of the samples!
def makeFilelist(paths, maxFiles=-1, eos=False):
    filelist = []
    for path in paths:
        filelist.extend(glob.glob(path) if not eos else buildXrdFileList(path, "eoscms.cern.ch"))
    return filelist if maxFiles < 0 else filelist[:maxFiles]

def getDatasets(maxFiles=-1, filt=None):
    dataPostVFP = narf.Dataset(name = "dataPostVFP",
        filepaths = makeFilelist(["/scratch/shared/originalNANO/NanoV8Data_Mar2021/Run2016F_postVFP/*",
            "/scratch/shared/originalNANO/NanoV8Data_Mar2021/Run2016G/*",
            "/scratch/shared/originalNANO/NanoV8Data_Mar2021/Run2016H/*",
        ], maxFiles),
        is_data = True,
        lumi_csv = lumicsv,
        lumi_json = lumijson
    )

    ZmmPostVFP = narf.Dataset(name = "ZmumuPostVFP",
        filepaths = makeFilelist(
            ["/scratch/shared/NanoAOD/DYJetsToMuMu_M-50_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/MCPostVFPWeightFix/211114_184608/000*/*.root"], maxFiles),
        is_data = False,
        xsec = 1976.1,
    )

    ZmmPostVFP_bugfix = narf.Dataset(name = "ZmumuPostVFP_bugfix",
        filepaths = makeFilelist(
            ["/eos/cms/store/cmst3/group/wmass/w-mass-13TeV/NanoGen/DYJetsToMuMu_svn3900_BugFix_TuneCP5_13TeV-powheg-MiNNLO-pythia8-photos/RunIISummer15wmLHEGS/220125_235229/000*/*.root"], maxFiles),
        is_data = False,
        xsec = 1976.1,
    )

    BR_TAUToMU = 0.1739
    BR_TAUToE = 0.1782
    ZttPostVFP = narf.Dataset(name = "ZtautauPostVFP",
        filepaths = makeFilelist(
            ["/scratch/shared/originalNANO/DYJetsToTauTau_postVFP/*/*.root"], maxFiles),
        is_data = False,
        # At least one tau->e or mu decay, so everything that's not all other decays
        xsec = ZmmPostVFP.xsec*(1.-(1.-(BR_TAUToMU+BR_TAUToE)**2)),
    )

    WpmunuPostVFP = narf.Dataset(name = "WplusmunuPostVFP",
        filepaths = makeFilelist(
            ["/scratch/shared/originalNANO_newWithAltPDF/WplusJetsToMuNu_postVFP/*/*.root"], maxFiles),
        is_data = False,
        xsec = 11572.19,
    )
    
    WmmunuPostVFP = narf.Dataset(name = "WminusmunuPostVFP",
        filepaths = makeFilelist(
                ["/scratch/shared/originalNANO_newWithAltPDF/WminusJetsToMuNu_postVFP/*/*.root"], maxFiles),
            is_data = False,
            xsec = 8562.66,
    )

    WptaunuPostVFP = narf.Dataset(name = "WplustaunuPostVFP",
        filepaths = makeFilelist(
                ["/scratch/shared/originalNANO_newWithAltPDF/WplusJetsToTauNu_postVFP/*/*.root"], maxFiles),
            is_data = False,
            xsec = BR_TAUToMU*WpmunuPostVFP.xsec,
    )

    WmtaunuPostVFP = narf.Dataset(name = "WminustaunuPostVFP",
        filepaths = makeFilelist(
                ["/scratch/shared/originalNANO_newWithAltPDF/WminusJetsToTauNu_postVFP/*/*.root"], maxFiles),
            is_data = False,
            xsec = BR_TAUToMU*WmmunuPostVFP.xsec,
    )

    ttbarlnuPostVFP = narf.Dataset(name = "TTLeptonicPostVFP",
        filepaths = makeFilelist(
                ["/scratch/shared/originalNANO/TTbar_2l2nu_postVFP/*.root"], maxFiles),
            is_data = False,
            xsec = 88.29,
    )

    ttbarlqPostVFP = narf.Dataset(name = "TTSemileptonicPostVFP",
        filepaths = makeFilelist(
                ["/scratch/shared/originalNANO/TTbar_SemiLeptonic_postVFP/*.root"], maxFiles),
            is_data = False,
            xsec = 365.64,
    )

    # TODO: should really use the specific decay channels
    wwPostVFP = narf.Dataset(name = "WWPostVFP",
        filepaths = makeFilelist(
                ["/scratch/shared/originalNANO/WW_inclusive_postVFP/*.root"], maxFiles),
            is_data = False,
            xsec = 75.8,
    )

    allPostVFP = [dataPostVFP, WpmunuPostVFP, WmmunuPostVFP, WptaunuPostVFP, WmtaunuPostVFP, ZmmPostVFP, ZmmPostVFP_bugfix,
        ZttPostVFP, ttbarlnuPostVFP, ttbarlqPostVFP, wwPostVFP]

    if filt:
        print("Filtering")
        return list(filter(filt, allPostVFP))

    return allPostVFP

def buildXrdFileList(path, xrd):
    xrdpath = path[path.find('/store'):]
    logging.debug(f"Looking for path {xrdpath}")
    f = subprocess.check_output(['xrdfs', f'root://{xrd}', 'ls', xrdpath]).decode(sys.stdout.encoding)
    return filter(lambda x: "root" in x[-4:], f.split())
