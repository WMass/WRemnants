# WRemnants

## Instructions:

Activate the singularity image (to be done every time before running code)
```bash
singularity run /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/bendavid/cmswmassdocker/wmassdevrolling\:latest
```

Activate git Large File Storage (only need to do this once for a given user/home directory)
```
git lfs install
```
    
Get the code (after forking from the central WMass repository)
```bash
MY_GIT_USER=$(git config user.github)
git clone --recurse-submodules git@github.com:$MY_GIT_USER/WRemnants.git
cd WRemnants/
git remote add upstream git@github.com:WMass/WRemnants.git
```

Get updates from the central repository (and main branch)
```bash
git pull --recurse-submodules upstream main
git push origin main
```

Get combinetf. Need to run cmssw-cc7 (outside of the other singularity image) to work in a special centos7 environment, which allows you to work with CMSSW. If you plan to contribute to the combinetf code, you may first fork from: https://github.com/bendavid/HiggsAnalysis-CombinedLimit
```
    cmssw-cc7
    cd /some/path/to/download/code/
    export SCRAM_ARCH="slc7_amd64_gcc700"
    cmsrel CMSSW_10_6_19_patch2
    cd CMSSW_10_6_19_patch2/src/
    cmsenv
    git clone -o bendavid -b tensorflowfit git@github.com:bendavid/HiggsAnalysis-CombinedLimit.git HiggsAnalysis/CombinedLimit
    cd HiggsAnalysis/CombinedLimit
    scram b -j 8
    #
    # if everything worked fine, folder scripts/ contains "combineCards.py, combinetf.py, commentUncerts.py, pruneUncerts.py, text2hdf5.py, text2workspace.py"
    # the reference branch is bendavid/tensorflowfit
    # so your current local branch should also be tensorflowfit
    #
    # optional for developments, if you have your own remote fork
    git remote add origin git@github.com:<YOUR_GITHUB_USER>/HiggsAnalysis-CombinedLimit.git
    git checkout -b myBranch 
    git push origin myBranch
```

To run the fit with combinetf
```
    cmssw-cc7
    cd /path/to/CMSSW_10_6_19_patch2/src/HiggsAnalysis/CombinedLimit/
    cmsenv
    cd /wherever/you/like/
    <commands to run fit> # e.g. using WRemnants/scripts/combine/fitManager.py
```
    
        
### Contribute to the code

**Guidelines**
 * Use camel case practice for command line arguments and avoid the "dest" keyword.
 * Use snake case practice for function names.
 * Class names should start with capital letters.
 * When making a new PR, it should target only one subject at a time. This makes it more easy to validate and the integration faster. PR on top of other PR are ok when it is noted in the description, e.g. this PR is on top of PR XYZ.

### Run the code
Source the setup script.
It will create some environment variables to ease access to some folders:
 * WREM_BASE: it points to ./WRemnants/ where all the code is
 * COMBINE_STUDIES: folder with codes to create datacards and root files for combine, and run the fit
 * PLOTS: folder with some scripts for plots and dedicated studies for analysis
```bash
source WRemnants/setup.sh
```
### Theory agnostic analysis

Make histograms (only nominal and mass variations for now, systematics are being developed)
```
/usr/bin/time -v python scripts/histmakers/mw_with_mu_eta_pt.py -o outputFolder/ --met DeepMETReso  --theoryAgnostic
```

Prepare datacards and root files with TH2 (stat-only for now)
```
/usr/bin/time -v python scripts/combine/setupCombineWMass.py -i outputFolder/mw_with_mu_eta_pt_scetlib_dyturboCorr.hdf5  -o outputFolder/  --absolutePathInCard --theoryAgnostic
```
To remove the backgrounds and run signal only one can add __--excludeProcGroups Top Diboson Fake Zmumu Ztautau Wtaunu BkgWmunu__

Run the fit (for charge combination)
```
python WRemnants/scripts/combine/fitManager.py -i outputFolder/WMass_pt_eta_statOnly/ --skip-fit-data --theoryAgnostic --comb
```

            
### Traditional analysis
    
Make histograms for WMass (for Wlike the script is __mz_wlike_with_mu_eta_pt.py__).
```
python WRemnants/scripts/histmakers/mw_with_mu_eta_pt.py -o outputFolder/
```
More options are loaded from **WRemnants/utilities/common.py**

Make the datacards for single charges and prepare the TH2 histograms for combinetf.
```
python WRemnants/scripts/combine/setupCombineWMass.py -i outputFolder/mw_with_mu_eta_pt_scetlib_dyturboCorr.hdf5 -o outputFolder/
```
The input file is the output of the previous step.
The default path specified with __-o__ is the local folder. A subfolder with name identifying the specific analysis (e.g. WMass_pt_eta/) is automatically created inside it. Some options may add tags to the folder name: for example, using --doStatOnly will  call the folder WMass_pt_eta_statOnly/.
 
Combine the datacards for single charges and run the fit (Asimov only)
```
python WRemnants/scripts/combine/fitManager.py -i outputFolder/WMass_pt_eta/ --comb --skip-fit-data
```
Run the fits for single charges (Asimov only). These can be produced in the same output folder as the combination, since a postfix is automatically appended to the output card and fit result files.
```
python WRemnants/scripts/combine/fitManager.py -i outputFolder/WMass_pt_eta/ --fit-single-charge --skip-fit-data [-c plus|minus|both]
```

**NOTE**:
 * to run __fitManager.py__ one has to set a Centos 7 environment with __cmssw-cc7__. Then, one has to activate __cmsenv__ from the folder where combine is installed (once the environment is set one can keep working from inside WRemnants).
 * Each script has tons of options, to customize a gazillion of things, it's simpler to learn them by asking an expert rather that having an incomplete summary here (developments happen faster than documentation anyway).

### Making plots

There are many scripts to do every kind of plotting, and different people may have their own ones. We'll try to put a minimal list with examples here ASAP.

Plot Wmass histograms from hdf5 file (from Wmass histmaker) in the 4 iso-MT regions (can choose only some). It also makes some plots for fakes depending on the chosen region. It is also possible to select some specific processes to put in the plots.
```
python scripts/analysisTools/tests/testShapesIsoMtRegions.py mw_with_mu_eta_pt_scetlib_dyturboCorr.hdf5 outputFolder/ [--isoMtRegion 0 1 2 3]
```
    
Plot prefit shapes (requires root file from setupCombineWmass.py as input)
```
python scripts/analysisTools/w_mass_13TeV/plotPrefitTemplatesWRemnants.py WMassCombineInput.root outputFolder/ [-l 16.8] [--pseudodata <pseudodataHistName>] [--wlike]
```

Make study of fakes for mW analysis, checking mT dependence, with or without dphi cut (see example inside the script for more options). Even if the histmaker was run with the dphi cut, the script uses a dedicated histograms __mTStudyForFakes__ created before that cut, and with dphi in one axis.
```
python scripts/analysisTools/tests/testFakesVsMt.py mw_with_mu_eta_pt_scetlib_dyturboCorr.hdf5 outputFolder/ --rebinx 4 --rebiny 2 --mtBinEdges "0,5,10,15,20,25,30,35,40,45,50,55,60,65" --mtNominalRange "0,40" --mtFitRange "0,40" --fitPolDegree 1 --integralMtMethod sideband --maxPt 50  --met deepMet [--dphiMuonMetCut 0.25]
```

Make quick plots of any 1D distribution produced with any histmaker
```
python scripts/analysisTools/tests/testPlots1D.py mz_wlike_with_mu_eta_pt_scetlib_dyturboCorr.hdf5 outputFolder/ --plot transverseMass_uncorr transverseMass -x "Uncorrected Wlike m_{T} (GeV)" "Corrected Wlike m_{T} (GeV)"
```

Make plot with mW impacts from a single fit result
```
python scripts/analysisTools/w_mass_13TeV/makeImpactsOnMW.py fitresults_123456789.root -o outputFolder/  --scaleToMeV --showTotal -x ".*eff_(stat|syst)_" [--postfix plotNamePostfix]
```

Make plot with mW impacts comparing two fit results
```
python scripts/analysisTools/w_mass_13TeV/makeImpactsOnMW.py fitresults_123456789.root -o outputFolder/  --scaleToMeV --showTotal --compareFile fitresults_123456789_toCompare.root --printAltVal --legendEntries "Nominal" "Alternate" -x ".*eff_(stat|syst)_" [--postfix plotNamePostfix]
```

Print impacts without plotting (no need to specify output folder)
```
python w_mass_13TeV/makeImpactsOnMW.py fitresults_123456789.root --scaleToMeV --showTotal --justPrint
```
    