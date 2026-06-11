# WRemnants

WRemnants is the analysis framework for the CMS electroweak precision measurements such as the W boson mass, Z boson mass, strong coupling constraint, cross section measurements, and related studies on generator level, experimental calibrations, and future projections. It handles the full analysis chain from processing collision events (NanoAOD) into histograms, through systematic uncertainty estimation, to fit input preparation. The statistical inference is performed by the companion [rabbit](https://github.com/WMass/rabbit) framework.

## Instructions

### First time setup

Activate the container image (to be done every time before running code). 
Depending on the cluster you are working on you will need to set the directories that you want to access from within the container. E.g.
```bash
export APPTAINER_BIND="/tmp,/run,/cvmfs/etc/grid-security,/home/,/work/,/data"
```
And then start the container with
```bash
singularity run /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/bendavid/cmswmassdocker/wmassdevrolling\:latest
```
Where a flag `--nv` needs to be added to use NVIDIA GPUs (e.g. for the fit).

Activate git Large File Storage (only need to do this once for a given user/home directory)
``` bash
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

Activate git pre-commit hooks (only need to do this once when checking out)
``` bash
git config --local include.path ../.gitconfig
```
If the pre-commit hook is doing something undesired, it can be bypassed by adding “--no-verify” when doing “git commit”.

### Each session setup
Everytime a new session is started, the first thing to do is enabling the singularity (with an adapted APPTAINER_BIND variable)
```bash
export APPTAINER_BIND="/tmp,/run,/cvmfs/etc/grid-security,/home/,/work/,/data"
singularity run /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/bendavid/cmswmassdocker/wmassdevrolling\:latest
```
and to source the setup script to execute the setup of submodules and create some environment variables to ease access to some folders.
```bash
source WRemnants/setup.sh
```

### Project overview
The project contains several submodules that point to standalone repositories that may or may not be used also for other projects. Those are:
* [narf](https://github.com/bendavid/narf): This provides the computational backand for the event processing and boost histogram production using Root Data Frames (RDF).
* [wums](https://github.com/WMass/wums): This is a pure python based submodule containing utility functions that can be more widely used such as the input/output tools, common plotting functions, and histogram manipulation.
* [wremnants-data](https://gitlab.cern.ch/cms-wmass/wremnants-data): This repository contains resource files needed for the analysis, such as data quality .json and by lumisection .csv files, experimental scale factors such as for the efficiencies, theory correction files etc. . It is on CERN gitlab using the large file storage. 
* [rabbit](https://github.com/WMass/rabbit): This is the fitting framework using tensorflow 2.x as backend.

The WRemnants project itself is structured using different folders:
* `notebooks/`: jupyter-notebooks for data exploration and quick tests, mainly user specific
* `scripts/`: All executable files should go here such as
  * `scripts/analysisTools/`: analysis and user specific scripts
  * `scripts/ci/`: for the github continuous integration (CI) workflow. These scripts are executed automatically and get triggered e.g. by opening a pull request (PR).
  * `scripts/corrections/`: to compute correction files used in later steps of the analysis
  * `scripts/hepdata/`: for data preservation
  * `scripts/histmakers/`: for the processing of columnar data (mainly NanoAOD files) into histograms
  * `scripts/inspect/`: tools for inspection of input and output files
  * `scripts/plotting/`: data visualization
  * `scripts/rabbit/`: fit input data preparation
  * `scripts/recoil/`: studies around the hadronic recoil calibration
  * `scripts/studies/`: other studies
  * `scripts/tests/`: statistical tests
* `wremnants/`: Here are the main analysis classes and functions defined that get executed by the scripts
  * `wremnants/postprocessing/`: everything related to analysing the histograms such as tools for plotting and fit input data preparation
  * `wremnants/production/`: everything related to histogram production using RDF
  * `wremnants/templates/`: small analysis specific templates
  * `wremnants/utilities/`: things that are commonly and more widely used across the framework and not restricted to histogram production or postprocessing. Such as input/output tool functionality, common definitions, parsing options, etc.

A typical analysis is performed in a few steps:
1. **Histogram production**: The processing of columnar data (such as collision events in NanoAOD) is performed in `scripts/histmakers/`. A minimal skeleton can be found in `scripts/histmakers/histmaker_template.py`. Datasets to process are defined in `wremnants/production/datasets/` and new files for new data taking periods or data streams may be added.
2. **Postprocessing / plotting**: A ready-to-use script can be found in `scripts/plotting/makeDataMCStackPlot.py` to plot histograms produced by a histmaker. However it may be easier for a new user to write a new, custom plotting script.
3. **Fit input data preparation**: The central analyses use `scripts/rabbit/setupRabbit.py` to prepare the input file needed for [rabbit](https://github.com/WMass/rabbit). A new analysis may write a custom, more specific and simplified script. Some explanation of how to interface [rabbit](https://github.com/WMass/rabbit) is given in that framework and corresponding documentation.
4. **Fitting**: Statistical data analysis performed in [rabbit](https://github.com/WMass/rabbit). See the [rabbit](https://github.com/WMass/rabbit) documentation for more details.

### Contribute to the code

**Guidelines**
 * When making a new PR, it should target only one subject at a time. This makes it more easy to validate and the integration faster. PR on top of other PR are ok when it is noted in the description, e.g. this PR is on top of PR XYZ.
 * Follow a modular approach and avoid cross dependencies between functions and classes.
 * Don't pass "args" across functions and in particular the use of very specific args arguments. This makes it difficult to re-use existing functions across different scripts. 
 * Avoid using "magic strings" that have the purpose of activating a specific logic.
 * Use camel case practice for command line arguments and avoid the "dest" keyword.
 * Use snake case practice for function names.
 * Class names should start with capital letters.


## Run the existing code
The following is a description of the existing analysis workflows. New analyses should ideally follow a similar logic and use the same underlying functions, command line options etc. but it may be easier and cleaner to write new custom scripts.

**NOTE**:
 * Each script has tons of options, to customize a gazillion of things. Some defined on the top of each files and others, that are more commonly used across different files are defined in `wremnants/utilities/parsing.py`. It's simpler to learn them by asking an expert rather that having an incomplete summary here (developments happen faster than documentation anyway).

### Histogram production
    
Make histograms for WMass (similar for other scripts such as `mz_wlike_with_mu_eta_pt.py`, `mz_dilepton.py`, and others).
```bash
python WRemnants/scripts/histmakers/mw_with_mu_eta_pt.py -o outputFolder/
```

### Fit preparation production

Make the inputs for the fit.
```bash
python WRemnants/scripts/rabbit/setupRabbit.py -i outputFolder/mw_with_mu_eta_pt_scetlib_dyturboCorr.hdf5 -o outputFolder/
```
The input file is the output of the previous step.
The default path specified with `-o` is the local folder. A subfolder with name identifying the specific analysis (e.g. `WMass_pt_eta/`) is automatically created inside it. Some options may add tags to the folder name: for example, using `--doStatOnly` will call the folder `WMass_pt_eta_statOnly/`.

### Making plots

There are many scripts to do every kind of plotting, and different people may have their own ones. We'll try to put a minimal list with examples here ASAP.

Plot Wmass histograms from hdf5 file (from Wmass histmaker) in the 4 iso-MT regions (can choose only some). It also makes some plots for fakes depending on the chosen region. It is also possible to select some specific processes to put in the plots.
```
python scripts/analysisTools/tests/testShapesIsoMtRegions.py mw_with_mu_eta_pt_scetlib_dyturboCorr.hdf5 outputFolder/ [--isoMtRegion 0 1 2 3]
```
    
Plot prefit shapes (requires root file from setupRabbit.py as input)
```
python scripts/analysisTools/w_mass_13TeV/plotPrefitTemplatesWRemnants.py WMassrabbitInput.root outputFolder/ [-l 16.8] [--pseudodata <pseudodataHistName>] [--wlike]
```

Make study of fakes for mW analysis, checking mT dependence, with or without dphi cut (see example inside the script for more options). Even if the histmaker was run with the dphi cut, the script uses a dedicated histograms `mTStudyForFakes` created before that cut, and with dphi in one axis.
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

### Theory agnostic analysis

Make histograms (only nominal and mass variations for now, systematics are being developed)
```
/usr/bin/time -v python scripts/histmakers/mw_with_mu_eta_pt.py -o outputFolder/ --theoryAgnostic --noAuxiliaryHistograms
```

Prepare inputs for the fit (stat-only for now)
```
/usr/bin/time -v python scripts/rabbit/setupRabbit.py -i outputFolder/mw_with_mu_eta_pt_scetlib_dyturboCorr.hdf5  -o outputFolder/  --absolutePathInCard --theoryAgnostic
```
To remove the backgrounds and run signal only one can add `--excludeProcGroups Top Diboson Fake Zmumu DYlowMass Ztautau Wtaunu BkgWmunu`

Run the fit (for charge combination)
```
python WRemnants/scripts/rabbit/fitManager.py -i outputFolder/WMass_pt_eta_statOnly/ --skip-fit-data --theoryAgnostic --comb
```
### Theory agnostic analysis with POIs as NOIs

Make histograms (this has all systematics too unlike the standard theory agnostic setup)
```
/usr/bin/time -v python scripts/histmakers/mw_with_mu_eta_pt.py -o outputFolder/ --theoryAgnostic --poiAsNoi
```

Prepare inputs for the fit
```
/usr/bin/time -v python scripts/rabbit/setupRabbit.py -i outputFolder/mw_with_mu_eta_pt_scetlib_dyturboCorr.hdf5  -o outputFolder/ --absolutePathInCard --theoryAgnostic --poiAsNoi --priorNormXsec 0.5
```
To remove the backgrounds and run signal only one can add `--filterProcGroups Wmunu`

Run the fit (for charge combination). Note that it is the same command as the traditional analysis, without `--theoryAgnostic`
```
python WRemnants/scripts/rabbit/fitManager.py -i outputFolder/WMass_pt_eta/ --skip-fit-data --comb
```

### Tools for scale factors

Make W MC efficiencies for trigger and isolation (needed for anti-iso and anti-trigger SF)
```
/usr/bin/time -v python scripts/histmakers/mw_with_mu_eta_pt.py -o outputFolder/ --makeMCefficiency --onlyMainHistograms --noAuxiliaryHistograms --noScaleFactors --muonCorrMC none -p WmunuMCeffi_noSF_muonCorrMCnone --filterProcs Wmunu --dataPath root://eoscms.cern.ch//store/cmst3/group/wmass/w-mass-13TeV/NanoAOD/ -v 4 --maxFiles -1
    
python scripts/analysisTools/w_mass_13TeV/makeWMCefficiency3D.py /path/to/file.hdf5 /path/for/plots/makeWMCefficiency3D/ --rebinUt 2
```

Then, run 2D smoothing (has to manually edit the default input files inside for now, see other options inside too). Option `--extended` was used to select SF computed in a larger ut range, but now this might become the default (to be updated)
```
python scripts/analysisTools/w_mass_13TeV/run2Dsmoothing.py /path/for/plots/test2Dsmoothing/
```
