# Hayden notes

import pdb; pdb.set_trace()

## Setup

cd /work/submit/hayden17

APPTAINER_BIND="/tmp,/home/submit,/work/submit,/ceph/submit,/scratch/submit,/cvmfs,/etc/grid-security,/run" \
singularity run /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/bendavid/cmswmassdocker/wmassdevrolling:latest

source WRemnants/setup.sh

Singularity> cd /work/submit/hayden17/WRemnants

Singularity> python scripts/histmakers/histmaker_test.py --dataPath /scratch/submit/cms/wmass/NanoAOD/LowPU/2017G/ --era 2017G
Singularity>    python scripts/histmakers/histmaker_test.py      --dataPath /scratch/submit/cms/wmass/NanoAOD/LowPU/2017G/      --era 2017G      --filterProcs Zmumu2017G --theoryCorr scetlib_dyturboN3p0LL_LatticeNP_pdfas  -v 4

## Making Plots

IN=histmaker_test_scetlib_dyturboCorr.hdf5
OUT=~/public_html/
TAG=jan{date}
PROCS="Data Zmumu" - or Ztautau/Other

IN=histmaker_test_scetlib_dyturboN3p0LL_LatticeNP_pdfasCorr.hdf5
OUT=~/public_html/
TAG=jan16
PROCS="Data Zmumu"

Singularity> python scripts/plotting/makeDataMCStackPlot.py $IN   -o $OUT -f $TAG   -n nominal   --hists ptll   --rrange 0.995 1.005   --procFilters Zmumu --noData --flow none   variation --varName scetlib_dyturboN3p0LL_LatticeNP_pdfasCorr  --selectAxis vars --selectEntries pdfCT18ZNNLO_as_0116  pdfCT18ZNNLO_as_0120
  
python scripts/plotting/makeDataMCStackPlot.py $IN \
  -o $OUT -f $TAG \
  -n nominal \
  --hists ptll \
  --rrange 0.995 1.005 \
  --postfix w \
  --procFilters Other --noData --flow none \
  variation --varName prefiring --varLabel "up" "down"\
  --selectAxis prefire_variation --selectEntries 0 1
  
python scripts/plotting/makeDataMCStackPlot.py $IN \
  -o $OUT -f $TAG \
  -n nominal \
  --hists absYll \
  --rrange 0.995 1.005 \
  --postfix w \
  --procFilters Other --noData --flow none \
  variation --varName prefiring --varLabel "up" "down"\
  --selectAxis prefire_variation --selectEntries 0 1

python scripts/plotting/makeDataMCStackPlot.py $IN \
  -o $OUT -f $TAG \
  -n muleadeta \
  --hists mu_eta \
  --rrange 0.995 1.005 \
  --postfix wleading \
  --procFilters Other --noData --flow none \
  variation --varName prefiring --varLabel "up" "down"\
  --selectAxis prefire_variation --selectEntries 0 1
  
python scripts/plotting/makeDataMCStackPlot.py $IN \
  -o $OUT -f $TAG \
  -n mutraileta \
  --hists mu_eta \
  --rrange 0.995 1.005 \
  --postfix wtrailing \
  --procFilters Other --noData --flow none \
  variation --varName prefiring --varLabel "up" "down"\
  --selectAxis prefire_variation --selectEntries 0 1

for q in 0 1 2 3 4 5 6 7; do
  python scripts/plotting/makeDataMCStackPlot.py $IN -o $OUT -f $TAG \
    --procFilters $PROCS \
    --baseName ptll_vs_absYll_csQ${q} \
    --hists ptll-absYll \
    --postfix Q${q}
done

python scripts/plotting/makeDataMCStackPlot.py $IN -o $OUT -f $TAG --procFilters $PROCS --baseName nLepton --hists nLepton
python scripts/plotting/makeDataMCStackPlot.py $IN -o $OUT -f $TAG --procFilters $PROCS --baseName mll --hists mll
python scripts/plotting/makeDataMCStackPlot.py $IN -o $OUT -f $TAG --procFilters $PROCS --baseName ptll --hists ptll
python scripts/plotting/makeDataMCStackPlot.py $IN -o $OUT -f $TAG --procFilters $PROCS --baseName yll --hists yll
python scripts/plotting/makeDataMCStackPlot.py $IN -o $OUT -f $TAG --procFilters $PROCS --baseName phill --hists phill
python scripts/plotting/makeDataMCStackPlot.py $IN -o $OUT -f $TAG --procFilters $PROCS --baseName cosThetaStarll --hists cosThetaStarll
python scripts/plotting/makeDataMCStackPlot.py $IN -o $OUT -f $TAG --procFilters $PROCS --baseName phiStarll --hists phiStarll

python scripts/plotting/makeDataMCStackPlot.py $IN -o $OUT -f $TAG --procFilters $PROCS --baseName ptll_vs_yll --hists ptll-yll

python scripts/plotting/makeDataMCStackPlot.py $IN -o $OUT -f $TAG --procFilters $PROCS --baseName muleadpt --hists mu_pt --postfix leading
python scripts/plotting/makeDataMCStackPlot.py $IN -o $OUT -f $TAG --procFilters $PROCS --baseName mutrailpt --hists mu_pt --postfix trailing

python scripts/plotting/makeDataMCStackPlot.py $IN -o $OUT -f $TAG --procFilters $PROCS --baseName muleadeta --hists mu_eta --postfix leading
python scripts/plotting/makeDataMCStackPlot.py $IN -o $OUT -f $TAG --procFilters $PROCS --baseName mutraileta --hists mu_eta --postfix trailing

python scripts/plotting/makeDataMCStackPlot.py $IN -o $OUT -f $TAG --procFilters $PROCS --baseName mupospt --hists mu_pt --postfix positive
python scripts/plotting/makeDataMCStackPlot.py $IN -o $OUT -f $TAG --procFilters $PROCS --baseName munegpt --hists mu_pt --postfix negative

python scripts/plotting/makeDataMCStackPlot.py $IN -o $OUT -f $TAG --procFilters $PROCS --baseName muposeta --hists mu_eta --postfix positive
python scripts/plotting/makeDataMCStackPlot.py $IN -o $OUT -f $TAG --procFilters $PROCS --baseName munegeta --hists mu_eta --postfix negative

https://submit.mit.edu/~hayden17/jan{date}/



































# Description of Histmaker Template

Imports / setup
* import os Used for file/path utilities (here: getting the script name, building output filename).
* import ROOT Makes ROOT available; later you use ROOT C++ helpers in strings passed to RDataFrame (ROOT::VecOps::Nonzero, ROOT::Math::PtEtaPhiMVector).
* from utilities import parsing Your project’s helper that builds a standard CLI argument parser.
* from wremnants.datasets.datagroups import Datagroups Provides dataset grouping utilities and the analysisLabel helper.
* from wums import logging Your project’s logging utility.

Analysis label + argument parsing
* analysis_label = Datagroups.analysisLabel(os.path.basename(__file__)) Takes the current filename (like myanalysis.py) and converts it to an “analysis label” string used to pick dataset configs / modes.
* parser, initargs = parsing.common_parser(analysis_label) Creates an argparse parser with standard options (era, dataPath, maxFiles, filters, verbosity, etc.). initargs is whatever extra metadata this helper returns.
* args = parser.parse_args() Reads command-line arguments into args.
* logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger) Sets up a logger using the script name and CLI verbosity/color options.

More imports used for hist + running
* import hist The hist package: axes + histogram objects.
* import narf Framework to build and run the RDF computation graph over datasets.
* from wremnants.datasets.dataset_tools import getDatasets Loads dataset definitions (files, metadata, data-vs-MC, etc.) based on CLI args.
* from wremnants.histmaker_tools import write_analysis_output Writes the produced histograms / sums into an output file (HDF5 here).

Load datasets

datasets = getDatasets(
    maxFiles=args.maxFiles,
    filt=args.filterProcs,
    excl=args.excludeProcs,
    base_path=args.dataPath,
    mode=analysis_label,
    era=args.era,
)
* Calls getDatasets(...) to construct the list of dataset objects to run over.
* maxFiles: cap files per dataset (useful for quick tests).
* filt / excl: include/exclude process patterns.
* base_path: where the data files live.
* mode: selects a dataset “mode” (here based on your script name).
* era: selects dataset era/year.

Define histogram axes
* axis_nLepton = hist.axis.Integer(0, 5, name="nLepton", underflow=False) Integer axis with bins for 0,1,2,3,4 (no underflow bin). Used to histogram total lepton count.
* axis_mll = hist.axis.Regular(60, 76, 106, name="mll") 60 uniform bins from 76 to 106 for dimuon invariant mass.
* axis_ptll = hist.axis.Regular(60, 0, 120, name="ptll") 60 uniform bins from 0 to 120 for dimuon pT.
* axis_yll = hist.axis.Regular(48, -2.4, 2.4, name="yll") 48 uniform bins from -2.4 to 2.4 for dimuon rapidity.
* axis_mu_pt = hist.axis.Regular(60, 25, 150, name="mu_pt") 60 bins from 25 to 150 for single-muon pT.
* axis_mu_eta = hist.axis.Regular(48, -2.4, 2.4, name="mu_eta") 48 bins from -2.4 to 2.4 for single-muon eta.

The per-dataset computation graph
Function header
* def build_graph(df, dataset): This is called once per dataset. df is an RDataFrame-like object you add Define/Filter/HistoBoost nodes to. dataset holds metadata (name, is_data, etc.).
* logger.info(f"build graph for dataset: {dataset.name}") Logs which dataset you’re building the graph for.
* results = [] A list to collect histogram handles to return.

Event weights

if dataset.is_data:
    df = df.DefinePerSample("weight", "1.0")
else:
    df = df.Define("weight", "std::copysign(1.0, genWeight)")
* For data: define a per-sample constant weight of 1.
* For MC: define weight as the sign of genWeight (+1 or -1). This keeps negative-weight events (common in NLO MC) but ignores magnitude and ignores other corrections.

Sum of weights (bookkeeping)
* weightsum = df.SumAndCount("weight") Creates an action that will compute:
    * sum of weight
    * number of events contributing Useful for normalization / sanity checks.

Trigger selection
* df = df.Filter("HLT_HIMu17") Keeps only events that pass that trigger bit.

Simple lepton count
* df = df.Define("nLepton", "nElectron + nMuon") Adds a new column: total number of reconstructed electrons + muons in the event (as stored in the nano-like format).

Z→μμ candidate selection (all new physics logic)
“Good muon” mask

df = df.Define(
    "goodMu",
    "Muon_pt > 25 && abs(Muon_eta) < 2.4 && Muon_tightId && Muon_pfRelIso04_all < 0.15"
)
* Creates a boolean vector goodMu with one entry per muon in the event.
* A muon is “good” if:
    * Muon_pt > 25
    * |Muon_eta| < 2.4
    * Muon_tightId is true
    * relative isolation < 0.15
Indices of good muons
* df = df.Define("goodMu_idx", "ROOT::VecOps::Nonzero(goodMu)") Converts that boolean mask into a vector of indices where it’s true (e.g. [0, 2]).
Exactly two good muons
* df = df.Filter("goodMu_idx.size() == 2", "Exactly two good muons") Event-level requirement: there must be exactly two passing muons.

Opposite-sign requirement
* df = df.Define("i0", "int(goodMu_idx[0])").Define("i1", "int(goodMu_idx[1])") Pulls the two indices out into scalars i0 and i1 so you can index arrays like Muon_pt[i0].
* df = df.Filter("Muon_charge[i0] * Muon_charge[i1] < 0", "Opposite-sign muons") Requires charges to multiply to negative → one + and one −.

Build dimuon 4-vector and kinematics
* MU_MASS = 0.105658 Muon mass in GeV used for 4-vector construction.

df = df.Define("mu0_p4", f"ROOT::Math::PtEtaPhiMVector(Muon_pt[i0], Muon_eta[i0], Muon_phi[i0], {MU_MASS})") \
    .Define("mu1_p4", f"ROOT::Math::PtEtaPhiMVector(Muon_pt[i1], Muon_eta[i1], Muon_phi[i1], {MU_MASS})") \
    .Define("dimu_p4", "mu0_p4 + mu1_p4") \
    .Define("mll", "dimu_p4.M()") \
    .Define("ptll", "dimu_p4.Pt()") \
    .Define("yll", "dimu_p4.Rapidity()")
* mu0_p4, mu1_p4: build ROOT 4-vectors from each muon’s (pT, eta, phi, mass).
* dimu_p4: vector sum → the Z candidate 4-vector.
* mll: invariant mass of the pair.
* ptll: transverse momentum of the pair.
* yll: rapidity of the pair.
Z mass window
* df = df.Filter("mll > 76 && mll < 106", "Z mass window") Keeps only events with dimuon mass consistent with a Z candidate.

Histogramming
Z/dimuon histograms + nLepton

hist_nLepton = df.HistoBoost("nLepton", [axis_nLepton], ["nLepton"])
hist_mll  = df.HistoBoost("mll",  [axis_mll],  ["mll"])
hist_ptll = df.HistoBoost("ptll", [axis_ptll], ["ptll"])
hist_yll  = df.HistoBoost("yll",  [axis_yll],  ["yll"])
* Each HistoBoost(name, [axis], [column]) creates a histogram action:
    * It will fill that histogram using the column values from the currently filtered dataframe.
* Note: in many RDF setups, the weight column is automatically applied by HistoBoost; in others you’d need to pass it explicitly. In your setup it’s presumably handled by the framework.
Single-muon scalars and histograms

df = df.Define("mu0_pt", "Muon_pt[i0]").Define("mu1_pt", "Muon_pt[i1]") \
    .Define("mu0_eta", "Muon_eta[i0]").Define("mu1_eta", "Muon_eta[i1]")
* Extracts the two muons’ pT and eta into scalar columns so histogramming is straightforward.

hist_mu0_pt  = df.HistoBoost("mu0_pt",  [axis_mu_pt],  ["mu0_pt"])
hist_mu1_pt  = df.HistoBoost("mu1_pt",  [axis_mu_pt],  ["mu1_pt"])
hist_mu0_eta = df.HistoBoost("mu0_eta", [axis_mu_eta], ["mu0_eta"])
hist_mu1_eta = df.HistoBoost("mu1_eta", [axis_mu_eta], ["mu1_eta"])
* Fills separate histograms for muon 0 and muon 1 (as ordered in goodMu_idx).
Collect results + return
* results += [hist_mll, hist_ptll, hist_yll, hist_mu0_pt, hist_mu1_pt, hist_mu0_eta, hist_mu1_eta, hist_nLepton] Adds all histogram handles to the list to return.
* return results, weightsum Returns:
    * the histograms to be executed
    * the weight sum/count action

Run graph over all datasets
* resultdict = narf.build_and_run(datasets, build_graph) For each dataset:
    * constructs the RDF graph via build_graph
    * executes it
    * stores outputs (histograms + weightsum) in resultdict keyed by dataset/process.

Write output
* fout = f"{os.path.basename(__file__).replace('py', 'hdf5')}" Output filename = your script name but .hdf5 extension.
* write_analysis_output(resultdict, fout, args) Writes the results (hists + metadata) to that HDF5 file, using CLI args for bookkeeping.

If you want the one most important high-level summary: this script selects triggered events with exactly two tight isolated muons of opposite charge in the Z mass window, then saves histograms of the Z candidate and the two muons.













































































































































































































































# WRemnants (Old README)

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
