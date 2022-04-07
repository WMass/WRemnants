# Overview

This directory is dedicated to scripts for setting up the inputs to combinetf. Inputs are build from the output of the [anaysis code](histmakers). The primary operations performed on the ND histograms produced at the analysis stage are:

* Project histograms to produce 1D or 2D outputs (for mW/W-like, pt vs. eta)
* Slice histograms to produce systematic hists. Often this is done by slicing each bin of an ND histogram where a systematic variation is indexed on one axis
* Sum histograms of combined processes (e.g., combining small backgrounds to one process)
* Scale histograms (by xsec, lumi, sumweights)

The common tools to perform these tasks are in the [CardTool class](../wremnants/CardTool.py).

Currently the W mass/W-like and the low pileup use different input formats. This is handled in the [datagroups](../wremnants/datasets/datagroups.py) [classes](../wremnants/datasets/datagroupsLowPU.py) so that the input read in the combine scripts is transparent.

# Running

Each independent fit has a separate driver script in this directory. The driver scripts schedule the relevant systematics and set the input format to the CardTool class.

Different driver scripts may have special configuration arguments, but all require an input file and output folder. For the W mass, the simplest running command is:

```bash
python3 ./scripts/combine/setupCombineWmass.py -i mw_with_mu_eta_pt.pkl.lz4 -o outputFolder
```

Additional options are available to configure some systematics. The full list can be obtained with ```./scripts/combine/setupCombineWmass.py --help```
