# Overview

This directory is dedicated to scripts for setting up the inputs to combinetf2. Inputs are build from the output of the [anaysis code](histmakers). The primary operations performed on the python boost histograms produced at the analysis stage are:

* Project histograms to produce 1D or 2D outputs (for mW/W-like, pt vs. eta)
* Slice histograms to produce systematic hists. Often this is done by slicing each bin of an ND histogram where a systematic variation is indexed on one axis
* Sum histograms of combined processes (e.g., combining small backgrounds to one process)
* Scale histograms (by xsec, lumi, sumweights)

The common tools to perform these tasks are in the [Datagroups class](../wremnants/datastes/datagroups.py).

# Running

Each independent fit has a separate driver script in this directory. The driver scripts schedule the relevant systematics and set the input format to the CardTool class.

The driver script may have special configuration arguments. For the W mass, the simplest running command is:

```bash
python ./scripts/combine/setupCombine.py -i mw_with_mu_eta_pt.pkl.hdf5 -o outputFolder
```

Additional options are available to configure some systematics.
