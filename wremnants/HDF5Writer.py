from wremnants.combine_helpers import getTheoryFitData, setSimultaneousABCD, projectABCD
from utilities import common, logging, output_tools
import time
import numpy as np
import hist
import h5py
from utilities.h5pyutils import writeFlatInChunks, writeSparse
import math
import pandas as pd
import os
import narf
import pdb

logger = logging.child_logger(__name__)

class HDF5Writer(object):
    # keeps multiple card tools and writes them out in a single file to fit (appending the histograms)
    def __init__(self, card_name="card"):
        self.cardName = card_name
        self.cardTools = []
        # settings for writing out hdf5 files
        self.dtype="float64"
        self.chunkSize=4*1024**2
        self.logkepsilon=math.log(1e-3) #numerical cutoff in case of zeros in systematic variations

        self.theoryFit = False
        self.theoryFitData = None
        self.theoryFitDataCov = None
        self.theoryFitMCStat = True # Whether or not to include the MC stat uncertainty in the thoery fit (in the covariance matrix)

        self.dict_noigroups = {}
        self.dict_systgroups = {}

        self.systsstandard = set()
        self.systsnoconstraint = set()
        self.systsnoprofile = set()

        self.channels = {}
        self.masked_channels = []

        self.clipSystVariations=False
        self.clipSystVariationsSignal=False
        if self.clipSystVariations>0.:
            self.clip = np.abs(np.log(clipSystVariations))
        if self.clipSystVariationsSignal>0.:
            self.clipSig = np.abs(np.log(clipSystVariationsSignal))


    def set_fitresult(self, fitresult):
        # for theory fit, currently not supported for sumPOI groups
        self.theoryFit = True
        base_processes = ["W" if c.datagroups.wmass else "Z" for c in self.get_channels().values()]
        data, self.theoryFitDataCov = getTheoryFitData(fitresult, base_processes=base_processes)
        # theoryfit data for each channel
        self.theoryFitData = {c: d for c, d in zip(self.get_channels().keys(), data)}

    def add_channel(self, cardTool, name=None):
        if name is None:
            name = f"ch{len(self.channels)+1}"
        self.channels[name] = cardTool

    def get_channels(self):
        # return channels such that masked channels are last
        sorted_channels = dict(sorted(self.channels.items(), key=lambda x: x[1].xnorm))
        return sorted_channels

    def get_signals(self):
        signals = set()
        for chan, chanInfo in self.get_channels().items():
            signals.update(chanInfo.unconstrainedProcesses[:])
        return list(common.natural_sort(signals))

    def get_backgrounds(self):
        bkgs = set()
        for chan, chanInfo in self.get_channels().items():
            bkgs.update([p for p in chanInfo.predictedProcesses() if p not in chanInfo.unconstrainedProcesses])      
        return list(common.natural_sort(bkgs))

    def write(self, 
        args,
        forceNonzero=True, 
        check_systs=False, 
        allowNegativeExpectation=False,
    ):
        signals = self.get_signals() 
        bkgs = self.get_backgrounds() 

        dict_data_obs = {}
        dict_data_obs_cov = {}
        dict_sumw2 = {c : {} for c in self.get_channels()}
        dict_norm = {c : {} for c in self.get_channels()}
        dict_logkavg = {c : {} for c in self.get_channels()}
        dict_logkhalfdiff = {c : {} for c in self.get_channels()}

        #keep track of bins per channel
        ibins = []
        nbins = 0
        
        for chan, chanInfo in self.get_channels().items():
            masked = chanInfo.xnorm and not self.theoryFit
            logger.info(f"Now in channel {chan} masked={masked}")

            dg = chanInfo.datagroups
            axes = chanInfo.project[:]

            if chanInfo.xnorm:
                dg.globalAction = None # reset global action in case of rebinning or such
                dg.select_xnorm_groups() # only keep processes where xnorm is defined
                if dg.fakeName in dg.groups.keys():
                    dg.deleteGroup(dg.fakeName)

            # load data and nominal ans syst histograms
            dg.loadHistsForDatagroups(
                baseName=chanInfo.nominalName, syst=chanInfo.nominalName,
                procsToRead=dg.groups.keys(),
                label=chanInfo.nominalName, 
                scaleToNewLumi=chanInfo.lumiScale, 
                forceNonzero=forceNonzero)

            if not masked:                
                if self.theoryFit:
                    if self.theoryFitData is None or self.theoryFitDataCov is None:
                        raise RuntimeError("No data or covariance found to perform theory fit")
                    data_obs = self.theoryFitData[chan]
                else:
                    data_obs_hist = dg.groups[dg.dataName].hists[chanInfo.nominalName]

                    if chanInfo.ABCD:
                        setSimultaneousABCD(chanInfo)
                        if dg.fakeName not in bkgs:
                            bkgs.append(dg.fakeName)

                        if chanInfo.nameMT not in axes:
                            axes.append(chanInfo.nameMT)
                        if common.passIsoName not in axes:
                            axes.append(common.passIsoName)

                    if chanInfo.ABCD and set(chanInfo.fakerateAxes) != set(chanInfo.project):
                        data_obs = projectABCD(chanInfo, data_obs_hist)
                    else:
                        if data_obs_hist.axes.name != axes:
                            data_obs_hist = data_obs_hist.project(*axes)

                        data_obs = data_obs_hist.values(flow=False).flatten().astype(self.dtype)

                dict_data_obs[chan] = data_obs
                nbinschan = len(data_obs)
                nbins += nbinschan

            else:
                self.masked_channels.append(chan)
                axes = ["count"]
                nbinschan = 1

            ibins.append(nbinschan)

            procs_chan = chanInfo.predictedProcesses()
            for proc in procs_chan:
                logger.debug(f"Now  in channel {chan} at process {proc}")
                
                # nominal histograms of prediction
                norm_proc_hist = dg.groups[proc].hists[chanInfo.nominalName]

                if not masked:                
                    # check if variances are available
                    if norm_proc_hist.storage_type != hist.storage.Weight:
                        raise RuntimeError(f"Sumw2 not filled for {proc} but needed for binByBin uncertainties")

                    if chanInfo.ABCD and set(chanInfo.fakerateAxes) != set(chanInfo.project):
                        norm_proc, sumw2_proc = projectABCD(chanInfo, norm_proc_hist, return_variances=True)
                    else:
                        if norm_proc_hist.axes != axes:
                            norm_proc_hist = norm_proc_hist.project(*axes)

                        norm_proc = norm_proc_hist.values(flow=False).flatten().astype(self.dtype)
                        sumw2_proc = norm_proc_hist.variances(flow=False).flatten().astype(self.dtype)
                else:
                    norm_proc = norm_proc_hist.values(flow=False).flatten().astype(self.dtype)
                    if norm_proc.shape[0] != nbinschan:
                        raise Exception(f"Mismatch between number of bins in channel {chan} for expected ({nbinschan}) and template ({norm_proc.shape[0]})")

                # free memory
                dg.groups[proc].hists[chanInfo.nominalName] = None

                if not allowNegativeExpectation:
                    norm_proc = np.maximum(norm_proc, 0.)

                if not masked:                
                    dict_sumw2[chan][proc] = sumw2_proc

                dict_norm[chan][proc] = norm_proc

            dict_logkavg[chan] = {p : {} for p in procs_chan}
            dict_logkhalfdiff[chan] = {p : {} for p in procs_chan}

            # lnN systematics
            for name, syst in chanInfo.lnNSystematics.items():
                logger.info(f"Now in channel {chan} at lnN systematic {name}")

                if chanInfo.isExcludedNuisance(name): 
                    continue
                procs_syst = [p for p in syst["processes"] if p in procs_chan]
                if len(procs_syst) == 0:
                    continue

                ksyst = syst["size"]
                if type(ksyst) is list:
                    ksystup = ksyst[1]
                    ksystdown = ksyst[0]
                    if ksystup == 0. and ksystdown==0.:
                        continue
                    if ksystup == 0.:
                        ksystup = 1.
                    if ksystdown == 0.:
                        ksystdown = 1.
                    logkup_proc = math.log(ksystup)*np.ones([nbinschan],dtype=self.dtype)
                    logkdown_proc = -math.log(ksystdown)*np.ones([nbinschan],dtype=self.dtype)
                    logkavg_proc = 0.5*(logkup_proc + logkdown_proc)
                    logkhalfdiff_proc = 0.5*(logkup_proc - logkdown_proc)
                    logkup_proc = None
                    logkdown_proc = None
                else:
                    if ksyst == 0.:
                        continue
                    logkavg_proc = math.log(ksyst)*np.ones([nbinschan],dtype=self.dtype)
                    logkhalfdiff_proc = np.zeros([nbinschan],dtype=self.dtype)

                for proc in procs_syst:
                    logger.debug(f"Now at proc {proc}!")

                    # save for later
                    norm_proc = dict_norm[chan][proc]
                    #ensure that systematic tensor is sparse where normalization matrix is sparse
                    dict_logkavg[chan][proc][name] = np.where(np.equal(norm_proc,0.), 0., logkavg_proc)
                    dict_logkhalfdiff[chan][proc][name] = np.where(np.equal(norm_proc,0.), 0., logkhalfdiff_proc)

                self.book_systematic(syst, name)

            # shape systematics
            for systKey, syst in chanInfo.systematics.items():
                logger.info(f"Now in channel {chan} at shape systematic group {systKey}")

                if chanInfo.isExcludedNuisance(systKey): 
                    continue

                # some channels (e.g. xnorm) don't have all processes affected by the systematic
                procs_syst = [p for p in syst["processes"] if p in procs_chan]
                if len(procs_syst) == 0:
                    continue

                systName = systKey if not syst["name"] else syst["name"]

                # Needed to avoid always reading the variation for the fakes, even for procs not specified
                forceToNominal=[x for x in dg.getProcNames() if x not in 
                    dg.getProcNames([p for g in procs_syst for p in chanInfo.expandProcesses(g) if p != dg.fakeName])]

                dg.loadHistsForDatagroups(
                    chanInfo.nominalName, systName, label="syst",
                    procsToRead=procs_syst, 
                    forceNonzero=forceNonzero and systName != "qcdScaleByHelicity",
                    preOpMap=syst["actionMap"], preOpArgs=syst["actionArgs"],
                    # Needed to avoid always reading the variation for the fakes, even for procs not specified
                    forceToNominal=forceToNominal,
                    scaleToNewLumi=chanInfo.lumiScale,
                    nominalIfMissing=not chanInfo.xnorm # for masked channels not all systematics exist (we can skip loading nominal since Fake does not exist)
                )

                for proc in procs_syst:
                    logger.debug(f"Now at proc {proc}!")

                    hvar = dg.groups[proc].hists["syst"]

                    if syst["doActionBeforeMirror"] and syst["action"]:
                        logger.debug(f"Do action before mirror")
                        hvar = syst["action"](hvar, **syst["actionArgs"])
                    if syst["decorrByBin"]:
                        raise NotImplementedError("By bin decorrelation is not supported for writing output in hdf5")

                    var_map = chanInfo.systHists(hvar, systKey)
                    var_names = [x[:-2] if "Up" in x[-2:] else (x[:-4] if "Down" in x[-4:] else x) 
                        for x in filter(lambda x: x != "", var_map.keys())]
                    # Deduplicate while keeping order
                    var_names = list(dict.fromkeys(var_names))
                    norm_proc = dict_norm[chan][proc]

                    for var_name in var_names:
                        kfac = syst["scale"]

                        def get_logk(histname, var_type=""):
                            _hist = var_map[histname+var_type]

                            if not masked and chanInfo.ABCD and set(chanInfo.fakerateAxes) != set(chanInfo.project):
                                _syst = projectABCD(chanInfo, _hist)
                            else:
                                if _hist.axes != axes:
                                    _hist = _hist.project(*axes)
                                _syst = _hist.values(flow=False).flatten().astype(self.dtype)
                            
                            # check if there is a sign flip between systematic and nominal
                            _logk = kfac*np.log(_syst/norm_proc)
                            _logk_view = np.where(np.equal(np.sign(norm_proc*_syst),1), _logk, self.logkepsilon*np.ones_like(_logk))
                            _syst = None

                            if self.clipSystVariations>0.:
                                _logk = np.clip(_logk,-self.clip,self.clip)
                            if self.clipSystVariationsSignal>0. and proc in signals:
                                _logk = np.clip(_logk,-self.clipSig,self.clipSig)

                            return _logk_view

                        if syst["mirror"]:
                            logkavg_proc = get_logk(var_name)
                            logkhalfdiff_proc = np.zeros([nbinschan],dtype=self.dtype)
                        else:
                            logkup_proc = get_logk(var_name, "Up")
                            logkdown_proc = get_logk(var_name, "Down")

                            logkavg_proc = 0.5*(logkup_proc - logkdown_proc)
                            logkhalfdiff_proc = 0.5*(logkup_proc + logkdown_proc)

                            logkup_proc = None
                            logkdown_proc = None

                        #ensure that systematic tensor is sparse where normalization matrix is sparse
                        logkavg_proc = np.where(np.equal(norm_proc,0.), 0., logkavg_proc)
                        logkhalfdiff_proc = np.where(np.equal(norm_proc,0.), 0., logkhalfdiff_proc)

                        # save for later
                        dict_logkavg[chan][proc][var_name] = logkavg_proc
                        dict_logkhalfdiff[chan][proc][var_name] = logkhalfdiff_proc

                        self.book_systematic(syst, var_name)

                    # free memory
                    for var in var_map.keys():
                        var_map[var] = None
                    dg.groups[proc].hists["syst"] = None

        procs = signals + bkgs
        nproc = len(procs)

        logger.info(f"Write out nominal arrays")
        sumw = np.zeros([nbins], self.dtype)
        sumw2 = np.zeros([nbins], self.dtype)
        data_obs = np.zeros([nbins], self.dtype)
        ibin = 0
        for nbinschan, (chan, chanInfo) in zip(ibins, self.get_channels().items()):
            masked = chanInfo.xnorm and not self.theoryFit
            if masked:
                continue
            data_obs[ibin:ibin+nbinschan] = dict_data_obs[chan]

            for iproc, proc in enumerate(procs):
                if proc not in dict_norm[chan]:
                    continue

                sumw[ibin:ibin+nbinschan] += dict_norm[chan][proc]
                sumw2[ibin:ibin+nbinschan] += dict_sumw2[chan][proc]
            
            ibin += nbinschan

        systs = self.get_systs()
        nsyst = len(systs)

        nbinsfull = sum(ibins)

        ibin = 0
        if args.sparse:
            logger.info(f"Write out sparse array")

            idxdtype = 'int32'
            maxsparseidx = max(nbinsfull*nproc,2*nsyst)
            if maxsparseidx > np.iinfo(idxdtype).max:
                logger.info("sparse array shapes are too large for index datatype, switching to int64")
                idxdtype = 'int64'

            norm_sparse_size = 0
            norm_sparse_indices = np.zeros([norm_sparse_size,2],idxdtype)
            norm_sparse_values = np.zeros([norm_sparse_size],self.dtype)

            logk_sparse_size = 0
            logk_sparse_normindices = np.zeros([logk_sparse_size,1],idxdtype)
            logk_sparse_systindices = np.zeros([logk_sparse_size,1],idxdtype)
            logk_sparse_values = np.zeros([logk_sparse_size],self.dtype)

            for nbinschan, chan in zip(ibins, self.get_channels()):
                dict_norm_chan = dict_norm[chan]
                dict_logkavg_chan = dict_logkavg[chan]
                dict_logkhalfdiff_chan = dict_logkhalfdiff[chan]

                for iproc, proc in enumerate(procs):
                    if proc not in dict_norm_chan:
                        continue
                    norm_proc = dict_norm_chan[proc]

                    norm_indices = np.transpose(np.nonzero(norm_proc))
                    norm_values = np.reshape(norm_proc[norm_indices],[-1])
                        
                    nvals = len(norm_values)
                    oldlength = norm_sparse_size
                    norm_sparse_size = oldlength + nvals
                    norm_sparse_indices.resize([norm_sparse_size,2])
                    norm_sparse_values.resize([norm_sparse_size])
                    
                    out_indices = np.array([[ibin,iproc]]) + np.pad(norm_indices,((0,0),(0,1)),'constant')
                    norm_indices = None
                    
                    norm_sparse_indices[oldlength:norm_sparse_size] = out_indices
                    out_indices = None
                    
                    norm_sparse_values[oldlength:norm_sparse_size] = norm_values
                    norm_values = None
                    
                    norm_idx_map = np.cumsum(np.not_equal(norm_proc, 0.)) - 1 + oldlength

                    dict_logkavg_proc = dict_logkavg_chan[proc]
                    dict_logkhalfdiff_proc = dict_logkhalfdiff_chan[proc]
                    for isyst, syst in enumerate(systs):
                        if syst not in dict_logkavg_proc.keys():
                            continue

                        logkavg_proc = dict_logkavg_proc[syst]
                        logkhalfdiff_proc = dict_logkhalfdiff_proc[syst]

                        logkavg_proc_indices = np.transpose(np.nonzero(logkavg_proc))
                        logkavg_proc_values = np.reshape(logkavg_proc[logkavg_proc_indices],[-1])
                        
                        nvals_proc = len(logkavg_proc_values)
                        oldlength = logk_sparse_size
                        logk_sparse_size = oldlength + nvals_proc
                        logk_sparse_normindices.resize([logk_sparse_size,1])
                        logk_sparse_systindices.resize([logk_sparse_size,1])
                        logk_sparse_values.resize([logk_sparse_size])

                        #first dimension of output indices are NOT in the dense [nbin,nproc] space, but rather refer to indices in the norm_sparse vectors
                        #second dimension is flattened in the [2,nsyst] space, where logkavg corresponds to [0,isyst] flattened to isyst
                        #two dimensions are kept in separate arrays for now to reduce the number of copies needed later
                        out_normindices = norm_idx_map[logkavg_proc_indices]
                        logkavg_proc_indices = None
                        
                        logk_sparse_normindices[oldlength:logk_sparse_size] = out_normindices
                        logk_sparse_systindices[oldlength:logk_sparse_size] = isyst
                        out_normindices = None
                        
                        logk_sparse_values[oldlength:logk_sparse_size] = logkavg_proc_values
                        logkavg_proc_values = None
                        
                        logkhalfdiff_proc_indices = np.transpose(np.nonzero(logkhalfdiff_proc))
                        logkhalfdiff_proc_values = np.reshape(logkhalfdiff_proc[logkhalfdiff_proc_indices],[-1])
                                
                        nvals_proc = len(logkhalfdiff_proc_values)
                        oldlength = logk_sparse_size
                        logk_sparse_size = oldlength + nvals_proc
                        logk_sparse_normindices.resize([logk_sparse_size,1])
                        logk_sparse_systindices.resize([logk_sparse_size,1])
                        logk_sparse_values.resize([logk_sparse_size])
                                
                        #out_indices = np.array([[ibin,iproc,isyst,1]]) + np.pad(logkhalfdiff_proc_indices,((0,0),(0,3)),'constant')
                        #first dimension of output indices are NOT in the dense [nbin,nproc] space, but rather refer to indices in the norm_sparse vectors
                        #second dimension is flattened in the [2,nsyst] space, where logkhalfdiff corresponds to [1,isyst] flattened to nsyst + isyst
                        #two dimensions are kept in separate arrays for now to reduce the number of copies needed later
                        out_normindices = norm_idx_map[logkhalfdiff_proc_indices]
                        logkhalfdiff_proc_indices = None
                        
                        logk_sparse_normindices[oldlength:logk_sparse_size] = out_normindices
                        logk_sparse_systindices[oldlength:logk_sparse_size] = nsyst + isyst
                        out_normindices = None
                        
                        logk_sparse_values[oldlength:logk_sparse_size] = logkhalfdiff_proc_values
                    logkhalfdiff_proc_values = None        
    
                    # free memory
                    logkavg_proc = None
                    logkhalfdiff_proc = None

                # free memory
                norm_proc = None
                norm_idx_map = None

                ibin += nbinschan
            
            logger.info(f"Resize and sort sparse arrays into canonical order")
            #resize sparse arrays to actual length
            norm_sparse_indices.resize([norm_sparse_size,2])
            norm_sparse_values.resize([norm_sparse_size])
            logk_sparse_normindices.resize([logk_sparse_size,1])
            logk_sparse_systindices.resize([logk_sparse_size,1])
            logk_sparse_values.resize([logk_sparse_size])
            
            #straightforward sorting of norm_sparse into canonical order
            norm_sparse_dense_shape = (nbinsfull, nproc)
            norm_sort_indices = np.argsort(np.ravel_multi_index(np.transpose(norm_sparse_indices),norm_sparse_dense_shape))
            norm_sparse_indices = norm_sparse_indices[norm_sort_indices]
            norm_sparse_values = norm_sparse_values[norm_sort_indices]
                
            #now permute the indices of the first dimension of logk_sparse corresponding to the resorting of norm_sparse
            
            #compute the inverse permutation from the sorting of norm_sparse
            #since the final indices are filled from here, need to ensure it has the correct data type
            logk_permute_indices = np.argsort(norm_sort_indices).astype(idxdtype)
            norm_sort_indices = None
            logk_sparse_normindices = logk_permute_indices[logk_sparse_normindices]
            logk_permute_indices = None
            logk_sparse_indices = np.concatenate([logk_sparse_normindices, logk_sparse_systindices],axis=-1)
            logk_normindices = None

            #now straightforward sorting of logk_sparse into canonical order
            logk_sparse_dense_shape = (norm_sparse_indices.shape[0], 2*nsyst)
            logk_sort_indices = np.argsort(np.ravel_multi_index(np.transpose(logk_sparse_indices),logk_sparse_dense_shape))
            logk_sparse_indices = logk_sparse_indices[logk_sort_indices]
            logk_sparse_values = logk_sparse_values[logk_sort_indices]
            logk_sort_indices = None

        else:
            logger.info(f"Write out dense array")
            #initialize with zeros, i.e. no variation
            norm = np.zeros([nbinsfull,nproc], self.dtype)
            logk = np.zeros([nbinsfull,nproc,2,nsyst], self.dtype)

            for nbinschan, chan in zip(ibins, self.get_channels()):
                dict_norm_chan = dict_norm[chan]
                dict_logkavg_chan = dict_logkavg[chan]
                dict_logkhalfdiff_chan = dict_logkhalfdiff[chan]

                for iproc, proc in enumerate(procs):
                    if proc not in dict_norm_chan:
                        continue
                    norm[ibin:ibin+nbinschan, iproc] = dict_norm_chan[proc]

                    dict_logkavg_proc = dict_logkavg_chan[proc]
                    dict_logkhalfdiff_proc = dict_logkhalfdiff_chan[proc]
                    for isyst, syst in enumerate(systs):
                        if syst not in dict_logkavg_proc.keys():
                            continue

                        logk[ibin:ibin+nbinschan,iproc,0,isyst] = dict_logkavg_proc[syst]
                        logk[ibin:ibin+nbinschan,iproc,1,isyst] = dict_logkhalfdiff_proc[syst]
                        
                ibin += nbinschan
            
        #compute poisson parameter for Barlow-Beeston bin-by-bin statistical uncertainties
        kstat = np.square(sumw)/sumw2
        #numerical protection to avoid poorly defined constraint
        kstat = np.where(np.equal(sumw,0.), 1., kstat)

        #write results to hdf5 file
        procSize = nproc*np.dtype(self.dtype).itemsize
        systSize = 2*nsyst*np.dtype(self.dtype).itemsize
        amax = np.max([procSize,systSize])
        if amax > self.chunkSize:
            logger.warning(f"Maximum chunk size in bytes was increased from {self.chunkSize} to {amax} to align with tensor sizes and allow more efficient reading/writing.")
            self.chunkSize = amax

        #create HDF5 file (chunk cache set to the chunk size since we can guarantee fully aligned writes
        outfilename = self.get_output_filename(sparse=args.sparse, outfolder=args.outfolder, postfix=args.postfix, doStatOnly=args.doStatOnly)
        logger.info(f"Write output file {outfilename}")
        f = h5py.File(outfilename, rdcc_nbytes=self.chunkSize, mode='w')

        # propagate meta info into result file
        meta = {"meta_info" : output_tools.metaInfoDict(args=args)}
        narf.ioutils.pickle_dump_h5py("meta", meta, f)

        systsnoprofile = self.get_systsnoprofile()
        systsnoconstraint = self.get_systsnoconstraint()

        noigroups, noigroupidxs = self.get_noigroups()
        systgroups, systgroupidxs = self.get_systgroups()
        sumgroups, sumgroupsegmentids, sumgroupidxs = self.get_sumgroups(procs)
        chargegroups, chargegroupidxs = self.get_chargegroups()
        polgroups, polgroupidxs = self.get_polgroups()
        helgroups, helgroupidxs = self.get_helgroups()
        chargemetagroups, chargemetagroupidxs = self.get_chargemetagroups()
        ratiometagroups, ratiometagroupidxs = self.get_ratiometagroups()
        helmetagroups, helmetagroupidxs = self.get_helmetagroups()
        reggroups, reggroupidxs = self.get_reggroups()
        poly1dreggroups, poly1dreggroupfirstorder, poly1dreggrouplastorder, poly1dreggroupnames, poly1dreggroupbincenters = self.get_poly1dreggroups()
        poly2dreggroups, poly2dreggroupfirstorder, poly2dreggrouplastorder, poly2dreggroupfullorder, poly2dreggroupnames, poly2dreggroupbincenters0, poly2dreggroupbincenters1 = self.get_poly2dreggroups()

        #save some lists of strings to the file for later use
        def create_dataset(name, content, length=None, dtype=h5py.special_dtype(vlen=str), compression="gzip"):
            dimension=[len(content), length] if length else [len(content)]
            ds = f.create_dataset(f"h{name}", dimension, dtype=dtype, compression=compression)
            ds[...] = content

        create_dataset("procs", procs)
        create_dataset("signals", signals)
        create_dataset("systs", systs)
        create_dataset("systsnoprofile", systsnoprofile)
        create_dataset("systsnoconstraint", systsnoconstraint)
        create_dataset("systgroups", systgroups)
        create_dataset("systgroupidxs", systgroupidxs, dtype=h5py.special_dtype(vlen=np.dtype('int32')))
        create_dataset("chargegroups", chargegroups)
        create_dataset("chargegroupidxs", chargegroups, 2, dtype='int32')
        create_dataset("polgroups", polgroups)
        create_dataset("polgroupidxs", polgroups, 3, dtype='int32')
        create_dataset("helgroups", helgroups)
        create_dataset("helgroupidxs", helgroups, 6, dtype='int32')
        create_dataset("sumgroups", sumgroups)
        create_dataset("sumgroupsegmentids", sumgroupsegmentids, dtype='int32')
        create_dataset("sumgroupidxs", sumgroupidxs, dtype='int32')
        create_dataset("chargemetagroups", chargemetagroups)
        create_dataset("chargemetagroupidxs", chargemetagroups, 2, dtype='int32')
        create_dataset("ratiometagroups", ratiometagroups)
        create_dataset("ratiometagroupidxs", ratiometagroups, 2, dtype='int32')
        create_dataset("helmetagroups", helmetagroups)
        create_dataset("helmetagroupidxs", helmetagroups, 6, dtype='int32')
        create_dataset("reggroups", reggroups)
        create_dataset("reggroupidxs", reggroupidxs, dtype=h5py.special_dtype(vlen=np.dtype('int32')))
        create_dataset("poly1dreggroups", poly1dreggroups)
        create_dataset("poly1dreggroupfirstorder", poly1dreggroupfirstorder, dtype='int32')
        create_dataset("poly1dreggrouplastorder", poly1dreggrouplastorder, dtype='int32')
        create_dataset("poly1dreggroupnames", poly1dreggroupnames, dtype=h5py.special_dtype(vlen="S256"))
        create_dataset("poly1dreggroupbincenters", poly1dreggroupbincenters, dtype=h5py.special_dtype(vlen=np.dtype('float64')))
        create_dataset("poly2dreggroups", poly2dreggroups)
        create_dataset("poly2dreggroupfirstorder", poly2dreggroupfirstorder, 2, dtype='int32')
        create_dataset("poly2dreggrouplastorder", poly2dreggrouplastorder, 2, dtype='int32')
        create_dataset("poly2dreggroupfullorder", poly2dreggroupfullorder, 2, dtype='int32')
        create_dataset("poly2dreggroupnames", poly2dreggroupnames, dtype=h5py.special_dtype(vlen="S256"))
        create_dataset("poly2dreggroupbincenters0", poly2dreggroupbincenters0, dtype=h5py.special_dtype(vlen=np.dtype('float64')))
        create_dataset("poly2dreggroupbincenters1", poly2dreggroupbincenters1, dtype=h5py.special_dtype(vlen=np.dtype('float64')))
        create_dataset("noigroups", noigroups)
        create_dataset("noigroupidxs", noigroupidxs, dtype='int32')
        create_dataset("maskedchans", self.masked_channels)

        #create h5py datasets with optimized chunk shapes
        nbytes = 0

        constraintweights = self.get_constraintweights(self.dtype)
        nbytes += writeFlatInChunks(constraintweights, f, "hconstraintweights", maxChunkBytes = self.chunkSize)
        constraintweights = None

        nbytes += writeFlatInChunks(data_obs, f, "hdata_obs", maxChunkBytes = self.chunkSize)
        data_obs = None

        if self.theoryFit:
            data_cov = self.theoryFitDataCov
            if data_cov.shape != (nbins,nbins):
                raise RuntimeError(f"covariance matrix has incompatible shape of {data_cov.shape}, expected is {(nbins,nbins)}!")
            full_cov = np.add(data_cov,np.diag(sumw2)) if self.theoryFitMCStat else data_cov
            nbytes += writeFlatInChunks(np.linalg.inv(full_cov), f, "hdata_cov_inv", maxChunkBytes = self.chunkSize)
            data_cov = None
            full_cov = None

        nbytes += writeFlatInChunks(kstat, f, "hkstat", maxChunkBytes = self.chunkSize)
        kstat = None

        if args.sparse:
            nbytes += writeSparse(norm_sparse_indices, norm_sparse_values, norm_sparse_dense_shape, f, "hnorm_sparse", maxChunkBytes = self.chunkSize)
            norm_sparse_indices = None
            norm_sparse_values = None
            nbytes += writeSparse(logk_sparse_indices, logk_sparse_values, logk_sparse_dense_shape, f, "hlogk_sparse", maxChunkBytes = self.chunkSize)
            logk_sparse_indices = None
            logk_sparse_values = None
        else:
            nbytes += writeFlatInChunks(norm, f, "hnorm", maxChunkBytes = self.chunkSize)
            norm = None
            nbytes += writeFlatInChunks(logk, f, "hlogk", maxChunkBytes = self.chunkSize)
            logk = None

        logger.info(f"Total raw bytes in arrays = {nbytes}")
        
    def get_output_filename(self, outfolder, sparse=False, postfix=None, doStatOnly=False):

        if not os.path.isdir(outfolder):
            os.makedirs(outfolder)
        outfilename = f"{outfolder}/{self.cardName}"

        if doStatOnly:
            outfilename += "_statOnly"
        if postfix is not None:
            outfilename += f"_{postfix}"
        if sparse:
            outfilename += "_sparse"

        outfilename += ".hdf5"
        return outfilename

    def book_systematic(self, syst, name):
        if syst.get('noProfile', False):
            self.systsnoprofile.add(name)
        elif syst.get("noConstraint", False) or syst.get("noi", False):
            self.systsnoconstraint.add(name)
        else:
            self.systsstandard.add(name)
        group = syst["group"]
        if group is None:
            logger.warning(f"Systemtaic {name} is not a member of any group")
            return 
            #TODO: adding a group with the name of the member instead?
            # group = name 
        if syst.get("noi", False):
            if group not in self.dict_noigroups:
                self.dict_noigroups[group] = set([name])
            else:
                self.dict_noigroups[group].add(name)
        else:
            if group not in self.dict_systgroups:
                self.dict_systgroups[group] = set([name])
            else:
                self.dict_systgroups[group].add(name)

    def get_systsstandard(self):
        return list(common.natural_sort(self.systsstandard))

    def get_systsnoprofile(self):
        return list(common.natural_sort(self.systsnoprofile))

    def get_systsnoconstraint(self):
        return list(common.natural_sort(self.systsnoconstraint))

    def get_systs(self):
        return self.get_systsnoconstraint() + self.get_systsstandard() + self.get_systsnoprofile()

    def get_constraintweights(self, dtype):
        systs = self.get_systs()
        constraintweights = np.ones([len(systs)], dtype=dtype)
        for syst in self.get_systsnoconstraint():
            constraintweights[systs.index(syst)] = 0.
        return constraintweights

    def get_groups(self, group_dict):
        systs = self.get_systs()
        groups = []
        idxs = []
        for group, members in group_dict.items():
            groups.append(group)
            idx = []
            for syst in members:
                idx.append(systs.index(syst))
            idxs.append(idx)
        return groups, idxs

    def get_noigroups(self):
        #list of groups of systematics to be treated as additional outputs for impacts, etc (aka "nuisances of interest")
        systs = self.get_systs()
        groups = []
        idxs = []
        for group, members in self.dict_noigroups.items():
            groups.append(group)
            for syst in members:
                idxs.append(systs.index(syst))
        return groups, idxs

    def get_systgroups(self):
        #list of groups of systematics (nuisances) and lists of indexes
        return self.get_groups(self.dict_systgroups)

    def get_sumgroups(self, procs):
        #list of groups of signal processes to be summed
        sumgroups = []
        sumgroupsegmentids = []
        sumgroupidxs = []
        dict_sumgroups = {}
        for chanInfo in self.get_channels().values():
            dict_sumgroups.update(chanInfo.cardSumGroups)
        for igroup, (group, members) in enumerate(common.natural_sort_dict(dict_sumgroups).items()):
            sumgroups.append(group)
            for proc in members:
                sumgroupsegmentids.append(igroup)
                sumgroupidxs.append(procs.index(proc))
        return sumgroups, sumgroupsegmentids, sumgroupidxs

    def get_chargegroups(self):
        #list of groups of signal processes by charge
        return [], []

    def get_polgroups(self):
        #list of groups of signal processes by polarization
        return [], []

    def get_helgroups(self):
        #list of groups of signal processes by helicity xsec
        return [], []

    def get_chargemetagroups(self):
        #list of groups of signal processes by chargemeta
        return [], []

    def get_ratiometagroups(self):
        #list of groups of signal processes by ratiometa
        return [], []

    def get_helmetagroups(self):
        #list of groups of signal processes by helmeta
        return [], []

    def get_reggroups(self):
        #list of groups of signal processes for regularization
        return [], []

    def get_poly1dreggroups(self):
        #list of groups of signal processes for regularization
        return [], [], [], [], []

    def get_poly2dreggroups(self):
        #list of groups of signal processes for regularization
        return [], [], [], [], [], [], []