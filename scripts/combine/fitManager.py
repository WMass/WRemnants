#!/usr/bin/env python3

# examples (add -d for tests)
#
# charge combination
#
# python WRemnants/scripts/combine/fitManager.py -i /scratch/mciprian/CombineStudies/Wmass/abseta1p0/qcdScale_byPt/  --comb [--skip-fit-data]
#
# single charge (can select a single charge as -c plus, otherwise both are done in sequence)
#
# python WRemnants/scripts/combine/fitManager.py -i /scratch/mciprian/CombineStudies/Wmass/abseta1p0/qcdScale_byPt/  -c "plus,minus" --fit-single-charge [--skip-fit-data]

import os, re, copy, math, array

import argparse

## safe batch mode
import sys
args = sys.argv[:]
sys.argv = ['-b']
import ROOT
sys.argv = args
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True


def safeSystem(cmd, dryRun=False, quitOnFail=True):
    print(cmd)
    if not dryRun:
        res = os.system(cmd)
        if res:
            print('-'*30)
            print("safeSystem(): error occurred when executing the following command. Aborting")
            print(cmd)
            print('-'*30)
            if quitOnFail:
                quit()
        return res
    else:
        return 0

def createFolder(checkdir, dryRun=False):
    if not os.path.exists(checkdir):
        print("Creating folder", checkdir)
        safeSystem("mkdir -p " + checkdir, dryRun=dryRun)

def prepareChargeFit(options, charges=["plus"]):

    cardSubfolderFullName = options.inputdir + options.cardFolder
    postfix = options.postfix
    cardkeyname = 'card'
    if options.fitSingleCharge: 
    # this cardkeyname is needed only when a single datacard for a given charge is different from those that would be used
    # for the combination (e.g. because to facilitate the combination some lines that require both charges are 
    # added to the datacard for a specific charge)
        cardkeyname += "_singleCharge{ch}".format(ch=charges[0])
        if postfix == "":
            postfix = "only{ch}".format(ch=charges[0])
        else:
            postfix = postfix + "_only{ch}".format(ch=charges[0])
        
    datacards=[]; 
    channels=[]
    binname = "ZMassWLike" if options.isWlike else "WMass"

    for charge in charges:
        datacards.append(os.path.abspath(options.inputdir)+"/{b}_{ch}.txt".format(b=binname,ch=charge))
        channels.append('{b}_{ch}'.format(b=binname,ch=charge))
    # add masked channel, only one for both gen charges
    maskedChannels = []
    if options.theoryAgnostic:
        datacards.append(os.path.abspath(options.inputdir)+"/{b}_inclusive_xnorm.txt".format(b=binname))
        channels.append('inclusive') # FIXME: should track what is done by CardTool.py
        maskedChannels.append('inclusive')

    print('='*30)
    print("Looking for these cards")
    print('-'*30)
    for d in datacards:
        print(d)
    print('='*30)

    ### prepare the combineCards and txt2hdf5 commands
    if sum([os.path.exists(card) for card in datacards]) == len(datacards):
        if options.fitSingleCharge:
            print("I am going to run fit for single charge {ch}".format(ch=charges[0]))
        else:
            print("I found the cards for W+ and W-. Combining them now...")

        combinedCard = "{d}/{b}_{s}.txt".format(d=os.path.abspath(cardSubfolderFullName), b=binname, s=cardkeyname)
        ccCmd = "combineCards.py --noDirPrefix {cards} > {combinedCard} ".format(cards=' '.join(['{ch}={dcfile}'.format(ch=channels[i],dcfile=card) for i,card in enumerate(datacards)]), combinedCard=combinedCard)
        ## run the commands: need cmsenv in the combinetf release
        print 
        print
        if args.skip_card:
            safeSystem(ccCmd, dryRun=True)
            print("Unmodified combined card in ",combinedCard)
        else:
            safeSystem(ccCmd, dryRun=options.dryRun)
            print("New combined card in ",combinedCard)
        print

        if options.doOnlyCard:
            return
        
        txt2hdf5Cmd = 'text2hdf5.py {cf} --dataset {dn}'.format(cf=combinedCard, dn=options.dataname)
        if options.theoryAgnostic:
            maskchan = ["--maskedChan {mc}".format(mc=maskedChannel) for maskedChannel in maskedChannels]
            txt2hdf5Cmd += " --sparse {mc} --X-allow-no-background".format(mc=" ".join(maskchan))
        else:
            txt2hdf5Cmd += " --X-allow-no-signal"
            
        if len(postfix):
            txt2hdf5Cmd = txt2hdf5Cmd + " --postfix " + postfix
        if options.clipSystVariations > 0.0:
            txt2hdf5Cmd = txt2hdf5Cmd + " --clipSystVariations " + str(options.clipSystVariations)
        if options.clipSystVariationsSignal > 0.0:
            txt2hdf5Cmd = txt2hdf5Cmd + " --clipSystVariationsSignal " + str(options.clipSystVariationsSignal)

        print 
        if options.skip_text2hdf5: 
            print(txt2hdf5Cmd)
        else:
            print("Running text2hdf5.py, it might take time ...")
            safeSystem(txt2hdf5Cmd, dryRun=options.dryRun)
            
        metafilename = combinedCard.replace('.txt','.hdf5')
        if args.theoryAgnostic:
            metafilename = metafilename.replace('.hdf5','_sparse.hdf5')
        if len(postfix):
            metafilename = metafilename.replace('.hdf5','_%s.hdf5' % postfix)
            
        bbboptions = " --binByBinStat "
        if not options.noCorrelateXsecStat: bbboptions += "--correlateXsecStat "
        combineCmd = 'combinetf.py -t -1 {bbb} {metafile} --doImpacts --saveHists --computeHistErrors '.format(metafile=metafilename, bbb="" if options.noBBB else bbboptions)
        if options.combinetfOption:
            combineCmd += " %s" % options.combinetfOption
        if args.theoryAgnostic:
            combineCmd += " --POIMode mu --allowNegativePOI"
        else:
            combineCmd += " --POIMode none"                        
            
        fitdir_data = "{od}/fit/data/".format(od=os.path.abspath(cardSubfolderFullName))
        fitdir_Asimov = fitdir_data.replace("/fit/data/", "/fit/hessian/")
        fitdir_toys = fitdir_data.replace("/fit/data/", "/fit/toys/")
        for fitdir in [fitdir_data, fitdir_Asimov]:
            createFolder(fitdir, options.dryRun)
        print("")
        fitPostfix = "" if not len(postfix) else ("_"+postfix)

        print("Use the following command to run combine (add --seed <seed> to specify the seed, if needed). See other options in combinetf.py")
        print
        combineCmd_data = combineCmd.replace("-t -1 ", "-t 0 ")
        combineCmd_toys = combineCmd.replace("-t -1 ", "-t {} ".format(options.toys))

        bbbtext = "0" if options.noBBB else "1_cxs0" if options.noCorrelateXsecStat else "1_cxs1"
        combineCmd_data   = combineCmd_data + " --postfix Data{pf}_bbb{b} --outputDir {od} ".format(pf=fitPostfix, od=fitdir_data, b=bbbtext)
        combineCmd_Asimov = combineCmd      + " --postfix Asimov{pf}_bbb{b} --outputDir {od} ".format(pf=fitPostfix, od=fitdir_Asimov, b=bbbtext)
        combineCmd_toys   = combineCmd_toys + " --postfix Toys{pf}_bbb{b} --outputDir {od} ".format(pf=fitPostfix, od=fitdir_toys,  b=bbbtext)
        if not options.skip_combinetf and not options.skipFitData:
            safeSystem(combineCmd_data, dryRun=options.dryRun)
        else:
            print(combineCmd_data)
        print
        if not options.skip_combinetf and not options.skipFitAsimov:
            safeSystem(combineCmd_Asimov, dryRun=options.dryRun)
        else:
            print(combineCmd_Asimov)
        print
        if not options.skip_combinetf and options.toys:
            safeSystem(combineCmd_toys, dryRun=options.dryRun)
        else:
            print(combineCmd_toys)
        print
            
    else:
        print("Warning, I couldn't find the following cards. Check names and paths")
        for card in datacards:
            if not os.path.exists(card):
                print(card)

            
def combineCharges(options):
    prepareChargeFit(options, charges=['plus','minus'])

    
if __name__ == "__main__":    

    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input', dest='inputdir', default='', type=str, help='input directory with the root files inside (cards are eventually stored in the foldr specified with option --cf)')
    parser.add_argument('--cf','--card-folder', dest='cardFolder', default='nominal', type=str, help='Subfolder created inside inputdir to store all cards and fit results in a given configuration, for better bookkeeping when doing tests')
    parser.add_argument('-c','--charge', dest='charge', default='both', choices=['plus','minus','both'], type=str, help='process given charge. default is both')
    parser.add_argument(     '--comb'   , dest='combineCharges' , default=False, action='store_true', help='Combine W+ and W-, if single cards are done')
    parser.add_argument(      '--fit-single-charge', dest='fitSingleCharge', default=False, action='store_true', help='Prepare datacard for single-charge fit. For each charge, a postfix is appended to option --postfix, so no need to add the charge explicitly')
    parser.add_argument(     '--postfix',    dest='postfix', type=str, default="", help="Postfix for .hdf5 file created with text2hdf5.py");
    parser.add_argument('--wlike', dest='isWlike', action="store_true", default=False, help="Make cards for the W-like analysis. Default is Wmass");
    # options for card maker and fit
    parser.add_argument(       "--clipSystVariations", type=float, default=-1.,  help="Clipping of syst variations, passed to text2hdf5.py")
    parser.add_argument(       "--clipSystVariationsSignal", type=float, default=-1.,  help="Clipping of signal syst variations, passed to text2hdf5.py")
    parser.add_argument(       '--no-bbb'  , dest='noBBB', default=False, action='store_true', help='Do not use bin-by-bin uncertainties')
    parser.add_argument(       '--correlate-xsec-stat'  , dest='noCorrelateXsecStat', default=True, action='store_false', help='If given, use option --correlateXsecStat when using bin-by-bin uncertainties (for mass measurements it should not be needed because we do not use prefit cross sections)')
    parser.add_argument('-d',  '--dry-run'  , dest='dryRun', default=False, action='store_true', help='Do not execute command to make cards or fit')
    parser.add_argument(       '--doOnlyCard'  , dest='doOnlyCard', default=False, action='store_true', help='Do only card and exit (equivalent to using --no-text2hdf5 and --no-combinetf together)')
    parser.add_argument(       '--no-card', dest='skip_card' , default=False, action='store_true', help='Go directly to fit part without regenerating the card (can also skip text2hdf5 with --no-text2hdf5), useful when editing the card manually for tests')
    parser.add_argument(       '--no-text2hdf5'  , dest='skip_text2hdf5', default=False, action='store_true', help='skip running text2hdf5.py at the end, only prints command (useful if hdf5 file already exists, or for tests)')
    parser.add_argument(       '--no-combinetf'  , dest='skip_combinetf', default=False, action='store_true', help='skip running combinetf.py at the end, just print command (useful for tests)')
    parser.add_argument(       '--skip-fit-data', dest='skipFitData' , default=False, action='store_true', help='If True, fit only Asimov')
    parser.add_argument(       '--skip-fit-asimov', dest='skipFitAsimov' , default=False, action='store_true', help='If True, fit only data')
    parser.add_argument("-D", "--dataset",  dest="dataname", default="data_obs",  type=str,  help="Name of the observed dataset (pass name without x_ in the beginning). Useful to fit another pseudodata histogram")
    parser.add_argument("--combinetf-option",  dest="combinetfOption", default="",  type=str,  help="Pass other options to combinetf (TODO: some are already activated with other options, might move them here)")
    parser.add_argument("-t",  "--toys", type=int, default=0, help="Run combinetf for N toys if argument N is positive")
    parser.add_argument(       '--theoryAgnostic', action='store_true', help='Run theory agnostic fit, with masked channels and so on')
    args = parser.parse_args()

    if not args.dryRun:
        try:
            cmssw = os.environ['CMSSW_BASE']
        except:
            cmssw = ""
        if cmssw == "":
            print("\n")
            print("Error: to use combinetf you need to activate cmsenv from a release.")
            print("You should work from a cmssw-cc7 singularity environment to get the release.")
            print("Aborting ...")
            print("\n")
            quit()

    if not args.inputdir.endswith("/"):
        args.inputdir += "/"
            
    if args.cardFolder:
        if not args.cardFolder.endswith("/"):
            args.cardFolder += "/"
        cardFolderFullName = args.inputdir + args.cardFolder
        createFolder(cardFolderFullName, dryRun=False) # always create this folder, even for tests
        fcmd = open(cardFolderFullName+"cardMaker_command.txt", "w")
        fcmd.write("%s\n\n" % " ".join(sys.argv))
        fcmd.close()

    fitCharges = ["plus", "minus"] if args.charge == "both" else [args.charge]

    if not args.combineCharges and not args.fitSingleCharge:
        print("Warning: must pass one option between --fit-single-charge and --comb to fit single charge or combination.")
    
    if args.combineCharges:
        if args.fitSingleCharge:
            print("Error: options --fit-single-charge and --comb are incompatible. Abort")
            quit()
        if len(fitCharges) != 2:
            print("Error: --comb requires two charges, use -C 'plus,minus' and try again")
            quit()
            
    if args.fitSingleCharge:
        for charge in fitCharges:
            prepareChargeFit(args, charges=[charge])
            print('-'*30)
            print("Done with charge {ch}".format(ch=charge))
            print('-'*30)

    if args.combineCharges and len(fitCharges)==2:
        combineCharges(args)                
        print('-'*30)
        print("Done with charge combination")
        print('-'*30)
