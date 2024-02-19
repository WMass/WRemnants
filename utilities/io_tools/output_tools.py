import string
import os
import sys 
import subprocess
import time
import hdf5plugin
import h5py
import narf
import numpy as np
from utilities import common, logging
import glob
import shutil
import lz4.frame
import pickle

logger = logging.child_logger(__name__)

def readTemplate(templateFile, templateDict, filt=None):
    if not os.path.isfile(templateFile):
        raise ValueError("Template file %s is not a valid file!" % templateFile)
    with open(templateFile, "r") as tf:
        lines = filter(filt, tf.readlines()) if filt else tf.readlines()
        source = string.Template("".join(lines))
    filled = source.substitute(templateDict)
    return filled

def fillTemplatedFile(templateFile, outFile, templateDict, append=False):
    filled = readFromTempate(templateFile, templateDict)
    with open(outFile, "w" if not append else "a") as outFile:
        outFile.write(result)

def analysis_debug_output(results):
    logger.debug("")
    logger.debug("Unweighted (Weighted) events, before cut")
    logger.debug("-"*30)
    for key,val in results.items():
        if "event_count" in val:
            logger.debug(f"Dataset {key.ljust(30)}:  {str(val['event_count']).ljust(15)} ({round(val['weight_sum'],1)})")
            logger.debug("-"*30)
    logger.debug("")

def writeMetaInfoToRootFile(rtfile, exclude_diff='notebooks', args=None):
    import ROOT
    meta_dict = narf.ioutils.make_meta_info_dict(exclude_diff, args=args, wd=common.base_dir)
    d = rtfile.mkdir("meta_info")
    d.cd()
    
    for key, value in meta_dict.items():
        out = ROOT.TNamed(str(key), str(value))
        out.Write()

def write_analysis_output(results, outfile, args):
    analysis_debug_output(results)

    to_append = []
    if args.theoryCorr and not args.theoryCorrAltOnly:
        to_append.append(args.theoryCorr[0]+"Corr")
    if args.maxFiles is not None:
        to_append.append(f"maxFiles_{args.maxFiles}".replace("-","m"))
    if args.pdfs[0] != "ct18z": 
        to_append.append(args.pdfs[0])
    if hasattr(args, "ptqVgen") and args.ptqVgen:
        to_append.append("vars_qtbyQ")

    if to_append and not args.forceDefaultName:
        outfile = outfile.replace(".hdf5", f"_{'_'.join(to_append)}.hdf5")

    if args.postfix:
        outfile = outfile.replace(".hdf5", f"_{args.postfix}.hdf5")

    if args.outfolder:
        if not os.path.exists(args.outfolder):
            logger.info(f"Creating output folder {args.outfolder}")
            os.makedirs(args.outfolder)
        outfile = os.path.join(args.outfolder, outfile)

    if args.appendOutputFile:
        outfile = args.appendOutputFile
        if os.path.isfile(outfile):
            logger.info(f"Analysis output will be appended to file {outfile}")
            open_as="a"
        else:
            logger.warning(f"Analysis output requested to be appended to file {outfile}, but the file does not exist yet, it will be created instead")
            open_as="w"
    else:
        if os.path.isfile(outfile):
            logger.warning(f"Output file {outfile} exists already, it will be overwritten")
        open_as="w"

    time0 = time.time()
    with h5py.File(outfile, open_as) as f:
        for k, v in results.items():
            logger.debug(f"Pickle and dump {k}")
            narf.ioutils.pickle_dump_h5py(k, v, f)

        if "meta_info" not in f.keys():
            narf.ioutils.pickle_dump_h5py("meta_info", narf.ioutils.make_meta_info_dict(args=args, wd=common.base_dir), f)

    logger.info(f"Writing output: {time.time()-time0}")
    logger.info(f"Output saved in {outfile}")

def is_eosuser_path(path):
    if not path:
        return False
    path = os.path.realpath(path)
    return path.startswith("/eos/user") or path.startswith("/eos/home-")

def make_plot_dir(outpath, outfolder=None, eoscp=False):
    if eoscp and is_eosuser_path(outpath):
        outpath = os.path.join("temp", split_eos_path(outpath)[1])
        if not os.path.isdir(outpath):
            logger.info(f"Making temporary directory {outpath}")
            os.makedirs(outpath)

    full_outpath = outpath
    if outfolder:
        full_outpath = os.path.join(outpath, outfolder)
    if outpath and not os.path.isdir(outpath):
        raise IOError(f"The path {outpath} doesn't not exist. You should create it (and possibly link it to your web area)")
        
    if full_outpath and not os.path.isdir(full_outpath):
        try:
            os.makedirs(full_outpath)
            logger.info(f"Creating folder {full_outpath}")
        except FileExistsError as e:
            logger.warning(e)
            pass

    return full_outpath

def copy_to_eos(outpath, outfolder=None):
    eospath, outpath = split_eos_path(outpath)
    fullpath = outpath
    if outfolder:
        fullpath = os.path.join(outpath, outfolder)
        logger.info(f"Copying {outpath} to {eospath}")

    tmppath = os.path.join("temp", fullpath)

    for f in glob.glob(tmppath+"/*"):
        if os.path.isfile(f):
            command = ["xrdcp", "-f", f, "/".join(["root://eosuser.cern.ch", eospath, f.replace("temp/", "")])]

            logger.debug(f"Executing {' '.join(command)}")
            if subprocess.call(command):
                raise IOError("Failed to copy the files to eos! Perhaps you are missing a kerberos ticket and need to run kinit <user>@CERN.CH?"
                    " from lxplus you can run without eoscp and take your luck with the mount.")

    shutil.rmtree(tmppath) 

def write_theory_corr_hist(output_name, process, output_dict, args=None, file_meta_data=None): 
    outname = output_name
    output_filename = f"{outname}Corr{process}.pkl.lz4"
    logger.info(f"Write correction file {output_filename}")
    result_dict = {process : output_dict, "meta_data" : narf.ioutils.make_meta_info_dict(args, wd=common.base_dir)}
    if file_meta_data is not None:
        result_dict["file_meta_data"] = file_meta_data
    with lz4.frame.open(output_filename, "wb") as f:
        pickle.dump(result_dict, f, protocol = pickle.HIGHEST_PROTOCOL)

def split_eos_path(path):

    path = os.path.realpath(path)
    if not is_eosuser_path(path):
        raise ValueError(f"Expected a path on /eos/user, found {path}!")
        
    splitpath = [x for x in path.split("/") if x]
    # Can be /eos/user/<letter>/<username> or <letter-username>
    if "home-" in splitpath[1]:
        eospath = "/".join(["/eos/user", splitpath[1].split("-")[-1], splitpath[2]])
        basepath = "/".join(splitpath[3:])
    else:
        eospath = "/".join(splitpath[:4])
        basepath = "/".join(splitpath[4:])

    if path[0] == "/":
        eospath = "/"+eospath

    return eospath, basepath

