#!/usr/bin/env python3

from shutil import copyfile
import re, sys, os, os.path, subprocess, json, ROOT
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("inputfile", type=str, nargs=1, help="Input root file");
parser.add_argument('-c', '--class',          dest='className',      default='', type=str, help='Filter output based on the object class (e.g. -c TH1D)')
parser.add_argument('-x', '--exclude-regexp', dest='excludeRegexp', default='', type=str, help='Like option -r, but will reject object if name matches')
parser.add_argument('-f', '--output-format',  dest='outputFormat',  default='all', type=str, help='Print type, name and (for histograms only) integral (all, default); just name (name)')
parser.add_argument('-s', '--silent',  action='store_true', help='Silent mode: just print summary, not all entries')
parser.add_argument('-i', '--inherit',   default='', type=str, help='Filter output based on whether the object class inheriths from this (e.g. -i TH1 will match all root histogram objects)')  
parser.add_argument('-r', '--regexp',  default='', type=str, help='Filter output passing a regular expression to be matched in the object name')
parser.add_argument(      '--print-min', dest='printMin', action='store_true', help='If true, print also minimum bin content (only for histograms)')
parser.add_argument(      '--min-up-threshold', dest='minUpThreshold', default=None, type=float, help='If given, and --print-min is used, filter histograms where the minimum conent is below this value.')
parser.add_argument(      '--print-min-negative', dest='printMinNegative', action='store_true', help='If true, print also minimum bin content when negative (only for histograms)')
parser.add_argument(      '--print-integral-cap-negative', dest='printIntegralCapNegative',  action='store_true', help='If true, print integral when capping negative bin to 0 (print also ratio with actual integral')
parser.add_argument(      '--print-axis-label-histogram', dest='printAxisLabelHisto', default='', type=str, help='If given, print axis labels of histogram with this name (everything else in the file will be skipped, and other options might be ignored). For a TH2, the x axis is chosen')
parser.add_argument(      '--print-branch-tree', dest='printBranchTree', default='', type=str, help='If given, print names of branches in tree with this name (everything else in the file will be skipped, and other options might be ignored).')

args = parser.parse_args()

tf = ROOT.TFile.Open(args.inputfile[0],"READ")
nRead = 0
nNullIntegral = 0
nNegativeBin = 0
nNonPositiveBin = 0
histWithNegBins = []

minUpThreshold = args.minUpThreshold

if not args.silent:
    print('-'*50)
    if args.outputFormat == "all":    
        print("{: <10} {: <20}    {y}    {txt} ".format("Class","name",
                                                        y="integral(TH only)",
                                                        txt="integral(nonNeg)/integral" if args.printIntegralCapNegative else ""))
    elif args.outputFormat == "name":
        print("name")
    print('-'*50)
        
for k in tf.GetListOfKeys():
    name=k.GetName()
    if args.printAxisLabelHisto != "" and name != args.printAxisLabelHisto:
        continue
    if args.printBranchTree != "" and name != args.printBranchTree:
        continue

    if len(args.regexp) and not re.match(args.regexp,name): continue
    if len(args.excludeRegexp) and re.match(args.excludeRegexp,name): continue
    obj=k.ReadObj()
    if len(args.className) and obj.ClassName() != args.className: continue
    if len(args.inherit) and not obj.InheritsFrom(args.inherit): continue
    nRead += 1
    integral = -1
    integralOnlyNonNegativeBin = 0.0
    minBinContent = 1
    if (obj.ClassName().startswith("TH") and obj.InheritsFrom("TH1")):

        if args.printAxisLabelHisto != "" and name == args.printAxisLabelHisto:
            print('-'*30)
            print("Labels of histogram %s" % name)
            print('-'*30)
            for ib in range(1,1+obj.GetNbinsX()):
                print("{: <10} {l}".format(ib,l=obj.GetXaxis().GetBinLabel(ib)))
            print('-'*30)

        integral = obj.Integral() 
        minBinContent = obj.GetBinContent(obj.GetMinimumBin())
        if args.printIntegralCapNegative:
            for ibin in range(1,obj.GetNbinsX()+1):
                integralOnlyNonNegativeBin += max(0.0,obj.GetBinContent(ibin))

    if integral == 0.0: nNullIntegral += 1
    if minBinContent < 0:
        nNegativeBin += 1
        histWithNegBins.append(name)
    if minBinContent <= 0: nNonPositiveBin += 1
    if minUpThreshold and minBinContent > minUpThreshold:
        continue
    
    if not args.silent:
        if args.outputFormat == "all":
            #print("%s   %s   %s" % (obj.ClassName(), name, str(integral) if integral >= 0 else "")
            print("{: <10} {: <20}    {y}    {r}    {m} ".format(obj.ClassName(), name, 
                                                                 y=str(integral) if integral >= 0 else "",
                                                                 r=str(integralOnlyNonNegativeBin/integral) if args.printIntegralCapNegative else "",
                                                                 m=str(minBinContent) if (args.printMin or (args.printMinNegative and minBinContent < 0)) else ""))
        elif args.outputFormat == "name":
            print("%s" % name)


    if (obj.ClassName().startswith("TTree") or obj.InheritsFrom("TTree")):
        print('-'*30)
        print("Branches of tree %s" % name)
        print('-'*30)
        lok = obj.GetListOfLeaves()
        for ip,p in enumerate(lok):
            print("{: <10} {l}".format(ip,l=p.GetName()))
        print('-'*30)

if args.printAxisLabelHisto != "" or args.printBranchTree != "":
    quit()

print("########################################")
print("########################################")
print(" Summary")
print("----------------------------------------")
print("There were %d keys in the file" % tf.GetNkeys())
print("There were %d keys passing filters" % nRead)
print("There were %d histograms with integral = 0" % nNullIntegral)
print("There were %d histograms with one or more negative bins" % nNegativeBin)
print("There were %d histograms with one or more non positive bins" % nNonPositiveBin)
print()
print("List of histograms with negative bin content")
for i,name in enumerate(histWithNegBins):
    print(f"{i}) {name}")
    


