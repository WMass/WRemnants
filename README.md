# Hayden notes

get the other new files
works:
'scetlib_dyturboN3p0LL_LatticeNP_pdfasCorrZ.pkl.lz4'

scetlib_dyturboN3p0LL_LatticeNP_pdfasCorrZ.pkl.lz4

theory_corrs
['scetlib_dyturboN3p0LL_LatticeNP_pdfas']

scetlib_dyturboLatticeNP_CT18Z_N3p0LL_N2LO_CorrZ.pkl.lz4        
scetlib_dyturboLatticeNP_CT18Z_N3p0LL_N2LO_pdfas_CorrZ.pkl.lz4
scetlib_dyturboLatticeNP_CT18Z_N3p0LL_N2LO_pdfvars_CorrZ.pkl.lz4

(Pdb) fname
'/work/submit/hayden17/WRemnants/utilities/../wremnants-data/data//TheoryCorrections/5020GeV/scetlib_dyturboLatticeNP_CT18Z_N3p0LL_N2LO_pdfas_CorrZ.pkl.lz4'
(Pdb) label
'Z'
(Pdb) get_corr_name(generator, minnlo_ratio=minnlo_ratio)
'scetlib_dyturboLatticeNP_CT18Z_N3p0LL_N2LO_pdfas_minnlo_ratio'


(Pdb) corr
{'ZMUMU5020GEV': {'scetlib_dyturboLatticeNP_CT18Z_N3p0LL_N2LO_pdfas__minnlo_ratio': Hist(
  Regular(1, 60, 120, name='Q'),
  Variable([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 4, 5], underflow=False, name='absY'),
  Variable(array([  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,
        11.,  12.,  13.,  14.,  15.,  16.,  17.,  18.,  19.,  20.,  22.,
        24.,  26.,  28.,  30.,  33.,  37.,  44., 100.]), name='qT'),
  Regular(1, 0, 1, underflow=False, overflow=False, name='charge'),
  StrCategory(['pdfCT18ZNNLO_as_0118', 'pdfCT18ZNNLO_as_0116', 'pdfCT18ZNNLO_as_0120'], name='vars'),
  storage=Double()) # Sum: 1447.8529655611378 (5915.852965561138 with flow), 'scetlib_dyturboLatticeNP_CT18Z_N3p0LL_N2LO_pdfas__hist': Hist(
  Regular(1, 60, 120, name='Q'),
  Variable([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 4, 5], underflow=False, name='absY'),
  Variable(array([  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,
        11.,  12.,  13.,  14.,  15.,  16.,  17.,  18.,  19.,  20.,  22.,
        24.,  26.,  28.,  30.,  33.,  37.,  44., 100.]), name='qT'),
  Regular(1, 0, 1, underflow=False, overflow=False, name='charge'),
  StrCategory(['pdfCT18ZNNLO_as_0118', 'pdfCT18ZNNLO_as_0116', 'pdfCT18ZNNLO_as_0120'], name='vars'),
  storage=Weight()) # Sum: WeightedSum(value=1967.72, variance=6.46709e-06), 'minnlo_ref_hist': Hist(
  Variable([60, 120], name='Q'),
  Variable([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 4, 5], underflow=False, name='absY'),
  Variable(array([  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,
        11.,  12.,  13.,  14.,  15.,  16.,  17.,  18.,  19.,  20.,  22.,
        24.,  26.,  28.,  30.,  33.,  37.,  44., 100.]), underflow=False, name='qT'),
  Integer(0, 1, underflow=False, overflow=False, name='charge'),
  StrCategory(['as0118', 'as0116', 'as0120'], name='vars'),
  storage=Double()) # Sum: 1996.439537422493 (2094.833318129759 with flow)}, 'meta_data': {'time': '2026-01-16 14:58:39.187331', 'command': "scripts/corrections/make_theory_corr.py -m '/home/submit/david_w/ceph/AlphaS/results_histmaker/260116_5020GeV_gen/w_z_gen_dists_nnpdf31.hdf5' -g 'scetlib_dyturbo' --proc Zmumu5020GeV -o '/home/submit/david_w/public_html/AlphaS/260113_5TeV_theoryCorrections/Z/as' --axes Y qT -c '/home/submit/david_w/work/TheoryCorrections/SCETlib/com5p02_ct18z_newnps_n3+0ll_lattice_pdfvars/inclusive_Z_COM5p02_CT18Z_N3+0LL_lattice_allvars_higherprecision_pdfas_pdf_combined.pkl' '/home/submit/david_w/work/TheoryCorrections/SCETlib/com5p02_ct18z_newnps_n3+0ll_lattice_pdfvars_nnlo_sing/inclusive_Z_COM5p02_CT18Z_N3+0LL_lattice_higherprecision_pdfas_nnlo_sing_pdf_combined.pkl' '/home/submit/david_w/work/TheoryCorrections/DYTURBO/nnlo-scetlibmatch-5.02TeV/pdfvariations/CT18ZNNLO_as/z0/results_z-2d-nnlo-vj-CT18ZNNLO_as-member{i}-scetlibmatch.txt' -p 'LatticeNP_CT18Z_N3p0LL_N2LO_pdfas_' --outpath 'wremnants-data/data/TheoryCorrections/5020GeV/' --minnloh 'nominal_gen_pdfNNPDF31alphaS002'", 'args': {}, 'git_hash': '"a15224d664d2c12c458aef9cfa4add5a421a1ebe"\n', 'git_diff': 'diff --git a/wremnants-data b/wremnants-data\nindex 84b236cc..5dc16ca6 160000\n--- a/wremnants-data\n+++ b/wremnants-data\n@@ -1 +1 @@\n-Subproject commit 84b236cc04689363f37c83172e364ac559957747\n+Subproject commit 5dc16ca6d1417033f32a7a601c544e88d53db0e5\n'}, 'file_meta_data': {'w_z_gen_dists_nnpdf31.hdf5': {'time': '2026-01-16 14:51:26.217111', 'command': "scripts/histmakers/w_z_gen_dists.py -o '/home/submit/david_w/ceph/AlphaS/results_histmaker/260116_5020GeV_gen' --dataPath '/scratch/submit/cms/wmass/NanoAOD/LowPU/2017G/' --era 2017G --filterProcs Zmumu5020GeV Ztautau5020GeV --pdf nnpdf31", 'args': {'verbose': 3, 'noColorLogger': False, 'nThreads': 0, 'pdfs': ['nnpdf31'], 'altPdfOnlyCentral': False, 'maxFiles': None, 'filterProcs': ['Zmumu5020GeV', 'Ztautau5020GeV'], 'excludeProcs': [], 'postfix': None, 'forceDefaultName': False, 'theoryCorr': [], 'theoryCorrAltOnly': False, 'ewTheoryCorr': [], 'skipHelicity': False, 'noRecoil': False, 'recoilHists': False, 'recoilUnc': False, 'highptscales': False, 'dataPath': '/scratch/submit/cms/wmass/NanoAOD/LowPU/2017G/', 'noVertexWeight': False, 'validationHists': False, 'onlyMainHistograms': False, 'met': 'DeepMETPVRobust', 'outfolder': '/home/submit/david_w/ceph/AlphaS/results_histmaker/260116_5020GeV_gen', 'appendOutputFile': '', 'sequentialEventLoops': False, 'era': '2017G', 'scale_A': 1.0, 'scale_e': 1.0, 'scale_M': 1.0, 'nonClosureScheme': 'A-M-combined', 'correlatedNonClosureNP': True, 'dummyNonClosureA': False, 'dummyNonClosureAMag': 6.8e-05, 'dummyNonClosureM': False, 'dummyNonClosureMMag': 0.0, 'noScaleToData': False, 'aggregateGroups': ['Diboson', 'Top'], 'muRmuFPolVarFilePath': '/work/submit/david_w/WRemnants/utilities/../wremnants-data/data//MiNNLOmuRmuFPolVar/', 'muRmuFPolVarFileTag': 'x0p50_y4p00_ConstrPol5ExtYdep_Trad', 'nToysMC': -1, 'varianceScalingForToys': 1, 'randomSeedForToys': 0, 'sfFile': '', 'printParser': None, 'skipHelicityXsecs': False, 'propagatePDFstoHelicity': False, 'useTheoryAgnosticBinning': False, 'useUnfoldingBinning': False, 'genPtBinningAsReco': False, 'useCorrByHelicityBinning': False, 'singleLeptonHists': False, 'photonHists': False, 'skipEWHists': False, 'signedY': False, 'fiducial': None, 'auxiliaryHistograms': False, 'ptqVgen': None, 'helicity': False, 'theoryCorrections': False, 'addHelicityAxis': False, 'addCharmAxis': False, 'finePtVBinning': False, 'centralBosonPDFWeight': False}, 'git_hash': '"a15224d664d2c12c458aef9cfa4add5a421a1ebe"\n', 'git_diff': 'diff --git a/wremnants-data b/wremnants-data\nindex 84b236cc..5dc16ca6 160000\n--- a/wremnants-data\n+++ b/wremnants-data\n@@ -1 +1 @@\n-Subproject commit 84b236cc04689363f37c83172e364ac559957747\n+Subproject commit 5dc16ca6d1417033f32a7a601c544e88d53db0e5\n'}, 'inclusive_Z_COM5p02_CT18Z_N3+0LL_lattice_allvars_higherprecision_pdfas_pdf_combined.pkl': {'time': '2026-01-07 14:22:09.536935', 'command': '/eos/home-a/areimers/scetlib-cms-newnp/prod/scetlib_run//scetlib-run-qT.py 8 1 inclusive_Z_COM5p02_CT18Z_N3+0LL_lattice_allvars_higherprecision_pdfas.ini --no-datfiles --start-bin 0 --stop-bin 100 --fixed-var 0 --pdf-member CT18ZNNLO_as_0118 --alphas 0.118', 'git_hash': '"205a087c79fe996466df2f9c4b2055653027dd6e"\n', 'git_diff': 'diff --git a/prod/scetlib_run/condor_template b/prod/scetlib_run/condor_template\nindex 4cbce39..173fa13 100644\n--- a/prod/scetlib_run/condor_template\n+++ b/prod/scetlib_run/condor_template\n@@ -14,21 +14,20 @@ NProcesses = $$(NVARS)*$$(NBinSteps)*$$(NPDF)\n \n initialdir = ${submitdir}\n transfer_input_files = ${transfer_files}\n-request_memory       = 2000*$$(NCPU)\n+request_memory       = 4000\n request_disk         = 2048000\n request_cpus         = $$(NCPU)\n-requirements = (OpSysAndVer =?= "CentOS7")\n +AccountingGroup = "group_u_CMST3.all"\n +JobFlavour = "${queue}"\n \n BinNum = ($$(Process) % ($$(NBinSteps)))*$$(NBinsPerJob)\n VarNum = $$(Process) / ($$(NBinSteps)) % $$(NVARS)\n-PdfNum = $$(Process) / ($$(NBinSteps)*$$(NVARS)) % $$(NPDF)\n+PdfNum = $$$$([$$(Process) / ($$(NBinSteps)*$$(NVARS)) % $$(NPDF)])\n+\n+Arguments            = "${scetlib_dir} $$(NCPU) 1 $$(INPUT) --no-datfiles --start-bin $$$$([$$(BinNum)]) --stop-bin $$$$([$$(BinNum)+$$(NBinsPerJob)]) --fixed-var $$$$([$$(VarNum)]) --pdf-member $$(PdfNum) ${alphas_arg}"\n \n-Arguments            = "${scetlib_dir}/prod/scetlib_run $$(NCPU) 1 $$(INPUT) --no-datfiles --start-bin $$$$([$$(BinNum)]) --stop-bin $$$$([$$(BinNum)+$$(NBinsPerJob)]) --fixed-var $$$$([$$(VarNum)]) --pdf-member $$$$([$$(PdfNum)])" ${pdf_set}\n output               = condor_out/Job$$(Process).out\n error                = condor_out/Job$$(Process).err\n-Log                  = condor_out/Job$$(Process).log\n-\n-Queue $$(NProcesses)\n+Log                  = condor_out/AllLogs.log\n \n+Queue $$(NProcesses) ${queue_command}\ndiff --git a/prod/scetlib_run/scetlib-manage-condor.py b/prod/scetlib_run/scetlib-manage-condor.py\nindex 5d55e9e..199c8be 100644\n--- a/prod/scetlib_run/scetlib-manage-condor.py\n+++ b/prod/scetlib_run/scetlib-manage-condor.py\n@@ -22,11 +22,13 @@ parser.add_argument("-r", "--runcard", type=str, help="input (.ini) file", requi\n parser.add_argument("-b", "--bins-per-job", type=int, default=1)\n parser.add_argument("-j", "--ncpu", type=int, default=1)\n parser.add_argument("-p", "--pdf-members", type=str, nargs="*", default=None)\n+parser.add_argument("--pdf-set", type=str, default=None)\n parser.add_argument("-a", "--alphas", type=float, nargs="*", default=None)\n parser.add_argument("-f", "--submitdir", type=str, default=".")\n parser.add_argument("-s", "--scetlib-afs-path", type=str, default=f"/afs/cern.ch/work/{os.getlogin()[0]}/{os.getlogin()}/scetlib-cms")\n parser.add_argument("-q", "--queue", type=str, choices=["longlunch", "espresso", "workday", "tomorrow", "testmatch", "nextweek"], \n     help="Condor runtime queue to use", default="workday")\n+parser.add_argument("--make-vars-per-pdf", action=\'store_true\', help="in combine step (and only there), write out one file per pdfas variation, which contains all variations handled internally via .conf file.")\n \n subparsers = parser.add_subparsers(help="operation to perform", dest="operation", required=True)\n \n@@ -39,7 +41,7 @@ resubmit = subparsers.add_parser("resubmit", help="If there are failed jobs, pre\n \n args = parser.parse_args()\n if args.pdf_members:\n-    args.pdf_members = [int(p) if p.isnumeric() else p for p in args.pdf_members]\n+    args.pdf_members = [int(p) if p.isnumeric() and not isinstance(p, int) else p for p in args.pdf_members]\n \n logging.basicConfig(level=logging.INFO)\n logger = logging.getLogger("scetlib-manage-condor")\n@@ -55,8 +57,8 @@ def write_filled_template(template, out_file_name, template_dict):\n         outFile.write(filled)\n \n def prepare_submit_folder(folder, runcard, variations_file):\n-    if "/afs" not in folder[:4]:\n-        raise ValueError(f"Submit folder must be on afs! Found {folder} instead")\n+    # if "/afs" not in folder[:4]:\n+    #     raise ValueError(f"Submit folder must be on afs! Found {folder} instead")\n \n     if not os.path.isdir(folder):\n         os.mkdir(folder)\n@@ -161,24 +163,48 @@ def find_failed_jobs(runcard, submitdir, bins_per_job, pdfs):\n \n     return resubmit,pkl_files\n \n-def combine_pkl_files(comb, other, pdfs):\n+def combine_pkl_files(comb, other, pdfs, make_vars_per_pdf=False):\n+    # print("in combine_pkl_files(), pdfs:")\n+    # print(pdfs)\n     if comb["hist"] is None:\n         info = copy.deepcopy(other) \n+        # print("info:")\n+        # print(info)\n         if pdfs is not None:\n-            pdf_list = range(pdfs[0]) if len(pdfs) == 1 else pdfs\n-            axes = info["hist"].axes\n-            if not axes.name[-1] == "vars":\n-                raise ValueError("Connot combine ill-formed histogram. Expected variations axis in position -1" \\\n-                        " found axis names {axes.name}")\n-            var_ax = hist.axis.StrCategory([f"pdf{i}" for i in pdf_list], name="vars")\n-            info["hist"] = hist.Hist(*axes[:-1], var_ax, info["hist"]._storage_type())\n-            comb = info\n+            if make_vars_per_pdf:\n+                # pdf_list = range(pdfs[0]) if len(pdfs) == 1 else pdfs\n+                # info_per_pdf = {i: copy.deepcopy(info) for i in pdf_list}\n+                # # print("info_per_pdf:")\n+                # # print(info_per_pdf)\n+                # # axes = info["hist"].axes\n+                # # if not axes.name[-1] == "vars":\n+                # #     raise ValueError("Connot combine ill-formed histogram. Expected variations axis in position -1" \\\n+                # #             " found axis names {axes.name}")\n+                # # var_ax = hist.axis.StrCategory([f"pdf{i}" for i in pdf_list], name="vars")\n+                # # info["hist"] = hist.Hist(*axes[:-1], var_ax, info["hist"]._storage_type())\n+                # # comb = info\n+                # return info_per_pdf\n+                return info\n+            else:\n+                pdf_list = range(pdfs[0]) if len(pdfs) == 1 else pdfs\n+                axes = info["hist"].axes\n+                if not axes.name[-1] == "vars":\n+                    raise ValueError("Connot combine ill-formed histogram. Expected variations axis in position -1" \\\n+                            " found axis names {axes.name}")\n+                var_ax = hist.axis.StrCategory([f"pdf{i}" for i in pdf_list], name="vars")\n+                info["hist"] = hist.Hist(*axes[:-1], var_ax, info["hist"]._storage_type())\n+                comb = info\n         else:\n             return info\n \n     combh = comb["hist"]\n     otherh = other["hist"]\n     pdf_set = None\n+\n+    # print("combh.axes:")\n+    # print(combh.axes)\n+    # print("otherh.axes:")\n+    # print(otherh.axes)\n     \n     if combh.shape == otherh.shape:\n         combh += otherh\n@@ -218,7 +244,7 @@ def combine_pkl_files(comb, other, pdfs):\n     return comb\n \n \n-def combine_jobs(pkl_files, npdfs, skip_missing, outdir, submitdir):\n+def combine_jobs(pkl_files, npdfs, skip_missing, outdir, submitdir, make_vars_per_pdf):\n     comb_res = {\n         "bins" : None,\n         "hist" : None,\n@@ -226,11 +252,19 @@ def combine_jobs(pkl_files, npdfs, skip_missing, outdir, submitdir):\n         "config" : None,\n         "meta_data" : None,\n     }\n+    if make_vars_per_pdf:\n+        comb_res = {i: copy.deepcopy(comb_res) for i in npdfs}\n \n     if not pkl_files:\n         raise ValueError("No output files found!")\n \n+    idx = 0\n+    print(f"--> Going to merge {len(pkl_files)} pkl files")\n     for filename in pkl_files:\n+        # if idx > 1:\n+        #     raise ValueError("sss")\n+        if idx%5 == 0:\n+            print(f"--> Now at pkl file no. {idx}")\n         if not os.path.isfile(filename):\n             if skip_missing:\n                 logger.warning(f"Did not find file {filename}, most likely due to a failed job. Skipping it!")\n@@ -239,21 +273,55 @@ def combine_jobs(pkl_files, npdfs, skip_missing, outdir, submitdir):\n                 raise ValueError(f"Did not find file {filename}, most likely due to a failed job. Either run with --skip-missing or resubmit")\n \n         f = pickle.load(open(filename, "rb"))\n-        comb_res = combine_pkl_files(comb_res, f, npdfs)\n+        # print("filename for this pkl:")\n+        # print(filename)\n+        # print("current comb_res:")\n+        # print(comb_res)\n+        if make_vars_per_pdf:\n+            pdf_thisfile = [i for i in npdfs if i in filename]\n+            if len(pdf_thisfile) != 1:\n+                raise ValueError(f"Trying to combine files with one file per pdf, but could not unambiguously infer which PDF was used in this file. Filename: {filename}, matches multiple strings in pdf list: {pdf_thisfile}")\n+            pdf_thisfile = pdf_thisfile[0]\n+            # print("pdf_thisfile:")\n+            # print(pdf_thisfile)\n+            # print(f"adding hist f to combined result for pdf {pdf_thisfile}")\n+            # print(f"hist f to be added:")\n+            # print(f)\n+            comb_res[pdf_thisfile] = combine_pkl_files(comb_res[pdf_thisfile], f, npdfs, make_vars_per_pdf=make_vars_per_pdf)\n+            # print("\\n\\n comb_res after adding:")\n+            # print(comb_res)\n+        else:\n+            comb_res = combine_pkl_files(comb_res, f, npdfs, make_vars_per_pdf=make_vars_per_pdf)\n+        idx += 1 \n \n     condor_submit = os.path.join(submitdir, "condor.submit")\n     if os.path.isfile(condor_submit):\n         with open(condor_submit, "r") as condorf:\n-            comb_res["meta_data"]["condor_submit"] = condorf.read()\n+            if make_vars_per_pdf:\n+                for pdfname in npdfs:\n+                    comb_res[pdfname]["meta_data"]["condor_submit"] = condorf.read()\n+            else:\n+                comb_res["meta_data"]["condor_submit"] = condorf.read()\n \n     comb_name = re.sub("(pdf\\d+_)?bins*_\\d+(_\\d+)?(_var?s*_\\d+)?", "combined", filename)\n-    if npdfs and len(npdfs) > 1 and not npdfs[0].isnumeric():\n+    if npdfs and len(npdfs) > 1 and not (isinstance(npdfs[0], int) or npdfs[0].isnumeric()):\n         comb_name = comb_name.replace(npdfs[-1], "")\n \n-    outfile = os.path.join(outdir, os.path.basename(comb_name))\n-    with open(outfile, "wb") as f:\n-        pickle.dump(comb_res, f)\n-    logger.info(f"Wrote file {outfile}")\n+    if make_vars_per_pdf:\n+        for pdfname in npdfs:\n+            outfile = os.path.join(outdir, os.path.basename(comb_name.replace("combined.pkl", f"{pdfname}_combined.pkl")))\n+            with open(outfile, "wb") as f:\n+                pickle.dump(comb_res[pdfname], f)\n+            logger.info(f"Wrote file {outfile}")\n+\n+\n+\n+\n+    else:\n+        outfile = os.path.join(outdir, os.path.basename(comb_name))\n+        with open(outfile, "wb") as f:\n+            pickle.dump(comb_res, f)\n+        logger.info(f"Wrote file {outfile}")\n         \n \n def submit_condor(runcard, submitdir, bins_per_job, ncpu, pdf, alphas, queue, scetlib_afs_path):\n@@ -284,10 +352,12 @@ def submit_condor(runcard, submitdir, bins_per_job, ncpu, pdf, alphas, queue, sc\n         nvars=nvars,\n         scetlib_dir=scetlib_afs_path,\n         nbins_per_job=bins_per_job,\n-        runcard=runcard,\n-        transfer_files=\',\'.join([runcard, "base.conf"]+([variations_filename] if os.path.isfile(variations_filename) else [])),\n+        runcard=os.path.basename(runcard),\n+        transfer_files=\',\'.join([os.path.basename(runcard), "base.conf"]+([os.path.basename(variations_filename)] if os.path.isfile(variations_filename) else [])),\n         ncpu=ncpu,\n         npdf=npdf,\n+        pdfset=args.pdf_set,\n+        pdfset_arg="--pdf-set $PdfSet" if args.pdf_set else "",\n         queue=queue,\n         queue_command=queue_command,\n         alphas_arg="--alphas $(Alphas)" if alphas else "",\n@@ -323,7 +393,7 @@ elif args.operation == \'combine\':\n     resubmit,pkl_files = find_failed_jobs(args.runcard, args.submitdir, args.bins_per_job, args.pdf_members)\n     if resubmit and not args.skip_missing:\n         raise ValueError("Found failed jobs. To continue and ignore failed jobs, rerun with --skip-missing")\n-    combine_jobs(pkl_files, args.pdf_members, args.skip_missing, args.outdir, args.submitdir)\n+    combine_jobs(pkl_files, args.pdf_members, args.skip_missing, args.outdir, args.submitdir, args.make_vars_per_pdf)\n elif args.operation == \'resubmit\':\n     resubmit,pkl_files = find_failed_jobs(args.runcard, args.submitdir, args.bins_per_job, args.pdf_members)\n     resubmit_failed_jobs(resubmit, os.path.join(args.submitdir, "condor.submit"), args.queue)\ndiff --git a/prod/scetlib_run/wrap-scetlib-run-qT.sh b/prod/scetlib_run/wrap-scetlib-run-qT.sh\nindex aaffb50..dfe50a5 100755\n--- a/prod/scetlib_run/wrap-scetlib-run-qT.sh\n+++ b/prod/scetlib_run/wrap-scetlib-run-qT.sh\n@@ -2,8 +2,10 @@\n SCETLIB_DIR=$1\n \n args=${@:2}\n-work_dir=$(dirname $SCETLIB_DIR)\n+# work_dir=$(basename $SCETLIB_DIR)\n+work_dir=$SCETLIB_DIR\n export SINGULARITY_BIND="$work_dir,/cvmfs"\n+echo "SINGULARITY_BIND: ${SINGULARITY_BIND}"\n echo "Running /opt/env.sh; python3 ${SCETLIB_DIR}/prod/scetlib_run/scetlib-run-qT.py $args"\n singularity exec /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/bendavid/cmswmassdocker/wmassdevrolling\\:v15 bash \\\n     -c "source /opt/env.sh; python3 ${SCETLIB_DIR}/prod/scetlib_run//scetlib-run-qT.py $args"\n', 'condor_submit': '# Submit auto-generated with command /eos/home-a/areimers/scetlib-cms-newnp/prod/scetlib_run/scetlib-manage-condor.py -s /eos/home-a/areimers/scetlib-cms-newnp -r /home/a/areimers/wmass/TheoryCorrections/SCETlib/com5p02_ct18z_newnps_n3+0ll_lattice_pdfvars/inclusive_Z_COM5p02_CT18Z_N3+0LL_lattice_allvars_higherprecision_pdfas.ini -b 100 -j 8 -f /eos/home-a/areimers/workdir_scetlib/Z_COM5p02_CT18Z_N3p0LL_NewNPs_Lattice_AllVars_HigherPrecision_pdfas --pdf-members CT18ZNNLO_as_0118 CT18ZNNLO_as_0116 CT18ZNNLO_as_0120 --alphas 0.118 0.116 0.120 -q workday submit\nUniverse             = vanilla\nExecutable           = /eos/home-a/areimers/scetlib-cms-newnp/prod/scetlib_run/wrap-scetlib-run-qT.sh\nGetEnv               = false\n\nNVARS = 1\nNBINS = 3200\nNBinsPerJob = 100\nINPUT = inclusive_Z_COM5p02_CT18Z_N3+0LL_lattice_allvars_higherprecision_pdfas.ini\nNCPU = 8\nNPDF = 1\n\nNBinSteps = $(NBINS)/$(NBinsPerJob)\nNProcesses = $(NVARS)*$(NBinSteps)*$(NPDF)\n\ninitialdir = /eos/home-a/areimers/workdir_scetlib/Z_COM5p02_CT18Z_N3p0LL_NewNPs_Lattice_AllVars_HigherPrecision_pdfas\ntransfer_input_files = inclusive_Z_COM5p02_CT18Z_N3+0LL_lattice_allvars_higherprecision_pdfas.ini,base.conf\nrequest_memory       = 4000\nrequest_disk         = 2048000\nrequest_cpus         = $(NCPU)\n+AccountingGroup = "group_u_CMST3.all"\n+JobFlavour = "workday"\n\nBinNum = ($(Process) % ($(NBinSteps)))*$(NBinsPerJob)\nVarNum = $(Process) / ($(NBinSteps)) % $(NVARS)\nPdfNum = $$([$(Process) / ($(NBinSteps)*$(NVARS)) % $(NPDF)])\n\nArguments            = "/eos/home-a/areimers/scetlib-cms-newnp $(NCPU) 1 $(INPUT) --no-datfiles --start-bin $$([$(BinNum)]) --stop-bin $$([$(BinNum)+$(NBinsPerJob)]) --fixed-var $$([$(VarNum)]) --pdf-member $(PdfNum) --alphas $(Alphas)"\n\noutput               = condor_out/Job$(Process).out\nerror                = condor_out/Job$(Process).err\nLog                  = condor_out/AllLogs.log\n\nQueue $(NProcesses) PdfNum,AlphaS from (\n    CT18ZNNLO_as_0118,0.118\n    CT18ZNNLO_as_0116,0.116\n    CT18ZNNLO_as_0120,0.12\n)\n', 'config': {'Calculation_settings': {'calculation_piece': 'sing', 'lambda': '0.', 'b0_over_bmax': '0.', 'b0_over_bmax_global': '0.', 'mu0_min': '1.', 'mub_min': '1.', 'mus_min': '1.', 'nus_min': '0.', 'muf_min': '1.40', 'transition_points': '[0.2, 0.6, 1.0]', 'transition_type': 'slope', 'scale_setting': 'spectrum', 'weight_qt': 'none', 'mufo_fixed': '0.', 'kappafo': '1.', 'kappaf': '1.', 'phase_muh': '1.', 'muf_follows_mub': 'no', 'compensate_fo': 'yes', 'form_np_prescription': 'collins_soper4', 'scheme_radish': 'no', 'recoil_scheme': 'collins_soper', 'disable_asymmetry': 'yes', 'variations_filename': 'variations.conf', 'alphas_solution': 'analytic', 'alphas_rel_precision': '1.e-6', 'rge_solution': 'analytic', 'profile_functional_form': 'slope', 'muf_max': '5020.', 'fixed_order': 'nnlo', 'run_order': 'n3ll'}, 'Singlet_scheme': {'nonsinglet_use_exact_top_mass': 'no', 'vector_singlet_enabled': 'yes', 'vector_singlet_use_exact_top_mass': 'no', 'axial_singlet_enabled': 'yes', 'axial_singlet_use_exact_top_mass': 'no'}, 'Nonperturbative': {'b0_over_bmax_nu': '1.', 'lambda2_nu': '0.0870', 'lambda4_nu': '0.0074', 'lambda_inf_nu': '1.6853', 'np_model_nu': 'tanh_2', 'lambda2': '0.25', 'lambda4': '0.06', 'lambda_inf': '1.', 'np_model': 'tanh_2', 'delta_lambda2': '0.125', 'omega_down': '0.', 'omega_downbar': '0.', 'omega_up': '0.', 'omega_upbar': '0.', 'omega_strange': '0.', 'omega_strangebar': '0.', 'omega_charm': '0.', 'omega_charmbar': '0.', 'omega_bottom': '0.', 'omega_bottombar': '0.', 'omega_gluon': '0.', 'lambda2_down': '0.', 'lambda2_downbar': '0.', 'lambda2_up': '0.', 'lambda2_upbar': '0.', 'lambda2_strange': '0.', 'lambda2_strangebar': '0.', 'lambda2_charm': '0.', 'lambda2_charmbar': '0.', 'lambda2_bottom': '0.', 'lambda2_bottombar': '0.', 'lambda2_gluon': '0.'}, 'Integration': {'report_error_estimate': 'yes', 'target_precision_rel': '1.e-5', 'target_precision_abs': '1.e-9', 'precision_buffer_bt': '1.e-2', 'precision_buffer_decay': '1.e-1', 'tolerance_qt': '1.', 'precision_buffer_full': '1.', 'abs_precision_buffer_inner': '1.', 'change_var_q': 'arctan_Q2', 'change_var_q_q0': '91.15348061918276', 'change_var_qt': 'none', 'change_var_qt_q0': '1.'}, 'QCD': {'nf': '5', 'mu0': '91.1876', 'ecm': '5020.', 'pdf_set': 'CT18ZNNLO_as_0120', 'alphas_order': 'n3ll', 'pdf_member': '0', 'alphas_mu0': '0.118'}, 'TNPs': {'gamma_cusp': "(0., 'level0')", 'gamma_mu_q': "(0., 'level0')", 'gamma_nu': "(0., 'level0')", 'h_qqv': "(0., 'level0')", 's': "(0., 'level0')", 'b_qqv': "(0., 'relative')", 'b_qqbarv': "(0., 'relative')", 'b_qqs': "(0., 'relative')", 'b_qqds': "(0., 'relative')", 'b_qg': "(0., 'relative')"}, 'Electroweak': {'alphaem': '0.0077624494296896114', 'sin2_thw': '0.23153999447822571', 'mz': '91.153509740726733', 'gammaz': '2.4932018986110700', 'mw': '79.906853549493746', 'gammaw': '2.0904310808144846', 'ckm': '[ [0.97446, 0.222, 0.00365], [0.22438, 0.97359, 0.04214], [0.00896, 0.04133, 0.999105] ]'}, 'Process': {'boson': 'Z'}, 'Grid_Q': {'min': '60.', 'max': '120.', 'steps': '1', 'bins': 'true'}, 'Grid_Y': {'custom_grid': 'true', 'bins': 'true', 'values': '[-5.0, -4.0, -3.5, -3.25, -3, -2.75, -2.5, -2.25, -2., -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 4.0, 5.0]'}, 'Grid_qT': {'min': '0', 'max': '100', 'steps': '100', 'bins': 'true'}}}, 'inclusive_Z_COM5p02_CT18Z_N3+0LL_lattice_higherprecision_pdfas_nnlo_sing_pdf_combined.pkl': {'time': '2026-01-07 13:25:01.249389', 'command': '/eos/home-a/areimers/scetlib-cms-newnp/prod/scetlib_run//scetlib-run-qT.py 8 1 inclusive_Z_COM5p02_CT18Z_N3+0LL_lattice_higherprecision_pdfas_nnlo_sing.ini --no-datfiles --start-bin 0 --stop-bin 100 --fixed-var 0 --pdf-member CT18ZNNLO_as_0118 --alphas 0.118', 'git_hash': '"205a087c79fe996466df2f9c4b2055653027dd6e"\n', 'git_diff': 'diff --git a/prod/scetlib_run/condor_template b/prod/scetlib_run/condor_template\nindex 4cbce39..173fa13 100644\n--- a/prod/scetlib_run/condor_template\n+++ b/prod/scetlib_run/condor_template\n@@ -14,21 +14,20 @@ NProcesses = $$(NVARS)*$$(NBinSteps)*$$(NPDF)\n \n initialdir = ${submitdir}\n transfer_input_files = ${transfer_files}\n-request_memory       = 2000*$$(NCPU)\n+request_memory       = 4000\n request_disk         = 2048000\n request_cpus         = $$(NCPU)\n-requirements = (OpSysAndVer =?= "CentOS7")\n +AccountingGroup = "group_u_CMST3.all"\n +JobFlavour = "${queue}"\n \n BinNum = ($$(Process) % ($$(NBinSteps)))*$$(NBinsPerJob)\n VarNum = $$(Process) / ($$(NBinSteps)) % $$(NVARS)\n-PdfNum = $$(Process) / ($$(NBinSteps)*$$(NVARS)) % $$(NPDF)\n+PdfNum = $$$$([$$(Process) / ($$(NBinSteps)*$$(NVARS)) % $$(NPDF)])\n+\n+Arguments            = "${scetlib_dir} $$(NCPU) 1 $$(INPUT) --no-datfiles --start-bin $$$$([$$(BinNum)]) --stop-bin $$$$([$$(BinNum)+$$(NBinsPerJob)]) --fixed-var $$$$([$$(VarNum)]) --pdf-member $$(PdfNum) ${alphas_arg}"\n \n-Arguments            = "${scetlib_dir}/prod/scetlib_run $$(NCPU) 1 $$(INPUT) --no-datfiles --start-bin $$$$([$$(BinNum)]) --stop-bin $$$$([$$(BinNum)+$$(NBinsPerJob)]) --fixed-var $$$$([$$(VarNum)]) --pdf-member $$$$([$$(PdfNum)])" ${pdf_set}\n output               = condor_out/Job$$(Process).out\n error                = condor_out/Job$$(Process).err\n-Log                  = condor_out/Job$$(Process).log\n-\n-Queue $$(NProcesses)\n+Log                  = condor_out/AllLogs.log\n \n+Queue $$(NProcesses) ${queue_command}\ndiff --git a/prod/scetlib_run/scetlib-manage-condor.py b/prod/scetlib_run/scetlib-manage-condor.py\nindex 5d55e9e..199c8be 100644\n--- a/prod/scetlib_run/scetlib-manage-condor.py\n+++ b/prod/scetlib_run/scetlib-manage-condor.py\n@@ -22,11 +22,13 @@ parser.add_argument("-r", "--runcard", type=str, help="input (.ini) file", requi\n parser.add_argument("-b", "--bins-per-job", type=int, default=1)\n parser.add_argument("-j", "--ncpu", type=int, default=1)\n parser.add_argument("-p", "--pdf-members", type=str, nargs="*", default=None)\n+parser.add_argument("--pdf-set", type=str, default=None)\n parser.add_argument("-a", "--alphas", type=float, nargs="*", default=None)\n parser.add_argument("-f", "--submitdir", type=str, default=".")\n parser.add_argument("-s", "--scetlib-afs-path", type=str, default=f"/afs/cern.ch/work/{os.getlogin()[0]}/{os.getlogin()}/scetlib-cms")\n parser.add_argument("-q", "--queue", type=str, choices=["longlunch", "espresso", "workday", "tomorrow", "testmatch", "nextweek"], \n     help="Condor runtime queue to use", default="workday")\n+parser.add_argument("--make-vars-per-pdf", action=\'store_true\', help="in combine step (and only there), write out one file per pdfas variation, which contains all variations handled internally via .conf file.")\n \n subparsers = parser.add_subparsers(help="operation to perform", dest="operation", required=True)\n \n@@ -39,7 +41,7 @@ resubmit = subparsers.add_parser("resubmit", help="If there are failed jobs, pre\n \n args = parser.parse_args()\n if args.pdf_members:\n-    args.pdf_members = [int(p) if p.isnumeric() else p for p in args.pdf_members]\n+    args.pdf_members = [int(p) if p.isnumeric() and not isinstance(p, int) else p for p in args.pdf_members]\n \n logging.basicConfig(level=logging.INFO)\n logger = logging.getLogger("scetlib-manage-condor")\n@@ -55,8 +57,8 @@ def write_filled_template(template, out_file_name, template_dict):\n         outFile.write(filled)\n \n def prepare_submit_folder(folder, runcard, variations_file):\n-    if "/afs" not in folder[:4]:\n-        raise ValueError(f"Submit folder must be on afs! Found {folder} instead")\n+    # if "/afs" not in folder[:4]:\n+    #     raise ValueError(f"Submit folder must be on afs! Found {folder} instead")\n \n     if not os.path.isdir(folder):\n         os.mkdir(folder)\n@@ -161,24 +163,48 @@ def find_failed_jobs(runcard, submitdir, bins_per_job, pdfs):\n \n     return resubmit,pkl_files\n \n-def combine_pkl_files(comb, other, pdfs):\n+def combine_pkl_files(comb, other, pdfs, make_vars_per_pdf=False):\n+    # print("in combine_pkl_files(), pdfs:")\n+    # print(pdfs)\n     if comb["hist"] is None:\n         info = copy.deepcopy(other) \n+        # print("info:")\n+        # print(info)\n         if pdfs is not None:\n-            pdf_list = range(pdfs[0]) if len(pdfs) == 1 else pdfs\n-            axes = info["hist"].axes\n-            if not axes.name[-1] == "vars":\n-                raise ValueError("Connot combine ill-formed histogram. Expected variations axis in position -1" \\\n-                        " found axis names {axes.name}")\n-            var_ax = hist.axis.StrCategory([f"pdf{i}" for i in pdf_list], name="vars")\n-            info["hist"] = hist.Hist(*axes[:-1], var_ax, info["hist"]._storage_type())\n-            comb = info\n+            if make_vars_per_pdf:\n+                # pdf_list = range(pdfs[0]) if len(pdfs) == 1 else pdfs\n+                # info_per_pdf = {i: copy.deepcopy(info) for i in pdf_list}\n+                # # print("info_per_pdf:")\n+                # # print(info_per_pdf)\n+                # # axes = info["hist"].axes\n+                # # if not axes.name[-1] == "vars":\n+                # #     raise ValueError("Connot combine ill-formed histogram. Expected variations axis in position -1" \\\n+                # #             " found axis names {axes.name}")\n+                # # var_ax = hist.axis.StrCategory([f"pdf{i}" for i in pdf_list], name="vars")\n+                # # info["hist"] = hist.Hist(*axes[:-1], var_ax, info["hist"]._storage_type())\n+                # # comb = info\n+                # return info_per_pdf\n+                return info\n+            else:\n+                pdf_list = range(pdfs[0]) if len(pdfs) == 1 else pdfs\n+                axes = info["hist"].axes\n+                if not axes.name[-1] == "vars":\n+                    raise ValueError("Connot combine ill-formed histogram. Expected variations axis in position -1" \\\n+                            " found axis names {axes.name}")\n+                var_ax = hist.axis.StrCategory([f"pdf{i}" for i in pdf_list], name="vars")\n+                info["hist"] = hist.Hist(*axes[:-1], var_ax, info["hist"]._storage_type())\n+                comb = info\n         else:\n             return info\n \n     combh = comb["hist"]\n     otherh = other["hist"]\n     pdf_set = None\n+\n+    # print("combh.axes:")\n+    # print(combh.axes)\n+    # print("otherh.axes:")\n+    # print(otherh.axes)\n     \n     if combh.shape == otherh.shape:\n         combh += otherh\n@@ -218,7 +244,7 @@ def combine_pkl_files(comb, other, pdfs):\n     return comb\n \n \n-def combine_jobs(pkl_files, npdfs, skip_missing, outdir, submitdir):\n+def combine_jobs(pkl_files, npdfs, skip_missing, outdir, submitdir, make_vars_per_pdf):\n     comb_res = {\n         "bins" : None,\n         "hist" : None,\n@@ -226,11 +252,19 @@ def combine_jobs(pkl_files, npdfs, skip_missing, outdir, submitdir):\n         "config" : None,\n         "meta_data" : None,\n     }\n+    if make_vars_per_pdf:\n+        comb_res = {i: copy.deepcopy(comb_res) for i in npdfs}\n \n     if not pkl_files:\n         raise ValueError("No output files found!")\n \n+    idx = 0\n+    print(f"--> Going to merge {len(pkl_files)} pkl files")\n     for filename in pkl_files:\n+        # if idx > 1:\n+        #     raise ValueError("sss")\n+        if idx%5 == 0:\n+            print(f"--> Now at pkl file no. {idx}")\n         if not os.path.isfile(filename):\n             if skip_missing:\n                 logger.warning(f"Did not find file {filename}, most likely due to a failed job. Skipping it!")\n@@ -239,21 +273,55 @@ def combine_jobs(pkl_files, npdfs, skip_missing, outdir, submitdir):\n                 raise ValueError(f"Did not find file {filename}, most likely due to a failed job. Either run with --skip-missing or resubmit")\n \n         f = pickle.load(open(filename, "rb"))\n-        comb_res = combine_pkl_files(comb_res, f, npdfs)\n+        # print("filename for this pkl:")\n+        # print(filename)\n+        # print("current comb_res:")\n+        # print(comb_res)\n+        if make_vars_per_pdf:\n+            pdf_thisfile = [i for i in npdfs if i in filename]\n+            if len(pdf_thisfile) != 1:\n+                raise ValueError(f"Trying to combine files with one file per pdf, but could not unambiguously infer which PDF was used in this file. Filename: {filename}, matches multiple strings in pdf list: {pdf_thisfile}")\n+            pdf_thisfile = pdf_thisfile[0]\n+            # print("pdf_thisfile:")\n+            # print(pdf_thisfile)\n+            # print(f"adding hist f to combined result for pdf {pdf_thisfile}")\n+            # print(f"hist f to be added:")\n+            # print(f)\n+            comb_res[pdf_thisfile] = combine_pkl_files(comb_res[pdf_thisfile], f, npdfs, make_vars_per_pdf=make_vars_per_pdf)\n+            # print("\\n\\n comb_res after adding:")\n+            # print(comb_res)\n+        else:\n+            comb_res = combine_pkl_files(comb_res, f, npdfs, make_vars_per_pdf=make_vars_per_pdf)\n+        idx += 1 \n \n     condor_submit = os.path.join(submitdir, "condor.submit")\n     if os.path.isfile(condor_submit):\n         with open(condor_submit, "r") as condorf:\n-            comb_res["meta_data"]["condor_submit"] = condorf.read()\n+            if make_vars_per_pdf:\n+                for pdfname in npdfs:\n+                    comb_res[pdfname]["meta_data"]["condor_submit"] = condorf.read()\n+            else:\n+                comb_res["meta_data"]["condor_submit"] = condorf.read()\n \n     comb_name = re.sub("(pdf\\d+_)?bins*_\\d+(_\\d+)?(_var?s*_\\d+)?", "combined", filename)\n-    if npdfs and len(npdfs) > 1 and not npdfs[0].isnumeric():\n+    if npdfs and len(npdfs) > 1 and not (isinstance(npdfs[0], int) or npdfs[0].isnumeric()):\n         comb_name = comb_name.replace(npdfs[-1], "")\n \n-    outfile = os.path.join(outdir, os.path.basename(comb_name))\n-    with open(outfile, "wb") as f:\n-        pickle.dump(comb_res, f)\n-    logger.info(f"Wrote file {outfile}")\n+    if make_vars_per_pdf:\n+        for pdfname in npdfs:\n+            outfile = os.path.join(outdir, os.path.basename(comb_name.replace("combined.pkl", f"{pdfname}_combined.pkl")))\n+            with open(outfile, "wb") as f:\n+                pickle.dump(comb_res[pdfname], f)\n+            logger.info(f"Wrote file {outfile}")\n+\n+\n+\n+\n+    else:\n+        outfile = os.path.join(outdir, os.path.basename(comb_name))\n+        with open(outfile, "wb") as f:\n+            pickle.dump(comb_res, f)\n+        logger.info(f"Wrote file {outfile}")\n         \n \n def submit_condor(runcard, submitdir, bins_per_job, ncpu, pdf, alphas, queue, scetlib_afs_path):\n@@ -284,10 +352,12 @@ def submit_condor(runcard, submitdir, bins_per_job, ncpu, pdf, alphas, queue, sc\n         nvars=nvars,\n         scetlib_dir=scetlib_afs_path,\n         nbins_per_job=bins_per_job,\n-        runcard=runcard,\n-        transfer_files=\',\'.join([runcard, "base.conf"]+([variations_filename] if os.path.isfile(variations_filename) else [])),\n+        runcard=os.path.basename(runcard),\n+        transfer_files=\',\'.join([os.path.basename(runcard), "base.conf"]+([os.path.basename(variations_filename)] if os.path.isfile(variations_filename) else [])),\n         ncpu=ncpu,\n         npdf=npdf,\n+        pdfset=args.pdf_set,\n+        pdfset_arg="--pdf-set $PdfSet" if args.pdf_set else "",\n         queue=queue,\n         queue_command=queue_command,\n         alphas_arg="--alphas $(Alphas)" if alphas else "",\n@@ -323,7 +393,7 @@ elif args.operation == \'combine\':\n     resubmit,pkl_files = find_failed_jobs(args.runcard, args.submitdir, args.bins_per_job, args.pdf_members)\n     if resubmit and not args.skip_missing:\n         raise ValueError("Found failed jobs. To continue and ignore failed jobs, rerun with --skip-missing")\n-    combine_jobs(pkl_files, args.pdf_members, args.skip_missing, args.outdir, args.submitdir)\n+    combine_jobs(pkl_files, args.pdf_members, args.skip_missing, args.outdir, args.submitdir, args.make_vars_per_pdf)\n elif args.operation == \'resubmit\':\n     resubmit,pkl_files = find_failed_jobs(args.runcard, args.submitdir, args.bins_per_job, args.pdf_members)\n     resubmit_failed_jobs(resubmit, os.path.join(args.submitdir, "condor.submit"), args.queue)\ndiff --git a/prod/scetlib_run/wrap-scetlib-run-qT.sh b/prod/scetlib_run/wrap-scetlib-run-qT.sh\nindex aaffb50..dfe50a5 100755\n--- a/prod/scetlib_run/wrap-scetlib-run-qT.sh\n+++ b/prod/scetlib_run/wrap-scetlib-run-qT.sh\n@@ -2,8 +2,10 @@\n SCETLIB_DIR=$1\n \n args=${@:2}\n-work_dir=$(dirname $SCETLIB_DIR)\n+# work_dir=$(basename $SCETLIB_DIR)\n+work_dir=$SCETLIB_DIR\n export SINGULARITY_BIND="$work_dir,/cvmfs"\n+echo "SINGULARITY_BIND: ${SINGULARITY_BIND}"\n echo "Running /opt/env.sh; python3 ${SCETLIB_DIR}/prod/scetlib_run/scetlib-run-qT.py $args"\n singularity exec /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/bendavid/cmswmassdocker/wmassdevrolling\\:v15 bash \\\n     -c "source /opt/env.sh; python3 ${SCETLIB_DIR}/prod/scetlib_run//scetlib-run-qT.py $args"\n', 'condor_submit': '# Submit auto-generated with command /eos/home-a/areimers/scetlib-cms-newnp/prod/scetlib_run/scetlib-manage-condor.py -s /eos/home-a/areimers/scetlib-cms-newnp -r /home/a/areimers/wmass/TheoryCorrections/SCETlib/com5p02_ct18z_newnps_n3+0ll_lattice_pdfvars_nnlo_sing/inclusive_Z_COM5p02_CT18Z_N3+0LL_lattice_higherprecision_pdfas_nnlo_sing.ini -b 100 -j 8 -f /eos/home-a/areimers/workdir_scetlib/Z_COM5p02_CT18Z_N3p0LL_NewNPs_Lattice_HigherPrecision_pdfas_NNLO_Singular --pdf-members CT18ZNNLO_as_0118 CT18ZNNLO_as_0116 CT18ZNNLO_as_0120 --alphas 0.118 0.116 0.120 -q workday submit\nUniverse             = vanilla\nExecutable           = /eos/home-a/areimers/scetlib-cms-newnp/prod/scetlib_run/wrap-scetlib-run-qT.sh\nGetEnv               = false\n\nNVARS = 1\nNBINS = 3200\nNBinsPerJob = 100\nINPUT = inclusive_Z_COM5p02_CT18Z_N3+0LL_lattice_higherprecision_pdfas_nnlo_sing.ini\nNCPU = 8\nNPDF = 1\n\nNBinSteps = $(NBINS)/$(NBinsPerJob)\nNProcesses = $(NVARS)*$(NBinSteps)*$(NPDF)\n\ninitialdir = /eos/home-a/areimers/workdir_scetlib/Z_COM5p02_CT18Z_N3p0LL_NewNPs_Lattice_HigherPrecision_pdfas_NNLO_Singular\ntransfer_input_files = inclusive_Z_COM5p02_CT18Z_N3+0LL_lattice_higherprecision_pdfas_nnlo_sing.ini,base.conf\nrequest_memory       = 4000\nrequest_disk         = 2048000\nrequest_cpus         = $(NCPU)\n+AccountingGroup = "group_u_CMST3.all"\n+JobFlavour = "workday"\n\nBinNum = ($(Process) % ($(NBinSteps)))*$(NBinsPerJob)\nVarNum = $(Process) / ($(NBinSteps)) % $(NVARS)\nPdfNum = $$([$(Process) / ($(NBinSteps)*$(NVARS)) % $(NPDF)])\n\nArguments            = "/eos/home-a/areimers/scetlib-cms-newnp $(NCPU) 1 $(INPUT) --no-datfiles --start-bin $$([$(BinNum)]) --stop-bin $$([$(BinNum)+$(NBinsPerJob)]) --fixed-var $$([$(VarNum)]) --pdf-member $(PdfNum) --alphas $(Alphas)"\n\noutput               = condor_out/Job$(Process).out\nerror                = condor_out/Job$(Process).err\nLog                  = condor_out/AllLogs.log\n\nQueue $(NProcesses) PdfNum,AlphaS from (\n    CT18ZNNLO_as_0118,0.118\n    CT18ZNNLO_as_0116,0.116\n    CT18ZNNLO_as_0120,0.12\n)\n', 'config': {'Calculation_settings': {'calculation_piece': 'sing', 'lambda': '0.', 'b0_over_bmax': '0.', 'b0_over_bmax_global': '0.', 'mu0_min': '1.', 'mub_min': '1.', 'mus_min': '1.', 'nus_min': '0.', 'muf_min': '1.40', 'transition_points': '[0.2, 0.6, 1.0]', 'transition_type': 'slope', 'scale_setting': 'spectrum', 'weight_qt': 'none', 'mufo_fixed': '0.', 'kappafo': '1.', 'kappaf': '1.', 'phase_muh': '1.', 'muf_follows_mub': 'no', 'compensate_fo': 'yes', 'form_np_prescription': 'collins_soper4', 'scheme_radish': 'no', 'recoil_scheme': 'collins_soper', 'disable_asymmetry': 'yes', 'variations_filename': 'variations.conf', 'alphas_solution': 'analytic', 'alphas_rel_precision': '1.e-6', 'rge_solution': 'analytic', 'profile_functional_form': 'slope', 'muf_max': '5020.', 'fixed_order': 'nnlo', 'run_order': 'none'}, 'Singlet_scheme': {'nonsinglet_use_exact_top_mass': 'no', 'vector_singlet_enabled': 'yes', 'vector_singlet_use_exact_top_mass': 'no', 'axial_singlet_enabled': 'yes', 'axial_singlet_use_exact_top_mass': 'no'}, 'Nonperturbative': {'b0_over_bmax_nu': '1.', 'lambda2_nu': '0.0870', 'lambda4_nu': '0.0074', 'lambda_inf_nu': '1.6853', 'np_model_nu': 'tanh_2', 'lambda2': '0.25', 'lambda4': '0.06', 'lambda_inf': '1.', 'np_model': 'tanh_2', 'delta_lambda2': '0.125', 'omega_down': '0.', 'omega_downbar': '0.', 'omega_up': '0.', 'omega_upbar': '0.', 'omega_strange': '0.', 'omega_strangebar': '0.', 'omega_charm': '0.', 'omega_charmbar': '0.', 'omega_bottom': '0.', 'omega_bottombar': '0.', 'omega_gluon': '0.', 'lambda2_down': '0.', 'lambda2_downbar': '0.', 'lambda2_up': '0.', 'lambda2_upbar': '0.', 'lambda2_strange': '0.', 'lambda2_strangebar': '0.', 'lambda2_charm': '0.', 'lambda2_charmbar': '0.', 'lambda2_bottom': '0.', 'lambda2_bottombar': '0.', 'lambda2_gluon': '0.'}, 'Integration': {'report_error_estimate': 'yes', 'target_precision_rel': '1.e-5', 'target_precision_abs': '1.e-9', 'precision_buffer_bt': '1.e-2', 'precision_buffer_decay': '1.e-1', 'tolerance_qt': '1.', 'precision_buffer_full': '1.', 'abs_precision_buffer_inner': '1.', 'change_var_q': 'arctan_Q2', 'change_var_q_q0': '91.15348061918276', 'change_var_qt': 'none', 'change_var_qt_q0': '1.'}, 'QCD': {'nf': '5', 'mu0': '91.1876', 'ecm': '5020.', 'pdf_set': 'CT18ZNNLO_as_0120', 'alphas_order': 'n3ll', 'pdf_member': '0', 'alphas_mu0': '0.118'}, 'Electroweak': {'alphaem': '0.0077624494296896114', 'sin2_thw': '0.23153999447822571', 'mz': '91.153509740726733', 'gammaz': '2.4932018986110700', 'mw': '79.906853549493746', 'gammaw': '2.0904310808144846', 'ckm': '[ [0.97446, 0.222, 0.00365], [0.22438, 0.97359, 0.04214], [0.00896, 0.04133, 0.999105] ]'}, 'Process': {'boson': 'Z'}, 'Grid_Q': {'min': '60.', 'max': '120.', 'steps': '1', 'bins': 'true'}, 'Grid_Y': {'custom_grid': 'true', 'bins': 'true', 'values': '[-5.0, -4.0, -3.5, -3.25, -3, -2.75, -2.5, -2.25, -2., -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 4.0, 5.0]'}, 'Grid_qT': {'min': '0', 'max': '100', 'steps': '100', 'bins': 'true'}}}, 'results_z-2d-nnlo-vj-CT18ZNNLO_as-member{i}-scetlibmatch.txt': None}}
(Pdb) proc
'Z'
(Pdb) histname
'scetlib_dyturboLatticeNP_CT18Z_N3p0LL_N2LO_pdfas_minnlo_ratio'
(Pdb) 

## Setup

cd /work/submit/hayden17

APPTAINER_BIND="/tmp,/home/submit,/work/submit,/ceph/submit,/scratch/submit,/cvmfs,/etc/grid-security,/run" \
singularity run /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/bendavid/cmswmassdocker/wmassdevrolling:latest

source WRemnants/setup.sh

Singularity> cd /work/submit/hayden17/WRemnants

Singularity>    python scripts/histmakers/histmaker_test.py      --dataPath /scratch/submit/cms/wmass/NanoAOD/LowPU/2017G/      --era 2017G      --filterProcs Zmumu2017G --theoryCorr scetlib_dyturboN3p0LL_LatticeNP_pdfas  -v 4

Singularity>  python scripts/histmakers/histmaker_test.py      --dataPath /scratch/submit/cms/wmass/NanoAOD/LowPU/2017G/      --era 2017G      --filterProcs Zmumu2017G --theoryCorr scetlib_dyturboLatticeNP_CT18Z_N3p0LL_N2LO_pdfas

import pdb; pdb.set_trace()

## Making Plots

IN=histmaker_test_scetlib_dyturboLatticeNP_CT18Z_N3p0LL_N2LO_pdfasCorr.hdf5
OUT=~/public_html/
TAG=jan20
PROCS="Data Zmumu"

IN=histmaker_test_scetlib_dyturboN3p0LL_LatticeNP_pdfasCorr.hdf5
OUT=~/public_html/
TAG=jan{date}
PROCS="Data Zmumu" - or Ztautau/Other

Singularity> python scripts/plotting/makeDataMCStackPlot.py $IN   -o $OUT -f $TAG   -n nominal   --hists ptll   --rrange 0.995 1.005   --procFilters Zmumu --noData --flow none   variation --varName scetlib_dyturboLatticeNP_CT18Z_N3p0LL_N2LO_pdfasCorr  --selectAxis vars --selectEntries pdfCT18ZNNLO_as_0116  pdfCT18ZNNLO_as_0120

https://submit.mit.edu/~hayden17/jan{date}/












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
