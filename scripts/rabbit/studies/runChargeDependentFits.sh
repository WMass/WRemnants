declare -a combine_folders=("WMass_pt_eta_decorl-param-sep")
declare -a postfixes=("_none")
#declare -a postfixes=("_A-M-combined-one-group-ind")

source local_setup.sh
cd $combine_dir
cmsenv

for combine_folder in "${combine_folders[@]}"; do
    cd ${combine_files_dir}/${combine_folder}
    for postfix in "${postfixes[@]}"; do
        combineCards.py WMass_minus${postfix}.txt WMass_plus${postfix}.txt > WMass${postfix}.txt
        text2hdf5.py --X-allow-no-signal WMass${postfix}.txt
        text2hdf5.py --X-allow-no-signal WMass_plus${postfix}.txt
        text2hdf5.py --X-allow-no-signal WMass_minus${postfix}.txt
        combinetf.py WMass${postfix}.hdf5 -t -1 --doImpacts --binByBinStat -o fitresults-w_charge_inclusive${postfix}.root
        combinetf.py WMass_plus${postfix}.hdf5 -t -1 --doImpacts --binByBinStat -o fitresults-w_plus${postfix}.root
        combinetf.py WMass_minus${postfix}.hdf5 -t -1 --doImpacts --binByBinStat -o fitresults-w_minus${postfix}.root
    done
done

cd $WREM_BASE
