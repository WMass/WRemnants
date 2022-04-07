import narf

# TODO: Allow filtering
def getDatasets(maxFiles=-1, filt=None):
    allProcs = [
        narf.Dataset(
            name="TTTo2L2Nu",
            filepaths=[],
            xsec=1.,
            is_data=False,
        ),
        narf.Dataset(
            name="TTToSemiLeptonic",
            filepaths=[],
            xsec=1.,
            is_data=False,
        ),
        narf.Dataset(
            name="WWTo2L2Nu",
            filepaths=[],
            xsec=1.,
            is_data=False,
        ),
        narf.Dataset(
            name="WZTo3LNu",
            filepaths=[],
            xsec=1.,
            is_data=False,
        ),
        narf.Dataset(
            name="DYmumu_MiNNLO",
            filepaths=[],
            xsec=1.,
            is_data=False,
        ),
        narf.Dataset(
            name="WplusJetsToTauNu",
            filepaths=[],
            xsec=1.,
            is_data=False,
        ),
        narf.Dataset(
            name="WminusJetsToTauNu",
            filepaths=[],
            xsec=1.,
            is_data=False,
        ),
        narf.Dataset(
            name="WminusJetsToMuNu",
            filepaths=[],
            xsec=1.,
            is_data=False,
        ),
        narf.Dataset(
            name="WplusJetsToMuNu",
            filepaths=[],
            xsec=1.,
            is_data=False,
        ),
        narf.Dataset(
            filepaths=[],
            name="singlemuon",
            xsec=1.,
            is_data=True,
        ),
    ]

    if filt:
        return list(filter(filt, allProcs))

    return allProcs
