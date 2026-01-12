//MACRO TO MERGE ISO AND ANTIISO SF FROM DIRECT SMOOTHING AND FROM EFFICIENCY SMOOTHING (NEEDED BECAUSE, WHEN WE TAKE INTO ACCOUNT THE DEPENDENCE ON W UT, WE AVERAGE THE TNP DATA AND MC EFFICIENCIES OVER THE W UT, AND THIS IS A DIFFERENT SET OF SFs WITH RESPECT TO INTEGRATING THE 3D BINNED SFs OVER THE W UT DISTRIBUTION)

void implementfinalfiles() {
	TFile *file1=new TFile("/scratchnvme/bruschin/Newtest/WRemnants/wremnants/data/testMuonSF/allSmooth_GtoH3Dsfiso4_.root");
	TFile *file2=new TFile("/scratchnvme/bruschin/Newtest/WRemnants/wremnants/data/testMuonSF/allSmooth_GtoH3Deffiso.root");
	TFile *fileout=new TFile("/scratchnvme/bruschin/Newtest/WRemnants/wremnants/data/testMuonSF/allSmooth_GtoH3Dout.root","RECREATE");
	fileout->cd();
	for (auto&& keyAsObj : *(file1->GetListOfKeys())){
		auto key = (TKey*) keyAsObj;
		bool found=false;
		std::string string1("onlyDataVar"), string2("onlyMCVar");
		std::string name(key->GetName());
		std::size_t found1 = name.find(string1);
		std::size_t found2 = name.find(string2);
		if (std::string(key->GetClassName())==std::string("TH2D")) {
			TH2D* histo=(TH2D*)file1->Get(key->GetName());
			histo->SetName(key->GetName());
			histo->Write();
		}
		else {
			if ((found1!=std::string::npos)||(found2!=std::string::npos)) {
				TH3D* histo=(TH3D*)file2->Get(key->GetName());
				histo->SetName(key->GetName());
				histo->Write();
			}
			else {
				TH3D* histo=(TH3D*)file1->Get(key->GetName());
				histo->SetName(key->GetName());
				histo->Write();
			}
		}
	}
}
