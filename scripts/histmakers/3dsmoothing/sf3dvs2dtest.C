void sf3dvs2dtest() {
	TFile *file1=new TFile("/gpfs/ddn/cms/user/bruschin/Newtest/WRemnants/wremnants/data/testMuonSF/allSmooth_GtoH3Dsfiso2.root");
	TFile *file2=new TFile("/gpfs/ddn/cms/user/bruschin/Newtest/WRemnants/wremnants/data/testMuonSF/allSmooth_GtoHisosf.root");
	TFile *file3=new TFile("/gpfs/ddn/cms/user/bruschin/Newtest/WRemnants/wremnants/data/testMuonSF/allSmooth_GtoHisosf_2.root","RECREATE");
	TFile *filein2=new TFile("/gpfs/ddn/cms/user/bruschin/Newtest/smoothLeptonScaleFactorsredofullspline/GtoH/allSmooth_GtoH.root");
	file3->cd();
	for (auto&& keyAsObj : *(file1->GetListOfKeys())){
		auto key = (TKey*) keyAsObj;
		if (std::string(key->GetClassName())==std::string("TH2D")) {
			TH2D* histo=(TH2D*)file1->Get(key->GetName())->Clone();
			histo->SetName(key->GetName());
			histo->Write();
		}
		if (std::string(key->GetClassName())==std::string("TH3D")) {
			TH3D* histo=(TH3D*)file1->Get(key->GetName())->Clone();
			TH3D* Histo=(TH3D*)histo->Clone();
			if ((std::string(key->GetName())==std::string("SF_nomiAndAlt_onlyDataVar_GtoH_iso_both"))||(std::string(key->GetName())==std::string("SF_nomiAndAlt_GtoH_trigger_plus"))||(std::string(key->GetName())==std::string("SF_nomiAndAlt_GtoH_trigger_minus"))) {
				TH3D* histo2=(TH3D*)file2->Get(key->GetName());
				for (unsigned int i=0; i!=histo->GetXaxis()->GetNbins(); i++) {
					for (unsigned int j=0; j!=histo->GetYaxis()->GetNbins(); j++) {
						double sf=Histo->GetBinContent(i+1,j+1,1)*histo2->GetBinContent(1,j+1,1)/Histo->GetBinContent(1,j+1,1);
						histo->SetBinContent(i+1,j+1,1,sf);
					}
				}
			}
			if (std::string(key->GetName())==std::string("SF_nomiAndAlt_onlyDataVar_GtoH_antiiso_both")) {
				TH3D* histo2=(TH3D*)file2->Get("SF_nomiAndAlt_onlyDataVar_GtoH_iso_both");
				TH3D* histo3=(TH3D*)filein2->Get("effMC_nomiAndAlt_GtoH_iso_both");
				for (unsigned int i=0; i!=histo->GetXaxis()->GetNbins(); i++) {
					for (unsigned int j=0; j!=histo->GetYaxis()->GetNbins(); j++) {
						double sf=Histo->GetBinContent(i+1,j+1,1)*histo2->GetBinContent(1,j+1,1)/Histo->GetBinContent(1,j+1,1);
						double antiisosf=(1.-sf*histo3->GetBinContent(i+1,j+1,1))/(1.-histo3->GetBinContent(i+1,j+1,1));
						histo->SetBinContent(i+1,j+1,1,antiisosf);
					}
				}
			}
			if (std::string(key->GetName())==std::string("SF_nomiAndAlt_onlyMCVar_GtoH_antiiso_both")) {
				TH3D* histo2=(TH3D*)file2->Get("SF_nomiAndAlt_onlyDataVar_GtoH_iso_both");
				TH3D* histo3=(TH3D*)filein2->Get("effMC_nomiAndAlt_GtoH_iso_both");
				for (unsigned int i=0; i!=histo->GetXaxis()->GetNbins(); i++) {
					for (unsigned int j=0; j!=histo->GetYaxis()->GetNbins(); j++) {
						double sf=Histo->GetBinContent(i+1,j+1,1)*histo2->GetBinContent(1,j+1,1)/Histo->GetBinContent(1,j+1,1);
						double antiisosf=(1.-sf*histo3->GetBinContent(i+1,j+1,1))/(1.-histo3->GetBinContent(i+1,j+1,1));
						histo->SetBinContent(i+1,j+1,1,antiisosf);
					}
				}
			}
			histo->SetName(key->GetName());
			histo->Write();
		}
	}
}
