void decorrelatesf2d() {
	TFile* file1=new TFile("/gpfs/ddn/cms/user/bruschin/Newtest/WRemnants/wremnants/data/testMuonSF/allSmooth_GtoH3Dsfiso2.root");
	TFile* file2=new TFile("/gpfs/ddn/cms/user/bruschin/Newtest/WRemnants/wremnants/data/testMuonSF/allSmooth_GtoHisosf.root");
	TFile* out1=new TFile("/gpfs/ddn/cms/user/bruschin/Newtest/WRemnants/wremnants/data/testMuonSF/allSmooth_GtoHisosf_bin1.root","RECREATE");
	TFile* out2=new TFile("/gpfs/ddn/cms/user/bruschin/Newtest/WRemnants/wremnants/data/testMuonSF/allSmooth_GtoHisosf_bin2.root","RECREATE");
	TFile* out3=new TFile("/gpfs/ddn/cms/user/bruschin/Newtest/WRemnants/wremnants/data/testMuonSF/allSmooth_GtoHisosf_bin3.root","RECREATE");
	TFile* out4=new TFile("/gpfs/ddn/cms/user/bruschin/Newtest/WRemnants/wremnants/data/testMuonSF/allSmooth_GtoHisosf_bin4.root","RECREATE");
	out1->cd();
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
			if ((std::string(key->GetName())==std::string("SF_nomiAndAlt_onlyDataVar_GtoH_iso_both"))||(std::string(key->GetName())==std::string("SF_nomiAndAlt_GtoH_trigger_plus"))||(std::string(key->GetName())==std::string("SF_nomiAndAlt_GtoH_trigger_minus"))||(std::string(key->GetName())==std::string("SF_nomiAndAlt_onlyDataVar_GtoH_antiiso_both"))||(std::string(key->GetName())==std::string("SF_nomiAndAlt_onlyMCVar_GtoH_antiiso_both"))) {
				TH3D* histo2=(TH3D*)file2->Get(key->GetName());
				for (unsigned int i=0; i!=histo->GetXaxis()->GetNbins(); i++) {
					for (unsigned int j=0; j!=histo->GetYaxis()->GetNbins(); j++) {
						if (!((Histo->GetXaxis()->GetBinCenter(i+1)>-2.4)&&(Histo->GetXaxis()->GetBinCenter(i+1)<-1.0))) histo->SetBinContent(i+1,j+1,1,Histo->GetBinContent(i+1,j+1,1));
						else histo->SetBinContent(i+1,j+1,1,histo2->GetBinContent(i+1,j+1,1));
					}
				}
			}
			histo->SetName(key->GetName());
			histo->Write();
		}
	}
	out1->Close();
	out2->cd();
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
			if ((std::string(key->GetName())==std::string("SF_nomiAndAlt_onlyDataVar_GtoH_iso_both"))||(std::string(key->GetName())==std::string("SF_nomiAndAlt_GtoH_trigger_plus"))||(std::string(key->GetName())==std::string("SF_nomiAndAlt_GtoH_trigger_minus"))||(std::string(key->GetName())==std::string("SF_nomiAndAlt_onlyDataVar_GtoH_antiiso_both"))||(std::string(key->GetName())==std::string("SF_nomiAndAlt_onlyMCVar_GtoH_antiiso_both"))) {
				TH3D* histo2=(TH3D*)file2->Get(key->GetName());
				for (unsigned int i=0; i!=histo->GetXaxis()->GetNbins(); i++) {
					for (unsigned int j=0; j!=histo->GetYaxis()->GetNbins(); j++) {
						if (!((Histo->GetXaxis()->GetBinCenter(i+1)>-1.0)&&(Histo->GetXaxis()->GetBinCenter(i+1)<0.0))) histo->SetBinContent(i+1,j+1,1,Histo->GetBinContent(i+1,j+1,1));
						else histo->SetBinContent(i+1,j+1,1,histo2->GetBinContent(i+1,j+1,1));
					}
				}
			}
			histo->SetName(key->GetName());
			histo->Write();
		}
	}
	out2->Close();
	out3->cd();
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
			if ((std::string(key->GetName())==std::string("SF_nomiAndAlt_onlyDataVar_GtoH_iso_both"))||(std::string(key->GetName())==std::string("SF_nomiAndAlt_GtoH_trigger_plus"))||(std::string(key->GetName())==std::string("SF_nomiAndAlt_GtoH_trigger_minus"))||(std::string(key->GetName())==std::string("SF_nomiAndAlt_onlyDataVar_GtoH_antiiso_both"))||(std::string(key->GetName())==std::string("SF_nomiAndAlt_onlyMCVar_GtoH_antiiso_both"))) {
				TH3D* histo2=(TH3D*)file2->Get(key->GetName());
				for (unsigned int i=0; i!=histo->GetXaxis()->GetNbins(); i++) {
					for (unsigned int j=0; j!=histo->GetYaxis()->GetNbins(); j++) {
						if (!((Histo->GetXaxis()->GetBinCenter(i+1)>0.0)&&(Histo->GetXaxis()->GetBinCenter(i+1)<1.0))) histo->SetBinContent(i+1,j+1,1,Histo->GetBinContent(i+1,j+1,1));
						else histo->SetBinContent(i+1,j+1,1,histo2->GetBinContent(i+1,j+1,1));
					}
				}
			}
			histo->SetName(key->GetName());
			histo->Write();
		}
	}
	out3->Close();
	out4->cd();
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
			if ((std::string(key->GetName())==std::string("SF_nomiAndAlt_onlyDataVar_GtoH_iso_both"))||(std::string(key->GetName())==std::string("SF_nomiAndAlt_GtoH_trigger_plus"))||(std::string(key->GetName())==std::string("SF_nomiAndAlt_GtoH_trigger_minus"))||(std::string(key->GetName())==std::string("SF_nomiAndAlt_onlyDataVar_GtoH_antiiso_both"))||(std::string(key->GetName())==std::string("SF_nomiAndAlt_onlyMCVar_GtoH_antiiso_both"))) {
				TH3D* histo2=(TH3D*)file2->Get(key->GetName());
				for (unsigned int i=0; i!=histo->GetXaxis()->GetNbins(); i++) {
					for (unsigned int j=0; j!=histo->GetYaxis()->GetNbins(); j++) {
						if (!((Histo->GetXaxis()->GetBinCenter(i+1)>1.0)&&(Histo->GetXaxis()->GetBinCenter(i+1)<2.4))) histo->SetBinContent(i+1,j+1,1,Histo->GetBinContent(i+1,j+1,1));
						else histo->SetBinContent(i+1,j+1,1,histo2->GetBinContent(i+1,j+1,1));
					}
				}
			}
			histo->SetName(key->GetName());
			histo->Write();
		}
	}
	out4->Close();
}
