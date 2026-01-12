//MACRO WHICH IS USED TO PRODUCE ANTIISO SFs IN CASE OF DIRECT ISO SF SMOOTHING. IT NEEDS THE FILE WHICH CONTAINS ALL EFFICIENCY STEPS WITH ISO SF SMOOTHED DIRECTLY AND THE SMOOTHED W ISO EFFICIENCY (SMOOTHED WITH A SPLINE) 

void implementantiisosf() {
	TFile *filein=new TFile("/scratchnvme/bruschin/Newtest/newupdatesdeltaphi/smoothLeptonScaleFactorsdeepmet/GtoH/allSmooth_GtoHmerge.root");
	TFile *filein2=new TFile("/scratchnvme/bruschin/Newtest/smoothLeptonScaleFactorsredofullspline/GtoH/allSmooth_GtoH.root");
	TFile *fileout=new TFile("/scratchnvme/bruschin/Newtest/WRemnants/wremnants/data/testMuonSF/allSmooth_GtoH3Disosfdeepmet.root","RECREATE");
	fileout->cd();
	for (auto&& keyAsObj : *(filein->GetListOfKeys())){
		auto key = (TKey*) keyAsObj;
		if (std::string(key->GetClassName())==std::string("TH2D")) {
			TH2D* histo=(TH2D*)filein->Get(key->GetName())->Clone();
			histo->SetName(key->GetName());
			histo->Write();
		}
		if (std::string(key->GetClassName())==std::string("TH3D")) {
			TH3D* histo=(TH3D*)filein->Get(key->GetName())->Clone();
			histo->SetName(key->GetName());
			histo->Write();
			if (std::string(key->GetName())==std::string("SF_nomiAndAlt_GtoH_iso_both")) {
				TH3D* histo2=(TH3D*)filein2->Get("effMC_nomiAndAlt_GtoH_iso_both");
				TH3D* histo3=(TH3D*)filein->Get(key->GetName())->Clone("SF_nomiAndAlt_GtoH_antiiso_both");
				for (unsigned int i=0; i!=histo->GetXaxis()->GetNbins(); i++) {
					for (unsigned int j=0; j!=histo->GetYaxis()->GetNbins(); j++) {
						for (unsigned int h=0; h!=histo->GetZaxis()->GetNbins(); h++) {
							double antiisosf=(1.-histo->GetBinContent(i+1,j+1,h+1)*histo2->GetBinContent(i+1,j+1,1))/(1.-histo2->GetBinContent(i+1,j+1,1));
							histo3->SetBinContent(i+1,j+1,h+1,antiisosf);
						}
					}
				}
				histo3->Write();
			}
			if (std::string(key->GetName())==std::string("SF_nomiAndAlt_GtoH_isonotrig_both")) {
				TH3D* histo2=(TH3D*)filein->Get("effMC_nomiAndAlt_GtoH_isonotrig_both");
				TH3D* histo3=(TH3D*)filein->Get(key->GetName())->Clone("SF_nomiAndAlt_GtoH_antiisonotrig_both");
				for (unsigned int i=0; i!=histo->GetXaxis()->GetNbins(); i++) {
					for (unsigned int j=0; j!=histo->GetYaxis()->GetNbins(); j++) {
						for (unsigned int h=0; h!=histo->GetZaxis()->GetNbins(); h++) {
							double antiisosf=(1.-histo->GetBinContent(i+1,j+1,h+1)*histo2->GetBinContent(i+1,j+1,1))/(1.-histo2->GetBinContent(i+1,j+1,1));
							histo3->SetBinContent(i+1,j+1,h+1,antiisosf);
						}
					}
				}
				histo3->Write();
			}
		}
	}
	fileout->Close();
}
