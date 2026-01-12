//THIS SCRIPTS IS USED TO MERGE SCALE FACTORS FROM DIFFERENT FILES. THIS IS BECAUSE I PERFORMED THE SMOOTHING ONLY FOR TRIGGER AND ISO, THE REST IS EXACTLY THE SAME. THIS ALSO IMPLEMENTS THE CORRECTION TO ALTERNATE FITS (WHICH MOSTLY DON'T CONVERGE AS A FUNCTION OF UT, BUT THE NOMINAL ONES DO, SO WHAT IS DONE IS RESCALE THE ALT FIT EFFICIENCY TO THE 3D SMOOTH ONE, AS THE REASON BEHIND THE UNCERTAINTY BAR SHOULD BE THE SAME

void makenew2dsf() {
	TFile *file=new TFile("/scratchnvme/bruschin/Newtest/WRemnants/wremnants/data/testMuonSF/allSmooth_GtoH.root");
	TFile *file2=new TFile("/scratchnvme/bruschin/Newtest/newupdatesdeltaphi/smoothLeptonScaleFactorsdeepmet/GtoH/allSmooth_GtoH.root");
	TFile *file3=new TFile("/scratchnvme/bruschin/Newtest/newupdatesdeltaphi/smoothLeptonScaleFactorsdeepmet/GtoH/allSmooth_GtoHmerge.root","RECREATE");
	file3->cd();
	TH2D *Histo1, *Histo2;
	for (auto&& keyAsObj : *(file->GetListOfKeys())){
		auto key = (TKey*) keyAsObj;
		cout << key->GetName() << " " << key->GetClassName() << endl;
		std::string string(key->GetName()), trigplus("trigger_plus"), trigminus("trigger_minus"), iso("iso_both"), isonotrig("isonotrig_both");
		TNamed *object;
		if (string.find(std::string("dataMCVar"))!= std::string::npos) continue;
		if (string==std::string("effData_nomiAndAlt_GtoH_trigger_plus")) continue;
		if (string==std::string("effMC_nomiAndAlt_GtoH_trigger_plus")) continue;
		if (string==std::string("effData_nomiAndAlt_GtoH_trigger_minus")) continue;
		if (string==std::string("effMC_nomiAndAlt_GtoH_trigger_minus")) continue;
		if ((string.find(trigplus) != std::string::npos)||(string.find(trigminus) != std::string::npos)||(string.find(iso) != std::string::npos)) {
			if ((string.find("original_"))!= std::string::npos) {
				Histo1=(TH2D*)file2->Get(key->GetName())->Clone();
				Histo2=(TH2D*)file->Get(key->GetName())->Clone();
			}
			if (std::string(key->GetClassName())==std::string("TH3D")) {
				TH3D *histo1=(TH3D*)file2->Get(key->GetName())->Clone();
				TH3D *histo2=(TH3D*)file->Get(key->GetName())->Clone();
				for (unsigned int i=0; i!=histo1->GetXaxis()->GetNbins(); i++) {
					for (unsigned int j=0; j!=histo1->GetYaxis()->GetNbins(); j++) {
						int lastbin=histo1->GetZaxis()->GetNbins(), lastbin2=histo2->GetZaxis()->GetNbins();
						double corr=histo1->GetBinContent(i+1,j+1,1), wrong=histo2->GetBinContent(i+1,j+1,1), syst=histo2->GetBinContent(i+1,j+1,lastbin2);
						histo1->SetBinContent(i+1,j+1,lastbin,syst*corr/wrong);
					}
				}
				object=(TNamed*)histo1;
			}
			else if (string.find(std::string("DataAltSig"))!= std::string::npos) {
				TH2D *histo1=(TH2D*)file2->Get(key->GetName())->Clone();
				TH2D *histo2=(TH2D*)file->Get(key->GetName())->Clone();
				for (unsigned int i=0; i!=histo1->GetXaxis()->GetNbins(); i++) {
					for (unsigned int j=0; j!=histo1->GetYaxis()->GetNbins(); j++) {
						double corr=Histo1->GetBinContent(i+1,j+1), wrong=Histo2->GetBinContent(i+1,j+1), syst=histo2->GetBinContent(i+1,j+1);
						histo1->SetBinContent(i+1,j+1,syst*corr/wrong);
					}
				}
				object=(TNamed*)histo1;
			}
			else object=(TNamed*)file2->Get(key->GetName())->Clone();
		}
		else if (string.find(isonotrig) != std::string::npos) {
			if (std::string(key->GetClassName())==std::string("TH3D")) {
				TH3D *histo1=(TH3D*)file->Get(key->GetName())->Clone();
				histo1->SetName("histo1");
				int npar=(histo1->GetZaxis()->GetNbins()-2)/2, npar2=7;
				TH3D *histo3=new TH3D(key->GetName(),"",48,-2.4,2.4,41*5,24,65,npar2*2+2,0,npar2*2+2);
				for (unsigned int i=0; i!=histo1->GetXaxis()->GetNbins(); i++) {
					for (unsigned int j=0; j!=histo1->GetYaxis()->GetNbins(); j++) {
						int lastbin=histo1->GetZaxis()->GetNbins();
						double corr=histo1->GetBinContent(i+1,j+1,1);
						histo3->SetBinContent(i+1,j+1,lastbin,corr);
						histo3->SetBinContent(i+1,j+1,1,corr);
						for (int h=0; h!=npar; h++) {
							histo3->SetBinContent(i+1,j+1,2+h,histo1->GetBinContent(i+1,j+1,2+h));
							histo3->SetBinContent(i+1,j+1,2+h+npar2,histo1->GetBinContent(i+1,j+1,2+h+npar));
						}
						for (int h=npar; h!=npar2; h++) {
							histo3->SetBinContent(i+1,j+1,2+h,histo1->GetBinContent(i+1,j+1,1));
							histo3->SetBinContent(i+1,j+1,2+h+npar2,histo1->GetBinContent(i+1,j+1,1));
						}
					}
				}
				object=(TNamed*)histo3;
			}
			else if (string.find(std::string("DataAltSig"))!= std::string::npos) {
				TH2D *histo1=(TH2D*)file->Get(key->GetName())->Clone();
				object=(TNamed*)histo1;
			}
			else object=(TNamed*)file->Get(key->GetName())->Clone();
		}
		else {
			object=(TNamed*)file->Get(key->GetName())->Clone();
		}
		object->SetName(key->GetName());
		object->Write();
	}
	file3->Close();
}
