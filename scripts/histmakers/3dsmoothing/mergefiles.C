void mergefiles() {
	TFile *file=new TFile("/gpfs/ddn/cms/user/bruschin/Newtest/smoothLeptonScaleFactorswrongwayofdoingthingspol6/GtoH/allSmooth_GtoH.root");
	TFile *file2=new TFile("/gpfs/ddn/cms/user/bruschin/Newtest/smoothLeptonScaleFactorsredofull/GtoH/allSmooth_GtoH.root");
	TFile *file3=new TFile("/gpfs/ddn/cms/user/bruschin/Newtest/smoothLeptonScaleFactorswrongwayofdoingthingspol6/GtoH/allSmooth_GtoH2.root","RECREATE");
	file3->cd();
	for (auto&& keyAsObj : *(file2->GetListOfKeys())){
		auto key = (TKey*) keyAsObj;
		bool found=false;
		for (auto&& keyAsObj2 : *(file->GetListOfKeys())){
			auto key2 = (TKey*) keyAsObj2;
			if (std::string(key->GetName())==std::string(key2->GetName())) {
				found=true;
			}
		}
		TNamed* object;
		if (found) {
			object=(TNamed*)file->Get(key->GetName())->Clone();
		}
		else {
			object=(TNamed*)file2->Get(key->GetName())->Clone();
		}
		object->SetName(key->GetName());
		object->Write();
	}
	file3->Close();
}
