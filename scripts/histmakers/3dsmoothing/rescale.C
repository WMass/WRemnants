void rescale() {
	TFile *file=new TFile("/gpfs/ddn/cms/user/bruschin/WRemnants/wremnants/data/testMuonSF/allSmooth_GtoH.root");
	TFile *file2=new TFile("/gpfs/ddn/cms/user/bruschin/WRemnants/wremnants/data/testMuonSF/allSmooth_GtoH3D1bin.root");
	TFile *file3=new TFile("/gpfs/ddn/cms/user/bruschin/WRemnants/wremnants/data/testMuonSF/allSmooth_GtoH3DREDO.root");
	TFile *file4=new TFile("/gpfs/ddn/cms/user/bruschin/WRemnants/wremnants/data/testMuonSF/allSmooth_GtoH3DREDOrescaled.root","RECREATE");
	file4->cd();
	for (auto&& keyAsObj : *(file->GetListOfKeys())){
		auto key = (TKey*) keyAsObj;
		TH2D *_2dhisto1, *_2dhisto2, *_2dhisto3, *_2dhisto4;
		TH3D *_3dhisto1, *_3dhisto2, *_3dhisto3, *_3dhisto4;
		if (std::string(key->GetClassName())==std::string("TH2D")) {
			std::string name(key->GetName());
			_2dhisto1=(TH2D*)file->Get(name.c_str());
			_2dhisto1->SetName("histo1");
			_2dhisto2=(TH2D*)file2->Get(name.c_str());
			_2dhisto2->SetName("histo2");
			_2dhisto3=(TH2D*)file3->Get(name.c_str());
			_2dhisto3->SetName("histo3");
			std::vector<double> xaxis(_2dhisto1->GetXaxis()->GetNbins()+1), yaxis(_2dhisto1->GetYaxis()->GetNbins()+1);
			for (unsigned int i=0; i!=_2dhisto1->GetXaxis()->GetNbins()+1; i++) {
				xaxis[i]=_2dhisto1->GetXaxis()->GetBinLowEdge(i+1);
			}
			for (unsigned int i=0; i!=_2dhisto1->GetYaxis()->GetNbins()+1; i++) {
				yaxis[i]=_2dhisto1->GetYaxis()->GetBinLowEdge(i+1);
			}
			_2dhisto4=new TH2D(name.c_str(),_2dhisto1->GetTitle(),_2dhisto1->GetXaxis()->GetNbins(),xaxis.data(),_2dhisto1->GetYaxis()->GetNbins(),yaxis.data());
			for (unsigned int i=0; i!=_2dhisto4->GetXaxis()->GetNbins(); i++) {
				for (unsigned int j=0; j!=_2dhisto4->GetYaxis()->GetNbins(); j++) {
					double cont1=_2dhisto1->GetBinContent(i+1,j+1), cont2=_2dhisto2->GetBinContent(i+1,j+1), cont3=_2dhisto3->GetBinContent(i+1,j+1), err3=_2dhisto3->GetBinError(i+1,j+1);
					_2dhisto4->SetBinContent(i+1,j+1,cont1*cont3/cont2);
					_2dhisto4->SetBinError(i+1,j+1,cont1*err3/cont2);
				}
			}
			_2dhisto4->Write();
		}
		if (std::string(key->GetClassName())==std::string("TH3D")) {
			std::string name(key->GetName());
			_3dhisto1=(TH3D*)file->Get(name.c_str());
			_3dhisto1->SetName("histo1");
			_3dhisto2=(TH3D*)file2->Get(name.c_str());
			_3dhisto2->SetName("histo2");
			_3dhisto3=(TH3D*)file3->Get(name.c_str());
			_3dhisto3->SetName("histo3");
			std::vector<double> xaxis(_3dhisto1->GetXaxis()->GetNbins()+1), yaxis(_3dhisto1->GetYaxis()->GetNbins()+1), zaxis(_3dhisto1->GetZaxis()->GetNbins()+1);
			for (unsigned int i=0; i!=_3dhisto1->GetXaxis()->GetNbins()+1; i++) {
				xaxis[i]=_3dhisto1->GetXaxis()->GetBinLowEdge(i+1);
			}
			for (unsigned int i=0; i!=_3dhisto1->GetYaxis()->GetNbins()+1; i++) {
				yaxis[i]=_3dhisto1->GetYaxis()->GetBinLowEdge(i+1);
			}
			for (unsigned int i=0; i!=_3dhisto1->GetZaxis()->GetNbins()+1; i++) {
				zaxis[i]=_3dhisto1->GetZaxis()->GetBinLowEdge(i+1);
			}
			_3dhisto4=new TH3D(name.c_str(),_3dhisto1->GetTitle(),_3dhisto1->GetXaxis()->GetNbins(),xaxis.data(),_3dhisto1->GetYaxis()->GetNbins(),yaxis.data(),_3dhisto1->GetZaxis()->GetNbins(),zaxis.data());
			for (unsigned int i=0; i!=_3dhisto4->GetXaxis()->GetNbins(); i++) {
				for (unsigned int j=0; j!=_3dhisto4->GetYaxis()->GetNbins(); j++) {
					for (unsigned int h=0; h!=_3dhisto4->GetZaxis()->GetNbins(); h++) {
						double cont1=_3dhisto1->GetBinContent(i+1,j+1,h+1), cont2=_3dhisto2->GetBinContent(i+1,j+1,h+1), cont3=_3dhisto3->GetBinContent(i+1,j+1,h+1), err3=_3dhisto3->GetBinError(i+1,j+1,h+1);
						_3dhisto4->SetBinContent(i+1,j+1,h+1,cont1*cont3/cont2);
						_3dhisto4->SetBinError(i+1,j+1,h+1,cont1*err3/cont2);
					}
				}
			}
			_3dhisto4->Write();
		}
	}
}
