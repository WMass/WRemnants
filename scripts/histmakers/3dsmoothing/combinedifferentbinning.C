//TO BE USED IN CASE WE NEED TO COMBINE EFFICIENCIES SMOOTHED IN DIFFERENT PT RANGES (AT THE MOMENT THE SMOOTHER DOESN'T DO IT, SO A WORKAROUND IS TO CHANGE THE PT RANGE IN THE HISTOGRAMS AND THEN ADD THE REMAINING BINS FROM ANOTHER FILE).

void combinedifferentbinning() {
	TFile *file=new TFile("/gpfs/ddn/cms/user/bruschin/Newtest/smoothLeptonScaleFactorsredofullpol4/GtoH/allSmooth_GtoH.root");
	TFile *file2=new TFile("/gpfs/ddn/cms/user/bruschin/Newtest/smoothLeptonScaleFactorsredopol4/GtoH/allSmooth_GtoH.root");
	TFile *file3=new TFile("/gpfs/ddn/cms/user/bruschin/Newtest/smoothLeptonScaleFactorsredopol4/GtoH/allSmooth_GtoH2.root","RECREATE");
	file3->cd();
	for (auto&& keyAsObj : *(file2->GetListOfKeys())){
		auto key = (TKey*) keyAsObj;
		TH2D *_2dhisto1, *_2dhisto2, *_2dhisto3;
		TH3D *_3dhisto1, *_3dhisto2, *_3dhisto3;
		std::cout<<key->GetName()<<"\n";
		if (std::string(key->GetClassName())==std::string("TH2D")) {
			std::string name(key->GetName());
			_2dhisto1=(TH2D*)file->Get(name.c_str());
			_2dhisto1->SetName("histo1");
			_2dhisto2=(TH2D*)file2->Get(name.c_str());
			_2dhisto2->SetName("histo2");
			std::vector<double> xaxis(_2dhisto1->GetXaxis()->GetNbins()+1), yaxis(_2dhisto1->GetYaxis()->GetNbins()+1);
			for (unsigned int i=0; i!=_2dhisto1->GetXaxis()->GetNbins()+1; i++) {
				xaxis[i]=_2dhisto1->GetXaxis()->GetBinLowEdge(i+1);
			}
			for (unsigned int i=0; i!=_2dhisto1->GetYaxis()->GetNbins()+1; i++) {
				yaxis[i]=_2dhisto1->GetYaxis()->GetBinLowEdge(i+1);
			}
			_2dhisto3=new TH2D(name.c_str(),_2dhisto1->GetTitle(),_2dhisto1->GetXaxis()->GetNbins(),xaxis.data(),_2dhisto1->GetYaxis()->GetNbins(),yaxis.data());
			for (unsigned int i=0; i!=_2dhisto2->GetXaxis()->GetNbins(); i++) {
				for (unsigned int j=0; j!=_2dhisto2->GetYaxis()->GetNbins(); j++) {
					_2dhisto3->SetBinContent(i+1,j+1,_2dhisto2->GetBinContent(i+1,j+1));
					_2dhisto3->SetBinError(i+1,j+1,_2dhisto2->GetBinError(i+1,j+1));
				}
			}
			for (unsigned int i=0; i!=_2dhisto1->GetXaxis()->GetNbins(); i++) {
				for (unsigned int j=_2dhisto2->GetYaxis()->GetNbins(); j!=_2dhisto1->GetYaxis()->GetNbins(); j++) {
					_2dhisto3->SetBinContent(i+1,j+1,_2dhisto1->GetBinContent(i+1,j+1));
					_2dhisto3->SetBinError(i+1,j+1,_2dhisto1->GetBinError(i+1,j+1));
				}
			}
			_2dhisto3->Write();
		}
		if (std::string(key->GetClassName())==std::string("TH3D")) {
			std::string name(key->GetName());
			_3dhisto1=(TH3D*)file->Get(name.c_str());
			_3dhisto1->SetName("histo1");
			_3dhisto2=(TH3D*)file2->Get(name.c_str());
			_3dhisto2->SetName("histo2");
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
			_3dhisto3=new TH3D(name.c_str(),_3dhisto1->GetTitle(),_3dhisto1->GetXaxis()->GetNbins(),xaxis.data(),_3dhisto1->GetYaxis()->GetNbins(),yaxis.data(),_3dhisto1->GetZaxis()->GetNbins(),zaxis.data());
			for (unsigned int i=0; i!=_3dhisto2->GetXaxis()->GetNbins(); i++) {
				for (unsigned int j=0; j!=_3dhisto2->GetYaxis()->GetNbins(); j++) {
					for (unsigned int h=0; h!=_3dhisto2->GetZaxis()->GetNbins(); h++) {
						_3dhisto3->SetBinContent(i+1,j+1,h+1,_3dhisto2->GetBinContent(i+1,j+1,h+1));
						_3dhisto3->SetBinError(i+1,j+1,h+1,_3dhisto2->GetBinError(i+1,j+1,h+1));
					}
				}
			}
			for (unsigned int i=0; i!=_3dhisto1->GetXaxis()->GetNbins(); i++) {
				for (unsigned int j=_3dhisto2->GetYaxis()->GetNbins(); j!=_3dhisto1->GetYaxis()->GetNbins(); j++) {
					for (unsigned int h=0; h!=_3dhisto1->GetZaxis()->GetNbins(); h++) {
						_3dhisto3->SetBinContent(i+1,j+1,h+1,_3dhisto1->GetBinContent(i+1,j+1,h+1));
						_3dhisto3->SetBinError(i+1,j+1,h+1,_3dhisto1->GetBinError(i+1,j+1,h+1));
					}
				}
			}
			_3dhisto3->Write();
		}
	}
}
