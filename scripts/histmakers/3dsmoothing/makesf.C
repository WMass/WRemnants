#define PTSIZE 15

//FINAL CREATION OF HISTOGRAMS, USED TO PRODUCE FILES IN THE SAME FORMAT AS THE egm TOOL OUTPUT, TO BE PASSED TO THE SMOOTHER. YOU NEED TO CHANGE THE HISTNAMES (e.g. if you want trigger you need triggerPlus, triggerMCPlus...), AND THE 

void makesf() {
	TFile *file=new TFile("efficiencieswremnantsnewupdatesdeltaphi.root");
	TFile *file2=new TFile("efficiencieswremnantshighpt10bins.root");
	TH2D* isoplus=(TH2D*)file->Get("isoPlus");
	TH2D* isoMCplus=(TH2D*)file->Get("isoMCPlus");
	TH2D* isominus=(TH2D*)file->Get("isoMinus");
	TH2D* isoMCminus=(TH2D*)file->Get("isoMCMinus");
	TH2D* isoplus2=(TH2D*)file2->Get("isoPlus");
	TH2D* isoMCplus2=(TH2D*)file2->Get("isoMCPlus");
	TH2D* isominus2=(TH2D*)file2->Get("isoMinus");
	TH2D* isoMCminus2=(TH2D*)file2->Get("isoMCMinus");
	//double etabinning[49], ptbinning[PTSIZE+1] = {40.,42.,44.,47.,50.,55.,60.,65.};
	double etabinning[49], ptbinning[PTSIZE+1] = {24.,26.,28.,30.,32.,34.,36.,38.,40.,42.,44.,47.,50.,55.,60.,65.};
	for (unsigned int i=0; i!=49; i++) {
		etabinning[i] = -2.4 + i*0.1;
	}
	TH2D* sf=new TH2D("SF2D_nominal","SF2D_nominal",48,etabinning,PTSIZE,ptbinning);
	TH2D* data=new TH2D("EffData2D","EffData2D",48,etabinning,PTSIZE,ptbinning);
	TH2D* sf2=new TH2D("SF2D_dataAltSig","SF2D_nominal",48,etabinning,PTSIZE,ptbinning);
	TH2D* data2=new TH2D("EffDataAltSig2D","EffData2D",48,etabinning,PTSIZE,ptbinning);
	TH2D* mc=new TH2D("EffMC2D","EffMC2D",48,etabinning,PTSIZE,ptbinning);
	for (unsigned int i=0; i!=48; i++) {
		for (unsigned int j=0; j!=PTSIZE; j++) {
			//if (j<8) {
				data->SetBinContent(i+1,j+1,isoplus->GetBinContent(i+1,j+1)); data->SetBinError(i+1,j+1,isoplus->GetBinError(i+1,j+1));
				data2->SetBinContent(i+1,j+1,isoplus->GetBinContent(i+1,j+1)); data2->SetBinError(i+1,j+1,isoplus->GetBinError(i+1,j+1));
				mc->SetBinContent(i+1,j+1,isoMCplus->GetBinContent(i+1,j+1)); mc->SetBinError(i+1,j+1,isoMCplus->GetBinError(i+1,j+1));
				sf->SetBinContent(i+1,j+1,isoplus->GetBinContent(i+1,j+1)/isoMCplus->GetBinContent(i+1,j+1));
				sf2->SetBinContent(i+1,j+1,isoplus->GetBinContent(i+1,j+1)/isoMCplus->GetBinContent(i+1,j+1));
				sf->SetBinError(i+1,j+1,isoplus->GetBinError(i+1,j+1)/isoMCplus->GetBinContent(i+1,j+1));
				sf2->SetBinError(i+1,j+1,isoplus->GetBinError(i+1,j+1)/isoMCplus->GetBinContent(i+1,j+1));
			//}
			/*else {
				data->SetBinContent(i+1,j+1,isominus2->GetBinContent(i+1,j-8+1)); data->SetBinError(i+1,j+1,isominus2->GetBinError(i+1,j-8+1));
				data2->SetBinContent(i+1,j+1,isominus2->GetBinContent(i+1,j-8+1)); data2->SetBinError(i+1,j+1,isominus2->GetBinError(i+1,j-8+1));
				mc->SetBinContent(i+1,j+1,isoMCminus2->GetBinContent(i+1,j-8+1)); mc->SetBinError(i+1,j+1,isoMCminus2->GetBinError(i+1,j-8+1));
				sf->SetBinContent(i+1,j+1,isominus2->GetBinContent(i+1,j-8+1)/isoMCminus2->GetBinContent(i+1,j-8+1));
				sf2->SetBinContent(i+1,j+1,isominus2->GetBinContent(i+1,j-8+1)/isoMCminus2->GetBinContent(i+1,j-8+1));
				sf->SetBinError(i+1,j+1,isominus2->GetBinError(i+1,j-8+1)/isoMCminus2->GetBinContent(i+1,j-8+1));
				sf2->SetBinError(i+1,j+1,isominus2->GetBinError(i+1,j-8+1)/isoMCminus2->GetBinContent(i+1,j-8+1));
			}*/
		}
		/*for (unsigned int j=0; j!=PTSIZE; j++) {
			data->SetBinContent(i+1,j+1,isominus->GetBinContent(i+1,j+8+1)); data->SetBinError(i+1,j+1,isominus->GetBinError(i+1,j+8+1));
			data2->SetBinContent(i+1,j+1,isominus->GetBinContent(i+1,j+8+1)); data2->SetBinError(i+1,j+1,isominus->GetBinError(i+1,j+8+1));
			mc->SetBinContent(i+1,j+1,isoMCminus->GetBinContent(i+1,j+8+1)); mc->SetBinError(i+1,j+1,isoMCminus->GetBinError(i+1,j+8+1));
			sf->SetBinContent(i+1,j+1,isominus->GetBinContent(i+1,j+8+1)/isoMCminus->GetBinContent(i+1,j+8+1));
			sf2->SetBinContent(i+1,j+1,isominus->GetBinContent(i+1,j+8+1)/isoMCminus->GetBinContent(i+1,j+8+1));
			sf->SetBinError(i+1,j+1,isominus->GetBinError(i+1,j+8+1)/isoMCminus->GetBinContent(i+1,j+8+1));
			sf2->SetBinError(i+1,j+1,isominus->GetBinError(i+1,j+8+1)/isoMCminus->GetBinContent(i+1,j+8+1));
		}*/
	}
	TFile *output=new TFile("newupdatesdeltaphi/mu_iso_both/allEfficiencies_2D.root","RECREATE");
	output->cd();
	sf->Write();
	sf2->Write();
	data->Write();
	data2->Write();
	mc->Write();
	output->Close();
}
