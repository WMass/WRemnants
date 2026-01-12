#define PTSIZE 15

//SECOND STEP IN EFFICIENCY CALCULATION

#define UTBINS 10 //CHANGE IF NEEDED

void makeefficiencies() {
	TFile *file=new TFile("makeefficienciesnewupdatesdeltaphi.root");
	TH3D *errortriggerplus=(TH3D*)file->Get("TriggerErrorPlus");
	TH3D *errortriggerminus=(TH3D*)file->Get("TriggerErrorMinus");
	TH3D *errorisoplus=(TH3D*)file->Get("IsoErrorPlus");
	TH3D *errorisominus=(TH3D*)file->Get("IsoErrorMinus");
	TH2D *IDIPPlus=(TH2D*)file->Get("IDIPPlus");
	TH2D *IDIPMinus=(TH2D*)file->Get("IDIPMinus");
	TH2D *TriggerMCPassPlus=(TH2D*)file->Get("TriggerMCPassPlus");
	TH2D *TriggerMCFailPlus=(TH2D*)file->Get("TriggerMCFailPlus");
	TH2D *TriggerMCPassMinus=(TH2D*)file->Get("TriggerMCPassMinus");
	TH2D *TriggerMCFailMinus=(TH2D*)file->Get("TriggerMCFailMinus");
	TH2D *TriggerPlus=(TH2D*)file->Get("TriggerPlus");
	TH2D *TriggerMinus=(TH2D*)file->Get("TriggerMinus");
	TH2D *IsoMCPassPlus=(TH2D*)file->Get("IsoMCPassPlus");
	TH2D *IsoMCFailPlus=(TH2D*)file->Get("IsoMCFailPlus");
	TH2D *IsoMCPassMinus=(TH2D*)file->Get("IsoMCPassMinus");
	TH2D *IsoMCFailMinus=(TH2D*)file->Get("IsoMCFailMinus");
	TH2D *IsoPlus=(TH2D*)file->Get("IsoPlus");
	TH2D *IsoMinus=(TH2D*)file->Get("IsoMinus");
	TH2D *IsoptPlus=(TH2D*)file->Get("IsoptPlus");
	TH2D *IsoptMinus=(TH2D*)file->Get("IsoptMinus");
	double etabinning[49], ptbinning[PTSIZE+1] = {24.,26.,28.,30.,32.,34.,36.,38.,40.,42.,44.,47.,50.,55.,60.,65.};
	for (unsigned int i=0; i!=49; i++) {
		etabinning[i] = -2.4 + i*0.1;
	}
	TH2D *ErrorTriggerPlus = new TH2D("ErrorTriggerPlus","",48,etabinning,PTSIZE,ptbinning);
	TH2D *ErrorTriggerMinus = new TH2D("ErrorTriggerMinus","",48,etabinning,PTSIZE,ptbinning);
	TH2D *ErrorIsoPlus = new TH2D("ErrorIsoPlus","",48,etabinning,PTSIZE,ptbinning);
	TH2D *ErrorIsoMinus = new TH2D("ErrorIsoMinus","",48,etabinning,PTSIZE,ptbinning);
	TH2D *triggerMCPlus = new TH2D("triggerMCPlus","",48,etabinning,PTSIZE,ptbinning);
	TH2D *triggerMCMinus = new TH2D("triggerMCMinus","",48,etabinning,PTSIZE,ptbinning);
	TH2D *isoMCPlus = new TH2D("isoMCPlus","",48,etabinning,PTSIZE,ptbinning);
	TH2D *isoMCMinus = new TH2D("isoMCMinus","",48,etabinning,PTSIZE,ptbinning);
	TH2D *triggerPlus = new TH2D("triggerPlus","",48,etabinning,PTSIZE,ptbinning);
	TH2D *triggerMinus = new TH2D("triggerMinus","",48,etabinning,PTSIZE,ptbinning);
	TH2D *isoptPlus = new TH2D("isoptPlus","",48,etabinning,PTSIZE,ptbinning);
	TH2D *isoptMinus = new TH2D("isoptMinus","",48,etabinning,PTSIZE,ptbinning);
	TH2D *isoPlus = new TH2D("isoPlus","",48,etabinning,PTSIZE,ptbinning);
	TH2D *isoMinus = new TH2D("isoMinus","",48,etabinning,PTSIZE,ptbinning);
	double ptbinning2[PTSIZE+1],meanvalues[PTSIZE];
	for (unsigned int i=0; i!=PTSIZE; i++) {
		double denominator=0;
		for (unsigned int j=0; j!=48; j++) {
			denominator+=TriggerMCPassPlus->GetBinContent(j+1,i+1);
		}
		meanvalues[i]=isoptPlus->GetBinContent(1,i+1)/denominator;
	}
	
	for (unsigned int i=0; i!=48; i++) {
		for (unsigned int j=0; j!=PTSIZE; j++) {
			double Errortriggerplus=0, Errortriggerminus=0, Errorisoplus=0, Errorisominus=0;
			for (unsigned int h=0; h!=UTBINS; h++) {
				Errortriggerplus+=pow(errortriggerplus->GetBinContent(i+1,j+1,h+1),2);
				Errortriggerminus+=pow(errortriggerminus->GetBinContent(i+1,j+1,h+1),2);
				Errorisoplus+=pow(errorisoplus->GetBinContent(i+1,j+1,h+1),2);
				Errorisominus+=pow(errorisominus->GetBinContent(i+1,j+1,h+1),2);
			}
			Errortriggerplus=sqrt(Errortriggerplus);
			Errortriggerminus=sqrt(Errortriggerminus);
			Errorisoplus=sqrt(Errorisoplus);
			Errorisominus=sqrt(Errorisominus);
			ErrorTriggerPlus->SetBinContent(i+1,j+1,Errortriggerplus);
			ErrorTriggerMinus->SetBinContent(i+1,j+1,Errortriggerminus);
			ErrorIsoPlus->SetBinContent(i+1,j+1,Errorisoplus);
			ErrorIsoMinus->SetBinContent(i+1,j+1,Errorisominus);
			double pass=TriggerMCPassPlus->GetBinContent(i+1,j+1);
			double fail=TriggerMCFailPlus->GetBinContent(i+1,j+1);
			double errpass=TriggerMCPassPlus->GetBinError(i+1,j+1);
			double errfail=TriggerMCFailPlus->GetBinError(i+1,j+1);
			triggerMCPlus->SetBinContent(i+1,j+1,pass/IDIPPlus->GetBinContent(i+1,j+1));
			triggerMCPlus->SetBinError(i+1,j+1,sqrt(errpass*errpass*fail*fail+errfail*errfail*pass*pass) / pow(IDIPPlus->GetBinContent(i+1,j+1),2));
			pass=TriggerMCPassMinus->GetBinContent(i+1,j+1);
			fail=TriggerMCFailMinus->GetBinContent(i+1,j+1);
			errpass=TriggerMCPassMinus->GetBinError(i+1,j+1);
			errfail=TriggerMCFailMinus->GetBinError(i+1,j+1);
			triggerMCMinus->SetBinContent(i+1,j+1,pass/IDIPMinus->GetBinContent(i+1,j+1));
			triggerMCMinus->SetBinError(i+1,j+1,sqrt(errpass*errpass*fail*fail+errfail*errfail*pass*pass) / pow(IDIPMinus->GetBinContent(i+1,j+1),2));
			triggerPlus->SetBinContent(i+1,j+1,TriggerPlus->GetBinContent(i+1,j+1)/IDIPPlus->GetBinContent(i+1,j+1));
			triggerPlus->SetBinError(i+1,j+1,ErrorTriggerPlus->GetBinContent(i+1,j+1)/IDIPPlus->GetBinContent(i+1,j+1));
			triggerMinus->SetBinContent(i+1,j+1,TriggerMinus->GetBinContent(i+1,j+1)/IDIPMinus->GetBinContent(i+1,j+1));
			triggerMinus->SetBinError(i+1,j+1,ErrorTriggerMinus->GetBinContent(i+1,j+1)/IDIPMinus->GetBinContent(i+1,j+1));
			pass=IsoMCPassPlus->GetBinContent(i+1,j+1);
			fail=IsoMCFailPlus->GetBinContent(i+1,j+1);
			errpass=IsoMCPassPlus->GetBinError(i+1,j+1);
			errfail=IsoMCFailPlus->GetBinError(i+1,j+1);
			isoMCPlus->SetBinContent(i+1,j+1,pass/(pass+fail));
			isoMCPlus->SetBinError(i+1,j+1,sqrt(errpass*errpass*fail*fail+errfail*errfail*pass*pass) / pow(pass+fail,2));
			pass=IsoMCPassMinus->GetBinContent(i+1,j+1);
			fail=IsoMCFailMinus->GetBinContent(i+1,j+1);
			errpass=IsoMCPassMinus->GetBinError(i+1,j+1);
			errfail=IsoMCFailMinus->GetBinError(i+1,j+1);
			isoMCMinus->SetBinContent(i+1,j+1,pass/(pass+fail));
			isoMCMinus->SetBinError(i+1,j+1,sqrt(errpass*errpass*fail*fail+errfail*errfail*pass*pass) / pow(pass+fail,2));
			isoPlus->SetBinContent(i+1,j+1,IsoPlus->GetBinContent(i+1,j+1)/TriggerPlus->GetBinContent(i+1,j+1));
			isoPlus->SetBinError(i+1,j+1,ErrorIsoPlus->GetBinContent(i+1,j+1)/TriggerPlus->GetBinContent(i+1,j+1));
			isoMinus->SetBinContent(i+1,j+1,IsoMinus->GetBinContent(i+1,j+1)/TriggerMinus->GetBinContent(i+1,j+1));
			isoMinus->SetBinError(i+1,j+1,ErrorIsoMinus->GetBinContent(i+1,j+1)/TriggerMinus->GetBinContent(i+1,j+1));
		}
	}
	TFile *output = new TFile("efficiencieswremnantsnewupdatesdeltaphi.root","RECREATE");
	output->cd();
	triggerMCPlus->Write();
	triggerMCMinus->Write();
	isoMCPlus->Write();
	isoMCMinus->Write();
	triggerPlus->Write();
	triggerMinus->Write();
	isoPlus->Write();
	isoMinus->Write();
}
