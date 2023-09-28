#ifndef WREMNANTS_LOWPU_RECOIL_H
#define WREMNANTS_LOWPU_RECOIL_H


#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include "defines.h"

namespace wrem {

struct recoil {
  float para;
  float perp;
};



    
using Vec_recoil = ROOT::VecOps::RVec<recoil>;
    

std::map<std::string, TH3D*> recoil_hists;
std::map<std::string, TH3D*> recoil_hists_gen_W;
std::map<std::string, TH3D*> recoil_hists_gen_Z;
std::map<std::string, TH1D*> recoil_hists_param;

// recoil parametric maps
std::map<std::string, int> recoil_param_nGauss;

std::vector<TF1*> recoil_param_funcs_;
std::map<std::string, TF1> recoil_param_funcs__;
std::unordered_map<std::string, TF1*> recoil_param_funcs;
std::unordered_map<std::string, bool> recoil_param_transform;

std::map<std::string, int> recoil_unc_no; // number of uncertainties recoil_unc_no

std::map<std::string, int> recoil_binned_nGauss;
std::map<std::string, std::vector<float>> recoil_binned_qTbins;
std::map<std::string, std::vector<  std::vector<  std::vector< float > >  >> recoil_binned_components; // tag qTbin nGauss, component

bool applyRecoilCorrection = false;
bool recoil_verbose = false;
    
std::vector<float> qTbins;
float recoil_correction_qTmax = 1e99;


// qT reweighting
bool applyqTReweighting = false;
std::vector<float> qTrw_bins;
std::vector<float> qTrw_weights;




// MET xy correction
bool applyMETxyCorrection = false;
std::vector<float> met_xy_corr_x_data_nom, met_xy_corr_y_data_nom, met_xy_corr_x_mc_nom, met_xy_corr_y_mc_nom;

void insertFunction(char* name, char* expr, std::vector<float> pars, bool transform) {
    
    auto func = new TF1(name, expr, -300, 300);
    for(int j=0; j<pars.size(); j++) {
        func->SetParameter(j, pars.at(j));
    }
    recoil_param_funcs.insert({name, func});
    recoil_param_transform.insert({name, transform});
}

void recoil_init(char* name) {
    

}



void recoil_init_gen(char* name_W, char* name_Z) {
    
    TH3D *h3;

    TFile *recoil_fits_W = new TFile(name_W, "READ");
    TH3D *gen_plus_para = (TH3D*)recoil_fits_W->Get("gen_plus_para");
    TH3D *gen_minus_para = (TH3D*)recoil_fits_W->Get("gen_minus_para");
    TH3D *gen_plus_perp = (TH3D*)recoil_fits_W->Get("gen_plus_perp");
    TH3D *gen_minus_perp = (TH3D*)recoil_fits_W->Get("gen_minus_perp");
    gen_plus_para->SetDirectory(0);
    gen_minus_para->SetDirectory(0);
    gen_plus_perp->SetDirectory(0);
    gen_minus_perp->SetDirectory(0);
    recoil_hists_gen_W.insert({gen_plus_para->GetName(), gen_plus_para});
    recoil_hists_gen_W.insert({gen_minus_para->GetName(), gen_minus_para});
    recoil_hists_gen_W.insert({gen_plus_perp->GetName(), gen_plus_perp});
    recoil_hists_gen_W.insert({gen_minus_perp->GetName(), gen_minus_perp});
    recoil_fits_W->Close();
    
    TFile *recoil_fits_Z = new TFile(name_Z, "READ");
    TH3D *gen_para_Z = (TH3D*)recoil_fits_Z->Get("gen_para");
    TH3D *gen_perp_Z = (TH3D*)recoil_fits_Z->Get("gen_perp");
    gen_para_Z->SetDirectory(0);
    gen_perp_Z->SetDirectory(0);
    recoil_hists_gen_Z.insert({gen_para_Z->GetName(), gen_para_Z});
    recoil_hists_gen_Z.insert({gen_perp_Z->GetName(), gen_perp_Z});
    recoil_fits_Z->Close();
}


int getqTbin(float qT) {

    for(unsigned int i=0; i < qTbins.size()-1; i++) {
        
        if(qT >= qTbins.at(i) and qT < qTbins.at(i+1)) return i;
    }
    
    //cout << "[getQtBin] Bin not found qT=" << qT << endl;
    return -1;
}

double qTweight(float qT) {

	if(not applyqTReweighting) return 1.0;
    int idx = qTrw_bins.size()-2;
    for(unsigned int i=0; i < qTrw_bins.size()-1; i++) {
        if(qT >= qTrw_bins.at(i) and qT < qTrw_bins.at(i+1)) {
            idx = i;
            break;
        }
    }
    return qTrw_weights.at(idx);
}





Vec_d METLeptonCorrection(double MET_pt, double MET_phi, Vec_d lep_pt_uncorr, Vec_d lep_pt, Vec_d lep_phi) {

	// correct MET for muon scale corrections (in lab frame pT-phi)
	TVector2 lep1_raw(lep_pt_uncorr[0]*cos(lep_phi[0]), lep_pt_uncorr[0]*sin(lep_phi[0]));
	TVector2 lep2_raw(lep_pt_uncorr[1]*cos(lep_phi[1]), lep_pt_uncorr[1]*sin(lep_phi[1]));
    TVector2 lep1_corr(lep_pt[0]*cos(lep_phi[0]), lep_pt[0]*sin(lep_phi[0]));
    TVector2 lep2_corr(lep_pt[1]*cos(lep_phi[1]), lep_pt[1]*sin(lep_phi[1]));

	TVector2 MET(MET_pt*cos(MET_phi), MET_pt*sin(MET_phi));
	TVector2 MET_corr = MET + lep1_raw + lep2_raw - lep1_corr - lep2_corr;
	double MET_pt_corr = MET_corr.Mod();
    double MET_phi_corr = MET_corr.Phi_mpi_pi(MET_corr.Phi());
	
    Vec_d res(2, 0);
    res[0] = MET_pt_corr;
    res[1] = MET_phi_corr;
    
    //cout << "** Lepton angles ** " << MET_corr.Phi() << " " << MET_corr.Phi_mpi_pi(MET_corr.Phi()) << endl;
     

	
	return res;
	
}


Vec_d METLeptonCorrection(double MET_pt, double MET_phi, double lep_pt_uncorr, double lep_pt, double lep_phi) {

    Vec_d lep_pt_uncorr_(1, lep_pt_uncorr);
    Vec_d lep_pt_(1, lep_pt);
    Vec_d lep_phi_(1, lep_phi);
	return METLeptonCorrection(MET_pt, MET_phi, lep_pt_uncorr_, lep_pt_, lep_phi_);
}

Vec_d recoilComponents(double MET_pt, double MET_phi, double lep_pt, double lep_phi) {
    
    // computes the hadronic recoil, defined as -(MET + V), V being the vector sum of the reco (di)lepton system in the lab frame
    double pUx  = MET_pt*cos(MET_phi) + lep_pt*cos(lep_phi); // recoil x in lab frame
	double pUy  = MET_pt*sin(MET_phi) + lep_pt*sin(lep_phi); // recoil y in lab frame
	double pU   = std::hypot(pUx, pUy);
	double Ux	= - (pUx*cos(lep_phi) + pUy*sin(lep_phi));
	double Uy	= - (- pUx*sin(lep_phi) + pUy*cos(lep_phi));    

    Vec_d res(3, 0);
    res[0] = pU;
    res[1] = Ux;
    res[2] = Uy;
    
    //cout << " MET_phi=" << MET_phi << "  lep_phi=" << lep_phi<< "  pU=" << pU<< "  Ux=" << Ux << "  Uy=" << Uy << endl;
    
	return res;
}


Vec_d recoilComponentsGen(double MET_pt, double MET_phi, double lep_pt, double lep_phi, double gen_lep_phi) {
    
    // computes the hadronic recoil, defined as -(MET + V), V being the vector sum of the reco (di)lepton system in the lab frame
    // compute recoil as projection on the gen boson phi
    double pUx  = MET_pt*cos(MET_phi) + lep_pt*cos(lep_phi); // recoil x in lab frame
	double pUy  = MET_pt*sin(MET_phi) + lep_pt*sin(lep_phi); // recoil y in lab frame
	double pU   = std::hypot(pUx, pUy);
	double Ux	= - (pUx*cos(gen_lep_phi) + pUy*sin(gen_lep_phi));
	double Uy	= - (- pUx*sin(gen_lep_phi) + pUy*cos(gen_lep_phi));

    Vec_d res(3, 0);
    res[0] = pU;
    res[1] = Ux;
    res[2] = Uy;
        
	return res;
}

Vec_d METXYCorrection(double MET_pt, double MET_phi, int npv, int isData) {
    
    // preserve the MET magnitude??

    double deltaMax = 10; // protection for high/nonphysical corrections
    double delta;
    double pUx = MET_pt*cos(MET_phi);
    double pUy = MET_pt*sin(MET_phi);
    
    Vec_d res(2, 0);
    res[0] = hypot(pUx, pUy);
    res[1] = atan2(pUy, pUx);
    
    if(!applyMETxyCorrection) return res;

    if(isData == 1) {
    
        // x
        delta = 0;
        for(int k=0; k<met_xy_corr_x_data_nom.size(); k++) delta += met_xy_corr_x_data_nom.at(k)*std::pow(npv, k);
        if(std::abs(delta) > deltaMax) delta = 0;
        pUx -= delta;
        
        // y
        delta = 0;
        for(int k=0; k<met_xy_corr_y_data_nom.size(); k++) delta += met_xy_corr_y_data_nom.at(k)*std::pow(npv, k);
        if(std::abs(delta) > deltaMax) delta = 0;
        pUy -= delta;    
    }
    else {
        // x
        delta = 0;
        for(int k=0; k<met_xy_corr_x_mc_nom.size(); k++) delta += met_xy_corr_x_mc_nom.at(k)*std::pow(npv, k);
        if(std::abs(delta) > deltaMax) delta = 0;
        pUx -= delta;
        
        // y
        delta = 0;
        for(int k=0; k<met_xy_corr_y_mc_nom.size(); k++) delta += met_xy_corr_y_mc_nom.at(k)*std::pow(npv, k);
        if(std::abs(delta) > deltaMax) delta = 0;
        pUy -= delta;     
    }

    res[0] = hypot(pUx, pUy);
    res[1] = atan2(pUy, pUx);

    //cout << "** XY angles ** " << atan(pUy/pUx) << " " << res[1] << " (pUy=" << pUy << " pUx=" << pUx << ")" << endl;
    
    return res;
}





class GaussianSum {
    
    private:
    
        std::vector<double> mean;
        std::vector<double> sigma;
        std::vector<double> norm;
        double totNorm;
        double extraNorm =1;
        
        //int MaxIterations = 512; 
        //double _tol = 2.2204460492503131e-16;
        
        int MaxIterations = 1024; 
        double _tol = 2.2204460492503131e-16;
        
        bool verbose = true;
    
    
    public:
    
        GaussianSum() {};
        
        void addTerm(double mean_, double sigma_, double norm_) {
            
            mean.push_back(mean_);
            sigma.push_back(sigma_);
            norm.push_back(norm_*extraNorm);
            totNorm = std::accumulate(norm.begin(), norm.end(), 0.0f); // recompute norm
        }
        
        
        void setNorm(double norm) {
            
            extraNorm = norm;
        }
        
        double evalCDF(double pval) {

            double ret = 0;
            double tmp;
            for(unsigned int i = 0; i < mean.size(); i++) {
                
                tmp = (pval-mean.at(i))/sigma.at(i);
                //ret += norm.at(i)*std::erfc(-tmp/TMath::Sqrt2()); // erfc?
                ret += norm.at(i)*0.5*(1.0 + std::erf(tmp/TMath::Sqrt2()));
            }
            //return 0.5*ret/totNorm;
            return ret/totNorm;
        }
        
        double eval(double val) {

            double ret = 0;
            double tmp;
            for(unsigned int i = 0; i < mean.size(); i++) {
                ret += norm.at(i)*TMath::Gaus(val, mean.at(i), sigma.at(i), true);
            }
            return ret/totNorm;
        }
        
        void resetMeans() {
            
            for(unsigned int i = 0; i < mean.size(); i++) {
                mean.at(i) = 0;
            }
        }
        
        double getWeightedMean() {
            
            double wmean = 0;
            for(unsigned int i = 0; i < mean.size(); i++) {
                wmean += norm.at(i)*mean.at(i);
            }
            return wmean;
        }
        
        double findRoot(double value, double xMin, double xMax) {
            
            double a(xMin), b(xMax);
            
            if(recoil_verbose and (totNorm < 0.999 or totNorm > 1.001)) cout << "[nGaussian::findRoot]: Total norm not equal to 1: " << totNorm  << endl;
            
            double fa = this->evalCDF(a) - value;
            double fb = this->evalCDF(b) - value;
            
     
            if(fb*fa > 0) {
                if(recoil_verbose) cout << "[nGaussian::findRoot]: initial interval does not bracket a root (fa=" << fa << ", fb=" << fb << ", pval=" << value << ", xMin=" << a << ", xMax=" << b <<  ")" << endl;
                return 999999;
            }

            Bool_t ac_equal(kFALSE);
            double fc = fb;
            double c(0),d(0),e(0);
            for(Int_t iter= 0; iter <= MaxIterations; iter++) {
         
                if ((fb < 0 && fc < 0) || (fb > 0 && fc > 0)) {
                  
                    // Rename a,b,c and adjust bounding interval d
                    ac_equal = kTRUE;
                    c = a;
                    fc = fa;
                    d = b - a;
                    e = b - a;
                }
              
                if (fabs (fc) < fabs (fb)) {
                  
                    ac_equal = kTRUE;
                    a = b;
                    b = c;
                    c = a;
                    fa = fb;
                    fb = fc;
                    fc = fa;
                }
             
                double tol = 0.5 * _tol * fabs(b);
                double m = 0.5 * (c - b);
             
             
                if (fb == 0 || fabs(m) <= tol) {
                  
                    //cout << "RooBrentRootFinder: iter = " << iter << " m = " << m << " tol = " << tol << endl ;
                    return b;
                }
              
                if (fabs (e) < tol || fabs (fa) <= fabs (fb)) {
                  
                    // Bounds decreasing too slowly: use bisection
                    d = m;
                    e = m;
                }
                else {
                  
                    // Attempt inverse cubic interpolation
                    double p, q, r;
                    double s = fb / fa;
                  
                    if (ac_equal) {
                    
                        p = 2 * m * s;
                        q = 1 - s;
                    }
                    else {
                        
                        q = fa / fc;
                        r = fb / fc;
                        p = s * (2 * m * q * (q - r) - (b - a) * (r - 1));
                        q = (q - 1) * (r - 1) * (s - 1);
                    }
                  
                    // Check whether we are in bounds
                    if (p > 0) {
                        
                        q = -q;
                    }
                    else {
                        
                        p = -p;
                    }
                  
                    double min1= 3 * m * q - fabs (tol * q);
                    double min2= fabs (e * q);
                    
                    if (2 * p < (min1 < min2 ? min1 : min2)) {
               
                        // Accept the interpolation
                        e = d;
                        d = p / q;
                    }
                    else {
               
                        // Interpolation failed: use bisection.
                        d = m;
                        e = m;
                    }
                }
                
                // Move last best guess to a
                a = b;
                fa = fb;
                // Evaluate new trial root
                if (fabs (d) > tol) {
                  
                    b += d;
                }
                else {
                  
                    b += (m > 0 ? +tol : -tol);
                }
                
                fb = this->evalCDF(b) - value;
            }
            
            
            if(recoil_verbose) cout << "[nGaussian::findRoot]: maximum iterations exceeded" << endl;
            //return b; // return best-estimate of root
            return 999999; // return default value, reject this 
        }
    
};



// recoil correction
Vec_d recoilCorrectionBinned(double pU1, double pU2, double qTbinIdx, double qT, int corrType=0, int corrParam=1) {
    
    Vec_d res(3, 0);
    //cout << pU1 << " " << pU2 << " " << qTbinIdx << endl; 
    
    res[1] = pU1;
    res[2] = pU2;
    res[0] = std::hypot(res[1], res[2]);
    //return res;
    
    TH3D *h3;
    int corrIdx = 1;
    
    double norm_para_data = 1;
    double norm_perp_data = 1;
    double norm_para_mc = 1;
    double norm_perp_mc = 1;
    
    //cout << "************* " << qT << endl;
    
    GaussianSum u1_data;
    h3 = recoil_hists["data_para"];
    corrIdx = 1;
    if(corrType == 1) corrIdx = corrParam;
    norm_para_data =  h3->GetBinContent(qTbinIdx+1, 10, 1);
    //u1_data.setNorm(norm_para_data);
    u1_data.addTerm(h3->GetBinContent(qTbinIdx+1, 1, corrIdx), h3->GetBinContent(qTbinIdx+1, 2, corrIdx), h3->GetBinContent(qTbinIdx+1, 3, corrIdx));
    u1_data.addTerm(h3->GetBinContent(qTbinIdx+1, 4, corrIdx), h3->GetBinContent(qTbinIdx+1, 5, corrIdx), h3->GetBinContent(qTbinIdx+1, 6, corrIdx));
    u1_data.addTerm(h3->GetBinContent(qTbinIdx+1, 7, corrIdx), h3->GetBinContent(qTbinIdx+1, 8, corrIdx), h3->GetBinContent(qTbinIdx+1, 9, corrIdx));
    
/*
    if(corrType == 1) {
        if(qTbinIdx == 40) {
            cout << corrIdx << endl;
            cout << h3->GetBinContent(qTbinIdx+1, 1, corrIdx) << "  " << h3->GetBinContent(qTbinIdx+1, 2, corrIdx) << "  " << h3->GetBinContent(qTbinIdx+1, 3, corrIdx) << endl;
            cout << h3->GetBinContent(qTbinIdx+1, 4, corrIdx) << "  " << h3->GetBinContent(qTbinIdx+1, 5, corrIdx) << "  " << h3->GetBinContent(qTbinIdx+1, 6, corrIdx) << endl;
            cout << h3->GetBinContent(qTbinIdx+1, 7, corrIdx) << "  " << h3->GetBinContent(qTbinIdx+1, 8, corrIdx) << "  " << h3->GetBinContent(qTbinIdx+1, 9, corrIdx) << endl;
        }
    }*/
    /*
    if(qT > 280) {
        cout << "************* " << qT << " " << qTbinIdx << endl;
    cout << h3->GetBinContent(qTbinIdx+1, 1, corrIdx) << "  " << h3->GetBinContent(qTbinIdx+1, 2, corrIdx) << "  " << h3->GetBinContent(qTbinIdx+1, 3, corrIdx) << endl;
    cout << h3->GetBinContent(qTbinIdx+1, 4, corrIdx) << "  " << h3->GetBinContent(qTbinIdx+1, 5, corrIdx) << "  " << h3->GetBinContent(qTbinIdx+1, 6, corrIdx) << endl;
    cout << h3->GetBinContent(qTbinIdx+1, 7, corrIdx) << "  " << h3->GetBinContent(qTbinIdx+1, 8, corrIdx) << "  " << h3->GetBinContent(qTbinIdx+1, 9, corrIdx) << endl;
    }
    */
    //if(qTbinIdx == 3) {
    //    cout << h3->GetBinContent(qTbinIdx+1, 1, corrIdx) << " " << h3->GetBinContent(qTbinIdx+1, 2, corrIdx) << " " << h3->GetBinContent(qTbinIdx+1, 3, corrIdx) << endl;
    //}
    
    
    
    GaussianSum u2_data;
    h3 = recoil_hists["data_perp"];
    corrIdx = 1;
    if(corrType == 2) corrIdx = corrParam;
    norm_perp_data =  h3->GetBinContent(qTbinIdx+1, 10, 1);
    //u2_data.setNorm(norm_perp_data);
    u2_data.addTerm(h3->GetBinContent(qTbinIdx+1, 1, corrIdx), h3->GetBinContent(qTbinIdx+1, 2, corrIdx), h3->GetBinContent(qTbinIdx+1, 3, corrIdx));
    u2_data.addTerm(h3->GetBinContent(qTbinIdx+1, 4, corrIdx), h3->GetBinContent(qTbinIdx+1, 5, corrIdx), h3->GetBinContent(qTbinIdx+1, 6, corrIdx));
    u2_data.addTerm(h3->GetBinContent(qTbinIdx+1, 7, corrIdx), h3->GetBinContent(qTbinIdx+1, 8, corrIdx), h3->GetBinContent(qTbinIdx+1, 9, corrIdx));
    
    /*
    // parametric
    
    GaussianSum u2_data;
    double mean = 0;
    double sigma1 = recoilParameterization_sigma(5.02232e+00, 1.39195e+00, 6.33123e-02, qT);
    double sigma2 = recoilParameterization_sigma(3.71235e-01, 1.07052e+02, 6.74092e-01, qT);
    double sigma3 = recoilParameterization_sigma(5.69628e-01, 1.19781e+02, 6.57171e-01, qT);
    double n1 = 0.3;
    double n2 = 0.6;
    double n3 = 1. - n1 - n2;
    u2_data.addTerm(mean, sigma1, n1);
    u2_data.addTerm(mean, sigma2, n2);
    u2_data.addTerm(mean, sigma3, n3);
    
    */
    
    ///////////////////

    
    GaussianSum u1_mc;
    h3 = recoil_hists["mc_para"];
    corrIdx = 1;
    if(corrType == 3) corrIdx = corrParam;
    norm_para_mc =  h3->GetBinContent(qTbinIdx+1, 10, 1);
    //u1_mc.setNorm(norm_para_mc);
    u1_mc.addTerm(h3->GetBinContent(qTbinIdx+1, 1, corrIdx), h3->GetBinContent(qTbinIdx+1, 2, corrIdx), h3->GetBinContent(qTbinIdx+1, 3, corrIdx));
    u1_mc.addTerm(h3->GetBinContent(qTbinIdx+1, 4, corrIdx), h3->GetBinContent(qTbinIdx+1, 5, corrIdx), h3->GetBinContent(qTbinIdx+1, 6, corrIdx));
    u1_mc.addTerm(h3->GetBinContent(qTbinIdx+1, 7, corrIdx), h3->GetBinContent(qTbinIdx+1, 8, corrIdx), h3->GetBinContent(qTbinIdx+1, 9, corrIdx));
    

    
    GaussianSum u2_mc;
    h3 = recoil_hists["mc_perp"];
    corrIdx = 1;
    if(corrType == 4) corrIdx = corrParam;
    norm_perp_mc =  h3->GetBinContent(qTbinIdx+1, 10, 1);
    //u2_mc.setNorm(norm_perp_mc);
    u2_mc.addTerm(h3->GetBinContent(qTbinIdx+1, 1, corrIdx), h3->GetBinContent(qTbinIdx+1, 2, corrIdx), h3->GetBinContent(qTbinIdx+1, 3, corrIdx));
    u2_mc.addTerm(h3->GetBinContent(qTbinIdx+1, 4, corrIdx), h3->GetBinContent(qTbinIdx+1, 5, corrIdx), h3->GetBinContent(qTbinIdx+1, 6, corrIdx));
    u2_mc.addTerm(h3->GetBinContent(qTbinIdx+1, 7, corrIdx), h3->GetBinContent(qTbinIdx+1, 8, corrIdx), h3->GetBinContent(qTbinIdx+1, 9, corrIdx));
    
    /*
    GaussianSum u2_mc;
    double mean_u2_mc = 0;
    double sigma1_u2_mc = recoilParameterization_sigma(3.01405e+00, 4.94889e+00, 1.54923e-01, qT);
    double sigma2_u2_mc = recoilParameterization_sigma(5.49714e+00, 8.45153e+00, 1.21679e-01, qT);
    double sigma3_u2_mc = recoilParameterization_sigma(8.08396e+00, 7.06204e+00, 9.56741e-02, qT);
    double sigma4_u2_mc = recoilParameterization_sigma(9.64523e-02, 1.20944e+02, 1.03714e+00, qT);
    double n1_u2_mc = 0.22;
    double n2_u2_mc = 0.5;
    double n3_u2_mc = 0.25;
    double n4_u2_mc = 1. - n1_u2_mc - n2_u2_mc - n3_u2_mc;
    u2_mc.addTerm(mean_u2_mc, sigma1_u2_mc, n1_u2_mc);
    u2_mc.addTerm(mean_u2_mc, sigma2_u2_mc, n2_u2_mc);
    u2_mc.addTerm(mean_u2_mc, sigma3_u2_mc, n3_u2_mc);
    u2_mc.addTerm(mean_u2_mc, sigma4_u2_mc, n4_u2_mc);
    */



    double pVal_u1_mc = u1_mc.evalCDF(pU1 + qT);
    double pVal_u2_mc = u2_mc.evalCDF(pU2);
    /*
    double pU1_mc = u1_mc.findRoot(pVal_u1_mc, -1000, 1000);
    double pU2_mc = u2_mc.findRoot(pVal_u2_mc, -500, 500);
    
    if(abs(pU2_mc-pU2) > 0.1) {
        
        cout << pU2_mc << " " << pU2 << endl;
    }*/
    
    /*
    if(qTbinIdx == 30) {
             cout << norm_para_data << " " << norm_perp_data << " " << norm_para_mc << " " << norm_perp_mc << " " << pVal_u1_mc << " " << pVal_u2_mc << endl;
    }
    else return res;
    */
    
    double pU1_data = u1_data.findRoot(pVal_u1_mc, -1000, 1000);
    double pU2_data = u2_data.findRoot(pVal_u2_mc, -500, 500);
    //double pU2_data=  999999;
    //double pU1_data=  999999;

  
    if(pU1_data == 999999) cout << "PARA ************* " << qT << endl;
    if(pU2_data == 999999) cout << "PERP ************* " << qT << endl;
    
    
    
    //if(qTbinIdx == 1) cout << norm_para_data << " " << norm_perp_data << " " << norm_para_mc << " " << norm_perp_mc << endl;

    if(pU1_data == 999999) pU1_data = pU1;
    else pU1_data -= qT;
    
    
    //else pU1_data -= qT;
    //else {
    //   if((pU1+qT)*(pU1_data+qT) > 0) pU1_data -= qT; // SS
    //   else pU1_data += qT; // OS
    //}
    //else pU1_data -= qT;
    
    if((pU1+qT)*(pU1_data+qT) < 0) {
        
        //cout << "OS " << qT << " " << qTbinIdx << " " << pU1+qT << " " << pU1_data+qT << " " << pU2 << " " << pU2_data << endl;
    }
    
    if(pU2_data == 999999) pU1_data = pU2;    
        
    // correct MC to DATA
    res[1] = pU1_data;
    res[2] = pU2_data;
    res[0] = std::hypot(res[1], res[2]);

	return res;
}


Vec_d recoilCorrectionBinnedWtoZ(int q, double pU1, double pU2, double qTbinIdx, double qT, int corrType=0, int corrParam=1) {
    
    Vec_d res(3, 0);
    //cout << pU1 << " " << pU2 << " " << qTbinIdx << endl; 
    
    res[1] = pU1;
    res[2] = pU2;
    res[0] = std::hypot(res[1], res[2]);
    return res;
    
    TH3D *h3;
    int corrIdx = 1;
    
    
    //cout << "************* " << qT << endl;
    
    GaussianSum u1_Z;
    h3 = recoil_hists_gen_Z["gen_para"];
    corrIdx = 1;
    if(corrType == 1) corrIdx = corrParam;
    u1_Z.addTerm(h3->GetBinContent(qTbinIdx+1, 1, corrIdx), h3->GetBinContent(qTbinIdx+1, 2, corrIdx), h3->GetBinContent(qTbinIdx+1, 3, corrIdx));
    u1_Z.addTerm(h3->GetBinContent(qTbinIdx+1, 4, corrIdx), h3->GetBinContent(qTbinIdx+1, 5, corrIdx), h3->GetBinContent(qTbinIdx+1, 6, corrIdx));
    u1_Z.addTerm(h3->GetBinContent(qTbinIdx+1, 7, corrIdx), h3->GetBinContent(qTbinIdx+1, 8, corrIdx), h3->GetBinContent(qTbinIdx+1, 9, corrIdx));
    
    
    GaussianSum u2_Z;
    h3 = recoil_hists_gen_Z["gen_perp"];
    corrIdx = 1;
    if(corrType == 2) corrIdx = corrParam;
    u2_Z.addTerm(h3->GetBinContent(qTbinIdx+1, 1, corrIdx), h3->GetBinContent(qTbinIdx+1, 2, corrIdx), h3->GetBinContent(qTbinIdx+1, 3, corrIdx));
    u2_Z.addTerm(h3->GetBinContent(qTbinIdx+1, 4, corrIdx), h3->GetBinContent(qTbinIdx+1, 5, corrIdx), h3->GetBinContent(qTbinIdx+1, 6, corrIdx));
    u2_Z.addTerm(h3->GetBinContent(qTbinIdx+1, 7, corrIdx), h3->GetBinContent(qTbinIdx+1, 8, corrIdx), h3->GetBinContent(qTbinIdx+1, 9, corrIdx));
    
    
    
    GaussianSum u1_W;
    if(q > 0) h3 = recoil_hists_gen_W["gen_plus_para"];
    else h3 = recoil_hists_gen_W["gen_minus_para"];
    corrIdx = 1;
    if(corrType == 3) corrIdx = corrParam;
    u1_W.addTerm(h3->GetBinContent(qTbinIdx+1, 1, corrIdx), h3->GetBinContent(qTbinIdx+1, 2, corrIdx), h3->GetBinContent(qTbinIdx+1, 3, corrIdx));
    u1_W.addTerm(h3->GetBinContent(qTbinIdx+1, 4, corrIdx), h3->GetBinContent(qTbinIdx+1, 5, corrIdx), h3->GetBinContent(qTbinIdx+1, 6, corrIdx));
    u1_W.addTerm(h3->GetBinContent(qTbinIdx+1, 7, corrIdx), h3->GetBinContent(qTbinIdx+1, 8, corrIdx), h3->GetBinContent(qTbinIdx+1, 9, corrIdx));
    

    
    GaussianSum u2_W;
    if(q > 0) h3 = recoil_hists_gen_W["gen_plus_perp"];
    else h3 = recoil_hists_gen_W["gen_minus_perp"];
    corrIdx = 1;
    if(corrType == 4) corrIdx = corrParam;
    u2_W.addTerm(h3->GetBinContent(qTbinIdx+1, 1, corrIdx), h3->GetBinContent(qTbinIdx+1, 2, corrIdx), h3->GetBinContent(qTbinIdx+1, 3, corrIdx));
    u2_W.addTerm(h3->GetBinContent(qTbinIdx+1, 4, corrIdx), h3->GetBinContent(qTbinIdx+1, 5, corrIdx), h3->GetBinContent(qTbinIdx+1, 6, corrIdx));
    u2_W.addTerm(h3->GetBinContent(qTbinIdx+1, 7, corrIdx), h3->GetBinContent(qTbinIdx+1, 8, corrIdx), h3->GetBinContent(qTbinIdx+1, 9, corrIdx));


    
    double pVal_u1_W = u1_W.evalCDF(pU1 + qT);
    double pVal_u2_W = u2_W.evalCDF(pU2);

    
    double pU1_Z = u1_Z.findRoot(pVal_u1_W, -1000, 1000);
    double pU2_Z = u2_Z.findRoot(pVal_u2_W, -500, 500);
  
    if(pU1_Z == 999999) cout << "************* " << qT << endl;
    if(pU2_Z == 999999) cout << "************* " << qT << endl;
    
    if(pU1_Z == 999999) pU1_Z = pU1;
    else pU1_Z -= qT;
    

    if(pU2_Z == 999999) pU2_Z = pU2;    
        
    // correct W to Z
    res[1] = pU1_Z;
    res[2] = pU2_Z;
    res[0] = std::hypot(res[1], res[2]);

	return res;
}


double qT_tf(double qT, bool transform) {
    if(transform) {
        return ((qT-0)-(recoil_correction_qTmax-qT))/(recoil_correction_qTmax-0); // map to [-1, 1]s
    }
    else return qT;
}



GaussianSum constructParametricGauss(string tag, double qT, int syst=-1) {
    //cout << "tag=" << tag << "qT=" << qT << endl;
    GaussianSum gauss;
    double totNorm = 0.;
    double mean, sigma, norm;
    string systLabel = "";
    
    //if(tag == "target_perp" and syst==1) syst = -1;
    if(syst != -1) systLabel = "_syst" + to_string(syst);
    if(recoil_param_nGauss.find(tag) != recoil_param_nGauss.end()) { // parametric
        //cout << " PARAMETIC " << endl; 
        int nGauss = recoil_param_nGauss[tag];
        for(int i=1; i<=nGauss; i++) {
            mean = recoil_param_funcs[tag + "_mean" + to_string(i) + systLabel]->Eval(qT_tf(qT, recoil_param_transform[tag + "_mean" + to_string(i)]));
           
            //if(tag == "target_para" or tag == "source_para") mean -= qT;
            sigma = recoil_param_funcs[tag + "_sigma" + to_string(i) + systLabel]->Eval(qT_tf(qT, recoil_param_transform[tag + "_sigma" + to_string(i)]));
            if(sigma <= 0) continue;

            if(i == nGauss) norm = 1.-totNorm;
            else {
                norm = recoil_param_funcs[tag + "_norm" + to_string(i) + systLabel]->Eval(qT_tf(qT, recoil_param_transform[tag + "_norm" + to_string(i)]));
                totNorm += norm;
            }
            if(totNorm > 1) {
                cout << "Total PDF norm exceeding 1, adjust (tag=" << tag << ", qT=" << qT << " syst" << syst << ")" << endl;
                return constructParametricGauss(tag, qT); // return the nominal
                norm = 1. - totNorm;
                if(norm < 0) norm = 0;
            }
        
            gauss.addTerm(mean, sigma, norm);
            //cout << " " << tag << " qT=" << qT << "  iGauss=" << i << " norm=" << norm << " mean=" << mean << " sigma=" << sigma << endl;
        }
    }
    else if (recoil_binned_nGauss.find(tag) != recoil_binned_nGauss.end()) {
        //cout << " BINNED " << endl; 
        // find the bin
        int qTbin = -1;
        int nGauss = recoil_binned_nGauss[tag];
        for(unsigned int i=0; i < recoil_binned_qTbins[tag].size()-1; i++) {
            if(qT >= recoil_binned_qTbins[tag].at(i) and qT < recoil_binned_qTbins[tag].at(i+1)) {
                qTbin = i;
                break;
            }
        }
        if(qTbin == -1) {
            qTbin = recoil_binned_qTbins[tag].size()-2;
            cout << "Bin outside range for tag=" << tag << " and qT=" << qT << endl; 
        }
        
        // construnct the Gauss
        for(int i=0; i<nGauss; i++) {
            mean = recoil_binned_components[tag].at(qTbin).at(i).at(0);
            sigma = recoil_binned_components[tag].at(qTbin).at(i).at(1);
            norm = recoil_binned_components[tag].at(qTbin).at(i).at(2);
            gauss.addTerm(mean, sigma, norm);
            //cout << " ---> qT=" << qT << " qTbin=" << qTbin << " nGauss=" << i << " mean=" << mean << " sigma=" << sigma << " norm=" << norm << endl;
        }  
    }
    else {
        exit(1);
    }
    return gauss;    
}


// recoil correction for Z (MC to DATA)
Vec_d recoilCorrectionParametric(double para, double perp, double qT, string systTag="", int systIdx=-1) {
    
    Vec_d res(3, 0);
    //cout << pU1 << " " << pU2 << " " << qTbinIdx << endl; 
    
    res[1] = para;
    res[2] = perp;
    res[0] = std::hypot(res[1], res[2]);
    
	if(!applyRecoilCorrection) return res;
    //if(qT > recoil_correction_qTmax) return res; // protection for high qT
    if(qT > recoil_correction_qTmax) qT = recoil_correction_qTmax; // protection for high qT

    GaussianSum data_para;
    GaussianSum data_perp;
    GaussianSum dy_para;
    GaussianSum dy_perp;
    
    if(systIdx != -1 and systTag == "target_para") data_para = constructParametricGauss("target_para", qT, systIdx);
    else if(systIdx != -1 and systTag == "target_para_bkg") data_para = constructParametricGauss("target_para_bkg", qT, systIdx);
    else data_para = constructParametricGauss("target_para", qT);
    if(systIdx != -1 and systTag == "target_perp") data_perp = constructParametricGauss("target_perp", qT, systIdx);
    else if(systIdx != -1 and systTag == "target_perp_bkg") data_perp = constructParametricGauss("target_perp_bkg", qT, systIdx);
    else data_perp = constructParametricGauss("target_perp", qT);
    if(systIdx != -1 and systTag == "source_para") dy_para = constructParametricGauss("source_para", qT, systIdx);
    else dy_para = constructParametricGauss("source_para", qT);
    if(systIdx != -1 and systTag == "source_perp") dy_perp = constructParametricGauss("source_perp", qT, systIdx);
    else dy_perp = constructParametricGauss("source_perp", qT);
    
    double dy_para_mean = dy_para.getWeightedMean();
    double data_para_mean = data_para.getWeightedMean();
    double dmean = data_para_mean - dy_para_mean;
    
    //data_para.resetMeans();
    //dy_para.resetMeans();
 
    double pVal_para_dy = dy_para.evalCDF(para);// dy_para_mean works
    double pVal_perp_dy = dy_perp.evalCDF(perp);
    
    double para_orig = dy_para.findRoot(pVal_para_dy, -1000, 1000);
    double perp_orig = dy_perp.findRoot(pVal_perp_dy, -1000, 1000);
    
    
    double para_corr = data_para.findRoot(pVal_para_dy, -1000, 1000);
    double perp_corr = data_perp.findRoot(pVal_perp_dy, -1000, 1000);

  
    if(para_corr == 999999 and recoil_verbose) cout << "PARA ************* " << qT << endl;
    if(perp_corr == 999999 and recoil_verbose) cout << "PERP ************* " << qT << endl;
    
    if(para_corr == 999999) para_corr = para;
    //else para_corr += data_para_mean; // -data_para_mean
    if(perp_corr == 999999) perp_corr = perp;    
        
    res[1] = para_corr + para - para_orig;
    res[2] = perp_corr + perp - perp_orig;
    res[0] = std::hypot(res[1]-qT, res[2]);

	return res;    
}


// recoil correction
Vec_recoil recoilCorrectionParametricUnc(double para, double perp, double qT, string tag) {
    
    /*
    if(tag == "target_para_bkg") {
        
        Vec_d ret_up = recoilCorrectionParametric(para, perp, qT, "target_para_bkg", 0);
        Vec_d ret_dw = recoilCorrectionParametric(para, perp, qT, "target_para_bkg", 1);
        Vec_d ret_nom = recoilCorrectionParametric(para, perp, qT, "target_para");
        
        if(ret_nom[1] > 60) cout << "qT=" << qT << " para=" << para << " para_nom=" << ret_nom[1] << " para_up=" << ret_up[1] <<  "para_dw=" << ret_dw[1] << endl;
    }
    */
    
    
    int nSysts = recoil_unc_no[tag];
    Vec_recoil res(nSysts);
    for(int iSyst = 0; iSyst<nSysts; iSyst++) {
        Vec_d ret = recoilCorrectionParametric(para, perp, qT, tag, iSyst);
        res[iSyst].para = ret[1];
        res[iSyst].perp = ret[2];
    }
    return res;
}


Vec_d recoilCorrectionParametricUncWeights(double eval, double qT, string tag_nom, string tag_pert) {
    
    GaussianSum nom;
    GaussianSum pert;
    
    if(qT > recoil_correction_qTmax) qT = recoil_correction_qTmax; // protection for high qT
    
    int nSysts = recoil_unc_no[tag_pert];
    Vec_d res(nSysts);
    nom = constructParametricGauss(tag_nom, qT);
    double nom_eval = nom.eval(eval);
    double w;
    for(int iSyst = 0; iSyst<nSysts; iSyst++) {

        pert = constructParametricGauss(tag_pert, qT, iSyst);
        w = pert.eval(eval)/nom_eval;
        if(w < 0.1 or w > 10) { // protection
            //if(recoil_verbose) cout << "Weight too large w=" << w << " tag_pert=" << tag_pert << " eval=" << eval <<" qT=" << qT << " res[iSyst]=" << res[iSyst] << " nom_eval=" << nom_eval << " pert_eval" << pert.eval(eval) << endl;
            w = 1;
        }
        if(nom_eval < 1e-10) res[iSyst] = 1;
        else res[iSyst] = w;
        //cout << tag_pert << " qT=" << qT << " res[iSyst]=" << res[iSyst] << " " << nom_eval << endl;
    }
    return res;
}





Vec_d METCorrection(double MET_pt, double MET_phi, double rec_para, double rec_perp, double V_pt, double V_phi) {

    Vec_d res(2, 0);

        
    double lMX = -V_pt*cos(V_phi) - rec_para*cos(V_phi) + rec_perp*sin(V_phi);
    double lMY = -V_pt*sin(V_phi) - rec_para*sin(V_phi) - rec_perp*cos(V_phi);

        
    //res[0] = sqrt(lMX*lMX + lMY*lMY);
    res[0] = sqrt((rec_para+V_pt)*(rec_para+V_pt) + rec_perp*rec_perp);
    res[1] = atan2(lMY, lMX);
    //if(lMX > 0) res[1] = atan(lMY/lMX);
    //else res[1] = (fabs(lMY)/lMY)*3.14159265 + atan(lMY/lMX);
  
	return res;
}

Vec_d METCorrectionGen(double rec_para, double rec_perp, double lep_pt, double lep_phi, double V_phi) {

    Vec_d res(2, 0);

        
    double lMX = -lep_pt*cos(lep_phi) - rec_para*cos(V_phi) + rec_perp*sin(V_phi);
    double lMY = -lep_pt*sin(lep_phi) - rec_para*sin(V_phi) - rec_perp*cos(V_phi);

        
    res[0] = sqrt(lMX*lMX + lMY*lMY);
    //res[0] = sqrt((rec_para+lep_pt)*(rec_para+lep_pt) + rec_perp*rec_perp);
    res[1] = atan2(lMY, lMX);
    //if(lMX > 0) res[1] = atan(lMY/lMX);
    //else res[1] = (fabs(lMY)/lMY)*3.14159265 + atan(lMY/lMX);
  
	return res;
}

/*
Vec_d METCorrection_StatUnc(double MET_pt, double MET_phi, double rec_para, double rec_perp, double V_pt, double V_phi) {

    Vec_d res(2, 0);

        
    double lMX = -V_pt*cos(V_phi) - rec_para*cos(V_phi) + rec_perp*sin(V_phi);
    double lMY = -V_pt*sin(V_phi) - rec_para*sin(V_phi) - rec_perp*cos(V_phi);

        
    res[0] = sqrt(lMX*lMX + lMY*lMY);
    if(lMX > 0) res[1] = atan(lMY/lMX);
    else res[1] = (fabs(lMY)/lMY)*3.14159265 + atan(lMY/lMX);
  
	return res;
}*/




Vec_d recoilCorrectionParametric_MET_pt_unc(Vec_recoil recoil, double V_pt, double V_phi) {
    
    int nBins = recoil.size();
    Vec_d res(nBins, -1e5);
    for(int iSyst=0; iSyst<nBins; iSyst++) {
       
        double lMX = -V_pt*cos(V_phi) - recoil[iSyst].para*cos(V_phi) + recoil[iSyst].perp*sin(V_phi);
        double lMY = -V_pt*sin(V_phi) - recoil[iSyst].para*sin(V_phi) - recoil[iSyst].perp*cos(V_phi);     
        res[iSyst] = sqrt(lMX*lMX + lMY*lMY);
    }
	return res;
}

Vec_d recoilCorrectionParametric_MET_phi_unc(Vec_recoil recoil, double V_pt, double V_phi) {
    
    int nBins = recoil.size();
    Vec_d res(nBins, -1e5);
    for(int iSyst=0; iSyst<nBins; iSyst++) {
       
        double lMX = -V_pt*cos(V_phi) - recoil[iSyst].para*cos(V_phi) + recoil[iSyst].perp*sin(V_phi);
        double lMY = -V_pt*sin(V_phi) - recoil[iSyst].para*sin(V_phi) - recoil[iSyst].perp*cos(V_phi);     
        res[iSyst] = atan2(lMY, lMX);
    }
	return res;
}

Vec_d recoilCorrectionParametric_para_qT_unc(Vec_recoil recoil, double qT) {
    
    int nBins = recoil.size();
    Vec_d res(nBins, -1e5);
    for(int iSyst=0; iSyst<nBins; iSyst++) res[iSyst] = recoil[iSyst].para - qT;
	return res;
}

Vec_d recoilCorrectionParametric_para_unc(Vec_recoil recoil, double qT) {
    
    int nBins = recoil.size();
    Vec_d res(nBins, -1e5);
    for(int iSyst=0; iSyst<nBins; iSyst++) res[iSyst] = recoil[iSyst].para;
	return res;
}

Vec_d recoilCorrectionParametric_perp_unc(Vec_recoil recoil, double qT) {
    
    int nBins = recoil.size();
    Vec_d res(nBins, -1e5);
    for(int iSyst=0; iSyst<nBins; iSyst++) res[iSyst] = recoil[iSyst].perp;
	return res;
}

Vec_d recoilCorrectionParametric_magn_unc(Vec_recoil recoil, double qT) {
    
    int nBins = recoil.size();
    Vec_d res(nBins, -1e5);
    for(int iSyst=0; iSyst<nBins; iSyst++) res[iSyst] = sqrt((recoil[iSyst].para-qT)*(recoil[iSyst].para-qT) + recoil[iSyst].perp*recoil[iSyst].perp);
	return res;
}

Vec_d recoilCorrectionParametric_mT_unc(Vec_d met_pt, Vec_d met_phi, float pt, float phi, float ptOther, float phiOther) {
    
    int nBins = met_pt.size();
    Vec_d res(nBins, -1e5);
    for(int iSyst=0; iSyst<nBins; iSyst++) {
        
        TVector2 pl = TVector2();
        pl.SetMagPhi(ptOther, phiOther);

        TVector2 met_wlike = TVector2();
        met_wlike.SetMagPhi(met_pt[iSyst], met_phi[iSyst]);
        met_wlike = pl + met_wlike;

        res[iSyst] = std::sqrt(2*pt*met_wlike.Mod()*(1-std::cos(phi-met_wlike.Phi())));  
    }
	return res;
}

Vec_d recoilCorrectionParametric_MET_pt_gen_unc(Vec_recoil recoil, double lep_pt, double lep_phi, double V_phi) {
    
    int nBins = recoil.size();
    Vec_d res(nBins, -1e5);
    for(int iSyst=0; iSyst<nBins; iSyst++) {
       
        double lMX = -lep_pt*cos(lep_phi) - recoil[iSyst].para*cos(V_phi) + recoil[iSyst].perp*sin(V_phi);
        double lMY = -lep_pt*sin(lep_phi) - recoil[iSyst].para*sin(V_phi) - recoil[iSyst].perp*cos(V_phi);   
        res[iSyst] = sqrt(lMX*lMX + lMY*lMY);
    }
	return res;
}

Vec_d recoilCorrectionParametric_MET_phi_gen_unc(Vec_recoil recoil, double lep_pt, double lep_phi, double V_phi) {
    
    int nBins = recoil.size();
    Vec_d res(nBins, -1e5);
    for(int iSyst=0; iSyst<nBins; iSyst++) {
       
        double lMX = -lep_pt*cos(lep_phi) - recoil[iSyst].para*cos(V_phi) + recoil[iSyst].perp*sin(V_phi);
        double lMY = -lep_pt*sin(lep_phi) - recoil[iSyst].para*sin(V_phi) - recoil[iSyst].perp*cos(V_phi);   
        res[iSyst] = atan2(lMY, lMX);
    }
	return res;
}

Vec_d recoilCorrectionParametric_mT_2_unc(Vec_d met_pt, Vec_d met_phi, float pt, float phi) {
    
    int nBins = met_pt.size();
    Vec_d res(nBins, -1e5);
    for(int iSyst=0; iSyst<nBins; iSyst++) res[iSyst] = std::sqrt(2*pt*met_pt[iSyst]*(1-std::cos(phi-met_phi[iSyst])));
	return res;
}


/*
Binned recoil correction functions
*/
Vec_d recoilCorrectionBinned_magn_StatUnc(Vec_d pU, int qTbinIdx) {
    
    int nqTbins = qTbins.size();
    Vec_d res(nqTbins, -1e5);
    res[qTbinIdx] = pU[0];
	return res;
}

Vec_d recoilCorrectionBinned_para_StatUnc(Vec_d pU, int qTbinIdx) {
    
    int nqTbins = qTbins.size();
    Vec_d res(nqTbins, -1e5);
    res[qTbinIdx] = pU[1];
	return res;
}

Vec_d recoilCorrectionBinned_para_qT_StatUnc(Vec_d pU, int qTbinIdx, double qT) {
    
    int nqTbins = qTbins.size();
    Vec_d res(nqTbins, -1e5); // BE CAREFUL, SHOULD BE UNDERFOW
    res[qTbinIdx] = pU[1] + qT;
	return res;
}

Vec_d recoilCorrectionBinned_perp_StatUnc(Vec_d pU, int qTbinIdx) {
    
    int nqTbins = qTbins.size();
    Vec_d res(nqTbins, -1e5); // // BE CAREFUL, SHOULD BE UNDERFOW
    res[qTbinIdx] = pU[2];
	return res;
}


Vec_d recoilCorrectionBinned_MET_pt_StatUnc(Vec_d pU, int qTbinIdx, double MET_pt, double MET_phi, double V_pt, double V_phi) {
    
    int nqTbins = qTbins.size();
    Vec_d res(nqTbins, -1e5);
    
            
    double lMX = -V_pt*cos(V_phi) - pU[1]*cos(V_phi) + pU[2]*sin(V_phi);
    double lMY = -V_pt*sin(V_phi) - pU[1]*sin(V_phi) - pU[2]*cos(V_phi);

    res[qTbinIdx] = sqrt(lMX*lMX + lMY*lMY);
	return res;
}

Vec_d recoilCorrectionBinned_MET_phi_StatUnc(Vec_d pU, int qTbinIdx, double MET_pt, double MET_phi, double V_pt, double V_phi) {
    
    int nqTbins = qTbins.size();
    Vec_d res(nqTbins, -1e5);
    
            
    double lMX = -V_pt*cos(V_phi) - pU[1]*cos(V_phi) + pU[2]*sin(V_phi);
    double lMY = -V_pt*sin(V_phi) - pU[1]*sin(V_phi) - pU[2]*cos(V_phi);

    res[qTbinIdx] = atan2(lMY, lMX);
	return res;
}






Vec_d recoilCorrectionBinned_MET_pt_gen_StatUnc(Vec_d pU, int qTbinIdx, double lep_pt, double lep_phi, double V_phi) {
    
    int nqTbins = qTbins.size();
    Vec_d res(nqTbins, -1e5);
    
            
    double lMX = -lep_pt*cos(lep_phi) - pU[1]*cos(V_phi) + pU[2]*sin(V_phi);
    double lMY = -lep_pt*sin(lep_phi) - pU[1]*sin(V_phi) - pU[2]*cos(V_phi);

    res[qTbinIdx] = sqrt(lMX*lMX + lMY*lMY);
	return res;
}

Vec_d recoilCorrectionBinned_MET_phi_gen_StatUnc(Vec_d pU, int qTbinIdx, double lep_pt, double lep_phi, double V_phi) {
    
    int nqTbins = qTbins.size();
    Vec_d res(nqTbins, -1e5);
    
            
    double lMX = -lep_pt*cos(lep_phi) - pU[1]*cos(V_phi) + pU[2]*sin(V_phi);
    double lMY = -lep_pt*sin(lep_phi) - pU[1]*sin(V_phi) - pU[2]*cos(V_phi);

    res[qTbinIdx] = atan2(lMY, lMX);
	return res;
}




Vec_d recoilCorrectionBinned_mt_StatUnc(Vec_d met_pt, Vec_d met_phi, int qTbinIdx, float pt, float phi, float ptOther, float phiOther) {
    
    int nqTbins = qTbins.size();
    Vec_d res(nqTbins, -1e5);
    
    TVector2 pl = TVector2();
    pl.SetMagPhi(ptOther,phiOther);

    TVector2 met_wlike = TVector2();
    met_wlike.SetMagPhi(met_pt[qTbinIdx], met_phi[qTbinIdx]);
    met_wlike = pl + met_wlike;

    res[qTbinIdx] = std::sqrt(2*pt*met_wlike.Mod()*(1-std::cos(phi-met_wlike.Phi())));  
	return res;
}


Vec_d recoilCorrectionBinned_mt_2_StatUnc(Vec_d met_pt, Vec_d met_phi, int qTbinIdx, float pt, float phi) {
    
    int nqTbins = qTbins.size();
    Vec_d res(nqTbins, -1e5);
  
    res[qTbinIdx] = std::sqrt(2*pt*met_pt[qTbinIdx]*(1-std::cos(phi-met_phi[qTbinIdx])));;
	return res;
}






// recoil correction
/*
Vec_d recoilCorrectionBinned_StatUnc(double pU1, double pU2, double qTbinIdx, int corrType, int corrIdx) {
    
    int nqTbins = qTbins.size();

    Vec_d t(3, -1);
    Vec_d res(nqTbins, -1); // per qT bin
    
    t = recoilCorrectionBinned(pU1, pU2, qTbinIdx, corrType, corrIdx);
    res[qTbinIdx] = t[0];
    
	return res;
}


Vec_recoilType recoilCorrectionBinned_StatUnc(double pU1, double pU2, double qTbinIdx, int corrType, int corrIdx) {
    
    int nqTbins = qTbins.size();

    Vec_d t(3, -1);
    
    RecoilType tmp;
    tmp.pu1 = 0;
    tmp.pu2 = 0;
    
    Vec_recoilType res(nqTbins, tmp);
    
    t = recoilCorrectionBinned(pU1, pU2, qTbinIdx, corrType, corrIdx);
    res[qTbinIdx].pu1 = t[1];
    res[qTbinIdx].pu1 = t[2];
    
	return res;
}

Vec_d recoilCorrectionBinned_magn_StatUnc(Vec_recoilType pU) {
    
    unsigned size = pU.size();
    Vec_d res(size, -1);
    
    for(unsigned int i=0; i < size; i++) {
        res[i] = sqrt(pU[i].pu1*pU[i].pu1 + pU[i].pu2*pU[i].pu2);
    }

	return res;
}

*/
Vec_d recoilCorrectionBinned_StatUnc(double pU1, double pU2, int qTbinIdx, int corrType, int corrIdx) {
    
    Vec_d res = recoilCorrectionBinned(pU1, pU2, qTbinIdx, corrType, corrIdx);
	return res;
}






}

#endif
