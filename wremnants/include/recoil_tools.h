#include <ROOT/RVec.hxx>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <cmath>
#include "defines.h"
#include "tfliteutils.h"


namespace wrem {

using ROOT::VecOps::RVec;


ROOT::Math::PtEtaPhiMVector getIdx_dr2(const float& eta, const float& phi, const Vec_f& vec_eta, const Vec_f& vec_phi, const Vec_f& vec_pt, const Vec_f& vec_mass, const float dr2 = 0.09) {

  for (unsigned int jvec = 0; jvec < vec_eta.size(); ++jvec) {
    if (deltaR2(eta, phi, vec_eta[jvec], vec_phi[jvec]) < dr2) {
        
        return ROOT::Math::PtEtaPhiMVector(vec_pt[jvec], vec_eta[jvec], vec_phi[jvec], vec_mass[jvec]);
    }
  }
  auto k = ROOT::Math::PtEtaPhiMVector(0, 0, 0, 0);
  return k;

}

TVector2 get_z_mom(const float pt1, const float phi1, const float pt2, const float phi2) {

    TVector2 l1 = TVector2();
    l1.SetMagPhi(pt1, phi1);
  
    TVector2 l2 = TVector2();
    l2.SetMagPhi(pt2, phi2);

    TVector2 z = l1 + l2;
    return z;
}


Vec_d compute_recoil_from_met(const double met_pt, const double met_phi, const Vec_d lep_pt, const Vec_d lep_phi, const double v_pt, const double v_phi) {

    // computes the hadronic recoil, defined as -(MET + V), V being the vector sum of the reco (di)lepton system in the lab frame
    double pUx = met_pt*cos(met_phi) + lep_pt[0]*cos(lep_phi[0]) + lep_pt[1]*cos(lep_phi[1]); // recoil x in lab frame
    double pUy = met_pt*sin(met_phi) + lep_pt[0]*sin(lep_phi[0]) + lep_pt[1]*sin(lep_phi[1]); // recoil y in lab frame
    double Upara = - (pUx*cos(v_phi) + pUy*sin(v_phi));
    double Uperp = - (- pUx*sin(v_phi) + pUy*cos(v_phi));

    Vec_d res(2, 0);
    res[0] = Upara;
    res[1] = Uperp;

    return res;
}

Vec_d compute_recoil_from_met(const double met_pt, const double met_phi, const double lep_pt, const double lep_phi, const double v_pt, const double v_phi) {

    // computes the hadronic recoil, defined as -(MET + V), V being the vector sum of the reco (di)lepton system in the lab frame
    double pUx = met_pt*cos(met_phi) + lep_pt*cos(lep_phi); // recoil x in lab frame
    double pUy = met_pt*sin(met_phi) + lep_pt*sin(lep_phi); // recoil y in lab frame
    double Upara = - (pUx*cos(v_phi) + pUy*sin(v_phi));
    double Uperp = - (- pUx*sin(v_phi) + pUy*cos(v_phi));

    Vec_d res(2, 0);
    res[0] = Upara;
    res[1] = Uperp;

    return res;
}


Vec_d met_lepton_correction(const double met_pt, const double met_phi, const Vec_d lep_pt_uncorr, const Vec_d lep_phi_uncorr, const Vec_d lep_pt_corr, const Vec_d lep_phi_corr) {

    // correct MET for muon scale corrections (in lab frame pT-phi)
    TVector2 lep1_uncorr(lep_pt_uncorr[0]*cos(lep_phi_uncorr[0]), lep_pt_uncorr[0]*sin(lep_phi_uncorr[0]));
    TVector2 lep2_uncorr(lep_pt_uncorr[1]*cos(lep_phi_uncorr[1]), lep_pt_uncorr[1]*sin(lep_phi_uncorr[1]));
    TVector2 lep1_corr(lep_pt_corr[0]*cos(lep_phi_corr[0]), lep_pt_corr[0]*sin(lep_phi_corr[0]));
    TVector2 lep2_corr(lep_pt_corr[1]*cos(lep_phi_corr[1]), lep_pt_corr[1]*sin(lep_phi_corr[1]));

    TVector2 met(met_pt*cos(met_phi), met_pt*sin(met_phi));
    TVector2 met_corr = met + lep1_uncorr + lep2_uncorr - lep1_corr - lep2_corr;
    double met_pt_corr = met_corr.Mod();
    double met_phi_corr = met_corr.Phi_mpi_pi(met_corr.Phi());

    Vec_d res(2, 0);
    res[0] = met_pt_corr;
    res[1] = met_phi_corr;

    return res;
}

Vec_d met_lepton_correction(const double met_pt, const double met_phi, const double lep_pt_uncorr, const double lep_phi_uncorr, const double lep_pt_corr, const double lep_phi_corr) {

    TVector2 lep_uncorr = TVector2();
    lep_uncorr.SetMagPhi(lep_pt_uncorr, lep_phi_uncorr);

    TVector2 lep_corr = TVector2();
    lep_corr.SetMagPhi(lep_pt_corr, lep_phi_corr);

    TVector2 met = TVector2();
    met.SetMagPhi(met_pt, met_phi);

    TVector2 met_corr = met + lep_uncorr - lep_corr;

    Vec_d res(2, 0);
    res[0] = met_corr.Mod();
    res[1] = met_corr.Phi_mpi_pi(met_corr.Phi());

    //cout << met_pt << " " << met_phi << " " << res[0] << " " << res[1] << " " << lep_pt_uncorr << " " << lep_phi_uncorr << " " << lep_pt_corr << " " << lep_phi_corr << endl;

    return res;
}




Vec_d compute_met_from_recoil(const double rec_para, const double rec_perp, const Vec_d lep_pt, const Vec_d lep_phi, const double v_pt, const double v_phi) {

    Vec_d res(2, 0);
    double lMX = - (rec_para*cos(v_phi) - rec_perp*sin(v_phi) + lep_pt[0]*cos(lep_phi[0]) + lep_pt[1]*cos(lep_phi[1]));
    double lMY = - (rec_para*sin(v_phi) + rec_perp*cos(v_phi) + lep_pt[0]*sin(lep_phi[0]) + lep_pt[1]*sin(lep_phi[1]));

    res[0] = std::hypot(lMY, lMX);
    res[1] = std::atan2(lMY, lMX);

    return res;
}


// OLD function used for W
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

// OLD function
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

Vec_d compute_met_from_recoil(const double rec_para, const double rec_perp, const double lep_pt, const double lep_phi, const double v_pt, const double v_phi) {

    Vec_d res(2, 0);
    double lMX = - (rec_para*cos(v_phi) - rec_perp*sin(v_phi) + lep_pt*cos(lep_phi));
    double lMY = - (rec_para*sin(v_phi) + rec_perp*cos(v_phi) + lep_pt*sin(lep_phi));

    res[0] = std::hypot(lMX, lMY);
    res[1] = std::atan2(lMY, lMX);

    return res;
}

double compute_recoil_from_met_and_lepton(const double met_pt, const double met_phi, const double lep_pt, const double lep_phi) {

    double pUx = met_pt*cos(met_phi) + lep_pt*cos(lep_phi);
    double pUy = met_pt*sin(met_phi) + lep_pt*sin(lep_phi);
    return std::hypot(pUx, pUy);
}



ROOT::Math::PxPyPzEVector proxy_gen_v(const ROOT::Math::PtEtaPhiMVector lep_gen, const ROOT::Math::PtEtaPhiMVector antilep_gen, const ROOT::Math::PtEtaPhiMVector trg_lep, int trg_charge) {
    ROOT::Math::PxPyPzEVector res;

    if(trg_charge == 1) {
        res = ROOT::Math::PxPyPzEVector(lep_gen) + ROOT::Math::PxPyPzEVector(trg_lep);
    }
    else {
        res = ROOT::Math::PxPyPzEVector(antilep_gen) + ROOT::Math::PxPyPzEVector(trg_lep);
    }
    return res;
}


template <size_t N_SYST, typename First, typename... Rest>
Eigen::TensorFixedSize<double, Eigen::Sizes<N_SYST>> concatWeights(First nominal_weight, Rest... args) {
    Eigen::TensorFixedSize<double, Eigen::Sizes<N_SYST>> result;
    result.setZero();
    size_t index = 0;
    ((result(index++) = static_cast<double>(args)), ...);
    result = nominal_weight*result;
    return result;
}




}