#include <ROOT/RVec.hxx>
#include "Math/GenVector/PtEtaPhiM4D.h"
#include "TFile.h"
#include "TTree.h"
#include "TRandom.h"
#include <eigen3/Eigen/Dense>
#include <memory>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <typeinfo>
#include <algorithm>
#include "defines.h"
#include "tfliteutils.h"

namespace wrem {

using ROOT::VecOps::RVec;

template<typename T>
ROOT::VecOps::RVec<ROOT::VecOps::RVec<T>> splitNestedRVec(const ROOT::VecOps::RVec<T> &vec, const ROOT::VecOps::RVec<int> &counts) {
  using ROOT::VecOps::RVec;

  RVec<RVec<T>> res;
  res.reserve(counts.size());

  int total = 0;
  for (unsigned int i = 0; i < counts.size(); ++i) {
    const int count = counts[i];
    // avoid copies by adopting memory
    res.emplace_back(const_cast<T*>(vec.begin()) + total, count);
    total += count;
  }

  return res;
}

template <std::ptrdiff_t NJac = 3, std::ptrdiff_t NReplicas = 0>
class CVHCorrectorSingle {
public:

  using V = ROOT::Math::PtEtaPhiMVector;

  CVHCorrectorSingle(const std::string &filename) {
    TFile *fcor = TFile::Open(filename.c_str());

    TTree *idxmaptree = static_cast<TTree*>(fcor->Get("idxmaptree"));

    unsigned int idx;
    idxmaptree->SetBranchAddress("idx", &idx);

    idxmap_->reserve(idxmaptree->GetEntries());

    for (long long ientry=0; ientry < idxmaptree->GetEntries(); ++ientry) {
      idxmaptree->GetEntry(ientry);
      idxmap_->push_back(idx);
    }

    TTree *parmtree = static_cast<TTree*>(fcor->Get("parmtree"));

    float x;
    parmtree->SetBranchAddress("x", &x);

    std::array<float, NReplicas> xrep;
    if (NReplicas>0) {
      parmtree->SetBranchAddress("xreplicas", xrep.data());
    }

    x_->reserve(parmtree->GetEntries());
    xreplicas_->resize(parmtree->GetEntries(), NReplicas);

    for (long long ientry=0; ientry < parmtree->GetEntries(); ++ientry) {
      parmtree->GetEntry(ientry);
      x_->push_back(x);

      if (NReplicas > 0) {
        for (std::ptrdiff_t irep = 0; irep < NReplicas; ++irep) {
          (*xreplicas_)(ientry, irep) = xrep[irep];
        }
      }
    }
  }


  std::pair<V, int> operator() (float pt, float eta, float phi, int charge, const RVec<int> &idxs, const RVec<float> &jac) const {

    if (pt < 0.) {
      return std::make_pair<V, int>(V(), -99);
    }

    const Eigen::Matrix<double, 3, 1> curvmom = CurvMom(pt, eta, phi, charge);

    return CorMomCharge(curvmom, idxs, jac, *x_);
  }

protected:

  Eigen::Matrix<double, 3, 1> CurvMom(float pt, float eta, float phi, int charge) const {
    const double theta = 2.*std::atan(std::exp(-double(eta)));
    const double lam = M_PI_2 - theta;
    const double p = double(pt)/std::sin(theta);
    const double qop = double(charge)/p;

    return Eigen::Matrix<double, 3, 1>(qop, lam, phi);
  }

  template<typename U>
  std::pair<V, int> CorMomCharge(const Eigen::Matrix<double, 3, 1> &curvmom, const RVec<int> &idxs, const RVec<float> &jac, const U &parms) const {

    const auto nparms = idxs.size();

    const Eigen::Map<const Eigen::Matrix<float, NJac, Eigen::Dynamic, Eigen::RowMajor>> jacMap(jac.data(), NJac, nparms);

    Eigen::VectorXd xtrk(nparms);
    for (unsigned int i = 0; i < nparms; ++i) {
      xtrk[i] = parms[(*idxmap_)[idxs[i]]];
    }

    const Eigen::Matrix<double, 3, 1> curvmomcor = curvmom + jacMap.template topRows<3>().template cast<double>()*xtrk;

    if (curvmomcor.array().isNaN().any() || curvmomcor.array().isInf().any()) {
      return std::make_pair<V, int>(V(), -99);
    }

    const double qopcor = curvmomcor[0];
    const double lamcor = std::clamp(curvmomcor[1], -M_PI_2, M_PI_2);
    const double phicor = curvmomcor[2];

    const double pcor = 1./std::abs(qopcor);
    const int qcor = std::copysign(1., qopcor);

    const double ptcor = pcor*std::cos(lamcor);

    const double thetacor = std::clamp(M_PI_2 - lamcor, 0., M_PI);
    const double etacor = -std::log(std::tan(0.5*thetacor));

    return std::make_pair<V, int>(V(ptcor, etacor, phicor, wrem::muon_mass), int(qcor));

  }

  std::shared_ptr<std::vector<unsigned int>> idxmap_ = std::make_shared<std::vector<unsigned int>>();
  std::shared_ptr<std::vector<double>> x_ = std::make_shared<std::vector<double>>();
  std::shared_ptr<Eigen::MatrixXd> xreplicas_ = std::make_shared<Eigen::MatrixXd>();
};


class CVHCorrectorValidation : public CVHCorrectorSingle<5, 0> {
public:
  using base_t = CVHCorrectorSingle<5, 0>;

  using V = typename base_t::V;

  using base_t::base_t;

  std::pair<V, int> operator() (const RVec<float> &refParms, const RVec<unsigned int> &idxs, const RVec<float> &jac) const {

    const Eigen::Matrix<double, 3, 1> curvmom(refParms[0], refParms[1], refParms[2]);

    return CorMomCharge(curvmom, idxs, jac, *base_t::x_);
  }
};

template <std::ptrdiff_t NReplicas>
class CVHCorrectorReplicas : public CVHCorrectorSingle<5, NReplicas> {
public:
  using base_t = CVHCorrectorSingle<5, NReplicas>;

  using V = typename base_t::V;

  using base_t::base_t;

  std::array<std::pair<V, int>, NReplicas> operator() (const RVec<float> &refParms, const RVec<unsigned int> &idxs, const RVec<float> &jac) const {

    std::array<std::pair<V, int>, NReplicas> res;

    const Eigen::Matrix<double, 3, 1> curvmom(refParms[0], refParms[1], refParms[2]);

    for (std::ptrdiff_t irep = 0; irep < NReplicas; ++irep) {
      res[irep] = CorMomCharge(curvmom, idxs, jac, base_t::xreplicas_->col(irep));
    }

    return res;

  }
};

class CVHCorrector : public CVHCorrectorSingle<> {
public:


  CVHCorrector(const std::string &filename) : CVHCorrectorSingle(filename) {}
  CVHCorrector(const CVHCorrectorSingle &corsingle) : CVHCorrectorSingle(corsingle) {}

  RVec<std::pair<V, int>> operator () (const RVec<float> &ptv, const RVec<float> &etav, const RVec<float> &phiv, const RVec<int> &chargev, const RVec<RVec<int>> &idxsv, const RVec<RVec<float>> &jacv) {
    RVec<std::pair<V, int>> res;
    res.reserve(ptv.size());

    for (unsigned int i = 0; i < ptv.size(); ++i) {
      res.emplace_back(CVHCorrectorSingle::operator()(ptv[i], etav[i], phiv[i], chargev[i], idxsv[i], jacv[i]));
    }

    return res;
  }
};

template <typename HIST>
class calibration_uncertainty_helper {

public:

  using tensor_t = typename HIST::storage_type::value_type::tensor_t;
  static constexpr auto sizes = narf::tensor_traits<tensor_t>::sizes;
  static constexpr auto nvars = sizes[0];
  static constexpr auto nparms = sizes[1];

  using out_tensor_t = Eigen::TensorFixedSize<double, Eigen::Sizes<nvars, 2>>;

  calibration_uncertainty_helper(HIST &&hist, double minGenPt, double maxWeight) : 
    hist_(std::make_shared<const HIST>(std::move(hist))),
    minGenPt_(minGenPt),
    maxWeight_(maxWeight) {}

  out_tensor_t operator() (
    double qop,
    double pt,
    float eta,
    float phi,
    int charge,
    RVec<float> &momcov,
    double genQop,
    float genPt,
    float genEta,
    float genPhi,
    int genCharge,
    double nominal_weight) const {

    //TODO move this into a helper

    out_tensor_t res;
    res.setConstant(nominal_weight);
    res *= scale_res_weight(qop, pt, eta, phi, charge, momcov, genQop, genPt, genEta, genPhi, genCharge);
    return res;
  }

private:

  out_tensor_t scale_res_weight(
    double qop, double pt, double eta, double phi, int charge, const RVec<float> &cov,
    double genQop, double genPt, double genEta, double genPhi, int genCharge,
    bool abQop = false, bool fullParam = false
  ) const {

    const double theta = 2.*std::atan(std::exp(-double(eta)));
    const double lam = M_PI_2 - theta;
    const double p = double(pt)/std::sin(theta);
    const double Qop = double(charge)/p;
    Eigen::Vector3d parms(
      (abQop? qop : Qop),
      (fullParam? lam : 0),
      (fullParam? phi : 0)
    );

    const double gentheta = 2.*std::atan(std::exp(-double(genEta)));
    const double genlam = M_PI_2 - gentheta;
    const double genp = double(genPt)/std::sin(gentheta);
    const double genqop = double(genCharge)/genp;
    Eigen::Vector3d genparms(
      (abQop? genQop : genqop),
      (fullParam? genlam : 0),
      (fullParam? genPhi : 0)
    );

    const Eigen::Vector3d deltaparms = parms - genparms;

    const Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>> covMap(cov.data(), 3, 3);

    Eigen::Matrix<double, 3, 3> covd = covMap.cast<double>();

    if (fullParam) {
      // fill in lower triangular part of the matrix, which is stored as zeros to save space
      covd.triangularView<Eigen::Lower>() = covd.triangularView<Eigen::Upper>().transpose();
    } else {
      covd.row(0) << covd(0,0), 0, 0;
      covd.row(1) << 0, 1, 0;
      covd.row(2) << 0, 0, 1;

    }

    const Eigen::Matrix<double, 3, 3> covinv = covd.inverse();
    const double covdet = covd.determinant();

    const double lnp = -0.5*deltaparms.transpose()*covinv*deltaparms;

    // no need to initialize since all elements are explicit filled
    out_tensor_t res;

    const auto &varparms = hist_->at(hist_->template axis<0>().index(genEta)).data();

    for (std::ptrdiff_t ivar = 0; ivar < nvars; ++ivar) {
      Eigen::Vector3d parmvar = Eigen::Vector3d::Zero();
      Eigen::Matrix<double, 3, 3> covvar = Eigen::Matrix<double, 3, 3>::Zero();

      const double A = varparms(ivar, 0);
      const double e = varparms(ivar, 1);
      const double M = varparms(ivar, 2);
      const double sig = varparms(ivar, 3);

      // scale
      parmvar[0] += (abQop? A*genQop : A*genqop);
      parmvar[0] += (abQop? -e*genQop/genPt : -e*genqop/genPt);
      parmvar[0] += (abQop? genCharge*M*genPt*genQop : genCharge*M*genPt*genqop);

      // resolution
      covvar += sig*covd;

      for (std::ptrdiff_t idownup = 0; idownup < 2; ++idownup) {

        const double dir = idownup == 0 ? -1. : 1.;

        const Eigen::Vector3d deltaparmsalt = deltaparms + dir*parmvar;
        const Eigen::Matrix<double, 3, 3> covdalt = covd + dir*covvar;

        const Eigen::Matrix<double, 3, 3> covinvalt = covdalt.inverse();
        const double covdetalt = covdalt.determinant();

        const double lnpalt = -0.5*deltaparmsalt.transpose()*covinvalt*deltaparmsalt;

        const double weight = std::sqrt(covdet/covdetalt)*std::exp(lnpalt - lnp);

        if (false) {
          if (std::isnan(weight) || std::isinf(weight) || std::fabs(weight) == 0.) {
            std::cout << "invalid weight: " << weight << std::endl;
            std::cout << "pt " << pt << std::endl;
            std::cout << "genPt " << genPt << std::endl;
            std::cout << "qop " << qop << std::endl;
            std::cout << "lam " << lam << std::endl;
            std::cout << "genlam " << genlam << std::endl;
            std::cout << "phi " << phi << std::endl;
            std::cout << "genlam " << genPhi << std::endl;
            std::cout << "covdet " << covdet << std::endl;
            std::cout << "covdetalt " << covdet << std::endl;
            std::cout << "lnp " << lnp << std::endl;
            std::cout << "lnpalt " << lnpalt << std::endl;
            std::cout << "covd\n" << covd << std::endl;
            std::cout << "covdalt\n" << covdalt << std::endl;
            std::cout << "covinv\n" << covinv << std::endl;
            std::cout << "covinvalt\n" << covinvalt << std::endl;
          }
        }

        // protect against outliers
        // if (weight > 0.9998 && weight < 1.0002) {cout << "smearing weight is " << weight << "covd is " << covd << std::endl;}
        res(ivar, idownup) = std::min(weight, maxWeight_);
      }
    }
    
    return res;
  }
  std::shared_ptr<const HIST> hist_;
  double minGenPt_;
  double maxWeight_;

};

    float smearGenPt(RVec<float> cov, int charge, float pt, float theta) {
        float sigma2_qop = cov[0];
        double sigma2_pt = pow((pt * pt / (charge * sin(theta))), 2) * sigma2_qop;
        return gRandom -> Gaus(pt, sqrt(sigma2_pt));
    }
    
    double smearGenQop(RVec<float> cov, double qop) {
        double sigma2_qop = cov[0];
        return gRandom -> Gaus(qop, sqrt(sigma2_qop));
    }

    int getGoodGenMuons0IdxInReco(RVec<bool> goodMuons, RVec<bool> goodMuonsByGenTruth) {
        int first_goodGenMuon_idx_in_goodMuons = 0;
        for (int i = 0; i < goodMuonsByGenTruth.size(); i++) {
            if (goodMuonsByGenTruth[i]) {break;}
            else {first_goodGenMuon_idx_in_goodMuons += 1;}
        }
        int goodGenMuons0_idx_in_recos = 0;
        int num_goodMuons_found = 0;
        for (int i = 0; i < goodMuons.size(); i++) {
            if (goodMuons[i]) {
                num_goodMuons_found += 1;
                if (num_goodMuons_found == first_goodGenMuon_idx_in_goodMuons + 1) {
                    break;
                }
            }
            goodGenMuons0_idx_in_recos += 1;
        }
        return goodGenMuons0_idx_in_recos;
    }

    RVec<bool> filterRecoMuonsByGenTruth(
        RVec<bool> muonSelection,
        RVec<int> Muon_genPartIdx,
        RVec<int> GenPart_pdgId,
        RVec<int> GenPart_statusFlags,
        RVec<int> GenPart_status,
        bool requirePrompt = true
    ) {
        ROOT::VecOps::RVec<bool> res(muonSelection.size());
        for (int i = 0; i < muonSelection.size(); i++) {
            res[i] = (
                muonSelection[i] &&
                (!(requirePrompt) || GenPart_statusFlags[Muon_genPartIdx[i]] & 0x01) &&
                (abs(GenPart_pdgId[Muon_genPartIdx[i]]) == MUON_PDGID) &&
                (GenPart_status[Muon_genPartIdx[i]] == 1)
            );
        }
        return res;
    }

    RVec<bool> filterRecoMuonsByCovMat(
        RVec<bool> muonSelection,
        RVec<float> covMat,
        RVec<int> covMatCounts
    ) {
        ROOT::VecOps::RVec<bool> res(muonSelection.size());
        int covMat_start_idx = 0;
        for (int i = 0; i < muonSelection.size(); i++) {
            res[i] = (
                muonSelection[i] &&
                //covMatCounts[i] &&
                covMat[covMat_start_idx] > 0
            );
            covMat_start_idx += covMatCounts[i];
        }
        return res;
    }

    template<typename T> RVec<RVec<T>> getCovMatForSelectedRecoMuons(
        RVec<float> covmat,
        RVec<int> covmat_counts,
        RVec<bool> selection,
        int nMuons = -1 // limit the number of muons to be N; -1 to take all selected muons
    ) {
        if (covmat_counts.size() != selection.size()) {
            throw std::invalid_argument("selection masks should match the size of all RECO muons");
        }
        
        using ROOT::VecOps::RVec;
        RVec<RVec<T>> res;
        res.reserve(nMuons > 0 ? nMuons : 1);

        int covmat_start_idx = 0;
        for (int i = 0; i < covmat_counts.size(); i++) {
            if (selection[i]) {
                ROOT::VecOps::RVec<int> idxRange(covmat_counts[i]);
                for (int i = 0; i < idxRange.size(); i++) {
                    idxRange[i] = i + covmat_start_idx;
                }
                res.emplace_back(Take(covmat, idxRange));
            }
            covmat_start_idx += covmat_counts[i];
        }
        return res;
    }
        
    RVec<float> getCovMatForGoodMuons0(
        RVec<float> covmat,
        RVec<int> covmat_counts,
        RVec<bool> goodMuons,
        RVec<bool> goodMuonsByGenTruth
    ) {
        
        int goodGenMuons0_idx_in_recos = wrem::getGoodGenMuons0IdxInReco(
            goodMuons, goodMuonsByGenTruth
        );
        int covmat_start_idx = 0;
        for (int i = 0; i < goodGenMuons0_idx_in_recos; i++) {
            covmat_start_idx += covmat_counts[i];
        }
        ROOT::VecOps::RVec<int> idxRange(9);
        for (int i = 0; i < 9; i++) {
            idxRange[i] = i + covmat_start_idx;
        }
        return Take(covmat, idxRange);
    }
  
    using ROOT::VecOps::RVec;

    double calculateTheta(float eta) {
        return 2.*std::atan(std::exp(-double(eta)));
    }

    double calculateLam(float eta) {
        double theta = calculateTheta(eta);
        return M_PI_2 - theta;
    }

    double calculateQop(float pt, float eta, int charge) {
        double theta = calculateTheta(eta);
        return (charge * std::sin(theta) / double(pt));
    }

    double calculatePt(double qop, float eta, int charge) {
        double theta = calculateTheta(eta);
        return (charge * std::sin(theta) / qop);
    }

    double calculateQopUnc(float eta, int charge, double kUnc) {
        double theta = calculateTheta(eta);
        return (charge * std::sin(theta) * kUnc);
    }

    double calculateQopUnc(float pt, float eta, int charge, double ptUnc) {
        double theta = calculateTheta(eta);
        return ((-1. * charge * std::sin(theta)) / pow(pt, 2)) * ptUnc;
    }

    double calculateQopUnc(float pt, float eta, int charge, double AUnc, double eUnc, double MUnc) {
        float k = 1 / pt;
        double kUnc = (AUnc - eUnc * k) * k + charge * MUnc;
        return calculateQopUnc(eta, charge, kUnc);
    }

    Eigen::TensorFixedSize<double, Eigen::Sizes<2>> calculateSmearingWeightsDownUp(
        double genQop, double recoQop, double recoQopUnc, double sigma2Qop
    ) {
        const double lnp = -0.5 * pow((recoQop - genQop), 2) / sigma2Qop;
        Eigen::TensorFixedSize<double, Eigen::Sizes<2>> res;
        for (std::ptrdiff_t idownup = 0; idownup < 2; ++idownup) {
            const double dir = idownup == 0 ? -1. : 1.;
            const double lnpAlt = -0.5 * pow((recoQop + dir * recoQopUnc - genQop), 2) / sigma2Qop;
            const double weight = std::exp(lnpAlt - lnp);
            res(idownup) = weight;
        }
        return res;
    }

// jpsi correction central value for one muon
template <typename T>
class JpsiCorrectionsHelper {

public:
    using hist_t = T;
    using tensor_t = typename T::storage_type::value_type::tensor_t;

    JpsiCorrectionsHelper(T&& corrections) :
        correctionHist_(std::make_shared<const T>(std::move(corrections))) {}

    // helper for bin lookup which implements the compile-time loop over axes
    template<typename... Xs, std::size_t... Idxs>
    const tensor_t &get_tensor_impl(std::index_sequence<Idxs...>, const Xs&... xs) {
        return correctionHist_->at(correctionHist_->template axis<Idxs>().index(xs)...).data();
    }

    // variadic templated bin lookup
    template<typename... Xs>
    const tensor_t &get_tensor(const Xs&... xs) {
        return get_tensor_impl(std::index_sequence_for<Xs...>{}, xs...);
    }
    
    // for central value of pt
    float operator() (float cvhPt, float cvhEta, int charge) {
        const auto &params = get_tensor(cvhEta);
        const double A = params(0);
        const double e = params(1);
        const double M = params(2);
        double k = 1.0 / cvhPt;
        double magnetic = 1.0 + A;
        double material = -1.0 * e * k;
        double alignment = charge * M;
        double kCrctd = (magnetic + material) * k + alignment;
        return (1.0 / kCrctd);
    }

private:
    std::shared_ptr<const T> correctionHist_;
};

// jpsi corrections central value for multiple muons
template <typename T>
class JpsiCorrectionsRVecHelper : public JpsiCorrectionsHelper<T> {

using base_t = JpsiCorrectionsHelper<T>;

public:
    //inherit constructor
    using base_t::base_t;

    RVec<float> operator() (const RVec<float>& pts, const RVec<float> etas, RVec<int> charges) {
        RVec<float> corrected_pt(pts.size(), 0.);
        assert(etas.size() == pts.size() && etas.size() == charges.size());
        for (size_t i = 0; i < pts.size(); i++) {
            corrected_pt[i] = JpsiCorrectionsHelper<T>::operator()(pts[i], etas[i], charges[i]);
        }

        return corrected_pt;
    }
};

// unlike the JpsiCorrectionHelper which only deals with one muon,
// and uses inheritance to work on a vector of muons,
// this helper is already vectorized
template <typename T>
class JpsiCorrectionsUncHelper {

public:
    using hist_t = T;
    using tensor_t = typename T::storage_type::value_type::tensor_t;
    static constexpr auto sizes = narf::tensor_traits<tensor_t>::sizes;
    static constexpr auto nUnc = sizes[sizes.size() - 1]; // 1 for central value
    using out_tensor_t = Eigen::TensorFixedSize<double, Eigen::Sizes<nUnc, 2>>;

    JpsiCorrectionsUncHelper(T&& corrections) :
        correctionHist_(std::make_shared<const T>(std::move(corrections))) {}

    // helper for bin lookup which implements the compile-time loop over axes
    template<typename... Xs, std::size_t... Idxs>
    const tensor_t &get_tensor_impl(std::index_sequence<Idxs...>, const Xs&... xs) {
        return correctionHist_->at(correctionHist_->template axis<Idxs>().index(xs)...).data();
    }

    // variadic templated bin lookup
    template<typename... Xs>
    const tensor_t &get_tensor(const Xs&... xs) {
        return get_tensor_impl(std::index_sequence_for<Xs...>{}, xs...);
    }

    // for smearing weights derived from propagating uncs on A, e, M to uncs on qop

    out_tensor_t operator() (
        const RVec<double> &genQops, const RVec<float> &genPhis, const RVec<float> &genEtas,
        const RVec<double> &recoQops, const RVec<float> &recoPhis, const RVec<float> &recoEtas,
        const RVec<int> &recoCharges, const RVec<double> &recoPts, const RVec<RVec<float>> &covs,
        double nominal_weight = 1.0, bool fullParam = false
    ) {
        out_tensor_t res;
        res.setConstant(nominal_weight);

        const std::size_t nmuons = recoPts.size();

        for (std::size_t i = 0; i < nmuons; ++i) {
            res *= smearingWeight_oneMuon(
                genQops[i], genPhis[i], genEtas[i],
                recoQops[i], recoPhis[i], recoEtas[i], recoCharges[i], recoPts[i],
                covs[i], fullParam
            );
        }

        return res;
    }

private:
    out_tensor_t smearingWeight_oneMuon(
        double genQop,  float genPhi,  float genEta,
        double recoQop, float recoPhi, float recoEta, int recoCharge, float recoPt,  
        const RVec<float> &cov, bool fullParam = false
    ) {
        Eigen::Vector3d parms(
            recoQop,
            (fullParam? calculateLam(recoEta) : 0),
            (fullParam? recoPhi: 0)
        ); // (qop, lam, phi)

        Eigen::Vector3d genparms(
            genQop,
            (fullParam? calculateLam(genEta) : 0),
            (fullParam? genPhi: 0)
        );

        const Eigen::Vector3d deltaparms = parms - genparms;

        const Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>> covMap(cov.data(), 3, 3);

        Eigen::Matrix<double, 3, 3> covd = covMap.cast<double>();
    
        if (fullParam) {
            // fill in lower triangular part of the matrix, which is stored as zeros to save space
            covd.triangularView<Eigen::Lower>() = covd.triangularView<Eigen::Upper>().transpose();
        } else {
            covd.row(0) << covd(0,0), 0, 0;
            covd.row(1) << 0, 1, 0;
            covd.row(2) << 0, 0, 1;
        }

        const Eigen::Matrix<double, 3, 3> covinv = covd.inverse();
        const double covdet = covd.determinant();
    
        const double lnp = -0.5*deltaparms.transpose()*covinv*deltaparms;

        const auto &params = get_tensor(recoEta);

        // no need to initialize since all elements will be explicitly filled
        out_tensor_t res;
    
        for (std::ptrdiff_t ivar = 0; ivar < nUnc; ++ivar) {
            const double AUnc = params(0, ivar);
            const double eUnc = params(1, ivar);
            const double MUnc = params(2, ivar);
            double recoK = 1.0 /recoPt;
            double recoKUnc = (AUnc - eUnc * recoK) * recoK + recoCharge * MUnc;
            Eigen::Vector3d parmvar = Eigen::Vector3d::Zero();   
            parmvar[0] = recoCharge * std::sin(calculateTheta(recoEta)) * recoKUnc;
            for (std::ptrdiff_t idownup = 0; idownup < 2; ++idownup) {
                const double dir = idownup == 0 ? -1. : 1.;
                const Eigen::Vector3d deltaparmsalt = deltaparms + dir*parmvar;
                const Eigen::Matrix<double, 3, 3> covdalt = covd; //+ dir*covvar;
    
                const Eigen::Matrix<double, 3, 3> covinvalt = covdalt.inverse();
                const double covdetalt = covdalt.determinant();
    
                const double lnpalt = -0.5*deltaparmsalt.transpose()*covinvalt*deltaparmsalt;
    
                const double weight = std::sqrt(covdet/covdetalt)*std::exp(lnpalt - lnp);
    
                res(ivar, idownup) = weight;
            }
        }
        return res;
    }

    std::shared_ptr<const T> correctionHist_;
};

template <typename T, size_t NEtaBins>
class ZNonClosureParametrizedHelper {

public:
    using hist_t = T;
    using tensor_t = typename T::storage_type::value_type::tensor_t;
    using out_tensor_t = Eigen::TensorFixedSize<double, Eigen::Sizes<NEtaBins, 2>>;
    using out_tensor_chip_t = Eigen::TensorFixedSize<double, Eigen::Sizes<2>>;

    ZNonClosureParametrizedHelper(T&& corrections):
        correctionHist_(std::make_shared<const T>(std::move(corrections))) {}

    out_tensor_t operator() (
        const RVec<double> &genQops, const RVec<double> &recoQops, const RVec<float> &recoEtas,
        const RVec<double> &recoPts, const RVec<int> &recoCharges, const RVec<RVec<float>> &covs,
        double nominal_weight = 1.0, int calVarFlags = 7   //A = 1, e = 2, M = 4
    ) {
        out_tensor_t res;
        res.setConstant(nominal_weight);

        for (std::size_t i = 0; i < recoPts.size(); ++i) {
            const unsigned int iEta = std::clamp(
                correctionHist_ -> template axis<0>().index(recoEtas[i]), 0, (int)NEtaBins - 1
            ); // clip under/overflow bins 
            const auto &params = correctionHist_->at(iEta).data();
            const double sigma2Qop = covs[i][0];
    
            double recoK = 1.0 /recoPts[i];
            double recoKUnc = 0.0;
            enum calVarFlagsScheme {AFlag = 1, eFlag = 2, MFlag = 4};
            if (calVarFlags & AFlag) {
                const double AUnc = params(0);
                recoKUnc += AUnc * recoK;
            }
            if (calVarFlags & eFlag) {
                const double eUnc = params(1);
                recoKUnc += -1.0 * eUnc * recoK * recoK;
            }
            if (calVarFlags & MFlag) {
                const double MUnc = params(2);
                recoKUnc += recoCharges[i] * MUnc;
            }
            double recoQopUnc = recoCharges[i] * std::sin(calculateTheta(recoEtas[i])) * recoKUnc;
    
            out_tensor_chip_t smearing_weight = \
                calculateSmearingWeightsDownUp(genQops[i], recoQops[i], recoQopUnc, sigma2Qop);
            res(iEta, 0) *= smearing_weight(0);
            res(iEta, 1) *= smearing_weight(1);
        }
        return res;
    }
private:
    std::shared_ptr<const T> correctionHist_;
};

// the parametrized Z non-closure helper without the de-correlation in output nuisances
// TODO: vectorize the helper for multiple muons
template <typename T, size_t NEtaBins>
class ZNonClosureParametrizedHelperCorl {

public:
    using hist_t = T;
    using tensor_t = typename T::storage_type::value_type::tensor_t;
    using out_tensor_t = Eigen::TensorFixedSize<double, Eigen::Sizes<2>>;

    ZNonClosureParametrizedHelperCorl(T&& corrections):
        correctionHist_(std::make_shared<const T>(std::move(corrections))) {}

    out_tensor_t operator() (
        const RVec<double> &genQops, const RVec<double> &recoQops, const RVec<float> &recoEtas,
        const RVec<double> &recoPts, const RVec<int> &recoCharges, const RVec<RVec<float>> &covs,
        double nominal_weight = 1.0, int calVarFlags = 7   //A = 1, e = 2, M = 4
    ) {
        out_tensor_t res;
        res.setConstant(nominal_weight);

        for (std::size_t i = 0; i < recoPts.size(); ++i) {
            const unsigned int iEta = std::clamp(
                correctionHist_ -> template axis<0>().index(recoEtas[i]), 0, (int)NEtaBins - 1
            ); // clip under/overflow bins
            const auto &params = correctionHist_->at(iEta).data();
            const double sigma2Qop = covs[i][0];
    
            double recoK = 1.0 / recoPts[i];
            double recoKUnc = 0.0;
            enum calVarFlagsScheme {AFlag = 1, eFlag = 2, MFlag = 4};
            if (calVarFlags & AFlag) {
                const double AUnc = params(0);
                recoKUnc += AUnc * recoK;
            }
            if (calVarFlags & eFlag) {
                const double eUnc = params(1);
                recoKUnc += -1.0 * eUnc * recoK * recoK;
            }
            if (calVarFlags & MFlag) {
                const double MUnc = params(2);
                recoKUnc += recoCharges[i] * MUnc;
            }
            double recoQopUnc = recoCharges[i] * std::sin(calculateTheta(recoEtas[i])) * recoKUnc;
            res *= calculateSmearingWeightsDownUp(genQops[i], recoQops[i], recoQopUnc, sigma2Qop);
        }
        return res;
    }

private:
    std::shared_ptr<const T> correctionHist_;
};

template <typename T, size_t NEtaBins, size_t NPtBins>
class ZNonClosureBinnedHelper {

public:
    using hist_t = T;
    using out_tensor_t = Eigen::TensorFixedSize<double, Eigen::Sizes<NEtaBins, NPtBins, 2>>;
    using out_tensor_chip_t = Eigen::TensorFixedSize<double, Eigen::Sizes<2>>;

    ZNonClosureBinnedHelper(T&& corrections) :
        correctionHist_(std::make_shared<const T>(std::move(corrections))) {}

    // smaering weights for Z non-closure number on pt 
    out_tensor_t operator() (
        const RVec<double> &genQops, const RVec<double> &recoQops, const RVec<float> &recoEtas,
        const RVec<double> &recoPts, const RVec<int> &recoCharges, const RVec<RVec<float>> &covs,
        double nominal_weight = 1.0
    ) {
        out_tensor_t res;
        res.setConstant(nominal_weight);

        for (std::size_t i = 0; i < recoPts.size(); ++i) {
            const unsigned int iEta = std::clamp(
                correctionHist_->template axis<0>().index(recoEtas[i]), 0, int(NEtaBins) - 1
            );
            const unsigned int iPt = std::clamp(
                correctionHist_->template axis<1>().index(recoPts[i]), 0, int(NPtBins) - 1
            );
            const double nonClosure = correctionHist_->at(iEta, iPt).value();
            const double recoKUnc = (nonClosure - 1) * (1 / recoPts[i]);
            const double recoQopUnc = calculateQopUnc(recoEtas[i], recoCharges[i], recoKUnc);
            const double sigma2Qop = covs[i][0];
            out_tensor_chip_t smearing_weight = \
                calculateSmearingWeightsDownUp(genQops[i], recoQops[i], recoQopUnc, sigma2Qop);
            res(iEta, iPt, 0) *= smearing_weight(0);
            res(iEta, iPt, 1) *= smearing_weight(1);
        }
        return res;
    }

private:
    std::shared_ptr<const T> correctionHist_;
};

// the binned Z non-closure helper without the de-correlation in output nuisances
// TODO: vectorize the helper for multiple muons
template <typename T, size_t NEtaBins, size_t NPtBins>
class ZNonClosureBinnedHelperCorl {

public:
    using hist_t = T;
    using out_tensor_t = Eigen::TensorFixedSize<double, Eigen::Sizes<2>>;

    ZNonClosureBinnedHelperCorl(T&& corrections) :
        correctionHist_(std::make_shared<const T>(std::move(corrections))) {}

    // smaering weights for Z non-closure number on pt 
    out_tensor_t operator() (
        const RVec<double> &genQops, const RVec<double> &recoQops, const RVec<float> &recoEtas,
        const RVec<double> &recoPts, const RVec<int> &recoCharges, const RVec<RVec<float>> &covs,
        double nominal_weight = 1.0
    ) {
        out_tensor_t res;
        res.setConstant(nominal_weight);

        for (std::size_t i = 0; i < recoPts.size(); ++i) {
            const unsigned int iEta = std::clamp(
                correctionHist_->template axis<0>().index(recoEtas[i]), 0, int(NEtaBins) - 1
            );
            const unsigned int iPt = std::clamp(
                correctionHist_->template axis<1>().index(recoPts[i]), 0, int(NPtBins) - 1
            );
            const double nonClosure = correctionHist_->at(iEta, iPt).value();
            const double recoKUnc = (nonClosure - 1) * (1 / recoPts[i]);
            const double recoQopUnc = calculateQopUnc(recoEtas[i], recoCharges[i], recoKUnc);
            const double sigma2Qop = covs[i][0];
            res *= calculateSmearingWeightsDownUp(genQops[i], recoQops[i], recoQopUnc, sigma2Qop);
        }
        return res;
    }

private:
    std::shared_ptr<const T> correctionHist_;
};

template <typename T>
class CorrectionHelperBase {

public:
    CorrectionHelperBase(unsigned int nSlots, T&& corrections) :
        myRndGens(nSlots),
        correctionHist_(std::make_shared<const T>(std::move(corrections)))
        {
            init_random_generators();
        }


    CorrectionHelperBase(unsigned int nSlots, T&& corrections, T&& uncertainties) : 
        myRndGens(nSlots),
        correctionHist_(std::make_shared<const T>(std::move(corrections))),
        uncertaintyHist_(std::make_shared<const T>(std::move(uncertainties))
        ){
            init_random_generators();
        }

    void init_random_generators(){
        int seed = 1; // not 0 because seed 0 has a special meaning
        for (auto &&gen : myRndGens)
        {
            gen.SetSeed(seed++);
        }
    }

    // helper for bin lookup which implements the compile-time loop over axes
    template<typename... Xs, std::size_t... Idxs>
    const float get_value_impl(std::index_sequence<Idxs...>, const Xs&... xs) {
      return correctionHist_->at(correctionHist_->template axis<Idxs>().index(xs)...);
    }

    // variadic templated bin lookup
    template<typename... Xs>
    const float get_value(const Xs&... xs) {
        return get_value_impl(std::index_sequence_for<Xs...>{}, xs...);
    }

    // helper for bin lookup which implements the compile-time loop over axes
    template<typename... Xs, std::size_t... Idxs>
    const float get_error_impl(std::index_sequence<Idxs...>, const Xs&... xs) {
      return uncertaintyHist_->at(uncertaintyHist_->template axis<Idxs>().index(xs)...);
    }

    // variadic templated bin lookup
    template<typename... Xs>
    const float get_error(const Xs&... xs) {
        return get_error_impl(std::index_sequence_for<Xs...>{}, xs...);
    }

    virtual float get_correction(unsigned int slot, float pt, float eta) {return 0;}

    float get_random(unsigned int slot, float mean, float std){
        return myRndGens[slot].Gaus(mean, std);
    }


    RVec<float> operator() (unsigned int slot, const RVec<float>& pts, const RVec<float>& etas) {
        RVec<float> corrected_pt(pts.size(), 0.);
        assert(etas.size() == pts.size());
        for (size_t i = 0; i < pts.size(); i++) {
            corrected_pt[i] = get_correction(slot, pts[i], etas[i]);
        }
        return corrected_pt;
    }


private:
    std::vector<TRandom3> myRndGens; 

    std::shared_ptr<const T> correctionHist_;
    std::shared_ptr<const T> uncertaintyHist_;
};


template <typename T>
class BiasCalibrationHelper : public CorrectionHelperBase<T> {

using base_t = CorrectionHelperBase<T>;

public:
    //inherit constructor
    using base_t::base_t;

    float get_correction(unsigned int slot, float pt, float eta) override {
        const double bias = base_t::get_value(eta, pt);
        const double error = base_t::get_error(eta, pt);

        if(error>0.)
            return pt*(1.0 + base_t::get_random(slot, bias, error));
        else 
            return pt*(1.0+bias);
    }
};

template<typename HIST>
class SmearingHelper{

public:
    SmearingHelper(HIST&& smearings) :
        hash_(std::hash<std::string>()("SmearingHelper")),
        hsmear_(std::make_shared<const HIST>(std::move(smearings)))
        {}

    RVec<float> operator() (const unsigned int run, const unsigned int lumi, const unsigned long long event, const RVec<float>& pts, const RVec<float>& etas) const {
        std::seed_seq seq{hash_, std::size_t(run), std::size_t(lumi), std::size_t(event)};
        std::mt19937 rng(seq);

        RVec<float> corrected_pt(pts.size(), 0.);
        for (size_t i = 0; i < pts.size(); i++) {
            const float pt = pts[i];
            const float eta = etas[i];

            const double sigmasq = narf::get_value(*hsmear_, eta, pt);

            if (sigmasq > 0.) {
                const double k = 1./pt;
                std::normal_distribution gaus{k, std::sqrt(sigmasq)*k};
                const double ksmeared = gaus(rng);
                corrected_pt[i] = 1./ksmeared;
            }
            else {
                corrected_pt[i] = pt;
            }
        }
        return corrected_pt;
    }

private:
    const std::size_t hash_;
    std::shared_ptr<const HIST> hsmear_;
};

template<typename HIST>
class SmearingHelperParametrized{

public:
    SmearingHelperParametrized(HIST&& smearings) :
        hash_(std::hash<std::string>()("SmearingHelperParametrized")),
        hsmear_(std::make_shared<const HIST>(std::move(smearings)))
        {}

    RVec<float> operator() (const unsigned int run, const unsigned int lumi, const unsigned long long event, const RVec<float>& pts, const RVec<float>& etas) const {
        std::seed_seq seq{hash_, std::size_t(run), std::size_t(lumi), std::size_t(event)};
        std::mt19937 rng(seq);

        RVec<float> corrected_pt(pts.size(), 0.);
        for (size_t i = 0; i < pts.size(); i++) {
            const float pt = pts[i];
            const float eta = etas[i];

            auto const &resolution_parms = narf::get_value(*hsmear_, eta).data();

            std::array<double, 2> resolution_data_mc;
            for (std::size_t idatamc = 0; idatamc < 2; ++idatamc) {
                const double a = resolution_parms(0, idatamc);
                const double c = resolution_parms(1, idatamc);
                const double b = resolution_parms(2, idatamc);
                const double d = resolution_parms(3, idatamc);

                resolution_data_mc[idatamc] = a + c*pt*pt + b/(1. + d/pt/pt);
            }

            const double sigmasq = resolution_data_mc[0] - resolution_data_mc[1];


            if (sigmasq > 0.) {
                const double k = 1./pt;
                std::normal_distribution gaus{k, std::sqrt(sigmasq)*k};
                const double ksmeared = gaus(rng);
                corrected_pt[i] = 1./ksmeared;
            }
            else {
                corrected_pt[i] = pt;
            }
        }
        return corrected_pt;
    }

    const HIST &hist() const { return *hsmear_; }

private:
    const std::size_t hash_;
    std::shared_ptr<const HIST> hsmear_;
};


template<typename HIST, std::size_t NVar>
class SmearingUncertaintyHelper{

public:
    using out_tensor_t = Eigen::TensorFixedSize<double, Eigen::Sizes<NVar>>;

    SmearingUncertaintyHelper(HIST&& smearings) :
        hsmear_(std::make_shared<const HIST>(std::move(smearings)))
        {}

    out_tensor_t operator() (const RVec<float>& pts, const RVec<float>& etas, const RVec<std::pair<double, double>> &weights, const double nominal_weight = 1.0) const {

        out_tensor_t res;
        res.setConstant(nominal_weight);

        for (size_t i = 0; i < pts.size(); i++) {
            const float pt = pts[i];
            const float eta = etas[i];
            const double dweightdsigmasq = weights[i].second;

            const double p = pt*std::cosh(eta);
            const double qopsq = 1./p/p;

            auto const &dsigmarelsq = narf::get_value(*hsmear_, eta, pt).data();
            const out_tensor_t iweight = 1. + dweightdsigmasq*dsigmarelsq*qopsq;
            const out_tensor_t iweight_clamped = wrem::clip_tensor(iweight, 10.);

            res *= iweight_clamped;
        }
        return res;
    }

private:
    std::shared_ptr<const HIST> hsmear_;
};


template<typename HISTNOM, typename HISTVAR, std::size_t NVar>
class SmearingUncertaintyHelperParametrized : public SmearingHelperParametrized<HISTNOM> {

public:
    using base_t = SmearingHelperParametrized<HISTNOM>;
    using out_tensor_t = Eigen::TensorFixedSize<double, Eigen::Sizes<NVar>>;

    SmearingUncertaintyHelperParametrized(const base_t &helper, HISTVAR&& hvar) :
        base_t(helper),
        hvar_(std::make_shared<const HISTVAR>(std::move(hvar)))
        {}

    out_tensor_t operator() (const RVec<float>& pts, const RVec<float>& etas, const RVec<std::pair<double, double>> &weights, const double nominal_weight = 1.0) const {

        out_tensor_t res;
        res.setConstant(nominal_weight);

        for (size_t i = 0; i < pts.size(); i++) {
            const float pt = pts[i];
            const float eta = etas[i];
            const double dweightdsigmasq = weights[i].second;

            const double k = 1./pt;

            auto const &resolution_parms = narf::get_value(base_t::hist(), eta).data();

            const std::size_t idatamc = 0;
            const double anom = resolution_parms(0, idatamc);
            const double cnom = resolution_parms(1, idatamc);
            const double bnom = resolution_parms(2, idatamc);
            const double dnom = resolution_parms(3, idatamc);

            const double sigmasqnom =  anom + cnom*pt*pt + bnom/(1. + dnom/pt/pt);

            auto const &resolution_parms_var = narf::get_value(*hvar_, eta).data();

            out_tensor_t iweight;
            for (std::size_t ivar = 0; ivar < NVar; ++ivar) {
                const double avar = resolution_parms_var(0, ivar);
                const double cvar = resolution_parms_var(1, ivar);
                const double bvar = resolution_parms_var(2, ivar);
                const double dvar = resolution_parms_var(3, ivar);

                const double sigmasqvar =  avar + cvar*pt*pt + bvar/(1. + dvar/pt/pt);

                const double dsigmarelsq = sigmasqvar - sigmasqnom;

                iweight(ivar) = 1. + dweightdsigmasq*dsigmarelsq*k*k;
            }

            const out_tensor_t iweight_clamped = wrem::clip_tensor(iweight, 10.);

            res *= iweight_clamped;
        }
        return res;
    }

private:
    std::shared_ptr<const HISTNOM> hnom_;
    std::shared_ptr<const HISTVAR> hvar_;
};

template<typename T>
double test_smearing_uncertainty_helper(const T &helper) {
    RVec<float> pts = { 20., 30., 40.};
    RVec<float> etas = { -1.4, 0., 1.4 };
    RVec<std::pair<double, double>> response_weights;

    response_weights.emplace_back(1e-3, 1e-3);
    response_weights.emplace_back(1e-3, 1e-3);
    response_weights.emplace_back(1e-3, 1e-3);

    auto const res = helper(pts, etas, response_weights, 1.0);

    return res(0);

}

template <typename T>
class JpsiCorrectionsUncHelperSplines {
public:
    using hist_t = T;
    using tensor_t = typename T::storage_type::value_type::tensor_t;
    static constexpr auto sizes = narf::tensor_traits<tensor_t>::sizes;
    static constexpr auto nUnc = sizes[sizes.size() - 1];
    using out_tensor_t = Eigen::TensorFixedSize<double, Eigen::Sizes<nUnc, 2>>;

    JpsiCorrectionsUncHelperSplines(const std::string &filename, T&& corrections) : 
        correctionHist_(std::make_shared<const T>(std::move(corrections))) {
    }

    // helper for bin lookup which implements the compile-time loop over axes
    template<typename... Xs, std::size_t... Idxs>
    const tensor_t &get_tensor_impl(std::index_sequence<Idxs...>, const Xs&... xs) {
        return correctionHist_->at(correctionHist_->template axis<Idxs>().index(xs)...).data();
    }

    // variadic templated bin lookup
    template<typename... Xs>
    const tensor_t &get_tensor(const Xs&... xs) {
        return get_tensor_impl(std::index_sequence_for<Xs...>{}, xs...);
    }

    out_tensor_t operator() (
        const RVec<float> &recPts, const RVec<float> &recEtas, const RVec<int> &recCharges,
        const RVec<float> &genPts, const RVec<float> &genEtas, const RVec<int> &genCharges,
        const RVec<std::pair<double, double>> &response_weights, double nominal_weight = 1.0) {

        auto const nmuons = recPts.size();

        out_tensor_t alt_weights_all;
        alt_weights_all.setConstant(nominal_weight);

        for (std::size_t i = 0; i < nmuons; ++i) {
            auto const &recPt = recPts[i];
            auto const &recEta = recEtas[i];
            auto const &recCharge = recCharges[i];

            auto const &genPt = genPts[i];
            auto const &genEta = genEtas[i];
            auto const &genCharge = genCharges[i];

            const double qopgen = genCharge*1./(genPt*std::cosh(genEta));

            const double dweightdqop = response_weights[i].first;
            const auto &params = get_tensor(recEta);
            out_tensor_t delta_qop;

            for (std::ptrdiff_t ivar = 0; ivar < nUnc; ++ivar) {
                const double AUnc = params(0, ivar);
                const double eUnc = params(1, ivar);
                const double MUnc = params(2, ivar);  
                double recoQopUnc = calculateQopUnc(recPt, recEta, recCharge, AUnc, eUnc, MUnc);
                for (std::ptrdiff_t idownup = 0; idownup < 2; ++idownup) {
                    const double dir = idownup == 0 ? -1. : 1.;
                    delta_qop(ivar, idownup) = recoQopUnc * dir;
                }
            }

            const out_tensor_t alt_weights = dweightdqop * delta_qop + 1.;
            const out_tensor_t alt_weights_clamped = wrem::clip_tensor(alt_weights, 10.);

            // total weight is the product over all the muons
            alt_weights_all *= alt_weights_clamped;
        }
        return alt_weights_all;
    }

private:
    std::shared_ptr<const T> correctionHist_;
};

template <typename T, size_t NEtaBins>
class ZNonClosureParametrizedHelperSplines {

public:
    using hist_t = T;
    using out_tensor_t = Eigen::TensorFixedSize<double, Eigen::Sizes<NEtaBins, 2>>;
    using out_tensor_chip_t = Eigen::TensorFixedSize<double, Eigen::Sizes<2>>;

    ZNonClosureParametrizedHelperSplines(const std::string &filename, T&& corrections) : 
        correctionHist_(std::make_shared<const T>(std::move(corrections))) {
    }

    out_tensor_t operator() (
        const RVec<float> &recPts, const RVec<float> &recEtas, const RVec<int> &recCharges,
        const RVec<float> &genPts, const RVec<float> &genEtas, const RVec<int> &genCharges,
        const RVec<std::pair<double, double>> &response_weights, double nominal_weight = 1.0,
        int calVarFlags = 7 //A = 1, e = 2, M = 4
    ) {

        auto const nmuons = recPts.size();
        out_tensor_t res;
        res.setConstant(nominal_weight);

        for (std::size_t i = 0; i < nmuons; ++i) {
            auto const &recPt = recPts[i];
            auto const &recEta = recEtas[i];
            auto const &recCharge = recCharges[i];

            auto const &genPt = genPts[i];
            auto const &genEta = genEtas[i];
            auto const &genCharge = genCharges[i];

            const double qopgen = genCharge*1./(genPt*std::cosh(genEta));

            const double &dweightdqop = response_weights[i].first;

            out_tensor_chip_t delta_qop;

            const unsigned int iEta = std::clamp(
                correctionHist_ -> template axis<0>().index(recEtas[i]), 0, (int)NEtaBins - 1
            ); // clip under/overflow bins 
            const auto &params = correctionHist_->at(iEta).data();
            double recoK = 1.0 /recPts[i];
            double recoKUnc = 0.0;
            enum calVarFlagsScheme {AFlag = 1, eFlag = 2, MFlag = 4};
            if (calVarFlags & AFlag) {
                const double AUnc = params(0);
                recoKUnc += AUnc * recoK;
            }
            if (calVarFlags & eFlag) {
                const double eUnc = params(1);
                recoKUnc += -1.0 * eUnc * recoK * recoK;
            }
            if (calVarFlags & MFlag) {
                const double MUnc = params(2);
                recoKUnc += recCharges[i] * MUnc;
            }
            double recoQopUnc = recCharge * std::sin(calculateTheta(recEta)) * recoKUnc;

            for (std::ptrdiff_t idownup = 0; idownup < 2; ++idownup) {
                const double dir = idownup == 0 ? -1. : 1.;
                delta_qop(idownup) = recoQopUnc * dir;
            }

            const out_tensor_chip_t alt_weights = dweightdqop*delta_qop + 1.;
            const out_tensor_chip_t alt_weights_clamped = wrem::clip_tensor(alt_weights, 10.);

            // total weight is the product over all the muons
            res(iEta, 0) *= alt_weights_clamped(0);
            res(iEta, 1) *= alt_weights_clamped(1);
        }
        return res;
    }

private:
    std::shared_ptr<const T> correctionHist_;
};

// the parametrized Z non-closure helper without the de-correlation in output nuisances
// TODO: vectorize the helper for multiple muons
template <typename T, size_t NEtaBins>
class ZNonClosureParametrizedHelperSplinesCorl {

public:
    using hist_t = T;
    using tensor_t = typename T::storage_type::value_type::tensor_t;
    using out_tensor_t = Eigen::TensorFixedSize<double, Eigen::Sizes<2>>;

    ZNonClosureParametrizedHelperSplinesCorl(T&& corrections):
        correctionHist_(std::make_shared<const T>(std::move(corrections))) {}

    out_tensor_t operator() (
        const RVec<float> &recPts, const RVec<float> &recEtas, const RVec<int> &recCharges,
        const RVec<float> &genPts, const RVec<float> &genEtas, const RVec<int> &genCharges,
        const RVec<std::pair<double, double>> &response_weights, double nominal_weight = 1.0,
        int calVarFlags = 7 //A = 1, e = 2, M = 4
    ) {

        auto const nmuons = recPts.size();
        out_tensor_t res;
        res.setConstant(nominal_weight);

        for (std::size_t i = 0; i < nmuons; ++i) {
            auto const &recPt = recPts[i];
            auto const &recEta = recEtas[i];
            auto const &recCharge = recCharges[i];

            auto const &genPt = genPts[i];
            auto const &genEta = genEtas[i];
            auto const &genCharge = genCharges[i];

            const double qopgen = genCharge*1./(genPt*std::cosh(genEta));

            const double &dweightdqop = response_weights[i].first;

            out_tensor_t delta_qop;

            const unsigned int iEta = std::clamp(
                correctionHist_ -> template axis<0>().index(recEtas[i]), 0, (int)NEtaBins - 1
            ); // clip under/overflow bins 
            const auto &params = correctionHist_->at(iEta).data();
            double recoK = 1.0 /recPts[i];
            double recoKUnc = 0.0;
            enum calVarFlagsScheme {AFlag = 1, eFlag = 2, MFlag = 4};
            if (calVarFlags & AFlag) {
                const double AUnc = params(0);
                recoKUnc += AUnc * recoK;
            }
            if (calVarFlags & eFlag) {
                const double eUnc = params(1);
                recoKUnc += -1.0 * eUnc * recoK * recoK;
            }
            if (calVarFlags & MFlag) {
                const double MUnc = params(2);
                recoKUnc += recCharges[i] * MUnc;
            }
            double recoQopUnc = recCharge * std::sin(calculateTheta(recEta)) * recoKUnc;

            for (std::ptrdiff_t idownup = 0; idownup < 2; ++idownup) {
                const double dir = idownup == 0 ? -1. : 1.;
                delta_qop(idownup) = recoQopUnc * dir;
            }

            const out_tensor_t alt_weights = dweightdqop*delta_qop + 1.;
            const out_tensor_t alt_weights_clamped = wrem::clip_tensor(alt_weights, 10.);

            // total weight is the product over all the muons
            res *= alt_weights_clamped;
        }
        return res;
    }

private:
    std::shared_ptr<const T> correctionHist_;
};

template <typename T, size_t NEtaBins, size_t NPtBins>
class ZNonClosureBinnedHelperSplines {

public:
    using hist_t = T;
    using out_tensor_t = Eigen::TensorFixedSize<double, Eigen::Sizes<NEtaBins, NPtBins, 2>>;
    using out_tensor_chip_t = Eigen::TensorFixedSize<double, Eigen::Sizes<2>>;

    ZNonClosureBinnedHelperSplines(const std::string &filename, T&& corrections) : 
        correctionHist_(std::make_shared<const T>(std::move(corrections))) {
    }

    out_tensor_t operator() (
        const RVec<float> &recPts, const RVec<float> &recEtas, const RVec<int> &recCharges,
        const RVec<float> &genPts, const RVec<float> &genEtas, const RVec<int> &genCharges,
        const RVec<std::pair<double, double>> &response_weights, double nominal_weight = 1.0
    ) {

        auto const nmuons = recPts.size();

        out_tensor_t res;
        res.setConstant(nominal_weight);

        for (std::size_t i = 0; i < nmuons; ++i) {
            auto const &recPt = recPts[i];
            auto const &recEta = recEtas[i];
            auto const &recCharge = recCharges[i];

            auto const &genPt = genPts[i];
            auto const &genEta = genEtas[i];
            auto const &genCharge = genCharges[i];

            const double qopgen = genCharge*1./(genPt*std::cosh(genEta));

            const double &dweightdqop = response_weights[i].first;

            out_tensor_chip_t delta_qop;

            const unsigned int iEta = std::clamp(
                correctionHist_->template axis<0>().index(recEtas[i]), 0, int(NEtaBins) - 1
            );
            const unsigned int iPt = std::clamp(
                correctionHist_->template axis<1>().index(recPts[i]), 0, int(NPtBins) - 1
            );
            const double nonClosure = correctionHist_->at(iEta, iPt).value();
            const double recoKUnc = (nonClosure - 1) * (1 / recPts[i]);
            const double recoQopUnc = calculateQopUnc(recEtas[i], recCharges[i], recoKUnc);

            for (std::ptrdiff_t idownup = 0; idownup < 2; ++idownup) {
                const double dir = idownup == 0 ? -1. : 1.;
                delta_qop(idownup) = recoQopUnc * dir;
            }

            const out_tensor_chip_t alt_weights = dweightdqop*delta_qop + 1.;
            const out_tensor_chip_t alt_weights_clamped = wrem::clip_tensor(alt_weights, 10.);

            // total weight is the product over all the muons
            res(iEta, iPt, 0) *= alt_weights_clamped(0);
            res(iEta, iPt, 1) *= alt_weights_clamped(1);
        }
        return res;
    }

private:
    std::shared_ptr<const T> correctionHist_;
};

class SplinesDifferentialWeightsHelper {
public:
    using out_t = RVec<std::pair<double, double> >;

    SplinesDifferentialWeightsHelper(const std::string &filename) : 
        helper_(
            std::make_shared<narf::tflite_helper>(
                filename, "serving_default", ROOT::GetThreadPoolSize()
            )
        ) {}

    out_t operator() (
        const RVec<float> &recPts, const RVec<float> &recEtas, const RVec<int> &recCharges,
        const RVec<float> &genPts, const RVec<float> &genEtas, const RVec<int> &genCharges
    ) {

        auto const nmuons = recPts.size();
        out_t res;
        res.reserve(nmuons);

        for (std::size_t i = 0; i < nmuons; ++i) {

            auto const &recPt = recPts[i];
            auto const &recEta = recEtas[i];
            auto const &recCharge = recCharges[i];

            auto const &genPt = genPts[i];
            auto const &genEta = genEtas[i];
            auto const &genCharge = genCharges[i];

            const double qoprec = recCharge*1./(recPt*std::cosh(recEta));
            const double qopgen = genCharge*1./(genPt*std::cosh(genEta));

            // compute qoprec/qopgen needed to compute the weights
            const double qopr = qoprec/qopgen;

            // fill input tensors
            Eigen::TensorFixedSize<double, Eigen::Sizes<>> genPt_tensor;
            Eigen::TensorFixedSize<double, Eigen::Sizes<>> genEta_tensor;
            Eigen::TensorFixedSize<double, Eigen::Sizes<>> genCharge_tensor;
            Eigen::TensorFixedSize<double, Eigen::Sizes<>> qopr_tensor;

            genPt_tensor(0) = genPt;
            genEta_tensor(0) = genEta;
            genCharge_tensor(0) = genCharge;
            qopr_tensor(0) = qopr;

            // define output tensors
            Eigen::TensorFixedSize<double, Eigen::Sizes<>> dweightdmu_tensor;
            Eigen::TensorFixedSize<double, Eigen::Sizes<>> dweightdsigmasq_tensor;


            // build tuples of inputs and outputs (use std::tie so the tuples contain references to the tensors above)
            auto const inputs = std::tie(genPt_tensor, genEta_tensor, genCharge_tensor, qopr_tensor);
            auto outputs = std::tie(dweightdmu_tensor, dweightdsigmasq_tensor);

            // call the tensorflow lite model to fill the outputs
            (*helper_)(inputs, outputs);

            // get the output values
            const double dweightdscale = dweightdmu_tensor(0)/qopgen;
            const double dweightdsigmasq = dweightdsigmasq_tensor(0)/qopgen/qopgen;

            res.emplace_back(dweightdscale, dweightdsigmasq);
        }
        return res;
    }

private:
    std::shared_ptr<narf::tflite_helper> helper_;
};

class SmearingHelperSimpleWeight {

public:
    SmearingHelperSimpleWeight(const double sigmarel) : sigmarel_(sigmarel) {}

    double operator() (const RVec<float> &recPts, const RVec<float> &recEtas, const RVec<int> &recCharges, const RVec<std::pair<double, double>> &weights, const double nominal_weight = 1.0) {

        double res = nominal_weight;
        for (std::size_t i = 0; i < recPts.size(); ++i) {
            const double pt = recPts[i];
            const double eta = recEtas[i];
            const double charge = recCharges[i];
            const double dweightdsigmasq = weights[i].second;

            const double qop = charge/pt/std::cosh(eta);

            const double dsigmasq = sigmarel_*sigmarel_*qop*qop;

            const double dweight = dweightdsigmasq*dsigmasq;
            const double iweight = std::clamp(1. + dweight, -10., 10.);
            res *= iweight;
        }

        return res;
    }


private:
    double sigmarel_;

};

class SmearingHelperSimpleTransform {

public:
    SmearingHelperSimpleTransform(const double sigmarel) : sigmarel_(sigmarel) {}

    RVec<double> operator() (const RVec<float> &recPts, const RVec<float> &recEtas, const RVec<int> &recCharges, const RVec<std::pair<double, double>> &weights) {

        RVec<double> res;
        res.reserve(recPts.size());
        for (std::size_t i = 0; i < recPts.size(); ++i) {
            const double pt = recPts[i];
            const double eta = recEtas[i];
            const double charge = recCharges[i];
            const double dweightdscale = weights[i].first;

            const double qop = charge/pt/std::cosh(eta);

            const double dsigmasq = sigmarel_*sigmarel_*qop*qop;

            const double qopout = qop + 0.5*dweightdscale*dsigmasq;
            const double pout = std::fabs(1./qopout);

            const double ptout = pout/std::cosh(eta);


            res.emplace_back(ptout);
        }

        return res;
    }


private:
    double sigmarel_;
};

class SmearingHelperSimple {

public:
    SmearingHelperSimple(const double sigmarel, const unsigned int nslots = 1) : sigmarel_(sigmarel) {
        const unsigned int nslotsactual = std::max(nslots, 1U);
        rng_.reserve(nslotsactual);
        auto const hash = std::hash<std::string>()("SmearingHelperSimple");
        for (std::size_t islot = 0; islot < nslotsactual; ++islot) {
            std::seed_seq seq{hash, islot};
            rng_.emplace_back(seq);
        }
    }

    RVec<double> operator() (const unsigned int slot, const RVec<float> &recPts, const RVec<float> &recEtas, const RVec<int> &recCharges) {

        RVec<double> res;
        res.reserve(recPts.size());
        for (std::size_t i = 0; i < recPts.size(); ++i) {
            const double pt = recPts[i];
            const double eta = recEtas[i];
            const double charge = recCharges[i];

            const double qop = charge/pt/std::cosh(eta);

            const double dsigma = sigmarel_*qop;

            std::normal_distribution gaus{qop, dsigma};

            const double qopout = gaus(rng_[slot]);
            const double pout = std::fabs(1./qopout);

            const double ptout = pout/std::cosh(eta);


            res.emplace_back(ptout);
        }

        return res;
    }


private:
    double sigmarel_;
    std::vector<mt19937> rng_;

};

template<std::size_t N>
class SmearingHelperSimpleMulti {

public:
    SmearingHelperSimpleMulti(const double sigmarel) : hash_(std::hash<std::string>()("SmearingHelperSimpleMulti")), sigmarel_(sigmarel) {}

    RVec<double> operator() (const unsigned int run, const unsigned int lumi, const unsigned long long event,
                            const RVec<float> &recPts, const RVec<float> &recEtas, const RVec<int> &recCharges) {

        std::seed_seq seq{hash_, std::size_t(run), std::size_t(lumi), std::size_t(event)};
        std::mt19937 rng(seq);

        RVec<double> res;
        res.reserve(N*recPts.size());
        for (std::size_t i = 0; i < recPts.size(); ++i) {
            const double pt = recPts[i];
            const double eta = recEtas[i];
            const double charge = recCharges[i];

            const double qop = charge/pt/std::cosh(eta);

            const double dsigma = sigmarel_*qop;

            std::normal_distribution gaus{qop, dsigma};

            for (std::size_t irep = 0; irep < N; ++irep) {
                const double qopout = gaus(rng);

                res.emplace_back(qopout);
            }
        }

        return res;
    }


private:
    const std::size_t hash_;
    const double sigmarel_;
};

template <typename T>
RVec<T> replicate_rvec(const RVec<T> &v, std::size_t n) {
    RVec<T> res;
    res.reserve(n*v.size());

    for (const T &el : v) {
        for (std::size_t irep=0; irep < n; ++irep) {
            res.emplace_back(el);
        }
    }

    return res;
}

class ScaleHelperSimpleWeight {

public:
    ScaleHelperSimpleWeight(const double scalerel) : scalerel_(scalerel) {}

    double operator() (const RVec<float> &recPts, const RVec<float> &recEtas, const RVec<int> &recCharges, const RVec<std::pair<double, double>> &weights, const double nominal_weight = 1.0) {

        double res = nominal_weight;
        for (std::size_t i = 0; i < recPts.size(); ++i) {
            const double pt = recPts[i];
            const double eta = recEtas[i];
            const double charge = recCharges[i];
            const double dweightdmu = weights[i].first;

            const double qop = charge/pt/std::cosh(eta);

            const double dmu = scalerel_*qop;

            const double dweight = dweightdmu*dmu;
            const double iweight = std::clamp(1. + dweight, -10., 10.);
            res *= iweight;
        }

        return res;
    }


private:
    double scalerel_;

};

}
