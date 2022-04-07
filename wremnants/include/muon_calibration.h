#include <ROOT/RVec.hxx>
#include "Math/GenVector/PtEtaPhiM4D.h"
#include "TFile.h"
#include "TTree.h"
#include <eigen3/Eigen/Dense>
#include <memory>

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

  using V = ROOT::Math::PtEtaPhiM4D<double>;

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

    const double qopcor = curvmomcor[0];
    const double lamcor = curvmomcor[1];
    const double phicor = curvmomcor[2];

    const double pcor = 1./std::abs(qopcor);
    const int qcor = std::copysign(1., qopcor);

    const double ptcor = pcor*std::cos(lamcor);

    const double thetacor = M_PI_2 - lamcor;
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

  calibration_uncertainty_helper(HIST &&hist, double minGenPt, double maxWeight) : hist_(std::make_shared<const HIST>(std::move(hist))),
                                                                                  minGenPt_(minGenPt),
                                                                                  maxWeight_(maxWeight) {}

  out_tensor_t operator() (const RVec<float> &pts,
                      const RVec<float> &etas,
                      const RVec<float> &phis,
                      const RVec<int> &charges,
                      const RVec<int> &matchedGenIdxs,
                      const RVec<RVec<float>> &momcovs,
                      const RVec<int> &selection,
                      const RVec<float> &genPts,
                      const RVec<float> &genEtas,
                      const RVec<float> &genPhis,
                      const RVec<int> &genPdgIds,
                      const RVec<int> &genStatusFlags,
                      double nominal_weight = 1.0) const {

    //TODO move this into a helper

    const std::size_t nmuons = pts.size();

    out_tensor_t res;
    res.setConstant(nominal_weight);

    for (std::size_t i = 0; i < nmuons; ++i) {
      if (!selection[i]) {
        continue;
      }

      if (pts[i] < 0.) {
        continue;
      }

      const int genidx = matchedGenIdxs[i];

      if (genidx < 0) {
        continue;
      }

      // matched genparts should be status 1 muons, but we need to explicitly check if they are prompt
      if (!(genStatusFlags[genidx] & 0x01)) {
        continue;
      }
      
      // minimum gen pt cut to avoid threshold effects from reco pt cut for track refit during nano production
      if (minGenPt_ >= 0. && genPts[genidx] < minGenPt_) {
        continue;
      }

      const int gencharge = genPdgIds[genidx] > 0 ? -1 : 1;

      res *= scale_res_weight(pts[i], etas[i], phis[i], charges[i], genPts[genidx], genEtas[genidx], genPhis[genidx], gencharge, momcovs[i]);
    }

    return res;

  }

private:

  out_tensor_t scale_res_weight(double pt, double eta, double phi, int charge, double genPt, double genEta, double genPhi, int genCharge, const RVec<float> &cov) const {

    const double theta = 2.*std::atan(std::exp(-double(eta)));
    const double lam = M_PI_2 - theta;
    const double p = double(pt)/std::sin(theta);
    const double qop = double(charge)/p;

    const Eigen::Vector3d parms(qop, lam, phi);

    const double gentheta = 2.*std::atan(std::exp(-double(genEta)));
    const double genlam = M_PI_2 - gentheta;
    const double genp = double(genPt)/std::sin(gentheta);
    const double genqop = double(genCharge)/genp;
    const double genqopt = double(genCharge)/genPt;

    const Eigen::Vector3d genparms(genqop, genlam, genPhi);

    const Eigen::Vector3d deltaparms = parms - genparms;

    const Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>> covMap(cov.data(), 3, 3);

    Eigen::Matrix<double, 3, 3> covd = covMap.cast<double>();
    // fill in lower triangular part of the matrix, which is stored as zeros to save space
    covd.triangularView<Eigen::Lower>() = covd.triangularView<Eigen::Upper>().transpose();
    
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
      parmvar[0] += A*genqop;
      parmvar[0] += -e*genqop/genPt;
      parmvar[0] += genCharge*M*genPt*genqop;

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
            std::cout << "genqop " << genqop << std::endl;
            std::cout << "qoperr " << std::sqrt(covd(0,0)) << std::endl;
            std::cout << "lam " << lam << std::endl;
            std::cout << "genlam " << genlam << std::endl;
            std::cout << "lamerr " << std::sqrt(covd(1,1)) << std::endl;
            std::cout << "phi " << phi << std::endl;
            std::cout << "genlam " << genPhi << std::endl;
            std::cout << "phierr " << std::sqrt(covd(2,2)) << std::endl;
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
        res(ivar, idownup) = std::min(weight, maxWeight_);

      }
    }

    return res;
  }

  std::shared_ptr<const HIST> hist_;
  double minGenPt_;
  double maxWeight_;

};


}
