#ifndef WREMNANTS_EtaPtCorrelatedEfficiency_h
#define WREMNANTS_EtaPtCorrelatedEfficiency_h

#include <TF1.h>
#include <TFile.h>
#include <TH2D.h>
#include <TH3D.h>
#include <TMath.h>
#include <TROOT.h>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

namespace wrem {

// ================================================
// Some functions to be used for EtaPtCorrelatedEfficiency
// ================================================
// TODO:
// put functions in another header file
// write a base class to make all functions derive from, at least for
// polynomials write a generic class for polynomials
class pol3_custom {
public:
  pol3_custom() {};
  pol3_custom(const double &xMin, const double &xRange) {
    xMinNorm_ = xMin;
    xRangeNorm_ = xRange;
  };
  double operator()(double *x, double *p) {
    double xscaled = (x[0] - xMinNorm_) / xRangeNorm_;
    return p[0] + p[1] * xscaled + p[2] * std::pow(xscaled, 2) +
           p[3] * std::pow(xscaled, 3);
  }
  void setPolynomialArgument(const double &xMin, const double &xRange) {
    xMinNorm_ = xMin;
    xRangeNorm_ = xRange;
  }
  int getNparams() { return nparams_; }

protected:
  // to normalize polynomial argument
  int nparams_ = 4;
  double xMinNorm_ = 0.0;
  double xRangeNorm_ = 1.0;
};

class pol4_custom {
public:
  pol4_custom() {};
  pol4_custom(const double &xMin, const double &xRange) {
    xMinNorm_ = xMin;
    xRangeNorm_ = xRange;
  };
  double operator()(double *x, double *p) {
    double xscaled = (x[0] - xMinNorm_) / xRangeNorm_;
    return p[0] + p[1] * xscaled + p[2] * std::pow(xscaled, 2) +
           p[3] * std::pow(xscaled, 3) + p[4] * std::pow(xscaled, 4);
  }
  void setPolynomialArgument(const double &xMin, const double &xRange) {
    xMinNorm_ = xMin;
    xRangeNorm_ = xRange;
  }
  int getNparams() { return nparams_; }

protected:
  // to normalize polynomial argument
  int nparams_ = 5;
  double xMinNorm_ = 0.0;
  double xRangeNorm_ = 1.0;
};

// // TODO: could use a generic polynomial using previous classes
// class erfPol2_custom {
// public:
//     erfPol2_custom() {};
//     erfPol2_custom(const double& xMin, const double& xRange) {
//         xMinNorm_ = xMin;
//         xRangeNorm_ = xRange;
//     };
//     double operator() (double *x, double *p) {
//         double xscaled = (x[0] - xMinNorm_) / xRangeNorm_;
//         return (p[0] + p[1]*xscaled + p[2]*std::pow(xscaled,2)) * (1.0 +
//         TMath::Erf((x[0]-p[3])/p[4]));
//     }
//     void setPolynomialArgument(const double& xMin, const double& xRange) {
//         xMinNorm_ = xMin;
//         xRangeNorm_ = xRange;
//     }
//     int getNparams() { return nparams_; }
// protected:
//     // to normalize polynomial argument
//     int nparams_ = 5;
//     double xMinNorm_ = 0.0;
//     double xRangeNorm_ = 1.0;
// };

// ================================================

class EtaPtCorrelatedEfficiency {

  // TODO: if a destructor is explicitly defined, add copy constructor and
  // assignment operator, even though I won't use any of those

public:
  EtaPtCorrelatedEfficiency(TH3D *histocov, TH2D *histoerf, double ptmin,
                            double ptmax);
  // ~EtaPtCorrelatedEfficiency();
  double DoEffSyst(int etabin, double pt, double *variations,
                   bool getDiff = false);
  std::vector<double> DoEffSyst(int etabin, int ipar);
  std::vector<double> DoEffSyst(int etabin);
  // void setPtRange(double ptmin, double ptmax) { ptmin_ = ptmin; ptmax_ =
  // ptmax; } // not used currently, should modify function ranges accordingly
  void setSmoothingFunction(const std::string &name);
  void setEigenShift(double shift) { eigenShift_ = shift; }

protected:
  Eigen::MatrixXd covariance(int etabin);
  void DoHessianShifts(int etabin, int ipar, const std::vector<double> &inpars,
                       std::vector<double> &outpars);

  std::string smoothFunctionName_ = "cheb3";
  TH3D *covhist_;
  TH2D *parhist_;
  int ndim_ = 4;
  double ptmin_ = 0.0;
  double ptmax_ = 100.0;
  double eigenShift_ = 1.0;
  TF1 *function_ = nullptr;
  // list of predefined functions
  TF1 *tf1_pol3_ = new TF1("tf1_pol3_", "pol3", ptmin_, ptmax_);
  TF1 *tf1_pol2_ = new TF1("tf1_pol2_", "pol2", ptmin_, ptmax_);
  TF1 *tf1_erf_ = new TF1("tf1_erf_", "[0] * (1.0 + TMath::Erf((x-[1])/[2]))",
                          ptmin_, ptmax_);
  pol3_custom pol3_tf_;
  TF1 *tf1_pol3_tf_ = nullptr;
  pol4_custom pol4_tf_;
  TF1 *tf1_pol4_tf_ = nullptr;
  // erfPol2_custom erfPol2_tf_;
  // TF1* tf1_erfPol2_tf_ = nullptr;
};

EtaPtCorrelatedEfficiency::EtaPtCorrelatedEfficiency(TH3D *histocov,
                                                     TH2D *histoerf,
                                                     double ptmin, double ptmax)
    : pol3_tf_(ptmin, ptmax), pol4_tf_(ptmin, ptmax)
// erfPol2_tf_(ptmin, ptmax)
{
  covhist_ = histocov;
  int ny = covhist_->GetNbinsY();
  int nz = covhist_->GetNbinsZ();
  assert(ny == nz);
  ndim_ = ny;
  parhist_ = histoerf;
  ptmin_ = ptmin;
  ptmax_ = ptmax;
  setSmoothingFunction("pol3_tf");
}
// EtaPtCorrelatedEfficiency::~EtaPtCorrelatedEfficiency() {
//     function_ = nullptr;
//     delete tf1_pol3_;
//     delete tf1_pol2_;
//     delete tf1_erf_;
//     delete tf1_pol3_tf_;
//     delete tf1_pol4_tf_;
// //     delete tf1_erfPol2_tf_;
// }

void EtaPtCorrelatedEfficiency::setSmoothingFunction(const std::string &name) {
  // TODO: if case I add more functions, find a smarte way to find the good one
  smoothFunctionName_ = name;
  if (name.find("pol3_tf") != std::string::npos) {
    tf1_pol3_tf_ = new TF1("tf1_pol3_tf_", pol3_tf_, ptmin_, ptmax_,
                           pol3_tf_.getNparams());
    function_ = tf1_pol3_tf_;
    ndim_ = tf1_pol3_tf_->GetNpar();
  } else if (name.find("pol4_tf") != std::string::npos) {
    tf1_pol4_tf_ = new TF1("tf1_pol4_tf_", pol4_tf_, ptmin_, ptmax_,
                           pol4_tf_.getNparams());
    function_ = tf1_pol4_tf_;
    ndim_ = tf1_pol4_tf_->GetNpar();
    // } else if (name.find("erfPol2_tf") != std::string::npos) {
    //     tf1_erfPol2_tf_ = new TF1("tf1_erfPol2_tf_", erfPol2_tf_, ptmin_,
    //     ptmax_, erfPol2_tf_.getNparams()); function_ = tf1_erfPol2_tf_; ndim_
    //     = tf1_erfPol2_tf_->GetNpar();
  } else if (name.find("pol3") != std::string::npos) {
    function_ = tf1_pol3_;
    ndim_ = 4;
  } else if (name.find("pol2") != std::string::npos) {
    function_ = tf1_pol2_;
    ndim_ = 3;
  } else if (name.find("erf") != std::string::npos) {
    function_ = tf1_erf_;
    ndim_ = 3;
  } else {
    std::cout << "Smoothing function " << name << " not implemented. Abort"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  return;
}

Eigen::MatrixXd EtaPtCorrelatedEfficiency::covariance(int etabin) {
  Eigen::MatrixXd covMat(ndim_, ndim_);
  for (int i = 0; i < ndim_; ++i) {
    for (int j = 0; j < ndim_; ++j) {
      covMat(i, j) = covhist_->GetBinContent(etabin, i + 1, j + 1);
    }
  }
  // std::cout << "covariance matrix = " << std::endl << covMat << std::endl;
  return covMat;
}

void EtaPtCorrelatedEfficiency::DoHessianShifts(
    int etabin, int ipar, const std::vector<double> &inpars,
    std::vector<double> &outpars) {

  // diagonalize the covariance matrix
  Eigen::MatrixXd covMat = covariance(etabin);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(covMat);
  Eigen::VectorXd eigenv = es.eigenvalues();
  Eigen::MatrixXd transformation = es.eigenvectors();
  // std::cout << "Transformation = " << std::endl << transformation <<
  // std::endl;

  // transform the pars in the diagonal basis
  const unsigned int npars = transformation.rows();
  const unsigned int neigenvectors = transformation.cols();
  Eigen::VectorXd inparv(npars);
  for (unsigned int jpar = 0; jpar < npars; ++jpar) {
    inparv[jpar] = inpars[jpar];
  }
  // std::cout << "inparv = " << std::endl << inparv << std::endl;
  Eigen::VectorXd diagbasisv = transformation.transpose() * inparv;
  // std::cout << "diagbasisv = " << std::endl << diagbasisv << std::endl;

  // shift one of them by the diagonal uncertainty (uncorrelated in this basis)
  diagbasisv[ipar] += eigenShift_ * sqrt(eigenv[ipar]);

  // transform the pars back in the original basis
  // Eigen::VectorXd outparv = transformation*diagbasisv;
  Eigen::VectorXd outparv =
      inparv + eigenShift_ * sqrt(eigenv[ipar]) * transformation.col(ipar);
  for (unsigned int ieig = 0; ieig < neigenvectors; ++ieig) {
    outpars[ieig] = outparv[ieig];
  }
  // std::cout << "outparv = " << std::endl << outparv << std::endl;
  return;
}

// method to return the actual function variations for all parameters filling
// the externally provided array "variations"
double EtaPtCorrelatedEfficiency::DoEffSyst(int etabin, double pt,
                                            double *variations, bool getDiff) {

  std::vector<double> inpars(ndim_);
  std::vector<double> outpars(ndim_);

  for (int ipar = 0; ipar < ndim_; ++ipar) {
    inpars[ipar] = parhist_->GetBinContent(etabin, ipar + 1);
  }

  double nominal = function_->EvalPar(&pt, inpars.data());

  for (int ipar = 0; ipar < ndim_; ++ipar) {
    DoHessianShifts(etabin, ipar, inpars, outpars);
    variations[ipar] = function_->EvalPar(&pt, outpars.data());
    if (getDiff)
      variations[ipar] -= nominal;
  }
  return nominal;
}

// method to return a single row of parameters
std::vector<double> EtaPtCorrelatedEfficiency::DoEffSyst(int etabin, int ipar) {

  if (ipar >= ndim_) {
    std::cout << "EtaPtCorrelatedEfficiency::DoEffSyst(int etabin, int ipar):  "
                 "ipar >= "
              << ndim_ << " (" << ipar << ")" << std::endl;
    exit(EXIT_FAILURE);
  }
  std::vector<double> inpars(ndim_);
  std::vector<double> outpars(ndim_);

  for (int jpar = 0; jpar < ndim_; ++jpar) {
    inpars[jpar] = parhist_->GetBinContent(etabin, jpar + 1);
  }
  DoHessianShifts(etabin, ipar, inpars, outpars);
  std::vector<double> ret(ndim_, 0.0);
  for (int jpar = 0; jpar < ndim_; ++jpar) {
    ret[jpar] = outpars[jpar];
  }
  return ret;
}

// method to return all parameters in a single vector
std::vector<double> EtaPtCorrelatedEfficiency::DoEffSyst(int etabin) {

  std::vector<double> inpars(ndim_);
  std::vector<double> outpars(ndim_);

  for (int jpar = 0; jpar < ndim_; ++jpar) {
    inpars[jpar] = parhist_->GetBinContent(etabin, jpar + 1);
  }

  std::vector<double> ret(ndim_ * ndim_, 0.0);
  for (int ipar = 0; ipar < ndim_; ++ipar) {
    DoHessianShifts(etabin, ipar, inpars, outpars);
    for (int jpar = 0; jpar < ndim_; ++jpar) {
      ret[ipar * ndim_ + jpar] = outpars[jpar];
    }
  }
  return ret;
}

} // namespace wrem

#endif
