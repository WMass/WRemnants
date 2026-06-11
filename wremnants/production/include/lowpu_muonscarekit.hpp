#ifndef WREMNANTS_LOWPU_MUONSCAREKIT_H
#define WREMNANTS_LOWPU_MUONSCAREKIT_H

#include "defines.hpp"
#include <TMath.h>
#include <TFile.h>
#include <TH2D.h>
#include <TH3D.h>
#include <boost/math/special_functions/erf.hpp>

namespace wrem {

struct MuonScarekitCB {
    static const double pi;
    static const double sqrtPiOver2;
    static const double sqrt2;

    double m, s, a, n;
    double B, C, D, N, NA, Ns, NC, F, G, k;
    double cdfMa, cdfPa;

    MuonScarekitCB(double mean, double sigma, double alpha, double nn)
        : m(mean), s(sigma), a(alpha), n(nn) { init(); }

    void init() {
        double fa = fabs(a);
        double ex = exp(-fa * fa / 2);
        double A  = pow(n / fa, n) * ex;
        double C1 = n / fa / (n - 1) * ex;
        double D1 = 2 * sqrtPiOver2 * boost::math::erf(fa / sqrt2);
        B = n / fa - fa;
        C = (D1 + 2 * C1) / C1;
        D = (D1 + 2 * C1) / 2;
        N = 1.0 / s / (D1 + 2 * C1);
        k = 1.0 / (n - 1);
        NA = N * A; Ns = N * s; NC = Ns * C1;
        F = 1 - fa * fa / n;
        G = s * n / fa;
        cdfMa = cdf(m - a * s);
        cdfPa = cdf(m + a * s);
    }

    double cdf(double x) const {
        double d = (x - m) / s;
        if (d < -a) return NC / pow(F - s * d / G, n - 1);
        if (d >  a) return NC * (C - pow(F + s * d / G, 1 - n));
        return Ns * (D - sqrtPiOver2 * boost::math::erf(-d / sqrt2));
    }

    double invcdf(double u) const {
        if (u < cdfMa) return m + G * (F - pow(NC / u, k));
        if (u > cdfPa) return m - G * (F - pow(C - u / NC, -k));
        return m - sqrt2 * s * boost::math::erf_inv((D - u / Ns) / sqrtPiOver2);
    }
};

const double MuonScarekitCB::pi          = 3.14159265358979;
const double MuonScarekitCB::sqrtPiOver2 = sqrt(MuonScarekitCB::pi / 2.0);
const double MuonScarekitCB::sqrt2       = sqrt(2.0);

namespace muonscarekit_impl {
    TFile* tf_scale = TFile::Open(
        "wremnants-data/data/lowPU/muonscarekit/step3_correction.root", "READ");
    TH2D* h_M_DATA = (TH2D*)tf_scale->Get("M_DATA");
    TH2D* h_A_DATA = (TH2D*)tf_scale->Get("A_DATA");
    TH2D* h_M_SIG  = (TH2D*)tf_scale->Get("M_SIG");
    TH2D* h_A_SIG  = (TH2D*)tf_scale->Get("A_SIG");

    TFile* tf_cb   = TFile::Open(
        "wremnants-data/data/lowPU/muonscarekit/step2_fitresults.root", "READ");
    TH3D* h_cb     = (TH3D*)tf_cb->Get("h_results_cb");
    TH3D* h_poly   = (TH3D*)tf_cb->Get("h_results_poly");

    TFile* tf_k    = TFile::Open(
        "wremnants-data/data/lowPU/muonscarekit/step4_k.root", "READ");
    TH2D* h_k_data = (TH2D*)tf_k->Get("k_hist_DATA");
    TH2D* h_k_sig  = (TH2D*)tf_k->Get("k_hist_SIG");
}

Vec_f applyMuonScarekitData(Vec_f pt, Vec_f eta, Vec_f phi, Vec_i charge) {
    using namespace muonscarekit_impl;
    unsigned int size = pt.size();
    Vec_f res(size);
    for (unsigned int i = 0; i < size; ++i) {
        double M = h_M_DATA->GetBinContent(h_M_DATA->FindBin(eta[i], phi[i]));
        double A = h_A_DATA->GetBinContent(h_A_DATA->FindBin(eta[i], phi[i]));
        res[i] = static_cast<float>(1.0 / (M / pt[i] + charge[i] * A));
    }
    return res;
}

Vec_f applyMuonScarekitMC(Vec_f pt, Vec_f eta, Vec_f phi, Vec_i charge,
                           Vec_i nTrackerLayers,
                           unsigned int run, unsigned int lumi) {
    using namespace muonscarekit_impl;
    unsigned int size = pt.size();
    Vec_f res(size);

    for (unsigned int i = 0; i < size; ++i) {
        double M = h_M_SIG->GetBinContent(h_M_SIG->FindBin(eta[i], phi[i]));
        double A = h_A_SIG->GetBinContent(h_A_SIG->FindBin(eta[i], phi[i]));
        double pt_scale = 1.0 / (M / pt[i] + charge[i] * A);

        Int_t etabin = h_cb->GetXaxis()->FindBin(fabs((double)eta[i]));
        Int_t nlbin  = h_cb->GetYaxis()->FindBin((double)nTrackerLayers[i]);

        double mean_cb  = h_cb->GetBinContent(etabin, nlbin, 1);
        double sig_cb   = h_cb->GetBinContent(etabin, nlbin, 2);
        double n_cb     = h_cb->GetBinContent(etabin, nlbin, 3);
        double alpha_cb = h_cb->GetBinContent(etabin, nlbin, 4);

        double a_poly = h_poly->GetBinContent(etabin, nlbin, 1);
        double b_poly = h_poly->GetBinContent(etabin, nlbin, 2);
        double c_poly = h_poly->GetBinContent(etabin, nlbin, 3);
        double sigma_poly = a_poly + b_poly * pt_scale + c_poly * pt_scale * pt_scale;
        if (sigma_poly < 0.0) sigma_poly = 0.0;

        Int_t absetabin  = h_k_data->GetXaxis()->FindBin(fabs((double)eta[i]));
        double k_data_v  = h_k_data->GetBinContent(absetabin, 3);
        double k_sig_v   = h_k_sig->GetBinContent(absetabin, 3);
        double k_mc      = (k_sig_v < k_data_v)
                           ? sqrt(k_data_v * k_data_v - k_sig_v * k_sig_v)
                           : 0.0;

        if (k_mc == 0.0 || sigma_poly == 0.0 || n_cb <= 1.0 + 1e-6 || sig_cb <= 0.0 || alpha_cb <= 0.0) {
            res[i] = static_cast<float>(pt_scale);
            continue;
        }

        MuonScarekitCB cb(mean_cb, sig_cb, alpha_cb, n_cb);
        double rndm_cb = cb.invcdf(gRandom->Rndm());

        res[i] = static_cast<float>(pt_scale * (1.0 + k_mc * sigma_poly * rndm_cb));
    }
    return res;
}

} // namespace wrem
#endif
