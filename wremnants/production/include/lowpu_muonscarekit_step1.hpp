#ifndef WREMNANTS_LOWPU_MUONSCAREKIT_H
#define WREMNANTS_LOWPU_MUONSCAREKIT_H

#include "defines.hpp"
#include <TFile.h>
#include <TH2D.h>

namespace wrem {

namespace muonscarekit_step1_impl {
    TFile* tf_scale = TFile::Open(
        "wremnants-data/data/lowPU/muonscarekit/step1_C.root", "READ");
    TH2D* h_M_DATA = (TH2D*)tf_scale->Get("M_DATA");
    TH2D* h_A_DATA = (TH2D*)tf_scale->Get("A_DATA");
    TH2D* h_M_SIG  = (TH2D*)tf_scale->Get("M_SIG");
    TH2D* h_A_SIG  = (TH2D*)tf_scale->Get("A_SIG");
}

// Step 1 scale correction only — no resolution smearing
Vec_f applyMuonScarekitData(Vec_f pt, Vec_f eta, Vec_f phi, Vec_i charge) {
    using namespace muonscarekit_step1_impl;
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
    using namespace muonscarekit_step1_impl;
    unsigned int size = pt.size();
    Vec_f res(size);
    for (unsigned int i = 0; i < size; ++i) {
        double M = h_M_SIG->GetBinContent(h_M_SIG->FindBin(eta[i], phi[i]));
        double A = h_A_SIG->GetBinContent(h_A_SIG->FindBin(eta[i], phi[i]));
        res[i] = static_cast<float>(1.0 / (M / pt[i] + charge[i] * A));
    }
    return res;
}

} // namespace wrem
#endif
