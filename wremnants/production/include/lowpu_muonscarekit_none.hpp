#ifndef WREMNANTS_LOWPU_MUONSCAREKIT_H
#define WREMNANTS_LOWPU_MUONSCAREKIT_H

#include "defines.hpp"

namespace wrem {

// No corrections — return raw pt unchanged
Vec_f applyMuonScarekitData(Vec_f pt, Vec_f eta, Vec_f phi, Vec_i charge) {
    return pt;
}

Vec_f applyMuonScarekitMC(Vec_f pt, Vec_f eta, Vec_f phi, Vec_i charge,
                           Vec_i nTrackerLayers,
                           unsigned int run, unsigned int lumi) {
    return pt;
}

} // namespace wrem
#endif
