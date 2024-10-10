#ifndef WREMNANTS_VERTEX_H
#define WREMNANTS_VERTEX_H

#include <TH2D.h>
#include <memory>

#include "utils.hpp"

namespace wrem {

class vertex_helper {
public:
  vertex_helper(const TH2D &weights)
      : vertexweights_(make_shared_TH1<const TH2D>(weights)) {}

  // returns the vertex weight
  double operator()(float genVtx_z, float nTrueInt) const {
    int xbin = std::clamp(vertexweights_->GetXaxis()->FindFixBin(genVtx_z), 1,
                          vertexweights_->GetNbinsX());
    int ybin = std::clamp(vertexweights_->GetYaxis()->FindFixBin(nTrueInt), 1,
                          vertexweights_->GetNbinsY());
    return vertexweights_->GetBinContent(xbin, ybin);
  }

private:
  std::shared_ptr<const TH2D> vertexweights_;
};

} // namespace wrem

#endif
