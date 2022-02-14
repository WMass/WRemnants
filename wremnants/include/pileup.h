#ifndef WREMNANTS_PILEUP_H
#define WREMNANTS_PILEUP_H

#include "TH1D.h"
#include <memory>

namespace wrem {

class pileup_helper {
public:

  pileup_helper(const TH1D &puweights) :
    puweights_(make_shared_TH1<const TH1D>(puweights)) {}

  // returns the pileup weight
  double operator() (float nTrueInt) const {
    return puweights_->GetBinContent(puweights_->FindFixBin(nTrueInt));
  }


private:
  std::shared_ptr<const TH1D> puweights_;
};

}

#endif
