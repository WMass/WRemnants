#ifndef WREMNANTS_DEFINES_H
#define WREMNANTS_DEFINES_H

#include <ROOT/RVec.hxx>
#include <ROOT/RDataFrame.hxx>
#include <ROOT/RDF/RInterface.hxx>

namespace wrem {

using Vec_b = ROOT::VecOps::RVec<bool>;
using Vec_d = ROOT::VecOps::RVec<double>;
using Vec_f = ROOT::VecOps::RVec<float>;
using Vec_i = ROOT::VecOps::RVec<int>;
using Vec_ui = ROOT::VecOps::RVec<unsigned int>;

bool isOddEvent(ULong64_t evt) {

  return (evt%2) ? 1 : 0;

}

bool isEvenEvent(ULong64_t evt) {

  return (evt%2) ? 0 : 1;

}

}

#endif
