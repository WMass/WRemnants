#ifndef WREMNANTS_DEFINES_H
#define WREMNANTS_DEFINES_H

#include <ROOT/RDF/RInterface.hxx>
#include <ROOT/RDataFrame.hxx>
#include <ROOT/RVec.hxx>

namespace wrem {

using Vec_b = ROOT::VecOps::RVec<bool>;
using Vec_d = ROOT::VecOps::RVec<double>;
using Vec_f = ROOT::VecOps::RVec<float>;
using Vec_i = ROOT::VecOps::RVec<int>;
using Vec_ui = ROOT::VecOps::RVec<unsigned int>;

enum class AnalysisType { Wmass = 0, Wlike, Dilepton };

enum class Era { Era_2016PreVFP, Era_2016PostVFP, Era_2017, Era_2018 };

const unsigned int MUON_PDGID = 13;

bool isOddEvent(ULong64_t evt) { return (evt % 2) ? 1 : 0; }

bool isEvenEvent(ULong64_t evt) { return (evt % 2) ? 0 : 1; }

} // namespace wrem

#endif
