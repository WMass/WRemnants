#ifndef WREMNANTS_MCSTAT_FOLDS_H
#define WREMNANTS_MCSTAT_FOLDS_H

#include <cstdint>

namespace wrem {

// splitmix64 finalizer: a bijective 64-bit mixer, so that consecutive inputs
// give uncorrelated outputs
inline std::uint64_t splitmix64(std::uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

// MC-stat fold index in [0, nfolds) of a generated event, for the two-half /
// k-fold de-biasing of the fit. The index is a hash of the event identifiers
// only, so it is the same for every histogram filled from the event and does
// not depend on its kinematics. The raw event number is not used directly
// because its parity is used elsewhere in the analysis (e.g. to choose the
// W-like lepton).
inline int mc_stat_fold(unsigned int run, unsigned int lumi,
                        unsigned long long event, unsigned int nfolds,
                        unsigned long long seed) {
  std::uint64_t h = splitmix64(seed);
  h = splitmix64(h ^ run);
  h = splitmix64(h ^ lumi);
  h = splitmix64(h ^ event);
  return static_cast<int>(h % nfolds);
}

} // namespace wrem

#endif
