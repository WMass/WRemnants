#ifndef WREMNANTS_ELECTRON_ID_H
#define WREMNANTS_ELECTRON_ID_H

#include <limits>
#include <bitset>

#include "ROOT/RVec.hxx"

namespace wrem {
        
    namespace electron_id {
        namespace wp {
            constexpr int nbit = 3;

            // following the Electron_cutBased branch syntax
            // https://cms-nanoaod-integration.web.cern.ch/integration/cms-swCMSSW_10_6_X/mc106Xul17v2_doc.html#Electron
            // cut-based ID Fall17 V2 (0:fail, 1:veto, 2:loose, 3:medium, 4:tight)
            // i.e. the comparison must be done as integer using >=, and NOT treated as a bitset!!
            constexpr int tight = 4;
            constexpr int medium = 3;
            constexpr int loose = 2;
            constexpr int veto = 1;
            constexpr int fail = 0;
        }

        namespace cut {  
            template <int ...cuts>
            constexpr int bitset(int wp, std::integer_sequence<int, cuts...>)
            {
                return ((wp << cuts) + ...);
            }

            constexpr int ncut = 10;

            constexpr int icut_min_pt = 0;
            constexpr int icut_sc_eta = 3;
            constexpr int icut_deta_seed = 6;
            constexpr int icut_dphi_in = 9;
            constexpr int icut_sieie_5x5 = 12;
            constexpr int icut_hoe = 15;
            constexpr int icut_inve_over_invp = 18;
            constexpr int icut_relpfiso = 21;
            constexpr int icut_conv_veto = 24;
            constexpr int icut_miss_hit = 27;

            constexpr auto iseq_min_pt = std::integer_sequence<int, icut_min_pt>{};
            constexpr auto iseq_sc_eta = std::integer_sequence<int, icut_sc_eta>{};
            constexpr auto iseq_deta_seed = std::integer_sequence<int, icut_deta_seed>{};
            constexpr auto iseq_dphi_in = std::integer_sequence<int, icut_dphi_in>{};
            constexpr auto iseq_sieie_5x5 = std::integer_sequence<int, icut_sieie_5x5>{};
            constexpr auto iseq_hoe = std::integer_sequence<int, icut_hoe>{};
            constexpr auto iseq_inve_over_invp = std::integer_sequence<int, icut_inve_over_invp>{};
            constexpr auto iseq_relpfiso = std::integer_sequence<int, icut_relpfiso>{};
            constexpr auto iseq_conv_veto = std::integer_sequence<int, icut_conv_veto>{};
            constexpr auto iseq_miss_hit = std::integer_sequence<int, icut_miss_hit>{};
            constexpr auto iseq_all_but_pfiso = std::integer_sequence<int,
                                                                    icut_min_pt,
                                                                    icut_sc_eta,
                                                                    icut_deta_seed,
                                                                    icut_dphi_in,
                                                                    icut_sieie_5x5,
                                                                    icut_hoe,
                                                                    icut_inve_over_invp, 
                                                                    icut_conv_veto,
                                                                    icut_miss_hit>{};
            constexpr auto iseq_all = std::integer_sequence<int,
                                                            icut_min_pt,
                                                            icut_sc_eta,
                                                            icut_deta_seed,
                                                            icut_dphi_in,
                                                            icut_sieie_5x5,
                                                            icut_hoe,
                                                            icut_inve_over_invp,
                                                            icut_relpfiso,
                                                            icut_conv_veto,
                                                            icut_miss_hit>{};

            template <int WP> constexpr int bitset_min_pt = bitset(WP, iseq_min_pt);
            template <int WP> constexpr int bitset_sc_eta = bitset(WP, iseq_sc_eta);
            template <int WP> constexpr int bitset_deta_seed = bitset(WP, iseq_deta_seed);
            template <int WP> constexpr int bitset_dphi_in = bitset(WP, iseq_dphi_in);
            template <int WP> constexpr int bitset_sieie_5x5 = bitset(WP, iseq_sieie_5x5);
            template <int WP> constexpr int bitset_hoe = bitset(WP, iseq_hoe);
            template <int WP> constexpr int bitset_inve_over_invp = bitset(WP, iseq_inve_over_invp);
            template <int WP> constexpr int bitset_relpfiso = bitset(WP, iseq_relpfiso);
            template <int WP> constexpr int bitset_conv_veto = bitset(WP, iseq_conv_veto);
            template <int WP> constexpr int bitset_miss_hit = bitset(WP, iseq_miss_hit);
            template <int WP> constexpr int bitset_all_but_pfiso = bitset(WP, iseq_all_but_pfiso);
            template <int WP> constexpr int bitset_all = bitset(WP, iseq_all);
        }

        template <std::size_t N>
        constexpr std::size_t last_n(std::size_t bits)
        {
        static_assert(N < sizeof(std::size_t), "ERROR: this function makes no sense for too large N!");
        return bits & std::bitset<N>(std::numeric_limits<std::size_t>::max()).to_ullong();
        }

        template <int bits>
        ROOT::VecOps::RVec<int> pass_id(const ROOT::VecOps::RVec<int> &bitmap)
        {
            using cut::ncut;
            using wp::nbit;

            ROOT::VecOps::RVec<int> passID(bitmap.size(), 0);
            for (std::size_t ibit = 0ull; ibit < bitmap.size(); ++ibit) {
            bool pass_cuts = true;

            for (int icut = 0; icut < ncut; ++icut) {
                const int nshift = icut * nbit;
                pass_cuts = pass_cuts and (last_n<nbit>(bitmap[ibit] >> nshift) >= last_n<nbit>(bits >> nshift));
            }

            passID[ibit] = pass_cuts;
            }

            return passID;
        }

        template <int WP>
        ROOT::VecOps::RVec<int> (*pass_cutbased_noiso)(const ROOT::VecOps::RVec<int>&) = &pass_id<cut::bitset_all_but_pfiso<WP>>;

        template <int WP>
        ROOT::VecOps::RVec<int> (*pass_cutbased)(const ROOT::VecOps::RVec<int>&) = &pass_id<cut::bitset_all<WP>>;

        template <int WP>
        ROOT::VecOps::RVec<int> (*pass_iso)(const ROOT::VecOps::RVec<int>&) = &pass_id<cut::bitset_relpfiso<WP>>;

    }
}


#endif