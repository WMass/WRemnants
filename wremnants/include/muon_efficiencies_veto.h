#ifndef WREMNANTS_MUON_EFFICIENCIES_VETO_H
#define WREMNANTS_MUON_EFFICIENCIES_VETO_H

#include <boost/histogram/axis.hpp>
#include <array>
#include "defines.h"

namespace wrem {

	template<typename HIST_SF, int NEtaBins, int NPtEigenBins, int NSysts, int Steps>
	class muon_efficiency_veto_helper_base {

	public:
		muon_efficiency_veto_helper_base(HIST_SF &&sf_veto_plus, HIST_SF &&sf_veto_minus, double minpt, double maxpt, double mineta, double maxeta) :
			sf_veto_plus_(std::make_shared<const HIST_SF>(std::move(sf_veto_plus))),
			sf_veto_minus_(std::make_shared<const HIST_SF>(std::move(sf_veto_minus))),
			minpt_(minpt),
			maxpt_(maxpt),
			mineta_(mineta),
			maxeta_(maxeta) {
		}

		double scale_factor(float pt, float eta, int charge) const {

			double sf = 1.0;

			if ((pt > minpt_) && (pt < maxpt_) && (eta > mineta_) && (eta < maxeta_)) {
				auto const ix = sf_veto_plus_->template axis<0>().index(eta);
				auto const iy = sf_veto_plus_->template axis<1>().index(pt);
				if (charge>0) sf = sf_veto_plus_->at(ix,iy,0).value();
				else sf = sf_veto_minus_->at(ix,iy,0).value();
			}

			return sf;

		}

		using syst_tensor_t = Eigen::TensorFixedSize<double, Eigen::Sizes<Steps, NSysts>>;

		syst_tensor_t sf_syst_var(float pt, float eta, int charge) const {

			syst_tensor_t res;
			res.setConstant(1.0);

			if ((pt > minpt_) && (pt < maxpt_) && (eta > mineta_) && (eta < maxeta_)) {
				auto const ix = sf_veto_plus_->template axis<0>().index(eta);
				auto const iy = sf_veto_plus_->template axis<1>().index(pt);
				for (int step = 0; step < Steps; step++) {
					for (int ns = 0; ns < NSysts; ns++) {
						if ((ns == 0) || (ns == (ix+1))) {
							if (charge>0) res(step, ns) = sf_veto_plus_->at(ix,iy,1+2*NPtEigenBins+step).value()/sf_veto_plus_->at(ix,iy,0).value();
							else res(step, ns) = sf_veto_minus_->at(ix,iy,1+2*NPtEigenBins+step).value()/sf_veto_minus_->at(ix,iy,0).value();
						}
						else res(step, ns) = 1.;
					}
				}
			}
			return res;

		}

		using stat_tensor_t = Eigen::TensorFixedSize<double, Eigen::Sizes<NEtaBins, NPtEigenBins, 2>>;

		stat_tensor_t sf_stat_var(float pt, float eta, int charge) const {
			stat_tensor_t res;
			res.setConstant(1.0);

			if ((pt > minpt_) && (pt < maxpt_) && (eta > mineta_) && (eta < maxeta_)) {
				auto const ix = sf_veto_plus_->template axis<0>().index(eta);
				auto const iy = sf_veto_plus_->template axis<1>().index(pt);
				for (int tensor_eigen_idx = 1; tensor_eigen_idx <= NPtEigenBins; tensor_eigen_idx++) {
					if (charge > 0) res(ix, tensor_eigen_idx-1, 1) *= sf_veto_plus_->at(ix,iy,tensor_eigen_idx).value()/sf_veto_plus_->at(ix,iy,0).value();
					else res(ix, tensor_eigen_idx-1, 0) *= sf_veto_minus_->at(ix,iy,tensor_eigen_idx).value()/sf_veto_minus_->at(ix,iy,0).value();
				}
			}
				
			return res;
		}

	protected:

		std::shared_ptr<const HIST_SF> sf_veto_plus_;
		std::shared_ptr<const HIST_SF> sf_veto_minus_;
		double minpt_;
		double maxpt_;
		double mineta_;
		double maxeta_;

	};

	template<typename HIST_SF, int NEtaBins, int NPtEigenBins, int NSysts, int Steps>
	class muon_efficiency_veto_helper :
		public muon_efficiency_veto_helper_base<HIST_SF, NEtaBins, NPtEigenBins, NSysts, Steps> {

	public:

		using base_t = muon_efficiency_veto_helper_base<HIST_SF, NEtaBins, NPtEigenBins, NSysts, Steps>;

		using base_t::base_t;

		muon_efficiency_veto_helper(const base_t &other) : base_t(other) {}

		double operator() (float pt, float eta, int charge) {
			return base_t::scale_factor(pt, eta, charge);
		}

	};

	template<typename HIST_SF, int NEtaBins, int NPtEigenBins, int NSysts, int Steps>
	class muon_efficiency_veto_helper_syst :
		public muon_efficiency_veto_helper_base<HIST_SF, NEtaBins, NPtEigenBins, NSysts, Steps> {

	public:

		using base_t = muon_efficiency_veto_helper_base<HIST_SF, NEtaBins, NPtEigenBins, NSysts, Steps>;
		using tensor_t = typename base_t::syst_tensor_t;

		using base_t::base_t;

		muon_efficiency_veto_helper_syst(const base_t &other) : base_t(other) {}
		
		tensor_t operator() (float pt, float eta, int charge, double nominal_weight = 1.0) {
			return nominal_weight*base_t::sf_syst_var(pt, eta, charge);
		}

	};

	template<typename HIST_SF, int NEtaBins, int NPtEigenBins, int NSysts, int Steps>
	class muon_efficiency_veto_helper_stat :
		public muon_efficiency_veto_helper_base<HIST_SF, NEtaBins, NPtEigenBins, NSysts, Steps> {
		
	public:

		using base_t = muon_efficiency_veto_helper_base<HIST_SF, NEtaBins, NPtEigenBins, NSysts, Steps>;
		using tensor_t = typename base_t::stat_tensor_t;

		using base_t::base_t;

		muon_efficiency_veto_helper_stat(const base_t &other) : base_t(other) {}

		tensor_t operator() (float pt, float eta, int charge, double nominal_weight = 1.0) {
			return nominal_weight*base_t::sf_stat_var(pt, eta, charge);
		}

	};

}

#endif
