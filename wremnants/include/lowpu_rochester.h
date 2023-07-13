#ifndef WREMNANTS_LOWPU_ROCHESTER_H
#define WREMNANTS_LOWPU_ROCHESTER_H

#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include "defines.h"

//#include <boost/math/special_functions/erf.hpp>
#include "TMath.h"

namespace wrem {
    


struct CrystalBall{
    static const double pi;
    static const double sqrtPiOver2;
    static const double sqrt2;

    double m;
    double s;
    double a;
    double n;

    double B;
    double C;
    double D;
    double N;

    double NA;
    double Ns;
    double NC;
    double F;
    double G;
    double k;

    double cdfMa;
    double cdfPa;

    CrystalBall():m(0),s(1),a(10),n(10){
	init();
    }

    void init(){
	double fa = fabs(a);
	double ex = exp(-fa*fa/2);
	double A  = pow(n/fa, n) * ex;
	double C1 = n/fa/(n-1) * ex; 
	//double D1 = 2 * sqrtPiOver2 * erf(fa/sqrt2);
	double D1 = 2 * sqrtPiOver2 * TMath::Erf(fa/sqrt2);

	B = n/fa-fa;
	C = (D1+2*C1)/C1;   
	D = (D1+2*C1)/2;   

	N = 1.0/s/(D1+2*C1); 
	k = 1.0/(n-1);  

	NA = N*A;       
	Ns = N*s;       
	NC = Ns*C1;     
	F = 1-fa*fa/n; 
	G = s*n/fa;    

	cdfMa = cdf(m-a*s);
	cdfPa = cdf(m+a*s);
    }

    double pdf(double x) const{ 
	double d=(x-m)/s;
	if(d<-a) return NA*pow(B-d, -n);
	if(d>a) return NA*pow(B+d, -n);
	return N*exp(-d*d/2);
    }

    double pdf(double x, double ks, double dm) const{ 
	double d=(x-m-dm)/(s*ks);
	if(d<-a) return NA/ks*pow(B-d, -n);
	if(d>a) return NA/ks*pow(B+d, -n);
	return N/ks*exp(-d*d/2);
    }

    double cdf(double x) const{
	double d = (x-m)/s;
	if(d<-a) return NC / pow(F-s*d/G, n-1);
	if(d>a) return NC * (C - pow(F+s*d/G, 1-n) );
	//return Ns * (D - sqrtPiOver2 * erf(-d/sqrt2));
	return Ns * (D - sqrtPiOver2 * TMath::Erf(-d/sqrt2));
    }

    double invcdf(double u) const{
	if(u<cdfMa) return m + G*(F - pow(NC/u, k));
	if(u>cdfPa) return m - G*(F - pow(C-u/NC, -k) );
	//return m - sqrt2 * s * boost::math::erf_inv((D - u/Ns )/sqrtPiOver2);
	return m - sqrt2 * s * TMath::ErfInverse((D - u/Ns )/sqrtPiOver2);
    }
};


struct RocRes{
    enum TYPE {MC, Data, Extra};

    struct ResParams{
	double eta; 
	double kRes[2]; 
	std::vector<double> nTrk[2]; 
	std::vector<double> rsPar[3]; 
	std::vector<CrystalBall> cb;
	ResParams():eta(0){for(auto& k: kRes) k=1;}
    };

    int NETA;
    int NTRK;
    int NMIN;

    std::vector<ResParams> resol;

    RocRes();

    int etaBin(double x) const;
    int trkBin(double x, int h, TYPE T=MC) const;
    void reset();

    double rndm(int H, int F, double v) const;
    double Sigma(double pt, int H, int F) const;
    double kSpread(double gpt, double rpt, double eta, int nlayers, double w) const;
    double kSpread(double gpt, double rpt, double eta) const;
    double kSmear(double pt, double eta, TYPE type, double v, double u) const;
    double kSmear(double pt, double eta, TYPE type, double v, double u, int n) const;
    double kExtra(double pt, double eta, int nlayers, double u, double w) const;
    double kExtra(double pt, double eta, int nlayers, double u) const;
};

class RoccoR{

    public: // was private?
	enum TVAR{Default, Replica, Symhes};

	static const double MPHI; 

	int NETA;
	int NPHI; 
	double DPHI;
	std::vector<double> etabin;

	struct CorParams{double M; double A;};

	struct RocOne{
	    RocRes RR;
	    std::vector<std::vector<CorParams>> CP[2];
	};

	int nset;
	std::vector<int> nmem;
	std::vector<int> tvar;
	std::vector<std::vector<RocOne>> RC;
	int etaBin(double eta) const;
	int phiBin(double phi) const;
	template <typename T> double error(T f) const;

    //public:
	enum TYPE{MC, DT};

	RoccoR(); 
	RoccoR(std::string filename); 

	void init(std::string filename);
	void reset();
	bool empty() const {return RC.empty();} 
	const RocRes& getRes(int s=0, int m=0) const {return RC[s][m].RR;}
	double getM(int T, int H, int F, int s=0, int m=0) const{return RC[s][m].CP[T][H][F].M;}
	double getA(int T, int H, int F, int s=0, int m=0) const{return RC[s][m].CP[T][H][F].A;}
	double getK(int T, int H, int s=0, int m=0)        const{return RC[s][m].RR.resol[H].kRes[T];}
	double kGenSmear(double pt, double eta, double v, double u, RocRes::TYPE TT=RocRes::Data, int s=0, int m=0) const;
	double kScaleMC(int Q, double pt, double eta, double phi, int s=0, int m=0) const;

	double kScaleDT(int Q, double pt, double eta, double phi, int s=0, int m=0) const;
	double kSpreadMC(int Q, double pt, double eta, double phi, double gt, int s=0, int m=0) const;
	double kSmearMC(int Q, double pt, double eta, double phi, int n, double u, int s=0, int m=0) const;

	double kScaleDTerror(int Q, double pt, double eta, double phi) const;
	double kSpreadMCerror(int Q, double pt, double eta, double phi, double gt) const;
	double kSmearMCerror(int Q, double pt, double eta, double phi, int n, double u) const;

	//old, should only be used with 2017v0
	double kScaleFromGenMC(int Q, double pt, double eta, double phi, int n, double gt, double w, int s=0, int m=0) const; 
	double kScaleAndSmearMC(int Q, double pt, double eta, double phi, int n, double u, double w, int s=0, int m=0) const;  
	double kScaleFromGenMCerror(int Q, double pt, double eta, double phi, int n, double gt, double w) const; 
	double kScaleAndSmearMCerror(int Q, double pt, double eta, double phi, int n, double u, double w) const;  
};


const double CrystalBall::pi = 3.14159;
const double CrystalBall::sqrtPiOver2 = sqrt(CrystalBall::pi/2.0);
const double CrystalBall::sqrt2 = sqrt(2.0);



RocRes::RocRes(){
    reset();
}

void RocRes::reset(){
    NETA=0;
    NTRK=0;
    NMIN=0;
    std::vector<ResParams>().swap(resol);
}

int RocRes::etaBin(double eta) const{
    double abseta=fabs(eta);
    for(int i=0; i<NETA-1; ++i) if(abseta<resol[i+1].eta) return i;
    return NETA-1;
}

int RocRes::trkBin(double x, int h, TYPE T) const{
    for(int i=0; i<NTRK-1; ++i) if(x<resol[h].nTrk[T][i+1]) return i;
    return NTRK-1;
}

double RocRes::Sigma(double pt, int H, int F) const{
    double dpt=pt-45;
    const ResParams &rp = resol[H];
    return rp.rsPar[0][F] + rp.rsPar[1][F]*dpt + rp.rsPar[2][F]*dpt*dpt;
}

double RocRes::rndm(int H, int F, double w) const{
    const ResParams &rp = resol[H];
    return rp.nTrk[MC][F]+(rp.nTrk[MC][F+1]-rp.nTrk[MC][F])*w; 
}

double RocRes::kSpread(double gpt, double rpt, double eta, int n, double w) const{
    int H = etaBin(fabs(eta));
    int F = n>NMIN ? n-NMIN : 0;
    double v = rndm(H, F, w);
    int D = trkBin(v, H, Data);
    double kold = gpt / rpt;
    const ResParams &rp = resol[H];
    double u = rp.cb[F].cdf( (kold-1.0)/rp.kRes[MC]/Sigma(gpt,H,F) ); 
    double knew = 1.0 + rp.kRes[Data]*Sigma(gpt,H,D)*rp.cb[D].invcdf(u);

    if(knew<0) return 1.0;
    return kold/knew;
}


double RocRes::kSpread(double gpt, double rpt, double eta) const{
    int H = etaBin(fabs(eta));
    const auto &k = resol[H].kRes;
    double x = gpt/rpt;
    return x / (1.0 + (x-1.0)*k[Data]/k[MC]);
}

double RocRes::kSmear(double pt, double eta, TYPE type, double v, double u) const{
    int H = etaBin(fabs(eta));
    int F = trkBin(v, H); 
    const ResParams &rp = resol[H];
    double x = rp.kRes[type] * Sigma(pt, H, F) * rp.cb[F].invcdf(u);
    return 1.0/(1.0+x);
}

double RocRes::kSmear(double pt, double eta, TYPE type, double w, double u, int n) const{
    int H = etaBin(fabs(eta));
    int F = n-NMIN;
    if(type==Data) F = trkBin(rndm(H, F, w), H, Data);
    const ResParams &rp = resol[H];
    double x = rp.kRes[type] * Sigma(pt, H, F) * rp.cb[F].invcdf(u);
    return 1.0/(1.0+x);
}

double RocRes::kExtra(double pt, double eta, int n, double u, double w) const{
    int H = etaBin(fabs(eta));
    int F = n>NMIN ? n-NMIN : 0;
    const ResParams &rp = resol[H];
    double v = rp.nTrk[MC][F]+(rp.nTrk[MC][F+1]-rp.nTrk[MC][F])*w;
    int D = trkBin(v, H, Data);
    double RD = rp.kRes[Data]*Sigma(pt, H, D);
    double RM = rp.kRes[MC]*Sigma(pt, H, F);
    double x = RD>RM ? sqrt(RD*RD-RM*RM)*rp.cb[F].invcdf(u) : 0;
    if(x<=-1) return 1.0;
    return 1.0/(1.0 + x); 
}

double RocRes::kExtra(double pt, double eta, int n, double u) const{
    int H = etaBin(fabs(eta));
    int F = n>NMIN ? n-NMIN : 0;
    const ResParams &rp = resol[H];
    double d = rp.kRes[Data];
    double m = rp.kRes[MC];
    double x = d>m ? sqrt(d*d-m*m) * Sigma(pt, H, F) * rp.cb[F].invcdf(u) : 0;
    if(x<=-1) return 1.0;
    return 1.0/(1.0 + x); 
}


RoccoR::RoccoR(){}

RoccoR::RoccoR(std::string filename){
    init(filename);
}

void RoccoR::reset(){
    NETA=0;
    NPHI=0;
    std::vector<double>().swap(etabin);
    nset=0;
    std::vector<int>().swap(nmem);
    std::vector<std::vector<RocOne>>().swap(RC);

}


void RoccoR::init(std::string filename){
    std::ifstream in(filename.c_str());
    if(in.fail()) throw std::invalid_argument("RoccoR::init could not open file " + filename);

    int RMIN(0), RTRK(0), RETA(0);
    std::vector<double> BETA;

    std::string tag;
    int type, sys, mem, var, bin;	
    std::string s;
    while(std::getline(in, s)){
	std::stringstream ss(s); 
	std::string first4=s.substr(0,4);
	if(first4=="NSET"){
	    ss >> tag >> nset;
	    nmem.resize(nset);
	    tvar.resize(nset);
	    RC.resize(nset);
	}
	else if(first4=="NMEM") {
	    ss >> tag;
	    for(int i=0; i<nset; ++i) {
		ss >> nmem[i];
		RC[i].resize(nmem[i]);
	    }
	}
	else if(first4=="TVAR") {
	    ss >> tag;
	    for(int i=0; i<nset; ++i) ss >> tvar[i];
	}
	else if(first4=="RMIN") ss >> tag >> RMIN;
	else if(first4=="RTRK") ss >> tag >> RTRK;
	else if(first4=="RETA") {
	    ss >> tag >> RETA;
	    BETA.resize(RETA+1);
	    for(auto &h: BETA) ss >> h;

	}
	else if(first4=="CPHI") {
	    ss >> tag >> NPHI; 
	    DPHI=2*CrystalBall::pi/NPHI;
	}
	else if(first4=="CETA")  {
	    ss >> tag >> NETA;
	    etabin.resize(NETA+1);
	    for(auto& h: etabin) ss >> h;
	}
	else{ 
	    ss >> sys >> mem >> tag;
	    auto &rc = RC[sys][mem]; 
	    rc.RR.NETA=RETA;
	    rc.RR.NTRK=RTRK;
	    rc.RR.NMIN=RMIN;
	    auto &resol = rc.RR.resol;
	    if(resol.empty()){
		resol.resize(RETA);
		for(size_t ir=0; ir<resol.size(); ++ir){
		    auto &r = resol[ir];
		    r.eta = BETA[ir];
		    r.cb.resize(RTRK);
		    for(auto i:{0,1})r.nTrk[i].resize(RTRK+1);
		    for(auto i:{0,1,2})r.rsPar[i].resize(RTRK);
		}
	    }
	    auto &cp = rc.CP;
	    for(TYPE T:{MC,DT}){
		if(cp[T].empty()){
		    cp[T].resize(NETA);
		    for(auto &i: cp[T]) i.resize(NPHI);
		}
	    }

	    if(tag=="R"){
		ss >> var >> bin; 
		for(int i=0; i<RTRK; ++i) {
		    switch(var){
			case 0: ss >> resol[bin].rsPar[var][i]; break;
			case 1: ss >> resol[bin].rsPar[var][i]; break;
			case 2: ss >> resol[bin].rsPar[var][i]; resol[bin].rsPar[var][i]/=100; break; 
			case 3: ss >> resol[bin].cb[i].s; break; 
			case 4: ss >> resol[bin].cb[i].a; break; 
			case 5: ss >> resol[bin].cb[i].n; break; 
			default: break;
		    }
		}
	    }
	    else if(tag=="T") {
		ss >> type >> bin; 
		for(int i=0; i<RTRK+1; ++i) ss >> resol[bin].nTrk[type][i];
	    }
	    else if(tag=="F") {
		ss >> type; 
		for(int i=0; i<RETA; ++i) ss >> resol[i].kRes[type];

	    }
	    else if(tag=="C") {
		ss >> type >> var >> bin; 
		for(int i=0; i<NPHI; ++i){
		    auto &x = cp[type][bin][i];
		    if(var==0) { ss >> x.M; x.M = 1.0+x.M/100;}
		    else if(var==1){ ss >> x.A; x.A/=100; }
		}
	    }
	}
    }

    for(auto &rcs: RC)
	for(auto &rcm: rcs)
	    for(auto &r: rcm.RR.resol)
		for(auto &i: r.cb) i.init();

    in.close();
}

const double RoccoR::MPHI=-CrystalBall::pi;

int RoccoR::etaBin(double x) const{
    for(int i=0; i<NETA-1; ++i) if(x<etabin[i+1]) return i;
    return NETA-1;
}

int RoccoR::phiBin(double x) const{
    int ibin=(x-MPHI)/DPHI;
    if(ibin<0) return 0; 
    if(ibin>=NPHI) return NPHI-1;
    return ibin;
}

double RoccoR::kScaleDT(int Q, double pt, double eta, double phi, int s, int m) const{
    int H = etaBin(eta);
    int F = phiBin(phi);
    return 1.0/(RC[s][m].CP[DT][H][F].M + Q*RC[s][m].CP[DT][H][F].A*pt);
}

double RoccoR::kScaleMC(int Q, double pt, double eta, double phi, int s, int m) const{
    int H = etaBin(eta);
    int F = phiBin(phi);
    return 1.0/(RC[s][m].CP[MC][H][F].M + Q*RC[s][m].CP[MC][H][F].A*pt);
}

double RoccoR::kSpreadMC(int Q, double pt, double eta, double phi, double gt, int s, int m) const{
    const auto& rc=RC[s][m];
    int H = etaBin(eta);
    int F = phiBin(phi);
    double k=1.0/(rc.CP[MC][H][F].M + Q*rc.CP[MC][H][F].A*pt);
    return k*rc.RR.kSpread(gt, k*pt, eta);
}

double RoccoR::kSmearMC(int Q, double pt, double eta, double phi, int n, double u, int s, int m) const{
    const auto& rc=RC[s][m];
    int H = etaBin(eta);
    int F = phiBin(phi);
    double k=1.0/(rc.CP[MC][H][F].M + Q*rc.CP[MC][H][F].A*pt);
    return k*rc.RR.kExtra(k*pt, eta, n, u);
}


double RoccoR::kScaleFromGenMC(int Q, double pt, double eta, double phi, int n, double gt, double w, int s, int m) const{
    const auto& rc=RC[s][m];
    int H = etaBin(eta);
    int F = phiBin(phi);
    double k=1.0/(rc.CP[MC][H][F].M + Q*rc.CP[MC][H][F].A*pt);
    return k*rc.RR.kSpread(gt, k*pt, eta, n, w);
}

double RoccoR::kScaleAndSmearMC(int Q, double pt, double eta, double phi, int n, double u, double w, int s, int m) const{
    const auto& rc=RC[s][m];
    int H = etaBin(eta);
    int F = phiBin(phi);
    double k=1.0/(rc.CP[MC][H][F].M + Q*rc.CP[MC][H][F].A*pt);
    return k*rc.RR.kExtra(k*pt, eta, n, u, w);
}

double RoccoR::kGenSmear(double pt, double eta, double v, double u, RocRes::TYPE TT, int s, int m) const{
    if(empty()) return 1.0;
    return RC[s][m].RR.kSmear(pt, eta, TT, v, u);
}

template <typename T>
double RoccoR::error(T f) const{
    double sum=0;
    for(int s=0; s<nset; ++s){
	for(int i=0; i<nmem[s]; ++i) {
	    double d = f(s,i) - f(0,0); 
	    sum += d*d/nmem[s];
	}
    }
    return sqrt(sum);
}

double RoccoR::kScaleDTerror(int Q, double pt, double eta, double phi) const{
    return error([this, Q, pt, eta, phi](int s, int m) {return kScaleDT(Q, pt, eta, phi, s, m);});
}

double RoccoR::kSpreadMCerror(int Q, double pt, double eta, double phi, double gt) const{
    return error([this, Q, pt, eta, phi, gt](int s, int m){return kSpreadMC(Q, pt, eta, phi, gt, s, m);});
}

double RoccoR::kSmearMCerror(int Q, double pt, double eta, double phi, int n, double u) const{
    return error([this, Q, pt, eta, phi, n, u](int s, int m){return kSmearMC(Q, pt, eta, phi, n, u, s, m);});
}

double RoccoR::kScaleFromGenMCerror(int Q, double pt, double eta, double phi, int n, double gt, double w) const{
    return error([this, Q, pt, eta, phi, n, gt, w](int s, int m) {return kScaleFromGenMC(Q, pt, eta, phi, n, gt, w, s, m);});
}

double RoccoR::kScaleAndSmearMCerror(int Q, double pt, double eta, double phi, int n, double u, double w) const{
    return error([this, Q, pt, eta, phi, n, u, w](int s, int m) {return kScaleAndSmearMC(Q, pt, eta, phi, n, u, w, s, m);});
}




//RoccoR * rochester = new RoccoR("wremnants/data/lowPU/rochester/RoccoR2017.txt"); // https://gitlab.cern.ch/akhukhun/roccor
RoccoR * rochester = new RoccoR("wremnants/data/lowPU/rochester/RoccoR_lowPU_v0.txt"); // v0 version for lowPU

Vec_f applyRochesterMC(Vec_f pt, Vec_f eta, Vec_f phi, Vec_f ch, Vec_i gen_idx, Vec_f gen_pt, Vec_i nTrackerLayers, int fluctuation=0) {

    unsigned int size = pt.size();
    Vec_f res(size, 0.0);

    // https://gitlab.cern.ch/akhukhun/roccor
    for(unsigned int i = 0; i < size; ++i) {

        if(gen_idx.at(i) >= 0) res[i] = 1.0000*pt[i] * rochester->kSpreadMC(ch[i], pt[i], eta[i], phi[i], gen_pt.at(gen_idx.at(i)), fluctuation, 0);
        else res[i] = 1.0000*pt[i] * rochester->kSmearMC(ch[i], pt[i], eta[i], phi[i], nTrackerLayers.at(i), gRandom->Rndm(), fluctuation, 0);
    }

    return res;

}


Vec_f applyRochesterData(Vec_f pt, Vec_f eta, Vec_f phi, Vec_f ch, int fluctuation=0) {

    unsigned int size = pt.size();
    Vec_f res(size, 0.0);

    // https://gitlab.cern.ch/akhukhun/roccor
    for(unsigned int i = 0; i < size; ++i) {

        res[i] = pt[i] * rochester->kScaleDT(ch[i], pt[i], eta[i], phi[i], fluctuation, 0);
    }
    return res;

}


}

#endif
