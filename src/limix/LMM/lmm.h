// Copyright(c) 2014, The LIMIX developers(Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
//
//Licensed under the Apache License, Version 2.0 (the "License");
//you may not use this file except in compliance with the License.
//You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//Unless required by applicable law or agreed to in writing, software
//distributed under the License is distributed on an "AS IS" BASIS,
//WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//See the License for the specific language governing permissions and
//limitations under the License.

#ifndef ALMM_H_
#define ALMM_H_

#include "limix/types.h"
#include "limix/utils/matrix_helper.h"
#include <math.h>
#include "limix/utils/brentc.h"


namespace limix {

//Abstract base class for LMM models*/
class ALMM
{
protected:
	//Data and sample information
	MatrixXd snps;
	MatrixXd pheno;
	MatrixXd covs;
	MatrixXd Usnps;
	MatrixXd Upheno;
	MatrixXd Ucovs;
	MatrixXd K;
	//permutation, if needed:
	VectorXi perm;

	//number of samples, snps and pheno
	muint_t num_samples,num_pheno,num_snps,num_covs;
	//common settings
	muint_t num_intervalsAlt;
	muint_t num_intervals0;
	mfloat_t ldeltamin0;
	mfloat_t ldeltamax0;
	mfloat_t ldeltaInit;
	mfloat_t ldeltaminAlt;
	mfloat_t ldeltamaxAlt;
	//results:
	MatrixXd pv;
	//state of decompostion cache:
	bool UK_cached;
	bool Usnps_cached;
	bool Upheno_cached;
	bool Ucovs_cached;
	int testStatistics;

	virtual void clearCache();
	template <typename Derived1>
	inline void applyPermutation(const Eigen::MatrixBase<Derived1> & M_) ;
public:
	ALMM();
	virtual ~ALMM();

	//constants: test statistics
	static const int TEST_LRT=0;
	static const int TEST_F=1;
	muint_t getNumIntervals0() const;
    void setNumIntervals0(muint_t num_intervals0);
    void setNumIntervalsAlt(muint_t num_intervalsAlt);
    muint_t getNumIntervalsAlt() const;
    mfloat_t getLdeltamin0() const;
    void setLdeltamin0(mfloat_t ldeltamin0);
    void setLdeltaminAlt(mfloat_t ldeltaminAlt);
    mfloat_t getLdeltaminAlt() const;
	mfloat_t getLdeltamaxAlt() const;
    void setLdeltamaxAlt(mfloat_t ldeltamaxAlt);
    mfloat_t getLdeltamax0() const;
    void setLdeltamax0(mfloat_t ldeltamax0);
	void setLdeltaInit(mfloat_t logdelta);
	mfloat_t getLdeltaInit() const;

    //setter fors data:
    muint_t getNumSamples() const;
    //getters:
    void agetPheno(MatrixXd *out) const;
    void agetPv(MatrixXd *out) const;
    void agetSnps(MatrixXd *out) const;
    void agetCovs(MatrixXd *out) const;
    //setters:
    void setCovs(const MatrixXd & covs);
    void setPheno(const MatrixXd & pheno);
    void setSNPs(const MatrixXd & snps);
    //abstract function
    virtual void process()=0; // =0;

	virtual void agetK(MatrixXd* out) const;
	virtual void setK(const MatrixXd& K);
	void setPermutation(const VectorXi& perm);
	void agetPermutation(VectorXi* out) const;

	//setting of certain behaviour
	void setVarcompApprox0(mfloat_t ldeltamin0 = -5, mfloat_t ldeltamax0=5,muint_t num_intervals0=100)
	{
		this->ldeltamax0 = ldeltamax0;
		this->ldeltamin0 = ldeltamin0;
		this->num_intervals0 = num_intervals0;
		this->ldeltaminAlt =0;
		this->ldeltamaxAlt =0;
		this->num_intervalsAlt = 0;
	}
	void setVarcompExact(mfloat_t ldeltamin = -5, mfloat_t ldeltamax=5,muint_t num_intervals=100)
	{
			this->ldeltamaxAlt = ldeltamax;
			this->ldeltaminAlt = ldeltamin;
			this->num_intervalsAlt = num_intervals;
			this->setTestStatistics(this->TEST_F);
	}


	//convenience wrappers:
	VectorXi getPermutation() const;
	MatrixXd getK() const;
	MatrixXd getPheno() const;
	MatrixXd getPv() const;
	MatrixXd getSnps() const;
	MatrixXd getCovs() const;
    int getTestStatistics() const;
    void setTestStatistics(int testStatistics);
};

class CLMMCore
{
protected:
	//instances of eigensolvers for caching
	Eigen::SelfAdjointEigenSolver<MatrixXd2> solver2;
	Eigen::SelfAdjointEigenSolver<MatrixXd3> solver3;

	//variables for optimization etc.
	MatrixXd XSX;
	MatrixXd XSY;
	MatrixXd beta;
	MatrixXd res;
	VectorXd Sdi;
	MatrixXd XSdi;
	MatrixXd U_X;
	MatrixXd S_X;
	MatrixXd sigg2;
	MatrixXd nllgrid;
	VectorXd ldeltagrid;

public:
	CLMMCore() : solver2(2),solver3(3)
	{
	}
	//selfadjoint eigen solver which is optimized for 1d/2d/3d matrices
	template <typename Derived, typename OtherDerived>
	inline void SelfAdjointEigenSolver(const Eigen::MatrixBase<Derived>& U, const Eigen::MatrixBase<Derived>& S, const Eigen::MatrixBase<OtherDerived>& M);
	//public functions for nlleval, optdelta etc.
	template <typename Derived1, typename Derived2,typename Derived3,typename Derived4, typename Derived5,typename Derived6,typename Derived7,typename Derived8>
	inline void nLLevalEx(const Eigen::MatrixBase<Derived1>& AObeta_, const Eigen::MatrixBase<Derived2>& AObeta_ste_, const Eigen::MatrixBase<Derived3>& AOsigma_, const Eigen::MatrixBase<Derived4>& AOF_tests_,const Eigen::MatrixBase<Derived5>& AOnLL_,const Eigen::MatrixBase<Derived6>& UY, const Eigen::MatrixBase<Derived7>& UX, const Eigen::MatrixBase<Derived8>& S,mfloat_t ldelta,bool calc_ftest=false,bool calc_ste=false, bool REML = false);
	template <typename Derived1, typename Derived2,typename Derived3,typename Derived4,typename Derived5>
	inline void optdeltaEx(const Eigen::MatrixBase<Derived1> & AO_delta_,const Eigen::MatrixBase<Derived2> & AO_NLL_, const Eigen::MatrixBase<Derived3>& UY,const Eigen::MatrixBase<Derived4>& UX,const Eigen::MatrixBase<Derived5>& S,muint_t numintervals,mfloat_t ldeltamin,mfloat_t ldeltamax,bool REML = false);
};

//Standard mixed liner model
class CLMM :  public CLMMCore, public ALMM
{
protected:
	//caching variables
	MatrixXd U;
	VectorXd S;
	//verbose results:
	bool storeVerboseResults;
	bool calc_stes; //!<calculate standard erorrs?
	MatrixXd nLL0, nLLAlt,f_tests,ldeltaAlt,ldelta0,beta_snp,beta_snp_ste,lsigma;


public:
	CLMM();
	virtual ~CLMM();

	//function to add testing kernel

	/*! process asssociation test*/
	virtual void process();// ;
	/*! public function to update the covaraince decomposition*/
	virtual void updateDecomposition() ;

	/*! set the decompsotion elemnts of K directly \param U: eigenvectors, \param S: eigen vectors*/
	void setKUS(const MatrixXd& K,const MatrixXd& U, const VectorXd& S)
	{
		setK(K,U,S);
	}

	void setK(const MatrixXd& K,const MatrixXd& U, const VectorXd& S);
	void setK(const MatrixXd& K)
	{ ALMM::setK(K);}

	void agetNLL0(MatrixXd* out)
	{
		(*out) = nLL0;
	}
	void agetNLLAlt(MatrixXd *out)
	{
		(*out) = nLLAlt;
	}
	void agetFtests(MatrixXd *out)
	{
		(*out) = f_tests;
	}

	void agetLdeltaAlt(MatrixXd *out)
	{
		(*out) = ldeltaAlt;
	}
	void agetLdelta0(MatrixXd *out)
	{
		(*out) = ldelta0;
	}

	void agetLSigma(MatrixXd *out)
	{
		(*out) = lsigma;
	}

	void agetBetaSNP(MatrixXd *out)
	{
		(*out) = beta_snp;
	}

	void agetBetaSNPste(MatrixXd *out)
	{
		(*out) = beta_snp_ste;
	}

	MatrixXd getLSigma()
	{
		return lsigma;
	}

	MatrixXd getBetaSNP()
	{
		return beta_snp;
	}

	MatrixXd getBetaSNPste()
	{
		return beta_snp_ste;
	}

	MatrixXd getNLL0()
	{
		return nLL0;
	}
	MatrixXd getNLLAlt()
	{
		return nLLAlt;
	}
	MatrixXd getFtests()
	{
		return f_tests;
	}

	MatrixXd getLdelta0()
	{
		return ldelta0;
	}

	MatrixXd getLdeltaAlt()
	{
		return ldeltaAlt;
	}
	void setLdeltaInit(mfloat_t logdelta)
	{
		this->ldeltaInit=logdelta;
	}
	mfloat_t getLdeltaInit()
	{
		return this->ldeltaInit;
	}
	/*! calculate standard errors?*/
	bool isCalcStes() const {
		return calc_stes;
	}
	/*! calculate standard errors?*/
	void setCalcStes(bool calcStes) {
		calc_stes = calcStes;
	}
};
typedef sptr<CLMM> PLMM;


class CInteractLMM : public CLMM
{
protected:

	//interaction data: foreground model
	MatrixXd I;
	//interaction data: null model
	MatrixXd I0;
	//rotated interaction term
	MatrixXd Uinter;
	muint_t num_inter,num_inter0;
	bool refitDelta0Pheno;

public:
	CInteractLMM();
	virtual ~CInteractLMM();

	//setter, getter
	void setInter(const MatrixXd& Inter);
	void agetInter(MatrixXd* out) const;
	//setter, getter
	void setInter0(const MatrixXd& Inter);
	void agetInter0(MatrixXd* out) const;


	//processing;
	virtual void process(); //;
	virtual void updateDecomposition() ;


	//convenience
	MatrixXd getInter() const;
	MatrixXd getInter0() const;
    bool isRefitDelta0Pheno() const;
    void setRefitDelta0Pheno(bool refitDelta0Pheno);
};
typedef sptr<CInteractLMM> PInteractLMM;






/* Standalone helper functions.
 * These will also be wrapped in python
 */


void train_associations_SingleSNP(MatrixXd* PV, MatrixXd* LL, MatrixXd* ldelta,
		const MatrixXd& X, const MatrixXd& Y, const MatrixXd& U,
		const MatrixXd& S, const MatrixXd& C, int numintervals,
		double ldeltamin, double ldeltamax);
double optdelta(const MatrixXd& UY,const MatrixXd& UX,const MatrixXd& S,int numintervals,double ldeltamin,double ldeltamax, bool REML = false);
void optdeltaAllY(MatrixXd* out, const MatrixXd& UY, const MatrixXd& UX, const MatrixXd& S, const MatrixXd& ldeltagrid);
double nLLeval(MatrixXd* F_tests, double ldelta,const MatrixXd& UY,const MatrixXd& UX,const MatrixXd& S, bool REML=false);
void nLLevalAllY(MatrixXd* out, double ldelta,const MatrixXd& UY,const MatrixXd& UX,const VectorXd& S);



/* Inline functions */
template <typename Derived1>
inline void ALMM::applyPermutation(const Eigen::MatrixBase<Derived1>& M_) 
{
	//cast out arguments
	Eigen::MatrixBase<Derived1>& M = const_cast< Eigen::MatrixBase<Derived1>& >(M_);

    if(isnull(perm))
        return;

    if(perm.rows() != M.rows()){
        throw CLimixException("ALMM:Permutation vector has incompatible length");
    }
    //create temporary copy
    MatrixXd Mc = M;
    //apply permutation;
    for(muint_t i = 0;i < (muint_t)((((Mc.rows()))));++i){
        M.row(i) = Mc.row(perm(i));
    }
}




template <typename Derived, typename OtherDerived>
inline void CLMMCore::SelfAdjointEigenSolver(const Eigen::MatrixBase<Derived>& U, const Eigen::MatrixBase<Derived>& S, const Eigen::MatrixBase<OtherDerived>& M)
{
	//1. check size of matrix
	muint_t dim = M.rows();
    if (dim==1)
    {
		//trivial
    	const_cast<Eigen::MatrixBase<Derived>& >(U) = MatrixXd::Ones(1,1);
    	const_cast<Eigen::MatrixBase<Derived>& >(S) = M;
    }
    else if (dim==2)
    {
    	solver2.computeDirect(M);
    	const_cast<Eigen::MatrixBase<Derived>& >(U) = solver2.eigenvectors();
    	const_cast<Eigen::MatrixBase<Derived>& >(S) = solver2.eigenvalues();
    }
    else if (dim==3)
    {
    	//use eigen direct solver
    	solver3.computeDirect(M);
    	const_cast<Eigen::MatrixBase<Derived>& >(U) = solver3.eigenvectors();
    	const_cast<Eigen::MatrixBase<Derived>& >(S) = solver3.eigenvalues();
    }
    else
    {
    	//use dynamic standard solver
    	Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver(M);
    	const_cast<Eigen::MatrixBase<Derived>& >(U) = eigensolver.eigenvectors();
    	const_cast<Eigen::MatrixBase<Derived>& >(S) = eigensolver.eigenvalues();
	}
}


template <typename Derived1, typename Derived2,typename Derived3,typename Derived4, typename Derived5,typename Derived6,typename Derived7,typename Derived8>
inline void CLMMCore::nLLevalEx(const Eigen::MatrixBase<Derived1>& AObeta_, const Eigen::MatrixBase<Derived2>& AObeta_ste_, const Eigen::MatrixBase<Derived3>& AOsigma_, const Eigen::MatrixBase<Derived4>& AOF_tests_,const Eigen::MatrixBase<Derived5>& AOnLL_,const Eigen::MatrixBase<Derived6>& UY, const Eigen::MatrixBase<Derived7>& UX, const Eigen::MatrixBase<Derived8>& S,mfloat_t ldelta,bool calc_ftest,bool calc_ste, bool REML)
{
	//cast out arguments
	Eigen::MatrixBase<Derived1>& AObeta = const_cast< Eigen::MatrixBase<Derived1>& >(AObeta_);
	Eigen::MatrixBase<Derived2>& AObeta_ste = const_cast< Eigen::MatrixBase<Derived2>& >(AObeta_ste_);
	Eigen::MatrixBase<Derived3>& AOsigma = const_cast< Eigen::MatrixBase<Derived3>& >(AOsigma_);
	Eigen::MatrixBase<Derived4>& AOF_tests = const_cast< Eigen::MatrixBase<Derived4>& >(AOF_tests_);
	Eigen::MatrixBase<Derived5>& AOnLL = const_cast< Eigen::MatrixBase<Derived5>& >(AOnLL_);


	//number of samples
	muint_t n = UX.rows();
	//number of dimensions for fitting (X)
	muint_t d = UX.cols();
	//number of phenotypes to evaluate:
	muint_t p = UY.cols();

	//resize output arguments as needed
	AOnLL.derived().resize(1,p);
	AObeta.derived().resize(d,p);
	AOsigma.derived().resize(1,p);

	if (calc_ftest || calc_ste)
		AOF_tests.derived().resize(d,p);
	if (calc_ste)
		AObeta_ste.derived().resize(d,p);


	assert(UY.rows() == S.rows());
	assert(UY.rows() == UX.rows());

	mfloat_t delta = exp(ldelta);
	Sdi = S.array() + delta;
	mfloat_t ldet = 0.0;
	mfloat_t ldetXX = 0.0;
	mfloat_t ldetXSX = 0.0;
	//calc log det and invert at the same time Sdi elementwise
	for(size_t ind = 0;ind < n;++ind)
	{
		//ldet
		ldet += log(Sdi.data()[ind]);
		//inverse:
		Sdi.data()[ind] = 1.0 / (Sdi.data()[ind]);
	}
	//check whether ldet is NAN, => set to infinity
#ifdef _MSC_VER
	if (ldet!=ldet)
	{
		ldet=std::numeric_limits<mfloat_t>::infinity();
	}
#else
	if(ldet==NAN)
	{
		ldet=INFINITY;
	}
#endif

	if (calc_ftest || calc_ste)
		AOF_tests.setConstant(0.0);


	XSdi = UX.array().transpose();
	XSdi.array().rowwise() *= Sdi.array().transpose();
	XSX.noalias() = XSdi * UX;
	XSY.noalias() = XSdi * UY;

	//least squares solution of XSX*beta = XSY
	//Call internal solver which uses fixed solvers for 2d and 3d matrices
	SelfAdjointEigenSolver(U_X, S_X, XSX);
	AObeta.noalias() = U_X.transpose() * XSY;

	//loop over genotype dimensions:
	for(size_t dim = 0;dim < d;++dim)
	{
		if(S_X(dim) > 3E-8)
		{
			AObeta.row(dim).array() /= S_X(dim);
			ldetXSX+=log(S_X(dim));
			if (calc_ftest || calc_ste)
			{
				for(size_t dim2 = 0;dim2 < d;++dim2)
					AOF_tests.array().row(dim2) += U_X(dim2, dim) * U_X(dim2, dim) / S_X(dim, 0);
			}
		}
		else
			AObeta.row(dim).setConstant(0.0);
	}

	AObeta = U_X * AObeta;
	res.noalias() = UY - UX * AObeta;
	//sqared residuals
	res.array() *= res.array();
	res.array().colwise() *= Sdi.array();
	
	/*
	std::cout << "UX" << UX << "\n";
	std::cout << "UY" << UX << "\n";
	std::cout << "Sdi" << Sdi << "n";

	std::cout << "AObeta" << AObeta << "\n";
	std::cout << "res" << res << "n";
	std::cout << "ldet" << ldet << "\n";
	*/

	//compute REML/ML specific terms
	if (REML){
		AOsigma = res.colwise().sum() / (mfloat_t)(n-d);
		
		MatrixXd XX = UX.transpose() * UX;
        Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver(XX);
        //MatrixXd U_XX = eigensolver.eigenvectors();
        MatrixXd S_XX = eigensolver.eigenvalues();
		for(size_t dim = 0;dim < d;++dim){
            if(S_XX(dim, 0) > 3E-8){
                ldetXX += log(S_XX(dim, 0));
            }
        }

	}else{
		AOsigma = res.colwise().sum() / (mfloat_t)n;
	}

	//compute standard errors
	if(calc_ste)
	{
		AObeta_ste  = AOF_tests.cwiseInverse();
		AObeta_ste.array().rowwise() *= AOsigma.array().row(0);
		//Christoph: I think this is a bug:
		//ste should not be divided by additional n, That factor is already considered in sigma
		//if (REML){
		//	AObeta_ste/=n-d;
		//}else{
		//	AObeta_ste/=n;
		//}
		AObeta_ste =AObeta_ste.cwiseSqrt();
	}
	//compute the F-statistics
	if(calc_ftest)
	{
		AOF_tests.array() = AObeta.array() * AObeta.array() / AOF_tests.array();
		AOF_tests.array().rowwise() /= AOsigma.array().row(0);
	}


	//WARNING: elementwise log of sigg2
	logInplace(AOsigma);
	//calc likelihood:
	if (REML){
		AOnLL = 0.5*(n-d)*AOsigma;
		AOnLL.array() += 0.5*((n-d)*L2pi+ldet +ldetXX + ldetXSX+ (n-d));
	}else{
		AOnLL = 0.5*n*AOsigma;
		AOnLL.array() += 0.5*(n*L2pi+ldet + n);
	}
}




template <typename Derived1, typename Derived2,typename Derived3,typename Derived4,typename Derived5>
inline void CLMMCore::optdeltaEx(const Eigen::MatrixBase<Derived1> & AO_delta_,const Eigen::MatrixBase<Derived2> & AO_NLL_, const Eigen::MatrixBase<Derived3>& UY,const Eigen::MatrixBase<Derived4>& UX,const Eigen::MatrixBase<Derived5>& S,muint_t numintervals,mfloat_t ldeltamin,mfloat_t ldeltamax,bool REML)
{

	//cast out arguments
	Eigen::MatrixBase<Derived1>& AO_delta = const_cast< Eigen::MatrixBase<Derived1>& >(AO_delta_);
	Eigen::MatrixBase<Derived1>& AO_NLL = const_cast< Eigen::MatrixBase<Derived1>& >(AO_NLL_);


	//number of phenotypes
	muint_t Np = UY.cols();

	//grid variable with the current likelihood evaluations
	AO_delta.derived().resize(Np,1);
	AO_NLL.derived().resize(Np,1);


	nllgrid.resize(numintervals,Np);
	ldeltagrid.resize(numintervals);
    //current delta
    mfloat_t ldelta = ldeltamin;
    mfloat_t ldeltaD = (ldeltamax - ldeltamin);
    ldeltaD /= ((mfloat_t)(numintervals) - 1.0);

    MatrixXd f_tests_,AObeta_,AObeta_ste_,AOsigma_;

    //forall elements in grid do:
    for(muint_t ii = 0;ii < (numintervals);++ii)
    {
    	ldeltagrid(ii) = ldelta;
    	//get nll for all phenotypes jointly (using current ldelta)
    	this->nLLevalEx(f_tests_,AObeta_,AObeta_ste_,AOsigma_, nllgrid.block(ii,0,1,Np),UY, UX, S,ldelta,false,REML);
    	//move on delta
    	ldelta += ldeltaD;
    }
    //std::cout << nllgrid << "\n";
    //get index of minimum for all phenotypes and store
    MatrixXd::Index min_index;
    for (muint_t ip=0;ip<Np;++ip)
    {
      	AO_NLL(ip) = nllgrid.col(ip).minCoeff(&min_index);
      	AO_delta(ip) = ldeltagrid(min_index);
      	/*
      	std::cout << nllgrid.col(ip) << "\n";
    	std::cout << "\n\n";
    	std::cout << "min index:" << min_index << "\n";
    	std::cout << ldeltagrid << "\n";
    	std::cout << "delta:"<< AO_delta(ip) << "\n";
		*/
    }
    //std::cout << AO_NLL << "\n";
    //std::cout << AO_delta << "\n";

}

class nLLevalFunctor: public BrentFunctor{
	MatrixXd f_tests;
	MatrixXd X;
	MatrixXd Y;
	MatrixXd S;
	bool REML;
public:
	nLLevalFunctor(	
		const MatrixXd Y,
		const MatrixXd X,
		const MatrixXd S,
		const bool REML);
   ~nLLevalFunctor();
   virtual mfloat_t operator()(const mfloat_t logdelta);
};


} /* namespace limix */
#endif /* ALMM_H_ */
