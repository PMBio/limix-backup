/*
 * ALmm.h
 *
 *  Created on: Nov 27, 2011
 *      Author: stegle
 */

#ifndef ALMM_H_
#define ALMM_H_

#include "gpmix/types.h"
#include "gpmix/utils/matrix_helper.h"

namespace gpmix {

//rename argout operators for swig interface
#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
//ignore C++ versions
%ignore ALMM::getPheno;
%ignore ALMM::getPv;
%ignore ALMM::getSnps;
%ignore ALMM::getCovs;
%ignore ALMM::getK;
%ignore ALMM::getPermutation;



//rename argout versions for python; this overwrites the C++ convenience functions
%rename(getPheno) ALMM::agetPheno;
%rename(getPv) ALMM::agetPv;
%rename(getSnps) ALMM::agetSnps;
%rename(getCovs) ALMM::agetCovs;
%rename(getK) ALMM::agetK;
%rename(getPermutation) ALMM::agetPermutation;
#endif

//Abstract base class for LMM models*/
class ALMM {
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
	muint_t num_intervalsAlt,num_intervals0;
	mfloat_t ldeltamin0,ldeltamax0;
	mfloat_t ldeltaminAlt,ldeltamaxAlt;
	//results:
	MatrixXd pv;
	//state of decompostion cache:
	bool UK_cached;
	bool Usnps_cached;
	bool Upheno_cached;
	bool Ucovs_cached;
	int testStatistics;

	virtual void clearCache();
	void applyPermutation(MatrixXd& V) throw(CGPMixException);
public:
	ALMM();
	virtual ~ALMM();

	//constants: test statistics
	static const int TEST_LLR=0;
	static const int TEST_F=1;

	//setter fors data:
	mfloat_t getLdeltamin0() const;
	muint_t getNumIntervalsAlt() const;
	muint_t getNumSamples() const;
	mfloat_t getLdeltaminAlt() const;
	void setLdeltaminAlt(mfloat_t ldeltaminAlt);
	void setLdeltamin0(mfloat_t ldeltamin0);
	void setNumIntervalsAlt(muint_t num_intervalsAlt);

	//getters:
	void agetPheno(MatrixXd *out) const;
	void agetPv(MatrixXd *out) const;
	void agetSnps(MatrixXd *out) const;
	void agetCovs(MatrixXd* out) const;
	//setters:
	void setCovs(const MatrixXd& covs);
	void setPheno(const MatrixXd& pheno);
	void setSNPs(const MatrixXd& snps);

	//abstract function
	virtual void process() throw(CGPMixException) =0;
	virtual void updateDecomposition() throw(CGPMixException) =0;

	virtual void agetK(MatrixXd* out) const;
	virtual void setK(const MatrixXd& K);
	void setPermutation(const VectorXi& perm);
	void agetPermutation(VectorXi* out) const;

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
	VectorXd XSY;
	VectorXd beta;
	MatrixXd res;
	VectorXd Sdi;
	MatrixXd XSdi;
	MatrixXd U_X;
	MatrixXd S_X;
	MatrixXd sigg2;

public:
	CLMMCore() : solver2(2),solver3(3)
	{
	}
	//selfadjoint eigen solver which is optimized for 1d/2d/3d matrices
	template <typename Derived, typename OtherDerived>
	inline void SelfAdjointEigenSolver(const Eigen::MatrixBase<Derived>& U, const Eigen::MatrixBase<Derived>& S, const Eigen::MatrixBase<OtherDerived>& M);
	//public functions for nlleval, optdelta etc.
	mfloat_t optdelta(const MatrixXd& UY,const MatrixXd& UX,const MatrixXd& S,int numintervals,double ldeltamin,double ldeltamax);
	mfloat_t nLLeval(VectorXd* F_tests, double ldelta,const MatrixXd& UY,const MatrixXd& UX,const VectorXd& S);

	template <typename Derived1, typename Derived2,typename Derived3,typename Derived4, typename Derived5>
	inline void nLLevalEx(const Eigen::MatrixBase<Derived1>& AOF_tests_,const Eigen::MatrixBase<Derived2>& AOnLL_,const Eigen::MatrixBase<Derived3>& UY, const Eigen::MatrixBase<Derived4>& UX, const Eigen::MatrixBase<Derived5>& S,double ldelta,bool calc_ftest=false);

};


/*
mfloat_t CLMMCore::optdelta(const MatrixXd & UY, const MatrixXd & UX, const MatrixXd & S, int numintervals, double ldeltamin, double ldeltamax)
{
	//grid variable with the current likelihood evaluations
	MatrixXd nllgrid     = MatrixXd::Ones(numintervals,1).array()*HUGE_VAL;
	MatrixXd ldeltagrid = MatrixXd::Zero(numintervals, 1);
	//current delta
	mfloat_t ldelta = ldeltamin;
	mfloat_t ldeltaD = (ldeltamax - ldeltamin);
	ldeltaD /= ((mfloat_t)((((((numintervals)))))) - 1);
	mfloat_t nllmin = HUGE_VAL;
	mfloat_t ldeltaopt_glob = 0;
	for(int i = 0;i < (numintervals);i++){
		nllgrid(i, 0) = this->nLLeval(NULL, ldelta, UY, UX, S);
		ldeltagrid(i, 0) = ldelta;
		if(nllgrid(i, 0) < nllmin){
			nllmin = nllgrid(i, 0);
			ldeltaopt_glob = ldelta;
		}
		//move on delta
		ldelta += ldeltaD;
	}
	return ldeltaopt_glob;
}
*/

template <typename Derived1, typename Derived2,typename Derived3,typename Derived4,typename Derived5>
inline void CLMMCore::nLLevalEx(const Eigen::MatrixBase<Derived1>& AOF_tests_,const Eigen::MatrixBase<Derived2>& AOnLL_,const Eigen::MatrixBase<Derived3>& UY, const Eigen::MatrixBase<Derived4>& UX, const Eigen::MatrixBase<Derived5>& S,double ldelta,bool calc_ftest)
{
	//cast out arguments
	Eigen::MatrixBase<Derived1>& AOF_tests = const_cast< Eigen::MatrixBase<Derived1>& >(AOF_tests_);
	Eigen::MatrixBase<Derived2>& AOnLL = const_cast< Eigen::MatrixBase<Derived2>& >(AOnLL_);


	//number of samples
	muint_t n = UX.rows();
	//number of dimensions for fitting (X)
	muint_t d = UX.cols();
	//number of phenotypes to evaluate:
	//muint_t p = UY.cols();

	assert(UY.rows() == S.rows());
	assert(UY.rows() == UX.rows());

	mfloat_t delta = exp(ldelta);
	Sdi = S.array() + delta;
	mfloat_t ldet = 0.0;
	//calc log det and invert at the same time Sdi elementwise
	for(size_t ind = 0;ind < n;++ind)
	{
		//ldet
		ldet += log(Sdi.data()[ind]);
		//inverse:
		Sdi.data()[ind] = 1.0 / (Sdi.data()[ind]);
	}

	if (calc_ftest)
		AOF_tests.setConstant(0.0);

	XSdi = UX.array().transpose();
	XSdi.array().rowwise() *= Sdi.array().transpose();
	XSX.noalias() = XSdi * UX;
	XSY.noalias() = XSdi * UY;

	//least squares solution of XSX*beta = XSY
	//Call internal solver which uses fixed solvers for 2d and 3d matrices
	SelfAdjointEigenSolver(U_X, S_X, XSX);
	beta.noalias() = U_X.transpose() * XSY;

	//loop over genotype dimensions:
	for(size_t dim = 0;dim < d;++dim)
	{
		if(S_X(dim) > 3E-8)
		{
			beta.row(dim).array() /= S_X(dim);
			if (calc_ftest)
			{
				for(size_t dim2 = 0;dim2 < d;++dim2)
					AOF_tests.array().row(dim2) += U_X(dim2, dim) * U_X(dim2, dim) / S_X(dim, 0);
			}
		}
		else
			beta.row(dim).setConstant(0.0);
	}
	beta = U_X * beta;
	res.noalias() = UY - UX * beta;
	//sqared residuals
	res.array() *= res.array();
	res.array() *= Sdi.array();

	sigg2 = res.colwise().sum() / (n);
	//compute the F-statistics
	if(calc_ftest)
	{
		AOF_tests.array() = beta.array() * beta.array() / AOF_tests.array();
		AOF_tests.array().rowwise() /= sigg2.array().row(0);
	}

	//WARNING: elementwise log of sigg2
	logInplace(sigg2);
	//calc likelihood:
	AOnLL = 0.5*n*sigg2;
	AOnLL.array() += 0.5*(n*L2pi+ldet + n);

}





#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
//ignore C++ versions
#endif
//Standard mixed liner model
class CLMM :  public CLMMCore, public ALMM
{
protected:
	//caching variables
	MatrixXd U;
	VectorXd S;


public:
	CLMM();
	virtual ~CLMM();

	//function to add testing kernel

	//processing;
	virtual void process() throw(CGPMixException);
	virtual void updateDecomposition() throw(CGPMixException);

	void setK(const MatrixXd& K,const MatrixXd& U, const VectorXd& S);
	void setK(const MatrixXd& K)
	{ ALMM::setK(K);}
};


//interaction tests
#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
//ignore C++ versions
%ignore CInteractLMM::getInter;

//rename argout versions for python; this overwrites the C++ convenience functions
%rename(getInter) CInteractLMM::agetInter;
#endif

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
	virtual void process() throw(CGPMixException);
	virtual void updateDecomposition() throw(CGPMixException);


	//convenience
	MatrixXd getInter() const;
	MatrixXd getInter0() const;
    bool isRefitDelta0Pheno() const;
    void setRefitDelta0Pheno(bool refitDelta0Pheno);
};



/*Simple Kronecker model.
 * Kroneckers are merely used to efficiently rotate the phenotypes and genotypes
 */

class CSimpleKroneckerLMM: public ALMM
{
protected:
	MatrixXd C;
	MatrixXd R;
	MatrixXd U_R;
	MatrixXd U_C;
	VectorXd S_R;
	VectorXd S_C;
	VectorXd S;
	//kronecker structure: Wkron - p x D where D is the number of weights to be fitted
	MatrixXd Wkron;
	//kronecker structure: same but for the background model:
	MatrixXd Wkron0;

	void kron_snps(MatrixXd* out,const MatrixXd& x,const MatrixXd& kron);

	void kron_rot(MatrixXd* out,const MatrixXd&  x);
public:
	CSimpleKroneckerLMM();
	virtual ~CSimpleKroneckerLMM();
	//processing;
	virtual void process() throw(CGPMixException);
	virtual void updateDecomposition() throw(CGPMixException);

	void setK_C(const MatrixXd& C);
	void setK_C(const MatrixXd& C,const MatrixXd& U_C, const VectorXd& S_C);

	void setK_R(const MatrixXd& R);
	void setK_R(const MatrixXd& R,const MatrixXd& U_R, const VectorXd& S_R);

	void getK_R(MatrixXd* out) const;
	void getK_C(MatrixXd* out) const;
	void setWkron(const MatrixXd& Wkron);
    void setWkron0(const MatrixXd& Wkron0);


#ifndef SWIG
    MatrixXd getK_R() const;
    MatrixXd getK_C() const;
    MatrixXd getWkron() const;
    MatrixXd getWkron0() const;
#endif

};

/*
Efficient mixed liner model
However, there are limitations as to which hypothesis can be tested
TODO: add documentation here
*/
class CKroneckerLMM : public ALMM
{
	//TODO: check what is a symmetric matrix type!
protected:
	MatrixXd C;
	MatrixXd R;
	MatrixXd U_R;
	MatrixXd U_C;
	VectorXd S_R;
	VectorXd S_C;
	MatrixXd WkronDiag0;
	MatrixXd WkronBlock0;
	MatrixXd WkronDiag;
	MatrixXd WkronBlock;

public:
	CKroneckerLMM();
	virtual ~CKroneckerLMM();

	//processing;
	virtual void process() throw(CGPMixException);
	virtual void updateDecomposition() throw(CGPMixException);

	void setK_C(const MatrixXd& C);
	void setK_C(const MatrixXd& C,const MatrixXd& U_C, const VectorXd& S_C);

	void setK_R(const MatrixXd& R);
	void setK_R(const MatrixXd& R,const MatrixXd& U_R, const VectorXd& S_R);

	void getK_R(MatrixXd* out) const;
	void getK_C(MatrixXd* out) const;
	void setKronStructure(const MatrixXd& WkronDiag0, const MatrixXd& WkronBlock0, const MatrixXd& WkronDiag, const MatrixXd& WkronBlock);
	static mfloat_t nLLeval(MatrixXd* F_tests, mfloat_t ldelta, const MatrixXd& WkronDiag, const MatrixXd& WkronBlock, const MatrixXd& UX, const MatrixXd& UYU, const VectorXd& S_C, const VectorXd& S_R);
	static mfloat_t optdelta(const MatrixXd& UX, const MatrixXd& UYU, const VectorXd& S_C, const VectorXd& S_R, const muint_t numintervals, const mfloat_t ldeltamin, const mfloat_t ldeltamax, const MatrixXd& WkronDiag, const MatrixXd& WkronBlock);
#ifndef SWIG
	MatrixXd getK_R() const;
	MatrixXd getK_C() const;
#endif
};



/* Standalone helper functions.
 * These will also be wrapped in python
 */


void train_associations_SingleSNP(MatrixXd* PV, MatrixXd* LL, MatrixXd* ldelta,
		const MatrixXd& X, const MatrixXd& Y, const MatrixXd& U,
		const MatrixXd& S, const MatrixXd& C, int numintervals,
		double ldeltamin, double ldeltamax);
double optdelta(const MatrixXd& UY,const MatrixXd& UX,const MatrixXd& S,int numintervals,double ldeltamin,double ldeltamax);
void optdeltaAllY(MatrixXd* out, const MatrixXd& UY, const MatrixXd& UX, const MatrixXd& S, const MatrixXd& ldeltagrid);
double nLLeval(MatrixXd* F_tests, double ldelta,const MatrixXd& UY,const MatrixXd& UX,const MatrixXd& S);
void nLLevalAllY(MatrixXd* out, double ldelta,const MatrixXd& UY,const MatrixXd& UX,const VectorXd& S);



/* Inline functions */

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


} /* namespace gpmix */
#endif /* ALMM_H_ */
