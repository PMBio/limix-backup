/*
 * ALmm.h
 *
 *  Created on: Nov 27, 2011
 *      Author: stegle
 */

#ifndef ALMM_H_
#define ALMM_H_

#include "gpmix/types.h"

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


class CFastFixedEigenSolver
{
protected:
	//instances of eigensolvers for caching
	Eigen::SelfAdjointEigenSolver<MatrixXd2> solver2;
	Eigen::SelfAdjointEigenSolver<MatrixXd3> solver3;

public:
	CFastFixedEigenSolver() : solver2(2),solver3(3)
	{
	}
	template <typename Derived, typename OtherDerived>
	inline void SelfAdjointEigenSolver(const Eigen::MatrixBase<Derived>& U, const Eigen::MatrixBase<Derived>& S, const Eigen::MatrixBase<OtherDerived>& M);
};





#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
//ignore C++ versions
#endif
//Standard mixed liner model
class CLMM : public ALMM, CFastFixedEigenSolver
{
protected:
	//caching variables
	MatrixXd U;
	VectorXd S;

	//variables for optimization etc.
	MatrixXd XSX;
	VectorXd XSY;
	VectorXd beta;
	MatrixXd res;
	VectorXd Sdi;
	MatrixXd XSdi;
	MatrixXd U_X;
	MatrixXd S_X;

	double optdelta(const MatrixXd& UY,const MatrixXd& UX,const MatrixXd& S,int numintervals,double ldeltamin,double ldeltamax);
	double nLLeval(VectorXd* F_tests, double ldelta,const VectorXd& UY,const MatrixXd& UX,const VectorXd& S);
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
void nLLevalAllY(MatrixXd* out, double ldelta,const MatrixXd& UY,const MatrixXd& UX,const MatrixXd& S);



/* Inline functions */

template <typename Derived, typename OtherDerived>
inline void CFastFixedEigenSolver::SelfAdjointEigenSolver(const Eigen::MatrixBase<Derived>& U, const Eigen::MatrixBase<Derived>& S, const Eigen::MatrixBase<OtherDerived>& M)
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
