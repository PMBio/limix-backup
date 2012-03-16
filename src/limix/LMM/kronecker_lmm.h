/*
 * kronecker_lmm.h
 *
 *  Created on: Feb 11, 2012
 *      Author: stegle
 */

#ifndef KRONECKER_LMM_H_
#define KRONECKER_LMM_H_

#include "limix/LMM/lmm.h"
#include "limix/gp/gp_kronecker.h"
#include "limix/mean/CKroneckerMean.h"
#include "limix/LMM/CGPLMM.h"

namespace limix {



/*
Efficient linear mixed model
However, there are limitations as to which hypothesis can be tested
Note: this class is probably redundant. Complex testing models are now implemented as mean functions
in kronecker_gp.
*/
#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
%ignore CKroneckerLMM::agetKr;
%ignore CKroneckerLMM::agetKc;

%rename(getKr) CGPHyperParams::agetKr;
%rename(getKc) CGPHyperParams::agetKc;
#endif
class CKroneckerLMM : public CGPLMM
{
protected:
	MatrixXd Kr,Kc;
	//decompositiosn thare being used for testing:
	MatrixXd Ur,Uc;
	VectorXd Sr,Sc;

	//init function for manually specified kernels
	void initTestingK();
	//init function when using GP object
	void initTestingGP();

public:
	CKroneckerLMM(PGPkronecker gp) : CGPLMM(gp)
	{
	}
	CKroneckerLMM()
	{
	}
	virtual ~CKroneckerLMM()
	{}

	virtual void process() throw (CGPMixException);
	static mfloat_t nLLeval(mfloat_t ldelta, const MatrixXdVec& A,const MatrixXdVec& X, const MatrixXd& Y, const VectorXd& S_C, const VectorXd& S_R);
	static mfloat_t optdelta(mfloat_t& ldelta_opt, const MatrixXdVec& A,const MatrixXdVec& X, const MatrixXd& Y, const VectorXd& S_C, const VectorXd& S_R, mfloat_t ldeltamin, mfloat_t ldeltamax, muint_t numintervals);

	//setter and getter for manual operation
	void setKr(const MatrixXd& K)
	{
		this->Kr = K;
	}
	void setKc(const MatrixXd& K)
	{
		this->Kc =K;
	}

	void agetKr(MatrixXd* out)
	{
		(*out) = Kr;
	}
	void agetKc(MatrixXd* out)
	{
		(*out) = Kc;
	}
	MatrixXd getKr()
	{
		return Kr;
	}
	MatrixXd getKc()
	{
		return Kc;
	}
};



}


#endif /* KRONECKER_LMM_H_ */
