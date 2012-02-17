/*
 * kronecker_lmm.h
 *
 *  Created on: Feb 11, 2012
 *      Author: stegle
 */

#ifndef KRONECKER_LMM_H_
#define KRONECKER_LMM_H_

#include "gpmix/LMM/lmm.h"
#include "gpmix/gp/gp_kronecker.h"
#include "gpmix/mean/CKroneckerMean.h"
#include "gpmix/LMM/CGPLMM.h"

namespace gpmix {



/*
Efficient linear mixed model
However, there are limitations as to which hypothesis can be tested
Note: this class is probably redundant. Complex testing models are now implemented as mean functions
in kronecker_gp.
*/
#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
//%shared_ptr(gpmix::CKroneckerLMM)
#endif
class CKroneckerLMM : public CGPLMM
{
protected:

public:
	CKroneckerLMM(PGPkronecker gp) : CGPLMM(gp)
	{
	}
#ifdef returnW
	static mfloat_t nLLeval(std::vector<MatrixXd>& W, std::vector<MatrixXd>& F_tests, mfloat_t ldelta, const std::vector<MatrixXd>& A, const std::vector<MatrixXd>& X, const MatrixXd& Y, const VectorXd& S_C, const VectorXd& S_R);
#else
	static mfloat_t nLLeval(mfloat_t ldelta, const std::vector<MatrixXd>& A, const std::vector<MatrixXd>& X, const MatrixXd& Y, const VectorXd& S_C, const VectorXd& S_R);
#endif
	virtual void process() throw (CGPMixException);
	static mfloat_t optdelta(mfloat_t& ldelta_opt, const std::vector<MatrixXd>& A, const std::vector<MatrixXd>& X, const MatrixXd& Y, const VectorXd& S_C, const VectorXd& S_R, mfloat_t ldeltamin, mfloat_t ldeltamax, muint_t numintervals);
#if 0
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
	static mfloat_t nLLeval(std::vector<MatrixXd>& W, std::vector<MatrixXd>& F_tests, mfloat_t ldelta, const std::vector<MatrixXd>& A, const std::vector<MatrixXd>& X, const MatrixXd& Y, const VectorXd& S_C, const VectorXd& S_R);
	static mfloat_t optdelta(const MatrixXd& UX, const MatrixXd& UYU, const VectorXd& S_C, const VectorXd& S_R, const muint_t numintervals, const mfloat_t ldeltamin, const mfloat_t ldeltamax, const MatrixXd& WkronDiag, const MatrixXd& WkronBlock);
	MatrixXd getK_R() const;
	MatrixXd getK_C() const;
#endif
};



}


#endif /* KRONECKER_LMM_H_ */
