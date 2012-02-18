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

	virtual void process() throw (CGPMixException);
	static mfloat_t nLLeval(mfloat_t ldelta, const MatrixXdVec& A,const MatrixXdVec& X, const MatrixXd& Y, const VectorXd& S_C, const VectorXd& S_R);
	static mfloat_t optdelta(mfloat_t& ldelta_opt, const MatrixXdVec& A,const MatrixXdVec& X, const MatrixXd& Y, const VectorXd& S_C, const VectorXd& S_R, mfloat_t ldeltamin, mfloat_t ldeltamax, muint_t numintervals);

};



}


#endif /* KRONECKER_LMM_H_ */
