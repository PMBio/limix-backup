/*
 * CGPLMM.h
 *
 *  Created on: Feb 13, 2012
 *      Author: stegle
 */

#ifndef CGPLMM_H_
#define CGPLMM_H_

#include "gpmix/LMM/lmm.h"
#include "gpmix/gp/gp_kronecker.h"
#include "gpmix/gp/gp_opt.h"
#include "gpmix/mean/CKroneckerMean.h"


namespace gpmix {

/*
 * CGPLMM:
 * testing based on GP class
 */
#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
//ignore C++ versions
%ignore CGPLMM::getNLL0;
%ignore CGPLMM::getNLLAlt;

//rename argout versions for python; this overwrites the C++ convenience functions
%rename(getNLL0) CGPLMM::agetNLL0;
%rename(getNLLAlt) CGPLMM::agetNLLAlt;
#endif
class CGPLMM : public ALMM
{

protected:
	//pointer to GP intance used for testing
	PGPkronecker gp;
	//pointer to optimization enginge
	PGPopt opt;
	void checkConsistency() throw (CGPMixException);
	void initTesting() throw (CGPMixException);

	//design matrices for foreground and background model
	MatrixXd AAlt;
	MatrixXd A0;
	//weight matrices (starting point of opt)
	MatrixXd weightsAlt;
	MatrixXd weights;
	//hyperparam objects for 0 and alt
	CGPHyperParams hpAlt;
	CGPHyperParams hp;
	//negative log likelihoods for foreground/background model
	MatrixXd nLL0, nLLAlt;

	sptr<CKroneckerMean> meanAlt;
	sptr<CKroneckerMean> mean;
public:
	CGPLMM(PGPkronecker gp) : gp(gp)
	{
	}
	virtual ~CGPLMM()
	{};
	//overload pure virtual functions:
	virtual void process() throw (CGPMixException);
    MatrixXd getA() const;
    void setA(MatrixXd a);
    MatrixXd getA0() const;
    void setA0(MatrixXd a0);
    void agetA0(MatrixXd* out) const;
    void agetA(MatrixXd* out) const;
    PGPkronecker getGp() const;
    void setGp(PGPkronecker gp);

    //getter and setter
	void agetNLL0(MatrixXd* out)
	{
		(*out) = nLL0;
	}
	void agetNLLAlt(MatrixXd *out)
	{
		(*out) = nLLAlt;
	}
	MatrixXd& getNLL0()
	{
		return nLL0;
	}
	MatrixXd& getNLLAlt()
	{
		return nLLAlt;
	}
};


}

#endif /* CGPLMM_H_ */
