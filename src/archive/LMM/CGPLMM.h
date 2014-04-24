// Copyright(c) 2014, The LIMIX developers (Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
// All rights reserved.
//
// LIMIX is provided under a 2-clause BSD license.
// See license.txt for the complete license.

#ifndef CGPLMM_H_
#define CGPLMM_H_

#include "limix/LMM/lmm.h"
#include "limix/gp/gp_kronecker.h"
#include "limix/gp/gp_opt.h"
#include "limix/mean/CKroneckerMean.h"
#include "limix/mean/CSumLinear.h"



namespace limix {
/*
 * CGPLMM:
 * testing based on GP class
 */

#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
//ignore C++ versions
%ignore CGPLMM::getNLL0;
%ignore CGPLMM::getNLLAlt;
%ignore CGPLMM::getLdeltaAlt;
%ignore CGPLMM::getLdelta0;
%ignore CGPLMM::getAAlt;
%ignore CGPLMM::getA0;

//rename argout versions for python; this overwrites the C++ convenience functions
%rename(getNLL0) CGPLMM::agetNLL0;
%rename(getNLLAlt) CGPLMM::agetNLLAlt;
%rename(getLdeltaAlt) CGPLMM::agetLdeltaAlt;
%rename(getLdelta0) CGPLMM::agetLdelta0;
%rename(getAAlt) CGPLMM::agetAAlt;
%rename(getA0) CGPLMM::agetA0;
#endif

class CGPLMM : public ALMM
{

protected:
	//pointer to GP instance used for testing
	PGPkronecker gp;
	//pointer to optimization engine
	PGPopt opt;
	void checkConsistency() ;
	void initTesting() ;

	//design matrices for foreground and background model
	MatrixXd AAlt;
	MatrixXdVec VA0;
	//weight matrices (starting point of opt)
	MatrixXd weightsAlt;
	MatrixXd weights0;
	//hyperparam objects for 0 and alt
	CGPHyperParams hpAlt;
	CGPHyperParams hp0;
	//starting parameters for optimization in general
	CGPHyperParams params0;
	//filter parameters
	CGPHyperParams paramsMask;
	//negative log likelihoods for foreground/background model
	MatrixXd nLL0, nLLAlt,ldeltaAlt,ldelta0;
	PSumLinear meanAlt;
	PSumLinear mean0;
	//data term for alt model (to change fixed effect)
	PKroneckerMean altTerm;
	muint_t degreesFreedom;
public:
	CGPLMM(PGPkronecker gp);
	CGPLMM()
	{};
	virtual ~CGPLMM()
	{};
	//overload pure virtual functions:
	virtual void process() ;



	//set get AAlt
	void setAAlt(const MatrixXd& AAlt)
	{
		this->AAlt = AAlt;
	}
	MatrixXd getAAlt()
	{
		return AAlt;
	}
	void agetAAlt(MatrixXd* out)
	{
		(*out) = AAlt;
	}

	//VA0 and elements
	void setVA0(const MatrixXdVec& VA0)
	{
		this->VA0 = VA0;
	}
	MatrixXdVec getVA0()
	{
		return VA0;
	}
    void addA0(const MatrixXd& a)
    {
    	VA0.push_back(a);
    }
    void setA0(const MatrixXd& a0,muint_t i)
    {
        VA0[i] = a0;
    }
    MatrixXd getA0(muint_t i) const
    {
    	return VA0[i];
    }
    void agetA0(MatrixXd* out,muint_t i) const
    {
    	(*out) = VA0[i];
    }
    muint_t getDegreesFredom() { return degreesFreedom;}

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

	CGPHyperParams getParams0()
	{
		return params0;
	}
	void setParams0(const CGPHyperParams& params0)
	{
		this->params0 = params0;
	}

	CGPHyperParams getParamsMask()
	{
		return paramsMask;
	}
	void setParamsMask(const CGPHyperParams& p)
	{
		paramsMask = p;
	}

	void agetLdeltaAlt(MatrixXd *out)
	{
		(*out) = ldeltaAlt;
	}
	void agetLdelta0(MatrixXd *out)
	{
		(*out) = ldelta0;
	}

	MatrixXd getLdeltaAlt()
	{
		return ldeltaAlt;
	}
	MatrixXd getLdelta0()
	{
		return ldeltaAlt;
	}
};


}

#endif /* CGPLMM_H_ */
