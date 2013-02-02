/*
 * freeform.h
 *
 *  Created on: Jan 16, 2012
 *      Author: stegle
 */

#ifndef FREEFORM_H_
#define FREEFORM_H_

#include "covariance.h"

namespace limix {

#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
%ignore CFreeFormCF::getIparamDiag;
%ignore CFreeFormCF::K0Covar2Params;
%rename(getIparamDiag) CFreeFormCF::agetIparamDiag;
%rename(K0Covar2Params) CFreeFormCF::aK0Covar2Params;
#endif

class CFreeFormCF: public ACovarianceFunction {
protected:
	muint_t numberGroups;
	void agetL0grad_param(MatrixXd* out,muint_t i) const throw(CGPMixException);
	void agetK0grad_param(MatrixXd* out,muint_t i) const throw(CGPMixException);
	void projectKcross(MatrixXd* out,const MatrixXd& K0,const CovarInput& Xstar) const throw (CGPMixException);
	static muint_t calcNumberParams(muint_t numberGroups);
public:
	CFreeFormCF(muint_t numberGroups);
	virtual ~CFreeFormCF();

	void agetL0(MatrixXd* out) const;
	void agetK0(MatrixXd* out) const;


	void aKcross(MatrixXd* out, const CovarInput& Xstar ) const throw(CGPMixException);
	void aKcross_diag(VectorXd* out, const CovarInput& Xstar) const throw(CGPMixException);

	void aKgrad_param(MatrixXd* out,const muint_t i) const throw(CGPMixException);
	void aKcross_grad_X(MatrixXd* out,const CovarInput& Xstar, const muint_t d) const throw(CGPMixException);
	void aKdiag_grad_X(VectorXd* out,const muint_t d) const throw(CGPMixException);


	virtual void agetParamBounds(CovarParams* lower,CovarParams* upper) const;

	//class information
	inline std::string getName() const {return "CFreeform";};

	//helper function to convert from matrix to hyperparams
	static void aK0Covar2Params(VectorXd* out,const MatrixXd& K0,muint_t numberGroups);
	static VectorXd K0Covar2Params(const MatrixXd& K0,muint_t numberGroups);
	//set the a covariance matrix rather than parameters:
	void setParamsCovariance(const MatrixXd& K0);

	//information on parameter settings
	void agetIparamDiag(VectorXi* out) const;
	VectorXi getIparamDiag() const
	{
		VectorXi rv;
		agetIparamDiag(&rv);
		return rv;
	}

};
typedef sptr<CFreeFormCF> PFreeFormCF;


} /* namespace limix */
#endif /* FREEFORM_H_ */
