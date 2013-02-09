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
%rename(getK0) CFreeFormCF::agetK0;
#endif

enum CFreeFromCFConstraitType {freeform,diagonal,dense};

 class CFreeFormCF: public ACovarianceFunction {
protected:
	muint_t numberGroups;
	void agetK0grad_param(MatrixXd* out,muint_t i) const throw(CGPMixException);
	void projectKcross(MatrixXd* out,const MatrixXd& K0,const CovarInput& Xstar) const throw (CGPMixException);
	static muint_t calcNumberParams(muint_t numberGroups);

	CFreeFromCFConstraitType constraint;
	//helper function to convert from matrix to hyperparams

	void aK0Covar2Params(VectorXd* out,const MatrixXd& K0);
	virtual void agetL0grad_param(MatrixXd* out,muint_t i) const throw(CGPMixException);
	virtual void agetL0grad_param_dense(MatrixXd* out,muint_t i) const throw(CGPMixException);

public:

	CFreeFormCF(muint_t numberGroups,CFreeFromCFConstraitType constraint=freeform);
	virtual ~CFreeFormCF();

	virtual void agetL0(MatrixXd* out) const;
	virtual void agetL0_dense(MatrixXd* out) const;
	virtual void agetK0(MatrixXd* out) const;


	void aKcross(MatrixXd* out, const CovarInput& Xstar ) const throw(CGPMixException);
	void aKcross_diag(VectorXd* out, const CovarInput& Xstar) const throw(CGPMixException);

	void aKgrad_param(MatrixXd* out,const muint_t i) const throw(CGPMixException);
    void aKhess_param(MatrixXd* out, const muint_t i, const muint_t j) const throw(CGPMixException);
	void aKcross_grad_X(MatrixXd* out,const CovarInput& Xstar, const muint_t d) const throw(CGPMixException);
	void aKdiag_grad_X(VectorXd* out,const muint_t d) const throw(CGPMixException);

	virtual void agetParamBounds(CovarParams* lower,CovarParams* upper) const;
	virtual void agetParamMask0(CovarParams* out) const;

	//class information
	inline std::string getName() const {return "CFreeform";};

	//set the a covariance matrix rather than parameters:
	virtual void setParamsCovariance(const MatrixXd& K0) throw(CGPMixException);

	//information on parameter settings
	void agetIparamDiag(VectorXi* out) const;
	VectorXi getIparamDiag() const
	{
		VectorXi rv;
		agetIparamDiag(&rv);
		return rv;
	}

	CFreeFromCFConstraitType getConstraint() const {
		return constraint;
	}

	void setConstraint(CFreeFromCFConstraitType constraint) {
		this->constraint = constraint;
	}
};
typedef sptr<CFreeFormCF> PFreeFormCF;


} /* namespace limix */
#endif /* FREEFORM_H_ */
