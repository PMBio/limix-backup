/*
 * likelihood.h
 *
 *  Created on: Nov 11, 2011
 *      Author: clippert
 */

#ifndef LIKELIHOOD_H_
#define LIKELIHOOD_H_

#include <gpmix/covar/covariance.h>



namespace gpmix {


typedef VectorXd LikParams;

//For now, likelihood is a special case of covariance function
class ALikelihood : public ACovarianceFunction {
	//indicator if the class is synced with the cache
protected:
public:
	ALikelihood(const muint_t numberParams=1);
	virtual ~ALikelihood();

	//pure virtual functions we don't really need...
	virtual void aKcross(MatrixXd* out, const CovarInput& Xstar ) const throw(CGPMixException);
	void aKcross_diag(VectorXd* out, const CovarInput& Xstar) const throw(CGPMixException);
	virtual void aKcross_grad_X(MatrixXd* out,const CovarInput& Xstar, const muint_t d) const throw(CGPMixException);
	virtual void aKdiag_grad_X(VectorXd* out,const muint_t d) const throw(CGPMixException);
};

/* Null likelihood model: assuming all variation is explained in covar*/
class CLikNormalNULL : public ALikelihood {
protected:
	muint_t numRows;
public:
	CLikNormalNULL();
	~CLikNormalNULL();

	//pure virtual functions that need to be overwritten
	virtual void aK(MatrixXd* out) const throw (CGPMixException);
	virtual void aKdiag(VectorXd* out) const throw (CGPMixException);
	virtual void aKcross_diag(VectorXd* out, const CovarInput& Xstar) const throw(CGPMixException);
	virtual void aKgrad_param(MatrixXd* out, const muint_t row) const throw (CGPMixException);
	//overwrite setX. We merely ignore the number of columns here:
	virtual void setX(const CovarInput& X) throw (CGPMixException);

	string getName() const {return "LikNormalIso";};
};


/* Gaussian likelihood model*/
class CLikNormalIso : public ALikelihood {
protected:
	muint_t numRows;
public:
	CLikNormalIso();
	~CLikNormalIso();

	//pure virtual functions that need to be overwritten
	virtual void aK(MatrixXd* out) const throw (CGPMixException);
	virtual void aKdiag(VectorXd* out) const throw (CGPMixException);
	virtual void aKcross_diag(VectorXd* out, const CovarInput& Xstar) const throw(CGPMixException);
	virtual void aKgrad_param(MatrixXd* out, const muint_t row) const throw (CGPMixException);
	//overwrite setX. We merely ignore the number of columns here:
	virtual void setX(const CovarInput& X) throw (CGPMixException);

	string getName() const {return "LikNormalIso";};
};



} /* namespace gpmix */
#endif /* LIKELIHOOD_H_ */
