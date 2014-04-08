// Copyright(c) 2014, The LIMIX developers (Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
// All rights reserved.
//
// LIMIX is provided under a 2-clause BSD license.
// See license.txt for the complete license.

#ifndef LIKELIHOOD_H_
#define LIKELIHOOD_H_

#include "limix/covar/covariance.h"

namespace limix {


typedef VectorXd LikParams;


#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
//%sptr(gpmix::ALikelihood)
#endif

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
typedef sptr<ALikelihood> PLikelihood;



#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
//%sptr(gpmix::CLikNormalNULL)
#endif

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
    virtual void aKhess_param(MatrixXd* out, const muint_t i, const muint_t j) const throw(CGPMixException);
	//overwrite setX. We merely ignore the number of columns here:
	virtual void setX(const CovarInput& X) throw (CGPMixException);

	std::string getName() const {return "LikNormalIso";};
};
typedef sptr<CLikNormalNULL> PLikNormalNULL;


/* Gaussian likelihood model*/
#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
//%sptr(gpmix::CLikNormalIso)
#endif
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
    virtual void aKhess_param(MatrixXd* out, const muint_t i, const muint_t j) const throw(CGPMixException);
	//overwrite setX. We merely ignore the number of columns here:
	virtual void setX(const CovarInput& X) throw (CGPMixException);

	std::string getName() const {return "LikNormalIso";};
};
typedef sptr<CLikNormalIso> PLikNormalIso;

/* Likelihood model for SVD Covariances */
class CLikNormalSVD : public ALikelihood {
protected:
	muint_t numRows;
public:
	CLikNormalSVD();
	~CLikNormalSVD();

	//pure virtual functions that need to be overwritten
	virtual void aK(MatrixXd* out) const throw (CGPMixException);
	virtual void aKdiag(VectorXd* out) const throw (CGPMixException);
	virtual void aKcross_diag(VectorXd* out, const CovarInput& Xstar) const throw(CGPMixException);
	virtual void aKgrad_param(MatrixXd* out, const muint_t row) const throw (CGPMixException);
    virtual void aKhess_param(MatrixXd* out, const muint_t i, const muint_t j) const throw(CGPMixException);
	//overwrite setX. We merely ignore the number of columns here:
	virtual void setX(const CovarInput& X) throw (CGPMixException);

	virtual mfloat_t getSigmaK2();
	virtual mfloat_t getDelta();
	virtual mfloat_t getSigmaK2grad();
	virtual mfloat_t getDeltagrad();

	std::string getName() const {return "ClikNormalSVD";};
};
typedef sptr<CLikNormalSVD> PLikNormalSVD;




} /* namespace limix */
#endif /* LIKELIHOOD_H_ */
