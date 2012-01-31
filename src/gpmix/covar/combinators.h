/*
 * combinators.h
 *
 *  Created on: Dec 28, 2011
 *      Author: stegle
 */

#ifndef COMBINATORS_H_
#define COMBINATORS_H_

#include <gpmix/covar/covariance.h>
#include <vector>

namespace gpmix {
using namespace std;

//Define a vector of covariances which is needed to represent the sum and product CF
typedef vector<ACovarianceFunction*> ACovarVec;
//SWIG template declaration:
#if (!defined(SWIG_FILE_WITH_INIT) && defined(SWIG))
%template(ACovarVec) vector<ACovarianceFunction* >;
#endif





class AMultiCF : public ACovarianceFunction
{
protected:
	ACovarVec vecCovariances;
public:
	AMultiCF(const ACovarVec& covariances);
	AMultiCF(const muint_t numCovariances=0);
	//destructors
	virtual ~AMultiCF();

	virtual bool isInSync() const;
	virtual void makeSync();

	virtual muint_t Kdim() const throw(CGPMixException);

	//access to covariance arrays
	void addCovariance(ACovarianceFunction* covar) throw (CGPMixException);
	void setCovariance(muint_t i,ACovarianceFunction* covar) throw (CGPMixException);
	ACovarianceFunction* getCovariance(muint_t i) throw (CGPMixException);

	virtual muint_t getNumberDimensions() const throw (CGPMixException);
	virtual void setNumberDimensions(muint_t numberDimensions) throw (CGPMixException);
	virtual muint_t getNumberParams() const;

	//setX and getX
	virtual void setX(const CovarInput& X) throw (CGPMixException);
	virtual void agetX(CovarInput* Xout) const throw (CGPMixException);
	virtual void setXcol(const CovarInput& X,muint_t col) throw (CGPMixException);

	//set and get Params
	virtual void setParams(const CovarParams& params);
	virtual void agetParams(CovarParams* out);
};


class CSumCF : public AMultiCF {
public:
	CSumCF(const ACovarVec& covariances);
	CSumCF(const muint_t numCovariances=0);
	//destructors
	virtual ~CSumCF();

	//overloaded pure virtual members
	virtual void aKcross(MatrixXd* out, const CovarInput& Xstar ) const throw(CGPMixException);
	virtual void aKcross_diag(VectorXd* out, const CovarInput& Xstar) const throw(CGPMixException);
	virtual void aKgrad_param(MatrixXd* out,const muint_t i) const throw(CGPMixException);
	virtual void aKcross_grad_X(MatrixXd* out,const CovarInput& Xstar, const muint_t d) const throw(CGPMixException);
	virtual void aKdiag_grad_X(VectorXd* out,const muint_t d) const throw(CGPMixException);
	//optional overloadings:
	virtual void aK(MatrixXd* out) const;
	virtual void aKdiag(VectorXd* out) const;
	virtual void aKgrad_X(MatrixXd* out,const muint_t d) const throw(CGPMixException);
	virtual string getName() const;
};



class CProductCF : public AMultiCF {
public:
	CProductCF(const ACovarVec& covariances);
	CProductCF(const muint_t numCovariances=0);
	//destructors
	virtual ~CProductCF();

	//overloaded pure virtual members
	virtual void aKcross(MatrixXd* out, const CovarInput& Xstar ) const throw(CGPMixException);
	virtual void aKcross_diag(VectorXd* out, const CovarInput& Xstar) const throw(CGPMixException);
	virtual void aKgrad_param(MatrixXd* out,const muint_t i) const throw(CGPMixException);
	virtual void aKcross_grad_X(MatrixXd* out,const CovarInput& Xstar, const muint_t d) const throw(CGPMixException);
	virtual void aKdiag_grad_X(VectorXd* out,const muint_t d) const throw(CGPMixException);
	//optional overloadings:
	virtual void aK(MatrixXd* out) const;
	virtual void aKdiag(VectorXd* out) const;
	virtual void aKgrad_X(MatrixXd* out,const muint_t d) const throw(CGPMixException);
	virtual string getName() const;
};


}




#endif /* COMBINATORS_H_ */
