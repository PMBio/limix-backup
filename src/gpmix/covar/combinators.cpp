/*
 * combinators.cpp
 *
 *  Created on: Dec 28, 2011
 *      Author: stegle
 */


#include "combinators.h"
#include "gpmix/types.h"

namespace gpmix {


AMultiCF::AMultiCF(const ACovarVec& covariances)
{
	vecCovariances = covariances;
}

AMultiCF::AMultiCF(const muint_t numCovariances) : vecCovariances(numCovariances)
{
}

AMultiCF::~AMultiCF()
{
}

CSumCF::CSumCF(const ACovarVec& covariances) : AMultiCF(covariances)
{
};


CSumCF::CSumCF(const muint_t numCovariances) :AMultiCF(numCovariances)
{
}

CSumCF::~CSumCF()
{
}

muint_t AMultiCF::getXRows() const
{
	return vecCovariances.begin()[0]->getX().rows();
}



void AMultiCF::setCovariance(muint_t i, ACovarianceFunction *covar) throw (CGPMixException)
								{
	vecCovariances[i] = covar;
								}

ACovarianceFunction *AMultiCF::getCovariance(muint_t i) throw (CGPMixException)
								{
	return vecCovariances[i];
								}

void AMultiCF::addCovariance(ACovarianceFunction* covar) throw (CGPMixException)
								{
	vecCovariances.push_back(covar);
								}

muint_t AMultiCF::getNumberDimensions() const throw (CGPMixException)
{
	muint_t rv=0;
	for(ACovarVec::const_iterator iter = vecCovariances.begin(); iter!=vecCovariances.end();iter++)
	{
		ACovarianceFunction* cp = iter[0];
		if (cp!=NULL)
			rv+= cp->getNumberDimensions();
	}
	//loop through covariances and add up dimensionality;
	return rv;
}

muint_t AMultiCF::getNumberParams() const
{
	muint_t rv=0;
	for(ACovarVec::const_iterator iter = vecCovariances.begin(); iter!=vecCovariances.end();iter++)
	{
		ACovarianceFunction* cp = iter[0];
		if (cp!=NULL)
			rv+= cp->getNumberParams();
	}
	//loop through covariances and add up dimensionality;
	return rv;
}

void AMultiCF::setNumberDimensions(muint_t numberDimensions) throw (CGPMixException)
		{
	throw CGPMixException("Multiple covariance functions do not support setting X dimensions. Set dimensions of member covariance functions instead.");
		}


void AMultiCF::setX(const CovarInput& X) throw (CGPMixException)
{
	checkXDimensions(X);
	//current column index in X:
	muint_t c0=0;
	muint_t cols;
	//loop through covariances and assign
	for(ACovarVec::iterator iter = vecCovariances.begin(); iter!=vecCovariances.end();iter++)
	{
		ACovarianceFunction* cp = iter[0];
		if (cp!=NULL)
		{
			cols = cp->getNumberDimensions();
			cp->setX(X.block(0,c0,X.rows(),cols));
			//move pointer on
			c0+=cols;
		}
	}
}

void AMultiCF::agetX(CovarInput* Xout) const throw (CGPMixException)
{
	//1. determine size of Xout
	muint_t trows = getXRows();
	muint_t tcols = getNumberDimensions();
	(*Xout).resize(trows,tcols);

	//2. loop through and fill
	muint_t c0=0;
	muint_t cols;
	for(ACovarVec::const_iterator iter = vecCovariances.begin(); iter!=vecCovariances.end();iter++)
	{
		ACovarianceFunction* cp = iter[0];
		if (cp!=NULL)
		{
			cols = cp->getNumberDimensions();
			(*Xout).block(0,c0,trows,cols) = cp->getX();
			//move pointer on
			c0+=cols;
		}
	}
}

bool AMultiCF::isInSync() const
{
	//if at least one covariance is not in sync, return false
	for(ACovarVec::const_iterator iter = vecCovariances.begin(); iter!=vecCovariances.end();iter++)
		{
			ACovarianceFunction* cp = iter[0];
			if (cp!=NULL)
			{
				if (!cp->isInSync())
						return false;
			}
		}
	return true;
}

void AMultiCF::makeSync()
{
	for(ACovarVec::const_iterator iter = vecCovariances.begin(); iter!=vecCovariances.end();iter++)
			{
				ACovarianceFunction* cp = iter[0];
				if (cp!=NULL)
				{
					cp->makeSync();
				}
			}
}

void AMultiCF::setParams(const CovarParams& params)
{
	//1. check dimensionality
	checkParamDimensions(params);
	//2. loop through covariances
	muint_t i0=0;
	muint_t nparams;
	//loop through covariances and assign
	for(ACovarVec::iterator iter = vecCovariances.begin(); iter!=vecCovariances.end();iter++)
	{
		ACovarianceFunction* cp = iter[0];
		if (cp!=NULL)
		{
			nparams = cp->getNumberParams();
			cp->setParams(params.segment(i0,nparams));
			i0+=nparams;
		}
	}
}

void AMultiCF::agetParams(CovarParams* out)
{
	//1. reserve memory
	(*out).resize(getNumberParams());
	//2. loop through covariances
	muint_t i0=0;
	muint_t nparams;
	//loop through covariances and assign
	for(ACovarVec::iterator iter = vecCovariances.begin(); iter!=vecCovariances.end();iter++)
	{
		ACovarianceFunction* cp = iter[0];
		if (cp!=NULL)
		{
			nparams = cp->getNumberParams();
			(*out).segment(i0,nparams) = cp->getParams();
			i0+=nparams;
		}
	}
}



string CSumCF::getName() const
{
	return "SumCF";

}



void CSumCF::aK(MatrixXd* out) const
{
	muint_t trows = this->getXRows();
	(*out).setConstant(trows,trows,0);
	for(ACovarVec::const_iterator iter = vecCovariances.begin(); iter!=vecCovariances.end();iter++)
	{
		ACovarianceFunction* cp = iter[0];
		if (cp!=NULL)
		{
			(*out) += cp->K();
		}
	}
}


void CSumCF::aKdiag(VectorXd* out) const
{
	muint_t trows = this->getXRows();
	(*out).setConstant(trows,0);
	for(ACovarVec::const_iterator iter = vecCovariances.begin(); iter!=vecCovariances.end();iter++)
	{
		ACovarianceFunction* cp = iter[0];
		if (cp!=NULL)
		{
			(*out) += cp->Kdiag();
		}
	}
}


void CSumCF::aKcross(MatrixXd *out, const CovarInput & Xstar) const throw(CGPMixException)
				{
	//1. check that Xstar has consistent dimension
	if((muint_t)Xstar.cols()!=this->getNumberDimensions())
		throw CGPMixException("Kcross: col dimension of Xstar inconsistent!");
	//2. loop through covariances and add up
	(*out).setConstant(Xstar.rows(),this->getXRows(),0);
	muint_t c0=0;
	muint_t cols;
	for(ACovarVec::const_iterator iter = vecCovariances.begin(); iter!=vecCovariances.end();iter++)
	{
		ACovarianceFunction* cp = iter[0];
		if (cp!=NULL)
		{
			cols = cp->getNumberDimensions();
			MatrixXd Xstar_ = Xstar.block(0,c0,Xstar.rows(),cols);
			(*out) += cp->Kcross(Xstar_);
			//move on
			c0+=cols;
		}
	}
}

void CSumCF::aKgrad_param(MatrixXd *out, const muint_t i) const throw(CGPMixException)
						{
	//1. check that i is within the available range
	this->checkWithinParams(i);
	//2. loop through covariances until we found the correct one
	muint_t i0=0;
	muint_t params;
	for(ACovarVec::const_iterator iter = vecCovariances.begin(); iter!=vecCovariances.end();iter++)
	{
		ACovarianceFunction* cp = iter[0];
		if (cp!=NULL)
		{
			params = cp->getNumberParams();
			//is the parameter in that covariance function?
			if((i-i0)<params)
			{
				cp->aKgrad_param(out,i-i0);
				break;
			}
			//move on
			i0+=params;
		}
	}
}

void CSumCF::aKgrad_X(MatrixXd* out,const muint_t d) const throw(CGPMixException)
{
	//1. check that d is in range
	this->checkWithinDimensions(d);
	//2. loop over covarainces and get the one corresponding to d
	muint_t c0=0;
	muint_t cols;
	for(ACovarVec::const_iterator iter = vecCovariances.begin(); iter!=vecCovariances.end();iter++)
	{
		ACovarianceFunction* cp = iter[0];
		if (cp!=NULL)
		{
			cols = cp->getNumberDimensions();
			if ((d-c0)<cols)
			{
				cp->aKgrad_X(out,d-c0);
				break;
			}
			//move on
			c0+=cols;
		}
	}
}

void CSumCF::aKdiag_grad_X(VectorXd *out, const muint_t d) const throw(CGPMixException)
{
	//1. check that d is in range
	this->checkWithinDimensions(d);
	muint_t c0=0;
	muint_t cols;
	for(ACovarVec::const_iterator iter = vecCovariances.begin(); iter!=vecCovariances.end();iter++)
	{
		ACovarianceFunction* cp = iter[0];
		if (cp!=NULL)
		{
			cols = cp->getNumberDimensions();
			if ((d-c0)<cols)
			{
				cp->aKdiag_grad_X(out,d-c0);
				break;
			}
			//move on
			c0+=cols;
		}
	}
}


void CSumCF::aKcross_grad_X(MatrixXd *out, const CovarInput & Xstar, const muint_t d) const throw(CGPMixException)
{
	//1. check that d is in range
	this->checkWithinDimensions(d);

	//2. loop over covarainces and get the one corresponding to d
	muint_t c0=0;
	muint_t cols;
	for(ACovarVec::const_iterator iter = vecCovariances.begin(); iter!=vecCovariances.end();iter++)
	{
		ACovarianceFunction* cp = iter[0];
		if (cp!=NULL)
		{
			cols = cp->getNumberDimensions();
			if ((d-c0)<cols)
			{
				//get Xstar
				MatrixXd Xstar_ = Xstar.block(0,c0,Xstar.rows(),cols);
				//calc
				cp->aKcross_grad_X(out,Xstar_,d-c0);
				break;
			}
			//move on
			c0+=cols;
		}
	}
}




}
