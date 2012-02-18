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


muint_t AMultiCF::Kdim() const throw(CGPMixException)
{
	return vecCovariances.begin()[0]->Kdim();
}



void AMultiCF::setCovariance(muint_t i, PCovarianceFunction covar) throw (CGPMixException)
								{
	vecCovariances[i] = covar;
								}

PCovarianceFunction AMultiCF::getCovariance(muint_t i) throw (CGPMixException)
								{
	return vecCovariances[i];
								}

void AMultiCF::addCovariance(PCovarianceFunction covar) throw (CGPMixException)
								{
	vecCovariances.push_back(covar);
								}

muint_t AMultiCF::getNumberDimensions() const throw (CGPMixException)
{
	muint_t rv=0;
	for(ACovarVec::const_iterator iter = vecCovariances.begin(); iter!=vecCovariances.end();iter++)
	{
		PCovarianceFunction cp = iter[0];
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
		PCovarianceFunction cp = iter[0];
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
		PCovarianceFunction cp = iter[0];
		if (cp!=NULL)
		{
			cols = cp->getNumberDimensions();
			cp->setX(X.block(0,c0,X.rows(),cols));
			//move pointer on
			c0+=cols;
		}
	}
}

void AMultiCF::setXcol(const CovarInput& X,muint_t col) throw (CGPMixException)
{
	if(((col+(muint_t)X.cols())>getNumberDimensions()) || ((muint_t)X.rows()!=this->Kdim()))
	{
		std::ostringstream os;
		os << "setXcol out of range. Current X:"<<col<< "..."<<col+X.cols()<<" own:"<<this->getNumberDimensions() <<")";
		throw CGPMixException(os.str());
	}
	if (X.cols()>1)
	{
		std::ostringstream os;
		os << "setXcol (Combinator CF) only suports setting individual columns" << "\n";
		throw CGPMixException(os.str());
	}
	muint_t c0=0;
	muint_t cols;
	//loop through covariances and assign
	for(ACovarVec::iterator iter = vecCovariances.begin(); iter!=vecCovariances.end();iter++)
	{
		PCovarianceFunction cp = iter[0];
		if (cp!=NULL)
		{
			cols = cp->getNumberDimensions();
			if ((c0+cols)>=col)
			{
				cp->setXcol(X,col-c0);
				break;
			}
			c0+=cols;
		}
	}
}


void AMultiCF::agetX(CovarInput* Xout) const throw (CGPMixException)
{
	//1. determine size of Xout
	muint_t trows = Kdim();
	muint_t tcols = getNumberDimensions();
	(*Xout).resize(trows,tcols);

	//2. loop through and fill
	muint_t c0=0;
	muint_t cols;
	for(ACovarVec::const_iterator iter = vecCovariances.begin(); iter!=vecCovariances.end();iter++)
	{
		PCovarianceFunction cp = iter[0];
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
			PCovarianceFunction cp = iter[0];
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
				PCovarianceFunction cp = iter[0];
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
		PCovarianceFunction cp = iter[0];
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
		PCovarianceFunction cp = iter[0];
		if (cp!=NULL)
		{
			nparams = cp->getNumberParams();
			(*out).segment(i0,nparams) = cp->getParams();
			i0+=nparams;
		}
	}
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



std::string CSumCF::getName() const
{
	return "SumCF";

}



void CSumCF::aK(MatrixXd* out) const
{
	muint_t trows = this->Kdim();
	(*out).setConstant(trows,trows,0);
	for(ACovarVec::const_iterator iter = vecCovariances.begin(); iter!=vecCovariances.end();iter++)
	{
		PCovarianceFunction cp = iter[0];
		if (cp!=NULL)
		{
			(*out) += cp->K();
		}
	}
}


void CSumCF::aKdiag(VectorXd* out) const
{
	muint_t trows = this->Kdim();
	(*out).setConstant(trows,0);
	for(ACovarVec::const_iterator iter = vecCovariances.begin(); iter!=vecCovariances.end();iter++)
	{
		PCovarianceFunction cp = iter[0];
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
	(*out).setConstant(Xstar.rows(),this->Kdim(),0);
	muint_t c0=0;
	muint_t cols;
	for(ACovarVec::const_iterator iter = vecCovariances.begin(); iter!=vecCovariances.end();iter++)
	{
		PCovarianceFunction cp = iter[0];
		if (cp!=NULL)
		{
			cols = cp->getNumberDimensions();
			MatrixXd Xstar_ = Xstar.block(0,c0,Xstar.rows(),cols);
			MatrixXd t = cp->Kcross(Xstar_);
			(*out) += cp->Kcross(Xstar_);
			//move on
			c0+=cols;
		}
	}
}

void CSumCF::aKcross_diag(VectorXd* out, const CovarInput& Xstar) const throw(CGPMixException)
	{
	//1. check that Xstar has consistent dimension
	if((muint_t)Xstar.cols()!=this->getNumberDimensions())
		throw CGPMixException("Kcross: col dimension of Xstar inconsistent!");
	//2. loop through covariances and add up
	(*out).setConstant(Xstar.rows(),0);
	muint_t c0=0;
	muint_t cols;
	for(ACovarVec::const_iterator iter = vecCovariances.begin(); iter!=vecCovariances.end();iter++)
	{
		PCovarianceFunction cp = iter[0];
		if (cp!=NULL)
		{
			//std::cout << cp->getName() << "\n";
			cols = cp->getNumberDimensions();
			MatrixXd Xstar_ = Xstar.block(0,c0,Xstar.rows(),cols);
			VectorXd t = cp->Kcross_diag(Xstar_);
			//std::cout << t.rows() << "," << Xstar.rows() << "\n";
			(*out) += t;
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
		PCovarianceFunction cp = iter[0];
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
		PCovarianceFunction cp = iter[0];
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
		PCovarianceFunction cp = iter[0];
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
		PCovarianceFunction cp = iter[0];
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





/*ProductCF*/

CProductCF::CProductCF(const ACovarVec& covariances) : AMultiCF(covariances)
{
};


CProductCF::CProductCF(const muint_t numCovariances) :AMultiCF(numCovariances)
{
}

CProductCF::~CProductCF()
{
}


std::string CProductCF::getName() const
{
	return "ProductCF";

}



void CProductCF::aK(MatrixXd* out) const
{
	muint_t trows = this->Kdim();
	(*out).setConstant(trows,trows,1);
	for(ACovarVec::const_iterator iter = vecCovariances.begin(); iter!=vecCovariances.end();iter++)
	{
		PCovarianceFunction cp = iter[0];
		if (cp!=NULL)
		{
			(*out).array() *= cp->K().array();
		}
	}
}


void CProductCF::aKdiag(VectorXd* out) const
{
	muint_t trows = this->Kdim();
	(*out).setConstant(trows,1.0);
	for(ACovarVec::const_iterator iter = vecCovariances.begin(); iter!=vecCovariances.end();iter++)
	{
		PCovarianceFunction cp = iter[0];
		if (cp!=NULL)
		{
			(*out).array() *= cp->Kdiag().array();
		}
	}
}


void CProductCF::aKcross(MatrixXd *out, const CovarInput & Xstar) const throw(CGPMixException)
{
	//1. check that Xstar has consistent dimension
	if((muint_t)Xstar.cols()!=this->getNumberDimensions())
		throw CGPMixException("Kcross: col dimension of Xstar inconsistent!");
	//2. loop through covariances and add up
	(*out).setConstant(Xstar.rows(),this->Kdim(),1.0);
	muint_t c0=0;
	muint_t cols;
	for(ACovarVec::const_iterator iter = vecCovariances.begin(); iter!=vecCovariances.end();iter++)
	{
		PCovarianceFunction cp = iter[0];
		if (cp!=NULL)
		{
			cols = cp->getNumberDimensions();
			MatrixXd Xstar_ = Xstar.block(0,c0,Xstar.rows(),cols);
			MatrixXd t = cp->Kcross(Xstar_);
			(*out).array() *= cp->Kcross(Xstar_).array();
			//move on
			c0+=cols;
		}
	}
}

void CProductCF::aKcross_diag(VectorXd* out, const CovarInput& Xstar) const throw(CGPMixException)
	{
	//1. check that Xstar has consistent dimension
	if((muint_t)Xstar.cols()!=this->getNumberDimensions())
		throw CGPMixException("Kcross: col dimension of Xstar inconsistent!");
	//2. loop through covariances and add up
	(*out).setConstant(Xstar.rows(),1.0);
	muint_t c0=0;
	muint_t cols;
	for(ACovarVec::const_iterator iter = vecCovariances.begin(); iter!=vecCovariances.end();iter++)
	{
		PCovarianceFunction cp = iter[0];
		if (cp!=NULL)
		{
			cols = cp->getNumberDimensions();
			MatrixXd Xstar_ = Xstar.block(0,c0,Xstar.rows(),cols);
			VectorXd t = cp->Kcross_diag(Xstar_);
			(*out).array() *= t.array();
			//move on
			c0+=cols;
		}
	}

	}


void CProductCF::aKgrad_param(MatrixXd *out, const muint_t i) const throw(CGPMixException)
{
	//1. check that i is within the available range
	this->checkWithinParams(i);
	//2. loop through covariances until we found the correct one
	muint_t i0=0;
	muint_t params;
	(*out).setConstant(this->Kdim(),this->Kdim(),1.0);
	for(ACovarVec::const_iterator iter = vecCovariances.begin(); iter!=vecCovariances.end();iter++)
	{
		PCovarianceFunction cp = iter[0];
		if (cp!=NULL)
		{
			params = cp->getNumberParams();

			//is the parameter in that covariance function?
			if((i-i0)<params)
				(*out).array()*=cp->Kgrad_param(i-i0).array();
			else
				(*out).array()*=cp->K().array();
			//move on
			i0+=params;
		}
	}
}

void CProductCF::aKgrad_X(MatrixXd* out,const muint_t d) const throw(CGPMixException)
{
	//1. check that d is in range
	this->checkWithinDimensions(d);
	//2. loop over covarainces and get the one corresponding to d
	muint_t c0=0;
	muint_t cols;
	(*out).setConstant(this->Kdim(),this->Kdim(),1.0);
	for(ACovarVec::const_iterator iter = vecCovariances.begin(); iter!=vecCovariances.end();iter++)
	{
		PCovarianceFunction cp = iter[0];
		if (cp!=NULL)
		{
			cols = cp->getNumberDimensions();
			if ((d-c0)<cols)
				(*out).array()*=cp->Kgrad_X(d-c0).array();
			else
				(*out).array()*=cp->K().array();
			//move on
			c0+=cols;
		}
	}
}

void CProductCF::aKdiag_grad_X(VectorXd *out, const muint_t d) const throw(CGPMixException)
{
	//1. check that d is in range
	this->checkWithinDimensions(d);
	muint_t c0=0;
	muint_t cols;
	(*out).setConstant(this->Kdim(),1.0);
	for(ACovarVec::const_iterator iter = vecCovariances.begin(); iter!=vecCovariances.end();iter++)
	{
		PCovarianceFunction cp = iter[0];
		if (cp!=NULL)
		{
			cols = cp->getNumberDimensions();
			if ((d-c0)<cols)
				(*out).array()*=cp->Kdiag_grad_X(d-c0).array();
			else
				(*out).array()*=cp->Kdiag().array();
			//move on
			c0+=cols;
		}
	}
}


void CProductCF::aKcross_grad_X(MatrixXd *out, const CovarInput & Xstar, const muint_t d) const throw(CGPMixException)
{
	//1. check that d is in range
	this->checkWithinDimensions(d);

	//2. loop over covarainces and get the one corresponding to d
	muint_t c0=0;
	muint_t cols;
	(*out).setConstant(Xstar.rows(),this->Kdim(),1.0);
	for(ACovarVec::const_iterator iter = vecCovariances.begin(); iter!=vecCovariances.end();iter++)
	{
		PCovarianceFunction cp = iter[0];
		if (cp!=NULL)
		{
			//get Xstar
			cols = cp->getNumberDimensions();
			MatrixXd Xstar_ = Xstar.block(0,c0,Xstar.rows(),cols);
			if ((d-c0)<cols)
			{
				//calc
				(*out).array()*=cp->Kcross_grad_X(Xstar_,d-c0).array();
			}
			else
				(*out).array()*=cp->Kcross(Xstar_).array();
			//move on
			c0+=cols;
		}
	}
}



}
