// Copyright(c) 2014, The LIMIX developers(Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
//
//Licensed under the Apache License, Version 2.0 (the "License");
//you may not use this file except in compliance with the License.
//You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//Unless required by applicable law or agreed to in writing, software
//distributed under the License is distributed on an "AS IS" BASIS,
//WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//See the License for the specific language governing permissions and
//limitations under the License.

#include "combinators.h"
#include "limix/types.h"
#include "limix/utils/matrix_helper.h"

namespace limix {


AMultiCF::AMultiCF(const ACovarVec& covariances,muint_t numMaxCovariances)
{
	vecCovariances = covariances;
	this->numMaxCovariances = numMaxCovariances;
}

AMultiCF::AMultiCF(muint_t numCovariancesInit,muint_t numMaxCovariances) : vecCovariances(numCovariancesInit)
{
	this->numMaxCovariances = numMaxCovariances;
}


AMultiCF::~AMultiCF()
{
}


muint_t AMultiCF::Kdim() const 
{
	return vecCovariances.begin()[0]->Kdim();
}

void AMultiCF::setCovariance(muint_t i, PCovarianceFunction covar) 
{
	//check this is witin ther permitted range:
	if(i>numMaxCovariances)
	{
		throw CLimixException("AMultiCF: number of covariances limited.");
	}
	vecCovariances[i] = covar;
}

PCovarianceFunction AMultiCF::getCovariance(muint_t i) 
{
	return vecCovariances[i];
}

void AMultiCF::addCovariance(PCovarianceFunction covar) 
{
	if(vecCovariances.size()+1>numMaxCovariances)
	{
		throw CLimixException("AMultiCF: number of covariances limited.");
	}
	vecCovariances.push_back(covar);
}

muint_t AMultiCF::getNumberDimensions() const 
{
	muint_t rv=0;
	for(ACovarVec::const_iterator iter = vecCovariances.begin(); iter!=vecCovariances.end();iter++)
	{
		PCovarianceFunction cp = iter[0];
		if (cp!=nullptr)
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
		if (cp!=nullptr)
			rv+= cp->getNumberParams();
	}
	//loop through covariances and add up dimensionality;
	return rv;
}

void AMultiCF::setNumberDimensions(muint_t numberDimensions) 
		{
	throw CLimixException("Multiple covariance functions do not support setting X dimensions. Set dimensions of member covariance functions instead.");
		}


void AMultiCF::setX(const CovarInput& X) 
{
	checkXDimensions(X);
	//current column index in X:
	muint_t c0=0;
	muint_t cols;
	//loop through covariances and assign
	for(ACovarVec::iterator iter = vecCovariances.begin(); iter!=vecCovariances.end();iter++)
	{
		PCovarianceFunction cp = iter[0];
		if (cp!=nullptr)
		{
			cols = cp->getNumberDimensions();
			cp->setX(X.block(0,c0,X.rows(),cols));
			//move pointer on
			c0+=cols;
		}
	}
}

void AMultiCF::setXcol(const CovarInput& X,muint_t col) 
{
	if(((col+(muint_t)X.cols())>getNumberDimensions()) || ((muint_t)X.rows()!=this->Kdim()))
	{
		std::ostringstream os;
		os << "setXcol out of range. Current X:"<<col<< "..."<<col+X.cols()<<" own:"<<this->getNumberDimensions() <<")";
		throw CLimixException(os.str());
	}
	if (X.cols()>1)
	{
		std::ostringstream os;
		os << "setXcol (Combinator CF) only suports setting individual columns" << "\n";
		throw CLimixException(os.str());
	}
	muint_t c0=0;
	muint_t cols;
	//loop through covariances and assign
	for(ACovarVec::iterator iter = vecCovariances.begin(); iter!=vecCovariances.end();iter++)
	{
		PCovarianceFunction cp = iter[0];
		if (cp!=nullptr)
		{
			cols = cp->getNumberDimensions();
			//skip covaraince with 0 dimensions
			if (cols==0)
				continue;
			if ((c0+cols)>=col)
			{
				cp->setXcol(X,col-c0);
				break;
			}
			c0+=cols;
		}
	}
}

void AMultiCF::agetX(CovarInput* Xout) const 
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
		if (cp!=nullptr)
		{
			cols = cp->getNumberDimensions();
			//ignore covariances with 0 columns
			if (cols==0)
				continue;
			(*Xout).block(0,c0,trows,cols) = cp->getX();
			//move pointer on
			c0+=cols;
		}
	}
}

//synch child pointers are added to all covariance
//if a single one of the covara changes, an update is triggered
void AMultiCF::addSyncChild(Pbool l)
{
	//if at least one covariance is not in sync, return false
	for(ACovarVec::const_iterator iter = vecCovariances.begin(); iter!=vecCovariances.end();iter++)
		{
			PCovarianceFunction cp = iter[0];
			if (cp!=nullptr)
			{
				cp->addSyncChild(l);
			}
		}
}
void AMultiCF::delSyncChild(Pbool l)
{
	//if at least one covariance is not in sync, return false
	for(ACovarVec::const_iterator iter = vecCovariances.begin(); iter!=vecCovariances.end();iter++)
		{
			PCovarianceFunction cp = iter[0];
			if (cp!=nullptr)
			{
				cp->delSyncChild(l);
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
		if (cp!=nullptr)
		{
			nparams = cp->getNumberParams();
			cp->setParams(params.segment(i0,nparams));
			i0+=nparams;
		}
	}
}

void AMultiCF::agetParams(CovarParams* out) const
{
	//1. reserve memory
	(*out).resize(getNumberParams());
	//2. loop through covariances
	muint_t i0=0;
	muint_t nparams;
	//loop through covariances and assign
	for(ACovarVec::const_iterator iter = vecCovariances.begin(); iter!=vecCovariances.end();iter++)
	{
		PCovarianceFunction cp = iter[0];
		if (cp!=nullptr)
		{
			nparams = cp->getNumberParams();
			(*out).segment(i0,nparams) = cp->getParams();
			i0+=nparams;
		}
	}
}

void AMultiCF::setParamMask(const CovarParams& params)
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
		if (cp!=nullptr)
		{
			nparams = cp->getNumberParams();
			cp->setParamMask(params.segment(i0,nparams));
			i0+=nparams;
		}
	}
}


void AMultiCF::agetParamMask(CovarParams* out) const
{
	//1. reserve memory
	(*out).resize(getNumberParams());
	//2. loop through covariances
	muint_t i0=0;
	muint_t nparams;
	//loop through covariances and assign
	for(ACovarVec::const_iterator iter = vecCovariances.begin(); iter!=vecCovariances.end();iter++)
	{
		PCovarianceFunction cp = iter[0];
		if (cp!=nullptr)
		{
			nparams = cp->getNumberParams();
			(*out).segment(i0,nparams) = cp->getParamMask();
			i0+=nparams;
		}
	}
}


void AMultiCF::agetParamBounds0(CovarParams* lower, CovarParams* upper) const
{
    //1. create memory
    (*lower).resize(getNumberParams());
    (*upper).resize(getNumberParams());
    //2. loop through and allocate
    muint_t i0=0;
    muint_t nparams;
    for(ACovarVec::const_iterator iter = vecCovariances.begin(); iter!=vecCovariances.end();iter++)
    {
        PCovarianceFunction cp = iter[0];
        if (cp!=nullptr)
        {
            nparams = cp->getNumberParams();
            CovarParams _upper,_lower;
            cp->agetParamBounds0(&_lower,&_upper);
            (*lower).segment(i0,nparams) = _lower;
            (*upper).segment(i0,nparams) = _upper;
            i0+=nparams;
        }
    }
}
    

void AMultiCF::agetParamBounds(CovarParams* lower, CovarParams* upper) const
{
	//1. create memory
	(*lower).resize(getNumberParams());
	(*upper).resize(getNumberParams());
	//2. loop through and allocate
	muint_t i0=0;
	muint_t nparams;
	for(ACovarVec::const_iterator iter = vecCovariances.begin(); iter!=vecCovariances.end();iter++)
	{
		PCovarianceFunction cp = iter[0];
		if (cp!=nullptr)
		{
			nparams = cp->getNumberParams();
			CovarParams _upper,_lower;
			cp->agetParamBounds(&_lower,&_upper);
			(*lower).segment(i0,nparams) = _lower;
			(*upper).segment(i0,nparams) = _upper;
			i0+=nparams;
		}
	}
}

    
void AMultiCF::setParamBounds(const CovarParams& lower, const CovarParams& upper)  {
    //1. check dimensionality
    checkParamDimensions(lower);
    checkParamDimensions(upper);
    //2. loop through covariances
    muint_t i0=0;
    muint_t nparams;
    //loop through covariances and assign
    for(ACovarVec::iterator iter = vecCovariances.begin(); iter!=vecCovariances.end();iter++)
    {
        PCovarianceFunction cp = iter[0];
        if (cp!=nullptr)
        {
            nparams = cp->getNumberParams();
            cp->setParamBounds(lower.segment(i0,nparams),upper.segment(i0,nparams));
            i0+=nparams;
        }
    }
}


/* CSumCF */

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
		if (cp!=nullptr)
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
		if (cp!=nullptr)
		{
			(*out) += cp->Kdiag();
		}
	}
}


void CSumCF::aKcross(MatrixXd *out, const CovarInput & Xstar) const 
				{
	//1. check that Xstar has consistent dimension
	if((muint_t)Xstar.cols()!=this->getNumberDimensions())
		throw CLimixException("Kcross: col dimension of Xstar inconsistent!");
	//2. loop through covariances and add up
	(*out).setConstant(Xstar.rows(),this->Kdim(),0);
	muint_t c0=0;
	muint_t cols;
	for(ACovarVec::const_iterator iter = vecCovariances.begin(); iter!=vecCovariances.end();iter++)
	{
		PCovarianceFunction cp = iter[0];
		if (cp!=nullptr)
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

void CSumCF::aKcross_diag(VectorXd* out, const CovarInput& Xstar) const 
	{
	//1. check that Xstar has consistent dimension
	if((muint_t)Xstar.cols()!=this->getNumberDimensions())
		throw CLimixException("Kcross: col dimension of Xstar inconsistent!");
	//2. loop through covariances and add up
	(*out).setConstant(Xstar.rows(),0);
	muint_t c0=0;
	muint_t cols;
	for(ACovarVec::const_iterator iter = vecCovariances.begin(); iter!=vecCovariances.end();iter++)
	{
		PCovarianceFunction cp = iter[0];
		if (cp!=nullptr)
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


void CSumCF::aKgrad_param(MatrixXd *out, const muint_t i) const 
						{
	//1. check that i is within the available range
	this->checkWithinParams(i);
	//2. loop through covariances until we found the correct one
	muint_t i0=0;
	muint_t params;
	for(ACovarVec::const_iterator iter = vecCovariances.begin(); iter!=vecCovariances.end();iter++)
	{
		PCovarianceFunction cp = iter[0];
		if (cp!=nullptr)
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
    
void CSumCF::aKhess_param(MatrixXd* out, const muint_t i, const muint_t j) const 
{
    //1. check that i is within the available range
    this->checkWithinParams(i);
    this->checkWithinParams(j);
    //2. loop through covariances until we found the correct one
    muint_t i0=0;
    muint_t j0=0;
    muint_t params;
    for(ACovarVec::const_iterator iter = vecCovariances.begin(); iter!=vecCovariances.end();iter++)
    {
        PCovarianceFunction cp = iter[0];
        if (cp!=nullptr)
        {
            params = cp->getNumberParams();
            //is the parameter in that covariance function?
            if((i-i0)<params && (j-j0)<params)
            {
                cp->aKhess_param(out,i-i0,j-j0);
                break;
            }
            else if(((i-i0)<params) || ((j-j0)<params))
            {
                (*out)=MatrixXd::Zero(this->Kdim(),this->Kdim());
                break;
            }
            //move on
            i0+=params;
            j0+=params;
        }
    }
}

void CSumCF::aKgrad_X(MatrixXd* out,const muint_t d) const 
{
	//1. check that d is in range
	this->checkWithinDimensions(d);
	//2. loop over covarainces and get the one corresponding to d
	muint_t c0=0;
	muint_t cols;
	for(ACovarVec::const_iterator iter = vecCovariances.begin(); iter!=vecCovariances.end();iter++)
	{
		PCovarianceFunction cp = iter[0];
		if (cp!=nullptr)
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

void CSumCF::aKdiag_grad_X(VectorXd *out, const muint_t d) const 
{
	//1. check that d is in range
	this->checkWithinDimensions(d);
	muint_t c0=0;
	muint_t cols;
	for(ACovarVec::const_iterator iter = vecCovariances.begin(); iter!=vecCovariances.end();iter++)
	{
		PCovarianceFunction cp = iter[0];
		if (cp!=nullptr)
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


void CSumCF::aKcross_grad_X(MatrixXd *out, const CovarInput & Xstar, const muint_t d) const 
{
	//1. check that d is in range
	this->checkWithinDimensions(d);

	//2. loop over covarainces and get the one corresponding to d
	muint_t c0=0;
	muint_t cols;
	for(ACovarVec::const_iterator iter = vecCovariances.begin(); iter!=vecCovariances.end();iter++)
	{
		PCovarianceFunction cp = iter[0];
		if (cp!=nullptr)
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


/*CLinCombCF*/

CLinCombCF::CLinCombCF(const ACovarVec& covariances) : AMultiCF(covariances)
{
};


CLinCombCF::CLinCombCF(const muint_t numCovariances) :AMultiCF(numCovariances)
{
}

CLinCombCF::~CLinCombCF()
{
}

void CLinCombCF::setCoeff(const VectorXd& coeff)
{
	this->coeff = coeff;
}

void CLinCombCF::agetCoeff(VectorXd* out) const
{
	(*out) = this->coeff;
}

std::string CLinCombCF::getName() const
{
	return "CLinCombCF";
}



void CLinCombCF::aK(MatrixXd* out) const 
{
	muint_t trows = this->Kdim();
	(*out).setConstant(trows,trows,0);
	muint_t i=0;
	for(ACovarVec::const_iterator iter = vecCovariances.begin(); iter!=vecCovariances.end();iter++, i++)
	{
		PCovarianceFunction cp = iter[0];
		if (cp!=nullptr)
		{
			(*out) += this->coeff(i)*cp->K();
		}
	}
}


void CLinCombCF::aKdiag(VectorXd* out) const 
{
	muint_t trows = this->Kdim();
	(*out).setConstant(trows,0);
	muint_t i=0;
	for(ACovarVec::const_iterator iter = vecCovariances.begin(); iter!=vecCovariances.end();iter++, i++)
	{
		PCovarianceFunction cp = iter[0];
		if (cp!=nullptr)
		{
			(*out) += this->coeff(i)*cp->Kdiag();
		}
	}
}


void CLinCombCF::aKcross(MatrixXd *out, const CovarInput & Xstar) const 
{
	//1. check that Xstar has consistent dimension
	if((muint_t)Xstar.cols()!=this->getNumberDimensions())
		throw CLimixException("Kcross: col dimension of Xstar inconsistent!");
	//2. loop through covariances and add up
	(*out).setConstant(Xstar.rows(),this->Kdim(),0);
	muint_t c0=0;
	muint_t cols;
	muint_t i=0;
	for(ACovarVec::const_iterator iter = vecCovariances.begin(); iter!=vecCovariances.end();iter++, i++)
	{
		PCovarianceFunction cp = iter[0];
		if (cp!=nullptr)
		{
			cols = cp->getNumberDimensions();
			MatrixXd Xstar_ = Xstar.block(0,c0,Xstar.rows(),cols);
			MatrixXd t = this->coeff(i)*cp->Kcross(Xstar_);
			(*out) += t;
			//move on
			c0+=cols;
		}
	}
}

void CLinCombCF::aKcross_diag(VectorXd* out, const CovarInput& Xstar) const 
	{
	//1. check that Xstar has consistent dimension
	if((muint_t)Xstar.cols()!=this->getNumberDimensions())
		throw CLimixException("Kcross: col dimension of Xstar inconsistent!");
	//2. loop through covariances and add up
	(*out).setConstant(Xstar.rows(),0);
	muint_t c0=0;
	muint_t cols;
	muint_t i=0;
	for(ACovarVec::const_iterator iter = vecCovariances.begin(); iter!=vecCovariances.end();iter++,i++)
	{
		PCovarianceFunction cp = iter[0];
		if (cp!=nullptr)
		{
			//std::cout << cp->getName() << "\n";
			cols = cp->getNumberDimensions();
			MatrixXd Xstar_ = Xstar.block(0,c0,Xstar.rows(),cols);
			VectorXd t = this->coeff(i)*cp->Kcross_diag(Xstar_);
			//std::cout << t.rows() << "," << Xstar.rows() << "\n";
			(*out) += t;
			//move on
			c0+=cols;
		}
	}

	}


void CLinCombCF::aKgrad_param(MatrixXd *out, const muint_t i) const 
						{
	//1. check that i is within the available range
	this->checkWithinParams(i);
	//2. loop through covariances until we found the correct one
	muint_t i0=0;
	muint_t params;
	muint_t ii=0;
	for(ACovarVec::const_iterator iter = vecCovariances.begin(); iter!=vecCovariances.end();iter++,ii++)
	{
		PCovarianceFunction cp = iter[0];
		if (cp!=nullptr)
		{
			params = cp->getNumberParams();
			//is the parameter in that covariance function?
			if((i-i0)<params)
			{
				cp->aKgrad_param(out,i-i0);
				(*out)*=this->coeff(ii);
				break;
			}
			//move on
			i0+=params;
		}
	}
}

void CLinCombCF::aKhess_param(MatrixXd* out, const muint_t i, const muint_t j) const 
{
    //1. check that i is within the available range
    this->checkWithinParams(i);
    this->checkWithinParams(j);
    //2. loop through covariances until we found the correct one
    muint_t i0=0;
    muint_t j0=0;
    muint_t params;
    muint_t ii=0;
    for(ACovarVec::const_iterator iter = vecCovariances.begin(); iter!=vecCovariances.end();iter++,ii++)
    {
        PCovarianceFunction cp = iter[0];
        if (cp!=nullptr)
        {
            params = cp->getNumberParams();
            //is the parameter in that covariance function?
            if((i-i0)<params && (j-j0)<params)
            {
                cp->aKhess_param(out,i-i0,j-j0);
                (*out)*=this->coeff(ii);
                break;
            }
            else if(((i-i0)<params) || ((j-j0)<params))
            {
                (*out)=MatrixXd::Zero(this->Kdim(),this->Kdim());
                break;
            }
            //move on
            i0+=params;
            j0+=params;
        }
    }
}

void CLinCombCF::aKgrad_X(MatrixXd* out,const muint_t d) const 
{
	//1. check that d is in range
	this->checkWithinDimensions(d);
	//2. loop over covarainces and get the one corresponding to d
	muint_t c0=0;
	muint_t cols;
	muint_t ii=0;
	for(ACovarVec::const_iterator iter = vecCovariances.begin(); iter!=vecCovariances.end();iter++,ii++)
	{
		PCovarianceFunction cp = iter[0];
		if (cp!=nullptr)
		{
			cols = cp->getNumberDimensions();
			if ((d-c0)<cols)
			{
				cp->aKgrad_X(out,d-c0);
				(*out)*=this->coeff(ii);
				break;
			}
			//move on
			c0+=cols;
		}
	}
}

void CLinCombCF::aKdiag_grad_X(VectorXd *out, const muint_t d) const 
{
	//1. check that d is in range
	this->checkWithinDimensions(d);
	muint_t c0=0;
	muint_t cols;
	muint_t ii=0;
	for(ACovarVec::const_iterator iter = vecCovariances.begin(); iter!=vecCovariances.end();iter++,ii++)
	{
		PCovarianceFunction cp = iter[0];
		if (cp!=nullptr)
		{
			cols = cp->getNumberDimensions();
			if ((d-c0)<cols)
			{
				cp->aKdiag_grad_X(out,d-c0);
				(*out)*=this->coeff(ii);
				break;
			}
			//move on
			c0+=cols;
		}
	}
}


void CLinCombCF::aKcross_grad_X(MatrixXd *out, const CovarInput & Xstar, const muint_t d) const 
{
	//1. check that d is in range
	this->checkWithinDimensions(d);

	//2. loop over covarainces and get the one corresponding to d
	muint_t c0=0;
	muint_t cols;
	muint_t ii=0;
	for(ACovarVec::const_iterator iter = vecCovariances.begin(); iter!=vecCovariances.end();iter++,ii++)
	{
		PCovarianceFunction cp = iter[0];
		if (cp!=nullptr)
		{
			cols = cp->getNumberDimensions();
			if ((d-c0)<cols)
			{
				//get Xstar
				MatrixXd Xstar_ = Xstar.block(0,c0,Xstar.rows(),cols);
				//calc
				cp->aKcross_grad_X(out,Xstar_,d-c0);
				(*out)*=this->coeff(ii);
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
		if (cp!=nullptr)
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
		if (cp!=nullptr)
		{
			(*out).array() *= cp->Kdiag().array();
		}
	}
}


void CProductCF::aKcross(MatrixXd *out, const CovarInput & Xstar) const 
{
	//1. check that Xstar has consistent dimension
	if((muint_t)Xstar.cols()!=this->getNumberDimensions())
		throw CLimixException("Kcross: col dimension of Xstar inconsistent!");
	//2. loop through covariances and add up
	(*out).setConstant(Xstar.rows(),this->Kdim(),1.0);
	muint_t c0=0;
	muint_t cols;
	for(ACovarVec::const_iterator iter = vecCovariances.begin(); iter!=vecCovariances.end();iter++)
	{
		PCovarianceFunction cp = iter[0];
		if (cp!=nullptr)
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

void CProductCF::aKcross_diag(VectorXd* out, const CovarInput& Xstar) const 
	{
	//1. check that Xstar has consistent dimension
	if((muint_t)Xstar.cols()!=this->getNumberDimensions())
		throw CLimixException("Kcross: col dimension of Xstar inconsistent!");
	//2. loop through covariances and add up
	(*out).setConstant(Xstar.rows(),1.0);
	muint_t c0=0;
	muint_t cols;
	for(ACovarVec::const_iterator iter = vecCovariances.begin(); iter!=vecCovariances.end();iter++)
	{
		PCovarianceFunction cp = iter[0];
		if (cp!=nullptr)
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


void CProductCF::aKgrad_param(MatrixXd *out, const muint_t i) const 
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
		if (cp!=nullptr)
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
    
void CProductCF::aKhess_param(MatrixXd* out, const muint_t i, const muint_t j) const 
{
    //1. check that i and j are within the available range
    this->checkWithinParams(i);
    this->checkWithinParams(j);
    //2. loop through covariances
    muint_t i0=0;
    muint_t j0=0;
    muint_t params;
    (*out).setConstant(this->Kdim(),this->Kdim(),1.0);
    for(ACovarVec::const_iterator iter = vecCovariances.begin(); iter!=vecCovariances.end();iter++)
    {
        PCovarianceFunction cp = iter[0];
        if (cp!=nullptr)
        {
            params = cp->getNumberParams();
            //is the parameter in that covariance function?
            if((i-i0)<params && (j-j0)<params)
                (*out).array()*=cp->Khess_param(i-i0,j-j0).array();
            else if((i-i0)<params)
                (*out).array()*=cp->Kgrad_param(i-i0).array();
            else if((j-j0)<params)
                (*out).array()*=cp->Kgrad_param(j-j0).array();
            else
                (*out).array()*=cp->K().array();
            //move on
            i0+=params;
            j0+=params;
        }
    }
}

void CProductCF::aKgrad_X(MatrixXd* out,const muint_t d) const 
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
		if (cp!=nullptr)
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

void CProductCF::aKdiag_grad_X(VectorXd *out, const muint_t d) const 
{
	//1. check that d is in range
	this->checkWithinDimensions(d);
	muint_t c0=0;
	muint_t cols;
	(*out).setConstant(this->Kdim(),1.0);
	for(ACovarVec::const_iterator iter = vecCovariances.begin(); iter!=vecCovariances.end();iter++)
	{
		PCovarianceFunction cp = iter[0];
		if (cp!=nullptr)
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


void CProductCF::aKcross_grad_X(MatrixXd *out, const CovarInput & Xstar, const muint_t d) const 
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
		if (cp!=nullptr)
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



CKroneckerCF::CKroneckerCF() : AMultiCF(2,2) {
	vecCovariances.resize(2);
	kroneckerIndicator = MatrixXi(0,0);
}

CKroneckerCF::CKroneckerCF(PCovarianceFunction col,
		PCovarianceFunction row): AMultiCF(2,2) {
	kroneckerIndicator = MatrixXi(0,0);
	setRowCovariance(row);
	setColCovariance(col);
}

CKroneckerCF::~CKroneckerCF() {
}

muint_t CKroneckerCF::Kdim() const 
{
	//do we have a Kronecker index?
	if(!isnull(kroneckerIndicator))
	{
		return kroneckerIndicator.rows();
	}
	else
	{
		return vecCovariances.begin()[0]->Kdim()*vecCovariances.begin()[1]->Kdim();
	}
}

void CKroneckerCF::setRowCovariance(PCovarianceFunction cov) {
	setCovariance(1,cov);
}

void CKroneckerCF::setColCovariance(PCovarianceFunction cov) {
	setCovariance(0,cov);
}

PCovarianceFunction CKroneckerCF::getRowCovariance() 
{
	return vecCovariances[0];
}

PCovarianceFunction CKroneckerCF::getColCovariance() 
{
	return vecCovariances[1];
}

void CKroneckerCF::setXr(const CovarInput& Xr) 
{
	vecCovariances[0]->setX(Xr);
}

void CKroneckerCF::setXc(const CovarInput& Xc) 
{
	vecCovariances[1]->setX(Xc);
}

void CKroneckerCF::setKroneckerIndicator(const MatrixXi& kroneckerIndicator)
{
	this->kroneckerIndicator = kroneckerIndicator;
};
void CKroneckerCF::getKroneckerIndicator(MatrixXi* out) const
{
	(*out) = this->kroneckerIndicator;
};


std::string CKroneckerCF::getName() const {
	return "KroneckerCF";
}


bool CKroneckerCF::isKronecker() const
{
	return isnull(this->kroneckerIndicator);
}

void CKroneckerCF::aKcross(MatrixXd* out, const CovarInput& Xstar) const 
{
}

void CKroneckerCF::aKcross_diag(VectorXd* out, const CovarInput& Xstar) const 
{
}

void CKroneckerCF::aKgrad_param(MatrixXd* out, const muint_t i) const 
{
	//1. check that i is within the available range
	checkWithinParams(i);
	//2. loop through covariances until we found the correct one
	muint_t i0 = (muint_t)vecCovariances[0]->getNumberParams();

	if(i<i0)
		aMatrixIndexProduct((*out),vecCovariances[0]->Kgrad_param(i),vecCovariances[1]->K(),kroneckerIndicator);
	else
		aMatrixIndexProduct((*out),vecCovariances[0]->K(),vecCovariances[1]->Kgrad_param(i-i0),kroneckerIndicator);
}

void CKroneckerCF::aKhess_param(MatrixXd* out, const muint_t i,
		const muint_t j) const 
{
	//1. check that i is within the available range
	checkWithinParams(i);
	checkWithinParams(j);
	//2. swap if i>j
	muint_t i1;
	muint_t j1;
	if (i<j)	{i1=i; j1=j;}
	else		{i1=j; j1=i;}
	//3. loop through covariances until we found the correct one
	muint_t i0 = (muint_t)vecCovariances[0]->getNumberParams();
	if(j1<i0)
		aMatrixIndexProduct((*out),vecCovariances[0]->Khess_param(i1,j1),vecCovariances[1]->K(),kroneckerIndicator);
	else if(i1>=i0)
		aMatrixIndexProduct((*out),vecCovariances[0]->K(),vecCovariances[1]->Khess_param(i1-i0,j1-i0),kroneckerIndicator);
	else
		aMatrixIndexProduct((*out),vecCovariances[0]->Kgrad_param(i1),vecCovariances[1]->Kgrad_param(j1-i0),kroneckerIndicator);
}


void CKroneckerCF::aKcross_grad_X(MatrixXd* out, const CovarInput& Xstar,
		const muint_t d) const 
{
}

void CKroneckerCF::aKdiag_grad_X(VectorXd* out, const muint_t d) const 
{
}

void CKroneckerCF::aK(MatrixXd* out) const 
{
	MatrixXd Kc;
	vecCovariances[0]->aK(&Kc);
	MatrixXd Kr;
	vecCovariances[1]->aK(&Kr);
	aMatrixIndexProduct((*out),Kc,Kr,kroneckerIndicator);
}

void CKroneckerCF::aKdiag(VectorXd* out) const 
{
  MatrixXd Kc;
  vecCovariances[0]->aK(&Kc);
  MatrixXd Kr;
  vecCovariances[1]->aK(&Kr);

  //aMatrixIndexProduct_diag((*out),Kc,Kr,kroneckerIndicator);
  akron_diag(*out,Kc,Kr);
}

void CKroneckerCF::aKgrad_X(MatrixXd* out, const muint_t d) const 
{
}
 
/*!
 * create Kronecker index, assuming: first coumn in out rows, second, columns for Kc \kron Kr
 */
void CKroneckerCF::createKroneckerIndex(MatrixXi* out, muint_t Ncols, muint_t Nrows)
{
	//resize result structure
	out->resize(Nrows*Ncols,2);
	//double loop over rows and columns
	for (muint_t c=0;c<Ncols;++c)
	{
		for (muint_t r=0;r<Nrows;++r)
		{
			(*out)(c*Nrows+r,0) = c;
			(*out)(c*Nrows+r,1) = r;
		}
	}
} //end ::getKroneckerIndex

} /// end :limix


