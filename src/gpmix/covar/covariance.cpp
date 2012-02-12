/*
ear  * ACovariance.cpp
 *
 *  Created on: Nov 10, 2011
 *      Author: stegle
 */

#include "covariance.h"
#include "gpmix/utils/matrix_helper.h"

namespace gpmix {

ACovarianceFunction::ACovarianceFunction(muint_t numberParams)
{
	this->numberParams =numberParams;
	this->insync = false;
	this->params = VectorXd(numberParams);
	//default: 0 dimensions
	this->numberDimensions = 0;
}


ACovarianceFunction::~ACovarianceFunction()
{

}


muint_t ACovarianceFunction::Kdim() const throw(CGPMixException)
{
	if(isnull(X))
		throw CGPMixException("ACovarianceFunction: cannot query covariance dimension without X!");
	//standard: use X to determine dimension:
	return X.rows();
}



//set the parameters to a new value.
void ACovarianceFunction::setParams(const CovarParams& params)
{
	checkParamDimensions(params);
	this->params = params;
	this->insync = false;
}

void ACovarianceFunction::agetParams(CovarParams* out)
{(*out) = params;};

bool ACovarianceFunction::isInSync() const
{return insync;}

void ACovarianceFunction::makeSync()
{ insync = true;}

void ACovarianceFunction::setNumberDimensions(muint_t numberDimensions)
{
	this->numberDimensions=numberDimensions;
	this->numberParams = 1;
}

muint_t ACovarianceFunction::getNumberParams() const
{return numberParams;}

muint_t ACovarianceFunction::getNumberDimensions() const
{return numberDimensions;}



void ACovarianceFunction::setX(const CovarInput & X) throw (CGPMixException)
{
	checkXDimensions(X);
	this->X = X;
	this->insync = false;
}

void ACovarianceFunction::setXcol(const CovarInput& X,muint_t col) throw (CGPMixException)
{
	//1. check dimensions etc.
	if(((col+X.cols())>(muint_t)this->X.cols()) || (X.rows()!=this->X.rows()))
	{
		std::ostringstream os;
		os << "setXcol out of range. Current X:"<<this->getNumberDimensions() <<")";
		throw CGPMixException(os.str());
	}
	this->X.block(0,col,X.rows(),X.cols()) = X;
}

void ACovarianceFunction::agetX(CovarInput *Xout) const throw(CGPMixException)
{
	(*Xout) = this->X;
}



void ACovarianceFunction::aK(MatrixXd* out) const
{
	aKcross(out,X);
}

void ACovarianceFunction::aKdiag(VectorXd *out) const
{
	MatrixXd Kfull = K();
	(*out) = Kfull.diagonal();
	return;
}

void ACovarianceFunction::aKgrad_X(MatrixXd *out, const muint_t d) const throw(CGPMixException)
{
	aKcross_grad_X(out,X,d);
}

bool ACovarianceFunction::check_covariance_Kgrad_theta(ACovarianceFunction& covar,mfloat_t relchange,mfloat_t threshold)
{
	mfloat_t RV=0;

	//copy of parameter vector
	CovarParams L = covar.getParams();
	//create copy
	CovarParams L0 = L;
	//dimensions
	for(mint_t i=0;i<L.rows();i++)
	{
		mfloat_t change = relchange*L(i);
		change = std::max(change,1E-5);
		L(i) = L0(i) + change;
		covar.setParams(L);
		MatrixXd Lplus = covar.K();
		L(i) = L0(i) - change;
		covar.setParams(L);
		MatrixXd Lminus = covar.K();
		//numerical gradient
		MatrixXd diff_numerical  = (Lplus-Lminus)/(2.*change);
		//analytical gradient
		covar.setParams(L0);
		MatrixXd diff_analytical = covar.Kgrad_param(i);
		RV += (diff_numerical-diff_analytical).squaredNorm();
	}
	return (RV < threshold);
}

bool ACovarianceFunction::check_covariance_Kgrad_x(ACovarianceFunction& covar,mfloat_t relchange,mfloat_t threshold,bool check_diag)
{
	mfloat_t RV=0;
	//copy inputs for which we calculate gradients
	CovarInput X = covar.getX();
	CovarInput X0 = X;
	for (int ic=0;ic<X.cols();ic++)
	{
		//analytical gradient is per columns all in one go:
		MatrixXd Kgrad_x = covar.Kgrad_X(ic);
		MatrixXd Kgrad_x_diag = covar.Kdiag_grad_X(ic);
		for (int ir=0;ir<X.rows();ir++)
		{
			mfloat_t change = relchange*X0(ir,ic);
			change = std::max(change,1E-5);
			X(ir,ic) = X0(ir,ic) + change;
			covar.setX(X);
			MatrixXd Lplus = covar.K();
			X(ir,ic) = X0(ir,ic) - change;
			covar.setX(X);
			MatrixXd Lminus = covar.K();
			X(ir,ic) = X0(ir,ic);
			covar.setX(X);
			//numerical gradient
			MatrixXd diff_numerical = (Lplus-Lminus)/(2.*change);
			//build analytical gradient matrix
			MatrixXd diff_analytical = MatrixXd::Zero(X.rows(),X.rows());
			diff_analytical.row(ir) = Kgrad_x.row(ir);
			diff_analytical.col(ir) += Kgrad_x.row(ir);
			RV+= (diff_numerical-diff_analytical).squaredNorm();
			//difference
			if (check_diag)
			{
				double delta =(diff_numerical(ir,ir)-Kgrad_x_diag(ir));
				RV+= delta*delta;
			}
		} //end for ir
	}
	return (RV < threshold);
}



}/* namespace gpmix */


