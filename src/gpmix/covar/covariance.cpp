/*
 * ACovariance.cpp
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
}


ACovarianceFunction::~ACovarianceFunction()
{
}



//set the parameters to a new value.
inline void ACovarianceFunction::setParams(const CovarParams& params)
{
	if ((muint_t)(params.cols()) != this->getNumberParams()){
		ostringstream os;
		os << "Wrong number of params for covariance funtion " << this->getName() << ". numberParams = " << this->getNumberParams() << ", params.cols() = " << params.cols();
		throw gpmix::CGPMixException(os.str());
	}
	this->params = params;
	this->insync = false;
}


void ACovarianceFunction::K(MatrixXd* out) const
{
	Kcross(out,X);
}

void ACovarianceFunction::Kdiag(VectorXd *out) const
{
	MatrixXd Kfull = K();
	//out->resize(Kfull.rows());
	(*out) = Kfull.diagonal();
	return;
}

void ACovarianceFunction::Kgrad_X(MatrixXd *out, const muint_t d) const
{
	Kcross_grad_X(out,X,d);
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
		change = max(change,1E-5);
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

bool ACovarianceFunction::check_covariance_Kgrad_x(ACovarianceFunction& covar,mfloat_t relchange,mfloat_t threshold)
{
	mfloat_t RV=0;
	//copy inputs for which we calculate gradients
	CovarInput X = covar.getX();
	CovarInput X0 = X;
	for (int ic=0;ic<X.cols();ic++)
	{
		//analytical gradient is per columns all in one go:
		MatrixXd Kgrad_x = covar.Kgrad_X(ic);
		for (int ir=0;ir<X.rows();ir++)
		{
			mfloat_t change = relchange*X0(ir,ic);
			change = max(change,1E-5);
			X(ir,ic) = X0(ir,ic) + change;
			covar.setX(X);
			MatrixXd Lplus = covar.K();
			X(ir,ic) = X0(ir,ic) - change;
			covar.setX(X);
			MatrixXd Lminus = covar.K();
			X(ir,ic) = X0(ir,ic);
			//numerical gradient
			MatrixXd diff_numerical = (Lplus-Lminus)/(2.*change);
			//build analytical gradient matrix
			MatrixXd diff_analytical = MatrixXd::Zero(X.rows(),X.rows());
			for (int n=0;n<X.rows();n++)
			{
				diff_analytical.row(n) = Kgrad_x.row(n);
				diff_analytical.col(n) += Kgrad_x.row(n);
			}
			RV+= (diff_numerical-diff_analytical).squaredNorm();
		} //end for ir
	}
	return (RV < threshold);
}



}/* namespace gpmix */


