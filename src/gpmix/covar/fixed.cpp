/*
 * fixed.cpp
 *
 *  Created on: Nov 11, 2011
 *      Author: stegle
 */

#include "fixed.h"
#include "gpmix/types.h"
#include "assert.h"

namespace gpmix {

	CFixedCF::CFixedCF(const MatrixXd K0) : ACovarianceFunction(0)
	{
		this->K0 = K0;
		//check that matrix is squared
		if (K0.rows()!=K0.cols())
				throw CGPMixException("Fixed CF requires squared covariance matrix");
	}

	CFixedCF::~CFixedCF() {
		// TODO Auto-generated destructor stub
	}

	MatrixXd CFixedCF::K(const CovarParams params, const CovarInput x1, const CovarInput x2) const
	{
		if (this->K0.rows()!=x1.rows())
		{
			throw CGPMixException("Unaligned input dimensons in FixedCF");
		}
		float_t A = (float_t)std::exp((long double) (2.0*params(0)));
		return A*this->K0;
	}

	VectorXd CFixedCF::Kdiag(const CovarParams params,const CovarInput x1) const
	{
		float_t A = (float_t)std::exp((long double) (2.0*params(0)));
		return A*this->K0.diagonal();
	}

	MatrixXd CFixedCF::Kgrad_theta(const CovarParams params, const CovarInput x1,const uint_t i) const
	{
		if(i==0)
		{
			MatrixXd K = this->K(params,x1,x1);
			K*=2.0;
			return K;
		}
		else
			throw CGPMixException("unknown hyperparameter derivative requested in CLinearCFISO");
	}

	MatrixXd CFixedCF::Kgrad_x(const CovarParams params, const CovarInput x1, const CovarInput x2, const uint_t d) const
	{
		//create empty matrix
		MatrixXd RV = MatrixXd::Zero(x1.rows(),x2.rows());
		return RV;
	}

	MatrixXd CFixedCF::Kgrad_xdiag(const CovarParams params, const CovarInput x1, const uint_t d) const
	{
		VectorXd RV = VectorXd::Zero(x1.rows());
		return RV;
	}


} /* namespace gpmix */
