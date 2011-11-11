/*
 * fixed.cpp
 *
 *  Created on: Nov 11, 2011
 *      Author: stegle
 */

#include "fixed.h"

namespace gpmix {

	CFixedCF::CFixedCF(const MatrixXd K0)
	{
		this->K0 = K0;
	}

	CFixedCF::~CFixedCF() {
		// TODO Auto-generated destructor stub
	}

	MatrixXd CFixedCF::K(const CovarParams params,const CovarInput x1,const CovarInput x2)
	{
		//TODO: check that x1 and x2 are identical
		return this->K0;
	}

	VectorXd CFixedCF::Kdiag(const CovarParams params,const CovarInput x1)
	{
		return this->K0.diagonal();
	}

	MatrixXd CFixedCF::Kgrad_theta(const CovarParams params, const CovarInput x1,const uint_t i)
	{
		return MatrixXd();
	}

	MatrixXd CFixedCF::Kgrad_x(const CovarParams params, const CovarInput x1, const CovarInput x2, const uint_t d)
	{
		return MatrixXd();
	}

	MatrixXd CFixedCF::Kgrad_xdiag(const CovarParams params, const CovarInput x1, const uint_t d)
	{
		return MatrixXd();
	}

} /* namespace gpmix */
