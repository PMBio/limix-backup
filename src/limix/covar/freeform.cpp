/*
 * freeform.cpp
 *
 *  Created on: Jan 16, 2012
 *      Author: stegle
 */

#include "freeform.h"
#include <math.h>
#include <cmath>


namespace limix {

CFreeFormCF::CFreeFormCF(muint_t numberGroups)
{
	//1 input dimension which selects the group:
	this->numberDimensions = 1;
	//number of groups and parameters:
	this->numberGroups= numberGroups;
	this->numberParams = calcNumberParams(numberGroups);

}

muint_t CFreeFormCF::calcNumberParams(muint_t numberGroups)
{
	return (0.5*numberGroups*(numberGroups-1) + numberGroups);
}

CFreeFormCF::~CFreeFormCF()
{
}

void CFreeFormCF::aK0Covar2Params(VectorXd* out,const MatrixXd& K0,muint_t numberGroups)
{
	//0. check that the matrix has the correct size
	if(((muint_t)K0.rows()!=numberGroups) || ((muint_t)K0.cols()!=numberGroups))
	{
		throw CGPMixException("aK0Covar2Params: rows and columns need to be compatiable with the number of groups");
	}

	//1. calculate cholesky of K0
	MatrixXdChol chol(K0);
	//2. get L matrix
	MatrixXd L = chol.matrixL();
	//3. create output argument and fill
	(*out) = VectorXd::Zero(calcNumberParams(numberGroups));
	//4. loop over groups
	muint_t pindex=0;
	for(muint_t ir=0;ir<numberGroups;++ir)
		for (muint_t ic=0;ic<(ir+1);++ic)
		{
			(*out)(pindex) = L(ir,ic);
			++pindex;
		}
}

VectorXd CFreeFormCF::K0Covar2Params(const MatrixXd& K0,muint_t numberGroups) {
	VectorXd RV;
	aK0Covar2Params(&RV,K0, numberGroups);
	return RV;
}


void CFreeFormCF::setParamsCovariance(const MatrixXd& K0)
{
	CovarParams params;
	aK0Covar2Params(&params,K0,this->numberGroups);
	this->setParams(params);
}


void CFreeFormCF::agetL0(MatrixXd* out) const
{
	/*contruct cholesky factor from hyperparameters*/
	(*out).setConstant(numberGroups,numberGroups,0);
	muint_t pindex=0;
	//for rows
	for(muint_t ir=0;ir<numberGroups;++ir)
		for (muint_t ic=0;ic<(ir+1);++ic)
		{
			//diagonal is exponentiated
			(*out)(ir,ic) = params(pindex);
			++pindex;
		}
}

void CFreeFormCF::agetL0grad_param(MatrixXd* out,muint_t i) const throw(CGPMixException)
{
	/*construct cholesky factors from hyperparameters*/
		(*out).setConstant(numberGroups,numberGroups,0);
		muint_t pindex=0;
		//for rows
		for(muint_t ir=0;ir<numberGroups;++ir)
			for (muint_t ic=0;ic<(ir+1);++ic)
			{
				if (pindex==i)
				{
					(*out)(ir,ic) = 1;
				}
				++pindex;
			}
}

void CFreeFormCF::agetParamBounds(CovarParams* lower,CovarParams* upper) const
{
	//all parameters but the diagonal elements are unbounded:
	*lower = -INFINITY*VectorXd::Ones(this->getNumberParams());
	*upper = +INFINITY*VectorXd::Ones(this->getNumberParams());
	//get diagonal elements
	VectorXi isDiagonal =getIparamDiag();
	//set diagonal elements to be bounded [0,inf]
	for (muint_t i=0;i<getNumberParams();++i)
	{
		if(isDiagonal(i))
		{
			(*lower)(i) = 0;
		}
	}
}


void CFreeFormCF::agetK0(MatrixXd* out) const
{
	//create template matrix K
	MatrixXd L;
	agetL0(&L);
	(*out).noalias() = L*L.transpose();
}



void CFreeFormCF::agetK0grad_param(MatrixXd* out,muint_t i) const throw(CGPMixException)
{
	MatrixXd L;
	MatrixXd Lgrad_parami;

	agetL0(&L);
	agetL0grad_param(&Lgrad_parami,i);
	//use chain rule K = LL^T
	(*out).noalias() = Lgrad_parami*L.transpose() + L*Lgrad_parami.transpose();
}

void CFreeFormCF::projectKcross(MatrixXd* out,const MatrixXd& K0,const CovarInput& Xstar) const throw (CGPMixException)
{
	//2. loop thourugh entries and create effective covariance:
	//index for param
	(*out).setConstant(Xstar.rows(),X.rows(),0.0);
	MatrixXd Km_rc;
	VectorXd Xb;
	VectorXd Xstarb;
	for (muint_t ir=0;ir<numberGroups;++ir)
		for(muint_t ic=0;ic<numberGroups;++ic)
		{
			//get relevant data entries (N,N) binary matrix
			Xstarb = (Xstar.array()==(mfloat_t)ir).cast<mfloat_t>();
			Xb 	   = (X.array()==(mfloat_t)ic).cast<mfloat_t>();
			//add component to K
			Km_rc.noalias() = K0(ir,ic)* (Xstarb*Xb.transpose());
			(*out) += Km_rc;
		}
}

void CFreeFormCF::aKcross(MatrixXd* out, const CovarInput& Xstar ) const throw(CGPMixException)
{
	checkXDimensions(Xstar);
	//1. get K0 Matrix, the template for all others
	MatrixXd K0;
	agetK0(&K0);
	projectKcross(out,K0,Xstar);
}

void CFreeFormCF::aKcross_diag(VectorXd* out, const CovarInput& Xstar) const throw(CGPMixException)
{
	checkXDimensions(Xstar);
	//dignal matrix of Kstar
	//1. get K0 Matrix, the template for all others
	MatrixXd KK;
	agetK0(&KK);
	(*out).resize(Xstar.rows());
	for (muint_t ir=0;ir<(muint_t)Xstar.rows();++ir)
	{
		//get index
		muint_t index = Xstar(ir,0);
		//set element
		(*out)(ir) = KK(index,index);
	}
}

void CFreeFormCF::aKgrad_param(MatrixXd* out,const muint_t i) const throw(CGPMixException)
{
	checkWithinParams(i);
	// same as Kcross, however using a different base matrix K0
	//1. get K0 Matrix, the template for all others
	MatrixXd K0;
	agetK0grad_param(&K0,i);
	projectKcross(out,K0,X);
}

void CFreeFormCF::aKcross_grad_X(MatrixXd* out,const CovarInput& Xstar, const muint_t d) const throw(CGPMixException)
{
}

void CFreeFormCF::aKdiag_grad_X(VectorXd* out,const muint_t d) const throw(CGPMixException)
{
}


void CFreeFormCF::agetIparamDiag(VectorXi* out) const
{
	(*out) = VectorXi::Zero(getNumberParams(),1);
	//for rows
	muint_t pindex=0;
	for(muint_t ir=0;ir<numberGroups;++ir)
		for (muint_t ic=0;ic<(ir+1);++ic)
		{
			//diagonal is exponentiated
			if (ic==ir)
				(*out)(pindex) = 1;
			++pindex;
		}
}

} /* namespace limix */
