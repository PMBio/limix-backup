/*
 * lasso.cpp
 *
 *  Created on: Dec 24, 2011
 *      Author: stegle
 */

#include "lasso.h"


namespace gpmix {


void ridge_regression(MatrixXd* out, const MatrixXd& X, const MatrixXd& y,mfloat_t mu)
{
	muint_t n = X.rows();
	muint_t d = X.cols();
	if (d>n)
	{
		MatrixXd ms;
		//diagonal amtrix with ones
		ms = MatrixXd::Zero(n,n);
		ms.diagonal().array() += 1;
		ms += 1.0/(mu) * X*X.transpose();
		(*out) = X.transpose()*ms.jacobiSvd().solve(y);
		(*out) *= (1.0/mu);
	    //return 1./mu*SP.dot(X.T, LA.solve(SP.eye(n)+1./mu*SP.dot(X,X.T),y))
	}
	else
	{
		MatrixXd ms(d,d);
		ms.diagonal().array()+=mu;
		ms += X.transpose()*X;
		(*out) = ms.jacobiSvd().solve(X.transpose()*y);
		//return LA.solve(SP.dot(X.T,X) + mu*SP.eye(d),SP.dot(X.T,y)))
	}
}



void lasso_irr(MatrixXd* w_out,const MatrixXd& X,const MatrixXd& y, mfloat_t mu, mfloat_t optTol,mfloat_t threshold, muint_t maxIter)
{
	//0. get dimensionality and check for consistency
	muint_t n = X.rows();
	muint_t d = X.cols();
	assert (n==(muint_t)y.rows());

	//1. initialize weights with ridge regression
	std::cout << d;

}

} //end:: namepsace gpmix





