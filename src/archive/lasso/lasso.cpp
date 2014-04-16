// Copyright(c) 2014, The LIMIX developers (Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
// All rights reserved.
//
// LIMIX is provided under a 2-clause BSD license.
// See license.txt for the complete license.

#include "lasso.h"
#include <iostream>

namespace limix {


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
		//(*out) = X.transpose()*ms.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(y);
		(*out) = X.transpose()*ms.ldlt().solve(y);
		(*out) *= (1.0/mu);
		//return 1./mu*SP.dot(X.T, LA.solve(SP.eye(n)+1./mu*SP.dot(X,X.T),y))
	}
	else
	{
		MatrixXd ms(d,d);
		ms.diagonal().array()+=mu;
		ms += X.transpose()*X;
		//(*out) = ms.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(X.transpose()*y);
		(*out) = ms.ldlt().solve(X.transpose()*y);
		//return LA.solve(SP.dot(X.T,X) + mu*SP.eye(d),SP.dot(X.T,y)))
	}
}

//#define LASSO_IRR

void lasso_irr(MatrixXd* w_out,const MatrixXd& Xfull,const MatrixXd& y, mfloat_t mu, mfloat_t optTol,mfloat_t threshold, muint_t maxIter)
{
	//0. get dimensionality and check for consistency
	muint_t n = Xfull.rows();
	muint_t d = Xfull.cols();
	assert (n==(muint_t)y.rows());

	if (d<n)
	{
		std::cout << "lasso_irr is optimized for n<d. Please consider transposing your input data or update the code base.";
	}

	//1. initialize weights with ridge regression
	mu /= 2.0;
	mfloat_t imu = (1.0/mu);

	MatrixXd w;
	//old weights :ones
	MatrixXd w_old = MatrixXd::Ones(d,1);
	//absolute values of w for cashing:
	MatrixXd w_abs;
	//init with ridge regression
	ridge_regression(&w,Xfull,y,mu);

	// intialize NN
	MatrixXd NN = MatrixXd::Zero(n,n);
	MatrixXd NNiter = MatrixXd::Zero(n,n);
	MatrixXd XmuwXIy;
	mfloat_t mw;

	// solve iteratively reweighted least squares problems
	muint_t i_iter;
	for (i_iter=0;i_iter++<maxIter;i_iter++)
	{
		//assignment is copy in Eigen:
		w_old = w;
		//get abs weight values to identify non-zeros:
		w_abs = w.array().abs();

		//number of non-zero entries?
		muint_t n_zero = 0;

		//set NN zero:
		NN.setZero();
		//loop over non-zero entries in w_tmp
		for (muint_t i_d=0;i_d<(muint_t)w_abs.rows();i_d++)
		{
	#ifdef LASSO_IRR
			//1. check whether non-zero
			if (w_abs(i_d,0)<threshold)
				continue;
	#endif
			//ok, otherwise increment non-zeros
			n_zero++;
			// (1.0/mu) * abs
			mw = imu*w_abs(i_d,0);
			//reweighted inner product of this patricual feature onto NN
			NNiter.noalias() = (Xfull.col(i_d)*Xfull.col(i_d).transpose());
			NNiter *= mw;
			NN += NNiter;
		}//end for all non-zero SNPs
		//check whethere any feature is non-zero
		if (n_zero==0)
			break;
		//add 1. diagonal to NN
		NN += MatrixXd::Identity(n,n);
		//solve linear system
		XmuwXIy = NN.ldlt().solve(y).transpose();

		//compile new weight vector:
		for (muint_t i_d=0;i_d<(muint_t)w_abs.rows();i_d++)
		{
	#ifdef LASSO_IRR
			if (w_abs(i_d,0)<threshold)
				continue;
	#endif
			//update w:
			mw = imu*w_abs(i_d,0);
			w.block(i_d,0,1,1).noalias() = mw*XmuwXIy*Xfull.col(i_d);
			//move internal non-zero weight index forward
		} //end
		//check tolerenace of weight changes:
		if ((w-w_old).array().abs().sum()< optTol)
			break;
	}//end for maxIter

	//set everything below threshold explicitly to zero
	muint_t n_zero = 0;
	w_abs = w.array().abs();
	for (muint_t i_d=0;i_d<(muint_t)w.rows();i_d++)
		if (w_abs(i_d,0)<threshold)
			w(i_d,0) = 0;
		else
			n_zero++;

	//debug information
	std::cout << "number of iterations: " << (i_iter+1) << "\n";
	std::cout << "converged until: " << ((w-w_old).array().abs().sum()) << "\n";
	std::cout << "number of non-zero entries: "<<n_zero << "\n";

	(*w_out) = w;
}





} //end:: namepsace gpmix





