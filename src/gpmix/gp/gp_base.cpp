/*
 * gp_base.cpp
 *
 *  Created on: Nov 11, 2011
 *      Author: stegle
 */

#include "gp_base.h"

#ifndef PI
#define PI 3.14159265358979323846
#endif

namespace gpmix {

	CGPbase::CGPbase(ACovarianceFunction& covar, ALikelihood& lik) : covar(covar), lik(lik) {
		this->covar = covar;
		this->lik = lik;
		this->clearCache();
	}

	CGPbase::~CGPbase() {
		// TODO Auto-generated destructor stub
	}

	void CGPbase::set_data(MatrixXd& Y)
	{
		this->Y = Y;
	}

	void CGPbase::clearCache()
	{
		this->K=MatrixXd();
		this->Kinv=MatrixXd();
		this->KinvY=MatrixXd();
		this->cholK=Eigen::LDLT<gpmix::MatrixXd>();
		this->DKinv_KinvYYKinv = MatrixXd();
		this->covar.makeSync();
		this->lik.makeSync();
	}

	MatrixXd CGPbase::getKinv()
	{
		if(!this->covar.isInSync() || !this->lik.isInSync())
		{
			this->clearCache();
		}
		if (this->Kinv.cols()==0)
		{
			Eigen::LDLT<gpmix::MatrixXd> chol = this->getCholK();
			this->Kinv = MatrixXd::Identity(this->get_samplesize(),this->get_samplesize());
			chol.solveInPlace(this->Kinv);
		}
		return this->Kinv;
	}

	MatrixXd CGPbase::getKinvY()
	{
		if(!this->covar.isInSync() || !this->lik.isInSync())
		{
			this->clearCache();
		}
		if (this->KinvY.cols()==0 && this->Kinv.cols()!=0)
		{
			this->KinvY = this->Kinv * this->Y;
		}
		else if (this->KinvY.cols()==0)
		{
			Eigen::LDLT<gpmix::MatrixXd> chol = this->getCholK();
			this->KinvY = chol.solve(this->Y);
		}
		return this->KinvY;
	}

	MatrixXd CGPbase::getDKinv_KinvYYKinv()
	{
		if(!this->covar.isInSync() || !this->lik.isInSync())
		{
			this->clearCache();
		}
		if (this->DKinv_KinvYYKinv.cols()==0)
		{
			MatrixXd KiY = this->getKinvY();
			this->DKinv_KinvYYKinv = ((mfloat_t)this->get_target_dimension())*this->getKinv() - KiY * KiY.transpose();//WARNING:conversion uint64_t to double
		}
		return this->DKinv_KinvYYKinv;
	}

	Eigen::LDLT<gpmix::MatrixXd> CGPbase::getCholK()
	{
		if(!this->covar.isInSync() || !this->lik.isInSync())
		{
			this->clearCache();
		}
		if (this->cholK.cols()==0)
		{
			this->cholK = Eigen::LDLT<gpmix::MatrixXd>(  this->getK() );
		}
		return this->cholK;
	}

	MatrixXd CGPbase::getK()
	{
		if(!this->covar.isInSync() || !this->lik.isInSync())
		{
			this->clearCache();
		}
		if (this->K.cols()==0)
		{
			this->K=this->covar.K(); //This line breaks for se kernel
			this->lik.applyToK(this->covar.getX(),this->K);
		}
		return this->K;
	}

//	void CGPbase::set_params(CGPHyperParams& hyperparams)
//	{
//		this->clearCache();
//		this->params = hyperparams;
//	}

	mfloat_t CGPbase::LML()
	{
		//update the covariance parameters
		Eigen::LDLT<gpmix::MatrixXd>chol = CGPbase::getCholK();

		mfloat_t lml_det = 0.0;
		for (muint_t i = 0; i<(muint_t)chol.vectorD().rows(); ++i)
		{
			lml_det+=gpmix::log((mfloat_t)chol.vectorD()(i));//WARNING: mfloat_t cast
		}
      
		mfloat_t lml_quad = 0.0;
		MatrixXd KinvY = this->getKinvY();
		//loop over independent columns of Y:
		for (muint_t colY=0; colY<(muint_t)this->Y.cols();++colY)
		{
			lml_quad += this->Y.col(colY).transpose() * KinvY.col(colY);
		}

		mfloat_t lml_const = this->Y.cols() * this->Y.rows() * gpmix::log((2.0 * PI));

		return 0.5 * (lml_quad + this->Y.cols() * lml_det + lml_const);
	}



	CovarParams CGPbase::LMLgrad_covar()
	{
		CovarParams grad_covar(covar.getNumberParams());

		MatrixXd W = this->getDKinv_KinvYYKinv();
		for(muint_t row = 0 ; row<(muint_t)grad_covar.rows(); ++row)//WARNING: conversion
		{
			MatrixXd Kd = this->covar.Kgrad_param(row);
			grad_covar(row) = 0.5*(Kd.array() * W.array()).sum();
		}
		return grad_covar;
	}

	LikParams CGPbase::LMLgrad_lik()
	{
		LikParams grad_lik(lik.getNumberParams());

		MatrixXd W = this->getDKinv_KinvYYKinv();
		for(muint_t row = 0 ; row<lik.getNumberParams(); ++row)	//WARNING: conversion
		{
			MatrixXd Kd = this->lik.K_grad_params(this->covar.getX(), row);
			grad_lik(row) = 0.5*(Kd.array() * W.array()).sum();
		}
		return grad_lik;
	}

/*	MatrixXd CGPbase::predictMean(MatrixXd& Xstar)
	{
		MatrixXd KstarCross = this->covar.K(this->params.get("covar"), Xstar, this->X);
		return KstarCross * this->getKinvY();
	}

	MatrixXd CGPbase::predictVar(MatrixXd& Xstar)
	{
		MatrixXd KstarDiag = this->covar.Kdiag(this->params.get("covar"), Xstar);
		KstarDiag+=this->lik.Kdiag(this->params.get("lik"), Xstar);
		MatrixXd Kcross = this->covar.K(this->params.get("covar"), this->X, Xstar);
		MatrixXd v = this->getCholK().solve(Kcross);
		MatrixXd vv = (v.array()*v.array()).matrix().colwise().sum();
		MatrixXd S2 = KstarDiag - vv.transpose();

		return S2;
	}
*/
} /* namespace gpmix */
