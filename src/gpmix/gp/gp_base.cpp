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
		this->params=CGPHyperParams();
		this->clearCache();
	}

	CGPbase::~CGPbase() {
		// TODO Auto-generated destructor stub
	}

	void CGPbase::clearCache()
	{
		this->K=MatrixXd();
		this->Kinv=MatrixXd();
		this->KinvY=MatrixXd();
		this->cholK=Eigen::LDLT<gpmix::MatrixXd>();
		this->DKinv_KinvYYKinv = MatrixXd();
	}

	void CGPbase::set_data(MatrixXd& X, MatrixXd& Y, CGPHyperParams& hyperparams)
	{
		this->clearCache();
		this->X = X;
		this->Y = Y;
		this->params = hyperparams;
		//this->meanY = Y.colwise().mean();
	}


	MatrixXd CGPbase::getKinv()
	{
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
		if (this->DKinv_KinvYYKinv.cols()==0)
		{
			MatrixXd KiY = this->getKinvY();
			this->DKinv_KinvYYKinv = ((float_t)this->get_target_dimension())*this->getKinv() - KiY * KiY.transpose();//WARNING:conversion uint64_t to double
		}
		return this->DKinv_KinvYYKinv;
	}

	Eigen::LDLT<gpmix::MatrixXd> CGPbase::getCholK()
	{
		if (this->cholK.cols()==0)
		{
			this->cholK = Eigen::LDLT<gpmix::MatrixXd>(  this->getK() );
		}
		return this->cholK;
	}

	MatrixXd CGPbase::getK()
	{
		if (this->K.cols()==0)
		{
			this->K=this->covar.K(this->params.get("covar"),this->X); //This line breaks for se kernel
			this->lik.applyToK(this->params.get("lik"),this->K);
		}
		return this->K;
	}

//	void CGPbase::set_params(CGPHyperParams& hyperparams)
//	{
//		this->clearCache();
//		this->params = hyperparams;
//	}

	float_t CGPbase::LML()
	{
		//update the covariance parameters
		Eigen::LDLT<gpmix::MatrixXd>chol = CGPbase::getCholK();

		float_t lml_det = 0.0;
		for (uint_t i = 0; i<(uint_t)chol.vectorD().rows(); ++i)
		{
			lml_det+=gpmix::log((float_t)chol.vectorD()(i));//WARNING: float_t cast
		}
      
		float_t lml_quad = 0.0;
		MatrixXd KinvY = this->getKinvY();
		//loop over independent columns of Y:
		for (uint_t colY=0; colY<(uint_t)this->Y.cols();++colY)
		{
			lml_quad += this->Y.col(colY).transpose() * KinvY.col(colY);
		}

		float_t lml_const = this->Y.cols() * this->Y.rows() * gpmix::log((2.0 * PI));

		return 0.5 * (lml_quad + this->Y.cols() * lml_det + lml_const);
	}

	CGPHyperParams CGPbase::LMLgrad(){

		CGPHyperParams grad = CGPHyperParams();
		grad.set( "covar", this->LMLgrad_covar() );
		grad.set( "lik", this->LMLgrad_lik() );
		
		return grad;
	}

	MatrixXd CGPbase::LMLgrad_covar()
	{
		MatrixXd params_covar = this->params.get("covar");
		MatrixXd grad_covar = MatrixXd::Zero(params_covar.rows(),params_covar.cols());

		MatrixXd W = this->getDKinv_KinvYYKinv();
		for(uint_t row = 0 ; row<(uint_t)params_covar.rows(); ++row)//WARNING: conversion
		{
			for(uint_t col = 0 ; col<(uint_t)params_covar.cols(); ++col)//WARNING: conversion
			{
				MatrixXd Kd = this->covar.Kgrad_theta(params_covar, this->X, row*params_covar.cols() + col);
				grad_covar(row,col) = 0.5*(Kd.array() * W.array()).sum();
			}
		}
		return grad_covar;
	}

	MatrixXd CGPbase::LMLgrad_lik()
	{
		MatrixXd params_lik = this->params.get("lik");
		MatrixXd grad_lik(params_lik.rows(),params_lik.cols());

		MatrixXd W = this->getDKinv_KinvYYKinv();
		for(uint_t row = 0 ; row<(uint_t)params_lik.rows(); ++row)	//WARNING: conversion
		{
			for(uint_t col = 0 ; col<(uint_t)params_lik.cols(); ++col)	//WARNING: conversion
			{
				MatrixXd Kd = this->lik.K_grad_theta(params_lik, this->X, row*params_lik.cols() + col);
				grad_lik(row,col) = 0.5*(Kd.array() * W.array()).sum();
			}
		}
		return grad_lik;
	}

	MatrixXd CGPbase::predictMean(MatrixXd& Xstar)
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

} /* namespace gpmix */
