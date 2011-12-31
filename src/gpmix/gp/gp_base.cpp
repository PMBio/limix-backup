/*
 * gp_base.cpp
 *
 *  Created on: Nov 11, 2011
 *      Author: stegle
 */

#include "gp_base.h"
#include "gpmix/utils/matrix_helper.h"

#ifndef PI
#define PI 3.14159265358979323846
#endif

namespace gpmix {



CGPbase::CGPbase(ACovarianceFunction& covar, ALikelihood& lik) : covar(covar), lik(lik) {
	this->covar = covar;
	this->lik = lik;
	//this->clearCache();
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
	this->cache.clear();
	this->covar.makeSync();
	this->lik.makeSync();
}

bool CGPbase::isInSync() const
{
	return covar.isInSync() && lik.isInSync();
}

MatrixXd* CGPbase::getKinv()
{
	if (!isInSync())
		this->clearCache();
	if (isnull(cache.Kinv))
	{
		Eigen::LLT<gpmix::MatrixXd>* chol = this->getCholK();
		cache.Kinv = MatrixXd::Identity(this->get_samplesize(),this->get_samplesize());
		(*chol).solveInPlace(cache.Kinv);
		//for now
	}
	return (&this->cache.Kinv);
}

MatrixXd* CGPbase::getKinvY()
{
	//Invalidate Cache?
	if (!isInSync())
		this->clearCache();

	if (isnull(cache.KinvY))
	{
		Eigen::LLT<gpmix::MatrixXd>* chol = this->getCholK();
		cache.KinvY = (*chol).solve(this->Y);
	}
	return &cache.KinvY;
}

MatrixXd* CGPbase::getDKinv_KinvYYKinv()
{
	if (!isInSync())
		this->clearCache();
	if (isnull(cache.DKinv_KinvYYKinv))
	{
		MatrixXd* KiY  = getKinvY();
		MatrixXd* Kinv = getKinv();
		cache.DKinv_KinvYYKinv = ((mfloat_t)(this->get_target_dimension())) * (*Kinv) - (*KiY) * (*KiY).transpose();
	}
	return &cache.DKinv_KinvYYKinv;
}

Eigen::LLT<gpmix::MatrixXd>* CGPbase::getCholK()
{
	if (!isInSync())
		this->clearCache();

	if (isnull(cache.cholK))
	{
		cache.cholK = Eigen::LLT<gpmix::MatrixXd>((*this->getK()));
	}
	return &cache.cholK;
}

MatrixXd* CGPbase::getK()
{
	if (!isInSync())
		this->clearCache();
	if (isnull(cache.K))
	{
		covar.aK(&cache.K);
		cache.K += lik.K();
	}
	return &cache.K;
}


mfloat_t CGPbase::LML()
{
	//update the covariance parameters
	Eigen::LLT<gpmix::MatrixXd>* chol = getCholK();
	//1. logdet
	mfloat_t lml_det  = 0.5*Y.cols()*logdet((*chol));
	//2. quadratic term
	mfloat_t lml_quad = 0.0;
	MatrixXd* KinvY = this->getKinvY();
	//quadratic form
	lml_quad = 0.5*((*KinvY).array() * Y.array()).sum();
	//constants
	mfloat_t lml_const = 0.5*Y.cols() * Y.rows() * gpmix::log((2.0 * PI));
	return lml_quad + lml_det + lml_const;
}

void CGPbase::aLMLgrad_covar(VectorXd* out)
{
	//vector with results
	VectorXd grad_covar(covar.getNumberParams());
	//W:
	MatrixXd* W = this->getDKinv_KinvYYKinv();
	//Kd cachine result
	MatrixXd Kd;
	for(muint_t param = 0;param < (muint_t)(grad_covar.rows());param++){
		covar.aKgrad_param(&Kd,param);
		grad_covar(param) = 0.5 * (Kd.array() * (*W).array()).sum();
	}
	(*out) = grad_covar;
}

void CGPbase::agetY(MatrixXd* out) const
{
	(*out) = Y;
}

void CGPbase::setY(const MatrixXd& Y)
{
	this->Y = Y;
}

void CGPbase::aLMLgrad_lik(VectorXd* out)
{
	LikParams grad_lik(lik.getNumberParams());
	MatrixXd* W = this->getDKinv_KinvYYKinv();
	MatrixXd Kd;
	for(muint_t row = 0 ; row<lik.getNumberParams(); ++row)	//WARNING: conversion
	{
		lik.aKgrad_param(&Kd,row);
		grad_lik(row) = 0.5*(Kd.array() * (*W).array()).sum();
	}
	(*out) = grad_lik;
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
