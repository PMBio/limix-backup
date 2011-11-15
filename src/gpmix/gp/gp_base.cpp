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
	}

	CGPbase::~CGPbase() {
		// TODO Auto-generated destructor stub
	}

	void CGPbase::set_data(MatrixXd X, MatrixXd Y)
	{
		this->X = X;
		this->Y = Y;
		this->meanY = Y.colwise().mean();
	}

	bool CGPCache::is_cached(CGPHyperParams params)
	{
		return false;
	}

	CGPCache::CGPCache()
	{
	}

	CGPCache::~CGPCache()
	{
	}

	MatrixXd CGPCache::Kinv(CGPHyperParams params, MatrixXd X, bool check_passed = false, bool is_checked = false)
	{
		return MatrixXd(1,1);
	}

	MatrixXd CGPCache::KinvY(CGPHyperParams params, MatrixXd X, MatrixXd Y, bool check_passed = false, bool is_checked = false)
	{
		return MatrixXd(1,1);
	}

	Eigen::LDLT<gpmix::MatrixXd> CGPCache::CholK(CGPHyperParams params, MatrixXd X, bool check_passed = false, bool is_checked = false)
	{
		return 0;
	}

	void CGPCache::clear()
	{
		this->cachedparams.clear();
	}

	void CGPbase::getCovariance(CGPHyperParams& hyperparams)
	{
		if ((!this->is_cached(Hyperparams)) || (this->active_set_indices_changed) )
		{
			MatrixXd K = this->covar.K(hyperparams.get("covar"), this->X);
			this->cache_cov->lik.applyToK(hyperparams.get("lik"), K);
			this->cache_cov->chol = Eigen::LDLT<gpmix::MatrixXd>(K);
			this->cache_cov->KinvY = this->chol.solve(this->Y);
		}

	}

	float_t CGPbase::LML(CGPHyperParams& hyperparams)
	{
		//update the covariance parameters
		this->getCovariances(hyperparams);

		float_t lml_det = 0.0;
		for (uint_t i = 0; i<(uint_t)chol.vectorD().rows(); ++i)
		{
			lml_det+=gpmix::log((float_t)this->chol.vectorD()(i));//WARNING: float_t cast
		}
      
		float_t lml_quad = 0.0;
		//loop over independent columns of Y:
		for (uint_t colY=0; colY<(uint_t)this->Y.cols();++colY)
		{
			lml_quad += this->Y.col(colY).transpose() * KinvY.col(colY);
		}

		float_t lml_const = this->Y.cols() * this->Y.rows() * gpmix::log((2.0 * PI));

		return 0.5 * (lml_quad + this->Y.cols() * lml_det + lml_const);
	}

	CGPHyperParams CGPbase::LMLgrad(CGPHyperParams& hyperparams){
		//update the covariance parameters
		this->getCovariances(hyperparams);

		CGPHyperParams grad;
		return grad;
	}

} /* namespace gpmix */
