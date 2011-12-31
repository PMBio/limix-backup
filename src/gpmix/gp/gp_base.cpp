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


/* CGPHyperParmas */

void CGPHyperParams::agetParamArray(VectorXd* out)
{
	//1. get size
	(*out).resize(this->getNumberParams());
	//2. loop through entries
	muint_t ncurrent=0;
	for(CGPHyperParamsMap::const_iterator iter = param_map.begin(); iter!=param_map.end();iter++)
	{
		//1. get param object which can be a matrix or array:
		MatrixXd value = (*iter).second;
		muint_t nc = value.rows()*value.cols();
		//2. flatten to vector shape
		value.resize(nc,1);
		//3. add to vector
		(*out).segment(ncurrent,nc) = value;
		//4. increase pointer
		ncurrent += nc;
	}
}

void CGPHyperParams::setParamArray(const VectorXd& param) throw (CGPMixException)
{
	//1. check that param has correct shape
	if ((muint_t)param.rows()!=getNumberParams())
	{
		ostringstream os;
		os << "Wrong number of params HyperParams structure (HyperParams structure:"<<getNumberParams()<<", paramArray:"<<param.rows()<<")!";
		throw gpmix::CGPMixException(os.str());
	}

	//2. loop through elements and slot in params
	muint_t ncurrent=0;
	for(CGPHyperParamsMap::const_iterator iter = param_map.begin(); iter!=param_map.end();iter++)
	{
		string name = (*iter).first;
		muint_t nc = (*iter).second.rows()*(*iter).second.cols();
		//get  elements:
		MatrixXd value = param.segment(ncurrent,nc);
		//reshape
		value.resize((*iter).second.rows(),(*iter).second.cols());
		//set
		set(name,value);
		//move on
		ncurrent += nc;
	}
}

muint_t CGPHyperParams::getNumberParams()
{
	// return effective number of params. Note that parameter entries can either be of matrix of column nature
	// thus sizes is rows*cols
	muint_t nparams=0;
	for(CGPHyperParamsMap::const_iterator iter = param_map.begin(); iter!=param_map.end();iter++)
	{
		MatrixXd value = (*iter).second;
		nparams+=value.rows()*value.cols();
	}
	return nparams;
}

void CGPHyperParams::set(const string& name, const MatrixXd& value)
{
	param_map[name]=value;
}

void CGPHyperParams::aget(MatrixXd* out, const string& name)
{
	(*out) = param_map[name];
}

void CGPHyperParams::clear()
{
	this->param_map.clear();
}

void CGPHyperParams::agetNames(VectorXs* out)
{
	(*out).resize(param_map.size());
	map<string,MatrixXd>::iterator it=param_map.begin();
	for (muint_t i=0; i<param_map.size();++i)
	{
		(*out)(i)=(*it).first;
		++it;
	}
}


/* CGPbase */

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

void CGPbase::set_params(const CGPHyperParams& hyperparams)
{
	this->params = hyperparams;
	//update individual components
	//TODO:
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
		cache.Kinv = MatrixXd::Identity(this->getNumberSamples(),this->getNumberSamples());
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
		cache.DKinv_KinvYYKinv = ((mfloat_t)(this->getNumberDimension())) * (*Kinv) - (*KiY) * (*KiY).transpose();
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





void CGPbase::agetY(MatrixXd* out) const
{
	(*out) = Y;
}

void CGPbase::setY(const MatrixXd& Y)
{
	this->Y = Y;
}

void CGPbase::agetX(CovarInput* out) const
{
	this->covar.agetX(out);
}
void CGPbase::setX(const CovarInput& X) throw (CGPMixException)
{
	//use covariance to set everything
	this->covar.setX(X);
	this->lik.setX(X);
}


/* Marginal likelihood */
mfloat_t CGPbase::LML() throw (CGPMixException)
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

/*Gradients:*/

void CGPbase::aLMLgrad_covar(VectorXd* out) throw (CGPMixException)
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


void CGPbase::aLMLgrad_lik(VectorXd* out) throw (CGPMixException)
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

void CGPbase::aLMLgrad(VectorXd* out) throw (CGPMixException)
{

}


void CGPbase::apredictMean(MatrixXd* out, const MatrixXd& Xstar) throw (CGPMixException)
{
	/*
	MatrixXd KstarCross = covar.
	return KstarCross * this->getKinvY();
	*/
}

void CGPbase::apredictVar(MatrixXd* out,const MatrixXd& Xstar) throw (CGPMixException)
{
	/*
	MatrixXd KstarDiag = this->covar.Kdiag(this->params.get("covar"), Xstar);
	KstarDiag+=this->lik.Kdiag(this->params.get("lik"), Xstar);
	MatrixXd Kcross = this->covar.K(this->params.get("covar"), this->X, Xstar);
	MatrixXd v = this->getCholK().solve(Kcross);
	MatrixXd vv = (v.array()*v.array()).matrix().colwise().sum();
	MatrixXd S2 = KstarDiag - vv.transpose();
	return S2;
	*/
}

} /* namespace gpmix */
