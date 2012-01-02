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

void CGPHyperParams::agetParamArray(VectorXd* out) const
{
	//1. get size
	(*out).resize(this->getNumberParams());
	//2. loop through entries
	muint_t ncurrent=0;
	for(CGPHyperParamsMap::const_iterator iter = this->begin(); iter!=this->end();iter++)
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
	for(CGPHyperParamsMap::const_iterator iter = this->begin(); iter!=this->end();iter++)
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

muint_t CGPHyperParams::getNumberParams() const
{
	// return effective number of params. Note that parameter entries can either be of matrix of column nature
	// thus sizes is rows*cols
	muint_t nparams=0;
	for(CGPHyperParamsMap::const_iterator iter = this->begin(); iter!=this->end();iter++)
	{
		MatrixXd value = (*iter).second;
		nparams+=value.rows()*value.cols();
	}
	return nparams;
}

void CGPHyperParams::set(const string& name, const MatrixXd& value)
{
	(*this)[name] = value;
}

void CGPHyperParams::aget(MatrixXd* out, const string& name)
{
	(*out) = (*this)[name];
}


vector<string> CGPHyperParams::getNames() const
{
	vector<string> rv(this->size());
	for(CGPHyperParamsMap::const_iterator iter = this->begin(); iter!=this->end();iter++)
	{
		rv.push_back((*iter).first);
	}
	return rv;
}

bool CGPHyperParams::exists(string name) const
{
	CGPHyperParamsMap::const_iterator iter = this->find(name);
	if (iter==this->end())
		return false;
	else
		return true;
}


/* CGPCholCache */

void CGPCholCache::clearCache()
{
	gp.covar.makeSync();
	gp.lik.makeSync();

	//set null:
	this->K=MatrixXd();
	this->Kinv=MatrixXd();
	this->KinvY=MatrixXd();
	this->cholK=MatrixXdChol();
	this->DKinv_KinvYYKinv = MatrixXd();
}


bool CGPCholCache::isInSync() const
{
	return (gp.covar.isInSync() && gp.lik.isInSync());
}

MatrixXd* CGPCholCache::getKinv()
{
	if (!isInSync())
		clearCache();
	if (isnull(Kinv))
	{
		MatrixXdChol* chol = this->getCholK();
		Kinv = MatrixXd::Identity(K.rows(),K.rows());
#if 0
		(*chol).solveInPlace(cache.Kinv);
#else
		//alterative
		MatrixXd L = chol->matrixL();
		L.triangularView<Eigen::Lower>().solveInPlace(Kinv);
		Kinv.transpose()*=Kinv.triangularView<Eigen::Lower>();
#endif
	}
	return (&Kinv);
}

MatrixXd* CGPCholCache::getKinvY()
{
	//Invalidate Cache?
	if (!isInSync())
		this->clearCache();

	if (isnull(KinvY))
	{
		MatrixXdChol* chol = this->getCholK();
		KinvY = (*chol).solve(gp.Y);
	}
	return &KinvY;
}

MatrixXd* CGPCholCache::getDKinv_KinvYYKinv()
{
	if (!isInSync())
		this->clearCache();
	if (isnull(DKinv_KinvYYKinv))
	{
		MatrixXd* KiY  = getKinvY();
		MatrixXd* Kinv = getKinv();
		DKinv_KinvYYKinv = ((mfloat_t)(gp.getNumberDimension())) * (*Kinv) - (*KiY) * (*KiY).transpose();
	}
	return &DKinv_KinvYYKinv;
}

MatrixXdChol* CGPCholCache::getCholK()
{
	if (!isInSync())
		this->clearCache();

	if (isnull(cholK))
	{
		cholK = MatrixXdChol((*this->getK()));
	}
	return &cholK;
}

MatrixXd* CGPCholCache::getK()
{
	if (!isInSync())
		this->clearCache();
	if (isnull(K))
	{
		gp.covar.aK(&K);
		K += gp.lik.K();
	}
	return &K;
}



/* CGPbase */

CGPbase::CGPbase(ACovarianceFunction& covar, ALikelihood& lik) : cache(*this),covar(covar), lik(lik) {
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

void CGPbase::updateX(ACovarianceFunction& covar,const VectorXi& gplvmDimensions,const MatrixXd& X) throw (CGPMixException)
{
	if (X.cols()!=gplvmDimensions.rows())
	{
		ostringstream os;
		os << "CGPLvm X param update dimension missmatch. X("<<X.rows()<<","<<X.cols()<<") <-> gplvm_dimensions:"<<gplvmDimensions.rows()<<"!";
		throw CGPMixException(os.str());
	}
	//update
	for (muint_t ic=0;ic<(muint_t)X.cols();ic++)
		covar.setXcol(X.col(ic),gplvmDimensions(ic));
}


void CGPbase::updateParams() throw(CGPMixException)
{
	if (this->params.exists("covar"))
		this->covar.setParams(this->params["covar"]);
	if (this->params.exists("lik"))
		this->lik.setParams(this->params["lik"]);
	if (params.exists("X"))
		this->updateX(covar,gplvmDimensions,params["X"]);
}

void CGPbase::setParams(const CGPHyperParams& hyperparams) throw(CGPMixException)
{
	this->params = hyperparams;
	updateParams();
}


CGPHyperParams CGPbase::getParams() const
{
	return this->params;
}

void CGPbase::setParamArray(const VectorXd& hyperparams) throw (CGPMixException)
{
	this->params.setParamArray(hyperparams);
	updateParams();
}
void CGPbase::agetParamArray(VectorXd* out) const
{
	this->params.agetParamArray(out);
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
	if (isnull(gplvmDimensions))
			this->gplvmDimensions = VectorXi::LinSpaced(X.cols(),0,X.cols()-1);
}


/* Marginal likelihood */

//wrappers:
mfloat_t CGPbase::LML(const CGPHyperParams& params) throw (CGPMixException)
{
	setParams(params);
	return LML();
}

mfloat_t CGPbase::LML(const VectorXd& params) throw (CGPMixException)
{
	setParamArray(params);
	return LML();
}

mfloat_t CGPbase::LML() throw (CGPMixException)
{
	//update the covariance parameters
	MatrixXdChol* chol = cache.getCholK();
	//logdet:
	mfloat_t lml_det  = 0.5*Y.cols()*logdet((*chol));
	//2. quadratic term
	mfloat_t lml_quad = 0.0;
	MatrixXd* KinvY = cache.getKinvY();
	//quadratic form
	lml_quad = 0.5*((*KinvY).array() * Y.array()).sum();
	//constants
	mfloat_t lml_const = 0.5*Y.cols() * Y.rows() * gpmix::log((2.0 * PI));
	return lml_quad + lml_det + lml_const;
};


/* Gradient interface functions:*/
void CGPbase::aLMLgrad(VectorXd* out,const CGPHyperParams& params) throw (CGPMixException)
		{
	setParams(params);
	aLMLgrad(out);
		}

void CGPbase::aLMLgrad(VectorXd* out,const VectorXd& paramArray) throw (CGPMixException)
		{
	setParamArray(paramArray);
	aLMLgrad(out);
		}

void CGPbase::aLMLgrad(VectorXd* out) throw (CGPMixException)
		{
	CGPHyperParams rv = LMLgrad();
	rv.agetParamArray(out);
		}

CGPHyperParams CGPbase::LMLgrad(const CGPHyperParams& params) throw (CGPMixException)
		{
	setParams(params);
	return LMLgrad();
		}
CGPHyperParams CGPbase::LMLgrad(const VectorXd& paramArray) throw (CGPMixException)
		{
	setParamArray(paramArray);
	return LMLgrad();
		}

/* Main routine: gradient calculation*/
CGPHyperParams CGPbase::LMLgrad() throw (CGPMixException)
{
	CGPHyperParams rv;
	//calculate gradients for parameter components in params:
	if (params.exists("covar"))
	{
		VectorXd grad_covar;
		aLMLgrad_covar(&grad_covar);
		rv.set("covar",grad_covar);
	}
	if (params.exists("lik"))
	{
		VectorXd grad_lik;
		aLMLgrad_lik(&grad_lik);
		rv.set("lik",grad_lik);
	}
	if (params.exists("X"))
		{
			MatrixXd grad_X;
			aLMLgrad_X(&grad_X);
			rv.set("X",grad_X);
		}
	return rv;
}


void CGPbase::aLMLgrad_covar(VectorXd* out) throw (CGPMixException)
				{
	//vector with results
	VectorXd grad_covar(covar.getNumberParams());
	//W:
	MatrixXd* W = cache.getDKinv_KinvYYKinv();
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
	MatrixXd* W = cache.getDKinv_KinvYYKinv();
	MatrixXd Kd;
	for(muint_t row = 0 ; row<lik.getNumberParams(); ++row)	//WARNING: conversion
	{
		lik.aKgrad_param(&Kd,row);
		grad_lik(row) = 0.5*(Kd.array() * (*W).array()).sum();
	}
	(*out) = grad_lik;
				}

void CGPbase::aLMLgrad_X(MatrixXd* out) throw (CGPMixException)
{
	//0. set output dimensions
	(*out).resize(this->getNumberSamples(),this->gplvmDimensions.rows());

	//1. get W:
	MatrixXd* W = cache.getDKinv_KinvYYKinv();
	//loop through GLVM dimensions and calculate gradient

	MatrixXd WKgrad_X;
	VectorXd Kdiag_grad_X;
	for (muint_t ic=0;ic<(muint_t)this->gplvmDimensions.rows();ic++)
	{
		muint_t col = gplvmDimensions(ic);
		//get gradient
		covar.aKgrad_X(&WKgrad_X,col);
		covar.aKdiag_grad_X(&Kdiag_grad_X,col);
		WKgrad_X.diagonal() = Kdiag_grad_X;
		//precalc elementwise product of W and K
		WKgrad_X.array()*=(*W).array();
		MatrixXd t = (2*WKgrad_X.rowwise().sum() - WKgrad_X.diagonal());
		(*out).col(ic) = 0.5* (2*WKgrad_X.rowwise().sum() - WKgrad_X.diagonal());
	}
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
