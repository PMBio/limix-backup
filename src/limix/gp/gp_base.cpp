/*
 * gp_base.cpp
 *
 *  Created on: Nov 11, 2011
 *      Author: stegle
 */

#include "gp_base.h"
#include "limix/mean/CLinearMean.h"
#include "limix/utils/matrix_helper.h"
#include "limix/utils/logging.h"
#include <sstream>

namespace limix {


/* CGPHyperParmas */

CGPHyperParams::CGPHyperParams(const CGPHyperParams &_param) : std::map<std::string,MatrixXd>(_param)
{

}

void CGPHyperParams::agetParamArray(VectorXd* out) const throw(CGPMixException)
{
	CGPHyperParams mask;
	agetParamArray(out,mask);
}

void CGPHyperParams::setParamArray(const VectorXd& param) throw (CGPMixException)
{
	CGPHyperParams mask;
	setParamArray(param,mask);
}


MatrixXd CGPHyperParams::filterMask(const MatrixXd& array,const MatrixXd& mask) const
{
	//TODO: mask currently only supported for rows of array not on cols:
	MatrixXd RV;
	AfilterMask(RV,array,mask.col(0),VectorXd::Ones(array.cols()));
	return RV;
}

void CGPHyperParams::expandMask(MatrixXd& out, const MatrixXd& array,const MatrixXd& mask) const
{
	//TODO: mask currently only supported for rows of array not on cols:
	AexpandMask(out,array,mask.col(0),VectorXd::Ones(array.cols()));
}

void CGPHyperParams::agetParamArray(VectorXd* out,const CGPHyperParams& mask) const throw(CGPMixException)
{
	//1. get size
	(*out).resize(this->getNumberParams(mask));
	//2. loop through entries
	muint_t ncurrent=0;
	for(CGPHyperParams::const_iterator iter = this->begin(); iter!=this->end();iter++)
	{
		//1. get param object which can be a matrix or array:
		MatrixXd value = (*iter).second;
		std::string name = (*iter).first;
		//2. check whether filter applied for this field
		CGPHyperParams::const_iterator itf = mask.find(name);
		if (itf!=mask.end())
			value=filterMask(value,(*itf).second);
		muint_t nc = value.rows()*value.cols();
		//2. flatten to vector shape
		value.resize(nc,1);
		//3. add to vector
		(*out).segment(ncurrent,nc) = value;
		//4. increase pointer
		ncurrent += nc;
	}
}

void CGPHyperParams::setParamArray(const VectorXd& param,const CGPHyperParams& mask) throw (CGPMixException)
{
	//1. check that param has correct shape
	if ((muint_t)param.rows()!=getNumberParams(mask))
	{
		std::ostringstream os;
		os << "Wrong number of params HyperParams structure (HyperParams structure:"<<getNumberParams()<<", paramArray:"<<param.rows()<<")!";
		throw CGPMixException(os.str());
	}

	muint_t nc;
	MatrixXd value;
	//2. loop through elements and slot in params
	muint_t ncurrent=0;
	for(CGPHyperParams::const_iterator iter = this->begin(); iter!=this->end();iter++)
	{
		std::string name = (*iter).first;
		CGPHyperParams::const_iterator itf = mask.find(name);
		if (itf!=mask.end())
		{
			const MatrixXd& mask_ = (*itf).second;
			//number of elements depends on mask:
			muint_t vr = mask_.col(0).count();
			//TODO: masking of columns not supported
			muint_t vc = (*iter).second.cols();
			nc = vr*vc;
			//get value and resize
			MatrixXd value_ = param.segment(ncurrent,nc);
			value_.resize(vr,vc);
			//expand mask:
			value  = this->get(name);
			//expand
			expandMask(value,value_,mask_);
		}
		else
		{
			nc = (*iter).second.rows()*(*iter).second.cols();
			//get  elements:
			value = param.segment(ncurrent,nc);
			//reshape
			value.resize((*iter).second.rows(),(*iter).second.cols());
			//set
			set(name,value);
		}
		set(name,value);
		//std::cout << "name:" << name << "value:" << value << "\n\n";
		//move on
		ncurrent += nc;
	}//end for

}

muint_t CGPHyperParams::getNumberParams() const
{
	CGPHyperParams mask;
	return getNumberParams(mask);
}

muint_t CGPHyperParams::getNumberParams(const CGPHyperParams& mask) const
{
	muint_t nparams=0;
	for(CGPHyperParams::const_iterator iter = this->begin(); iter!=this->end();iter++)
	{
		std::string name = (*iter).first;
		MatrixXd value = (*iter).second;
		CGPHyperParams::const_iterator itf = mask.find(name);
		if (itf!=mask.end())
		{
			//mask case: number of parameters are determined by mask:
			const MatrixXd& mask_ = (*itf).second;
			//TODO: only support for row filter
			nparams += mask_.col(0).count()*value.cols();
		}
		else
		{
			//no mask for this value
			nparams+=value.rows()*value.cols();
		}
	}
	return nparams;
}


void CGPHyperParams::set(const std::string& name, const MatrixXd& value)
{
	(*this)[name] = value;
}

void CGPHyperParams::aget(MatrixXd* out, const std::string& name)
{
	(*out) = (*this)[name];
}


std::vector<std::string> CGPHyperParams::getNames() const
{
	std::vector<std::string> rv(this->size());
	for(CGPHyperParams::const_iterator iter = this->begin(); iter!=this->end();iter++)
	{
		rv.push_back((*iter).first);
	}
	return rv;
}

bool CGPHyperParams::exists(std::string name) const
{
	CGPHyperParams::const_iterator iter = this->find(name);
	if (iter==this->end())
		return false;
	else
		return true;
}

std::string CGPHyperParams::toString() const
{
	std::ostringstream os;
	os<<(*this);
	return os.str();
}

std::ostream& operator <<(std::ostream &os,const CGPHyperParams &obj)
{
	for(CGPHyperParams::const_iterator iter = obj.begin(); iter!=obj.end();iter++)
	{
		os << (*iter).first << ":" << "\n";
		os << (*iter).second << "\n\n";
	}
	return os;
}



/* CGPCholCache */
CGPCholCache::CGPCholCache(CGPbase* gp)
{
	this->gp = gp;
	syncLik = Pbool(new bool);
	syncData = Pbool(new bool);
	syncCovar = Pbool(new bool);


	this->covar = PCovarianceFunctionCache(new CCovarianceFunctionCache(gp->covar));
	gp->lik->addSyncChild(syncLik);
	covar->addSyncChild(syncCovar);
	gp->dataTerm->addSyncChild(syncData);
	//own sync state depends on lik, covar & dataterm
	addSyncParent(syncLik);
	addSyncParent(syncCovar);
	addSyncParent(syncData);
	//set all cache varibles to Null
	this->KEffCacheNull=true;
	this->KEffCholNull=true;
	this->KEffInvCacheNull=true;
	this->DKinv_KEffInvYYKEffInvCacheNull=true;
	this->KEffInvYCacheNull=true;
	this->YeffectiveCacheNull=true;
};


void CGPCholCache::setCovar(PCovarianceFunction covar)
{
	this->covar->setCovar(covar);
}


void CGPCholCache::validateCache()
{
	//std::cout << *syncCovar << "," << *syncData << "," << *syncLik << "\n";
	//1. variables that depend on any of the caches
	if ((! *syncCovar) || (! *syncData) || (! *syncLik))
	{
		KEffInvYCacheNull =true;
		DKinv_KEffInvYYKEffInvCacheNull = true;
	}
	//2. varaibles that depend on covar or lik
	if ((! *syncCovar) || (! *syncLik))
	{
		KEffCacheNull = true;
		KEffCholNull = true;
		KEffInvCacheNull = true;
	}
	//3. variables that on data term only
	if (! *syncData)
	{
		YeffectiveCacheNull = true;
	}
	//restore sync
	setSync();
	//std::cout << *syncCovar << "," << *syncData << "," << *syncLik << "\n";
}


MatrixXd& CGPCholCache::rgetKEff()
{
	validateCache();
	if(KEffCacheNull)
	{
		KEffCache  = covar->rgetK();
		KEffCache += gp->lik->K();
		KEffCacheNull=false;
	}
	return KEffCache;
}


MatrixXdChol& CGPCholCache::rgetKEffChol()
{
	validateCache();
	if(KEffCholNull)
	{
		KEffCholCache = MatrixXdChol(rgetKEff());
		KEffCholNull=false;
	}
	return this->KEffCholCache;
}


MatrixXd& CGPCholCache::rgetKEffInv()
{
	validateCache();
	if (KEffInvCacheNull)
	{
		MatrixXdChol& chol = rgetKEffChol();
		KEffInvCache = MatrixXd::Identity(chol.rows(),chol.rows());
		//faster alternative for chol.SolvInPlace(caache.Kinv)
		MatrixXd L = chol.matrixL();
		L.triangularView<Eigen::Lower>().solveInPlace(KEffInvCache);
		KEffInvCache.transpose()*=KEffInvCache.triangularView<Eigen::Lower>();
		KEffInvCacheNull=false;
	}
	return KEffInvCache;
}

MatrixXd& CGPCholCache::rgetYeffective()
{
	//Invalidate Cache?
	validateCache();
	if (YeffectiveCacheNull)
	{
		YeffectiveCache = gp->dataTerm->evaluate();
		YeffectiveCacheNull=false;
	}
	return YeffectiveCache;
}

MatrixXd& CGPCholCache::rgetKEffInvY()
{
	//Invalidate Cache?
	validateCache();
	if (KEffInvYCacheNull)
	{
		KEffInvYCache = rgetKEffChol().solve(this->rgetYeffective());
		KEffInvYCacheNull=false;
	}
	return KEffInvYCache;
}

MatrixXd& CGPCholCache::getDKEffInv_KEffInvYYKinv()
{
	//Invalidate Cache?
	validateCache();
	if (DKinv_KEffInvYYKEffInvCacheNull)
	{
		MatrixXd& KiY  = rgetKEffInvY();
		MatrixXd& Kinv = rgetKEffInv();
		DKinv_KEffinvYYKEffinvCache = ((mfloat_t)(gp->getNumberDimension())) * (Kinv) - (KiY) * (KiY).transpose();
		DKinv_KEffInvYYKEffInvCacheNull=false;
	}
	return DKinv_KEffinvYYKEffinvCache;
}



/* CGPbase */
CGPbase::CGPbase(PCovarianceFunction covar, PLikelihood lik,PDataTerm dataTerm)
{
	this->covar = covar;
	if(!dataTerm)
		this->dataTerm = PDataTerm(new CData());
	else
		this->dataTerm = dataTerm;
	if(!lik)
	{
		this->lik = PLikelihood(new CLikNormalIso());
	}
	else
		this->lik = lik;

	//init cache
	this->cache = PGPCholCache(new CGPCholCache(this));
}


void CGPbase::setCovar(PCovarianceFunction covar)
{
	this->covar = covar;
	this->cache->setCovar(covar);
}
void CGPbase::setLik(PLikelihood lik)
{
	this->lik = lik;
}

void CGPbase::setDataTerm(PDataTerm dataTerm)
{
	this->dataTerm = dataTerm;
}



CGPbase::~CGPbase()
{
	//check whether we need to fee dataTerm, Likelihood or Covar
	//TODO
}

void CGPbase::set_data(MatrixXd& Y)
{
	this->dataTerm->setY(Y);
}

void CGPbase::updateX(ACovarianceFunction& covar,const VectorXi& gplvmDimensions,const MatrixXd& X) throw (CGPMixException)
{
	if (X.cols()!=gplvmDimensions.rows())
	{
		std::ostringstream os;
		os << "CGPLvm X param update dimension missmatch. X("<<X.rows()<<","<<X.cols()<<") <-> gplvm_dimensions:"<<gplvmDimensions.rows()<<"!";
		throw CGPMixException(os.str());
	}
	//update
	for (muint_t ic=0;ic<(muint_t)X.cols();ic++)
	{
		covar.setXcol(X.col(ic),gplvmDimensions(ic));
	}
}


void CGPbase::updateParams() throw(CGPMixException)
{
	if (this->params.exists("covar"))
		this->covar->setParams(this->params["covar"]);
	if (this->params.exists("lik"))
		this->lik->setParams(this->params["lik"]);
	if (params.exists("X"))
		this->updateX(*covar,gplvmDimensions,params["X"]);
	if (params.exists("dataTerm"))
		this->dataTerm->setParams(this->params["dataTerm"]);
}

void CGPbase::setParams(const CGPHyperParams& hyperparams) throw(CGPMixException)
{
	this->params = hyperparams;
	updateParams();
}

void CGPbase::setParams(const CGPHyperParams& hyperparams,const CGPHyperParams& mask) throw(CGPMixException)
{
	//TODO: implemenation missing
	//std::cout << "implement me" << "\n";
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

void CGPbase::setParamArray(const VectorXd& hyperparams,const CGPHyperParams& mask) throw (CGPMixException)
{
	this->params.setParamArray(hyperparams,mask);
	updateParams();
}


void CGPbase::agetParamArray(VectorXd* out) const
{
	this->params.agetParamArray(out);
}

void CGPbase::agetY(MatrixXd* out)
{
	(*out) = this->cache->rgetYeffective();
}

void CGPbase::setY(const MatrixXd& Y)
{
	this->dataTerm->setY(Y);
	this->lik->setX(MatrixXd::Zero(Y.rows(),0));
}


void CGPbase::agetX(CovarInput* out) const
{
	this->covar->agetX(out);
}
void CGPbase::setX(const CovarInput& X) throw (CGPMixException)
{
	//use covariance to set everything
	this->covar->setX(X);
    this->lik->setX(X);
	if (isnull(gplvmDimensions))
	{
		if (X.cols()==1)
			//special case for a single dimensions...
			this->gplvmDimensions = VectorXi::Zero(1);
		else
			this->gplvmDimensions = VectorXi::LinSpaced(X.cols(),0,X.cols()-1);
	}
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
	MatrixXdChol& chol = cache->rgetKEffChol();

	//get effective Y:
	MatrixXd& Yeff = cache->rgetYeffective();
	//cout << Yeff<<endl;
	//log-det:
	mfloat_t lml_det  = 0.5* (Yeff).cols()*logdet((chol));

	//2. quadratic term
	mfloat_t lml_quad = 0.0;
	MatrixXd& KinvY = cache->rgetKEffInvY();
	//quadratic form
	lml_quad = 0.5*((KinvY).array() * (Yeff).array()).sum();

	//sum of the log-Jacobian term
	mfloat_t logJac = this->dataTerm->sumJacobianGradParams().sum();

	//constants
	mfloat_t lml_const = 0.5 * (Yeff).cols() * (Yeff).rows() * log((2.0 * PI));
	return lml_quad + lml_det + lml_const - logJac;
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
	if (params.exists("dataTerm"))
	{
		MatrixXd grad_dataTerm;
		aLMLgrad_dataTerm(&grad_dataTerm);
		rv.set("dataTerm",grad_dataTerm);
	}
	return rv;
}


void CGPbase::aLMLgrad_covar(VectorXd* out) throw (CGPMixException)
{
	//vector with results
	VectorXd grad_covar(covar->getNumberParams());
	//W:
	MatrixXd& W = cache->getDKEffInv_KEffInvYYKinv();
	//Kd cachine result
	MatrixXd Kd;
	for(muint_t param = 0;param < (muint_t)(grad_covar.rows());param++){
		covar->aKgrad_param(&Kd,param);
		grad_covar(param) = 0.5 * (Kd.array() * (W).array()).sum();
	}
	(*out) = grad_covar;
}


void CGPbase::aLMLgrad_lik(VectorXd* out) throw (CGPMixException)
{
	LikParams grad_lik(lik->getNumberParams());
	MatrixXd& W = cache->getDKEffInv_KEffInvYYKinv();
	MatrixXd Kd;
	for(muint_t row = 0 ; row<lik->getNumberParams(); ++row)	//WARNING: conversion
	{
		lik->aKgrad_param(&Kd,row);
		grad_lik(row) = 0.5*(Kd.array() * (W).array()).sum();
	}
	(*out) = grad_lik;
}

void CGPbase::aLMLgrad_X(MatrixXd* out) throw (CGPMixException)
{
	//0. set output dimensions
	(*out).resize(this->getNumberSamples(),this->gplvmDimensions.rows());

	//1. get W:
	MatrixXd& W = cache->getDKEffInv_KEffInvYYKinv();
	//loop through GLVM dimensions and calculate gradient

	MatrixXd WKgrad_X;
	VectorXd Kdiag_grad_X;
	for (muint_t ic=0;ic<(muint_t)this->gplvmDimensions.rows();ic++)
	{
		muint_t col = gplvmDimensions(ic);
		//get gradient
		covar->aKgrad_X(&WKgrad_X,col);
		covar->aKdiag_grad_X(&Kdiag_grad_X,col);
		WKgrad_X.diagonal() = Kdiag_grad_X;
		//precalc elementwise product of W and K
		WKgrad_X.array()*=(W).array();
		MatrixXd t = (2*WKgrad_X.rowwise().sum() - WKgrad_X.diagonal());
		(*out).col(ic) = 0.5* (2*WKgrad_X.rowwise().sum() - WKgrad_X.diagonal());
	}
}

void CGPbase::aLMLgrad_dataTerm(MatrixXd* out) throw (CGPMixException)
{
	//0. set output dimensions
	(*out) = dataTerm->gradParams(this->cache->rgetKEffInvY());
}
    

/* Main routine: Hessian calculation*/
void CGPbase::aLMLhess(MatrixXd* out, stringVec vecLabels) throw (CGPMixException)
{
        
    //Checks whether there are ripetions
    bool redundancy=0;
    for(muint_t i=0; i<vecLabels.size(); i++)
        for(muint_t j=i+1; j<vecLabels.size(); j++)
            if (vecLabels.at(i)==vecLabels.at(j)) redundancy=1;
    if (redundancy==1)   throw CGPMixException("Ripetition not allowed");
        
        
    //Checks if the labels are appropriate and extabilish the dimensions of the hessian
    muint_t hess_dimens=0;
    std::string sp1;
    for(stringVec::const_iterator iter = vecLabels.begin(); iter!=vecLabels.end();iter++)
    {
        sp1 = iter[0];
        if (sp1=="covar")           hess_dimens+=covar->getNumberParams();
        else if (sp1=="lik")        hess_dimens+=lik->getNumberParams();
        else if (sp1=="X")          throw CGPMixException("Not implemented");
        else if (sp1=="dataTerm")   throw CGPMixException("Not implemented");
        else                        throw CGPMixException("Hyperparameter list not valid");
    }
        
    (*out).resize(hess_dimens,hess_dimens);
    //(*out)=MatrixXd::Zero(hess_dimens,hess_dimens);
    
    muint_t i=0;
    muint_t j=0;
    muint_t j1=0;
    
    std::string sp2;
    MatrixXd hess_part;
    for(stringVec::const_iterator iter1 = vecLabels.begin(); iter1!=vecLabels.end();iter1++) {
        sp1 = iter1[0];
        if (sp1=="covar") {
            aLMLhess_covar(&hess_part);
            (*out).block(i,j,covar->getNumberParams(),covar->getNumberParams())=hess_part;
            j+=covar->getNumberParams();
            j1=j;
        }
        else if (sp1=="lik") {
            aLMLhess_lik(&hess_part);
            (*out).block(i,j,lik->getNumberParams(),lik->getNumberParams())=hess_part;
            j+=lik->getNumberParams();
            j1=j;
        }
        for(stringVec::const_iterator iter2 = iter1; iter2!=vecLabels.end();iter2++) {
            sp2 = iter2[0];
            if (sp1=="covar" && sp2=="lik") {
                aLMLhess_covarlik(&hess_part);
                (*out).block(i,j1,covar->getNumberParams(),lik->getNumberParams())=hess_part;
                (*out).block(j1,i,lik->getNumberParams(),covar->getNumberParams())=hess_part.transpose();
                j1+=covar->getNumberParams();
            }
            else if (sp1=="lik" && sp2=="covar") {
                aLMLhess_covarlik(&hess_part);
                (*out).block(i,j1,lik->getNumberParams(),covar->getNumberParams())=hess_part.transpose();
                (*out).block(j1,i,covar->getNumberParams(),lik->getNumberParams())=hess_part;
                j1+=covar->getNumberParams();
            }
        }
        i=j;
    }
    
}
    
void CGPbase::aLMLhess_covar(MatrixXd* out) throw (CGPMixException)
{
    //set output dimensions
    (*out).resize(covar->getNumberParams(),covar->getNumberParams());
    //W:
    MatrixXd& W = cache->getDKEffInv_KEffInvYYKinv();
    //KyInv:
    MatrixXd& KyInv = cache->rgetKEffInv();
    //KyInvY=alpha e alpha*alpha.T:
    MatrixXd& alpha = cache->rgetKEffInvY();
    MatrixXd alpha2 = (alpha)*(alpha).transpose();    //This might be included in CGPCholCache
    
    //Kd cachine result
    MatrixXd Khess_param;
    MatrixXd Kgrad_param;
    MatrixXd T;
    for(muint_t i = 0; i<(muint_t)(covar->getNumberParams()); i++) {
        covar->aKgrad_param(&Kgrad_param,i);
        T=KyInv*Kgrad_param*alpha2+alpha2*Kgrad_param*KyInv-KyInv*Kgrad_param*KyInv;
        for(muint_t j = 0; j<(muint_t)(covar->getNumberParams());j++) {
            covar->aKgrad_param(&Kgrad_param,j);
            covar->aKhess_param(&Khess_param,i,j);
            (*out)(i,j) = (T.array()*Kgrad_param.array()+W.array()*Khess_param.array()).sum();
        }
    }
    (*out)*=0.5;
}
    
void CGPbase::aLMLhess_lik(MatrixXd* out) throw (CGPMixException)
{
    //set output dimensions
    (*out).resize(lik->getNumberParams(),lik->getNumberParams());
    //W:
    MatrixXd& W = cache->getDKEffInv_KEffInvYYKinv();
    //KyInv:
    MatrixXd& KyInv = cache->rgetKEffInv();
    //KyInvY=alpha e alpha*alpha.T:
    MatrixXd& alpha = cache->rgetKEffInvY();
    MatrixXd alpha2 = (alpha)*(alpha).transpose();    //This might be included in CGPCholCache
    
    //Kd cachine result
    MatrixXd Khess_param;
    MatrixXd Kgrad_param;
    MatrixXd T;
    for(muint_t i = 0; i<(muint_t)(lik->getNumberParams()); i++) {
        lik->aKgrad_param(&Kgrad_param,i);
        T=KyInv*Kgrad_param*alpha2+alpha2*Kgrad_param*KyInv-KyInv*Kgrad_param*KyInv;
        for(muint_t j = 0; j<(muint_t)(lik->getNumberParams());j++) {
            lik->aKgrad_param(&Kgrad_param,j);
            lik->aKhess_param(&Khess_param,i,j);
            (*out)(i,j) = (T.array()*Kgrad_param.array()+W.array()*Khess_param.array()).sum();
        }
    }
    (*out)*=0.5;
}
    
void CGPbase::aLMLhess_covarlik(MatrixXd* out) throw (CGPMixException)
{
    //set output dimensions
    (*out).resize(covar->getNumberParams(),lik->getNumberParams());
    //KyInv:
    MatrixXd& KyInv = cache->rgetKEffInv();
    //KyInvY=alpha e alpha*alpha.T:
    MatrixXd& alpha = cache->rgetKEffInvY();
    MatrixXd alpha2 = (alpha)*(alpha).transpose();    //This might be included in CGPCholCache
    
    //Kd cachine result
    MatrixXd Khess_param;
    MatrixXd Kgrad_param;
    MatrixXd T;
    for(muint_t i = 0; i<(muint_t)(covar->getNumberParams()); i++) {
        covar->aKgrad_param(&Kgrad_param,i);
        T=KyInv*Kgrad_param*alpha2+alpha2*Kgrad_param*KyInv-KyInv*Kgrad_param*KyInv;
        for(muint_t j = 0; j<(muint_t)(lik->getNumberParams());j++) {
            lik->aKgrad_param(&Kgrad_param,j);
            (*out)(i,j) = (T.array()*Kgrad_param.array()).sum();
        }
    }
    (*out)*=0.5;
}
    
void CGPbase::agetCov_laplace(MatrixXd* out, stringVec vecLabels) throw (CGPMixException)
{
    MatrixXd hess;
    aLMLhess(&hess,vecLabels);
    (*out)=hess.inverse();
}
    
CGPHyperParams CGPbase::agetStd_laplace() throw (CGPMixException)
{
    MatrixXd Sigma;
    stringVec sv;
    sv.push_back("covar");
    sv.push_back("lik");
    agetCov_laplace(&Sigma,sv);
    VectorXd std=Sigma.diagonal().unaryExpr(std::ptr_fun(sqrt));
    CGPHyperParams out;
    out["covar"]=std.block(0,0,this->covar->getNumberParams(),1);
    out["lik"]=std.block(this->covar->getNumberParams(),0,this->lik->getNumberParams(),1);
    return out;
}
    

void CGPbase::apredictMean(MatrixXd* out, const MatrixXd& Xstar) throw (CGPMixException)
{
	MatrixXd KstarCross;
	this->covar->aKcross(&KstarCross,Xstar);
	MatrixXd& KinvY = this->cache->rgetKEffInvY();
	(*out).noalias() = KstarCross * KinvY;
}

void CGPbase::apredictVar(MatrixXd* out,const MatrixXd& Xstar) throw (CGPMixException)
{
	MatrixXd KstarCross;
	this->covar->aKcross(&KstarCross,Xstar);
	VectorXd KstarDiag;
	//get self covariance
	this->covar->aKcross_diag(&KstarDiag,Xstar);
	//add noise
	KstarDiag+=this->lik->Kcross_diag(Xstar);

	MatrixXd KK = KstarCross*this->cache->rgetKEffInv();
	KK.array()*=KstarCross.array();
	(*out) = KstarDiag - KK.rowwise().sum();
}


CGPHyperParams CGPbase::getParamBounds(bool upper) const
{
	CGPHyperParams rv;
	//query covariance function bounds
	CovarParams covar_lower,covar_upper;
	this->covar->agetParamBounds(&covar_lower,&covar_upper);
	if (upper)
		rv["covar"] = covar_upper;
	else
		rv["covar"] = covar_lower;

	CovarParams lik_lower,lik_upper;
	this->lik->agetParamBounds(&lik_lower,&lik_upper);
	if(upper)
		rv["lik"] = lik_upper;
	else
		rv["lik"] = lik_lower;

	return rv;
}


CGPHyperParams CGPbase::getParamMask() const {
	CGPHyperParams rv;
	rv["covar"] = this->covar->getParamMask();

	rv["lik"]   = this->lik->getParamMask();

	return rv;
}

    
double CGPbase::LMLgrad_num(CGPbase& gp, const muint_t i) throw (CGPMixException)
{

    double out, LML_plus, LML_minus;
    
    mfloat_t relchange=1E-5;
    
    CGPHyperParams L = gp.getParams();
    CGPHyperParams L0 = L;
    
    const muint_t i0 = L["covar"].rows();
    
    mfloat_t change;
    
    if (i<i0) {
        change = relchange*L["covar"](i);
        change = std::max(change,1E-5);
        L["covar"](i) = L0["covar"](i) + change;
    }
    else {
        change = relchange*L["lik"](i-i0);
        change = std::max(change,1E-5);
        L["lik"](i-i0) = L0["lik"](i-i0) + change;
    }
    gp.setParams(L);
    
    LML_plus=gp.LML();
    
    if (i<i0) {
        L["covar"](i) = L0["covar"](i) - change;
    }
    else {
        L["lik"](i-i0) = L0["lik"](i-i0) - change;
    }
    gp.setParams(L);
    
    LML_minus=gp.LML();
    
    
    out=(LML_plus-LML_minus)/(2.0*change);
    
    gp.setParams(L0);
    
    return out;
    
}
    
    
double CGPbase::LMLhess_num(CGPbase& gp, const muint_t i, const muint_t j) throw (CGPMixException)
{
    
    double out, LMLgrad_plus, LMLgrad_minus;
    
    mfloat_t relchange=1E-5;

    CGPHyperParams L = gp.getParams();
    CGPHyperParams L0 = L;
    
    const muint_t i0 = L["covar"].rows();
    
    mfloat_t change;
    
    if (j<i0) {
        change = relchange*L["covar"](j);
        change = std::max(change,1E-5);
        L["covar"](j) = L0["covar"](j) + change;
    }
    else {
        change = relchange*L["lik"](j-i0);
        change = std::max(change,1E-5);
        L["lik"](j-i0) = L0["lik"](j-i0) + change;
    }
    gp.setParams(L);
    
    if (i<i0)   LMLgrad_plus=gp.LMLgrad()["covar"](i);
    else        LMLgrad_plus=gp.LMLgrad()["lik"](i-i0);
    
    if (j<i0) {
        L["covar"](j) = L0["covar"](j) - change;
    }
    else {
        L["lik"](j-i0) = L0["lik"](j-i0) - change;
    }
    gp.setParams(L);
    
    if (i<i0)   LMLgrad_minus=gp.LMLgrad()["covar"](i);
    else        LMLgrad_minus=gp.LMLgrad()["lik"](i-i0);
    
    
    out=(LMLgrad_plus-LMLgrad_minus)/(2.0*change);
    
    gp.setParams(L0);
     
    return out;
    
}

/*  CGP Variance Decomposition  */

//CGPvarDecomp::CGPvarDecomp(): CGPbase(PTDiagonalCF(),PLikNormalNULL(),PLinearMean())//CGPbase(PTDiagonalCF(1),PLikNormalNULL(),PLinearMean(MatrixXd::Ones(1,1),MatrixXd::Ones(1,1)))
//{
//}

CGPvarDecomp::CGPvarDecomp(PCovarianceFunction covar, PLikelihood lik,PDataTerm dataTerm, const VectorXd& lambda, const muint_t P, const MatrixXd& pheno, const VectorXd& initParams): CGPbase(covar,lik,dataTerm)
{
    this->P=P;
	this->lambda=lambda;
	this->N=pheno.rows();
	this->pheno=pheno;
	this->initParams=initParams;
}

CGPvarDecomp::~CGPvarDecomp()
{
}

void CGPvarDecomp::initGPs() throw(CGPMixException)
{
	/*
	// initialize lik
	lik = PLikNormalNULL(new CLikNormalNULL());

	// initialise C2
	MatrixXd K0 = MatrixXd::Ones(P,P);
	C2 = PTFixedCF(new CTFixedCF(this->P,K0));
	C2->setX(this->trait);

	for (muint_t i=0; i<N; ++i)	C1.push_back(PTFixedCF(new CTFixedCF(this->P,this->lambda(i)*K0)));

	// initialise C1s and GPs and LinearMeans

	for (muint_t i=0; i<N; ++i) {
		PTFixedCF C1_i(new CTFixedCF(this->P,this->lambda(i)*K0));
		C1_i->setX(this->trait);
		PSumCF covar_i(new CSumCF());
		covar_i->addCovariance(C1_i);
		covar_i->addCovariance(C2);
		PLinearMean mean_i(new CLinearMean(this->pheno.block(0,i,P,1),MatrixXd::Identity(P,P)));
		PGPbase gp_i(new CGPbase(covar_i,lik,mean_i));
		C1.push_back(C1_i);
		covar.push_back(covar_i);
		vecLinearMeans.push_back(mean_i);
		vecGPs.push_back(*gp_i);
	}
	*/

	/*
	muint_t i = 0;
	ACovarVec::const_iterator C1it = C1.begin();
	ACovarVec::const_iterator covarit = covar.begin();
	ALinearMeanVec::const_iterator vecLinearMeansit = vecLinearMeans.begin();
	AGPbaseVec::const_iterator vecGPsit = vecGPs.begin();
	for(; C1it != C1.end() && covarit != covar.end() && vecGPsit != vecGPs.end() && vecLinearMeansit != vecLinearMeans.end(); ++C1it, ++covarit, ++vecGPsit, ++vecLinearMeansit, ++i)
	{
		PCovarianceFunction C1_i = C1it[0];
		PCovarianceFunction covar_i = covarit[0];
		PLinearMean vecLinearMeans_i = vecLinearMeansit[0];
		CGPbase vecGPs_i = vecGPsit[0];
		// Covariance
		C1_i = PTFixedCF(new CTFixedCF(this->P,this->lambda(i)*K0));
		C1_i->setX(this->trait);
		covar_i = PSumCF(new CSumCF());
		static_pointer_cast<CSumCF>(covar_i)->addCovariance(C1_i);
		static_pointer_cast<CSumCF>(covar_i)->addCovariance(C2);
		// LinearMeans
		vecLinearMeans_i = PLinearMean(new CLinearMean(this->pheno.block(0,i,P,1),MatrixXd::Identity(P,P)));
		// GPs
		vecGPs_i.setCovar(covar_i);
		vecGPs_i.setLik(lik);
		vecGPs_i.setDataTerm(vecLinearMeans_i);
		vecGPs_i.setY(pheno.block(0,i,N,1));
	}
	*/

	// Initialize Parameters
	CGPHyperParams params;
	params["covar"] = initParams;
	params["dataTerm"] = MatrixXd::Zero(P,1);
	this->setParams(params);
}


void CGPvarDecomp::updateParams() throw(CGPMixException)
{
	AGPbaseVec::const_iterator vecGPsit = vecGPs.begin();
	for(; vecGPsit != vecGPs.end(); ++vecGPsit)
	{
		PGPbase vecGPs_i = vecGPsit[0];
		vecGPs_i->setParams(this->params);
		this->state++;
	}
}

/*
void CGPvarDecomp::setParams(const CGPHyperParams& hyperparams) throw(CGPMixException)
{
	AGPbaseVec::const_iterator vecGPsit = vecGPs.begin();
	for(; vecGPsit != vecGPs.end(); ++vecGPsit)
	{
		CGPbase vecGPs_i = vecGPsit[0];
		vecGPs_i.setParams(hyperparams);
	}
}
*/

mfloat_t CGPvarDecomp::LML() throw (CGPMixException)
{
	mfloat_t out=0;
	AGPbaseVec::const_iterator vecGPsit = vecGPs.begin();
	for(; vecGPsit != vecGPs.end(); ++vecGPsit)
	{
		PGPbase vecGPs_i = vecGPsit[0];
		out+=vecGPs_i->LML();
	}
	return out;
};

void CGPvarDecomp::aLMLgrad_covar(VectorXd* out) throw (CGPMixException)
{
	VectorXd grad_covar = VectorXd::Zero((this->params["covar"]).rows());
	(*out) = VectorXd::Zero((this->params["covar"]).rows());
	AGPbaseVec::const_iterator vecGPsit = vecGPs.begin();
	for(; vecGPsit != vecGPs.end(); ++vecGPsit)
	{
		PGPbase vecGPs_i = vecGPsit[0];
		vecGPs_i->aLMLgrad_covar(&grad_covar);
		(*out)+=grad_covar;
	}
}


void CGPvarDecomp::aLMLgrad_lik(VectorXd* out) throw (CGPMixException)
{
}

void CGPvarDecomp::aLMLgrad_X(MatrixXd* out) throw (CGPMixException)
{
}


void CGPvarDecomp::aLMLgrad_dataTerm(MatrixXd* out) throw (CGPMixException)
{
	MatrixXd grad_dataTerm = MatrixXd::Zero((this->params["dataTerm"]).rows(),(this->params["dataTerm"]).cols());
	(*out) = MatrixXd::Zero((this->params["dataTerm"]).rows(),(this->params["dataTerm"]).cols());
	AGPbaseVec::const_iterator vecGPsit = vecGPs.begin();
	for(; vecGPsit != vecGPs.end(); ++vecGPsit)
	{
		PGPbase vecGPs_i = vecGPsit[0];
		vecGPs_i->aLMLgrad_dataTerm(&grad_dataTerm);
		(*out)+=grad_dataTerm;
	}
}


} /* namespace limix */
