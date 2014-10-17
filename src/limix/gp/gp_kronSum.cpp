// Copyright(c) 2014, The LIMIX developers(Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
//
//Licensed under the Apache License, Version 2.0 (the "License");
//you may not use this file except in compliance with the License.
//You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//Unless required by applicable law or agreed to in writing, software
//distributed under the License is distributed on an "AS IS" BASIS,
//WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//See the License for the specific language governing permissions and
//limitations under the License.

#include "gp_kronSum.h"
#include "limix/utils/matrix_helper.h"
#include <ctime>

namespace limix {

mfloat_t te1(clock_t beg){
	clock_t end = clock();
	mfloat_t TE = mfloat_t(end - beg) / CLOCKS_PER_SEC;
	return TE;
}

/* CGPkronSumCache */

CGPkronSumCache::CGPkronSumCache(CGPkronSum* gp)
{
	this->syncCovarc1 = Pbool(new bool);
	this->syncCovarc2 = Pbool(new bool);
	this->syncCovarr1 = Pbool(new bool);
	this->syncCovarr2 = Pbool(new bool);
	this->syncLik = Pbool(new bool);
	this->syncData = Pbool(new bool);
	//add to list of sync parents
	this->addSyncParent(syncCovarc1);
	this->addSyncParent(syncCovarc2);
	this->addSyncParent(syncCovarr1);
	this->addSyncParent(syncCovarr2);
	this->addSyncParent(syncLik);
	this->addSyncParent(syncData);

	this->gp = gp;
	this->covarc1 = PCovarianceFunctionCacheOld(new CCovarianceFunctionCacheOld(this->gp->covarc1));
	this->covarc2 = PCovarianceFunctionCacheOld(new CCovarianceFunctionCacheOld(this->gp->covarc2));
	this->covarr1 = PCovarianceFunctionCacheOld(new CCovarianceFunctionCacheOld(this->gp->covarr1));
	this->covarr2 = PCovarianceFunctionCacheOld(new CCovarianceFunctionCacheOld(this->gp->covarr2));
	//add sync liestener
	covarc1->addSyncChild(this->syncCovarc1);
	covarc2->addSyncChild(this->syncCovarc2);
	covarr1->addSyncChild(this->syncCovarr1);
	covarr2->addSyncChild(this->syncCovarr2);
	this->gp->lik->addSyncChild(this->syncLik);
	this->gp->dataTerm->addSyncChild(this->syncData);
	//set all cache Variables to Null
	SVDcstarCacheNull=true;
	LambdacCacheNull=true;
	SVDrstarCacheNull=true;
	LambdarCacheNull=true;
	DCacheNull=true;
	YrotPartCacheNull=true;
	YrotCacheNull=true;
	YtildeCacheNull=true;
}


/*!
This is optimised for the case where rows are fixed and colomns vary
(because of the partial rotation on Y)

if R1 or Omega change
- row rotations (inner rotation performed only if Omega changes)
- Lambdar,D
- YrotPart, Yrot, Ytilde
if C1 or Sigma change
- col rotations (inner rotation performed only if Sigma changes)
- Lambdac,D
- Yrot, Ytilde
*/
void CGPkronSumCache::validateCache()
{
	muint_t maskRR = this->gp->covarr1->getParamMask().sum()+this->gp->covarr2->getParamMask().sum();
	if((!*syncCovarc1) || (!*syncCovarc2))
	{
		SVDcstarCacheNull=true;
		LambdacCacheNull=true;
		DCacheNull=true;
		YrotCacheNull=true;
		YtildeCacheNull=true;
	}
	if(((!*syncCovarr1) || (!*syncCovarr2)) && maskRR>0)
	{
		SVDrstarCacheNull=true;
		LambdarCacheNull=true;
		DCacheNull=true;
		YrotPartCacheNull=true;
		YrotCacheNull=true;
		YtildeCacheNull=true;
	}
	if((!*syncData)) {
		YrotPartCacheNull=true;
		YrotCacheNull=true;
		YtildeCacheNull=true;
	}
	//set all sync
	setSync();
}

/*!
Computes the eigen deceomposition of the rotated column covariance (Cstar).

- Cstar    = diag(Ssigma)^(-0.5) Usigma.T C Usigma.T diag(Ssigma)^(-0.5)
- Cstar    = Ucstar diag(Scstar) Ucstar.T
*/
void CGPkronSumCache::updateSVDcstar()
{
	MatrixXd USisqrt;
	SsigmaCache=this->covarc2->rgetSK();
	aUS2alpha(USisqrt,this->covarc2->rgetUK(),SsigmaCache,-0.5);
	MatrixXd Cstar = USisqrt.transpose()*this->covarc1->rgetK()*USisqrt;
	// ADD SOME DIAGONAL STUFF?
	Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver(Cstar);
	UcstarCache = eigensolver.eigenvectors();
	ScstarCache = eigensolver.eigenvalues();
}


MatrixXd& CGPkronSumCache::rgetSsigma()
{
	validateCache();
	if (SVDcstarCacheNull) {
		updateSVDcstar();
		SVDcstarCacheNull=false;
	}
	return SsigmaCache;
}

/*!
Returns the eigenvalues (Scstar) of the rotated column covariance (Cstar).

- Cstar    = diag(Ssigma)^(-0.5) Usigma.T C Usigma.T diag(Ssigma)^(-0.5)
- Cstar    = Ucstar diag(Scstar) Ucstar.T
*/
MatrixXd& CGPkronSumCache::rgetScstar()
{
	validateCache();
	clock_t beg = clock();
	if (SVDcstarCacheNull) {
		updateSVDcstar();
		SVDcstarCacheNull=false;
	}
    this->gp->rtSVDcols=te1(beg);
	return ScstarCache;
}


/*!
Returns the Matrix of eigenvectors (Ucstar) of the rotated column covariance (Cstar).

- Cstar    = diag(Ssigma)^(-0.5) Usigma.T C Usigma.T diag(Ssigma)^(-0.5)
- Cstar    = Ucstar diag(Scstar) Ucstar.T
*/
MatrixXd& CGPkronSumCache::rgetUcstar()
{
	validateCache();
	if (SVDcstarCacheNull) {
		updateSVDcstar();
		SVDcstarCacheNull=false;
	}
	return UcstarCache;
}

MatrixXd& CGPkronSumCache::rgetLambdac()
{
	validateCache();
	clock_t beg = clock();
	if (LambdacCacheNull) {
		aUS2alpha(LambdacCache,rgetUcstar().transpose(),this->covarc2->rgetSK(),-0.5);
		LambdacCache*=this->covarc2->rgetUK().transpose();
		LambdacCacheNull=false;
	}
    this->gp->rtLambdac=te1(beg);
	return LambdacCache;
}

/*!
Computes the eigen deceomposition of the rotated row covariance (Rstar).

- Rstar    = diag(Somega)^(-0.5) Uomega.T R Uomega.T diag(Somega)^(-0.5)
- Rstar    = Urstar diag(Srstar) Urstar.T
*/
void CGPkronSumCache::updateSVDrstar()
{
	clock_t beg = clock();
	MatrixXd USisqrt;
	SomegaCache=this->covarr2->rgetSK();
	aUS2alpha(USisqrt,this->covarr2->rgetUK(),SomegaCache,-0.5);
	MatrixXd Rstar = USisqrt.transpose()*this->covarr1->rgetK()*USisqrt;
	Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver(Rstar);
	UrstarCache = eigensolver.eigenvectors();
	SrstarCache = eigensolver.eigenvalues();
	if (this->gp->debug)	std::cout<<"SVD rows:   "<<te1(beg)<<std::endl;
}


/*!
Returns the eigenvalues (Scstar) of the rotated row covariance (Rstar).

- Rstar    = diag(Somega)^(-0.5) Uomega.T R Uomega.T diag(Somega)^(-0.5)
- Rstar    = Urstar diag(Srstar) Urstar.T
*/
MatrixXd& CGPkronSumCache::rgetSomega()
{
	validateCache();
	if (SVDrstarCacheNull) {
		updateSVDrstar();
		SVDrstarCacheNull=false;
	}
	return SomegaCache;
}

/*!
Returns the matrix of eigenvectors (Ucstar) of the rotated row covariance (Rstar).

- Rstar    = diag(Somega)^(-0.5) Uomega.T R Uomega.T diag(Somega)^(-0.5)
- Rstar    = Urstar diag(Srstar) Urstar.T
*/
MatrixXd& CGPkronSumCache::rgetSrstar()
{
	validateCache();
	clock_t beg = clock();
	if (SVDrstarCacheNull) {
		updateSVDrstar();
		SVDrstarCacheNull=false;
	}
    this->gp->rtSVDrows=te1(beg);
	return SrstarCache;
}

MatrixXd& CGPkronSumCache::rgetUrstar()
{
	validateCache();
	if (SVDrstarCacheNull) {
		updateSVDrstar();
		SVDrstarCacheNull=false;
	}
	return UrstarCache;
}

MatrixXd& CGPkronSumCache::rgetLambdar()
{
	validateCache();
	clock_t beg = clock();
	if (LambdarCacheNull) {
		aUS2alpha(LambdarCache,rgetUrstar().transpose(),this->covarr2->rgetSK(),-0.5);
		LambdarCache*=this->covarr2->rgetUK().transpose();
		LambdarCacheNull=false;
	}
    this->gp->rtLambdar=te1(beg);
	return LambdarCache;
}

MatrixXd& CGPkronSumCache::rgetYrotPart()
{
	validateCache();
	clock_t beg = clock();
	if (YrotPartCacheNull) {
		//Rotate columns of Y
		YrotPartCache.resize(gp->getY().rows(),gp->getY().cols());
		MatrixXd& Lambdar = rgetLambdar();
		YrotPartCache.noalias()=Lambdar*gp->dataTerm->evaluate();
		YrotPartCacheNull=false;
	}
    this->gp->rtYrotPart=te1(beg);
	return YrotPartCache;
}

MatrixXd& CGPkronSumCache::rgetYrot()
{
	validateCache();
	clock_t beg = clock();
	if (YrotCacheNull) {
		//Rotate rows of Y
		YrotCache.resize(gp->getY().rows(),gp->getY().cols());
		MatrixXd& Lambdac = rgetLambdac();
		MatrixXd& YrotPart = rgetYrotPart();
		YrotCache.noalias()=YrotPart*Lambdac.transpose();
		YrotCacheNull=false;
	}
    this->gp->rtYrot=te1(beg);
	return YrotCache;
}

MatrixXd& CGPkronSumCache::rgetD()
{
	validateCache();
	clock_t beg = clock();
	if (DCacheNull) {
	    DCache.resize(gp->getN(),gp->getP());
	    MatrixXd& Scstar = rgetScstar();
	    MatrixXd& Srstar = rgetSrstar();
	    for (muint_t n=0; n<this->gp->getN(); n++)	{
	        for (muint_t p=0; p<this->gp->getP(); p++)	{
	        	DCache(n,p)=1/(Scstar(p,0)*Srstar(n,0)+1);
	    	}
	    }
	    DCacheNull=false;
	}
    this->gp->rtD=te1(beg);
	return DCache;
}

MatrixXd& CGPkronSumCache::rgetYtilde()
{
	validateCache();
	clock_t beg = clock();
	if (YtildeCacheNull) {
	    MatrixXd& Yrot = rgetYrot();
	    MatrixXd& D = rgetD();
	    YtildeCache = D.array()*Yrot.array();
	    YtildeCacheNull=false;
	}
    this->gp->rtYtilde=te1(beg);
	return YtildeCache;
}

/* CGPkronSum */

CGPkronSum::CGPkronSum(const MatrixXd& Y,
						PCovarianceFunction covarr1, PCovarianceFunction covarc1,
						PCovarianceFunction covarr2, PCovarianceFunction covarc2,
    					PLikelihood lik, PDataTerm dataTerm) : CGPbase(covarr1,lik,dataTerm)
{
	this->covarc1=covarc1;
	this->covarc2=covarc2;
	this->covarr1=covarr1;
	this->covarr2=covarr2;
	this->cache = PGPkronSumCache(new CGPkronSumCache(this));
	this->setY(Y);
	this->N=Y.rows();
	this->P=Y.cols();

	this->lambda_g=0;
	this->lambda_n=0;

	rtLML1a=0;
	rtLML1b=0;
	rtLML1c=0;
	rtLML1d=0;
	rtLML1e=0;
	rtLML2=0;
	rtLML3=0;
	rtLML4=0;
	rtGrad=0;
	rtCC1part1a=0;
	rtCC1part1b=0;
	rtCC1part1c=0;
	rtCC1part1d=0;
	rtCC1part1e=0;
	rtCC1part1f=0;
	rtCC1part2=0;
	rtCC2part1=0;
	rtCC2part2=0;

	debug=false;
}


CGPkronSum::~CGPkronSum()
{
}

void CGPkronSum::updateParams() 
{

	//is this needed?
	//CGPbase::updateParams();
	if(this->params.exists("covarc1"))
		this->covarc1->setParams(this->params["covarc1"]);

	if(this->params.exists("covarc2"))
		this->covarc2->setParams(this->params["covarc2"]);

	if(this->params.exists("covarr1"))
		this->covarr1->setParams(this->params["covarr1"]);

	if(this->params.exists("covarr2"))
		this->covarr2->setParams(this->params["covarr2"]);

	if(this->params.exists("dataTerm"))
	{
		this->dataTerm->setParams(this->params["dataTerm"]);
	}

}

CGPHyperParams CGPkronSum::getParamBounds(bool upper) const
{
	CGPHyperParams rv;
	//query covariance function bounds
	CovarParams covarc1_lower,covarc1_upper;
	this->covarc1->agetParamBounds(&covarc1_lower,&covarc1_upper);
	if (upper)	rv["covarc1"] = covarc1_upper;
	else		rv["covarc1"] = covarc1_lower;
	CovarParams covarc2_lower,covarc2_upper;
	this->covarc2->agetParamBounds(&covarc2_lower,&covarc2_upper);
	if (upper)	rv["covarc2"] = covarc2_upper;
	else		rv["covarc2"] = covarc2_lower;
	CovarParams covarr1_lower,covarr1_upper;
	this->covarr1->agetParamBounds(&covarr1_lower,&covarr1_upper);
	if (upper)	rv["covarr1"] = covarr1_upper;
	else		rv["covarr1"] = covarr1_lower;
	CovarParams covarr2_lower,covarr2_upper;
	this->covarr2->agetParamBounds(&covarr2_lower,&covarr2_upper);
	if (upper)	rv["covarr2"] = covarr2_upper;
	else		rv["covarr2"] = covarr2_lower;
	return rv;
}


CGPHyperParams CGPkronSum::getParamMask() const {
	CGPHyperParams rv;
	rv["covarc1"] = this->covarc1->getParamMask();
	rv["covarc2"] = this->covarc2->getParamMask();
	rv["covarr1"] = this->covarr1->getParamMask();
	rv["covarr2"] = this->covarr2->getParamMask();

	return rv;
}

void CGPkronSum::agetKEffInvYCache(MatrixXd* out) 
{
	//Computing Kinv
	MatrixXd& Ytilde = cache->rgetYtilde();
	MatrixXd& Lambdar = cache->rgetLambdar();
	MatrixXd& Lambdac = cache->rgetLambdac();
	(*out) = Lambdar.transpose()*Ytilde*Lambdac;
}


mfloat_t CGPkronSum::LML() 
{
	clock_t beg = clock();
    //get stuff from cache
    MatrixXd Ssigma = cache->rgetSsigma();
    rtLML1a+=te1(beg);
    beg = clock();
    MatrixXd Somega = cache->rgetSomega();
    rtLML1b+=te1(beg);
    beg = clock();
    MatrixXd& Scstar = cache->rgetScstar();
    rtLML1c+=te1(beg);
    beg = clock();
    MatrixXd& Srstar = cache->rgetSrstar();
    rtLML1d+=te1(beg);
    beg = clock();
    MatrixXd& Yrot = cache->rgetYrot();
    if (rtLML1e==0) {
    	rtLML1e+=te1(beg);
    }
    else	rtLML1e+=te1(beg);
    MatrixXd& Ytilde = cache->rgetYtilde();

    beg = clock();
    //1. logdet:
    mfloat_t lml_det = 0;
    mfloat_t temp = 0;
    for (muint_t p=0; p<(muint_t)Yrot.cols(); p++)
    	temp+=std::log(Ssigma(p,0));
    lml_det += Yrot.rows()*temp;
    temp = 0;
    for (muint_t n=0; n<(muint_t)Yrot.rows(); n++)
    	temp+=std::log(Somega(n,0));
    lml_det += Yrot.cols()*temp;
    temp = 0;
    for (muint_t n=0; n<(muint_t) Yrot.rows(); n++)
        for (muint_t p=0; p<(muint_t) Yrot.cols(); p++)
        	temp+=std::log(Scstar(p,0)*Srstar(n,0)+1);
    lml_det += temp;
    lml_det *= 0.5;
    rtLML2+=te1(beg);

    beg = clock();
    //2. quadratic term
    mfloat_t lml_quad = (Ytilde.array()*Yrot.array()).sum();
    lml_quad *= 0.5;
    rtLML3+=te1(beg);

    beg = clock();
    //3. constants
    mfloat_t lml_const = 0.5*Yrot.cols()*Yrot.rows() * limix::log((2.0 * PI));
    rtLML4+=te1(beg);

	//4. penalization
	mfloat_t lml_pen_g = 0;
	if (lambda_g>0) {
		MatrixXd C1 = covarc1->K();
		for (muint_t ir=0; ir<C1.rows(); ir++)
			for (muint_t ic=0; ic<ir; ic++)
				lml_pen_g+= C1(ir,ic)*C1(ir,ic);
		lml_pen_g*=lambda_g;
	}
	mfloat_t lml_pen_n = 0;
	if (lambda_n>0) {
		MatrixXd C2 = covarc2->K();
		for (muint_t ir=0; ir<C2.rows(); ir++)
			for (muint_t ic=0; ic<ir; ic++)
				lml_pen_n+= C2(ir,ic)*C2(ir,ic);
		lml_pen_n*=lambda_n;
	}

    return lml_quad + lml_det + lml_const+ lml_pen_g + lml_pen_n;
};


CGPHyperParams CGPkronSum::LMLgrad() 
{
    CGPHyperParams rv;
    //calculate gradients for parameter components in params:
	clock_t beg = clock();
    if(params.exists("covarc1")){
        VectorXd grad_covar;
        aLMLgrad_covarc1(&grad_covar);
        if (lambda_g>0) {
            MatrixXd C1, C1grad;
            for (muint_t i=0; i<(muint_t)params["covarc1"].rows(); i++) {
                C1     = covarc1->K();
                C1grad = covarc1->Kgrad_param(i);
                for (muint_t ir=0; ir<C1.rows(); ir++)
                    for (muint_t ic=0; ic<ir; ic++)
                        grad_covar(i)+= 2*lambda_g*C1(ir,ic)*C1grad(ir,ic);
            }
        }
        rv.set("covarc1", grad_covar);
    }
    if(params.exists("covarc2")){
        VectorXd grad_covar;
        aLMLgrad_covarc2(&grad_covar);
        if (lambda_n>0) {
            MatrixXd C2, C2grad;
            for (muint_t i=0; i<params["covarc2"].rows(); i++) {
                C2     = covarc2->K();
                C2grad = covarc2->Kgrad_param(i);
                for (muint_t ir=0; ir<C2.rows(); ir++)
                    for (muint_t ic=0; ic<ir; ic++)
                        grad_covar(i)+= 2*lambda_n*C2(ir,ic)*C2grad(ir,ic);
            }
        }
        rv.set("covarc2", grad_covar);
    }
    if(params.exists("covarr1")){
        VectorXd grad_covar;
        if (covarr1->getParamMask().sum()==0)
        	grad_covar=VectorXd::Zero(covarr1->getNumberParams(),1);
        else
        	aLMLgrad_covarr1(&grad_covar);
        rv.set("covarr1", grad_covar);
    }
    if(params.exists("covarr2")){
        VectorXd grad_covar;
        if (covarr1->getParamMask().sum()==0)
        	grad_covar=VectorXd::Zero(covarr2->getNumberParams(),1);
        else
        aLMLgrad_covarr2(&grad_covar);
        rv.set("covarr2", grad_covar);
    }
    this->rtLMLgradCovar=te1(beg);
	beg = clock();
    if (params.exists("dataTerm"))
    {
    	MatrixXd grad_dataTerm;
    	aLMLgrad_dataTerm(&grad_dataTerm);
    	rv.set("dataTerm",grad_dataTerm);
    }
    this->rtLMLgradDataTerm=te1(beg);
    rtGrad+=rtLMLgradCovar+rtLMLgradDataTerm;
    return rv;
}

void CGPkronSum::aLMLgrad_covarc1(VectorXd *out) 
{
	clock_t beg = clock();
    //get stuff from cache
    MatrixXd& D = cache->rgetD();
    this->rtCC1part1a+=te1(beg);
	beg = clock();
    MatrixXd& Srstar = cache->rgetSrstar();
    this->rtCC1part1b+=te1(beg);
	beg = clock();
    MatrixXd& Yrot = cache->rgetYrot();
    this->rtCC1part1c+=te1(beg);
	beg = clock();
	MatrixXd& Lambdac = cache->rgetLambdac();
    this->rtCC1part1d+=te1(beg);
	beg = clock();
	MatrixXd& Ytilde = cache->rgetYtilde();
    this->rtCC1part1e+=te1(beg);
	beg = clock();
	//param mask
	VectorXd paramMask = this->covarc1->getParamMask();

	beg = clock();
    //start loop trough covariance paramenters
    (*out).resize(covarc1->getNumberParams(),1);
    MatrixXd CgradRot(this->getY().rows(),this->getY().rows());
	for (muint_t i=0; i<covarc1->getNumberParams(); i++) {
		if (paramMask(i)==0) {
			(*out)(i,0)=0;
		}
		else {
			CgradRot=Lambdac*covarc1->Kgrad_param(i)*Lambdac.transpose();
			//1. grad logdet
			mfloat_t grad_det = 0;
	    	for (muint_t n=0; n<(muint_t)Yrot.rows(); n++)
	        	for (muint_t p=0; p<(muint_t)Yrot.cols(); p++)
	        		grad_det+=CgradRot(p,p)*Srstar(n,0)*D(n,p);
	    	grad_det*=0.5;
			//2. grad quadratic term
	    	// Decomposition in columns and row
	   		MatrixXd _Ytilde(Yrot.rows(),Yrot.cols());
			for (muint_t n=0; n<(muint_t)Yrot.rows(); n++)
				for (muint_t p=0; p<(muint_t)Yrot.cols(); p++)
					_Ytilde(n,p) = Srstar(n,0)*Ytilde(n,p);
			mfloat_t grad_quad = -0.5*(Ytilde.array()*(_Ytilde*CgradRot.transpose()).array()).sum();
			(*out)(i,0)=grad_det+grad_quad;
		}
    }
    this->rtCC1part2+=te1(beg);
}

void CGPkronSum::aLMLgrad_covarc2(VectorXd *out) 
{
	clock_t beg = clock();
    //get stuff from cache
    MatrixXd& D = cache->rgetD();
    MatrixXd& Yrot = cache->rgetYrot();
	MatrixXd& Lambdac = cache->rgetLambdac();
	MatrixXd& Ytilde = cache->rgetYtilde();
	//param mask
	VectorXd paramMask = this->covarc1->getParamMask();

	beg = clock();
    //start loop trough covariance paramenters
    (*out).resize(covarc2->getNumberParams(),1);
    MatrixXd SigmaGradRot(this->getY().rows(),this->getY().rows());
	for (muint_t i=0; i<covarc2->getNumberParams(); i++) {
		if (paramMask(i)==0) {
			(*out)(i,0)=0;
		}
		else {
			SigmaGradRot=Lambdac*covarc2->Kgrad_param(i)*Lambdac.transpose();
			//1. grad logdet
			mfloat_t grad_det = 0;
	    	for (muint_t n=0; n<(muint_t)Yrot.rows(); n++)
	        	for (muint_t p=0; p<(muint_t)Yrot.cols(); p++)
	        		grad_det+=SigmaGradRot(p,p)*D(n,p);
	    	grad_det*=0.5;
			//2. grad quadratic term
	    	// Decomposition in columns and row
			mfloat_t grad_quad = -0.5*(Ytilde.array()*(Ytilde*SigmaGradRot.transpose()).array()).sum();
			(*out)(i,0)=grad_det+grad_quad;
		}
    }
    this->rtCC2part2+=te1(beg);
}

void CGPkronSum::aLMLgrad_covarr1(VectorXd *out) 
{
	clock_t beg = clock();
    //get stuff from cache
    MatrixXd& Scstar = cache->rgetScstar();
    MatrixXd& Srstar = cache->rgetSrstar();
    MatrixXd& Yrot = cache->rgetYrot();
	MatrixXd& Lambdac = cache->rgetLambdac();
	MatrixXd& Lambdar = cache->rgetLambdar();
	MatrixXd& Ytilde = cache->rgetYtilde();
    this->rtCR1part1a+=te1(beg);
	beg = clock();
	//covar
	MatrixXd Crot = Lambdac*cache->covarc1->rgetK()*Lambdac.transpose();
    this->rtCR1part1b+=te1(beg);

	beg = clock();
    //start loop trough covariance paramenters
    (*out).resize(covarr1->getNumberParams(),1);
    MatrixXd RgradRot(this->getY().rows(),this->getY().rows());
	for (muint_t i=0; i<covarr1->getNumberParams(); i++) {
		beg = clock();
		RgradRot=Lambdar*covarr1->Kgrad_param(i)*Lambdar.transpose();
		this->is_it+=te1(beg);
		//1. grad logdet
		mfloat_t grad_det = 0;
	    for (muint_t n=0; n<(muint_t)Yrot.rows(); n++)	{
	        for (muint_t p=0; p<(muint_t)Yrot.cols(); p++)	{
	        	grad_det+=Crot(p,p)*RgradRot(n,n)/(Scstar(p,0)*Srstar(n,0)+1);
	    	}
	    }
	    grad_det*=0.5;
		//2. grad quadratic term
	    // Decomposition in columns and row
	    MatrixXd _Ytilde(Yrot.rows(),Yrot.cols());
	    MatrixXd YtildeR(Yrot.rows(),Yrot.cols());
		for (muint_t p=0; p<(muint_t)Yrot.cols(); p++)
			_Ytilde.block(0,p,Yrot.rows(),1).noalias()=RgradRot*Ytilde.block(0,p,Yrot.rows(),1);
		for (muint_t n=0; n<(muint_t)Yrot.rows(); n++)
			YtildeR.block(n,0,1,Yrot.cols()).noalias()=_Ytilde.block(n,0,1,Yrot.cols())*Crot.transpose();
		mfloat_t grad_quad = -0.5*(Ytilde.array()*YtildeR.array()).sum();
		(*out)(i,0)=grad_det+grad_quad;
    }
    this->rtCR1part2+=te1(beg);
}

void CGPkronSum::aLMLgrad_covarr2(VectorXd *out) 
{
	clock_t beg = clock();
    //get stuff from cache
    MatrixXd& Scstar = cache->rgetScstar();
    MatrixXd& Srstar = cache->rgetSrstar();
    MatrixXd& Yrot = cache->rgetYrot();
	MatrixXd& Lambdac = cache->rgetLambdac();
	MatrixXd& Lambdar = cache->rgetLambdar();
	MatrixXd& Ytilde = cache->rgetYtilde();
    this->rtCR2part1a+=te1(beg);
	beg = clock();
	//covar
	MatrixXd SigmaRot = Lambdac*cache->covarc2->rgetK()*Lambdac.transpose();
    this->rtCR2part1b+=te1(beg);

	beg = clock();
    //start loop trough covariance paramenters
    (*out).resize(covarr2->getNumberParams(),1);
    MatrixXd OmgaGradRot(this->getY().rows(),this->getY().rows());
	for (muint_t i=0; i<covarr2->getNumberParams(); i++) {
		OmgaGradRot=Lambdar*covarr2->Kgrad_param(i)*Lambdar.transpose();
		//1. grad logdet
		mfloat_t grad_det = 0;
	    for (muint_t n=0; n<(muint_t)Yrot.rows(); n++)	{
	        for (muint_t p=0; p<(muint_t)Yrot.cols(); p++)	{
	        	grad_det+=SigmaRot(p,p)*OmgaGradRot(n,n)/(Scstar(p,0)*Srstar(n,0)+1);
	    	}
	    }
	    grad_det*=0.5;
		//2. grad quadratic term
	    // Decomposition in columns and row
	    MatrixXd _Ytilde(Yrot.rows(),Yrot.cols());
	    MatrixXd YtildeR(Yrot.rows(),Yrot.cols());
		for (muint_t p=0; p<(muint_t)Yrot.cols(); p++)
			_Ytilde.block(0,p,Yrot.rows(),1).noalias()=OmgaGradRot*Ytilde.block(0,p,Yrot.rows(),1);
		for (muint_t n=0; n<(muint_t)Yrot.rows(); n++)
			YtildeR.block(n,0,1,Yrot.cols()).noalias()=_Ytilde.block(n,0,1,Yrot.cols())*SigmaRot.transpose();
		mfloat_t grad_quad = -0.5*(Ytilde.array()*YtildeR.array()).sum();
		(*out)(i,0)=grad_det+grad_quad;
    }
    this->rtCR2part2+=te1(beg);
}

void CGPkronSum::aLMLgrad_dataTerm(MatrixXd* out) 
{
	MatrixXd KinvY;
	this->agetKEffInvYCache(&KinvY);
	(*out) = this->dataTerm->gradParams(KinvY);
}

void CGPkronSum::aLMLhess_c1c1(MatrixXd *out) 
{
	/* TO DO

	//get stuff from cache
    MatrixXd& Scstar = cache->rgetScstar();
    MatrixXd& Srstar = cache->rgetSrstar();
	MatrixXd& Lambdac = cache->rgetLambdac();
	MatrixXd& Ytilde = cache->rgetYtilde();
	//covar
	MatrixXd& Rrot = cache->rgetRrot();

    //start loop trough covariance paramenters
    (*out).resize(covarc1->getNumberParams(),1);
    MatrixXd CgradRot_i(this->getY().rows(),this->getY().rows());
    MatrixXd CgradRot_j(this->getY().rows(),this->getY().rows());
    MatrixXd ChessRot(this->getY().rows(),this->getY().rows());
	for (muint_t i=0; i<covarc1->getNumberParams(); i++) {
		for (muint_t j=i; j<covarc1->getNumberParams(); j++) {
			CgradRot_i=Lambdac*covarc1->Kgrad_param(i)*Lambdac.transpose();
			CgradRot_j=Lambdac*covarc1->Kgrad_param(j)*Lambdac.transpose();
			ChessRot=Lambdac*covarc1->Khess_param(i,j)*Lambdac.transpose();
			//1. grad logdet
			mfloat_t grad_det = 0;
			for (muint_t n=0; n<(muint_t)Ytilde.rows(); n++)	{
				for (muint_t p=0; p<(muint_t)Ytilde.cols(); p++)	{
					grad_det+=CgradRot(p,p)*Rrot(n,n)/(Scstar(p,0)*Srstar(n,0)+1);
				}
			}
			grad_det*=0.5;
			//2. grad quadratic term
			// Decomposition in columns and row
			MatrixXd Ytilde0(Ytilde.rows(),Ytilde.cols());
			MatrixXd Ytilde1a(Ytilde.rows(),Ytilde.cols());
			MatrixXd Ytilde1b(Ytilde.rows(),Ytilde.cols());
			MatrixXd Ytilde2b(Ytilde.rows(),Ytilde.cols());
			for (muint_t p=0; p<(muint_t)Ytilde.cols(); p++)
				Ytilde0.block(0,p,Ytilde.rows(),1).noalias()=Rrot*Ytilde.block(0,p,Ytilde.rows(),1);
			for (muint_t n=0; n<(muint_t)Ytilde.rows(); n++) {
				Ytilde1a.block(n,0,1,Ytilde.cols()).noalias()=Ytilde0.block(n,0,1,Ytilde.cols())*CgradRot_i.transpose();
				Ytilde1b.block(n,0,1,Ytilde.cols()).noalias()=Ytilde0.block(n,0,1,Ytilde.cols())*CgradRot_j.transpose();
				Ytilde2b.block(n,0,1,Ytilde.cols()).noalias()=Ytilde0.block(n,0,1,Ytilde.cols())*ChessRot.transpose();
			}
			for (muint_t n=0; n<(muint_t)Ytilde.rows(); n++)
				for (muint_t p=0; p<(muint_t)Ytilde.cols(); p++)
					Ytilde1b(n,p)=Ytilde1b(n,p)/(Scstar(p,0)*Srstar(n,0)+1);
			mfloat_t grad_quad = -(Ytilde1a.array()*Ytilde1b.array()).sum();
			grad_quad += 0.5*(Ytilde.array()*Ytilde2b.array()).sum();
			(*out)(i,0)=grad_det+grad_quad;
    }
    */
}



} /* namespace limix */
