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

#include "gp_Sum.h"
#include "limix/utils/matrix_helper.h"
#include <ctime>


namespace limix {

void te(std::string s, clock_t beg){
	clock_t end = clock();
	mfloat_t TE = mfloat_t(end - beg) / CLOCKS_PER_SEC;
	std::cout<<s<<"... "<<TE<<std::endl;
}


CGPSumCache::CGPSumCache(CGPSum* gp)
{
	this->syncCovar1 = Pbool(new bool);
	this->syncCovar2 = Pbool(new bool);
	this->syncLik = Pbool(new bool);
	this->syncData = Pbool(new bool);
	//add to list of sync parents
	this->addSyncParent(syncCovar1);
	this->addSyncParent(syncCovar2);
	this->addSyncParent(syncLik);
	this->addSyncParent(syncData);

	this->gp = gp;
	this->covar1 = PCovarianceFunctionCacheOld(new CCovarianceFunctionCacheOld(this->gp->covar1));
	this->covar2 = PCovarianceFunctionCacheOld(new CCovarianceFunctionCacheOld(this->gp->covar2));
	//add sync liestener
	covar1->addSyncChild(this->syncCovar1);
	covar2->addSyncChild(this->syncCovar2);
	this->gp->lik->addSyncChild(this->syncLik);
	this->gp->dataTerm->addSyncChild(this->syncData);
	//set all cache Variables to Null
	SVDcstarCacheNull=true;
	YrotCacheNull=true;
	LambdaCacheNull=true;
}



void CGPSumCache::validateCache()
{
	if((!*syncCovar1) || (!*syncCovar2))
	{
		SVDcstarCacheNull=true;
		LambdaCacheNull=true;
		YrotCacheNull=true;
	}
	if((!*syncData)) {
		YrotCacheNull=true;
	}
	//set all sync
	setSync();
}

void CGPSumCache::updateSVDcstar()
{
	MatrixXd Cstar = this->covar2->rgetUK()*this->covar1->rgetK()*this->covar2->rgetUK().transpose();
	Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver(Cstar);
	UcstarCache = eigensolver.eigenvectors();
	ScstarCache = eigensolver.eigenvalues();
}

MatrixXd& CGPSumCache::rgetScstar()
{
	validateCache();
	if (SVDcstarCacheNull) {
		updateSVDcstar();
		SVDcstarCacheNull=false;
	}
	return ScstarCache;
}

MatrixXd& CGPSumCache::rgetUcstar()
{
	validateCache();
	if (SVDcstarCacheNull) {
		updateSVDcstar();
		SVDcstarCacheNull=false;
	}
	return UcstarCache;
}

MatrixXd& CGPSumCache::rgetLambda()
{
	validateCache();
	if (LambdaCacheNull) {
		aUS2alpha(LambdaCache,rgetUcstar().transpose(),this->covar2->rgetSK(),-0.5);
		LambdaCache*=this->covar2->rgetUK().transpose();
		LambdaCacheNull=false;
	}
	return LambdaCache;
}

MatrixXd& CGPSumCache::rgetYrot()
{
	validateCache();
	if (YrotCacheNull) {
		YrotCache.noalias()=rgetLambda()*gp->dataTerm->evaluate();
		YrotCacheNull=false;
	}
	return YrotCache;
}

CGPSum::CGPSum(const MatrixXd& Y,PCovarianceFunction covar1, PCovarianceFunction covar2,
    					   PLikelihood lik, PDataTerm dataTerm) : CGPbase(covar1,lik,dataTerm)
{
	this->covar1=covar1;
	this->covar2=covar2;
	this->cache = PGPSumCache(new CGPSumCache(this));
	this->setY(Y);
}

CGPSum::~CGPSum()
{
}

void CGPSum::updateParams() 
{

	//std::cout << params << "\n";

	CGPbase::updateParams();
	if(this->params.exists("covar1"))
		this->covar1->setParams(this->params["covar1"]);

	if(this->params.exists("covar2"))
		this->covar2->setParams(this->params["covar2"]);

	if(params.exists("X1"))
	{
		this->updateX(*covar1, gplvmDimensions1, params["X1"]);
	}

	if(params.exists("X2"))
		this->updateX(*covar2, gplvmDimensions2, params["X2"]);

	if(this->params.exists("dataTerm"))
	{
		this->dataTerm->setParams(this->params["dataTerm"]);
	}

}

    void CGPSum::setX1(const CovarInput & X) 
    {
        this->covar1->setX(X);
        //if(isnull(gplvmDimensions_r))
            //this->gplvmDimensions_r = VectorXi::LinSpaced(X.cols(), 0, X.cols() - 1);
    }
    void CGPSum::setX2(const CovarInput & X) 
    {
        this->covar2->setX(X);
        //if(isnull(gplvmDimensions_c))
            //this->gplvmDimensions_c = VectorXi::LinSpaced(X.cols(), 0, X.cols() - 1);
    }
    void CGPSum::setY(const MatrixXd & Y)
    {
        CGPbase::dataTerm->setY(Y);
        this->lik->setX(MatrixXd::Zero(Y.rows() * Y.cols(), 0));
    }
    void CGPSum::setCovar1(PCovarianceFunction covar)
    {
    	this->covar1 = covar;
		//this->cache->covar_r->setCovar(covar);
    }
    void CGPSum::setCovar2(PCovarianceFunction covar)
    {
        this->covar2 = covar;
        //this->cache->covar_c1->setCovar(covar);
    }

    void CGPSum::debugCache()
    {
    	clock_t beg = clock();
    	MatrixXd Ssigma = this->cache->covar2->rgetSK();
        te("getting Ssigma",beg);

    	beg = clock();
    	Ssigma = this->cache->covar2->rgetSK();
    	te("getting Ssigma again",beg);

    	beg = clock();
    	Ssigma = this->cache->covar2->rgetSK();
    	te("getting Ssigma",beg);

    	beg = clock();
    	MatrixXd& Scstar = cache->rgetScstar();
        te("getting Scstar again",beg);

    	beg = clock();
    	Scstar = cache->rgetScstar();
    	te("getting Scstar again",beg);

    	beg = clock();
    	MatrixXd& Yrot = cache->rgetYrot();
    	te("getting Yrot",beg);

    	beg = clock();
    	Yrot = cache->rgetYrot();
    	te("getting Yrot again",beg);

    	beg = clock();
    	Yrot = cache->rgetYrot();
    	te("getting Yrot again",beg);
    }


    mfloat_t CGPSum::LML() 
    {
        //get stuff from cache
        MatrixXd Ssigma = this->cache->covar2->rgetSK();
        MatrixXd& Scstar = cache->rgetScstar();
        MatrixXd& Yrot = cache->rgetYrot();

        //1. logdet:
        mfloat_t lml_det = 0;
        for (muint_t i=0; i<(muint_t) Ssigma.rows(); i++) {
        	lml_det+=std::log(Ssigma(i,0));
        	lml_det+=std::log(Scstar(i,0)+1);
        }
        lml_det *= 0.5;
        //2. quadratic term
        mfloat_t lml_quad = 0;
        for (muint_t i=0; i<(muint_t) Ssigma.rows(); i++) {
        	lml_quad+=std::pow(Yrot(i,0),2)/(Scstar(i,0)+1);
        }
        lml_quad *= 0.5;
        //3. constants
        mfloat_t lml_const = 0.5*this->getY().cols()*this->getY().rows() * limix::log((2.0 * PI));

        return lml_quad + lml_det + lml_const;
    };

    CGPHyperParams CGPSum::LMLgrad() 
    {
        CGPHyperParams rv;
        //calculate gradients for parameter components in params:
        if(params.exists("covar1")){
            VectorXd grad_covar;
            aLMLgrad_covar1(&grad_covar);
            rv.set("covar1", grad_covar);
        }
        if(params.exists("covar2")){
            VectorXd grad_covar;
            aLMLgrad_covar2(&grad_covar);
            rv.set("covar2", grad_covar);
        }
        if (params.exists("dataTerm"))
        {
        	MatrixXd grad_dataTerm;
        	aLMLgrad_dataTerm(&grad_dataTerm);
        	rv.set("dataTerm",grad_dataTerm);
        }
        /*
        if(params.exists("X1")){
            MatrixXd grad_X;
            aLMLgrad_X1(&grad_X);
            rv.set("X1", grad_X);
        }
        if(params.exists("X2")){
            MatrixXd grad_X;
            aLMLgrad_X2(&grad_X);
            rv.set("X2", grad_X);
        }
        */
        return rv;
    }

    void CGPSum::aLMLgrad_covar(VectorXd *out, bool cov1) 
    {
        //get stuff from cache
        MatrixXd ScstarP1i = cache->rgetScstar();
        ScstarP1i+=MatrixXd::Ones(this->getY().rows(),1);
        //in place inversion
        for (muint_t i=0;i<(muint_t)ScstarP1i.rows();i++){
            for (muint_t j=0;j<(muint_t)ScstarP1i.cols();j++){
            	ScstarP1i(i,j)=std::pow(ScstarP1i(i,j),-1);
            }
        }
    	MatrixXd& Lambda = cache->rgetLambda();
        MatrixXd& Yrot = cache->rgetYrot();

        //taking the right covariance function
        PCovarianceFunction _covar;
        if (cov1)	_covar = this->covar1;
        else		_covar = this->covar2;

        //start loop trough covariance paramenters
        (*out).resize(_covar->getNumberParams(),1);
        MatrixXd Ytilde;
        aS2alphaU(Ytilde,ScstarP1i,Yrot,1);
        MatrixXd Kgrad    =MatrixXd::Zero(this->getY().rows(),this->getY().rows());
        MatrixXd Kgrad_rot=MatrixXd::Zero(this->getY().rows(),this->getY().rows());
		for (muint_t i=0; i<_covar->getNumberParams(); i++) {
			Kgrad=_covar->Kgrad_param(i);
			Kgrad_rot=Lambda*Kgrad*Lambda.transpose();
			//1. grad logdet
			mfloat_t grad_det = 0.5*(ScstarP1i.array()*Kgrad_rot.diagonal().array()).sum();
			//2. grad quadratic term
			mfloat_t grad_quad = -0.5*(Ytilde.array()*(Kgrad_rot*Ytilde).array()).sum();
			(*out)(i,0)=grad_det+grad_quad;
        }
    }

    void CGPSum::aLMLgrad_covar1(VectorXd *out) 
    {
        aLMLgrad_covar(out,true);
    }

    void CGPSum::aLMLgrad_covar2(VectorXd *out) 
    {
        aLMLgrad_covar(out,false);
    }

    void CGPSum::aLMLgrad_dataTerm(MatrixXd* out) 
   {
     //0. set output dimensions
   	 //(*out) = this->dataTerm->gradParams(this->cache->rgetKinvY());
   }

    /*

    mfloat_t CGPSum::_gradLogDet(MatrixXd & dK, bool columns)
    {
        MatrixXd& Si = cache->rgetSi();
        MatrixXd rv;
        if(columns){
        	MatrixXd& U= cache->covar_c->rgetUK();
            VectorXd& S = cache->covar_r->rgetSK();
            MatrixXd d = (U.array() * (dK * U).array()).colwise().sum();
            rv.noalias() = S.transpose() * Si * d.transpose();
        }else{
            MatrixXd& U = cache->covar_r->rgetUK();
            VectorXd& S = cache->covar_c->rgetSK();
            MatrixXd d = (U.array() * (dK * U).array()).colwise().sum();
            rv.noalias() = d* Si * S;
        }
        return rv(0, 0);
    }

    mfloat_t CGPSum::_gradQuadrForm(MatrixXd & dK, bool columns)
    {
        MatrixXd& Ysi = cache->rgetYSi();
        MatrixXd UdKU;
        MatrixXd SYUdKU;
        if(columns){
            MatrixXd& U = cache->covar_c->rgetUK();
            VectorXd& S = cache->covar_r->rgetSK();
            UdKU.noalias() = U.transpose() * dK * U;
            //start with multiplying Y with Sc
            SYUdKU = Ysi;
            MatrixXd St = MatrixXd::Zero(Ysi.rows(), Ysi.cols());
            St.colwise() = S;
            SYUdKU.array() *= St.array();
            //dot product with UdKU
            SYUdKU = SYUdKU * UdKU.transpose();
        }else{
            MatrixXd& U = cache->covar_r->rgetUK();
            VectorXd& S = cache->covar_c->rgetSK();
            UdKU.noalias() = U.transpose() * dK * U;
            //start with multiplying Y with Sc
            SYUdKU = Ysi;
            MatrixXd St = MatrixXd::Zero(Ysi.rows(), Ysi.cols());
            St.rowwise() = S.transpose();
            SYUdKU.array() *= St.array();
            //dot product with UdKU
            SYUdKU = UdKU * SYUdKU;
        }
        SYUdKU.array() *= Ysi.array();
        return SYUdKU.sum();
    }

    void CGPSum::_gradQuadrFormX(VectorXd *rv, MatrixXd & dK, bool columns)
    {
        MatrixXd& Ysi = cache->rgetYSi();
        MatrixXd UY;
        MatrixXd UYS;
        MatrixXd UYSYU;
        if(columns){
            MatrixXd& U = cache->covar_c->rgetUK();
            VectorXd& S = cache->covar_r->rgetSK();
            UY.noalias() = U * Ysi.transpose();
            UYS = MatrixXd::Zero(UY.rows(), UY.cols());
            UYS.rowwise() = S.transpose();
        }else{
            MatrixXd& U = cache->covar_r->rgetUK();
            VectorXd& S = cache->covar_c->rgetSK();
            UY.noalias() = U * Ysi;
            UYS = MatrixXd::Zero(UY.rows(), UY.cols());
            UYS.rowwise() = S.transpose();
        }
        UYS.array() *= UY.array();
        UYSYU.noalias() = UYS * UY.transpose();
        MatrixXd trUYSYUdK =UYSYU.array() * dK.transpose().array();
        (*rv) = -2.0*trUYSYUdK.colwise().sum();
    }

    void CGPSum::_gradLogDetX(VectorXd *out, MatrixXd & dK, bool columns)
    {
        MatrixXd& Si = cache->rgetSi();
        if(columns){
            MatrixXd& U = cache->covar_c->rgetUK();
            VectorXd& S = cache->covar_r->rgetSK();
            MatrixXd D = 2.0*U.array() * (dK * U).array();
            (*out).noalias() = S.transpose() * Si * D.transpose();
        }else{
            MatrixXd& U = cache->covar_r->rgetUK();
            VectorXd& S = cache->covar_c->rgetSK();
            MatrixXd D = 2.0*U.array() * (dK * U).array();
            (*out).noalias() = D * Si * S;
        }
    }


    void CGPSum::aLMLgrad_X_r(MatrixXd *out) 
    {
        //0. set output dimensions
        (*out).resize(CGPbase::dataTerm->evaluate().rows(), this->gplvmDimensions_r.rows());
        MatrixXd dKx;
        VectorXd grad_column_quad;
        VectorXd grad_column_logdet;
        for(muint_t ic = 0;ic < (muint_t)((this->gplvmDimensions_r.rows()));ic++)
        {
            muint_t col = gplvmDimensions_r(ic);
            covar_r->aKgrad_X(&dKx, col);
            _gradQuadrFormX(&grad_column_quad, dKx, false);
            _gradLogDetX(&grad_column_logdet, dKx, false);
            (*out).col(ic) = 0.5 * (grad_column_quad + grad_column_logdet)* this->getLik()->getSigmaK2();
        }
    }

    void CGPSum::aLMLgrad_X_c(MatrixXd *out) 
     {
    	//0. set output dimensions
    	(*out).resize(CGPbase::dataTerm->evaluate().cols(), this->gplvmDimensions_c.rows());
    	MatrixXd dKx;
    	VectorXd grad_column_quad;
    	VectorXd grad_column_logdet;
    	for(muint_t ic = 0;ic < (muint_t)((this->gplvmDimensions_c.rows()));ic++)
    	{
    		muint_t col = gplvmDimensions_c(ic);
    		covar_c->aKgrad_X(&dKx, col);
    		_gradQuadrFormX(&grad_column_quad,dKx,true);
    		_gradLogDetX(&grad_column_logdet,dKx,true);
    		(*out).col(ic) = 0.5*(grad_column_quad + grad_column_logdet)* this->getLik()->getSigmaK2();
    	}
    }


    PGPKroneckerCache CGPSum::getCache()
    {
        return cache;
    }

    PCovarianceFunction CGPSum::getCovarC() const
    {
        return covar_c;
    }

    PCovarianceFunction CGPSum::getCovarR() const
    {
        return covar_r;
    }

    VectorXi CGPSum::getGplvmDimensionsC() const
    {
        return gplvmDimensions_c;
    }

    VectorXi CGPSum::getGplvmDimensionsR() const
    {
        return gplvmDimensions_r;
    }


    void CGPSum::setGplvmDimensionsC(VectorXi gplvmDimensionsC)
    {
        gplvmDimensions_c = gplvmDimensionsC;
    }

    void CGPSum::setGplvmDimensionsR(VectorXi gplvmDimensionsR)
    {
        gplvmDimensions_r = gplvmDimensionsR;
    }


 void CGPSum::apredictMean(MatrixXd* out, const MatrixXd& Xstar_r,const MatrixXd& Xstar_c) 
 {
	//1. calc cross variances for row and columns
 	MatrixXd Kstar_r,Kstar_c;
 	this->covar_r->aKcross(&Kstar_r,Xstar_r);
 	this->covar_c->aKcross(&Kstar_c,Xstar_c);
 	std::cout << "matrices loaded" << "\n";
 	//MatrixXd Kstar_rU = Kstar_r;//*this->cache.cache_r.getUK();
 	//MatrixXd Kstar_cU = Kstar_c;//*this->cache.cache_c.getUK();
 	std::cout << "allmost done" << "\n";
 	akronravel(*out,Kstar_r,Kstar_c,cache->rgetKinvY());
 	(*out) *= this->getLik()->getSigmaK2();
 }

 void CGPSum::apredictVar(MatrixXd* out,const MatrixXd& Xstar_r,const MatrixXd& Xstar_c) 
 {
	 throw CLimixException("CGPSum: apredictVar not implemented yet!");
 }

 */


} /* namespace limix */
