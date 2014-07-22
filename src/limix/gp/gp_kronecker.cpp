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

#include "gp_kronecker.h"
#include "limix/utils/matrix_helper.h"

namespace limix {



CGPKroneckerCache::CGPKroneckerCache(CGPkronecker* gp)
{
	this->syncCovar_r = Pbool(new bool);
	this->syncCovar_c = Pbool(new bool);
	this->syncLik = Pbool(new bool);
	this->syncData = Pbool(new bool);
	//add to list of sync parents
	this->addSyncParent(syncCovar_r);
	this->addSyncParent(syncCovar_c);
	this->addSyncParent(syncLik);
	this->addSyncParent(syncData);


	this->gp = gp;
	this->covar_r = PCovarianceFunctionCacheOld(new CCovarianceFunctionCacheOld(this->gp->covar_r));
	this->covar_c = PCovarianceFunctionCacheOld(new CCovarianceFunctionCacheOld(this->gp->covar_c));
	//add sync liestener
	covar_r->addSyncChild(this->syncCovar_r);
	covar_c->addSyncChild(this->syncCovar_c);
	this->gp->lik->addSyncChild(this->syncLik);
	this->gp->dataTerm->addSyncChild(this->syncData);
	//set all cache Variables to Null
	YSiCacheNull=true;
	KinvYCacheNull=true;
	KinvYCacheNull= true;
	SiCacheNull=true;
	YrotCacheNull=true;
}



void CGPKroneckerCache::validateCache()
{
	//std::cout << *syncCovar_c << "," << *syncCovar_r <<"," << *syncLik << "," << *syncData <<"\n";

	//1. variables that depend on any of the caches
	if((! *syncCovar_c) || (! *syncCovar_r) || (! *syncLik) || (! *syncData))
	{
		YSiCacheNull=true;
		KinvYCacheNull=true;
	}
	//covar or lik
	if((! *syncCovar_c) || (! *syncCovar_r) || (! *syncLik))
	{
		SiCacheNull=true;
	}
	//covar or data
	if((! *syncCovar_c) || (! *syncCovar_r) || (! *syncData))
	{
		YrotCacheNull=true;
	}
	//set all sync
	setSync();
	//std::cout << *syncCovar_c << "," << *syncCovar_r <<"," << *syncLik << "," << *syncData <<"\n";
}

MatrixXd& CGPKroneckerCache::rgetYrot()
{
	validateCache();
	if(YrotCacheNull)
    {
            akronravel(YrotCache, (covar_r->rgetUK()).transpose(), (covar_c->rgetUK()).transpose(), gp->dataTerm->evaluate());
            YrotCacheNull=false;
    }
        return YrotCache;
}

MatrixXd& CGPKroneckerCache::rgetSi()
{
	validateCache();
	if(SiCacheNull)
    {
		akrondiag(SiCache, (covar_r->rgetSK()), (covar_c->rgetSK()));
        //1. add Delta
        SiCache.array() += gp->getLik()->getDelta();
        //2. scale with sigmaK2
        SiCache.array()*=gp->getLik()->getSigmaK2();
        //elementwise inverse
        SiCache = SiCache.unaryExpr(std::ptr_fun(inverse));
        SiCacheNull=false;
    }
    return SiCache;
}

    MatrixXd& CGPKroneckerCache::rgetYSi()
    {
    	validateCache();
    	if(YSiCacheNull)
    	{
        	MatrixXd& Si   = rgetSi();
        	MatrixXd& Yrot = rgetYrot();
            YSiCache = (Si).array() * (Yrot).array();
            YSiCacheNull=false;
        }
        return YSiCache;
    }

    MatrixXd& CGPKroneckerCache::rgetKinvY()
    {
    	validateCache();
    	if(KinvYCacheNull)
    	{
        	MatrixXd& YSi = rgetYSi();
        	akronravel(KinvYCache,covar_r->rgetUK(), covar_c->rgetUK(),YSi);
        	KinvYCacheNull=false;
        }
        return KinvYCache;
    }





    CGPkronecker::CGPkronecker(PCovarianceFunction covar_r, PCovarianceFunction covar_c, PLikelihood lik,PDataTerm dataTerm)
    :CGPbase(covar_r, lik,dataTerm), covar_r(covar_r), covar_c(covar_c)
    {
      	if (lik)
       	{
       		this->lik = lik;
       	}
       	else
       	{
       		this->lik = PLikNormalSVD(new CLikNormalSVD());
       	}
       	if (typeid(*(this->lik))!=typeid(CLikNormalSVD))
       	    throw CLimixException("CGPLMM requires a SVD likelihood term");
       	cache = PGPKroneckerCache(new CGPKroneckerCache(this));
    }

    CGPkronecker::~CGPkronecker()
    {
        // TODO Auto-generated destructor stub
    }

    void CGPkronecker::updateParams() 
    {
    	//std::cout << params << "\n";

        CGPbase::updateParams();
        if(this->params.exists("covar_r"))
            this->covar_r->setParams(this->params["covar_r"]);

        if(this->params.exists("covar_c"))
            this->covar_c->setParams(this->params["covar_c"]);

        if(params.exists("X_r"))
        {
        	this->updateX(*covar_r, gplvmDimensions_r, params["X_r"]);
        }

        if(params.exists("X_c"))
            this->updateX(*covar_c, gplvmDimensions_c, params["X_c"]);

        if(this->params.exists("dataTerm"))
        {
        	this->dataTerm->setParams(this->params["dataTerm"]);
        }
    }

    void CGPkronecker::setX_r(const CovarInput & X) 
    {
        this->covar_r->setX(X);
        if(isnull(gplvmDimensions_r))
            this->gplvmDimensions_r = VectorXi::LinSpaced(X.cols(), 0, X.cols() - 1);

    }
    void CGPkronecker::setX_c(const CovarInput & X) 
    {
        this->covar_c->setX(X);
        if(isnull(gplvmDimensions_c))
            this->gplvmDimensions_c = VectorXi::LinSpaced(X.cols(), 0, X.cols() - 1);

    }
    void CGPkronecker::setY(const MatrixXd & Y)
    {
        CGPbase::dataTerm->setY(Y);
        this->lik->setX(MatrixXd::Zero(Y.rows() * Y.cols(), 0));
    }
    void CGPkronecker::setCovar_r(PCovarianceFunction covar)
    {
    	this->covar_r = covar;
    	this->cache->covar_r->setCovar(covar);
    }
    void CGPkronecker::setCovar_c(PCovarianceFunction covar)
    {
        	this->covar_c = covar;
        	this->cache->covar_c->setCovar(covar);
    }



    mfloat_t CGPkronecker::LML() 
    {
        //update the covariance parameters
        MatrixXd& Si = cache->rgetSi();

        //1. logdet:
        //loop through entries of Si: note we Si has non-vec shape, so we use the raw interface for this:
        muint_t size = Si.rows()*Si.cols();
        mfloat_t lml_det = 0;
        for(mfloat_t* Siraw = Si.data(); Siraw < Si.data()+size;++Siraw)
        {
            lml_det += limix::log(*Siraw);
        }
        lml_det *= -0.5;
        //2. quadratic term
        MatrixXd LMLq = cache->rgetYrot();
        LMLq.array() *= cache->rgetYSi().array();
        mfloat_t lml_quad = 0.5 * LMLq.sum();
        //3. constants
        mfloat_t lml_const = 0.5* LMLq.cols() * LMLq.rows() * limix::log((2.0 * PI));

        return lml_quad + lml_det + lml_const;
    };

    CGPHyperParams CGPkronecker::LMLgrad() 
    {
        CGPHyperParams rv;
        //calculate gradients for parameter components in params:
        if(params.exists("covar_r")){
            VectorXd grad_covar;
            aLMLgrad_covar_r(&grad_covar);
            rv.set("covar_r", grad_covar);
        }
        if(params.exists("covar_c")){
            VectorXd grad_covar;
            aLMLgrad_covar_c(&grad_covar);
            rv.set("covar_c", grad_covar);
        }
        if(params.exists("lik")){
            VectorXd grad_lik;
            aLMLgrad_lik(&grad_lik);
            rv.set("lik", grad_lik);
        }
        if(params.exists("X_r")){
            MatrixXd grad_X;
            aLMLgrad_X_r(&grad_X);
            rv.set("X_r", grad_X);
        }
        if(params.exists("X_c")){
            MatrixXd grad_X;
            aLMLgrad_X_c(&grad_X);
            rv.set("X_c", grad_X);
        }
        if (params.exists("dataTerm"))
        {
        	MatrixXd grad_dataTerm;
        	aLMLgrad_dataTerm(&grad_dataTerm);
        	rv.set("dataTerm",grad_dataTerm);
        }
        return rv;
    }

    mfloat_t CGPkronecker::_gradLogDet(MatrixXd & dK, bool columns)
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

    mfloat_t CGPkronecker::_gradQuadrForm(MatrixXd & dK, bool columns)
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

    void CGPkronecker::_gradQuadrFormX(VectorXd *rv, MatrixXd & dK, bool columns)
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

    void CGPkronecker::_gradLogDetX(VectorXd *out, MatrixXd & dK, bool columns)
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

    void CGPkronecker::aLMLgrad_covar(VectorXd *out, bool columns) 
    {
        PCovarianceFunction covar;
        if(columns)
            covar = covar_c;

        else
            covar = covar_r;

        MatrixXd dK;
        mfloat_t grad_logdet;
        mfloat_t grad_quad;
        (*out).resize(covar->getNumberParams());
        for(muint_t param = 0;param < (muint_t)(((covar->getNumberParams())));param++)
        {
            //1. get gradient matrix
            covar->aKgrad_param(&dK, param);
            //2. caclculate grad_quad and grad_logdet component
            grad_logdet = 0.5 * _gradLogDet(dK, columns);
            grad_quad = -0.5 * _gradQuadrForm(dK, columns);

            //(*out)[param] = (grad_logdet + grad_quad)
            //The gradients change with the sigma2:
            (*out)[param] = (grad_logdet + grad_quad) * this->getLik()->getSigmaK2();
        }
    }

    // this should be the new code...
    void CGPkronecker::aLMLgrad_lik(VectorXd *out) 
      {
          out->resize(lik->getNumberParams());
          //inner derivatives w.r.t Sigma and Delta
          mfloat_t dDelta   = getLik()->getDeltagrad();
          mfloat_t SigmaK2 = getLik()->getSigmaK2();
          MatrixXd& Si = cache->rgetSi();
          MatrixXd& YSi = cache->rgetYSi();
          MatrixXd& Y = cache->rgetYrot();
          //logdet
          mfloat_t grad_delta_logdet   = 0.5 * dDelta * SigmaK2* Si.sum();
          mfloat_t grad_sigmak2_logdet = Si.rows()*Si.cols();
          //gradquad
          //delta
          MatrixXd YSiYSi = YSi;
          YSiYSi.array() *= YSi.array();
          mfloat_t grad_delta_quad   = -0.5 * dDelta * SigmaK2 * YSiYSi.sum();
          //sigmaK2
          YSiYSi = YSi;
          YSiYSi.array()*=Y.array();
          mfloat_t grad_sigmak2_quad = -0.5 * SigmaK2 * 2.0 * YSiYSi.sum() / (getLik()->getSigmaK2());
          (*out)(0) = grad_sigmak2_logdet + grad_sigmak2_quad;
          (*out)(1) = grad_delta_logdet + grad_delta_quad;
      }

    void CGPkronecker::aLMLgrad_covar_r(VectorXd *out) 
    {
        aLMLgrad_covar(out, false);
    }

    void CGPkronecker::aLMLgrad_covar_c(VectorXd *out) 
    {
        aLMLgrad_covar(out, true);
    }

    void CGPkronecker::aLMLgrad_X_r(MatrixXd *out) 
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

    void CGPkronecker::aLMLgrad_X_c(MatrixXd *out) 
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


    PGPKroneckerCache CGPkronecker::getCache()
    {
        return cache;
    }

    PCovarianceFunction CGPkronecker::getCovarC() const
    {
        return covar_c;
    }

    PCovarianceFunction CGPkronecker::getCovarR() const
    {
        return covar_r;
    }

    VectorXi CGPkronecker::getGplvmDimensionsC() const
    {
        return gplvmDimensions_c;
    }

    VectorXi CGPkronecker::getGplvmDimensionsR() const
    {
        return gplvmDimensions_r;
    }


    void CGPkronecker::setGplvmDimensionsC(VectorXi gplvmDimensionsC)
    {
        gplvmDimensions_c = gplvmDimensionsC;
    }

    void CGPkronecker::setGplvmDimensionsR(VectorXi gplvmDimensionsR)
    {
        gplvmDimensions_r = gplvmDimensionsR;
    }


 void CGPkronecker::aLMLgrad_dataTerm(MatrixXd* out) 
{
 	//0. set output dimensions
	 (*out) = this->dataTerm->gradParams(this->cache->rgetKinvY());
}


 void CGPkronecker::apredictMean(MatrixXd* out, const MatrixXd& Xstar_r,const MatrixXd& Xstar_c) 
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

 void CGPkronecker::apredictVar(MatrixXd* out,const MatrixXd& Xstar_r,const MatrixXd& Xstar_c) 
 {
	 throw CLimixException("CGPKronecker: apredictVar not implemented yet!");
 }




} /* namespace limix */
