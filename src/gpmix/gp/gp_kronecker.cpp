/*
 * gp_kronecker.cpp
 *
 *  Created on: Jan 2, 2012
 *      Author: stegle
 */

#include "gp_kronecker.h"
#include "gpmix/utils/matrix_helper.h"

namespace gpmix {

CGPSVDCache::CGPSVDCache(CGPbase* gp, PCovarianceFunction covar) : CGPCholCache(gp,covar), covar(covar)
{
}



MatrixXd& CGPSVDCache::getUK()
{
	if (!isInSync())
		this->clearCache();
	if (UKNull)
	{
		Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver(getK0());
		UK = eigensolver.eigenvectors();
		SK = eigensolver.eigenvalues();
	}
	return UK;
}

ACovarianceFunction& CGPSVDCache::getCovar()
{
	return (*covar);
}


VectorXd& CGPSVDCache::getSK()
{
	if(!isInSync())
		this->clearCache();

	if(UKNull)
	{
		Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver(getK0());
		UK = eigensolver.eigenvectors();
		SK = eigensolver.eigenvalues();
	}
	return SK;
}

void CGPSVDCache::clearCache()
{
	covar->makeSync();
	K0Null=true;
	UKNull=true;
	SKNull=true;
}

bool CGPSVDCache::isInSync() const
{
	return covar->isInSync();
}


CGPKroneckerCache::CGPKroneckerCache(CGPkronecker* gp, PCovarianceFunction covar_r, PCovarianceFunction covar_c)
:gp(gp), cache_r(gp, covar_r), cache_c(gp, covar_c)
{
}



void CGPKroneckerCache::clearCache()
{
	YrotNull = true;
	SiNull = true;
	YSiNull = true;
	KinvYNull= true;
	gp->dataTerm->makeSync();
}

bool CGPKroneckerCache::isInSync() const
{
	return cache_r.covar->isInSync() && cache_c.covar->isInSync() && gp->dataTerm->isInSync() && gp->lik->isInSync();
}

   	   MatrixXd& CGPKroneckerCache::getYrot()
    {
   		//depnds on covar_r,covar_c,dataterm
      	bool in_sync = cache_r.isInSync() && cache_c.isInSync() && gp->dataTerm->isInSync();
        if(!in_sync)
        {
            akronravel(Yrot, (cache_r.getUK()).transpose(), (cache_c.getUK()).transpose(), gp->dataTerm->evaluate());
        }
        return Yrot;
    }

    MatrixXd& CGPKroneckerCache::getSi()
    {
    	bool is_sync = cache_r.covar->isInSync() && cache_c.covar->isInSync() && gp->lik->isInSync();
    	if(!is_sync)
        {
        	akrondiag(Si, (cache_r.getSK()), (cache_c.getSK()));
        	//1. add Delta
        	Si.array() += gp->lik->getDelta();
        	//2. scale with sigmaK2
#if 1
        	mfloat_t sg = gp->lik->getSigmaK2();
        	std::cout << sg << "using sigmaK2 factor:" << sg << "\n";
        	Si.array()*=sg;
#endif
        	//elementwise inverse
            Si = Si.unaryExpr(std::ptr_fun(inverse));
        }
        return Si;
    }

    MatrixXd& CGPKroneckerCache::getYSi()
    {
    	//sync state depends on all components
    	bool in_sync = cache_r.isInSync() && cache_c.isInSync() && gp->dataTerm->isInSync() && gp->dataTerm->isInSync();

    	if (!in_sync)
    	{
        	MatrixXd& Si   = getSi();
        	MatrixXd& Yrot = getYrot();
            YSi = (Si).array() * (Yrot).array();
        }
        return YSi;
    }

    MatrixXd& CGPKroneckerCache::getKinvY()
    {
    	bool in_sync = cache_r.isInSync() && cache_c.isInSync() && gp->dataTerm->isInSync() && gp->dataTerm->isInSync();
    	if (!in_sync)
    	{
        	MatrixXd& YSi = getYSi();
        	akronravel(KinvY,cache_r.getUK(), cache_c.getUK(),YSi);
        }
        return KinvY;
    }





    CGPkronecker::CGPkronecker(PCovarianceFunction covar_r, PCovarianceFunction covar_c, PLikNormalSVD lik,PDataTerm dataTerm)
    :CGPbase(covar_r, lik,dataTerm), covar_r(covar_r), covar_c(covar_c), cache(this, covar_r, covar_c)
    {
    	//check that likelihood is Iso
        	if (lik)
        	{
        		this->lik = lik;
        	}
        	else
        	{
        		this->lik = PLikNormalSVD(new CLikNormalSVD());
        	}
    }

    CGPkronecker::~CGPkronecker()
    {
        // TODO Auto-generated destructor stub
    }

    void CGPkronecker::updateParams() throw (CGPMixException)
    {
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

    void CGPkronecker::setX_r(const CovarInput & X) throw (CGPMixException)
    {
        this->covar_r->setX(X);
        if(isnull(gplvmDimensions_r))
            this->gplvmDimensions_r = VectorXi::LinSpaced(X.cols(), 0, X.cols() - 1);

    }
    void CGPkronecker::setX_c(const CovarInput & X) throw (CGPMixException)
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
    	this->cache.cache_r.setCovar(covar);
    }
    void CGPkronecker::setCovar_c(PCovarianceFunction covar)
    {
        	this->covar_c = covar;
        	this->cache.cache_c.setCovar(covar);
    }



    mfloat_t CGPkronecker::LML() throw (CGPMixException)
    {
        //update the covariance parameters
        MatrixXd& Si = cache.getSi();

        //1. logdet:
        //loop through entries of Si: note we Si has non-vec shape, so we use the raw interface for this:
        muint_t size = Si.rows()*Si.cols();
        mfloat_t lml_det = 0;
        for(mfloat_t* Siraw = Si.data(); Siraw < Si.data()+size;++Siraw)
        {
            lml_det += gpmix::log(*Siraw);
        }
        lml_det *= -0.5;
        //2. quadratic term
        MatrixXd LMLq = cache.getYrot();
        LMLq.array() *= cache.getYSi().array();
        mfloat_t lml_quad = 0.5 * LMLq.sum();
        //3. constants
        mfloat_t lml_const = 0.5* LMLq.cols() * LMLq.rows() * gpmix::log((2.0 * PI));

        return lml_quad + lml_det + lml_const;
    };

    CGPHyperParams CGPkronecker::LMLgrad() throw (CGPMixException)
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
        MatrixXd& Si = cache.getSi();
        MatrixXd rv;
        if(columns){
        	MatrixXd& U= cache.cache_c.getUK();
            VectorXd& S = cache.cache_r.getSK();
            MatrixXd d = (U.array() * (dK * U).array()).colwise().sum();
            rv.noalias() = S.transpose() * Si * d.transpose();
        }else{
            MatrixXd& U = cache.cache_r.getUK();
            VectorXd& S = cache.cache_c.getSK();
            MatrixXd d = (U.array() * (dK * U).array()).colwise().sum();
            rv.noalias() = d* Si * S;
        }
        return rv(0, 0);
    }

    mfloat_t CGPkronecker::_gradQuadrForm(MatrixXd & dK, bool columns)
    {
        MatrixXd& Ysi = cache.getYSi();
        MatrixXd UdKU;
        MatrixXd SYUdKU;
        if(columns){
            MatrixXd& U = cache.cache_c.getUK();
            VectorXd& S = cache.cache_r.getSK();
            UdKU.noalias() = U.transpose() * dK * U;
            //start with multiplying Y with Sc
            SYUdKU = Ysi;
            MatrixXd St = MatrixXd::Zero(Ysi.rows(), Ysi.cols());
            St.colwise() = S;
            SYUdKU.array() *= St.array();
            //dot product with UdKU
            SYUdKU = SYUdKU * UdKU.transpose();
        }else{
            MatrixXd& U = cache.cache_r.getUK();
            VectorXd& S = cache.cache_c.getSK();
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
        MatrixXd& Ysi = cache.getYSi();
        MatrixXd UY;
        MatrixXd UYS;
        MatrixXd UYSYU;
        if(columns){
            MatrixXd& U = cache.cache_c.getUK();
            VectorXd& S = cache.cache_r.getSK();
            UY.noalias() = U * Ysi.transpose();
            UYS = MatrixXd::Zero(UY.rows(), UY.cols());
            UYS.rowwise() = S.transpose();
        }else{
            MatrixXd& U = cache.cache_r.getUK();
            VectorXd& S = cache.cache_c.getSK();
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
        MatrixXd& Si = cache.getSi();
        if(columns){
            MatrixXd& U = cache.cache_c.getUK();
            VectorXd& S = cache.cache_r.getSK();
            MatrixXd D = 2.0*U.array() * (dK * U).array();
            (*out).noalias() = S.transpose() * Si * D.transpose();
        }else{
            MatrixXd& U = cache.cache_r.getUK();
            VectorXd& S = cache.cache_c.getSK();
            MatrixXd D = 2.0*U.array() * (dK * U).array();
            (*out).noalias() = D * Si * S;
        }
    }

    void CGPkronecker::aLMLgrad_covar(VectorXd *out, bool columns) throw (CGPMixException)
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
            //std::cout << "grad("<<param<<"), col="<< columns<< "; grad_quad:"<< grad_quad <<";"<< "grad_logdet:"<< grad_logdet << "\n";
            (*out)[param] = grad_logdet + grad_quad;
        }
    }

    void CGPkronecker::aLMLgrad_lik(VectorXd *out) throw (CGPMixException)
    {
        //TODO: we can only treat the boring standard noise level in this variant
        out->resize(lik->getNumberParams());
        // calc gradient manually, ..
        //MatrixXd dK_ = lik->Kgrad_param(0);
        //TODO:
        mfloat_t dK = 2.0*gpmix::exp( (mfloat_t)(2.0*lik->getParams()(0)));
        MatrixXd& Si = cache.getSi();
        MatrixXd& YSi = cache.getYSi();
        mfloat_t grad_logdet = 0.5 * dK * Si.sum();
        MatrixXd YSiYSi = YSi;
        YSiYSi.array() *= YSi.array();
        mfloat_t grad_quad = -0.5 * dK * YSiYSi.sum();
        (*out)(0) = grad_quad + grad_logdet;
    }

/*
 * this should be the new code...
    void CGPkronecker::aLMLgrad_lik(VectorXd *out) throw (CGPMixException)
      {
          //TODO: we can only treat the boring standard noise level in this variant
          out->resize(lik->getNumberParams());
          //inner derivatives w.r.t Sigma and Delta
          //mfloat_t dSigmaK2 = lik->getSigmaK2grad();
          mfloat_t dDelta   = lik->getDeltagrad();
          mfloat_t SigmaK2 = lik->getSigmaK2();
          //mfloat_t Delta   = lik->getDelta();

          MatrixXd& Si = cache.getSi();
          MatrixXd& YSi = cache.getYSi();
          MatrixXd& Y = cache.getYrot();


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

          mfloat_t grad_sigmak2_quad = -0.5* 2.0 * YSiYSi.sum() / (this->lik->getSigmaK2());
          (*out)(0) = grad_sigmak2_logdet + grad_sigmak2_quad;
          (*out)(1) = grad_delta_logdet + grad_delta_quad;
      }
 */



    void CGPkronecker::aLMLgrad_covar_r(VectorXd *out) throw (CGPMixException)
    {
        aLMLgrad_covar(out, false);
    }

    void CGPkronecker::aLMLgrad_covar_c(VectorXd *out) throw (CGPMixException)
    {
        aLMLgrad_covar(out, true);
    }

    void CGPkronecker::aLMLgrad_X_r(MatrixXd *out) throw (CGPMixException)
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
            (*out).col(ic) = 0.5 * (grad_column_quad + grad_column_logdet);
        }
    }

    CGPKroneckerCache& CGPkronecker::getCache()
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

 void CGPkronecker::aLMLgrad_X_c(MatrixXd *out) throw (CGPMixException)
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
		//std::cout << "_gradQuadrFormX("<< ic << ")=" << grad_column_quad << "\n";
		//std::cout << "_gradLogDetX("<< ic << ")=" << grad_column_logdet << "\n";
		(*out).col(ic) = 0.5*(grad_column_quad + grad_column_logdet);
	}
}

 void CGPkronecker::aLMLgrad_dataTerm(MatrixXd* out) throw (CGPMixException)
{
 	//0. set output dimensions
#if 1
	 (*out) = this->dataTerm->gradParams(this->cache.getKinvY());
#else
	 (*out) =
#endif
}

} /* namespace gpmix */
