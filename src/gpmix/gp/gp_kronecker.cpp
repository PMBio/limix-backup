/*
 * gp_kronecker.cpp
 *
 *  Created on: Jan 2, 2012
 *      Author: stegle
 */

#include "gp_kronecker.h"
#include "gpmix/utils/matrix_helper.h"

namespace gpmix {


MatrixXd* CGPSVDCache::getUK()
{
	if (!isInSync())
		this->clearCache();
	if (isnull(UK))
	{
		Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver((*getK0()));
		UK = eigensolver.eigenvectors();
		SK = eigensolver.eigenvalues();
	}
	return &UK;
}

    ACovarianceFunction *CGPSVDCache::getCovar() const
    {
        return covar;
    }


    VectorXd *CGPSVDCache::getSK()
    {
        if(!isInSync())
            this->clearCache();

        if(isnull(UK)){
            Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver((*getK0()));
            UK = eigensolver.eigenvectors();
            SK = eigensolver.eigenvalues();
        }
        return &SK;
    }

    void CGPSVDCache::clearCache()
    {
        covar->makeSync();
        K = MatrixXd();
        UK = MatrixXd();
        SK = VectorXd();
    }

    bool CGPSVDCache::isInSync() const
    {
        return covar->isInSync();
    }

    void CGPKroneckerCache::clearCache()
    {
        Yrot = MatrixXd();
        Si = MatrixXd();
        YSi = MatrixXd();
    }

    bool CGPKroneckerCache::isInSync() const
    {
        return cache_r.covar->isInSync() && cache_c.covar->isInSync();
    }

    MatrixXd *CGPKroneckerCache::getYrot()
    {
        if(!isInSync())
            this->clearCache();

        if(isnull(Yrot)){
            akronravel(&Yrot, (*cache_r.getUK()).transpose(), (*cache_c.getUK()).transpose(), gp->Y);
        }
        return &Yrot;
    }

    MatrixXd *CGPKroneckerCache::getSi()
    {
        if(!isInSync())
            this->clearCache();

        if(isnull(Si)){
            akrondiag(&Si, *(cache_r.getSK()), *(cache_c.getSK()));
            //add noise
            Si.array() += getKnoise();
            //elementwise inversion:
            Si = Si.unaryExpr(ptr_fun(inverse));
        }
        return &Si;
    }

    MatrixXd *CGPKroneckerCache::getYSi()
    {
        if(!isInSync())
            this->clearCache();

        if(isnull(YSi)){
        	MatrixXd* Si   = getSi();
        	MatrixXd* Yrot = getYrot();
            YSi = (*Si).array() * (*Yrot).array();
        }
        return &YSi;
    }

    mfloat_t CGPKroneckerCache::getKnoise()
    {
        if(!gp->lik.isInSync()){
            gp->lik.makeSync();
        }
    	Knoise = gpmix::exp( (mfloat_t)(2.0*gp->lik.getParams()(0)));
        return Knoise;
    }

    CGPKroneckerCache::CGPKroneckerCache(CGPbase *gp, ACovarianceFunction *covar_r, ACovarianceFunction *covar_c)
    :gp(gp), cache_r(gp, covar_r), cache_c(gp, covar_c)
    {
    }

    CGPkronecker::CGPkronecker(ACovarianceFunction & covar_r, ACovarianceFunction & covar_c, ALikelihood & lik)
    :CGPbase(covar_r, lik), covar_r(covar_r), covar_c(covar_c), cache(this, &covar_r, &covar_c)
    {
    }

    CGPkronecker::~CGPkronecker()
    {
        // TODO Auto-generated destructor stub
    }

    void CGPkronecker::updateParams() throw (CGPMixException)
    {
        CGPbase::updateParams();
        if(this->params.exists("covar_r"))
            this->covar_r.setParams(this->params["covar_r"]);

        if(this->params.exists("covar_c"))
            this->covar_c.setParams(this->params["covar_c"]);

        if(params.exists("X_r"))
        {
        	this->updateX(covar_r, gplvmDimensions_r, params["X_r"]);
        }

        if(params.exists("X_c"))
            this->updateX(covar_c, gplvmDimensions_c, params["X_c"]);

    }
    void CGPkronecker::setX_r(const CovarInput & X) throw (CGPMixException)
    {
        this->covar_r.setX(X);
        if(isnull(gplvmDimensions_r))
            this->gplvmDimensions_r = VectorXi::LinSpaced(X.cols(), 0, X.cols() - 1);

    }
    void CGPkronecker::setX_c(const CovarInput & X) throw (CGPMixException)
    {
        this->covar_c.setX(X);
        if(isnull(gplvmDimensions_c))
            this->gplvmDimensions_c = VectorXi::LinSpaced(X.cols(), 0, X.cols() - 1);

    }
    void CGPkronecker::setY(const MatrixXd & Y)
    {
        CGPbase::setY(Y);
        this->lik.setX(MatrixXd::Zero(Y.rows() * Y.cols(), 0));
    }

    mfloat_t CGPkronecker::LML() throw (CGPMixException)
    {
        //update the covariance parameters
        MatrixXd *Si = cache.getSi();

        //1. logdet:
        //loop through entries of Si: note we Si has non-vec shape, so we use the raw interface for this:
        muint_t size = Si->rows()*Si->cols();
        mfloat_t lml_det = 0;
        for(mfloat_t* Siraw = Si->data(); Siraw < Si->data()+size;++Siraw)
        {
            lml_det += gpmix::log(*Siraw);
        }
        lml_det *= -0.5;
        //2. quadratic term
        MatrixXd LMLq = (*cache.getYrot());
        LMLq.array() *= (*cache.getYSi()).array();
        mfloat_t lml_quad = 0.5 * LMLq.sum();
        //3. constants
        mfloat_t lml_const = 0.5*Y.cols() * Y.rows() * gpmix::log((2.0 * PI));

        //std::cout << "lml_quad:" << lml_quad << "lml_det:" << lml_det <<","<<"lml_const:"<<lml_const << "\n";
        return lml_quad + lml_det + lml_const;
    }
    ;
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
        return rv;
    }

    mfloat_t CGPkronecker::_gradLogDet(MatrixXd & dK, bool columns)
    {
        MatrixXd *Si = cache.getSi();
        MatrixXd *U;
        VectorXd *S;
        MatrixXd rv;
        if(columns){
            U = cache.cache_c.getUK();
            S = cache.cache_r.getSK();
            MatrixXd d = ((*U).array() * (dK * (*U)).array()).colwise().sum();
            rv = (*S).transpose() * (*Si) * d.transpose();
        }else{
            U = cache.cache_r.getUK();
            S = cache.cache_c.getSK();
            MatrixXd d = ((*U).array() * (dK * (*U)).array()).colwise().sum();
            rv = d* (*Si) * (*S);
        }
        return rv(0, 0);
    }

    mfloat_t CGPkronecker::_gradQuadrForm(MatrixXd & dK, bool columns)
    {
        MatrixXd *Ysi = cache.getYSi();
        MatrixXd UdKU;
        MatrixXd SYUdKU;
        if(columns){
            MatrixXd *U = cache.cache_c.getUK();
            VectorXd *S = cache.cache_r.getSK();
            UdKU = (*U).transpose() * dK * (*U);
            //start with multiplying Y with Sc
            SYUdKU = (*Ysi);
            MatrixXd St = MatrixXd::Zero((*Ysi).rows(), (*Ysi).cols());
            St.colwise() = (*S);
            SYUdKU.array() *= St.array();
            //dot product with UdKU
            SYUdKU = SYUdKU * UdKU.transpose();
        }else{
            MatrixXd *U = cache.cache_r.getUK();
            VectorXd *S = cache.cache_c.getSK();
            UdKU = (*U).transpose() * dK * (*U);
            //start with multiplying Y with Sc
            SYUdKU = (*Ysi);
            MatrixXd St = MatrixXd::Zero((*Ysi).rows(), (*Ysi).cols());
            St.rowwise() = (*S);
            SYUdKU.array() *= St.array();
            //dot product with UdKU
            SYUdKU = UdKU * SYUdKU;
        }
        SYUdKU.array() *= (*Ysi).array();
        return SYUdKU.sum();
    }

    void CGPkronecker::_gradQuadrFormX(VectorXd *rv, MatrixXd & dK, bool columns)
    {
        MatrixXd *Ysi = cache.getYSi();
        MatrixXd UY;
        MatrixXd UYS;
        MatrixXd UYSYU;
        if(columns){
            MatrixXd *U = cache.cache_c.getUK();
            VectorXd *S = cache.cache_r.getSK();
            UY = (*U) * (*Ysi).transpose();
            UYS = MatrixXd::Zero(UY.rows(), UY.cols());
            UYS.rowwise() = (*S);
        }else{
            MatrixXd *U = cache.cache_r.getUK();
            VectorXd *S = cache.cache_c.getSK();
            UY = (*U) * (*Ysi);
            UYS = MatrixXd::Zero(UY.rows(), UY.cols());
            UYS.rowwise() = (*S);
        }
        UYS.array() *= UY.array();
        UYSYU = UYS * UY.transpose();
        MatrixXd trUYSYUdK =UYSYU.array() * dK.transpose().array();
        (*rv) = -2.0*trUYSYUdK.colwise().sum();
    }

    void CGPkronecker::_gradLogDetX(VectorXd *out, MatrixXd & dK, bool columns)
    {
        MatrixXd *Si = cache.getSi();
        if(columns){
            MatrixXd *U = cache.cache_c.getUK();
            VectorXd *S = cache.cache_r.getSK();
            MatrixXd D = 2.0*(*U).array() * (dK * (*U)).array();
            (*out) = (*S).transpose() * (*Si) * D.transpose();
        }else{
            MatrixXd *U = cache.cache_r.getUK();
            VectorXd *S = cache.cache_c.getSK();
            MatrixXd D = 2.0*(*U).array() * (dK * (*U)).array();
            (*out) = D * (*Si) * (*S);
        }
    }

    void CGPkronecker::aLMLgrad_covar(VectorXd *out, bool columns) throw (CGPMixException)
    {
        ACovarianceFunction *covar;
        if(columns)
            covar = &covar_c;

        else
            covar = &covar_r;

        MatrixXd dK;
        mfloat_t grad_logdet;
        mfloat_t grad_quad;
        (*out).resize(covar->getNumberParams());
        for(muint_t param = 0;param < (muint_t)(((covar->getNumberParams())));param++){
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
        out->resize(lik.getNumberParams());
        MatrixXd dK_ = lik.Kgrad_param(0);
        mfloat_t dK = dK_(0, 0);
        MatrixXd *Si = cache.getSi();
        MatrixXd *YSi = cache.getYSi();
        mfloat_t grad_logdet = 0.5 * dK * (*Si).sum();
        MatrixXd YSiYSi = (*YSi);
        YSiYSi.array() *= (*YSi).array();
        mfloat_t grad_quad = -0.5 * dK * YSiYSi.sum();
        (*out)(0) = grad_quad + grad_logdet;
    }

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
        (*out).resize(Y.rows(), this->gplvmDimensions_r.rows());
        MatrixXd dKx;
        VectorXd grad_column_quad;
        VectorXd grad_column_logdet;
        for(muint_t ic = 0;ic < (muint_t)((this->gplvmDimensions_r.rows()));ic++){
            muint_t col = gplvmDimensions_r(ic);
            covar_r.aKgrad_X(&dKx, col);
            _gradQuadrFormX(&grad_column_quad, dKx, false);
            _gradLogDetX(&grad_column_logdet, dKx, false);
            (*out).col(ic) = 0.5 * (grad_column_quad + grad_column_logdet);
        }
    }

    CGPKroneckerCache CGPkronecker::getCache() const
    {
        return cache;
    }

    ACovarianceFunction & CGPkronecker::getCovarC() const
    {
        return covar_c;
    }

    ACovarianceFunction & CGPkronecker::getCovarR() const
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
        (*out).resize(Y.cols(), this->gplvmDimensions_c.rows());
        MatrixXd dKx;
        VectorXd grad_column_quad;
        VectorXd grad_column_logdet;
        for(muint_t ic = 0;ic < (muint_t)((this->gplvmDimensions_c.rows()));ic++)
		{
			muint_t col = gplvmDimensions_c(ic);
			covar_c.aKgrad_X(&dKx, col);
			_gradQuadrFormX(&grad_column_quad,dKx,true);
			_gradLogDetX(&grad_column_logdet,dKx,true);
			//std::cout << "_gradQuadrFormX("<< ic << ")=" << grad_column_quad << "\n";
			//std::cout << "_gradLogDetX("<< ic << ")=" << grad_column_logdet << "\n";
			(*out).col(ic) = 0.5*(grad_column_quad + grad_column_logdet);
		}
}



} /* namespace gpmix */
