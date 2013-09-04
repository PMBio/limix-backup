#include "kronecker_lmm.h"

namespace limix {

	CKroneckerLMM::CKroneckerLMM() {
	}

	CKroneckerLMM::~CKroneckerLMM() {
	}

	void CKroneckerLMM::process() throw (CGPMixException){

	}

	void CKroneckerLMM::updateDecomposition() throw(CGPMixException) {
        //check that dimensions match
        this->num_samples = snps.rows();
        this->num_snps = snps.cols();
        this->num_pheno = pheno.cols();
        this->num_covs = covs.cols();
        if(!num_samples == pheno.rows())
            throw new CGPMixException("phenotypes and SNP dimensions inconsistent");

        if(!num_samples == covs.rows())
            throw CGPMixException("covariates and SNP dimensions inconsistent");

        if(!(this->UK_cached)){
            //decomposition of K
			Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver2c(K2c);
            this->U2c = eigensolver2c.eigenvectors();
            this->S2c = eigensolver2c.eigenvalues();
			if (this->S2c(0)<=1e-12){
				throw new CGPMixException("The column covariance of the second covariance term has to be full rank, but is not.");
			}

			Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver2r(K2r);
            this->U2r = eigensolver2r.eigenvectors();
            this->S2r = eigensolver2r.eigenvalues();
			if (this->S2r(0)<=1e-12){
				throw new CGPMixException("The row covariance of the second covariance term has to be full rank, but is not.");
			}
            
			this->S2U2K1r = U2r.transpose() * this->K1r * U2r;
			for (size_t r1=0;r1<this->pheno.rows();++r1)
			{
				for (size_t r2=0;r2<this->pheno.rows();++r2)
				{
					this->S2U2K1r(r1,r2)/=sqrt(S2r(r1));
					this->S2U2K1r(r1,r2)/=sqrt(S2r(r2));
				}
			}

			this->S2U2K1c = U2c.transpose() * this->K1c * U2c;
			for (size_t c1=0;c1<this->pheno.cols();++c1)
			{
				for (size_t c2=0;c2<this->pheno.cols();++c2)
				{
					this->S2U2K1r(c1,c2)/=sqrt(S2r(c1));
					this->S2U2K1r(c1,c2)/=sqrt(S2r(c2));
				}
			}

			Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver1c(S2U2K1c);
            this->U1c = eigensolver1c.eigenvectors();
            this->S1c = eigensolver1c.eigenvalues();

			Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver1r(S2U2K1r);
            this->U1r = eigensolver1r.eigenvectors();
            this->S1r = eigensolver1r.eigenvalues();
            
			this->UK_cached = true;
        }
        
		if(!Usnps_cached){
            Usnps.noalias() = U2r.transpose() * snps;
			for (size_t r=0;r<this->pheno.rows();++r)
			{
				for (size_t s=0;s<this->snps.cols();++s)
				{
					this->Usnps(r,s)/=sqrt(S2r(s));
				}
			}
			Usnps.noalias()= U1r.transpose() * Usnps;
            Usnps_cached = true;
        }
        if(!Upheno_cached){
            Upheno.noalias() = U2r.transpose() * pheno * U2c;
			for (size_t r=0;r<this->pheno.rows();++r)
			{
				for (size_t c=0;c<this->pheno.cols();++c)
				{
					this->Upheno(r,c)/=sqrt(S2r(r));
					this->Upheno(r,c)/=sqrt(S2c(c));
				}
			}
			Upheno.noalias()= U1r.transpose() * Upheno *U1c;
			Upheno_cached = true;
        }
        if(!Ucovs_cached){
            Ucovs.noalias() = U2r.transpose() * covs;
			for (size_t r=0;r<this->pheno.rows();++r)
			{
				for (size_t d=0;d<this->covs.cols();++d)
				{
					this->Ucovs(r,d)/=sqrt(S2r(r));
				}
			}
			Ucovs.noalias()= U1r.transpose() * Ucovs;
            Ucovs_cached = true;
        }
		//need column design matrix
		if(!Ucoldesign_cached){//for covariates
            Ucoldesign.noalias() = coldesign * U2c;
            for (size_t r=0;r<this->Ucoldesign.rows();++r)
			{
				for (size_t d=0;d<this->Ucoldesign.cols();++d)
				{
					this->Ucoldesign(r,d)/=sqrt(S2c(d));
				}
			}
			Ucoldesign.noalias() = Ucoldesign * U1c;
            Ucoldesign_cached = true;
        }
		//need column design matrix
		if(!Usnpcoldesign_cached){//design for SNPs
            Usnpcoldesign.noalias() = snpcoldesign * U2c;
            for (size_t r=0;r<this->Usnpcoldesign.rows();++r)//should be 1
			{
				for (size_t d=0;d<this->Usnpcoldesign.cols();++d)
				{
					this->Usnpcoldesign(r,d)/=sqrt(S2c(d));
				}
			}
			Usnpcoldesign.noalias() = Usnpcoldesign * U1c;
            Usnpcoldesign_cached = true;
        }
	}



} // end namespace

