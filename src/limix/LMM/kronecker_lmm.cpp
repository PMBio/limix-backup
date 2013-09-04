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
			Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver1c(K1c);
            this->U1c = eigensolver1c.eigenvectors();
            this->S1c = eigensolver1c.eigenvalues();
			if (this->S1c(0)<=1e-12){
				throw new CGPMixException("The column covariance of the first covariance term has to be full rank, but is not.");
			}

			Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver1r(K1r);
            this->U1r = eigensolver1r.eigenvectors();
            this->S1r = eigensolver1r.eigenvalues();
			if (this->S1r(0)<=1e-12){
				throw new CGPMixException("The row covariance of the first covariance term has to be full rank, but is not.");
			}
            
			this->S1U1K2r = U1r.transpose() * this->K2r * U1r;
			for (size_t r1=0;r1<this->pheno.rows();++r1)
			{
				for (size_t r2=0;r2<this->pheno.rows();++r2)
				{
					this->S1U1K2r(r1,r2)/=sqrt(S1r(r1));
					this->S1U1K2r(r1,r2)/=sqrt(S1r(r2));
				}
			}

			this->S1U1K2c = U1c.transpose() * this->K2c * U1c;
			for (size_t c1=0;c1<this->pheno.cols();++c1)
			{
				for (size_t c2=0;c2<this->pheno.cols();++c2)
				{
					this->S1U1K2r(c1,c2)/=sqrt(S1r(c1));
					this->S1U1K2r(c1,c2)/=sqrt(S1r(c2));
				}
			}

			Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver2c(S1U1K2c);
            this->U2c = eigensolver2c.eigenvectors();
            this->S2c = eigensolver2c.eigenvalues();

			Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver1r(S1U1K2r);
            this->U2r = eigensolver1r.eigenvectors();
            this->S2r = eigensolver1r.eigenvalues();
            
			this->UK_cached = true;
        }
        
		if(!Usnps_cached){
            Usnps.noalias() = U1r.transpose() * snps;
			for (size_t r=0;r<this->pheno.rows();++r)
			{
				for (size_t s=0;s<this->snps.cols();++s)
				{
					this->Usnps(r,s)/=sqrt(S1r(s));
				}
			}
			Usnps.noalias()= U2r.transpose() * Usnps;
            Usnps_cached = true;
        }
        if(!Upheno_cached){
            Upheno.noalias() = U1r.transpose() * pheno * U1c;
			for (size_t r=0;r<this->pheno.rows();++r)
			{
				for (size_t c=0;c<this->pheno.cols();++c)
				{
					this->Upheno(r,c)/=sqrt(S1r(r));
					this->Upheno(r,c)/=sqrt(S1c(c));
				}
			}
			Usnps.noalias()= U2r.transpose() * Usnps;
			Upheno_cached = true;
        }
        if(!Ucovs_cached){
            Ucovs.noalias() = U1r.transpose() * covs;
			for (size_t r=0;r<this->pheno.rows();++r)
			{
				for (size_t d=0;d<this->covs.cols();++d)
				{
					this->Ucovs(r,d)/=sqrt(S1r(r));
				}
			}

            Ucovs_cached = true;
        }
		//need column design matrix
		if(!Ucoldesign_cached){
            Ucoldesign.noalias() = coldesign * U1c;
            for (size_t r=0;r<this->pheno.rows();++r)
			{
				for (size_t d=0;d<this->covs.cols();++d)
				{
					this->S1U1K2r(r,d)/=sqrt(S1r(r));
				}
			}
            Ucoldesign_cached = true;
        }
	}



} // end namespace

