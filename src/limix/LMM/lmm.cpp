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

#include "lmm.h"
#include "limix/utils/gamma.h"
#include "limix/utils/fisherf.h"
#include "limix/utils/matrix_helper.h"
#include <math.h>
//#include <omp.h>
#include<Eigen/Core>


namespace limix {


/*ALMM*/
ALMM::ALMM()
{
	//Default settings:
	num_intervals0 = 100;
	num_intervalsAlt = 0;
	ldeltaInit = 0.0;
	ldeltamin0 = -5.0;
	ldeltamax0 = 5.0;
	ldeltaminAlt = -5.0;
	ldeltamaxAlt = 5.0;
	UK_cached = false;
	Usnps_cached = false;
	Upheno_cached = false;
	Ucovs_cached = false;
	//standard: using likelihood ratios:
	testStatistics = ALMM::TEST_LRT;
}

ALMM::~ALMM()
{
}

mfloat_t ALMM::getLdeltamin0() const
{
	return ldeltamin0;
}

muint_t ALMM::getNumIntervalsAlt() const
{
	return num_intervalsAlt;
}

muint_t ALMM::getNumSamples() const
{
	return num_samples;
}

void ALMM::setLdeltamin0(mfloat_t ldeltamin0)
{
	this->ldeltamin0 = ldeltamin0;
}

void ALMM::setLdeltaInit(mfloat_t logdelta)
{
	this->ldeltaInit = logdelta;
}

mfloat_t ALMM::getLdeltaInit() const
{
	return this->ldeltaInit;
}

void ALMM::setNumIntervalsAlt(muint_t num_intervalsAlt)
{
	this->num_intervalsAlt = num_intervalsAlt;
}

MatrixXd ALMM::getPheno() const
{
	return pheno;
}

MatrixXd ALMM::getPv() const
{
	return pv;
}

void ALMM::agetPheno(MatrixXd *out) const
{
	(*out) = pheno;
}

void ALMM::agetPv(MatrixXd *out) const
{
	(*out) = pv;
}

void ALMM::agetSnps(MatrixXd *out) const
{
	(*out) = snps;
}

void ALMM::agetCovs(MatrixXd *out) const
{
	(*out) = covs;
}

void ALMM::agetK(MatrixXd *out) const
{
	(*out) = K;
}

MatrixXd ALMM::getK() const
{
	return K;
}

void ALMM::setK(const MatrixXd & K)
{
	this->K = K;
	this->clearCache();
}


void ALMM::clearCache()
{
	this->UK_cached = false;
	this->Usnps_cached = false;
	this->Ucovs_cached = false;
	this->Upheno_cached = false;
}

int ALMM::getTestStatistics() const
{
	return testStatistics;
}

    muint_t ALMM::getNumIntervals0() const
    {
        return num_intervals0;
    }

    void ALMM::setNumIntervals0(muint_t num_intervals0)
    {
        this->num_intervals0 = num_intervals0;
    }

    mfloat_t ALMM::getLdeltamax0() const
    {
        return ldeltamax0;
    }

    void ALMM::setLdeltamax0(mfloat_t ldeltamax0)
    {
        this->ldeltamax0 = ldeltamax0;
    }

    mfloat_t ALMM::getLdeltamaxAlt() const
    {
        return ldeltamaxAlt;
    }

    void ALMM::setLdeltamaxAlt(mfloat_t ldeltamaxAlt)
    {
        this->ldeltamaxAlt = ldeltamaxAlt;
    }

    void ALMM::setTestStatistics(int testStatistics)
    {
        this->testStatistics = testStatistics;
    }

    MatrixXd ALMM::getCovs() const
    {
        return covs;
    }

    void ALMM::setCovs(const MatrixXd & covs)
    {
        this->covs = covs;
        Ucovs_cached = false;
    }

    void ALMM::setPheno(const MatrixXd & pheno)
    {
        this->pheno = pheno;
        Upheno_cached = false;
    }

    void ALMM::setSNPs(const MatrixXd & snps)
    {
        this->snps = snps;
        Usnps_cached = false;
    }

    mfloat_t ALMM::getLdeltaminAlt() const
    {
        return ldeltaminAlt;
    }

    void ALMM::setLdeltaminAlt(mfloat_t ldeltaminAlt)
    {
        this->ldeltaminAlt = ldeltaminAlt;
    }

    void ALMM::setPermutation(const VectorXi & perm)
    {
        this->perm = perm;
    }

    MatrixXd ALMM::getSnps() const
    {
        return snps;
    }

    VectorXi ALMM::getPermutation() const
    {
        VectorXi rv;
        agetPermutation(&rv);
        return rv;
    }

    void ALMM::agetPermutation(VectorXi *out) const
    {
        (*out) = perm;
    }

    /*
    void ALMM::applyPermutation(MatrixXd & M) 
    {
        if(isnull(perm))
            return;

        if(perm.rows() != M.rows()){
            throw CLimixException("ALMM:Permutation vector has incompatible length");
        }
        //create temporary copy
        MatrixXd Mc = M;
        //apply permutation;
        for(muint_t i = 0;i < (muint_t)((((Mc.rows()))));++i){
            M.row(i) = Mc.row(perm(i));
        }
    }
    */

    /*CLMMCore*/


    /*CLMM*/
    CLMM::CLMM()
    :ALMM()
    {
    	storeVerboseResults= true;
    	calc_stes = true;
    }

    CLMM::~CLMM()
    {
        // TODO Auto-generated destructor stub
    }

    void CLMM::setK(const MatrixXd & K, const MatrixXd & U, const VectorXd & S)
    {
        this->K = K;
        this->U = U;
        this->S = S;
        clearCache();
    }

    /*CLMM*/
    void CLMM::updateDecomposition() 
    {
        //check that dimensions match
        this->num_samples = snps.rows();
        this->num_snps = snps.cols();
        this->num_pheno = pheno.cols();
        this->num_covs = covs.cols();
        if (num_samples==0)
            throw CLimixException("LMM requires a non-zero sample size");

        if (num_snps==0)
            throw CLimixException("LMM requires non-zero number of SNPs");

        if (num_pheno==0)
            throw CLimixException("LMM requires non-zero number of phenotypes");

        if(!(num_samples == (muint_t)pheno.rows()))
            throw CLimixException("phenotypes and SNP number samples (rows) inconsistent");

        if(!(num_samples == (muint_t)covs.rows()))
            throw CLimixException("covariates and SNP number samples (rows) inconsistent");

        if(isnull(K))
        {
            //no covariance? assume we perform linear regression
            K = VectorXd::Ones(this->num_samples).asDiagonal();
        }

		if (!(num_samples == (muint_t)K.rows()))
			throw CLimixException("number rows of kinship inconsistent");

		if (!(num_samples == (muint_t)K.cols()))
			throw CLimixException("number columns of kinship inconsistent");


        if(!(this->UK_cached)){
            //decomposition of K
            Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver(K);
            U = eigensolver.eigenvectors();
            S = eigensolver.eigenvalues();
            this->UK_cached = true;
        }
        if(!Usnps_cached){
            Usnps.noalias() = U.transpose() * snps;
            Usnps_cached = true;
        }
        if(!Upheno_cached){
            Upheno.noalias() = U.transpose() * pheno;
            Upheno_cached = true;
        }
        if(!Ucovs_cached){
            Ucovs.noalias() = U.transpose() * covs;
            Ucovs_cached = true;
        }
    }

    void CLMM::process()// 
    {
        //get decomposition
        updateDecomposition();
        /*
        std::cout << "U" << this->U << "\n";
        std::cout << "S" << this->S << "\n";
		*/

        //resize phenotype output
        this->pv.resize(this->num_pheno, this->num_snps);
        //result matrices: think about what to store in the end
        ldelta0.resize(num_pheno,1);
		ldelta0.setConstant(this->ldeltaInit);
        ldeltaAlt.resize(num_pheno,num_snps);
		ldeltaAlt.setConstant(this->ldeltaInit);
		nLL0.resize(num_pheno, 1);
        nLLAlt.resize(num_pheno,num_snps);
        lsigma.resize(num_pheno,num_snps);
        beta_snp.resize(num_pheno,num_snps);
        if(calc_stes)
        	beta_snp_ste.resize(num_pheno,num_snps);


        //reserve memory for snp-wise foreground model
        MatrixXd UXps(num_samples, num_covs + 1);
        //add in covariats from column 1 on:
        UXps.block(0, 1, num_samples, num_covs) = Ucovs;

        MatrixXd UXi = MatrixXd::Zero(num_samples, 1);
        MatrixXd f_tests_,AObeta_,AObeta_ste_,AOsigma_;

        bool calc_ftests = (this->testStatistics == ALMM::TEST_F);
        if(calc_ftests)
           	f_tests.resize(num_pheno,num_snps);
        //optimization on Null model?
        if (num_intervals0>0)
        {
        	//optimize on null model upfront:
        	optdeltaEx(ldelta0,nLL0,Upheno,Ucovs,S,num_intervals0,ldeltamin0,ldeltamax0);
        }
        //loop over SNPs
        for(muint_t is = 0;is < num_snps;++is)
        {
        	UXi = Usnps.block(0, is, num_samples, 1);
        	applyPermutation(UXi);
        	//set SNP
        	UXps.block(0,0,num_samples,1) = UXi;

        	//2. fit delta or use null model delta
            if(num_intervalsAlt > 0)
            	optdeltaEx(ldeltaAlt.col(is),nLLAlt.col(is),Upheno,UXps,S,num_intervalsAlt,ldeltaminAlt,ldeltamaxAlt);
            else
            	ldeltaAlt.col(is) = ldelta0;

            //3. loop over phenotypes and evaluate likelihood etc.
            for(muint_t ip=0;ip<num_pheno;++ip)
            {
            	//3.1 evaluate test statistics (foreground)
            	nLLevalEx(AObeta_,AObeta_ste_,AOsigma_,f_tests_,nLLAlt.block(ip, is, 1, 1), Upheno.block(0,ip,num_samples,1), UXps, S, ldeltaAlt(ip, is), calc_ftests,calc_stes);
            	lsigma(ip,is) = AOsigma_(0,0);
            	beta_snp(ip,is) = AObeta_(0,0);
            	if(calc_stes)
            		beta_snp_ste(ip,is) = AObeta_ste_(0,0);

            	//3.2 evaluate test statistics (background)
                if(this->testStatistics == ALMM::TEST_LRT)
                {
                	mfloat_t DL = nLL0(ip, 0) - nLLAlt(ip, is);
                	if (DL<0)
                		DL = 0;
                    this->pv(ip, is) = stats::Gamma::gammaQ(DL, (double)0.5 * 1.0);
                }
                else if(this->testStatistics == ALMM::TEST_F)
                    {
                    	//store ftest statitics for testing SNP
                    	f_tests(ip,is) = f_tests_(0,0);
                        this->pv(ip, is) = 1.0 - stats::FisherF::Cdf(f_tests_(0,0), 1.0, (double)(num_samples - f_tests_.rows()));
                    }
            } //end :: for ip

		} //end for SNP

    }

    //CInteractLmm
    CInteractLMM::CInteractLMM()
    {
        refitDelta0Pheno = false;
    }

    ;
    CInteractLMM::~CInteractLMM()
    {
    }

    ;
    void CInteractLMM::setInter(const MatrixXd & Inter)
    {
        this->I = Inter;
        this->num_inter = Inter.cols();
    }

    void CInteractLMM::agetInter(MatrixXd *out) const
    {
        (*out) = I;
    }

    MatrixXd CInteractLMM::getInter() const
    {
        MatrixXd rv;
        agetInter(&rv);
        return rv;
    }

    void CInteractLMM::setInter0(const MatrixXd & Inter)
    {
        this->I0 = Inter;
        this->num_inter0 = Inter.cols();
    }

    void CInteractLMM::agetInter0(MatrixXd *out) const
    {
        (*out) = I0;
    }

    bool CInteractLMM::isRefitDelta0Pheno() const
    {
        return refitDelta0Pheno;
    }

    void CInteractLMM::setRefitDelta0Pheno(bool refitDelta0Pheno)
    {
        this->refitDelta0Pheno = refitDelta0Pheno;
    }

    MatrixXd CInteractLMM::getInter0() const
    {
        MatrixXd rv;
        agetInter0(&rv);
        return rv;
    }

    //processing;
    void CInteractLMM::process() //
    {
    	//TODO: Ftest is not correct for simple cases where I0=0 as ftest_rows yields the wrong answer.
    	if((num_inter > 1) && (this->testStatistics == ALMM::TEST_F)){
            throw CLimixException("CInteractLMM:: cannot use Ftest for more than 1 interaction dimension!");
        }
        //get decomposition
        updateDecomposition();
        //resize phenotype output
        pv.resize(this->num_pheno, this->num_snps);
        ldelta0.resize(num_pheno,num_snps);
        ldelta0.setConstant(0.0);
        ldeltaAlt.resize(num_pheno,num_snps);
        nLL0.resize(num_pheno,num_snps);
        nLLAlt.resize(num_pheno,num_snps);
        lsigma.resize(num_pheno,num_snps);
        beta_snp.resize(num_pheno,num_snps);
        beta_snp_ste.resize(num_pheno,num_snps);


        //result matrices: think about what to store in the end
        //reserve memory for snp-wise foreground model
        //consist of [X*I,X*I0,cov]
        MatrixXd UXps(num_samples, num_inter + num_inter0 + num_covs);
        //store covariates upfront
        UXps.block(0, num_inter + num_inter0, num_samples, num_covs) = Ucovs;
        //create full interaction matirx including I0 and I
        MatrixXd II = MatrixXd::Zero(num_samples, num_inter + num_inter0);
        II.block(0, 0, num_samples, num_inter) = I;
        II.block(0, num_inter, num_samples, num_inter0) = I0;

        MatrixXd f_tests_,AObeta_,AObeta_ste_,AOsigma_;
        //reserver memory for ftests?
        bool calc_ftests = (this->testStatistics == ALMM::TEST_F);
        if(calc_ftests)
        {
        	f_tests.resize(num_pheno,num_snps);
        }

        //fit delta0 on full null model without any SNP in there (EmmaX)
        if (!refitDelta0Pheno && (testStatistics==ALMM::TEST_LRT))
        {
        	MatrixXd ldelta00;
        	MatrixXd nLL00;
        	optdeltaEx(ldelta00,nLL00,Upheno, Ucovs, S, num_intervals0, ldeltamin0, ldeltamax0);
        	ldelta0.colwise() = ldelta00.col(0);
        	//nLL0.colwise() = nLL00.col(0);
        }

        //loop over SNPs
        MatrixXd Xi = MatrixXd::Zero(num_samples, num_inter + num_inter0 + num_covs);
        MatrixXd UXi = MatrixXd::Zero(num_samples, num_inter + num_inter0 + num_covs);
        for(muint_t is = 0;is < num_snps;is++)
        {
        	//1. construct testing X:
        	//interaction model:
        	Xi = II;
        	Xi.array().colwise()*=snps.array().col(is);
        	UXi.noalias() = U.transpose()*Xi;
        	//permute if needed:
        	applyPermutation(UXi.block(0, 0, num_samples, num_inter));

        	//construct full foreground SNP set
        	UXps.block(0,0,num_samples,num_inter+num_inter0) = UXi;

    		if (refitDelta0Pheno && (testStatistics==ALMM::TEST_LRT))
    		{
    			//refit delta on new null model which has changed due to I0:
    			optdeltaEx(ldelta0.col(is),nLL0.col(is),Upheno, UXps.block(0,num_inter,num_samples,num_inter0+num_covs), S, num_intervals0, ldeltamin0, ldeltamax0);
    		}

    		if (num_intervalsAlt>0)
    			//fit delta on alt model also
    			optdeltaEx(ldeltaAlt.col(is),nLLAlt.col(is),Upheno, UXps, S, num_intervals0, ldeltamin0, ldeltamax0);
    		else
    			ldeltaAlt.col(is) = ldelta0.col(is);

        	//loop over phenotypes
        	for(muint_t ip = 0;ip < num_pheno;ip++)
        	{
        		//3. evaluate foreground likelihood
        		nLLevalEx(AObeta_,AObeta_ste_,AOsigma_,f_tests_,nLLAlt.block(ip,is,1,1),Upheno.col(ip), UXps, S,ldeltaAlt(ip, is),(calc_ftests));

        		//update lsigma and beta_snp:
        		lsigma(ip,is) = AOsigma_(0,0);
            	beta_snp(ip,is) = AObeta_(0,0);


        		//4. calc p-value
        		if (this->testStatistics==ALMM::TEST_LRT)
        		{
        			//for likelihood ratios, we require evaluation on the new null model due to I0:
        			nLLevalEx(AObeta_,AObeta_ste_,AOsigma_,f_tests_,nLL0.block(ip,is,1,1),Upheno.col(ip), UXps.block(0,num_inter,num_samples,num_inter0+num_covs), S,ldelta0(ip, is),false);
        			//adjust degrees of freedom and calc pv:
        			this->pv(ip, is) = stats::Gamma::gammaQ(nLL0(ip, is) - nLLAlt(ip, is), (double)0.5*(num_inter));
        		}
        		else if (this->testStatistics==ALMM::TEST_F)
        		{
        			f_tests(ip,is) = f_tests_(0,0);
        			this->pv(ip,is) = 1.0 - stats::FisherF::Cdf(f_tests_(0,0),1.0 , (double)(num_samples - f_tests_.rows()));
        		}
        	} //end for SNP
	}
    }
    void CInteractLMM::updateDecomposition() 
    {
        CLMM::updateDecomposition();
    }

	mfloat_t nLLevalFunctor::operator()(const mfloat_t logdelta)
	{
		return nLLeval(&f_tests, (double)logdelta, Y, X, S, REML);
	}

	nLLevalFunctor::nLLevalFunctor(
		const MatrixXd Y,
		const MatrixXd X,
		const MatrixXd S,
		const bool REML
		)
	{
		this->f_tests=MatrixXd();
		this->X=X;
		this->REML=REML;
		this->Y=Y;
		this->S=S;
	}

	nLLevalFunctor::~nLLevalFunctor(){}

    double optdelta(const MatrixXd & UY, const MatrixXd & UX, const MatrixXd & S, int numintervals, double ldeltamin, double ldeltamax, bool REML)
    {
        //grid variable with the current likelihood evaluations
        MatrixXd nllgrid     = MatrixXd::Ones(numintervals,1).array()*HUGE_VAL;
        MatrixXd ldeltagrid = MatrixXd::Zero(numintervals, 1);
        //current delta
        double ldelta = ldeltamin;
        double ldeltaD = (ldeltamax - ldeltamin);
        ldeltaD /= (double)(numintervals - 1);
        double nllmin = HUGE_VAL;
        double ldeltaopt_glob = 0;
        MatrixXd f_tests;
		muint_t nevals = 0;
        for(int i = 0;i < (numintervals);i++){
            nllgrid(i, 0) = nLLeval(&f_tests, ldelta, UY, UX, S, REML);
            ++nevals;
			ldeltagrid(i, 0) = ldelta;
            if(nllgrid(i, 0) < nllmin){
                nllmin = nllgrid(i, 0);
                ldeltaopt_glob = ldelta;
            }
            //move on delta
            ldelta += ldeltaD;
        } //end for all intervals
		nLLevalFunctor func(UY, UX, S, REML);
		for(muint_t i=1;i<(muint_t)(numintervals-1);i++){
		  if (nllgrid(i,0)<nllgrid(i+1,0) && nllgrid(i,0)<nllgrid(i-1,0)){
			  //check wether a local optimum exists in this triplet
			 mfloat_t current_nLL=std::numeric_limits<mfloat_t>::infinity();
			 size_t nevals_Int=0;
			 //If there is a local optimum, optimize over the current triplet:
			 mfloat_t brentTol=0.00000001;
			 muint_t brentMaxIter = 10000;
			 mfloat_t current_log_delta=BrentC::minimize(func,ldeltagrid(i-1,0),ldeltagrid(i+1,0),brentTol,current_nLL,nevals_Int,brentMaxIter,true);
			 nevals+=nevals_Int;
			 if (current_nLL<=nllmin){
				nllmin=current_nLL;
				ldeltaopt_glob=(current_log_delta);
			 }
		  }
	   }
		//std::cout << "\n\n nLL_i:\n" << nllgrid;
		//std::cout <<"\n\n delta_i:\n" << ldeltagrid;
        return ldeltaopt_glob;
    }

    /* internal functions */
    double nLLeval(MatrixXd *F_tests, double ldelta, const MatrixXd & UY, const MatrixXd & UX, const MatrixXd & S, bool REML)
    {
		size_t n = UX.rows();
        size_t d = UX.cols();
        size_t n_pheno = UY.cols();
        assert(UY.cols() == S.cols());
        assert(UY.rows() == S.rows());
        assert(UY.rows() == UX.rows());
        double delta = exp(ldelta);
        MatrixXd Sdi = S.array() + delta;
        double ldet = 0.0;
        for(size_t ind = 0;ind < n_pheno * n;++ind){
            ldet += log(Sdi.data()[ind]);
        }
        Sdi = Sdi.array().inverse();
        (*F_tests).resize(d, n_pheno);
        MatrixXd beta = MatrixXd(d, n_pheno);
        MatrixXd XSdi;
        //replice Sdi
		double ldetXSX = 0.0;
        for(muint_t phen = 0;phen < n_pheno;++phen){
            VectorXd Sdi_p = Sdi.block(0, phen, n, 1);
            XSdi = (UX.array() * Sdi_p.replicate(1, d).array()).transpose();
            MatrixXd XSX = XSdi * UX;
            MatrixXd XSY = XSdi * UY.block(0, phen, n, 1);
            //least sqaures solution of XSX*beta = XSY
            //decomposition of K
            Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver(XSX);
            MatrixXd U_X = eigensolver.eigenvectors();
            MatrixXd S_X = eigensolver.eigenvalues();
            beta.block(0, phen, d, 1) = U_X.transpose() * XSY;
            //MatrixXd S_i = MatrixXd::Zero(d,d);
            for(size_t dim = 0;dim < d;++dim){
                if(S_X(dim, 0) > 3E-8){
                    beta(dim, phen) /= S_X(dim, 0);
					ldetXSX += log(S_X(dim, 0));
                    for(size_t dim2 = 0;dim2 < d;++dim2){
                        (*F_tests)(dim2, phen) += U_X(dim2, dim) * U_X(dim2, dim) / S_X(dim, 0);
                    }
                    //S_i(dim,dim) = 1.0/S_X(dim,0);
                }
                else{
                    beta(dim, phen) = 0.0;
                }
            }

            beta.block(0, phen, d, 1) = U_X * beta.block(0, phen, d, 1);
        }

        MatrixXd res = UY - UX * beta;
        //squared residuals
        res.array() *= res.array();
        res.array() *= Sdi.array();
		double sigg2 = res.array().sum();
		double nLL;
		if (REML){
			MatrixXd XX = UX.transpose() * UX;
            Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver(XX);
            //MatrixXd U_XX = eigensolver.eigenvectors();
            MatrixXd S_XX = eigensolver.eigenvalues();
			double ldetXX  = 0.0;
			for(size_t dim = 0;dim < d;++dim){
                if(S_XX(dim, 0) > 3E-8){
                    ldetXX += log(S_XX(dim, 0));
                }
            }
			ldetXX*=n_pheno;
			sigg2/= ((n-d) * n_pheno);
			nLL = 0.5 * ((n-d) * n_pheno * L2pi + ldet + ldetXSX - ldetXX + (n-d) * n_pheno + (n-d) * n_pheno * log(sigg2));
		}else{
			sigg2/= (n * n_pheno);
			nLL = 0.5 * (n * n_pheno * L2pi + ldet + n * n_pheno + n * n_pheno * log(sigg2));
		}

		//compute the F-statistics
        (*F_tests).array() = beta.array() * beta.array() / (*F_tests).array();
        (*F_tests).array() /= sigg2;
        return nLL;
    }

    void optdeltaAllY(MatrixXd *out, const MatrixXd & UY, const MatrixXd & UX, const MatrixXd & S, const MatrixXd & ldeltagrid)
    {
        size_t n_p = UY.cols();
        size_t numintervals = ldeltagrid.rows();
        //grid variable with the current likelihood evaluations
        (*out) = MatrixXd::Ones(numintervals,n_p).array()*HUGE_VAL;
        //current delta
        for(size_t i = 0;i < numintervals;i++){
            MatrixXd row;
            nLLevalAllY(&row, ldeltagrid(i, 0), UY, UX, S);
            (*out).row(i) = row;
        } //end for all intervals
    }

    void train_associations_SingleSNP(MatrixXd *PV, MatrixXd *LL, MatrixXd *ldelta, const MatrixXd & X, const MatrixXd & Y, const MatrixXd & U, const MatrixXd & S, const MatrixXd & C, int numintervals, double ldeltamin, double ldeltamax)
    {
    	//get dimensions:
    	//samples
    	size_t nn = X.rows();
    	//snps
    	size_t ns = X.cols();
    	assert( ns == 1 );
    	//phenotypes
    	size_t np = Y.cols();
    	//covaraites
    	size_t nc = C.cols();
    	//make sure the size of N/Y is correct
    	assert((int)nn==(int)Y.rows());
    	assert((int)nn==(int)C.rows());
    	assert((int)nn==(int)U.rows());
    	assert((int)nn==(int)U.cols());


    	//resize output variable if needed
    	(*PV).resize(np,ns);
    	(*LL).resize(np,ns);
    	(*ldelta).resize(np,ns);

    	//transform everything
    	MatrixXd UX = U.transpose()*X;
    	MatrixXd UY = U.transpose()*Y;
    	MatrixXd Ucovariates = U.transpose()*C;


    	//reserve memory for snp-wise foreground model
    	MatrixXd UX_(nn,nc+1);
    	//store covariates upfront
    	UX_.block(0,0,nn,nc) = Ucovariates;
    	UX_.block(0,nc,nn,1) = UX;
    	MatrixXd ldeltagrid(numintervals,1);
    	for (size_t interval = 0; interval < (size_t)numintervals; ++interval)
    	{
    		ldeltagrid(interval,0) = ldeltamin + interval*((ldeltamax - ldeltamin)/(1.0*(numintervals-1)));
    	}
    	MatrixXd nllgrid;
    	optdeltaAllY(&nllgrid,UY, UX_, S, ldeltagrid);
    	//1. fit background covariances on phenotype and covariates alone
    	for (size_t ip=0;ip<np;ip++)
    	{
    		(*ldelta)(ip) = ldeltamin;
    		size_t i_min = 0;
    		for(size_t interval = 1; interval < (size_t)numintervals; ++interval)
    		{
    			if(nllgrid(interval,ip)<nllgrid(i_min,ip))
    			{
    				//printf("oldmin : %.4f, newmin : %.4f, newdelta : %.4f, interval : %i\n" ,nllgrid(i_min,ip) , nllgrid(interval,ip), ldeltagrid(interval,0),interval);
    				(*ldelta)(ip,0) = ldeltagrid(interval,0);
    				i_min = interval;
    			}
    		}
    		//get UY columns
    		MatrixXd UY_ = UY.block(0,ip,nn,1);
    		//fit delta on null model
    		MatrixXd f_tests(nc+1,1);
    		(*LL)(ip,0)   = -1.0*nLLeval(&f_tests, (*ldelta)(ip), UY_, UX_, S);
    		(*PV)(ip,0) = 1.0 - stats::FisherF::Cdf(f_tests(1,0), 1.0 , (double)(nn - f_tests.rows()));
    		//printf("ip : %i, ldelta : %.4f, PV: %.4f LL: %.4f\n", (int)ip, ldelta(ip),(PV(ip)),LL(ip));
    		//  for(size_t dim = 0; dim<(size_t)f_tests.rows();++dim)
    		//	{
    		//  printf("f[%i] : %.4f, ",(int)dim,f_tests(dim));
    		//}
    		//if( FisherF::Cdf(f_tests(f_tests.rows()-1,0), 1.0 , (double)(nn - f_tests.rows())) < 0.0 )
    		//{
    		//printf("\noutch!");//3.1756

    		// printf("  LLmin: %.4f nn:%i dim:%i F(3.1756):%.4f \n", nllgrid(i_min,ip), (int)nn, (int) f_tests.rows(),FisherF::Cdf(3.0, 1.0 , 18.0));
    		//LL(ip)   = -1.0 * nLLeval(f_tests, ldelta(ip), UY_, UX_, S);
    		//}


    		//printf("\n\n");
	}

}

/* Internal C++ functions */
void nLLevalAllY(MatrixXd *out, double ldelta, const MatrixXd & UY, const MatrixXd & UX, const VectorXd & S)
{
	size_t n = UX.rows();
	size_t p = UY.cols();

	MatrixXd XSdi,XSX,XSY,U_X,S_X;

	double delta = exp(ldelta);
	VectorXd Sdi = S.array() + delta;

	double ldet = 0.0;
	//calc log det and invert at the same time Sdi elementwise
	for(size_t ind = 0;ind < n;++ind)
	{
		//ldet
		ldet += log(Sdi.data()[ind]);
		//inverse:
		Sdi.data()[ind] = 1.0 / (Sdi.data()[ind]);
	}
	/*
	double ldet = Sdi.array().log().sum();
	Sdi = Sdi.array().inverse();
	*/

	XSdi = UX.array().transpose();
	XSdi.array().rowwise() *= Sdi.array().transpose();
	XSX.noalias() = XSdi * UX;
	XSY.noalias() = XSdi * UY;

	//least squares solution of XSX*beta = XSY
	MatrixXd beta = XSX.colPivHouseholderQr().solve(XSY);
	MatrixXd res = UY - UX * beta;
	//squared residuals
	res.array() *= res.array();
	res.array() *= Sdi.replicate(1, p).array();

	(*out) = MatrixXd(1, p);
	for(size_t phen = 0;phen < p;++phen){
		double sigg2 = res.col(phen).array().sum() / n;
		(*out)(0, phen) = 0.5 * (n * L2pi + ldet + n + n * log(sigg2));
	}
}


/*
mfloat_t CLMMCore::optdelta(const MatrixXd & UY, const MatrixXd & UX, const MatrixXd & S, int numintervals, double ldeltamin, double ldeltamax)
{
    //grid variable with the current likelihood evaluations
    MatrixXd nllgrid     = MatrixXd::Ones(numintervals,1).array()*HUGE_VAL;
    MatrixXd ldeltagrid = MatrixXd::Zero(numintervals, 1);
    //current delta
    mfloat_t ldelta = ldeltamin;
    mfloat_t ldeltaD = (ldeltamax - ldeltamin);
    ldeltaD /= ((mfloat_t)(((((((((numintervals))))))))) - 1);
    mfloat_t nllmin = HUGE_VAL;
    mfloat_t ldeltaopt_glob = 0;
    for(int i = 0;i < (numintervals);i++){
	nllgrid(i, 0) = this->nLLeval(NULL, ldelta, UY, UX, S);
	ldeltagrid(i, 0) = ldelta;
	if(nllgrid(i, 0) < nllmin){
		nllmin = nllgrid(i, 0);
		ldeltaopt_glob = ldelta;
	}
	//move on delta
	ldelta += ldeltaD;
    }
    return ldeltaopt_glob;
}

mfloat_t CLMMCore::nLLeval(VectorXd *F_tests, mfloat_t ldelta, const MatrixXd & UY, const MatrixXd & UX, const VectorXd & S)
{
    size_t n = UX.rows();
    size_t d = UX.cols();
    assert(UY.rows() == S.rows());
    assert(UY.rows() == UX.rows());
    mfloat_t delta = exp(ldelta);
    Sdi = S.array() + delta;
    mfloat_t ldet = 0.0;
    //calc log det and invert at the same time Sdi elementwise
    for(size_t ind = 0;ind < n;++ind){
        //ldet
        ldet += log(Sdi.data()[ind]);
        //inverse:
        Sdi.data()[ind] = 1.0 / (Sdi.data()[ind]);
    }
    //arrayInverseInplace(Sdi); (done in loop above)
    if (F_tests!=NULL)
{
	(*F_tests).resize(d);
	F_tests->setConstant(0.0);
}
    beta.resize(d);
    //replice Sdi
    //EIGEN_ASM_COMMENT("begin");
    XSdi = UX.array().transpose();
    XSdi.array().rowwise() *= Sdi.array().transpose();
    XSX.noalias() = XSdi * UX;
    XSY.noalias() = XSdi * UY;
    //EIGEN_ASM_COMMENT("end");
    //least sqaures solution of XSX*beta = XSY
    //Call internal solver which uses fixed solvers for 2d and 3d matrices
    SelfAdjointEigenSolver(U_X, S_X, XSX);
    beta.noalias() = U_X.transpose() * XSY;
    //loop over genotype dimensions:
    for(size_t dim = 0;dim < d;++dim)
{
	if(S_X(dim, 0) > 3E-8)
	{
		beta(dim) /= S_X(dim, 0);
		if (F_tests!=NULL)
		{
			for(size_t dim2 = 0;dim2 < d;++dim2)
				(*F_tests)(dim2) += U_X(dim2, dim) * U_X(dim2, dim) / S_X(dim, 0);
		}
	}
	else
		beta(dim) = 0.0;
}
    beta = U_X * beta;
    res.noalias() = UY - UX * beta;
    //sqared residuals
    res.array() *= res.array();
    res.array() *= Sdi.array();
    mfloat_t sigg2 = res.sum() / (n);
    //compute the F-statistics
    if (F_tests!=NULL)
{
	(*F_tests).array() = beta.array() * beta.array() / (*F_tests).array();
	(*F_tests).array() /= sigg2;
}
    mfloat_t nLL = 0.5 * (n * L2pi + ldet + n + n * log(sigg2));
    return nLL;
}
*/




/* namespace limix */





}


