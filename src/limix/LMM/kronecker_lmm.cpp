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
#include "kronecker_lmm.h"
#include "limix/utils/gamma.h"
#include "limix/utils/matrix_helper.h"
//#include <omp.h>
//#define debugkron = 0

namespace limix {

CKroneckerLMM::CKroneckerLMM() {
}

CKroneckerLMM::~CKroneckerLMM() {
}

void CKroneckerLMM::process() {
	//1. check dimensions
	muint_t num_snps = this->snps.cols();
	muint_t P = this->snpcoldesign.cols();
	muint_t N = this->snps.rows();
	muint_t num_terms=rowdesign0.size();
	//estimate effective degrees of freedom: difference in number of weights
	muint_t dof = snpcoldesign.rows();//actually this should be the rank of this design matrix
	muint_t dof0_inter = snpcoldesign0_inter.rows();//actually this should be the rank of this design matrix
	//if (num_terms!=coldesign0.size() || num_terms!=coldesignU0.size() || num_terms!=rowdesign0.size() || num_terms!=Urowdesign0.size())
	if (num_terms!=coldesign0.size() || num_terms!=rowdesign0.size() )
	{
		throw CLimixException("number terms in background model inconsistent");
	}
	for(muint_t c=0;c<rowdesign0.size();++c)
	{
		//if (P!=coldesign0[c].cols() || P!=coldesignU0[P].cols() || N!=rowdesign0[c].rows() || N!=Urowdesign0[c].rows())
		if (P!=(muint_t)coldesign0[c].cols() || N!=(muint_t)rowdesign0[c].rows() )
		{
			throw CLimixException("dimensions in background model inconsistent");
		}
	}

	//2. init result arrays
	this->nLL0.resize(1,num_snps);
	this->nLLAlt.resize(1,num_snps);
	this->ldelta0.resize(1,num_snps);
	this->ldeltaAlt.resize(1,num_snps);
    this->beta_snp.resize(snpcoldesign.rows(),num_snps);
	if (this->snpcoldesign0_inter.rows()!=0) //check if interaction design matrix is set
	{
		this->nLL0_inter.resize(1,num_snps);
		this->ldelta0_inter.resize(1,num_snps);
		this->pv.resize(3,num_snps);
	}
	else{
		this->pv.resize(1,num_snps);
	}
	//3. update the decompositions
	this->updateDecomposition();

	//evaluate null model
	mfloat_t ldelta0_ = this->ldeltaInit;//TODO: optionally fix to given value
	mfloat_t nLL0_ = 0.0;
	if (num_intervals0>0)
	{
		nLL0_ = CKroneckerLMM::optdelta(ldelta0_,this->coldesignU0,this->Urowdesign0,this->Upheno,this->S1c,this->S1r,this->S2c,this->S2r, ldeltamin0, ldeltamax0, num_intervals0);
	}
	else
	{
		nLL0_ = CKroneckerLMM::nLLeval(ldelta0_,this->coldesignU0,this->Urowdesign0,this->Upheno,this->S1c,this->S1r,this->S2c,this->S2r, this->W);
	}
	//store delta0
	ldelta0.setConstant(ldelta0_);
	nLL0.setConstant(nLL0_);
	//2. loop over SNPs
	mfloat_t deltaNLL;
	MatrixXdVec UrowdesignAlt = Urowdesign0;
	UrowdesignAlt.push_back(MatrixXd());
	MatrixXdVec coldesignU0_inter = coldesignU0;
	if (this->snpcoldesign0_inter.rows()!=0) //check if interaction design matrix is set
	{
		coldesignU0_inter.push_back(snpcoldesignU0_inter);
	}
	MatrixXdVec coldesignUAlt = coldesignU0;
	coldesignUAlt.push_back(snpcoldesignU);
	MatrixXd& Usnpcurrent = UrowdesignAlt[UrowdesignAlt.size()-1];

	for (muint_t is=0;is<num_snps;++is)
	{
		//0. update mean term
		Usnpcurrent = Usnps.block(0,is,num_samples,1);

		//2. evaluate null model for coldesign0_inter
		mfloat_t ldelta= ldelta0(0,is);
		mfloat_t nLL;
		if (this->snpcoldesign0_inter.rows()!=0) //check if interaction design matrix is set
		{
			if (num_intervals0_inter>0)
			{
				nLL = CKroneckerLMM::optdelta(ldelta,coldesignU0_inter,UrowdesignAlt,this->Upheno,this->S1c,this->S1r,this->S2c,this->S2r, ldeltaminAlt, ldeltamaxAlt, num_intervals0_inter);
			}
			else
			{
				nLL = CKroneckerLMM::nLLeval(ldelta,coldesignU0_inter,UrowdesignAlt,this->Upheno,this->S1c,this->S1r,this->S2c,this->S2r, this->W);
			}
			nLL0_inter(0,is) = nLL;
			ldelta0_inter(0,is) = ldelta;
			deltaNLL = nLL0(0,is) - nLL0_inter(0,is);
			//std::cout<< "nLL0(0,is)"<< nLL0(0,is)<< "nLLAlt(0,is)" << nLLAlt(0,is)<< "\n";
			if (deltaNLL<0.0)
			{
				std::cout << "outch" << "\n";
				deltaNLL = 0.0;
			}
		}

		//3. evaluate alternative model
		if (num_intervalsAlt>0)
		{
			nLL = CKroneckerLMM::optdelta(ldelta,coldesignUAlt,UrowdesignAlt,this->Upheno,this->S1c,this->S1r,this->S2c,this->S2r, ldeltaminAlt, ldeltamaxAlt, num_intervalsAlt);
		}
		else
		{
			nLL = CKroneckerLMM::nLLeval(ldelta,coldesignUAlt,UrowdesignAlt,this->Upheno,this->S1c,this->S1r,this->S2c,this->S2r, this->W);
		}
		nLLAlt(0,is) = nLL;
        beta_snp.block(0,is,snpcoldesignU.rows(),1) = W;
		ldeltaAlt(0,is) = ldelta;
		deltaNLL = nLL0(0,is) - nLLAlt(0,is);
		//std::cout<< "nLL0(0,is)"<< nLL0(0,is)<< "nLLAlt(0,is)" << nLLAlt(0,is)<< "\n";
		if (deltaNLL<0.0)
		{
			std::cout << "outch" << "\n";
			deltaNLL = 0.0;
		}
		//pvalues
		this->pv(0, is) = stats::Gamma::gammaQ(nLL0(0,is) - nLLAlt(0,is), (double)(0.5) * dof);
		if (this->snpcoldesign0_inter.rows()!=0) //check if interaction design matrix is set
		{
			this->pv(1, is) = stats::Gamma::gammaQ(nLL0_inter(0,is) - nLLAlt(0,is), (double)(0.5) * (dof-dof0_inter));
			this->pv(2, is) = stats::Gamma::gammaQ(nLL0(0,is) - nLL0_inter(0,is), (double)(0.5) * dof0_inter);
		}
	}

}


void CKroneckerLMM::addCovariates(const MatrixXd& covsR, const MatrixXd& covsCol)
{
	this->coldesign0.push_back(covsCol);
	this->rowdesign0.push_back(covsR);
}

void CKroneckerLMM::setCovariates(muint_t index,const MatrixXd& covsR, const MatrixXd& covsCol)
{
	this->coldesign0[index] = covsCol;
	this->rowdesign0[index] = covsR;
}


mfloat_t CKroneckerLMM::optdelta(mfloat_t& ldelta_opt, const MatrixXdVec& A,const MatrixXdVec& X, const MatrixXd& Y, const VectorXd& S_C1, const VectorXd& S_R1, const VectorXd& S_C2, const VectorXd& S_R2, mfloat_t ldeltamin, mfloat_t ldeltamax, muint_t numintervals)
{
    //grid variable with the current likelihood evaluations
    MatrixXd nllgrid    = MatrixXd::Ones(numintervals,1).array()*HUGE_VAL;
    MatrixXd ldeltagrid = MatrixXd::Zero(numintervals, 1);
    MatrixXd Wdummy;
    //current delta
    mfloat_t ldelta = ldeltamin;
    mfloat_t ldeltaD = (ldeltamax - ldeltamin);
    ldeltaD /= ((mfloat_t)(numintervals) - 1.0);
    mfloat_t nllmin = HUGE_VAL;
    mfloat_t ldeltaopt_glob = ldelta_opt;
    muint_t nevals = 0;
    for(muint_t i = 0;i < numintervals;i++){
    	nllgrid(i, 0) = CKroneckerLMM::nLLeval(ldelta, A, X, Y, S_C1, S_R1, S_C2, S_R2, Wdummy);
        ++nevals;
		ldeltagrid(i, 0) = ldelta;
        //std::cout<< "nnl( " << ldelta << ") = " << nllgrid(i,0) <<  "VS" << nllmin << ")\n\n";
        if(nllgrid(i, 0) < nllmin){
            //		std::cout << "new min (" << nllmin << ") -> " <<  nllgrid(i,0) << "\n\n";
            nllmin = nllgrid(i, 0);
            ldeltaopt_glob = ldelta;
        }
        //move on delta
        ldelta += ldeltaD;
    } //end for all intervals
	nLLevalKronFunctor func(A, X, Y, S_C1, S_R1, S_C2, S_R2, Wdummy);
	for(muint_t i=1;i<(numintervals-1);i++){
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
    ldelta_opt = ldeltaopt_glob;
    return nllmin;
}

mfloat_t CKroneckerLMM::nLLeval(mfloat_t ldelta, const MatrixXdVec& A,const MatrixXdVec& X, const MatrixXd& Y, const VectorXd& S_C1, const VectorXd& S_R1, const VectorXd& S_C2, const VectorXd& S_R2, MatrixXd& W)
{
//#define debugll
	muint_t R = (muint_t)Y.rows();
	muint_t C = (muint_t)Y.cols();
	assert(A.size() == X.size());
	assert(R == (muint_t)S_R1.rows());
	assert(C == (muint_t)S_C1.rows());
	assert(R == (muint_t)S_R2.rows());
	assert(C == (muint_t)S_C2.rows());
	muint_t nWeights = 0;
	for(muint_t term = 0; term < A.size();++term)
	{
		assert((muint_t)(X[term].rows())==R);
		assert((muint_t)(A[term].cols())==C);
		nWeights+=(muint_t)(A[term].rows()) * (muint_t)(X[term].cols());
	}
	mfloat_t delta = exp(ldelta);
	mfloat_t ldet = 0.0;//R * C * ldelta;

	//build D and compute the logDet of D
	MatrixXd D = MatrixXd(R,C);
	for (muint_t r=0; r<R;++r)
	{
		if(S_R2(r)>1e-10)
		{
			ldet += (mfloat_t)C * log(S_R2(r));//ldet
		}
		else
		{
			std::cout << "S_R2(" << r << ")="<< S_R2(r)<<"\n";
		}
	}
#ifdef debugll
	std::cout << ldet;
	std::cout << "\n";
#endif
	for (muint_t c=0; c<C;++c)
	{
		if(S_C2(c)>1e-10)
		{
			ldet += (mfloat_t)R * log(S_C2(c));//ldet
		}
		else
		{
			std::cout << "S_C2(" << c << ")="<< S_C2(c)<<"\n";
		}
	}
#ifdef debugll
	std::cout << ldet;
	std::cout << "\n";
#endif
	for (muint_t r=0; r<R;++r)
	{
		for (muint_t c=0; c<C;++c)
		{
			mfloat_t SSd = S_R1.data()[r]*S_C1.data()[c] + delta;
			ldet+=log(SSd);
			D(r,c) = 1.0/SSd;
		}
	}
#ifdef debugll
	std::cout << ldet;
	std::cout << "\n";
#endif
	MatrixXd DY = Y.array() * D.array();

	VectorXd XYA = VectorXd(nWeights);

	muint_t cumSumR = 0;

	MatrixXd covW = MatrixXd(nWeights,nWeights);
	for(muint_t termR = 0; termR < A.size();++termR){
		muint_t nW_AR = A[termR].rows();
		muint_t nW_XR = X[termR].cols();
		muint_t rowsBlock = nW_AR * nW_XR;
		MatrixXd XYAblock = X[termR].transpose() * DY * A[termR].transpose();
		XYAblock.resize(rowsBlock,1);
		XYA.block(cumSumR,0,rowsBlock,1) = XYAblock;

		muint_t cumSumC = 0;

		for(muint_t termC = 0; termC < A.size(); ++termC){//this does redundant computations, as the matrix is symmetric, change to start with termR instead
			muint_t nW_AC = A[termC].rows();
			muint_t nW_XC = X[termC].cols();
			muint_t colsBlock = nW_AC * nW_XC;
			MatrixXd block = MatrixXd::Zero(rowsBlock,colsBlock);
			if (R<C)
			{
				for(muint_t r=0; r<R; ++r)
				{
					MatrixXd AD = A[termR];
					AD.array().rowwise() *= D.row(r).array();
					MatrixXd AA = AD * A[termC].transpose();
					//sum up col matrices
					MatrixXd XX = X[termR].row(r).transpose() * X[termC].row(r);
					akron(block,AA,XX,true);
				}
			}
			else
			{//sum up col matrices
				for(muint_t c=0; c<C; ++c)
				{
					MatrixXd XD = X[termR];
					XD.array().colwise() *= D.col(c).array();
					MatrixXd XX = XD.transpose() * X[termC];
					//sum up col matrices
					MatrixXd AA = A[termR].col(c) * A[termC].col(c).transpose();
					akron(block,AA,XX,true);
				}
			}
			covW.block(cumSumR, cumSumC, rowsBlock, colsBlock) = block;
			cumSumC+=colsBlock;
		}
		cumSumR+=rowsBlock;
	}
	//std::cout << "covW = " << covW<<std::endl;
	MatrixXd W_vec = covW.colPivHouseholderQr().solve(XYA);
	//MatrixXd W_vec = covW * XYA;
	//std::cout << "W = " << W_vec<<std::endl;
	//std::cout << "XYA = " << XYA<<std::endl;

    // getting out Bsnp
    muint_t cumSum = 0;
    for(muint_t term = 0; term < A.size();++term)
    {
        muint_t currSize = X[term].cols() * A[term].rows();
        if (term==A.size()-1) {
            W = W_vec.block(cumSum,0,currSize,1);//
            //W.resize(X[term].cols(),A[term].rows());
        }
        cumSum+=currSize;
    }

	mfloat_t res = (Y.array()*DY.array()).sum();
	mfloat_t varPred = (W_vec.array() * XYA.array()).sum();
	res-= varPred;

	mfloat_t sigma = res/(mfloat_t)(R*C);

	mfloat_t nLL = 0.5 * ( R * C * (L2pi + log(sigma) + 1.0) + ldet);
#ifdef returnW
	covW = covW.inverse();	//here is another inverse!
	//std::cout << "covW.inverse() = " << covW<<std::endl;

	muint_t cumSum = 0;
	VectorXd F_vec = W_vec.array() * W_vec.array() /covW.diagonal().array() / sigma;//how to compute the inverse diagonal more efficiently?
	for(muint_t term = 0; term < A.size();++term)
	{
		muint_t currSize = X[term].cols() * A[term].rows();
		//W[term] = MatrixXd(X[term].cols(),A[term].rows());
		W[term] = W_vec.block(cumSum,0,currSize,1);//
		W[term].resize(X[term].cols(),A[term].rows());
		//F_tests[term] = MatrixXd(X[term].cols(),A[term].rows());
		F_tests[term] = F_vec.block(cumSum,0,currSize,1);//
		F_tests[term].resize(X[term].cols(),A[term].rows());
		cumSum+=currSize;
	}
#endif
	return nLL;
}


void CKroneckerLMM::updateDecomposition()  {
    //check that dimensions match
    this->num_samples = snps.rows();
    this->num_snps = snps.cols();
    this->num_pheno = pheno.cols();
    //this->num_covs = covs.cols();

    if (num_samples==0)
        throw CLimixException("LMM requires a non-zero sample size");

    if (num_snps==0)
        throw CLimixException("LMM requires non-zero SNPs");

    if (num_pheno==0)
        throw CLimixException("LMM requires non-zero phenotypes");

    if(!(num_samples == (muint_t) pheno.rows()) || !(num_samples == (muint_t) snps.rows()) )
        throw CLimixException("phenotypes and SNP dimensions inconsistent");

    //if(!num_samples == covs.rows())
    //    throw CLimixException("covariates and SNP dimensions inconsistent");

    //decomposition of K//should be a Cholesky for speed
	Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver2c(K2c);
    this->U2c = eigensolver2c.eigenvectors();
    this->S2c = eigensolver2c.eigenvalues();
	if (!(this->S2c(0)>1e-12)){
		throw CLimixException("The column covariance of the second covariance term has to be full rank, but is not.");
	}

	Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver2r(K2r);
    this->U2r = eigensolver2r.eigenvectors();
    this->S2r = eigensolver2r.eigenvalues();
	if (!(this->S2r(0)>1e-12)){
		throw CLimixException("The row covariance of the second covariance term has to be full rank, but is not.");
	}
        
	this->Rrot = this->U2r;
	this->Crot = this->U2c;
#ifdef debugkron
	std::cout << "Crot: \n"<< Crot << "\n";
	std::cout << "Rrot: \n"<< Rrot << "\n";
#endif		
	for (size_t r1=0;r1<(muint_t)this->pheno.rows();++r1)
	{
		for (size_t r2=0;r2<(muint_t)this->pheno.rows();++r2)
		{
			Rrot(r1,r2)/=sqrt(this->S2r(r2));
		}
	}
	for (size_t c1=0;c1<(muint_t)this->pheno.cols();++c1)
	{
		for (size_t c2=0;c2<(muint_t)this->pheno.cols();++c2)
		{
			Crot(c1,c2)/=sqrt(this->S2c(c2));
		}
	}
#ifdef debugkron
	std::cout << "Crot: \n"<< Crot << "\n";
	std::cout << "Rrot: \n"<< Rrot << "\n";
#endif
	MatrixXd S2U2K1r = this->Rrot.transpose() * this->K1r * this->Rrot;
	MatrixXd S2U2K1c = this->Crot.transpose() * this->K1c * this->Crot;
		

	Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver1c(S2U2K1c);
    this->U1c = eigensolver1c.eigenvectors();
    this->S1c = eigensolver1c.eigenvalues();

	Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver1r(S2U2K1r);
    this->U1r = eigensolver1r.eigenvectors();
    this->S1r = eigensolver1r.eigenvalues();
#ifdef debugkron
	MatrixXd K2ri_ = Rrot * Rrot.transpose(); 
	MatrixXd K2ci_ = Crot * Crot.transpose(); 
	std::cout << "K2ri_: \n"<<  K2ri_ << "\n";
	std::cout << "K2ci_: \n"<<  K2ci_ << "\n";
	std::cout << "K2ri : \n"<<  K2r.inverse() << "\n";
	std::cout << "K2ci : \n"<<  K2c.inverse() << "\n";
#endif
	Rrot *= this->U1r;//multiply from right with U1r
	Crot *= this->U1c;//multiply from right with U1c

#ifdef debugkron
	MatrixXd K2ri =   Rrot * Rrot.transpose(); 
	MatrixXd K2ci =   Crot * Crot.transpose(); 
	std::cout << "K2ri_: \n"<<  K2ri << "\n";
	std::cout << "K2ci_: \n"<<  K2ci << "\n";
	std::cout << "K2ri : \n"<<  K2r.inverse() << "\n";
	std::cout << "K2ci : \n"<<  K2c.inverse() << "\n";
		
	mfloat_t delta = 0.01;
		
	MatrixXd kronK1;
	akron(kronK1,K1c,K1r,false);
	std::cout << "K1r: \n"<<  K1r << "\n";
	std::cout << "K1c: \n"<<  K1c << "\n";
	std::cout << "K2r: \n"<<  K2r << "\n";
	std::cout << "K2c: \n"<<  K2c << "\n";
		
	MatrixXd kronK2;
	akron(kronK2,K2c,K2r,false);
	MatrixXd kronK = kronK1+delta*kronK2;
	std::cout << "kronK1: \n"<<  kronK1 << "\n";
	std::cout << "kronK2: \n"<<  kronK2 << "\n";
		
	MatrixXd kronKi = kronK.inverse();
		
	akron(kronK1,Crot,Rrot,false);
	std::cout << "Crot: \n"<< Crot << "\n";
	std::cout << "Rrot: \n"<< Rrot << "\n";
	for (size_t c1=0;c1<this->pheno.cols();++c1)
	{
		for (size_t c2=0;c2<this->pheno.cols();++c2)
		{
			for (size_t r1=0;r1<this->pheno.rows();++r1)
			{
				for (size_t r2=0;r2<this->pheno.rows();++r2)
				{
					mfloat_t S = this->S1c(c2)*this->S1r(r2) + delta;
					kronK1(c1*this->pheno.rows()+r1,c2*this->pheno.rows()+r2)/=sqrt(S);
				}
			}
		}
	}

	MatrixXd kronKi_ = kronK1*kronK1.transpose();
	std::cout << "Ki_: \n"<< kronKi_ << "\n";
	std::cout << "Ki : \n"<< kronKi << "\n";
	mfloat_t diff = (kronKi-kronKi_).norm();
	std::cout << "diff: "<< diff << "\n";
	
	mfloat_t logdet = log(kronK.determinant());
	mfloat_t logdet_ = 0.0;
	for (size_t r1=0;r1<this->pheno.rows();++r1)
	{
		logdet_+=pheno.cols()*log(this->S2r(r1));
	}
	for (size_t c1=0;c1<this->pheno.cols();++c1)
	{
		logdet_+=pheno.rows()*log(this->S2c(c1));
	}
	for (size_t c1=0;c1<this->pheno.cols();++c1)
	{
		for (size_t r1=0;r1<this->pheno.rows();++r1)
		{
			logdet_+=log(this->S1c(c1)*this->S1r(r1) + delta);
		}
	} 
	std::cout << "logdet :" << logdet<<"\n";
	std::cout << "logdet_:" << logdet_<<"\n";
	std::cout << "diff:" << logdet_-logdet<<"\n";

#endif

	//snps
    Usnps.noalias() = this->Rrot.transpose() * snps;

	//SNP column design matrix
	//design for SNPs
	this->snpcoldesignU.noalias() = snpcoldesign * this->Crot;
	if (this->snpcoldesign0_inter.rows()!=0) //check if interaction design matrix is set
	{
		this->snpcoldesignU0_inter.noalias() = snpcoldesign0_inter * this->Crot;
	}

	//phenotype
	Upheno.noalias() = this->Rrot.transpose() * pheno * this->Crot;

	//need column design matrix
	//rotate covariates
	this->Urowdesign0=MatrixXdVec();
	this->coldesignU0=MatrixXdVec();
	for(muint_t term = 0; term<coldesign0.size();++term)
	{
		Urowdesign0.push_back(Rrot.transpose() * rowdesign0[term]);
		coldesignU0.push_back(coldesign0[term] * Crot);
	}//end loop over covariates terms		
}


//the operator evaluates nLLeval
mfloat_t nLLevalKronFunctor::operator()(const mfloat_t logdelta)
   {   
	   return CKroneckerLMM::nLLeval(logdelta,A,X,Y,S_C1,S_R1,S_C2,S_R2, W);
   }
nLLevalKronFunctor::nLLevalKronFunctor(
		const MatrixXdVec A,
		const MatrixXdVec X,
		const MatrixXd Y,
		const MatrixXd S_C1,
		const MatrixXd S_R1,
		const MatrixXd S_C2,
		const MatrixXd S_R2,
        MatrixXd W){
	this->A=A;
	this->X=X;
	this->Y=Y;
	this->S_C1=S_C1;
	this->S_C2=S_C2;
	this->S_R1=S_R1;
	this->S_R2=S_R2;
    this->W=W;
}
nLLevalKronFunctor::~nLLevalKronFunctor(){}
} // end namespace

