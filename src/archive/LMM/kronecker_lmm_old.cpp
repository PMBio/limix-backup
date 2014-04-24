// Copyright(c) 2014, The LIMIX developers (Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
// All rights reserved.
//
// LIMIX is provided under a 2-clause BSD license.
// See license.txt for the complete license.

#include "kronecker_lmm_old.h"
#include "limix/utils/gamma.h"
#include "limix/utils/fisherf.h"
#include "limix/mean/CKroneckerMean.h"

//RTTI support
#include <typeinfo>

namespace limix {


void CKroneckerLMM_old::initTestingGP()
{
	//call base object init
	CGPLMM::initTesting();
	//store decompositions
	Ur = gp->getCache()->covar_r->rgetUK();
	Uc = gp->getCache()->covar_c->rgetUK();
	Sr = gp->getCache()->covar_r->rgetSK();
	Sc = gp->getCache()->covar_c->rgetSK();
}

void CKroneckerLMM_old::initTestingK()
{
	//assert that dimensions match
	checkConsistency();
	//additional checks for Kr and Kc
	if(Kr.rows()!=snps.rows())
	{
		throw CLimixException("KroneckerLMM: row covariance size missmatch");
	}
	if (Kc.rows()!=pheno.cols())
	{
		throw CLimixException("KroneckerLMM: column covariance size missmatch!");
	}
	//carry out decomposition for caches
	Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolverR(Kr);
	Ur = eigensolverR.eigenvectors();
	Sr = eigensolverR.eigenvalues();
	Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolverC(Kc);
	Uc = eigensolverC.eigenvectors();
	Sc = eigensolverC.eigenvalues();
}


mfloat_t CKroneckerLMM_old::nLLeval(mfloat_t ldelta, const MatrixXdVec& A,const MatrixXdVec& X, const MatrixXd& Y, const VectorXd& S_C, const VectorXd& S_R)
{
	muint_t R = (muint_t)Y.rows();
	muint_t C = (muint_t)Y.cols();
	assert(A.size() == X.size());
	assert(R == (muint_t)S_R.rows());
	assert(C == (muint_t)S_C.rows());
	muint_t nWeights = 0;
	for(muint_t term = 0; term < A.size();++term)
	{
		assert((muint_t)(X[term].rows())==R);
		assert((muint_t)(A[term].cols())==C);
		nWeights+=(muint_t)(A[term].rows()) * (muint_t)(X[term].cols());
	}
	mfloat_t delta = exp(ldelta);
	mfloat_t ldet = 0.0;

	//build D and compute the logDet of D
	MatrixXd D = MatrixXd(R,C);
	for (muint_t r=0; r<R;++r)
	{
		for (muint_t c=0; c<C;++c)
		{
			mfloat_t SSd = S_R.data()[r]*S_C.data()[c] + delta;
			ldet+=log(SSd);
			D(r,c) = 1.0/SSd;
		}
	}

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

		for(muint_t termC = 0; termC < A.size(); ++termC){
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

	mfloat_t res = (Y.array()*DY.array()).sum();
	mfloat_t varPred = (W_vec.array() * XYA.array()).sum();
	res-= varPred;

	mfloat_t sigma = res/(mfloat_t)(R*C);

	mfloat_t nLL = 0.5 * ( R * C * (L2pi + log(sigma) + 1.0) + ldet);
#ifdef returnW
	covW = covW.inverse();
	//std::cout << "covW.inverse() = " << covW<<std::endl;

	muint_t cumSum = 0;
	VectorXd F_vec = W_vec.array() * W_vec.array() /covW.diagonal().array() / sigma;
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


mfloat_t CKroneckerLMM_old::optdelta(mfloat_t& ldelta_opt, const MatrixXdVec& A,const MatrixXdVec& X, const MatrixXd& Y, const VectorXd& S_C, const VectorXd& S_R, mfloat_t ldeltamin, mfloat_t ldeltamax, muint_t numintervals)
{
    //grid variable with the current likelihood evaluations
    MatrixXd nllgrid    = MatrixXd::Ones(numintervals,1).array()*HUGE_VAL;
    MatrixXd ldeltagrid = MatrixXd::Zero(numintervals, 1);
    //current delta
    mfloat_t ldelta = ldeltamin;
    mfloat_t ldeltaD = (ldeltamax - ldeltamin);
    ldeltaD /= ((mfloat_t)(numintervals) - 1.0);
    mfloat_t nllmin = HUGE_VAL;
    mfloat_t ldeltaopt_glob = 0.0;
    MatrixXd f_tests;
    for(muint_t i = 0;i < numintervals;i++){
    	nllgrid(i, 0) = CKroneckerLMM_old::nLLeval(ldelta, A, X, Y, S_C, S_R);
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

    //std::cout << "\n\n nLL_i:\n" << nllgrid;
    ldelta_opt = ldeltaopt_glob;
    return nllmin;
}

void CKroneckerLMM_old::process() 
{
	//1. init testing engine
	//do we have a gp object?
	initTestingGP();
	//do we have manual Kr?
	if ((!isnull(Kr)) && (!isnull(Kc)))
	{
		//use manually specified Kr/Kc
		initTestingK();
	}
	//2. init result arrays
	nLL0.resize(1,num_snps);
	nLLAlt.resize(1,num_snps);
	ldelta0.resize(1,num_snps);
	ldeltaAlt.resize(1,num_snps);
	pv.resize(1,num_snps);

	//estimate effective degrees of freedom: difference in number of weights
	MatrixXd covsRot = Ur.transpose() * this->covs;
	MatrixXd snpsRot = Ur.transpose() * this->snps;

	//initialize vectors of rotated fixed effects and designs:
	MatrixXdVec X0Rot;
	MatrixXdVec XAltRot;
	MatrixXdVec A0Rot;
	MatrixXdVec AAltRot;

	//get alt terms and null model terms
	//TODO: mean terms don't work.
	VecLinearMean& terms0 = mean0->getTerms();
	VecLinearMean& termsAlt = meanAlt->getTerms();

	//null model terms
	for(VecLinearMean::const_iterator iter = terms0.begin(); iter!=terms0.end();iter++)
	{
		MatrixXd X = iter[0]->getFixedEffects();
		MatrixXd A = iter[0]->getA();
		A0Rot.push_back(A*Uc);
		X0Rot.push_back(Ur.transpose()*X);
	}
	//alternative model terms
	for(VecLinearMean::const_iterator iter = termsAlt.begin(); iter!=termsAlt.end();iter++)
	{
		MatrixXd X = iter[0]->getFixedEffects();
		MatrixXd A = iter[0]->getA();
		AAltRot.push_back(A*Uc);
		XAltRot.push_back(Ur.transpose()*X);
	}
	//create pointer to the last term in alt models which is SNP-dependent
	MatrixXd& XsnpRot = XAltRot[XAltRot.size()-1];
	//XsnpRot(0,0) = 99;
	//std::cout << XsnpRot <<"\n\n";
	//std::cout << XAltRot[XAltRot.size()-1] <<"\n\n";


	//evaluate cache details

	std::cout << "standardize on"<< "\n";
	Sc/=((Sc).sum()/Sc.rows());
	Sr/=((Sr).sum()/Sr.rows());


	MatrixXd Yrot;
	akronravel(Yrot,Ur.transpose(),Uc.transpose(),this->pheno);	//note that this one does not match with the Yrot from the gp.
	//evaluate null model
	mfloat_t ldelta0_ = 0.0;
	mfloat_t nLL0_ = CKroneckerLMM_old::optdelta(ldelta0_,A0Rot,X0Rot, Yrot, Sc, Sr, ldeltamin0, ldeltamax0, num_intervals0);
	//store delta0
	ldelta0.setConstant(ldelta0_);
	nLL0.setConstant(nLL0_);

	//2. loop over SNPs
	mfloat_t deltaNLL;
	for (muint_t is=0;is<num_snps;++is)
	{
		//0. update mean term
		XsnpRot = snpsRot.block(0,is,num_samples,1);
		//std::cout << XsnpRot <<"\n\n";
		//2. evaluate alternative model
		mfloat_t ldelta;
		mfloat_t nLL;
		if (num_intervalsAlt>0)
		{
			nLL = CKroneckerLMM_old::optdelta(ldelta,AAltRot,XAltRot, Yrot, Sc, Sr, ldeltaminAlt, ldeltamaxAlt, num_intervalsAlt);
		}
		else
		{
			ldelta = ldelta0(0,is);
			nLL = this->nLLeval(ldelta,AAltRot,XAltRot,Yrot,Sc,Sr);
		}
		nLLAlt(0,is) = nLL;
		ldeltaAlt(0,is) = ldelta;
		deltaNLL = nLL0(0,is) - nLLAlt(0,is);
		//std::cout<< "nLL0(0,is)"<< nLL0(0,is)<< "nLLAlt(0,is)" << nLLAlt(0,is)<< "\n";
		if (deltaNLL<0)
		{
			std::cout << "outch" << "\n";
			deltaNLL = 1E-10;
		}
		//3. pvalues
		this->pv(0, is) = stats::Gamma::gammaQ(deltaNLL, (double)(0.5) * getDegreesFredom());
	}
}

#if 0


    /*KroneckerLMM*/
    CKroneckerLMM::CKroneckerLMM()
    {
    }

    CKroneckerLMM::~CKroneckerLMM()
    {
    }

    void CKroneckerLMM::updateDecomposition() 
    {
        //TODO: think about caching procedures:
    }

    mfloat_t CKroneckerLMM::optdelta(const MatrixXd & UX, const MatrixXd & UYU, const VectorXd & S_C, const VectorXd & S_R, const muint_t numintervals, const mfloat_t ldeltamin, const mfloat_t ldeltamax, const MatrixXd & WkronDiag, const MatrixXd & WkronBlock)
    {
        //grid variable with the current likelihood evaluations
        MatrixXd nllgrid     = MatrixXd::Ones(numintervals,1).array()*HUGE_VAL;
        MatrixXd ldeltagrid = MatrixXd::Zero(numintervals, 1);
        //current delta
        mfloat_t ldelta = ldeltamin;
        mfloat_t ldeltaD = (ldeltamax - ldeltamin);
        ldeltaD /= ((mfloat_t)((((((((numintervals)))))))) - 1);
        mfloat_t nllmin = HUGE_VAL;
        mfloat_t ldeltaopt_glob = 0;
        MatrixXd f_tests;
        for(muint_t i = 0;i < numintervals;i++){
        	std::cout <<"commented out the nLL\n";//TODO commented out the nLL
            nllgrid(i, 0) = 0.0;//CKroneckerLMM::nLLeval(&f_tests, ldeltagrid(i), WkronDiag, WkronBlock, UX, UYU, S_C, S_R);
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

        //std::cout << "\n\n nLL_i:\n" << nllgrid;
        return ldeltaopt_glob;
    }


    void CKroneckerLMM::process() 
    {
        this->Usnps = this->U_R.transpose() * this->snps;
        this->Upheno = this->U_R.transpose() * this->pheno * this->U_C;
        this->Ucovs = this->U_R.transpose() * this->covs;
        //resize phenotype output
        this->pv.resize(1, this->snps.cols());
        //this->pv.resize(pheno.cols(), this->snps.cols());
        //result matrices: think about what to store in the end
        MatrixXd ldelta0(1, 1);
        MatrixXd ldelta(1, snps.cols());
        MatrixXd nLL0(1, 1);
        MatrixXd nLL(1, snps.cols());
        //reserve memory for snp-wise foreground model
        MatrixXd UX(snps.rows(), covs.cols() + 1);
        //store covariates upfront
        UX.block(0, 0, snps.rows(), covs.cols()) = Ucovs;
        MatrixXd f_tests;
        //fit delta on null model
        ldelta0(0) = CKroneckerLMM::optdelta(Ucovs, Upheno, this->S_C, this->S_R, this->num_intervals0, this->ldeltamin0, this->ldeltamax0, WkronDiag0, WkronBlock0);
    	std::cout <<"commented out the nLL\n";//TODO commented out the nLL
        nLL0(0) = 0.0;//this->nLLeval(&f_tests, ldelta0(0), WkronDiag0, WkronBlock0, Ucovs, Upheno, this->S_C, this->S_R);
        for(muint_t is = 0;is < (muint_t)(((((snps.cols())))));is++){
            //1. construct foreground testing SNP transformed
            UX.block(0, covs.cols(), snps.rows(), 1) = Usnps.block(0, is, snps.rows(), 1);
            //2. fit delta
            if(this->num_intervalsAlt > 0)
                //fit delta on alt model also
                ldelta(0, is) = CKroneckerLMM::optdelta(UX, Upheno, this->S_C, this->S_R, this->num_intervalsAlt, this->ldeltaminAlt, this->ldeltamaxAlt, WkronDiag, WkronBlock);

            else
                ldelta(0, is) = ldelta0(0);

            //3. evaluate
            MatrixXd f_tests;
        	std::cout <<"commented out the nLL\n";//TODO commented out the nLL
            nLL(0, is) = 0.0;//CKroneckerLMM::nLLeval(&f_tests, ldelta(0, is), WkronDiag, WkronBlock, UX, Upheno, this->S_C, this->S_R);
            //				gpmix::nLLeval(&f_tests, ldelta(0, is), Upheno, UX, S);
            //TODO:
            //rotate F_tests to get per pheno P-values
            //4. calc lod score
            //TODO: calculate dofs for arbitrary WkronDiag and WkronBlock, currently we expect all ones...
            mfloat_t dof = ((mfloat_t)(((((Upheno.cols()))))) / (mfloat_t)(((((WkronBlock.rows()))))));
            this->pv(0, is) = Gamma::gammaQ((double)(nLL0(0, 0) - nLL(0, is)), (double)(((((((0.5) * dof)))))));
    } //end for phenotypes
}

void CKroneckerLMM::setKronStructure(const MatrixXd & WkronDiag0, const MatrixXd & WkronBlock0, const MatrixXd & WkronDiag, const MatrixXd & WkronBlock)
{
    this->WkronDiag0 = WkronDiag0;
    this->WkronBlock0 = WkronBlock0;
    this->WkronDiag = WkronDiag;
    this->WkronBlock = WkronBlock;
}

void CKroneckerLMM::getK_C(MatrixXd *out) const
{
    (*out) = C;
}

void CKroneckerLMM::setK_C(const MatrixXd & C)
{
    this->C = C;
    Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver(C);
    this->U_C = eigensolver.eigenvectors();
    this->S_C = eigensolver.eigenvalues();
}

void CKroneckerLMM::setK_C(const MatrixXd & C, const MatrixXd & U_C, const VectorXd & S_C)
{
    this->C = C;
    this->U_C = U_C;
    this->S_C = S_C;
}

void CKroneckerLMM::getK_R(MatrixXd *out) const
{
    (*out) = R;
}

void CKroneckerLMM::setK_R(const MatrixXd & R)
{
    this->R = R;
    Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver(R);
    this->U_R = eigensolver.eigenvectors();
    this->S_R = eigensolver.eigenvalues();
}

void CKroneckerLMM::setK_R(const MatrixXd & R, const MatrixXd & U_R, const VectorXd & S_R)
{
    this->R = R;
    this->U_R = U_R;
    this->S_R = S_R;
}

MatrixXd CKroneckerLMM::getK_R() const
{
    return R;
}

MatrixXd CKroneckerLMM::getK_C() const
{
    return C;
}

#endif




/* namespace limix */
}


