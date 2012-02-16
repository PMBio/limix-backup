/*
 * ALmm.cpp
 *
 *  Created on: Nov 27, 2011
 *      Author: stegle
 */

#include "kronecker_lmm.h"
#include "gpmix/utils/gamma.h"
#include "gpmix/utils/fisherf.h"
#include "gpmix/mean/CKroneckerMean.h"

//RTTI support
#include <typeinfo>

namespace gpmix {




    /*KroneckerLMM*/
    CKroneckerLMM::CKroneckerLMM()
    {
    }

    CKroneckerLMM::~CKroneckerLMM()
    {
    }

    void CKroneckerLMM::updateDecomposition() throw (CGPMixException)
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

    mfloat_t CKroneckerLMM::nLLeval(std::vector<MatrixXd>& W, std::vector<MatrixXd>& F_tests, mfloat_t ldelta, const std::vector<MatrixXd>& A, const std::vector<MatrixXd>& X, const MatrixXd& Y, const VectorXd& S_C, const VectorXd& S_R)
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

        muint_t cumSumRowR = 0;
        muint_t cumSumColR = 0;

        MatrixXd covW = MatrixXd(nWeights,nWeights);
        for(muint_t termR = 0; termR < A.size();++termR){
        	muint_t nW_AR = A[termR].rows();
        	muint_t nW_XR = A[termR].cols();
        	muint_t rowsBlock = nW_AR * nW_XR;

        	XYA.block(cumSumRowR,0,rowsBlock,1) = X[termR].transpose() * DY * A[termR].transpose();

        	muint_t cumSumRowC = 0;
        	muint_t cumSumColC = 0;
        	for(muint_t termC = 0; termC < A.size(); ++termC){
            	muint_t nW_AC = A[termC].rows();
            	muint_t nW_XC = A[termC].cols();
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
            	covW.block(cumSumRowR * cumSumColR, cumSumRowC * cumSumColC,rowsBlock,colsBlock) = block;
            }
        }

        covW = covW.inverse();
        MatrixXd W_vec = covW * XYA;

        mfloat_t res = (Y.array()*DY.array()).sum();
        res -= (W_vec.array() * XYA.array()).sum();

        mfloat_t sigma = res/(mfloat_t)(R*C);

        mfloat_t nLL = 0.5 * ( R * C * (L2pi + log(sigma)) + ldet);

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

        return nLL;
    }

    void CKroneckerLMM::process() throw (CGPMixException)
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






/* namespace gpmix */
}


