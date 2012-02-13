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
            nllgrid(i, 0) = CKroneckerLMM::nLLeval(&f_tests, ldeltagrid(i), WkronDiag, WkronBlock, UX, UYU, S_C, S_R);
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

    mfloat_t CKroneckerLMM::nLLeval(MatrixXd *F_tests, mfloat_t ldelta, const MatrixXd & WkronDiag, const MatrixXd & WkronBlock, const MatrixXd & UX, const MatrixXd & UYU, const VectorXd & S_C, const VectorXd & S_R)
    {
        muint_t n = UX.rows();
        muint_t d = UX.cols();
        muint_t p = UYU.cols();
        assert(UYU.cols() == S_C.rows());
        assert(UYU.rows() == S_R.rows());
        assert(UYU.rows() == UX.rows());
        assert((muint_t)WkronDiag.cols() == d);
        assert((muint_t)WkronBlock.cols() == d);
        assert((muint_t)WkronDiag.rows() * (muint_t)WkronBlock.rows() == p);
        mfloat_t delta = exp(ldelta);
        mfloat_t ldet = 0.0;
        (*F_tests).resize(d, WkronDiag.rows());
        MatrixXd beta = MatrixXd(d, (muint_t)(((((WkronDiag.rows()))))));
        MatrixXd Sd = MatrixXd(S_R.rows(), S_C.rows());
        for(muint_t col = 0;col < (muint_t)(((((S_C.rows())))));++col){
            for(muint_t row = 0;row < (muint_t)(((((S_R.rows())))));++row){
                Sd(row, col) = S_R(row) * S_C(col) + delta;
                ldet += std::log((mfloat_t)(((((Sd(row, col)))))));
            }
        }

        muint_t phen = 0;
        MatrixXd XSdi = MatrixXd(UX.rows(), UX.cols());
        mfloat_t res = (UYU.array() * UYU.array() / Sd.array()).sum();
        for(muint_t i_diag = 0;i_diag < (muint_t)(((((WkronDiag.rows())))));++i_diag){
            MatrixXd XSX = MatrixXd::Zero(d, d);
            MatrixXd XSY = MatrixXd::Zero(d, 1);
            for(muint_t i_block = 0;i_block < (muint_t)(((((WkronBlock.rows())))));++i_block){
                VectorXd Sd_p = Sd.block(0, phen, n, 1);
                for(muint_t dim = 0;dim < d;++dim){
                    XSdi.block(0, dim, n, 1).array() = (UX.block(0, dim, n, 1).array() / Sd.block(0, phen, n, 1).array()) * (WkronDiag(i_diag, dim) * WkronBlock(i_block, dim));
                }
                XSX += XSdi.transpose() * UX;
                XSY += XSdi.transpose() * UYU.block(0, phen, n, 1);
                ++phen;
            }

            //least sqaures solution of XSX*beta = XSY
            //decomposition of K
            Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver(XSX);
            MatrixXd U_X = eigensolver.eigenvectors();
            MatrixXd S_X = eigensolver.eigenvalues();
            beta.block(0, i_diag, d, 1) = U_X.transpose() * XSY;
            //MatrixXd S_i = MatrixXd::Zero(d,d);
            for(size_t dim = 0;dim < d;++dim){
                if(S_X(dim, 0) > 3E-8){
                    beta(dim, i_diag) /= S_X(dim, 0);
                    for(size_t dim2 = 0;dim2 < d;++dim2){
                        (*F_tests)(dim2, i_diag) += U_X(dim2, dim) * U_X(dim2, dim) / S_X(dim, 0);
                    }
                    //S_i(dim,dim) = 1.0/S_X(dim,0);
                }
                else{
                    beta(dim, i_diag) = 0.0;
                }
            }

            beta.block(0, i_diag, d, 1) = U_X * beta.block(0, i_diag, d, 1);
            res -= (XSY.array() * beta.block(0, i_diag, d, 1).array()).sum();
        }

        //sqared residuals
        mfloat_t sigg2 = res / (n * p);
        //compute the F-statistics
        (*F_tests).array() = beta.array() * beta.array() / (*F_tests).array();
        (*F_tests).array() /= sigg2;
        double nLL = 0.5 * (n * p * L2pi + ldet + n * p + n * p * log(sigg2));
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
        nLL0(0) = this->nLLeval(&f_tests, ldelta0(0), WkronDiag0, WkronBlock0, Ucovs, Upheno, this->S_C, this->S_R);
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
            nLL(0, is) = CKroneckerLMM::nLLeval(&f_tests, ldelta(0, is), WkronDiag, WkronBlock, UX, Upheno, this->S_C, this->S_R);
            //				gpmix::nLLeval(&f_tests, ldelta(0, is), Upheno, UX, S);
            //TODO:
            //rotate F_tests to get per pheno P-values
            //4. calc lod score
            //TODO: calculate dofs for arbitrary WkronDiag and WkronBlock, currently we expect all ones...
            mfloat_t dof = ((mfloat_t)(((((Upheno.cols()))))) / (mfloat_t)(((((WkronBlock.rows()))))));
            this->pv(0, is) = Gamma::gammaQ(nLL0(0, 0) - nLL(0, is), (double)(((((((0.5) * dof)))))));
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


