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

#ifndef KRONECKER_LMM_H_
#define KRONECKER_LMM_H_

#include "limix/LMM/lmm.h"
#include "limix/gp/gp_kronSum.h"
#include "limix/utils/brentc.h"

namespace limix {

/*! \brief Core comptuational class for Kronecker LMM
 *
 * Fill me
 */
class CLMMKroneckerCore : public CLMMCore
{

	//TODO: create inline functions along the lines for CLMMCore optdeltaEx, nllEvalEx, etc.
	template <typename Derived1, typename Derived2,typename Derived3,typename Derived4, typename Derived5,typename Derived6,typename Derived7,typename Derived8>
	inline void nLLevalEx(const Eigen::MatrixBase<Derived1>& AObeta_, const Eigen::MatrixBase<Derived2>& AObeta_ste_, const Eigen::MatrixBase<Derived3>& AOsigma_, const Eigen::MatrixBase<Derived4>& AOF_tests_,const Eigen::MatrixBase<Derived5>& AOnLL_,const Eigen::MatrixBase<Derived6>& UY, const Eigen::MatrixBase<Derived7>& UX, const Eigen::MatrixBase<Derived8>& S,mfloat_t ldelta,bool calc_ftest=false,bool calc_ste=false, bool REML = false);
};


/*! \brief Kronecker mixed model inference for pre-fitted covariance matrices
 *
 * Class is derived from CLMM, however provides special functions for Kronecker phenotypes
 *
 * y ~ N(fixed(X), s2(K1c \kron K1r + \delta K2c \kron K2r))
 *
 * fixed effects are decomposed of kronecke terms that involve a predefined set of covariates and a linear set of SNPs
 *
 *
 */
class CKroneckerLMM : public CLMMKroneckerCore, public ALMM
{
protected:
	MatrixXd K1r;
	MatrixXd K1c;
	MatrixXd K2r;
	MatrixXd K2c;
	MatrixXd Rrot;	//(K1r^{-1/2} K2r K1r^{-1/2})
	MatrixXd Crot;	//(K1c^{-1/2} K2c K1c^{-1/2})
	//decompositions
	MatrixXd U1r;
	VectorXd S1r;
	MatrixXd U1c;
	VectorXd S1c;
	MatrixXd U2r,S2r;//These are actually decompositions of S1U1K2r (K1r^{-1/2} K2r K1r^{-1/2})
	MatrixXd U2c,S2c;//These are actually decompositions of S1U1K2c (K1c^{-1/2} K2c K1c^{-1/2})
	muint_t num_intervals0_inter;
	
	MatrixXdVec rowdesign0;
	MatrixXdVec Urowdesign0;
	MatrixXdVec coldesign0;
	MatrixXdVec coldesignU0;
	MatrixXd snps;
	MatrixXd Usnps;
	MatrixXd snpcoldesign0_inter;
	MatrixXd snpcoldesignU0_inter;
	MatrixXd snpcoldesign;
	MatrixXd snpcoldesignU;
	MatrixXd nLL0;
	MatrixXd nLL0_inter;
	MatrixXd nLLAlt;
	MatrixXd ldelta0;
	MatrixXd ldelta0_inter;
	MatrixXd ldeltaAlt;
    //Weights  
    MatrixXd W;
    MatrixXd beta_snp;
public:
	CKroneckerLMM();
	virtual ~CKroneckerLMM();

	virtual void process() ;
	virtual void updateDecomposition() ;

	/* getters and setters*/

	/*! set row covarince term1 */
	void setK1r(const MatrixXd& K1r)
	{ this->K1r = K1r;}
	/*! set col covarince term1 */
	void setK1c(const MatrixXd& K1c)
	{ this->K1c = K1c;}
	/*! set row covarince term2 */
	void setK2r(const MatrixXd& K2r)
	{this->K2r = K2r;}
	/*! set col covarince term2 */
	void setK2c(const MatrixXd& K2c)
	{this->K2c = K2c;}
	/*! set phenotype */
	void setPheno(const MatrixXd& Y)
	{this->pheno = Y;}
	/*! set SNPs */
	void setSNPs(const MatrixXd& snps)
	{this->snps = snps;}
	void setSNPcoldesign(const MatrixXd& design)
	{this->snpcoldesign = design;}
	void setSNPcoldesign0_inter(const MatrixXd& design)
	{this->snpcoldesign0_inter = design;}
	
	void setNumIntervals0_inter(muint_t num_intervals0_inter)
	{this->num_intervals0_inter=num_intervals0_inter;}

	muint_t getNumIntervals0_inter()
	{return this->num_intervals0_inter;}

	void agetNLL0(MatrixXd* out)
	{
		(*out) = nLL0;
	}
	void agetNLL0_inter(MatrixXd* out)
	{
		(*out) = nLL0_inter;
	}	
	void agetNLLAlt(MatrixXd *out)
	{
		(*out) = nLLAlt;
	}

	void agetLdeltaAlt(MatrixXd *out)
	{
		(*out) = ldeltaAlt;
	}
	void agetLdelta0(MatrixXd *out)
	{
		(*out) = ldelta0;
	}
	void agetLdelta0_inter(MatrixXd *out)
	{
		(*out) = ldelta0_inter;
	}
	void agetBetaSNP(MatrixXd *out)
	{
		(*out) = beta_snp;
	}


	/*! set Vecotr of covariates
	 * \param covsR: vector of row covariate terms
	 * \param covsCol: vector of column covariate terms
	 * */
	void setCovariates(const MatrixXdVec& covsR, const MatrixXdVec& covsCol)
	{
		this->coldesign0 = covsCol;
		this->rowdesign0 = covsR;
	}

	/*! add term to the backgroudn covaraite
	 * \param covR: row covariance design
	 * \param covC: column covariance design
	 * */
	void addCovariates(const MatrixXd& covR, const MatrixXd& covCol);

	/* set a particular covariate element
	\param index: index of the covarite vector be set
	\param covarR: covariate row design
	\param covC: covariate column design
	  */
	void setCovariates(muint_t index,const MatrixXd& covR, const MatrixXd& covC);


	static mfloat_t nLLeval(mfloat_t ldelta, const MatrixXdVec& A,const MatrixXdVec& X, const MatrixXd& Y, const VectorXd& S_C1, const VectorXd& S_R1, const VectorXd& S_C2, const VectorXd& S_R2, MatrixXd& W);
	static mfloat_t optdelta(mfloat_t& ldelta_opt, const MatrixXdVec& A,const MatrixXdVec& X, const MatrixXd& Y, const VectorXd& S_C1, const VectorXd& S_R1, const VectorXd& S_C2, const VectorXd& S_R2, mfloat_t ldeltamin, mfloat_t ldeltamax, muint_t numintervals);
	//set precompute decompositions
	//void setMatrices(const MatrixXd)
	//getters: TODO
};


/* inline functions */
template <typename Derived1, typename Derived2,typename Derived3,typename Derived4, typename Derived5,typename Derived6,typename Derived7,typename Derived8>
inline void CLMMKroneckerCore::nLLevalEx(const Eigen::MatrixBase<Derived1>& AObeta_, const Eigen::MatrixBase<Derived2>& AObeta_ste_, const Eigen::MatrixBase<Derived3>& AOsigma_, const Eigen::MatrixBase<Derived4>& AOF_tests_,const Eigen::MatrixBase<Derived5>& AOnLL_,const Eigen::MatrixBase<Derived6>& UY, const Eigen::MatrixBase<Derived7>& UX, const Eigen::MatrixBase<Derived8>& S,mfloat_t ldelta,bool calc_ftest,bool calc_ste, bool REML)
{
	//cast out arguments
	Eigen::MatrixBase<Derived1>& AObeta = const_cast< Eigen::MatrixBase<Derived1>& >(AObeta_);
	Eigen::MatrixBase<Derived2>& AObeta_ste = const_cast< Eigen::MatrixBase<Derived2>& >(AObeta_ste_);
	Eigen::MatrixBase<Derived3>& AOsigma = const_cast< Eigen::MatrixBase<Derived3>& >(AOsigma_);
	Eigen::MatrixBase<Derived4>& AOF_tests = const_cast< Eigen::MatrixBase<Derived4>& >(AOF_tests_);
	Eigen::MatrixBase<Derived5>& AOnLL = const_cast< Eigen::MatrixBase<Derived5>& >(AOnLL_);


} //end ::nLLevalEx


class nLLevalKronFunctor: public BrentFunctor{
	MatrixXd Y;
	MatrixXdVec X;
	MatrixXdVec A;
	MatrixXd S_C1;
	MatrixXd S_C2;
	MatrixXd S_R1;
	MatrixXd S_R2;
    MatrixXd W;
public:
   nLLevalKronFunctor(	
		const MatrixXdVec A,
		const MatrixXdVec X,
		const MatrixXd Y,
		const MatrixXd S_C1,
		const MatrixXd S_R1,
		const MatrixXd S_C2,
		const MatrixXd S_R2,
        MatrixXd W);
   ~nLLevalKronFunctor();
   virtual mfloat_t operator()(const mfloat_t logdelta);
};

} //end namespace LIMIX
#endif /* KRONECKER_LMM_H_ */
