/*
 * CMultiTraitVqtl.h
 *
 *  Created on: Jul 26, 2012
 *      Author: stegle
 */

#ifndef CMULTITRAITVQTL_H_
#define CMULTITRAITVQTL_H_

#include "limix/types.h"
#include "limix/gp/gp_base.h"
#include "limix/covar/linear.h"
#include "limix/covar/combinators.h"
#include "limix/covar/fixed.h"
#include "limix/likelihood/likelihood.h"




namespace limix {


//rename argout operators for swig interface
#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
//ignore C++ versions
%ignore Cvqtl::getPheno;
%ignore Cvqtl::getPv;
%ignore Cvqtl::getSnps;
%ignore Cvqtl::getCovs;
%ignore Cvqtl::getK;
%ignore Cvqtl::getPermutation;

%rename(getPheno) Cvqtl::agetPheno;
%rename(getSnps) Cvqtl::agetSnps;
%rename(getCovs) Cvqtl::agetCovs;
//%template(IndexVec) std::vector<MatrixXi>;
#endif



//vector on SNP sets

typedef std::vector<MatrixXd> MatrixXdVec;
//typedef std::vector<bool> Trait

class CMultiTraitVQTL {
protected:
	//Covariances for varaince decomposition:
	MatrixXdVec K_terms;

	//list of terms for each covariance
	ACovarVec covar_terms;
	//noise covariance
	PCovarianceFunction covar_noise;

	//genotype identity for noise model
	MatrixXd Kgeno;

	//phenotype Matrix
	MatrixXd pheno;
	//trait indicator
	MatrixXd trait;
	//fixed effects
	MatrixXd fixed;

	muint_t numtraits;

	//gp object for fitting
	PGPbase gp;
	PSumCF covar;

	void initGP();

public:
	CMultiTraitVQTL(muint_t numtraits);
	virtual ~CMultiTraitVQTL();

	//train
	void train();


	//getters and setters
	void agetK(MatrixXd* out,muint_t i) const;
	void agetKgeno(MatrixXd* out) const;
	void agetFixed(MatrixXd* out) const;
	void agetTrait(MatrixXd* out) const;
	PCovarianceFunction getCovar_term(muint_t i){return covar_terms[i];}
	PCovarianceFunction getCovar_noise(){return covar_noise;}
	PCovarianceFunction getCovar(){return this->covar;}
	PGPbase getGP() {return this->gp;}


	void setK(const MatrixXd& K,muint_t i);
	void addK(const MatrixXd& K);
	void setKgeno(const MatrixXd& Kgeno);
	void setFixed(const MatrixXd& fixed);
	void setTrait(const MatrixXd& trait);

	/*
	//setters and getters
	//getters:
    void agetPheno(MatrixXd *out) const;
    void agetSnps(MatrixXd *out) const;
    void agetCovs(MatrixXd *out) const;
    void setCovs(const MatrixXd & covs);
    void setPheno(const MatrixXd & pheno) throw(CGPMixException);
    void setSNPs(const MatrixXd & snps) throw(CGPMixException);
    void setPosition(const VectorXi& position);
    void setChrom(const VectorXi& chrom);
	*/
};
typedef sptr<CMultiTraitVQTL> PMultiTraitVQTL;


} /* namespace limix */
#endif /* CMULTITRAITVQTL_H */
