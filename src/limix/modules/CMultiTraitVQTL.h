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
#include "limix/gp/gp_opt.h"
//set for unique trait types
#include <set>




namespace limix {

//rename argout operators for swig interface
#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
//ignore C++ versions
%ignore CMultiTraitVQTL::getK;
%ignore CMultiTraitVQTL::getKgeno;
%ignore CMultiTraitVQTL::getFixed;
%ignore CMultiTraitVQTL::getTrait;
%ignore CMultiTraitVQTL::getPheno;
%ignore CMultiTraitVQTL::getVarianceComponent_term;
%ignore CMultiTraitVQTL::getVarianceComponent_noise;

%rename(getK) CMultiTraitVQTL::agetK;
%rename(getKgeno) CMultiTraitVQTL::agetKgeno;
%rename(getFixed) CMultiTraitVQTL::agetFixed;
%rename(getTrait) CMultiTraitVQTL::agetTrait;
%rename(getPheno) CMultiTraitVQTL::agetPheno;
%rename(getVarianceComponent_term) CMultiTraitVQTL::agetVarianceComponent_term;
%rename(getVarianceComponent_noise) CMultiTraitVQTL::agetVarianceComponent_noise;
#endif



//vector on SNP sets

typedef std::vector<MatrixXd> MatrixXdVec;
typedef std::vector<PProductCF> PProductCFVec;
typedef std::set<mfloat_t> mfloat_set;

//typedef std::vector<bool> Trait

class CMultiTraitVQTL {
protected:
	//Covariances for varaince decomposition:
	MatrixXdVec K_terms;


	//list of terms for each covariance
	PProductCFVec covar_terms;
	//noise covariance
	PProductCF covar_noise;

	//genotype identity for noise model
	MatrixXd Kgeno;

	//phenotype Matrix
	MatrixXd pheno;
	//trait indicator
	MatrixXd trait;
	//unique indiciates in trait
	MatrixXd utrait;

	//fixed effects
	MatrixXd fixed;

	muint_t numtraits;

	//gp object for fitting
	PGPbase gp;
	PGPopt opt;
	PSumCF covar;

	//check consitency of the parameters and datasets supplied
	void checkConsistency() throw(CGPMixException);
	//initialize GP instance
	void initGP() throw(CGPMixException);
	//calculate variance components form FreeForm Hyperparams und this->utrait
	void agetFreeFormVariance(MatrixXd* out,CovarParams params);

public:
	CMultiTraitVQTL();
	virtual ~CMultiTraitVQTL();

	//train
	bool train() throw(CGPMixException);


	//getters and setters
	void agetK(MatrixXd* out,muint_t i) const;
	void agetKgeno(MatrixXd* out) const;
	void agetFixed(MatrixXd* out) const;
	void agetTrait(MatrixXd* out) const;
	void agetPheno(MatrixXd* out) const;
	//get covariance functions
	PCovarianceFunction getCovar_term(muint_t i);
	PCovarianceFunction getCovar_noise(bool traitCovar=false);
	//convenience functions to directly access inferred variance components
	void agetVarianceComponent_term(MatrixXd* out,muint_t i);
	void agetVarianceComponent_noise(MatrixXd* out);
	MatrixXd getVarianceComponent_noise()
	{
		MatrixXd rv;
		agetVarianceComponent_noise(&rv);
		return rv;
	}
	MatrixXd getVarianceComponent_term(muint_t i)
	{
		MatrixXd rv;
		agetVarianceComponent_term(&rv,i);
		return rv;
	}


	PCovarianceFunction getCovar(){return this->covar;}
	PGPbase getGP() {return this->gp;}

	void setK(const MatrixXd& K,muint_t i,bool rescale=false);
	void addK(const MatrixXd& K,bool rescale=false);
	void setKgeno(const MatrixXd& Kgeno,bool rescale=false);
	void setFixed(const MatrixXd& fixed);
	void setTrait(const MatrixXd& trait);
	void setPheno(const MatrixXd& pheno);

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
