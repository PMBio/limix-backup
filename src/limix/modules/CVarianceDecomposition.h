/*
 * CVarianceDecomposition.h
 *
 *  Created on: Nov 27, 2012
 *      Author: stegle
 */

#ifndef CVARIANCEDECOMPOSITION_H_
#define CVARIANCEDECOMPOSITION_H_

#include "limix/types.h"
#include "limix/gp/gp_base.h"
#include "limix/covar/linear.h"
#include "limix/covar/combinators.h"
#include "limix/covar/fixed.h"
#include "limix/likelihood/likelihood.h"
#include "limix/gp/gp_opt.h"
#include "limix/covar/freeform.h"
//set for unique trait types
#include <set>


namespace limix {


//rename argout operators for swig interface
#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
//ignore C++ versions
%ignore AVarianceTerm::getHp0;
%ignore AVarianceTerm::getHpMask;
%ignore AVarianceTerm::getVarianceComponents;
%ignore AVarianceTerm::getK;
%ignore CVarianceDecomposition::getTrait;
%ignore CVarianceDecomposition::getPheno;
%ignore CVarianceDecomposition::getFixed;
%ginore CCategorialTraitVarianceTerm::getVinitMarginal;


%rename(getHp0) AVarianceTerm::agetHp0;
%rename(getHpMask) AVarianceTerm::agetHpMask;
%rename(getFittedVariance) AVarianceTerm::agetFittedVariance;
%rename(getK) AVarianceTerm::agetK;
%rename(getVinitMarginal) CCategorialTraitVarianceTerm::agetVinitMarginal;
%rename(getTrait) CVarianceDecomposition::agetTrait;
%rename(getPheno) CVarianceDecomposition::agetPheno;
%rename(getFixed) CVarianceDecomposition::agetFixed;
#endif


class AVarianceTerm;
class CCategorialTraitVarianceTerm;
class CSingleTraitVarianceTerm;
class CVarianceDecomposition;
typedef sptr<AVarianceTerm> PVarianceTerm;
typedef sptr<CCategorialTraitVarianceTerm> PCategorialTraitVarianceTerm;
typedef sptr<CSingleTraitVarianceTerm> PSingleTraitVarianceTerm;
typedef sptr<CVarianceDecomposition> PVarianceDecomposition;

typedef std::set<muint_t> muint_set;
typedef std::set<mfloat_t> mfloat_set;
typedef std::vector<PVarianceTerm> PVarianceTermVec;


/*
 * CVarainceTerm captures a specific term in the varaince model
 */
class AVarianceTerm {
protected:
	//data needed to specify a term
	MatrixXd K;
	//fixed CF component
	PFixedCF   Kcovariance;
	//init parameters
	MatrixXd hp0;
	//parameter mask
	MatrixXd hp_mask;
	//overall covariance member
	PCovarianceFunction covariance;
	bool isInitialized;
	mfloat_t Vinit;

public:
	virtual PCovarianceFunction getCovariance() const
	{
		return covariance;
	}
	virtual mfloat_t getVariance() const;
	virtual MatrixXd getVarianceComponents() const
	{
		MatrixXd out;
		agetFittedVariance(&out);
		return out;
	}
	virtual void initCovariance() throw (CGPMixException) = 0;

	//abstract functions
	virtual void agetFittedVariance(MatrixXd* out) const =0;

	AVarianceTerm();
	AVarianceTerm(const MatrixXd& K,mfloat_t Vinit);
	virtual ~AVarianceTerm();

	void agetK(MatrixXd* out) const;
	void setK(const MatrixXd& K);
	MatrixXd getK() const
	{
		MatrixXd out;
		this->agetK(&out);
		return out;
	}

	const MatrixXd& getHpMask() const {
		return hp_mask;
	}
	const void agetHpMask(MatrixXd* out) const {
		(*out) = hp_mask;
	}


	void setHpMask(const MatrixXd& hpMask) {
		hp_mask = hpMask;
	}

	const MatrixXd& getHp0() const {
		return hp0;
	}
	const void agetHp0(MatrixXd* out) const
	{
		(*out) = hp0;
	}

	void setHp0(const MatrixXd& hp0) {
		this->hp0 = hp0;
	}

	bool isIsInitialized() const {
		return isInitialized;
	}

	mfloat_t getVinit() const {
		return Vinit;
	}

	void setVinit(mfloat_t vinit) {
		Vinit = vinit;
	}
};


/*
 * Single trait covariance term
 */
class CSingleTraitVarianceTerm : public AVarianceTerm {
protected:
public:
	CSingleTraitVarianceTerm();
	CSingleTraitVarianceTerm(const MatrixXd& K,mfloat_t Vinit);
	virtual ~CSingleTraitVarianceTerm();

	virtual void initCovariance() throw (CGPMixException);
	virtual void agetFittedVariance(MatrixXd* out) const;

};


/*
 * Categorial multi trait variance term
 */
class CCategorialTraitVarianceTerm : public AVarianceTerm
{
protected:
	bool modelCrossCovariance;
	VectorXd trait;
	VectorXd utrait;
	//number of traits
	muint_t numtraits;
	//freeform covariance
	PFreeFormCF trait_covariance;
	VectorXd VinitMarginal;

public:
	CCategorialTraitVarianceTerm();
	CCategorialTraitVarianceTerm(const MatrixXd& K,const MatrixXd& trait,mfloat_t Vinit=NAN, bool fitCrossCovariance=true);
	virtual ~CCategorialTraitVarianceTerm();


	//init covariance function
	virtual void initCovariance() throw (CGPMixException);


	//getters and setters
	const VectorXd& getTrait() const {
		return trait;
	}
	void agetTrait(VectorXd* out) const;
	void setTrait(const VectorXd& trait);

	muint_t getNumtraits() const {
		return numtraits;
	}

	virtual void agetFittedVariance(MatrixXd* out) const;

	bool isModelCrossCovariance() const {
		return modelCrossCovariance;
	}

	void setModelCrossCovariance(bool modelCrossCovariance) {
		this->modelCrossCovariance = modelCrossCovariance;
	}

	const VectorXd& getVinitMarginal() const {
		return VinitMarginal;
	}

	void agetVinitMarginal(MatrixXd* out)
	{
		(*out) = VinitMarginal;
	}

	void setVinitMarginal(const VectorXd& vinitMarginal) {
		VinitMarginal = vinitMarginal;
	}
};

class CVarianceDecomposition
{
protected:
	//variance terms to fit:
	PVarianceTermVec terms;
	//phenotype Matrix
	MatrixXd pheno;
	//trait indicator
	MatrixXd trait;

	//fixed effects:
	MatrixXd fixed;
	//gp object, covaraince and optimization
	PGPbase gp;
	PGPopt opt;
	PSumCF covar;
	MatrixXd hp_covar0,hp_covar_mask;


public:
	static const muint_t singletrait = 0;
	static const muint_t categorial = 1;
	static const muint_t continuous = 2;

	CVarianceDecomposition();
	CVarianceDecomposition(const MatrixXd& pheno,const MatrixXd& trait);
	virtual ~CVarianceDecomposition();

	void initGP() throw (CGPMixException);
	bool train() throw(CGPMixException);

	void addTerm(PVarianceTerm term);
	void addTerm(const MatrixXd& K,muint_t type,mfloat_t Vinit=NAN, bool fitCrossCovariance=true);
	PVarianceTerm getTerm(muint_t i) const;

	PGPbase getGP(){return gp;};
	PGPopt  getOpt(){return opt;};


	static void aestimateHeritability(VectorXd* out,const MatrixXd& Y,const MatrixXd& fixed, const MatrixXd& K);

	/* getters and setters*/
	const MatrixXd& getFixed() const {
		return fixed;
	}

	void agetFixed(MatrixXd* out) const {
		(*out) = fixed;
	}

	void setFixed(const MatrixXd& fixed) {
		this->fixed = fixed;
	}

	const MatrixXd& getPheno() const {
		return pheno;
	}

	void agetPheno(MatrixXd* out) const
	{
		(*out) = pheno;
	}


	void setPheno(const MatrixXd& pheno) {
		this->pheno = pheno;
	}

	const MatrixXd& getTrait() const {
		return trait;
	}

	void agetTrait(MatrixXd* out) const
	{
		(*out) = trait;
	}

	void setTrait(const MatrixXd& trait) {
		this->trait = trait;
	}
};


} //end: namespace limix

#endif /* CVARIANCEDECOMPOSITION_H_ */
