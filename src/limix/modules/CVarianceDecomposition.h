// Copyright(c) 2014, The LIMIX developers (Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
// All rights reserved.
//
// LIMIX is provided under a 2-clause BSD license.
// See license.txt for the complete license.

#ifndef CVARIANCEDECOMPOSITION_H_
#define CVARIANCEDECOMPOSITION_H_

#include "limix/types.h"
#include "limix/covar/freeform.h"
#include "limix/covar/combinators.h"
#include "limix/mean/CLinearMean.h"
#include "limix/gp/gp_base.h"
#include "limix/gp/gp_opt.h"

namespace limix {

//typedef std::set<muint_t> muint_set;
//typedef std::set<mfloat_t> mfloat_set;
//typedef std::vector<PVarianceTerm> PVarianceTermVec;

/*
 * CVarainceTerm captures a specific term in the varaince model
 */
#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
//ignore C++ versions
//%ignore AVarianceTerm::getHpInit;
%rename(getK) AVarianceTerm::agetK;
%rename(getX) AVarianceTerm::agetX;
%rename(getScales) AVarianceTerm::agetScales;
#endif

class AVarianceTerm {
protected:
	//Covariance Functions
	PFixedCF Kcf;
	//Kinship matrix and SNP data
	bool Knull;
	MatrixXd K;
	MatrixXd X;
	//boold
	bool fitted, is_init;
public:
	AVarianceTerm();
	virtual ~AVarianceTerm();

	virtual std::string getName() = 0;
	virtual std::string getInfo() = 0;

	/*!
	 * set a filter with individuals
	 * This functionality is typically used to filter out non-observed individuals
	 */
	virtual void setSampleFilter(const MatrixXb& filter) throw (CGPMixException) = 0;

	virtual muint_t getNumberTraits()=0;
	virtual muint_t getNumberIndividuals() const throw(CGPMixException);

	//Kinship and SNP data
	virtual void setK(const MatrixXd& K) throw(CGPMixException);
	virtual void agetK(MatrixXd *out) const throw(CGPMixException);
	virtual PFixedCF getKcf() {return Kcf;};

	virtual PCovarianceFunction getTraitCovar() const throw(CGPMixException) =0;

	//Params Handling
	virtual void setScales(const VectorXd& scales) throw(CGPMixException) =0;
	virtual void agetScales(VectorXd* scales) const throw(CGPMixException) =0;
	virtual muint_t getNumberScales() const throw(CGPMixException) = 0;

	//initTerm and getCovariance
	virtual void initTerm() throw(CGPMixException) = 0;
	virtual PCovarianceFunction getCovariance() const throw(CGPMixException) =0;

};
typedef sptr<AVarianceTerm> PVarianceTerm;


/*
 * 	Single trait variance Term
 */

#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
#endif

class CSingleTraitTerm : public AVarianceTerm
{
protected:
public:
	CSingleTraitTerm();
	CSingleTraitTerm(const MatrixXd& K);
	virtual ~CSingleTraitTerm();

	virtual std::string getName() {return "CSingleTraitTerm";};
	virtual std::string getInfo() {return "POPPY";};

	virtual void setSampleFilter(const MatrixXb& filter) throw (CGPMixException);



	virtual muint_t getNumberTraits() {return (muint_t)1;};


	virtual PCovarianceFunction getTraitCovar() const throw(CGPMixException);

	// Params Handling
	virtual void setScales(const VectorXd& scales) throw(CGPMixException);
	virtual void agetScales(VectorXd* out) const throw(CGPMixException);
	virtual muint_t getNumberScales() const throw(CGPMixException);

	//initTerm and get covariance
	virtual void initTerm() throw(CGPMixException);
	virtual PCovarianceFunction getCovariance() const throw(CGPMixException);
};
typedef sptr<CSingleTraitTerm> PSingleTraitTerm;

/*
 * 	Single trait variance Term
 */

#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
#endif

class CMultiTraitTerm : public AVarianceTerm
{
protected:
	muint_t P;
	PCovarianceFunction traitCovariance;
	bool isNull;
	// TOTAL COVARIANCE
	PKroneckerCF covariance;
public:
	CMultiTraitTerm(muint_t P);
	CMultiTraitTerm(muint_t P, PCovarianceFunction traitCovar, const MatrixXd& K);
	virtual ~CMultiTraitTerm();

	virtual std::string getName() {return "CMultiTraitTerm";};
	virtual std::string getInfo() {return "POPPY";};

	/*!
	 * sets a filtr of observed individuals.
	 * Note, this filter breaks the Kronecker strucutre and is only applicable in a non-Kronecker settings
	 */
	virtual void setSampleFilter(const MatrixXb& filter) throw (CGPMixException);


	virtual muint_t getNumberTraits() {return (muint_t)this->P;};

	//MultiPhenoFramework
	void setTraitCovar(PCovarianceFunction traitCovar) throw(CGPMixException);
	virtual PCovarianceFunction getTraitCovar() const throw(CGPMixException);

	// Params Handling
	virtual void setScales(const VectorXd& scales) throw(CGPMixException);
	virtual void agetScales(VectorXd* out) const throw(CGPMixException);
	virtual muint_t getNumberScales() const throw(CGPMixException);

	//initTerm and get covariance
	virtual void initTerm() throw(CGPMixException);
	virtual PCovarianceFunction getCovariance() const throw(CGPMixException);
};
typedef sptr<CMultiTraitTerm> PMultiTraitTerm;


/*
 * 	Variance Decomposition
 */

typedef std::vector<PVarianceTerm> PVarianceTermVec;
typedef std::vector<MatrixXd> MatrixXdVec;

#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
%rename(getScales) CVarianceDecomposition::agetScales;
#endif

class CVarianceDecomposition
{
protected:
	//Dimensions
	muint_t N,P;
	//Fixed effects
	MatrixXdVec fixedEffs;
	MatrixXdVec designs;
	//terms
	PVarianceTermVec terms;
	//Gaussian Process
	PGPbase gp;
	PGPopt opt;
	//Total Covariance
	PSumCF covar;
	//pheno
	MatrixXd pheno;
	//missing values in pheno
	MatrixXb phenoNAN; //<<! indicator with possible missing values in pheno
	bool phenoNANany; //<<! any phenotype entry missing?
	//bool
	bool is_init;
	bool fast;
public:
	CVarianceDecomposition(const MatrixXd& pheno);
	~CVarianceDecomposition();

	//Delete All Fixed and Variance Terms
	virtual void clear();
	//Phenotype
	virtual void setPheno(const MatrixXd& pheno) throw(CGPMixException);
	virtual void getPheno(MatrixXd *out) const throw(CGPMixException);
	virtual muint_t getNumberTraits() const throw(CGPMixException) {return this->P;};
	virtual muint_t getNumberIndividuals() const throw(CGPMixException) {return this->N;};

	// Fixed Effect Terms
	virtual void addFixedEffTerm(const MatrixXd& design, const MatrixXd& fixed) throw(CGPMixException);
	virtual void addFixedEffTerm(const MatrixXd& F) throw(CGPMixException);
	virtual void getFixed(MatrixXd *out, const muint_t i) const throw(CGPMixException);
	virtual void getDesign(MatrixXd *out, const muint_t i) const throw(CGPMixException);
	virtual void clearFixedEffs();
	virtual muint_t getNumberFixedEffs() const;
	// access fixed

	// Variance Term
	virtual void addTerm(PVarianceTerm term) throw(CGPMixException);
	virtual void addTerm(const MatrixXd& K) throw(CGPMixException);	//add Single Trait Term
	virtual void addTerm(PCovarianceFunction traitCovar, const MatrixXd& K) throw(CGPMixException);	//add Multi Trait Term
	virtual PVarianceTerm getTerm(muint_t i) const throw(CGPMixException);
	virtual void clearTerms();
	virtual muint_t getNumberTerms() const;
	// access Variance Term Features
	virtual void setScales(const VectorXd& scales) const throw(CGPMixException);
	virtual void setScales(muint_t i,const VectorXd& scales) const throw(CGPMixException);
	virtual void agetScales(muint_t i, VectorXd* out) const throw(CGPMixException);
	virtual void agetScales(VectorXd* out) throw(CGPMixException);
	virtual muint_t getNumberScales() throw(CGPMixException);

	//get gp, Covar and mean
	virtual PGPbase getGP() {return this->gp;};
	virtual PSumCF getCovar() {return this->covar;};
	virtual PLinearMean getMean() {return static_pointer_cast<CLinearMean>(this->gp->getDataTerm());};
	//init and train GP
	virtual void initGPparams() throw(CGPMixException);
	virtual void initGP(bool fast=false) throw(CGPMixException);
	virtual void initGPbase() throw(CGPMixException);
	virtual void initGPkronSum() throw(CGPMixException);
	virtual bool trainGP() throw(CGPMixException);
	//access GP
	virtual void getFixedEffects(VectorXd* out) throw(CGPMixException);
	virtual mfloat_t getLML() throw(CGPMixException);
	virtual mfloat_t getLMLgrad() throw(CGPMixException);
	virtual mfloat_t getLMLgradGPbase() throw(CGPMixException);
	virtual mfloat_t getLMLgradGPkronSum() throw(CGPMixException);

	// estimate single trait heritability
	static void aestimateHeritability(VectorXd* out, const MatrixXd& Y, const MatrixXd& fixed, const MatrixXd& K);

};
typedef sptr<CVarianceDecomposition> PVarianceDecomposition;


} //end: namespace limix

#endif /* CVARIANCEDECOMPOSITION_H_ */
