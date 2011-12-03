/*
 * ALmm.h
 *
 *  Created on: Nov 27, 2011
 *      Author: stegle
 */

#ifndef ALMM_H_
#define ALMM_H_

#include "gpmix/types.h"


namespace gpmix {

//Abstract base class for LMM models*/
class ALMM {
protected:
	//Data and sample information
	MatrixXd snps;
	MatrixXd pheno;
	MatrixXd covs;
	MatrixXd Usnps;
	MatrixXd Upheno;
	MatrixXd Ucovs;
	//number of samples, snps and pheno
	muint_t num_samples,num_pheno,num_snps,num_covs;
	//common settings
	muint_t num_intervalsAlt,num_intervals0;
	mfloat_t ldeltamin0,ldeltamax0;
	mfloat_t ldeltaminAlt,ldeltamaxAlt;
	//results:
	MatrixXd pv;
	//state of decompostion cache:
	bool UK_cached;
	bool Usnps_cached;
	bool Upheno_cached;
	bool Ucovs_cached;


public:
	ALMM();
	virtual ~ALMM();

	//setter fors data:
	mfloat_t getLdeltamin0() const;
	muint_t getNumIntervalsAlt() const;
	muint_t getNumSamples() const;
	mfloat_t getLdeltaminAlt() const;
	void setLdeltaminAlt(mfloat_t ldeltaminAlt);
	void setLdeltamin0(mfloat_t ldeltamin0);
	void setNumIntervalsAlt(muint_t num_intervalsAlt);

	//getters:
	void getPheno(MatrixXd *out) const;
	void getPv(MatrixXd *out) const;
	void getSnps(MatrixXd *out) const;
	void getCovs(MatrixXd* out) const;
	void setCovs(const MatrixXd& covs);
	void setPheno(const MatrixXd& pheno);
	void setSNPs(const MatrixXd& snps);

	//virtual function
	virtual void process() =0;
	virtual void updateDecomposition() =0;


#ifndef SWIG
	//covenience versions:
	MatrixXd getPheno() const;
	MatrixXd getPv() const;
	MatrixXd getSnps() const;
	MatrixXd getCovs() const;
#endif

};


//Standard mixed liner model
class CLMM : public ALMM
{
protected:
	MatrixXd K;
	MatrixXd U;
	VectorXd S;
public:
	CLMM();
	virtual ~CLMM();

	//function to add testing kernel

	//processing;
	virtual void process();
	virtual void updateDecomposition();

	void getK(MatrixXd* out) const;
	void setK(const MatrixXd& K);
	void setK(const MatrixXd& K,const MatrixXd& U, const VectorXd& S);


#ifndef SWIG
	MatrixXd getK() const;
#endif
};

/*Simple Kronecker model.
 * Kroneckers are merely used to efficiently rotate the phenotypes and genotypes
 */

class CSimpleKroneckerLMM: public ALMM
{
protected:
	MatrixXd C;
	MatrixXd R;
	MatrixXd U_R;
	MatrixXd U_C;
	VectorXd S_R;
	VectorXd S_C;
public:
	CSimpleKroneckerLMM();
	virtual ~CSimpleKroneckerLMM();
	//processing;
	virtual void process();
	virtual void updateDecomposition();

	void setK_C(const MatrixXd& C);
	void setK_C(const MatrixXd& C,const MatrixXd& U_C, const VectorXd& S_C);

	void setK_R(const MatrixXd& R);
	void setK_R(const MatrixXd& R,const MatrixXd& U_R, const VectorXd& S_R);

	void getK_R(MatrixXd* out) const;
	void getK_C(MatrixXd* out) const;
#ifndef SWIG
	MatrixXd getK_R() const;
	MatrixXd getK_C() const;
#endif


};

/*
Efficient mixed liner model
However, there are limitations as to which hypothesis can be tested
TODO: add documentation here
*/
class CKroneckerLMM : public ALMM
{
	//TODO: check what is a symmetric matrix type!
protected:
	MatrixXd C;
	MatrixXd R;
	MatrixXd U_R;
	MatrixXd U_C;
	VectorXd S_R;
	VectorXd S_C;
	MatrixXd WkronDiag0;
	MatrixXd WkronBlock0;
	MatrixXd WkronDiag;
	MatrixXd WkronBlock;

public:
	CKroneckerLMM();
	virtual ~CKroneckerLMM();

	//processing;
	virtual void process();
	virtual void updateDecomposition();

	void setK_C(const MatrixXd& C);
	void setK_C(const MatrixXd& C,const MatrixXd& U_C, const VectorXd& S_C);

	void setK_R(const MatrixXd& R);
	void setK_R(const MatrixXd& R,const MatrixXd& U_R, const VectorXd& S_R);

	void getK_R(MatrixXd* out) const;
	void getK_C(MatrixXd* out) const;
	void setKronStructure(const MatrixXd& WkronDiag0, const MatrixXd& WkronBlock0, const MatrixXd& WkronDiag, const MatrixXd& WkronBlock);
	static mfloat_t nLLeval(MatrixXd* F_tests, mfloat_t ldelta, const MatrixXd& WkronDiag, const MatrixXd& WkronBlock, const MatrixXd& UX, const MatrixXd& UYU, const VectorXd& S_C, const VectorXd& S_R);
	static mfloat_t optdelta(const MatrixXd& UX, const MatrixXd& UYU, const VectorXd& S_C, const VectorXd& S_R, const muint_t numintervals, const mfloat_t ldeltamin, const mfloat_t ldeltamax, const MatrixXd& WkronDiag, const MatrixXd& WkronBlock);
#ifndef SWIG
	MatrixXd getK_R() const;
	MatrixXd getK_C() const;
#endif
};


/* Standalone helper functions.
 * These will also be wrapped in python
 */


void train_associations_SingleSNP(MatrixXd* PV, MatrixXd* LL, MatrixXd* ldelta,
		const MatrixXd& X, const MatrixXd& Y, const MatrixXd& U,
		const MatrixXd& S, const MatrixXd& C, int numintervals,
		double ldeltamin, double ldeltamax);
double optdelta(const MatrixXd& UY,const MatrixXd& UX,const MatrixXd& S,int numintervals,double ldeltamin,double ldeltamax);
void optdeltaAllY(MatrixXd* out, const MatrixXd& UY, const MatrixXd& UX, const MatrixXd& S, const MatrixXd& ldeltagrid);
double nLLeval(MatrixXd* F_tests, double ldelta,const MatrixXd& UY,const MatrixXd& UX,const MatrixXd& S);
void nLLevalAllY(MatrixXd* out, double ldelta,const MatrixXd& UY,const MatrixXd& UX,const MatrixXd& S);

} /* namespace gpmix */
#endif /* ALMM_H_ */
