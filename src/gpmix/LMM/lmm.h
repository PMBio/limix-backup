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
class ALmm {
protected:
	//Data and sample information
	MatrixXd snps;
	MatrixXd pheno;
	//number of samples, snps and pheno
	muint_t num_samples,num_pheno,num_snps;
	//common settings
	muint_t num_intervalsAlt,num_intervals0;
	mfloat_t ldeltamin0,ldeltamax0;
	mfloat_t ldeltamin,ldeltamax;
	//results:
	MatrixXd pv;

public:
	ALmm();
	virtual ~ALmm();

	//setter fors data:
	void setData(MatrixXd& snps,MatrixXd& pheno);
    mfloat_t getLdeltamin() const;
    mfloat_t getLdeltamin0() const;
    muint_t getNumIntervalsAlt() const;
    muint_t getNumSamples() const;
    void setLdeltamin(mfloat_t ldeltamin);
    void setLdeltamin0(mfloat_t ldeltamin0);
    void setNumIntervalsAlt(muint_t num_intervalsAlt);
    void setNumSamples(muint_t num_samples);

    //getters:
    void getPheno(MatrixXd* out) const;
    void getPv(MatrixXd* out) const;
    void getSnps(MatrixXd * out) const;

    //covenience versions:
    MatrixXd getPheno() const;
    MatrixXd getPv() const;
    MatrixXd getSnps() const;

    //virtual function
    virtual void process() =0;

};


//Standard mixed liner model
class CLmm : public ALmm
{
protected:
	MatrixXd K;
public:
	CLmm();
	virtual ~CLmm();

	//function to add testing kernel

	//processing;
	virtual void process();

	void getK(MatrixXd* out) const;
    MatrixXd getK() const;
    void setK(MatrixXd K);
};

//Standard mixed liner model
class CKroneckerLMM : public CLmm
{
protected:
	MatrixXd Kcol;
public:
	CKroneckerLMM();
	virtual ~CKroneckerLMM();


	//processing;
	virtual void process();

};



} /* namespace gpmix */
#endif /* ALMM_H_ */
