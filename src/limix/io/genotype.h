/*
 * genotype.h
 *
 *  Created on: May 16, 2013
 *      Author: stegle
 */

#ifndef GENOTYPE_H_
#define GENOTYPE_H_

#include "limix/types.h"

#include <string>
#include <map>
#include <vector>
#include <iostream>


namespace limix {

/*
 * Abstract class representing data structures with sample IDs (genotype/phenotype)
 */
class ASampleData {
public:

	ASampleData();
	virtual ~ASampleData();

	virtual void agetIDs(VectorXs* out) const throw(CGPMixException) = 0;
	virtual void setIDs(const VectorXs& in) throw (CGPMixException) =0;

};


/*
 * Abstract class representing genotype handling
 */
class AGenotype  : public ASampleData {

public:
	AGenotype();
	virtual ~AGenotype();

	//get genotype data
	virtual void agetGenotype(MatrixXd* out) const throw(CGPMixException) = 0;
	virtual void agetPosition(VectorXi* out) const throw(CGPMixException) = 0;
	virtual void agetChromosome(VectorXs* out) const throw(CGPMixException) = 0;

	//set genotype data
	virtual void setGenotype(const MatrixXd& in)  throw(CGPMixException) = 0;
	virtual void setPosition(const VectorXi& in)  throw(CGPMixException) = 0;
	virtual void setChromosome(const VectorXs& in)  throw(CGPMixException) = 0;
};
typedef sptr<AGenotype> PGenotype;

/*
 * In-memory implementation of genotype handling
 */

class CMemGenotype : public AGenotype
{
protected:
	PMatrixXd geno;
	PVectorXi pos;
	PVectorXs chrom;
	PVectorXs ids;

	void initMatrices();

public:
	CMemGenotype();
	CMemGenotype(PGenotype geno);
	virtual ~CMemGenotype();

	//virtual functions AGenotype
	virtual void agetGenotype(MatrixXd* out) const throw(CGPMixException);
	virtual void agetPosition(VectorXi* out) const throw(CGPMixException);
	virtual void agetChromosome(VectorXs* out) const throw(CGPMixException);

	virtual void setGenotype(const MatrixXd& in)  throw(CGPMixException);
	virtual void setPosition(const VectorXi& in)  throw(CGPMixException);
	virtual void setChromosome(const VectorXs& in)  throw(CGPMixException);

	//virtual functions ASampleData
	virtual void agetIDs(VectorXs* out) const throw(CGPMixException);
	virtual void setIDs(const VectorXs& in) throw (CGPMixException);
};
typedef sptr<CMemGenotype> PMemGenotype;


/* Common class to read from text files.
 * supports .gem, .vcf and .plink
 */
class CTextfileGenotype : public AGenotype
{
public:
	CTextfileGenotype();
	virtual ~CTextfileGenotype();
};
typedef sptr<CTextfileGenotype> PTextfileGenotype;



} //end: namespace limix

#endif /* GENOTYPE_H_ */
