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
#include "dataframe.h"
#include <string>
#include <iostream>
#include <fstream>


namespace limix {

class AGenotypeFilter
{
protected:
	std::string filter_chrom;
	uint64_t filter_start,filter_stop;
public:
	AGenotypeFilter()
	{
		filter_start = -1;
		filter_stop = -1;
		filter_chrom = "";
	};
	virtual ~AGenotypeFilter()
	{};
	virtual void setFilter(std::string chrom, uint64_t start,uint64_t stop)
	{
		this->filter_chrom = chrom;
		this->filter_start = start;
		this->filter_stop = stop;
	}
};

/*
 * Abstract class representing genotype handling
 * This is a DataFrame with an internal MatriXd object
 * rows: individuals
 * cols: genotypes
 */
class AGenotype  : public AGenotypeFilter,public CMemDataFrame<MatrixXd>
{
public:
	AGenotype()
	{};
	AGenotype(const AGenotype& copy) : CMemDataFrame<MatrixXd>(copy)
	{};
	virtual ~AGenotype()
	{};

	//add specific header functions for genotype and position
	virtual void agetPosition(VectorXi* out) const throw(CGPMixException) = 0;
	virtual PVectorXi getPosition() const throw(CGPMixException) = 0;

	virtual void agetChromosome(VectorXs* out) const throw(CGPMixException) = 0;
	virtual PVectorXs getChromosome() const throw(CGPMixException) = 0;
};

typedef sptr<AGenotype> PGenotype;

class ARWGenotype  : public AGenotypeFilter, public CMemRWDataFrame<MatrixXd>
{
public:
	ARWGenotype()
	{};
	ARWGenotype(const AGenotype& copy) : CMemRWDataFrame<MatrixXd>(copy)
	{};
	virtual ~ARWGenotype()
	{};

	//add specific header functions for genotype and position
	virtual void agetPosition(VectorXi* out) const throw(CGPMixException) = 0;
	virtual PVectorXi getPosition() const throw(CGPMixException) = 0;

	virtual void agetChromosome(VectorXs* out) const throw(CGPMixException) = 0;
	virtual PVectorXs getChromosome() const throw(CGPMixException) = 0;


	virtual void setPosition(const VectorXi& in) throw(CGPMixException) = 0;
	virtual void setPosition(PVectorXi in) throw(CGPMixException) =0;

	virtual void setChromosome(const VectorXs& in) throw(CGPMixException) = 0;
	virtual void setChromosome(PVectorXs in) throw(CGPMixException) =0;
};
typedef sptr<ARWGenotype> PRWGenotype;


/*
 * In-memory implementation of genotype handling
 */

//TODO: take filter into account for get operators.
class CMemGenotype : public AGenotype
{
protected:
	PVectorXi pos;
	PVectorXs chrom;
	void initMatrices();

public:
	CMemGenotype();
	CMemGenotype(PMatrixXd geno,PVectorXs chrom,PVectorXi pos,PVectorXs IDs);
	CMemGenotype(const AGenotype& copy);
	virtual ~CMemGenotype();

	//override Filter operations
	virtual void setFilter(std::string chrom, uint64_t start,uint64_t stop)
	{
		AGenotypeFilter::setFilter(chrom,start,stop);
	}

	//virtual functions AGenotype
	//R access
	virtual void agetPosition(VectorXi* out) const throw(CGPMixException);
	virtual PVectorXi getPosition() const throw(CGPMixException);

	virtual void agetChromosome(VectorXs* out) const throw(CGPMixException);
	virtual PVectorXs getChromosome() const throw(CGPMixException);


	/*
	//W access
	virtual void setPosition(const VectorXi& in) throw(CGPMixException);
	virtual void setPosition(PVectorXi in) throw(CGPMixException);

	virtual void setChromosome(const VectorXs& in) throw(CGPMixException);
	virtual void setChromosome(PVectorXs in) throw(CGPMixException);
	*/
};


typedef sptr<CMemGenotype> PMemGenotype;



/* Common class to read from text files.
 * supports .gem, .vcf and .plink
 */
class CTextfileGenotype : public AGenotype
{
protected:
	std::string filename;
public:
	CTextfileGenotype(std::string& filename);
	virtual ~CTextfileGenotype();
};
typedef sptr<CTextfileGenotype> PTextfileGenotype;


} //end: namespace limix

#endif /* GENOTYPE_H_ */
