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
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>
//#include <boost/iostreams/filter/bzip2.hpp>
#include <boost/iostreams/device/file_descriptor.hpp>


//namespace io = boost::iostreams;


namespace limix {

class AGenotypeFilter
{
protected:
	std::string filter_chrom;
	muint_t filter_start,filter_stop;
	bool inline check_SNP(std::string snp_chrom,muint_t snp_pos)
	{
		if (this->filter_chrom=="NAN")
			return true;
		if(snp_chrom!=filter_chrom)
			return false;
		return (snp_pos>=filter_start) && (snp_pos<filter_stop);
	}
public:
	AGenotypeFilter()
	{
		filter_start = -1;
		filter_stop = -1;
		filter_chrom = "NAN";
	};
	virtual ~AGenotypeFilter()
	{};
	virtual void setFilter(std::string chrom, muint_t start,muint_t stop)
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
	virtual void resizeMatrices(muint_t num_rows, muint_t num_columns);

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
class CTextfileGenotype : public CMemGenotype
{
protected:

	enum { GEN, VCF, BED } file_format;
	//stream for incoming files using the boost library
	boost::iostreams::filtering_istream in_stream;
	//filename
	std::string in_filename;

	//open files
	void openFile();

public:
	CTextfileGenotype(const std::string& filename);
	virtual ~CTextfileGenotype();

	void read(muint_t buffer_size=50000);
	void read_GEN(muint_t buffer_size);
};
typedef sptr<CTextfileGenotype> PTextfileGenotype;


} //end: namespace limix

#endif /* GENOTYPE_H_ */
