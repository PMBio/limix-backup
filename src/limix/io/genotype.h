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


//#include "gzstream.h"

/*
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/device/file_descriptor.hpp>
*/

//namespace io = boost::iostreams;


namespace limix {


//typedef std::map<std::string,std::vector<std::string>> stringMap;


class AGenotypeContainer;
class CTextfileGenotypeContainer;
class CGenotypeBlock;
typedef sptr<CGenotypeBlock> PGenotypeBlock;
typedef sptr<AGenotypeContainer> PGenotypeContainer;


/*!
 * Abstract container of genotype data
 */
class AGenotypeContainer
{
protected:
	std::string filterSNPchrom;
	muint_t filterSNPstart,filterSNPstop;

	//test whether a particular snp obays the filter
	bool inline check_SNP(std::string snp_chrom,muint_t snp_pos)
	{
		if (this->filterSNPchrom=="NAN")
			return true;
		if(snp_chrom!=filterSNPchrom)
			return false;
		return (snp_pos>=filterSNPstart) && (snp_pos<filterSNPstop);
	}

public:
	AGenotypeContainer()
	{
		filterSNPstart = -1;
		filterSNPstop = -1;
		filterSNPchrom = "NAN";
	};

	virtual ~AGenotypeContainer()
	{};
	//set filter of genotype class
	virtual void setSNPFilter(std::string chrom, muint_t start,muint_t stop)
	{
		this->filterSNPchrom = chrom;
		this->filterSNPstart = start;
		this->filterSNPstop = stop;
	}

	//virtual functions
	virtual PGenotypeBlock read(mint_t num_snps=-1) throw (CGPMixException) = 0;
};




/*CGenotypeBlock
 * In-memory representation of a genotype structure, which is also a container
 */
#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
%ignore CGenotypeBlock::getPosition;
%ignore CGenotypeBlock::getMatrix;
%rename(getMatrix) CGenotypeBlock::agetMatrix;
%rename(getPosition) CGenotypeBlock::agetPosition;
#endif
class CGenotypeBlock : public CRMemDataFrame<MatrixXd> //,public AGenotypeContainer
{
	friend class AGenotypeContainer;
	friend class CTextfileGenotypeContainer;
protected:
	PVectorXi pos;
	virtual void resizeMatrices(muint_t num_rows, muint_t num_columns);
	muint_t i_snp_read;

	void init(const stringVec& row_header_names,const stringVec& col_haeder_names);

public:
	CGenotypeBlock();
	CGenotypeBlock(const stringVec& row_header_names, const stringVec& col_header_names);
	CGenotypeBlock(const CGenotypeBlock& copy);
	CGenotypeBlock(PMatrixXd geno, PVectorXi pos,PHeaderMap row_header,PHeaderMap col_header);
	virtual ~CGenotypeBlock();

	//information about dimensions
	muint_t numSample(){
		return this->M->rows();
	}
	muint_t numSNPs()
	{
		return this->M->cols();
	}

	//virtual functions: CMemDataFrame
	virtual void agetPosition(VectorXi* out) const throw(CGPMixException);
	virtual PVectorXi getPosition() const throw(CGPMixException);

	//virtual function: AGenotypeContainer
	virtual PGenotypeBlock read(mint_t num_snps=-1) throw (CGPMixException);

};




/* Common class to read from text files.
 * supports .gem, .vcf and .plink
 */
class CTextfileGenotypeContainer : public AGenotypeContainer
{
protected:
	//file format
	enum { GEN, VCF, BED } file_format;
	//buffersize for unlimitted reading
	muint_t buffer_size;
	//stream for incoming files using the boost library
	//boost::iostreams::filtering_istream in_stream;
	//std::istream in_stream;
	std::istream* in_stream;
	//filename
	std::string in_filename;
	bool is_open;
	//open files
	void openFile() throw (CGPMixException);
	void read_header_GEN();
	PGenotypeBlock read_GEN(muint_t num_snps) throw (CGPMixException);
	PGenotypeBlock read_BED(muint_t num_snps) throw (CGPMixException);

	std::istream& getStream()
	{
		return *in_stream;
	}

public:
	CTextfileGenotypeContainer(const std::string& filename);
	virtual ~CTextfileGenotypeContainer();

	//virtual functions
	PGenotypeBlock read(mint_t num_snps=-1) throw (CGPMixException);
};
typedef sptr<CTextfileGenotypeContainer> PTextfileGenotypeContainer;



/* In Memory container
 *
 */

class CMemGenotypeContainer : public AGenotypeContainer
{
protected:
	PGenotypeBlock block;
	muint_t reading_row;
public:
	CMemGenotypeContainer(PGenotypeBlock block);
	virtual ~CMemGenotypeContainer();

	//virtual functions
	PGenotypeBlock read(mint_t num_snps=-1) throw (CGPMixException);
};
typedef sptr<CMemGenotypeContainer> PMemGenotypeContainer;


} //end: namespace limix

#endif /* GENOTYPE_H_ */
