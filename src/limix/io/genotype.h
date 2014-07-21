// Copyright(c) 2014, The LIMIX developers(Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
//
//Licensed under the Apache License, Version 2.0 (the "License");
//you may not use this file except in compliance with the License.
//You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//Unless required by applicable law or agreed to in writing, software
//distributed under the License is distributed on an "AS IS" BASIS,
//WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//See the License for the specific language governing permissions and
//limitations under the License.

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
	/*!
	read num_snp from the current stream and return as GenotypeBlock
	*/
	virtual PGenotypeBlock read(mint_t num_snps=-1)  = 0;
};




/*CGenotypeBlock
 * In-memory representation of a genotype structure, which is also a container
 */
class CGenotypeBlock : public CRMemDataFrame<MatrixXd> //,public AGenotypeContainer
{
	friend class AGenotypeContainer;
	friend class CTextfileGenotypeContainer;
protected:
	PVectorXi pos;
	/*!
	resize the internal storage (rows: samples, columns: SNPs)
	*/
	virtual void resizeMatrices(muint_t num_rows, muint_t num_columns);
	muint_t i_snp_read;

	/*!
	initialize the GenotypeBlock structure fro
	*/
	void init(const stringVec& row_header_names,const stringVec& col_haeder_names,muint_t rows,muint_t cols);

public:
	CGenotypeBlock();
	CGenotypeBlock(const stringVec& row_header_names, const stringVec& col_header_names,muint_t rows, muint_t cols);
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
	virtual void agetPosition(VectorXi* out) const ;
	virtual PVectorXi getPosition() const ;

	//virtual function: AGenotypeContainer
	virtual PGenotypeBlock read(mint_t num_snps=-1) ;

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
	std::ifstream* bin_in_stream;
	std::istream* in_stream;
	//filename
	std::string in_filename;
	bool is_open;
	//open files
	void openFile() ;
	void read_header_GEN();
	PGenotypeBlock read_GEN(mint_t num_snps) ;
	PGenotypeBlock read_BED(mint_t num_snps) ;

	std::istream& getStream()
	{
		return *in_stream;
	}

public:
	CTextfileGenotypeContainer(const std::string& filename);
	virtual ~CTextfileGenotypeContainer();

	//virtual functions
	PGenotypeBlock read(mint_t num_snps=-1) ;
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
	PGenotypeBlock read(mint_t num_snps=-1) ;
};
typedef sptr<CMemGenotypeContainer> PMemGenotypeContainer;


} //end: namespace limix

#endif /* GENOTYPE_H_ */
