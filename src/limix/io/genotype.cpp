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


#include "genotype.h"

//#include "vcflib/Variant.h"
#include "split.h"
#include <string>
#include <vector>
#include <numeric>

#ifdef ZLIB
//#include "zipstream.hpp"
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/zlib.hpp>
#endif

//namespace io = boost::iostreams;

namespace limix {

void CGenotypeBlock::init(const stringVec& row_header_names,const stringVec& col_header_names,muint_t rows, muint_t cols)
{
	for(stringVec::const_iterator iter = row_header_names.begin(); iter!=row_header_names.end();iter++)
	{
		std::string name = (*iter);
		PArray1DXs tmp=PArray1DXs(new Array1DXs(rows));
		(*this->rowHeader)[name] = tmp;

	}
	for(stringVec::const_iterator iter = col_header_names.begin(); iter!=col_header_names.end();iter++)
	{
		std::string name = (*iter);
		PArray1DXs tmp=PArray1DXs(new Array1DXs(cols));
		(*this->colHeader)[name] = tmp;
	}

	this->pos = PVectorXi(new VectorXi());
	i_snp_read = 0;
}


CGenotypeBlock::CGenotypeBlock(const stringVec& row_header_names,const stringVec& col_haeder_names,muint_t rows,muint_t cols)
{
	init(row_header_names,col_haeder_names,rows,cols);
}

CGenotypeBlock::CGenotypeBlock() {
	stringVec row_header_names,col_haeder_names;
	init(row_header_names,col_haeder_names,0,0);
}

CGenotypeBlock::CGenotypeBlock(const CGenotypeBlock& copy) : CRMemDataFrame<MatrixXd>(copy)
{
	pos = PVectorXi(new VectorXi());
	*(this->pos) = *(copy.pos);
	i_snp_read = 0;
}

CGenotypeBlock::CGenotypeBlock(PMatrixXd geno, PVectorXi pos,PHeaderMap row_header,PHeaderMap col_header) {
	this->M = geno;
	this->pos = pos;
	this->rowHeader = row_header;
	this->colHeader = col_header;
	i_snp_read = 0;
}

CGenotypeBlock::~CGenotypeBlock()
{
}

void CGenotypeBlock::agetPosition(VectorXi* out) const 
{
	(*out) = (*pos);
}
void CGenotypeBlock::resizeMatrices(muint_t num_samples, muint_t num_snps)
{
	//resize base matrix type
	CRMemDataFrame<MatrixXd>::resizeMatrices(num_samples,num_snps);
	//resize SNP specific elements
	this->pos->conservativeResize(num_snps);
}

PGenotypeBlock CGenotypeBlock::read(mint_t num_snps) 
{
	//build sub matrices:
	PMatrixXd new_geno = PMatrixXd(new MatrixXd(this->M->block(0,i_snp_read,this->numSample(),i_snp_read+num_snps)));
	PVectorXi new_pos = PVectorXi(new VectorXi(this->pos->segment(i_snp_read,i_snp_read+num_snps)));
	PHeaderMap new_rowHeader = this->rowHeader->copy(i_snp_read,num_snps);
	PHeaderMap new_colHeader = PHeaderMap(new CHeaderMap(*(this->colHeader)));
	PGenotypeBlock RV = PGenotypeBlock(new CGenotypeBlock(new_geno,new_pos,new_rowHeader,new_colHeader));
	return RV;
}


PVectorXi CGenotypeBlock::getPosition() const 
{
	return pos;
}

/* Text File genotype class */
limix::CTextfileGenotypeContainer::CTextfileGenotypeContainer(const std::string& filename){
	this->in_filename = filename;
	buffer_size = 100000;
	is_open = false;
}

limix::CTextfileGenotypeContainer::~CTextfileGenotypeContainer() {
}

void CTextfileGenotypeContainer::openFile() 
{
	//take filename apart and check whether ending is .gzip
    std::vector<std::string> filenameParts = split(in_filename, ".");

    std::string ext;

    //is the file gzip ?
    if (filenameParts.back() == "gz")
    {
      #ifdef ZLIB
    	bin_in_stream = new ifstream(in_filename.c_str(),ios::in | ios::binary);
    	in_stream = new funzipper(*bin_in_stream);
    	ext = filenameParts.at(filenameParts.size()-2);
      #else
	throw CLimixException("not compiled with zlib");
      #endif
    }
    else
    {
    	in_stream = new std::ifstream(in_filename.c_str());
    	ext = filenameParts.at(filenameParts.size()-1);
    }


    /*
    if (filenameParts.back() == "gz")
    {
    	in_stream.push(boost::iostreams::basic_gzip_decompressor<>());
    	//extension is the next element
    	ext = filenameParts.at(filenameParts.size()-2);
    }
    else if (filenameParts.back() == "bz2")
    {
    	throw CLimixException("bz2 not supported");
    	//in_stream.push(boost::iostreams::basic_bzip2_compressor<>());
    	ext = filenameParts.at(filenameParts.size()-2);
    }
    else
    {
    	ext = filenameParts.at(filenameParts.size()-1);
    }
    //open file
	in_stream.push(io::file_descriptor_source(in_filename));
     */

	//remember extension to call appropriate reader
	if(ext=="gen")
	{
		this->file_format = GEN;
		read_header_GEN();
	}
	else if(ext=="vcf")
		this->file_format = VCF;
	else if(ext=="bed")
		this->file_format = BED;
	else
		throw CLimixException("unknown file format");

	is_open = true;

}


PGenotypeBlock CTextfileGenotypeContainer::read(mint_t num_snps) 
{
	//open file
	if (!is_open)
		openFile();
	//which file format?
	if(this->file_format==GEN)
		return read_GEN(num_snps);
	else if (this->file_format==BED)
		return read_BED(num_snps);
	else
		throw CLimixException("unsupported file format in read");
}

void CTextfileGenotypeContainer::read_header_GEN()
{

}

/*
 * read num_snps lines from .bed file
 */
PGenotypeBlock CTextfileGenotypeContainer::read_BED(mint_t num_snps) 
{
	PGenotypeBlock RV = PGenotypeBlock(new CGenotypeBlock());
	throw CLimixException("BED readder not implemented");
	return RV;
}


PGenotypeBlock CTextfileGenotypeContainer::read_GEN(mint_t num_snps) 
{

	//creat result Structure
	const char* Scol_header_names[] = {"snp_id","chrom"};
	const char* Srow_header_names[] = {"sample_id"};

	PstringVec col_header_names = PstringVec(new stringVec(Scol_header_names,Scol_header_names+sizeof(Scol_header_names)/sizeof(Scol_header_names[0])));
	PstringVec row_header_names = PstringVec(new stringVec(Srow_header_names,Srow_header_names+sizeof(Srow_header_names)/sizeof(Srow_header_names[0])));


	PGenotypeBlock RV = PGenotypeBlock(new CGenotypeBlock(*row_header_names,*col_header_names,0,0));

	//number of read snps, current buffer
	muint_t i_snp,buffer,num_samples;

	buffer        = 0;
	num_samples   = 0;
	i_snp         = 0;

	//temporary variables for reading
	std::string line;
	std::string chrom,snp_id,snp_a,snp_b;
	muint_t snp_pos;
	while(std::getline(getStream(),line))
	{
		if((i_snp>=(muint_t)num_snps) && (num_snps>-1))
				break;

		//std::cout << line;
		//parse line
		std::vector<std::string> fields = split(line, ' ');
		//parse
		//fields.
		//std::cout << fields.size() << "\n";
		//std::cout << fields[0] << "\n";
		//header
		chrom = fields[0];
		snp_id = fields[1];
		snp_pos = atoi(fields[2].c_str());
		snp_a   = fields[3];
		snp_b   = fields[4];
		//check whether SNP in filter
		if (!check_SNP(chrom,snp_pos))
			continue;

		//1. line to read?
		if (num_samples==0)
			//figure out sample size
			num_samples = (fields.size()-5) / 3;
		else if (num_samples!=((fields.size()-5)/3))
			throw CLimixException("Line while reading as inconsistent length");

		//2. need to extend buffer?
		if (i_snp>=buffer)
		{
			if (num_snps>0)
				buffer += num_snps;
			else
				buffer += buffer_size;
			RV->resizeMatrices(num_samples,buffer);
		}
		//3. store position and chromosome
		RV->colHeader->setStr("chrom",i_snp,chrom);
		RV->colHeader->setStr("snp_id",i_snp,snp_id);


		(*RV->pos)(i_snp) = snp_pos;

		//4. read every individual
		//loop over individuals
		muint_t i_field;
		mfloat_t bin_state,state_0,state_1,state_2;

		for (muint_t i_sample=0;i_sample<num_samples;++i_sample)
		{
			i_field = i_sample*3 + 5;
			state_0 = atof(fields[i_field].c_str());
			state_1 = atof(fields[i_field+1].c_str());
			state_2 = atof(fields[i_field+2].c_str());
			bin_state = -1*state_0 + 0*state_1 + 1*state_2;
			(*RV->M)(i_sample,i_snp) = bin_state;
		}

		//increase counter
		i_snp++;
	}; //end for each line
	//resize memory again

	RV->resizeMatrices(num_samples,i_snp);

	//add rowHeader, which correspond to individual IDs
	std::stringstream out_str;
	for(muint_t i=0;i<RV->numSample();++i)
	{
		out_str << i;
		RV->rowHeader->setStr("sample_id",i,out_str.str());
	}

	return RV;
}

CMemGenotypeContainer::CMemGenotypeContainer(PGenotypeBlock block) {
	this->block = block;
	reading_row = 0;
}

CMemGenotypeContainer::~CMemGenotypeContainer() {
}

PGenotypeBlock CMemGenotypeContainer::read(mint_t num_snps) 
	{

	throw CLimixException("fix block to allow reading from a position");
	PGenotypeBlock RV = this->block->read(num_snps);
	reading_row += num_snps;

	return RV;
}

} //end ::limix



