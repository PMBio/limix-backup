/*
 * genotype.cpp
 *
 *  Created on: May 16, 2013
 *      Author: stegle
 */

#include "genotype.h"

//#include "vcflib/Variant.h"
#include "split.h"
#include <string>
#include <vector>
#include <numeric>

#ifdef ZLIB
#include "gzstream.h"
#endif

//namespace io = boost::iostreams;

namespace limix {


CGenotypeBlock::CGenotypeBlock() {
	this->pos = PVectorXi(new VectorXi());
	this->chrom = PVectorXs(new VectorXs());
	i_snp_read = 0;
}

CGenotypeBlock::CGenotypeBlock(const CGenotypeBlock& copy) : CMemDataFrame<MatrixXd>(copy)
{
	pos = PVectorXi(new VectorXi());
	chrom = PVectorXs(new VectorXs());
	*(this->pos) = *(copy.pos);
	*(this->chrom) = *(copy.chrom);
	i_snp_read = 0;
}

CGenotypeBlock::CGenotypeBlock(PMatrixXd geno, PVectorXs chrom, PVectorXi pos,PVectorXs sampleIDs,PVectorXs snpIDs) {
	this->M = geno;
	this->chrom = chrom;
	this->pos = pos;
	this->rowHeader = sampleIDs;
	this->colHeader = snpIDs;
	i_snp_read = 0;
}

CGenotypeBlock::~CGenotypeBlock()
{
}

void CGenotypeBlock::agetPosition(VectorXi* out) const throw(CGPMixException)
{
	(*out) = (*pos);
}
void CGenotypeBlock::resizeMatrices(muint_t num_samples, muint_t num_snps)
{
	//resize base matrix type
	CMemDataFrame<MatrixXd>::resizeMatrices(num_samples,num_snps);
	//resize SNP specific elements
	this->chrom->conservativeResize(num_snps);
	this->pos->conservativeResize(num_snps);
}

PGenotypeBlock CGenotypeBlock::read(mint_t num_snps) throw(CGPMixException)
{
	//build sub matrices:
	PMatrixXd geno = PMatrixXd(new MatrixXd(this->M->block(0,i_snp_read,this->numSample(),i_snp_read+num_snps)));
	PVectorXs chrom = PVectorXs(new VectorXs(this->chrom->segment(i_snp_read,i_snp_read+num_snps)));
	PVectorXi pos = PVectorXi(new VectorXi(this->pos->segment(i_snp_read,i_snp_read+num_snps)));
	PVectorXs colHeader = PVectorXs(new VectorXs(this->colHeader->segment(i_snp_read,i_snp_read+num_snps)));
	PVectorXs rowHeader = PVectorXs(new VectorXs(this->rowHeader->segment(i_snp_read,i_snp_read+num_snps)));

	PGenotypeBlock RV = PGenotypeBlock(new CGenotypeBlock(geno,chrom,pos,rowHeader,colHeader));
	return RV;
}


PVectorXi CGenotypeBlock::getPosition() const throw(CGPMixException)
{
	return pos;
}

PVectorXs CGenotypeBlock::getChromosome() const throw(CGPMixException)
{
	return chrom;
}

void CGenotypeBlock::agetChromosome(VectorXs* out) const throw(CGPMixException)
{
	(*out) = (*chrom);
}


/* Text File genotype class */
limix::CTextfileGenotypeContainer::CTextfileGenotypeContainer(const std::string& filename){
	this->in_filename = filename;
	buffer_size = 100000;
	is_open = false;
}

limix::CTextfileGenotypeContainer::~CTextfileGenotypeContainer() {
}

void CTextfileGenotypeContainer::openFile() throw (CGPMixException)
{
	//take filename apart and check whether ending is .gzip
    std::vector<std::string> filenameParts = split(in_filename, ".");

    std::string ext;

    //is the file gzip ?
    if (filenameParts.back() == "gz")
    {
      #ifdef ZLIB
    	in_stream = new igzstream(in_filename.c_str());
    	ext = filenameParts.at(filenameParts.size()-2);
      #else
	throw CGPMixException("not compiled with zlib");
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
    	throw CGPMixException("bz2 not supported");
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
		throw CGPMixException("unknown file format");

	is_open = true;

}


PGenotypeBlock CTextfileGenotypeContainer::read(mint_t num_snps) throw (CGPMixException)
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
		throw CGPMixException("unsupported file format in read");
}

void CTextfileGenotypeContainer::read_header_GEN()
{

}

/*
 * read num_snps lines from .bed file
 */
PGenotypeBlock CTextfileGenotypeContainer::read_BED(muint_t num_snps) throw (CGPMixException)
{
	PGenotypeBlock RV = PGenotypeBlock(new CGenotypeBlock());
	throw CGPMixException("BED readder not implemented");
	return RV;
}


PGenotypeBlock CTextfileGenotypeContainer::read_GEN(muint_t num_snps) throw (CGPMixException) {

	//creat result Structure
	PGenotypeBlock RV = PGenotypeBlock(new CGenotypeBlock());

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
			throw CGPMixException("Line while reading as inconsistent length");

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
		(*RV->chrom)(i_snp) = chrom;
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

	return RV;
}

CMemGenotypeContainer::CMemGenotypeContainer(PGenotypeBlock block) {
	this->block = block;
	reading_row = 0;
}

CMemGenotypeContainer::~CMemGenotypeContainer() {
}

PGenotypeBlock CMemGenotypeContainer::read(mint_t num_snps) throw(CGPMixException) {

	//read elements from the in-memory variant and create new block
	PMatrixXd geno = PMatrixXd(new MatrixXd());
	PVectorXi pos  = PVectorXi(new VectorXi());
	PVectorXs chrom = PVectorXs(new VectorXs());
	PVectorXs colHeader = PVectorXs(new VectorXs());
	PVectorXs rowHeader = PVectorXs(new VectorXs());


	*geno = this->block->getMatrix()->block(0,reading_row,block->numSample(),reading_row+num_snps);
	*pos = this->block->getPosition()->segment(reading_row,reading_row+num_snps);
	*chrom = this->block->getChromosome()->segment(reading_row,reading_row+num_snps);
	*colHeader = this->block->getColHeader()->segment(reading_row,reading_row+num_snps);
	*rowHeader = *this->block->getRowHeader();

	reading_row += num_snps;

	PGenotypeBlock RV = PGenotypeBlock(new CGenotypeBlock(geno,chrom,pos,rowHeader,colHeader));

	return RV;
}

} //end ::limix



