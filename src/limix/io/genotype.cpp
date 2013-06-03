/*
 * genotype.cpp
 *
 *  Created on: May 16, 2013
 *      Author: stegle
 */

#include "genotype.h"
#include "vcflib/Variant.h"
//#include "vcflib/split.h"
#include <string>
#include <vector>

namespace io = boost::iostreams;

namespace limix {


CMemGenotype::CMemGenotype() {
	this->pos = PVectorXi(new VectorXi());
	this->chrom = PVectorXs(new VectorXs());
}

CMemGenotype::CMemGenotype(const AGenotype& copy) : AGenotype(copy)
{
	this->pos = copy.getPosition();
	this->chrom = copy.getChromosome();
}


CMemGenotype::CMemGenotype(PMatrixXd geno, PVectorXs chrom, PVectorXi pos,
		PVectorXs IDs) {

}

CMemGenotype::~CMemGenotype()
{
}

void CMemGenotype::agetPosition(VectorXi* out) const throw(CGPMixException)
{
	(*out) = (*pos);
}

void CMemGenotype::resizeMatrices(muint_t num_samples, muint_t num_snps)
{
	//resize base matrix type
	CMemDataFrame<MatrixXd>::resizeMatrices(num_samples,num_snps);
	//resize SNP specific elements
	this->chrom->resize(num_snps);
	this->pos->resize(num_snps);
}

PVectorXi CMemGenotype::getPosition() const throw(CGPMixException)
{
	return pos;
}

PVectorXs CMemGenotype::getChromosome() const throw(CGPMixException)
{
	return chrom;
}

void CMemGenotype::agetChromosome(VectorXs* out) const throw(CGPMixException)
{
	(*out) = (*chrom);
}

/*
void CMemGenotype::setPosition(const VectorXi& in) throw(CGPMixException)
{
	(*pos) = in;
}

void CMemGenotype::setPosition(PVectorXi in) throw(CGPMixException)
{
	pos =in;
}

void CMemGenotype::setChromosome(const VectorXs& in) throw(CGPMixException)
{
	(*chrom) =in;
}

void CMemGenotype::setChromosome(PVectorXs in) throw(CGPMixException)
{
	chrom =in;
}
*/


/* Text File genotype class */
limix::CTextfileGenotype::CTextfileGenotype(const std::string& filename) {
	this->in_filename = filename;
}

limix::CTextfileGenotype::~CTextfileGenotype() {
}

void CTextfileGenotype::openFile()
{
	//take filename apart and check whether ending is .gzip
    vector<string> filenameParts = split(in_filename, ".");

    string ext;
    //is the file gzip ?
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
	//remember extension to call appropriate reader
	if(ext=="gen")
		this->file_format = GEN;
	else if(ext=="vcf")
		this->file_format = VCF;
	else if(ext=="bed")
		this->file_format = BED;
	else
		throw CGPMixException("unknown file format");
}

void CTextfileGenotype::read(muint_t buffer_size)
{
	//open file
	openFile();
	//which file format?
	if(this->file_format==GEN)
		read_GEN(buffer_size);
}

void CTextfileGenotype::read_GEN(muint_t block_size) {

	//number of read snps, current buffer
	muint_t i_snp,buffer,num_samples;

	buffer        = 0;
	num_samples   = 0;
	i_snp         = 0;

	//temporary variables for reading
	string line;
	string chrom,snp_id,snp_a,snp_b;
	muint_t snp_pos;
	while(std::getline(in_stream,line))
	{
		//std::cout << line;
		//parse line
		std::vector<string> fields = split(line, ' ');
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
			buffer += block_size;
			resizeMatrices(num_samples,buffer);
		}
		//3. store position and chromosome
		(*this->chrom)(i_snp) = chrom;
		(*this->pos)(i_snp) = snp_pos;

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
			(*this->M)(i_sample,i_snp) = bin_state;
		}

		//increase counter
		i_snp++;
	}; //end for each line
	//resize memory again
	resizeMatrices(num_samples,i_snp);
}

} //end ::limix

