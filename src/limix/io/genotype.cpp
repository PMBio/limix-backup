/*
 * genotype.cpp
 *
 *  Created on: May 16, 2013
 *      Author: stegle
 */

#include "genotype.h"
#include "vcflib/Variant.h"
#include "vcflib/split.h"

using namespace vcf;

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
limix::CTextfileGenotype::CTextfileGenotype(std::string& filename) {
	this->filename = filename;
}

limix::CTextfileGenotype::~CTextfileGenotype() {
}







} //end ::limix

