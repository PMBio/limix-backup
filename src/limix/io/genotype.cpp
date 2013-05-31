/*
 * genotype.cpp
 *
 *  Created on: May 16, 2013
 *      Author: stegle
 */

#include "genotype.h"

namespace limix {

ASampleData::ASampleData()
{

}


ASampleData::~ASampleData()
{

}
AGenotype::AGenotype() {
	// TODO Auto-generated constructor stub

}

AGenotype::~AGenotype() {
	// TODO Auto-generated destructor stub
}


CMemGenotype::CMemGenotype()
{
	initMatrices();
}

CMemGenotype::CMemGenotype(PGenotype geno)
{
	//1. init matrices
	initMatrices();
	//2. copy data
	MatrixXd genotype;
	geno->agetGenotype(&genotype);
	this->setGenotype(genotype);
	VectorXi position;
	geno->agetPosition(&position);
	this->setPosition(position);
	VectorXs chromosome;
	geno->agetChromosome(&chromosome);
	this->setChromosome(chromosome);
}

void CMemGenotype::initMatrices() {
	geno = PMatrixXd(new MatrixXd());
	pos  = PVectorXi(new VectorXi());
	chrom = PVectorXs(new VectorXs());
	ids   = PVectorXs(new VectorXs());
}


CMemGenotype::~CMemGenotype()
{
}


void CMemGenotype::agetGenotype(MatrixXd* out) const throw(CGPMixException)
{
	(*out) = (*geno);
}


void CMemGenotype::agetPosition(VectorXi* out) const throw(CGPMixException)
{
	(*out) = (*pos);
}

void CMemGenotype::agetChromosome(VectorXs* out) const throw(CGPMixException)
{
	(*out) = (*chrom);
}

void CMemGenotype::setGenotype(const MatrixXd& in)  throw(CGPMixException)
	{
		(*geno) = in;
	}


void CMemGenotype::setPosition(const VectorXi& in)  throw (CGPMixException)
	{
		(*pos) = in;
	}

void CMemGenotype::setChromosome(const VectorXs& in)  throw(CGPMixException)
	{
		(*chrom) = in;
	}

void CMemGenotype::agetIDs(VectorXs* out) const throw(CGPMixException){
	(*out) = (*ids);
}

void CMemGenotype::setIDs(const VectorXs& in) throw (CGPMixException)
	{
(*ids) = in;
	}

} //end ::limix
