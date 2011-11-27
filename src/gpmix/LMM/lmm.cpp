/*
 * ALmm.cpp
 *
 *  Created on: Nov 27, 2011
 *      Author: stegle
 */

#include "lmm.h"

namespace gpmix {


const double L2pi=1.8378770664093453;


/*ALMM*/

ALmm::ALmm() {
	//Default settings:
	num_intervals0 = 100;
	num_intervalsAlt = 0;
	ldeltamin0 = -5;
	ldeltamax0 = 5;

}

ALmm::~ALmm() {
}


mfloat_t ALmm::getLdeltamin() const
{
	return ldeltamin;
}

mfloat_t ALmm::getLdeltamin0() const
{
	return ldeltamin0;
}

muint_t ALmm::getNumIntervalsAlt() const
{
	return num_intervalsAlt;
}

muint_t ALmm::getNumSamples() const
{
	return num_samples;
}

void ALmm::setLdeltamin(mfloat_t ldeltamin)
{
	this->ldeltamin = ldeltamin;
}

void ALmm::setLdeltamin0(mfloat_t ldeltamin0)
{
	this->ldeltamin0 = ldeltamin0;
}

void ALmm::setNumIntervalsAlt(muint_t num_intervalsAlt)
{
	this->num_intervalsAlt = num_intervalsAlt;
}

MatrixXd ALmm::getPheno() const
{
	return pheno;
}

MatrixXd ALmm::getPv() const
{
	return pv;
}

void ALmm::getPheno(MatrixXd *out) const
{
	(*out) = pheno;
}

void ALmm::getPv(MatrixXd *out) const
{
	(*out) = pv;
}

void ALmm::getSnps(MatrixXd *out) const
{
	(*out) = snps;
}

MatrixXd ALmm::getSnps() const
{
	return snps;
}

void ALmm::setNumSamples(muint_t num_samples)
{
	this->num_samples = num_samples;}

void ALmm::setData(MatrixXd& snps, MatrixXd& pheno)
{
	//assert checks:
	if (!(snps.rows()==pheno.rows()))
	{
		throw CGPMixException("snps and phenotypes need same number of rows");
	}


	//set data
	this->snps = snps;
	this->pheno = pheno;
	num_samples = snps.rows();
	num_pheno   = pheno.cols();
	num_snps    = snps.cols();

}

/*CLMM*/

CLmm::CLmm() : ALmm(){
}

CLmm::~CLmm() {
	// TODO Auto-generated destructor stub
}

void CLmm::getK(MatrixXd* out) const
{
	(*out) = K;
}

MatrixXd CLmm::getK() const
{
	return K;
}

void CLmm::setK(MatrixXd K)
{
	this->K = K;
}

/*CLMM*/
void CLmm::process()
{
	//TODO
}


CKroneckerLMM::CKroneckerLMM()
{
}


CKroneckerLMM::~CKroneckerLMM()
{
}


void CKroneckerLMM::process()
{}


/* namespace gpmix */
}

