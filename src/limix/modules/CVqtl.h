/*
 * CVqtl.h
 *
 *  Created on: Jul 26, 2012
 *      Author: stegle
 */

#ifndef CVQTL_H_
#define CVQTL_H_

#include "limix/types.h"
#include "limix/gp/gp_base.h"
#include "limix/covar/linear.h"
#include "limix/covar/combinators.h"
#include "limix/covar/fixed.h"




namespace limix {


//rename argout operators for swig interface
#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
//ignore C++ versions
%ignore Cvqtl::getPheno;
%ignore Cvqtl::getPv;
%ignore Cvqtl::getSnps;
%ignore Cvqtl::getCovs;
%ignore Cvqtl::getK;
%ignore Cvqtl::getPermutation;

%rename(getPheno) Cvqtl::agetPheno;
%rename(getSnps) Cvqtl::agetSnps;
%rename(getCovs) Cvqtl::agetCovs;
//%template(IndexVec) std::vector<MatrixXi>;
#endif





//vector on SNP sets
typedef std::vector<VectorXi> VectorXiVec;

class CVqtl {
protected:
	//genotype matrix
	MatrixXd snps;
	//phenotype matrix
	MatrixXd pheno;
	MatrixXd covs;
	VectorXi position;
	VectorXi chrom;

	//gp object for fitting
	PGPbase gp;
	PSumCF covar;

	void initGP();
public:
	CVqtl();
	virtual ~CVqtl();

	//fit a kinship model with an arbitrary number of terms as denoted
	//in the std vector
	void fitVariances(MatrixXd* out, const MatrixXi& snp_index) throw(CGPMixException);
	mfloat_t testComponent(const MatrixXi snp_index_test, const MatrixXi& snp_index_covar) throw(CGPMixException);


	//setters and getters
	//getters:
    void agetPheno(MatrixXd *out) const;
    void agetSnps(MatrixXd *out) const;
    void agetCovs(MatrixXd *out) const;
    void setCovs(const MatrixXd & covs);
    void setPheno(const MatrixXd & pheno) throw(CGPMixException);
    void setSNPs(const MatrixXd & snps) throw(CGPMixException);
    void setPosition(const VectorXi& position);
    void setChrom(const VectorXi& chrom);

};

} /* namespace limix */
#endif /* CVQTL_H_ */
