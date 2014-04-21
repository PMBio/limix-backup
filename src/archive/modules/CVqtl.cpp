// Copyright(c) 2014, The LIMIX developers (Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
// All rights reserved.
//
// LIMIX is provided under a 2-clause BSD license.
// See license.txt for the complete license.

#include "CVqtl.h"
#include "limix/utils/matrix_helper.h"
#include "limix/gp/gp_opt.h"

namespace limix {
/*
template <typename Derived1>
inline void scale_K(const Eigen::MatrixBase<Derived1> & K_) 
{
	//cast out arguments
	Eigen::MatrixBase<Derived1>& K = const_cast< Eigen::MatrixBase<Derived1>& >(K);
	//ensure that it is a square matrix:
	if (K.rows()!=K.cols())
	{
		throw CLimixException("Kernel scaling requires square kernel matrix");
	}

	//diagonal
	mfloat_t c = K.trace();
	c -= 1.0/K.rows() * K.sum();

	mfloat_t scalar = (K.rows()-1) / c;
	K*=scalar;
}
*/

CVqtl::CVqtl() {
	// TODO Auto-generated constructor stub

}

CVqtl::~CVqtl() {
	// TODO Auto-generated destructor stub
}


void CVqtl::agetPheno(MatrixXd *out) const
{
	(*out) = pheno;
}


void CVqtl::agetSnps(MatrixXd *out) const
{
	(*out) = snps;
}

void CVqtl::agetCovs(MatrixXd *out) const
{
	(*out) = covs;
}

void CVqtl::setCovs(const MatrixXd& covs)
{
	this->covs = covs;
}

void CVqtl::setSNPs(const MatrixXd& snps) 
{
	this->snps = snps;
}

void CVqtl::setPheno(const MatrixXd& pheno) 
{
	if (pheno.cols()!=1)
	{
		throw CLimixException("Currently, CVqtl can only handle univariate phenotypes which need to be of shpae [N x 1]");
	}

	this->pheno = pheno;
}

void CVqtl::setPosition(const VectorXi& position)
{
	this->position = position;
}

void CVqtl::initGP()
{

}

void CVqtl::fitVariances(MatrixXd* out,const MatrixXi& snp_index) 
{
	if ((snp_index.cols()!=2) || (snp_index.rows()==0))
	{
		throw CLimixException("fit function needs a list with matrix indes of size [N x 2]");
	}

	//0. start covaraince fucntion
	covar = PSumCF(new CSumCF(snp_index.rows()));

	//1. create kernel matricse, one per element in snp_index
	for(muint_t iblock=0;iblock<(muint_t)snp_index.rows();++iblock)
	{
		//create block kernel matrix
		muint_t istart = snp_index(iblock,0);
		muint_t istop  = snp_index(iblock,1);
		MatrixXd Ki = this->snps.block(0,istart,this->snps.rows(),istop) * this->snps.block(0,istart,this->snps.rows(),istop).transpose();
		//scale kernel matrix
		VarianceScaleK(Ki);
		//create covaraince and add
		PFixedCF _covar(new CFixedCF(Ki));
		covar->addCovariance(_covar);
	}

	//1. create GP object
	gp =  PGPbase(new CGPbase(covar));
	gp->setY(this->pheno);
	//2. create hyperparams
	CGPHyperParams params;
	params["covar"] = MatrixXd::Ones(gp->getCovar()->getNumberParams(),1);
	params["lik"]   = MatrixXd::Ones(gp->getLik()->getNumberParams(),1);
	//3. create constraint to ensure that noise levl does not go to silly ranges
	//construct constraints
	CGPHyperParams upper;
	CGPHyperParams lower;
	upper["lik"] = 5.0*MatrixXd::Ones(1,1);
	lower["lik"] = -5.0*MatrixXd::Ones(1,1);
	CGPopt opt(gp);
	opt.setOptBoundLower(lower);
	opt.setOptBoundUpper(upper);
	opt.opt();
	//get varaince components
	MatrixXd variances = 2*gp->getParams()["covar"];
	expInplace(variances);


}

mfloat_t CVqtl::testComponent(const MatrixXi snp_index_test,
		const MatrixXi& snp_index_covar) 
{
	return 1.0;
}

void CVqtl::setChrom(const VectorXi& chrom)
{
	this->chrom = chrom;
}





} /* namespace limix */
