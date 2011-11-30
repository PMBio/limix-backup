/*
 * ALmm.cpp
 *
 *  Created on: Nov 27, 2011
 *      Author: stegle
 */

#include "lmm.h"
#include "gpmix/utils/gamma.h"
#include "gpmix/utils/fisherf.h"


namespace gpmix {

const double _PI = (double)(((2.0))) * std::acos((double)(((0.0))));
    const double L2pi = 1.8378770664093453;

/*ALMM*/
ALmm::ALmm()
{
	//Default settings:
	num_intervals0 = 100;
	num_intervalsAlt = 0;
	ldeltamin0 = -5;
	ldeltamax0 = 5;
	UK_cached = false;
	Usnps_cached = false;
	Upheno_cached = false;
	Ucovs_cached = false;
}

ALmm::~ALmm()
{
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

void ALmm::getCovs(MatrixXd *out) const
{
	(*out) = covs;
}

MatrixXd ALmm::getCovs() const
{
	return covs;
}

void ALmm::setCovs(const MatrixXd & covs)
{
	this->covs = covs;
}

void ALmm::setPheno(const MatrixXd & pheno)
{
	this->pheno = pheno;
}

void ALmm::setSNPs(const MatrixXd & snps)
{
	this->snps = snps;
}

mfloat_t ALmm::getLdeltaminAlt() const
{
	return ldeltaminAlt;
}

void ALmm::setLdeltaminAlt(mfloat_t ldeltaminAlt)
{
	this->ldeltaminAlt = ldeltaminAlt;
}

MatrixXd ALmm::getSnps() const
{
	return snps;
}

/*CLMM*/
CLmm::CLmm()
:ALmm()
{
}

CLmm::~CLmm()
{
	// TODO Auto-generated destructor stub
}

void CLmm::getK(MatrixXd *out) const
{
	(*out) = K;
}

MatrixXd CLmm::getK() const
{
	return K;
}

void CLmm::setK(const MatrixXd & K)
{
	this->K = K;
	this->UK_cached = false;
}

void CLmm::setK(const MatrixXd & K, const MatrixXd & U, const VectorXd & S)
{
	this->K = K;
	this->U = U;
	this->S = S;
	this->UK_cached = true;
}

/*CLMM*/
void CLmm::updateDecomposition()
{
	//check that dimensions match
	this->num_samples = snps.rows();
	this->num_snps = snps.cols();
	this->num_pheno = pheno.cols();
	this->num_covs = covs.cols();
	if(!num_samples == pheno.rows())
		throw new CGPMixException("phenotypes and SNP dimensions inconsistent");

	if(!num_samples == covs.rows())
		throw CGPMixException("covariates and SNP dimensions inconsistent");

	if(!(this->UK_cached)){
		//decomposition of K
		Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver(K);
		U = eigensolver.eigenvectors();
		S = eigensolver.eigenvalues();
	}
	if(!Usnps_cached){
		Usnps = U.transpose() * snps;
		Usnps_cached = true;
	}
	if(!Upheno_cached){
		Upheno = U.transpose() * pheno;
		Upheno_cached = true;
	}
	if(!Ucovs_cached){
		Ucovs = U.transpose() * covs;
		Ucovs_cached = true;
	}
}

void CLmm::process()
{
	//get decomposition
	updateDecomposition();
	//resize phenotype output
	this->pv.resize(this->num_pheno, this->num_snps);
	MatrixXd pvF = MatrixXd(this->num_pheno, this->num_snps);

	//result matries: think about what to store in the end
	MatrixXd ldelta0(num_pheno, 1);
	MatrixXd ldelta(num_pheno, num_snps);
	MatrixXd nLL0(num_pheno, 1);
	MatrixXd nLL(num_pheno, num_snps);
	//reserve memory for snp-wise foreground model
	MatrixXd UX_(num_samples, num_covs + 1);
	//store covariates upfront
	UX_.block(0, 0, num_samples, num_covs) = Ucovs;
	MatrixXd f_tests;
	for(muint_t ip = 0;ip < num_pheno;ip++){
		//get UY columns
		MatrixXd UY_ = Upheno.block(0, ip, num_samples, 1);
		//fit delta on null model
		ldelta0(ip) = optdelta(UY_, Ucovs, S, num_intervals0, ldeltamin0, ldeltamax0);
		nLL0(ip) = nLLeval(&f_tests, ldelta0(ip), UY_, Ucovs, S);
		for(muint_t is = 0;is < num_snps;is++){
			//1. construct foreground testing SNP transformed
			UX_.block(0, num_covs, num_samples, 1) = Usnps.block(0, is, num_samples, 1);
			//2. fit delta
			if(num_intervalsAlt > 0)
				//fit delta on alt model also
				ldelta(ip, is) = optdelta(UY_, UX_, S, num_intervalsAlt, ldelta0(ip) + ldeltaminAlt, ldelta0(ip) + ldeltamaxAlt);

			else
				ldelta(ip, is) = ldelta0(ip);

			//3. evaluate
			MatrixXd f_tests;
			nLL(ip, is) = nLLeval(&f_tests, ldelta(ip, is), UY_, UX_, S);
			//4. calc p-value
			this->pv(ip, is) = Gamma::gammaQ(nLL0(ip, 0) - nLL(ip, is), (double)((0.5)) * 1.0);
#if 0
			//compare p-value of LRT with F-test:
			pvF(ip,is) = 1.0 - FisherF::Cdf(f_tests(num_covs, 0), 1.0, (double)((UY_.rows() - f_tests.rows())));
#endif
		} //end for SNP
	} //end for phenotypes
}

CKroneckerLMM::CKroneckerLMM()
{
}

CKroneckerLMM::~CKroneckerLMM()
{
}

void CKroneckerLMM::updateDecomposition()
{
	//TODO: think about caching procedures:


}

void CKroneckerLMM::process()
{
	this->Usnps = this->U_R.transpose() * this->snps;
	this->Upheno= this->U_R.transpose() * this->pheno * this->U_C;
	this->Ucovs = this->U_R.transpose() * this->covs;

	MatrixXd S = MatrixXd(this->S_R.rows(),this->S_C.rows());

	for (muint_t col = 0; col<(muint_t)this->S_C.rows(); ++col)
	{
		for (muint_t row = 0; row<(muint_t)this->S_R.rows(); ++row)
		{
			S(row,col) =this->S_R(row) * this->S_C(col);
		}
	}

	//resize phenotype output
	this->pv.resize(this->pheno.cols(), this->snps.cols());
	//result matrices: think about what to store in the end
	MatrixXd ldelta0(1, 1);
	MatrixXd ldelta(1, snps.cols());
	MatrixXd nLL0(1, 1);
	MatrixXd nLL(1, snps.cols());
	//reserve memory for snp-wise foreground model
	MatrixXd UX(snps.rows(), covs.cols() + 1);
	//store covariates upfront
	UX.block(0, 0, snps.rows(), covs.cols()) = Ucovs;
	MatrixXd f_tests;
	//fit delta on null model
	ldelta0(0) = optdelta(Upheno, Ucovs, S, num_intervals0, ldeltamin0, ldeltamax0);
	nLL0(0) = nLLeval(&f_tests, ldelta0(0), Upheno, Ucovs, S);
	for(muint_t is = 0;is < (muint_t)snps.cols() ;is++){
		//1. construct foreground testing SNP transformed
		UX.block(0, covs.cols(), snps.rows(), 1) = Usnps.block(0, is, snps.rows(), 1);
		//2. fit delta
		if(num_intervalsAlt > 0)
			//fit delta on alt model also
			ldelta(0, is) = optdelta(Upheno, UX, S, num_intervalsAlt, ldelta0(0) + ldeltaminAlt, ldelta0(0) + ldeltamaxAlt);
		else
			ldelta(0, is) = ldelta0(0);
		//3. evaluate
		MatrixXd f_tests;
		nLL(0, is) = nLLeval(&f_tests, ldelta(0, is), Upheno, UX, S);

		//TODO:
		//rotate F_tests to get per pheno P-values

		//4. calc lod score
		this->pv(0, is) = Gamma::gammaQ(nLL0(0, 0) - nLL(0, is), (double)((0.5) * this->pheno.cols()));
	} //end for phenotypes
}


void CKroneckerLMM::getK_C(MatrixXd *out) const
  {
	(*out) = C;
  }

  void CKroneckerLMM::setK_C(const MatrixXd & C)
  {
	  this->C = C;
	  Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver(C);
	  this->U_C = eigensolver.eigenvectors();
	  this->S_C = eigensolver.eigenvalues();
  }

  void CKroneckerLMM::setK_C(const MatrixXd & C, const MatrixXd & U_C, const VectorXd & S_C)
  {
	this->C=C;
	this->U_C = U_C;
	this->S_C = S_C;
  }

  void CKroneckerLMM::getK_R(MatrixXd *out) const
  {
	(*out) = R;
  }

  void CKroneckerLMM::setK_R(const MatrixXd & R)
  {
	  this->R = R;
	  Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver(R);
	  this->U_R = eigensolver.eigenvectors();
	  this->S_R = eigensolver.eigenvalues();
  }

  void CKroneckerLMM::setK_R(const MatrixXd & R, const MatrixXd & U_R, const VectorXd & S_R)
  {
	this->R = R;
	this->U_R = U_R;
	this->S_R = S_R;
  }


  MatrixXd CKroneckerLMM::getK_R() const
  {
	return R;
  }

  MatrixXd CKroneckerLMM::getK_C() const
  {
	return C;
  }






double optdelta(const MatrixXd & UY, const MatrixXd & UX, const MatrixXd & S, int numintervals, double ldeltamin, double ldeltamax)
{
	//grid variable with the current likelihood evaluations
	MatrixXd nllgrid     = MatrixXd::Ones(numintervals,1).array()*HUGE_VAL;
	MatrixXd ldeltagrid = MatrixXd::Zero(numintervals, 1);
	//current delta
	double ldelta = ldeltamin;
	double ldeltaD = (ldeltamax - ldeltamin);
	ldeltaD /= ((double)(((numintervals))) - 1);
	double nllmin = HUGE_VAL;
	double ldeltaopt_glob = 0;
	MatrixXd f_tests;
	for(int i = 0;i < (numintervals);i++){
		nllgrid(i, 0) = nLLeval(&f_tests, ldelta, UY, UX, S);
		ldeltagrid(i, 0) = ldelta;
		//std::cout<< "nnl( " << ldelta << ") = " << nllgrid(i,0) <<  "VS" << nllmin << ")\n\n";
		if(nllgrid(i, 0) < nllmin){
			//		std::cout << "new min (" << nllmin << ") -> " <<  nllgrid(i,0) << "\n\n";
			nllmin = nllgrid(i, 0);
			ldeltaopt_glob = ldelta;
		}
		//move on delta
		ldelta += ldeltaD;
	} //end for all intervals

	//std::cout << "\n\n nLL_i:\n" << nllgrid;
	return ldeltaopt_glob;
}

/* internal functions */
double nLLeval(MatrixXd *F_tests, double ldelta, const MatrixXd & UY, const MatrixXd & UX, const MatrixXd & S)
{
	size_t n = UX.rows();
	size_t d = UX.cols();
	size_t n_pheno = UY.cols();

	assert(UY.cols() == S.cols());
	assert(UY.rows() == S.rows());
	assert(UY.rows() == UX.rows());

	double delta = exp(ldelta);
	MatrixXd Sdi = S.array() + delta;
	double ldet = 0.0;
	for(size_t ind = 0;ind < n_pheno * n;++ind){
		ldet += log(Sdi.data()[ind]);
	}
	Sdi = Sdi.array().inverse();
	(*F_tests).resize(d, n_pheno);
	MatrixXd beta =MatrixXd(d,n_pheno);
	//replice Sdi
	for (muint_t phen = 0; phen<n_pheno;++phen)
	{
		VectorXd Sdi_p = Sdi.block(0,phen,n,1);
		MatrixXd XSdi = (UX.array() * Sdi_p.replicate(1, d).array()).transpose();
		MatrixXd XSX = XSdi * UX;
		MatrixXd XSY = XSdi * UY.block(0,phen,n,1);
		//least sqaures solution of XSX*beta = XSY
		//decomposition of K
		Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver(XSX);
		MatrixXd U_X = eigensolver.eigenvectors();
		MatrixXd S_X = eigensolver.eigenvalues();
		beta.block(0,phen,d,1) = U_X.transpose() * XSY;
		//MatrixXd S_i = MatrixXd::Zero(d,d);
		for(size_t dim = 0;dim < d;++dim){
			if(S_X(dim, 0) > 3E-8)
			{
				beta(dim,phen) /= S_X(dim, 0);
				for(size_t dim2 = 0;dim2 < d;++dim2)
				{
					(*F_tests)(dim2, phen) += U_X(dim2, dim) * U_X(dim2, dim) / S_X(dim, 0);
				}
				//S_i(dim,dim) = 1.0/S_X(dim,0);
			}
			else
			{
				beta(dim,phen) = 0.0;
			}
		}
		beta.block(0,phen,d,1) = U_X * beta.block(0,phen,d,1);
	}
	MatrixXd res = UY - UX * beta;
	//sqared residuals
	res.array() *= res.array();
	res.array() *= Sdi.array();
	double sigg2 = res.array().sum() / (n * n_pheno);
	//compute the F-statistics
	(*F_tests).array() = beta.array() * beta.array() / (*F_tests).array();
	(*F_tests).array() /= sigg2;
	double nLL = 0.5 * (n * n_pheno * L2pi + ldet + n * n_pheno + n * n_pheno * log(sigg2));
	return nLL;
}

void optdeltaAllY(MatrixXd *out, const MatrixXd & UY, const MatrixXd & UX, const MatrixXd & S, const MatrixXd & ldeltagrid)
{
	size_t n_p = UY.cols();
	size_t numintervals = ldeltagrid.rows();
	//grid variable with the current likelihood evaluations
	(*out) = MatrixXd::Ones(numintervals,n_p).array()*HUGE_VAL;
	//current delta
	for(size_t i = 0;i < numintervals;i++){
		MatrixXd row;
		nLLevalAllY(&row, ldeltagrid(i, 0), UY, UX, S);
		(*out).row(i) = row;
		//ldeltagrid(0,i) = ldelta;
	} //end for all intervals
}

void train_associations_SingleSNP(MatrixXd *PV, MatrixXd *LL, MatrixXd *ldelta, const MatrixXd & X, const MatrixXd & Y, const MatrixXd & U, const MatrixXd & S, const MatrixXd & C, int numintervals, double ldeltamin, double ldeltamax)
{
	//get dimensions:
	//samples
	size_t nn = X.rows();
	//snps
	size_t ns = X.cols();
	assert( ns == 1 );
	//phenotypes
	size_t np = Y.cols();
	//covaraites
	size_t nc = C.cols();
	//make sure the size of N/Y is correct
	assert((int)nn==(int)Y.rows());
	assert((int)nn==(int)C.rows());
	assert((int)nn==(int)U.rows());
	assert((int)nn==(int)U.cols());
	//resize output variable if needed
	(*PV).resize(np, ns);
	(*LL).resize(np, ns);
	(*ldelta).resize(np, ns);
	//transform everything
	MatrixXd UX = U.transpose() * X;
	MatrixXd UY = U.transpose() * Y;
	MatrixXd Ucovariates = U.transpose() * C;
	//reserve memory for snp-wise foreground model
	MatrixXd UX_(nn, nc + 1);
	//store covariates upfront
	UX_.block(0, 0, nn, nc) = Ucovariates;
	UX_.block(0, nc, nn, 1) = UX;
	MatrixXd ldeltagrid(numintervals, 1);
	for(size_t interval = 0;interval < (size_t)(numintervals);++interval){
		ldeltagrid(interval, 0) = ldeltamin + interval * ((ldeltamax - ldeltamin) / (1.0 * (numintervals - 1)));
	}
	MatrixXd nllgrid;
	optdeltaAllY(&nllgrid, UY, UX_, S, ldeltagrid);
	//1. fit background covariances on phenotype and covariates alone
	for(size_t ip = 0;ip < np;ip++){
		(*ldelta)(ip) = ldeltamin;
		size_t i_min = 0;
		for(size_t interval = 1;interval < (size_t)(numintervals);++interval){
			if(nllgrid(interval, ip) < nllgrid(i_min, ip)){
				//printf("oldmin : %.4f, newmin : %.4f, newdelta : %.4f, interval : %i\n" ,nllgrid(i_min,ip) , nllgrid(interval,ip), ldeltagrid(interval,0),interval);
				(*ldelta)(ip, 0) = ldeltagrid(interval, 0);
				i_min = interval;
			}
		}

		//get UY columns
		MatrixXd UY_ = UY.block(0, ip, nn, 1);
		//fit delta on null model
		MatrixXd f_tests;
		(*LL)(ip, 0) = -1.0 * nLLeval(&f_tests, (*ldelta)(ip), UY_, UX_, S);
		(*PV)(ip, 0) = 1.0 - FisherF::Cdf(f_tests(nc, 0), 1.0, (double)((nn - f_tests.rows())));
	}

}

/* Internal C++ functions */
void nLLevalAllY(MatrixXd *out, double ldelta, const MatrixXd & UY, const MatrixXd & UX, const MatrixXd & S)
{
	size_t n = UX.rows();
	size_t d = UX.cols();
	size_t p = UY.cols();
	/*
std::cout << UX<< "\n\n";
std::cout << UY<< "\n\n";
std::cout << S<< "\n\n";
 */
	double delta = exp(ldelta);
	MatrixXd Sdi = S.array() + delta;
	double ldet = Sdi.array().log().sum();
	//std::cout << "ldet" << ldet << "\n\n";
	//elementwise inverse
	Sdi = Sdi.array().inverse();
	//std::cout << "Sdi" << Sdi << "\n\n";
	//replice Sdi
	MatrixXd XSdi = (UX.array() * Sdi.replicate(1, d).array()).transpose();
	MatrixXd XSX = XSdi * UX;
	MatrixXd XSY = XSdi * UY;
	//std::cout << "XSdi" << XSdi << "\n\n";
	//std::cout << "XSX" << XSX << "\n\n";
	//least squares solution of XSX*beta = XSY
	MatrixXd beta = XSX.colPivHouseholderQr().solve(XSY);
	MatrixXd res = UY - UX * beta;
	//squared residuals
	res.array() *= res.array();
	res.array() *= Sdi.replicate(1, p).array();
	//MatrixXd sigg2 = MatrixXd(1,p);
	(*out) = MatrixXd(1, p);
	for(size_t phen = 0;phen < p;++phen){
		double sigg2 = res.col(phen).array().sum() / n;
		(*out)(0, phen) = 0.5 * (n * L2pi + ldet + n + n * log(sigg2));
	}
}



/* namespace gpmix */
}

