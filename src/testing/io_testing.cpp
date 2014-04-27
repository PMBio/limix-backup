// Copyright(c) 2014, The LIMIX developers (Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
// All rights reserved.
//
// LIMIX is provided under a 2-clause BSD license.
// See license.txt for the complete license.
#if 1

#include <iostream>
#include "limix/types.h"
#include "limix/covar/covariance.h"
#include "limix/covar/linear.h"
#include "limix/utils/matrix_helper.h"
#include "limix/modules/CVarianceDecomposition.h"
#include "limix/io/genotype.h"
#include <string>
#include <iostream>


#include "limix/types.h"


using namespace std;
using namespace limix;


int main() {
	/*
	PMatrixXd TTf = PMatrixXd(new MatrixXd());
	(*TTf) = MatrixXd::Ones(100,100);

	PMatrixXi TTi = PMatrixXi(new MatrixXi());
	(*TTi) = MatrixXi::Ones(100,100);

	PArray1DXs TTs = PArray1DXs(new Array1DXs(100));
	PArrayXs TTss = PArrayXs(new ArrayXs(100,100));
	CFlexMatrix::PStringMatrix TT2s = CFlexMatrix::PStringMatrix(new CFlexMatrix::StringMatrix(100,100));

	ArrayXs CTTss(100,100);
	CFlexMatrix::StringMatrix CTTss2(100,100);

	PFlexMatrix testf = PFlexMatrix(new CFlexMatrix(TTf));
	PFlexMatrix testi = PFlexMatrix(new CFlexMatrix());
	PFlexVector tests = PFlexVector(new CFlexVector());
	PFlexMatrix tests2 = PFlexMatrix(new CFlexMatrix());

	CTTss=CTTss2;

	*testi = TTi;
	*tests2 = TT2s;

	TTss = TT2s;

	
	PMatrixXd TTf2 = *testf;
	MatrixXd TTf3 = *(PMatrixXd)(*testf);

	CFlexMatrix::PIntMatrix TTi2 = *testi;
	MatrixXi TTi3 = *testi;

	//assignment operators
	std::cout << TTi3;
	std::cout << testi;

	//std:: cout <<


	//pp =  test.getFloatArray();

	//1. simulate
	muint_t n  = 20;
	muint_t p = 2;
	muint_t s = 1000;
	muint_t ncov = 1;

	MatrixXd snps = (MatrixXd)randn((muint_t)n,(muint_t)s);
	MatrixXd pheno = (MatrixXd)randn((muint_t)n,(muint_t)p);
	MatrixXd covs = MatrixXd::Ones(n,ncov);
	*/

	string filename = "/Users/stegle/research/users/stegle/limix/vcf_gen/test.gen";

	PTextfileGenotypeContainer genoContainer = PTextfileGenotypeContainer(new CTextfileGenotypeContainer(filename));

	PGenotypeBlock geno = genoContainer->read();
	PVectorXi pos =geno->getPosition();
	PHeaderMap rowh = geno->getRowHeader();
	PHeaderMap colh = geno->getColHeader();
	PMatrixXd gen = geno->getMatrix();

	std::cout << pos->rows() << "\n";
	std::cout << gen->rows() << "\n";

}

#endif
