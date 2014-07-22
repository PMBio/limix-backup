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
