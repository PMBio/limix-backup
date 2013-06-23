#if 1
//============================================================================
// Name        : GPmix.cpp
// C++ testig file for variance component QTL
//============================================================================

#include <iostream>
#include "limix/types.h"
#include "limix/covar/covariance.h"
#include "limix/covar/linear.h"
#include "limix/utils/matrix_helper.h"
#include "limix/modules/CMultiTraitVQTL.h"
#include "limix/modules/CVarianceDecomposition.h"
#include "limix/io/genotype.h"
#include <string>
#include <iostream>


#include "limix/io/vcf/Variant.h"
#include "limix/io/split.h"
#include "limix/types.h"


using namespace vcf;
using namespace std;
using namespace limix;


int main() {

	//1. simulate
	muint_t n  = 20;
	muint_t p = 2;
	muint_t s = 1000;
	muint_t ncov = 1;

	MatrixXd snps = (MatrixXd)randn((muint_t)n,(muint_t)s);
	MatrixXd pheno = (MatrixXd)randn((muint_t)n,(muint_t)p);
	MatrixXd covs = MatrixXd::Ones(n,ncov);

	/*
	string filename = "/Users/stegle/research/users/stegle/limix/vcf_gen/sample2.gen";

	PTextfileGenotypeContainer genoContainer = PTextfileGenotypeContainer(new CTextfileGenotypeContainer(filename));

	PGenotypeBlock geno = genoContainer->read(4);
	PVectorXi pos =geno->getPosition();
	PMatrixXd gen = geno->getMatrix();

	std::cout << pos->rows() << "\n";
	std::cout << gen->rows() << "\n";

	std::cout << (*gen);

	std::cout << "\n\n";

	std::cout << (*pos);

	PGenotypeBlock geno2 = genoContainer->read(-1);
	PVectorXi pos2 =geno2->getPosition();
	PMatrixXd gen2 = geno2->getMatrix();

	std::cout << pos2->rows() << "\n";
	std::cout << gen2->rows() << "\n";
	*/

	//CMemGenotype geno;

	//PDMemDataFrame data;

	//PMemGenotype geno = PMemGenotype(new CMemGenotype());
	//geno->setGenotype(snps);

	//class to read .gen files






	//class to read VCF files
	string filename = "/Users/stegle/research/users/stegle/limix/vcf_gen/sample.vcf";
	VariantCallFile variantFile;
	variantFile.open(filename);

	/*
	if (!variantFile.is_open()) {
        return 1;
    }

	Variant var(variantFile);
    while (variantFile.getNextVariant(var)) {
        map<string, map<string, vector<string> > >::iterator s     = var.samples.begin();
        map<string, map<string, vector<string> > >::iterator sEnd  = var.samples.end();

        cout << var.sequenceName << "\t"
             << var.position     << "\t"
             << var.ref          << "\t";
        var.printAlt(cout);     cout << "\t";
        var.printAlleles(cout); cout << "\t";

        while(s!=sEnd)
        {

            map<string, vector<string> >& sample = s->second;
            string& genotype = sample["GT"].front(); // XXX assumes we can only have one GT value
            vector<string> gt = split(genotype, "|/");

            // report the sample and it's genotype
            cout << s->first << ":";
            for (vector<string>::iterator g = gt.begin(); g != gt.end(); ++g) {
                int index = atoi(g->c_str());
                cout << var.alleles[index];
                if (g != (gt.end()-1)) cout << "/";
            }
            cout << "\t";

            ++s;
        }
        cout << endl;
    }
	*/

}

#endif
