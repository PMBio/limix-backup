/*
 * matrix_helper.cpp
 *
 *  Created on: Nov 10, 2011
 *      Author: stegle
 */


#include <gpmix/utils/matrix_helper.h>
#include "matrix_helper.h"
#include <stdlib.h>

namespace gpmix{

bool isnull(const MatrixXd& m)
{
	return (m.cols()==0) & (m.rows()==0);
}
bool isnull(const Eigen::LLT<gpmix::MatrixXd>& m)
{
	return (m.cols()==0) & (m.rows()==0);
}

bool isnull(const Eigen::LDLT<gpmix::MatrixXd>& m)
{
	return (m.cols()==0) & (m.rows()==0);
}

mfloat_t logdet(Eigen::LLT<MatrixXd>& chol)
{
	//1. logdet
	VectorXd L = ((MatrixXd)(chol.matrixL())).diagonal();
	mfloat_t log_det = 0.0;
	for(muint_t i = 0;i < (muint_t)(L.rows());++i){
		log_det += gpmix::log((mfloat_t)(L(i))); //WARNING: mfloat_t cast
	}
	return 2*log_det;
}

mfloat_t logdet(Eigen::LDLT<gpmix::MatrixXd>& chol)
{
	//1. logdet
	VectorXd L = chol.vectorD();
	mfloat_t log_det = 0.0;
	//iterate: here log(sqrt)
	for(muint_t i = 0;i < (muint_t)(L.rows());++i)
	{
		log_det += gpmix::log(L(i));
	}
	//note: factor of 2 missing because D = sqrt(L.diag) in LLT decomposition
	return log_det;
}



#ifndef PI
#define PI 3.14159265358979323846
#endif

mfloat_t randn(mfloat_t mu, mfloat_t sigma) {
	static bool deviateAvailable=false;	//	flag
	static mfloat_t storedDeviate;			//	deviate from previous calculation
	double dist, angle;

	//	If no deviate has been stored, the standard Box-Muller transformation is
	//	performed, producing two independent normally-distributed random
	//	deviates.  One is stored for the next round, and one is returned.
	if (!deviateAvailable) {

		//	choose a pair of uniformly distributed deviates, one for the
		//	distance and one for the angle, and perform transformations
		dist=sqrt( -2.0 * log(double(rand()) / double(RAND_MAX)) );
		angle= 2.0 * PI * (double(rand()) / double(RAND_MAX));

		//	calculate and store first deviate and set flag
		storedDeviate=dist*cos(angle);
		deviateAvailable=true;

		//	calcaulate return second deviate
		return (mfloat_t)(dist * sin(angle) * sigma + mu);
	}

	//	If a deviate is available from a previous call to this function, it is
	//	returned, and the flag is set to false.
	else {
		deviateAvailable=false;
		return storedDeviate*sigma + mu;
	}
}






MatrixXd randn(const muint_t n, const muint_t m)
/* create a randn matrix, i.e. matrix of Gaussian distributed random numbers*/
{
	MatrixXd rv(n,m);
	for (muint_t i=0; i<n; i++)
		for (muint_t j=0; j<m; j++) {
			double r = randn(0.0,1.0);
			rv(i,j) = r;
		}
	return rv;
}


MatrixXd Mrandrand(const muint_t n,const muint_t m)
{
	MatrixXd rv(n,m);
	for (muint_t i=0;i<n;i++)
		for(muint_t j=0;j<m;j++)
		{
			rv(i,j) = ((double)rand())/RAND_MAX;
		}
	return rv;
}


}
