/*
 * matrix_helper.h
 *
 *  Created on: Nov 10, 2011
 *      Author: stegle
 */

#ifndef MATRIX_HELPER_H_
#define MATRIX_HELPER_H_

#include <math.h>
#include <cmath>
#include <gpmix/types.h>

namespace gpmix{

//create random matrix:
mfloat_t randn(mfloat_t mu=0.0, mfloat_t sigma=1.0);
//helper functions for eigen matrices
bool isnull(const MatrixXd& m);
bool isnull(const Eigen::LLT<gpmix::MatrixXd>& m);
//calculate log determinant form cholesky factor
mfloat_t logdet(Eigen::LLT<gpmix::MatrixXd>& chol);

double sum(MatrixXd& m);


MatrixXd randn(const muint_t n, const muint_t m);
MatrixXd Mrand(const muint_t n,const muint_t m);


}
#endif /* MATRIX_HELPER_H_ */
