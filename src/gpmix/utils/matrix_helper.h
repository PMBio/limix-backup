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
float_t randn(float_t mu=0.0, float_t sigma=1.0);
//helper functions for eigen matrices
bool isnull(const MatrixXd& m);
double sum(MatrixXd& m);


MatrixXd randn(const uint_t n, const uint_t m);
MatrixXd Mrand(const uint_t n,const uint_t m);


}
#endif /* MATRIX_HELPER_H_ */
