/*
 * lasso.h
 *
 *  Created on: Dec 24, 2011
 *      Author: stegle
 */

#ifndef LASSO_H_
#define LASSO_H_

#include "gpmix/types.h"

namespace gpmix {


void ridge_regression(MatrixXd* out, const MatrixXd& Xfull, const MatrixXd& y,mfloat_t mu);
void lasso_irr(MatrixXd* w_out,const MatrixXd& X,const MatrixXd& y, mfloat_t mu, mfloat_t optTol=1E-6,mfloat_t threshold=1E-6, muint_t maxIter = 10000);




} //::end namespace gpmix


#endif /* LASSO_H_ */
