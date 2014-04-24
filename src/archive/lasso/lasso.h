// Copyright(c) 2014, The LIMIX developers (Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
// All rights reserved.
//
// LIMIX is provided under a 2-clause BSD license.
// See license.txt for the complete license.

#ifndef LASSO_H_
#define LASSO_H_

#include "limix/types.h"


namespace limix {


void ridge_regression(MatrixXd* out, const MatrixXd& Xfull, const MatrixXd& y,mfloat_t mu);
void lasso_irr(MatrixXd* w_out,const MatrixXd& X,const MatrixXd& y, mfloat_t mu, mfloat_t optTol=1E-6,mfloat_t threshold=1E-6, muint_t maxIter = 10000);




} //::end namespace limix


#endif /* LASSO_H_ */
