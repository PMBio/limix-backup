// Copyright(c) 2014, The LIMIX developers (Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
// All rights reserved.
//
// LIMIX is provided under a 2-clause BSD license.
// See license.txt for the complete license.

#ifndef LMM_H
#define LMM_H

#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <stdio.h>
#include <cmath>

#include "Eigen/Eigen"
#include "limix/types.h"
#include "limix/utils/mathfunctions.h"
#include "limix/utils/gamma.h"
#include "limix/utils/beta.h"
#include "limix/utils/fisherf.h"
#include "limix/utils/brentc.h"

namespace lmm_old {

#ifndef SWIG
//global variable
//const double _PI = (double) 2.0 * std::acos((double) 0.0);
//const double _log2pi = std::log((double) 2.0 * _PI);

//standard Matrix types that maybe useful here:
//we use columnmajor here because it is more efficient for the LMM code (Note that rowMajor is the order in python)
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> MatrixXd;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> PVector;

// Proper C/C++ versions
inline double nLLeval(MatrixXd & F_tests, double ldelta, const MatrixXd& UY,
		const MatrixXd& UX, const MatrixXd& S);
inline double optdelta(const MatrixXd& UY, const MatrixXd& UX,
		const MatrixXd& S, int numintervals = 100, double ldeltamin = -10,
		double ldeltamax = 10);

#endif
//SWIG friendly interfaces:
void train_associations(MatrixXd* pvals, const MatrixXd& X, const MatrixXd& Y,
		const MatrixXd& K, const MatrixXd& C, int numintervalsAlt = 0,
		double ldeltaminAlt = -1, double ldeltamaxAlt = +1, int numintervals0 =
				100, double ldeltamin0 = -5, double ldeltamax0 = +5);
void train_interactions(MatrixXd* pvals, const MatrixXd& X, const MatrixXd& Y,
		const MatrixXd& K, const MatrixXd& C, const MatrixXd& I,
		int numintervalsAlt, double ldeltaminAlt, double ldeltamaxAlt,
		int numintervals0, double ldeltamin0, double ldeltamax0,
		bool refit_delta0_snp, bool use_ftest);

void train_associations_SingleSNP(MatrixXd* PV, MatrixXd* LL, MatrixXd* ldelta,
		const MatrixXd& X, const MatrixXd& Y, const MatrixXd& U,
		const MatrixXd& S, const MatrixXd& C, int numintervals,
		double ldeltamin, double ldeltamax);
//CLMM class which handles LMM computations

}// end namespace
#endif //LMM_H
