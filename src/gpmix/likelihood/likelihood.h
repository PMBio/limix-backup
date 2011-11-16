/*
 * likelihood.h
 *
 *  Created on: Nov 11, 2011
 *      Author: clippert
 */

#ifndef LIKELIHOOD_H_
#define LIKELIHOOD_H_

#include <gpmix/types.h>


namespace gpmix {


typedef MatrixXd LikParams;

class ALikelihood {
public:
	ALikelihood();
	virtual ~ALikelihood();
	virtual void applyToK(const LikParams& params, MatrixXd& K) const = 0;
	virtual MatrixXd K_grad_theta(const LikParams& params, MatrixXd X, uint_t row) const = 0;
	virtual MatrixXd K(const LikParams& params, MatrixXd& X) const = 0;
	virtual VectorXd Kdiag(const LikParams& params, MatrixXd& X) const = 0;
};

class CLikNormalIso : public ALikelihood {
public:
	CLikNormalIso();
	~CLikNormalIso();
	void applyToK(const LikParams& params, MatrixXd& K) const;
	MatrixXd K_grad_theta(const LikParams& params, MatrixXd X, uint_t row) const;
	MatrixXd K(const LikParams& params, MatrixXd& X) const;
	VectorXd Kdiag(const LikParams& params, MatrixXd& X) const;
};
















} /* namespace gpmix */
#endif /* LIKELIHOOD_H_ */
