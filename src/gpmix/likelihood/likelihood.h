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
};

class CLikNormalIso :ALikelihood {
public:
	CLikNormalIso();
	~CLikNormalIso();
	void applyToK(const LikParams& params, MatrixXd& K) const;
};
















} /* namespace gpmix */
#endif /* LIKELIHOOD_H_ */
