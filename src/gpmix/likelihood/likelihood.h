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


typedef VectorXd LikParams;

class ALikelihood {
	//indicator if the class is synced with the cache
	bool insync;
	LikParams params;
	muint_t numberParams;
public:
	ALikelihood(const muint_t numberParams);
	virtual ~ALikelihood();
	virtual void applyToK(const MatrixXd& X, MatrixXd& K) const = 0;
	virtual MatrixXd K_grad_params(const MatrixXd& X, const muint_t row) const = 0;
	virtual MatrixXd K(const MatrixXd& X) const = 0;
	virtual VectorXd Kdiag(const MatrixXd& X) const = 0;
	//class information
	virtual string getName() const = 0;

	//get the Vector of hyperparameters
	inline LikParams getParams() const {return params;}

	//set the parameters to a new value.
	virtual void setParams(LikParams& params);

	//check if object is  insync with cache
	inline bool isInSync() const {return insync;}

	//indicate that the cache has been cleared and is synced again
	inline void makeSync() { insync = true;}

	inline muint_t getNumberParams() const {return this->numberParams;};
};

class CLikNormalIso : public ALikelihood {
public:
	CLikNormalIso();
	~CLikNormalIso();
	void applyToK(const MatrixXd& X, MatrixXd& K) const;
	MatrixXd K_grad_params(const MatrixXd& X, const muint_t row) const;
	MatrixXd K(const MatrixXd& X) const;
	VectorXd Kdiag(const MatrixXd& X) const;
	//class information
	string getName() const {return "LikNormalIso";};
};




} /* namespace gpmix */
#endif /* LIKELIHOOD_H_ */
