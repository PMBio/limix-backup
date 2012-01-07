/*
 * ADataTerm.h
 *
 *  Created on: Jan 3, 2012
 *      Author: clippert
 */

#ifndef ADATATERM_H_
#define ADATATERM_H_

#include <gpmix/types.h>

namespace gpmix {

class ADataTerm {
protected:
	MatrixXd Y;
	bool insync;

public:
	ADataTerm();
	ADataTerm(MatrixXd& Y);
	virtual ~ADataTerm();
	virtual inline void setParams(MatrixXd& params){};
	virtual inline void setfixedEffects(MatrixXd& fixedEfects) {};
	virtual inline MatrixXd getParams(){ return MatrixXd();	};
	virtual inline MatrixXd getFixedEffects(){ return MatrixXd(); };
	virtual inline void setY(const MatrixXd& Y){
		this->insync = false;
		this->Y = Y;
	}
	virtual inline MatrixXd getY(){return Y;}
	virtual MatrixXd evaluate() {return Y;};
	virtual MatrixXd gradY() {return MatrixXd::Ones(Y.rows(),Y.cols());};
	virtual inline MatrixXd gradParams(){ return MatrixXd(); };
	virtual inline MatrixXd sumJacobianGradParams(){ return MatrixXd(); };
	virtual inline MatrixXd sumLogJacobian(){ return MatrixXd(); };
	bool isInSync() const;
	void makeSync();
};

} /* namespace gpmix */
#endif /* ADATATERM_H_ */
