/*
 * CData.h
 *
 *  Created on: Jan 4, 2012
 *      Author: clippert
 */

#ifndef CDATA_H_
#define CDATA_H_

#include "ADataTerm.h"

namespace gpmix {

class CData: public gpmix::ADataTerm {
public:
	CData();
	CData(MatrixXd& Y);
	~CData();
	inline MatrixXd evaluate(){return Y;};
	inline MatrixXd gradY(){return MatrixXd::Ones(Y.rows(), Y.cols());};
	inline MatrixXd gradParams(){ return MatrixXd(); };
	inline MatrixXd sumJacobianGradParams() {return MatrixXd();};
	inline MatrixXd sumLogJacobian(){return MatrixXd();};
};

} /* namespace gpmix */
#endif /* CDATA_H_ */
