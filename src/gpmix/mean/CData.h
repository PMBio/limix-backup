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
};

} /* namespace gpmix */
#endif /* CDATA_H_ */
