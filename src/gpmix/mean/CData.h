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


//rename argout operators for swig interface
#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
//%shared_ptr(gpmix::CData)
#endif
class CData: public gpmix::ADataTerm
{
public:
	CData();
	CData(MatrixXd& Y);
	~CData();
	virtual inline std::string getName() const {return "CData";};
	muint_t getRowsParams() {return 0;};
	muint_t getColsParams() {return 0;};
};



} /* namespace gpmix */
#endif /* CDATA_H_ */
