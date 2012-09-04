/*
 * CData.h
 *
 *  Created on: Jan 4, 2012
 *      Author: clippert
 */

#ifndef CDATA_H_
#define CDATA_H_

#include "ADataTerm.h"

namespace limix {


//rename argout operators for swig interface
#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
//%sptr(gpmix::CData)
#endif
class CData: public ADataTerm
{
public:
	CData();
	CData(MatrixXd& Y);
	~CData();
	virtual inline std::string getName() const {return "CData";};
	muint_t getRowsParams() {return 0;};
	muint_t getColsParams() {return 0;};
};



} /* namespace limix */
#endif /* CDATA_H_ */
