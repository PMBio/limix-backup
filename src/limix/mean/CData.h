// Copyright(c) 2014, The LIMIX developers (Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
// All rights reserved.
//
// LIMIX is provided under a 2-clause BSD license.
// See license.txt for the complete license.

#ifndef CDATA_H_
#define CDATA_H_

#include "ADataTerm.h"

namespace limix {


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
