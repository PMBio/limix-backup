// Copyright(c) 2014, The LIMIX developers (Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
// All rights reserved.
//
// LIMIX is provided under a 2-clause BSD license.
// See license.txt for the complete license.


#ifndef LOGGING_H_
#define LOGGING_H_
//disable boost
#define CPPLOG_NO_THREADING
#define CPPLOG_NO_SYSTEM_IDS

#include <cpplog/cpplog.h>
#include "limix/types.h"


namespace limix{

extern cpplog::StdErrLogger Log;


}


#endif /* LOGGING_H_ */
