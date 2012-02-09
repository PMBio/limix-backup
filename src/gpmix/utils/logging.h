/*
 * logging.h
 *
 *  Created on: Feb 5, 2012
 *      Author: stegle
 */

#ifndef LOGGING_H_
#define LOGGING_H_
//disable boost
#define CPPLOG_NO_THREADING
#define CPPLOG_NO_SYSTEM_IDS

#include <cpplog/cpplog.h>
#include <gpmix/types.h>

using namespace cpplog;


namespace gpmix{

extern StdErrLogger Log;


}


#endif /* LOGGING_H_ */
