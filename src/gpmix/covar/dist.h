/*
 * dist.h
 *
 *  Created on: Nov 11, 2011
 *      Author: stegle
 */

#ifndef DIST_H_
#define DIST_H_

#include <gpmix/types.h>

namespace gpmix{

// squared exponential distance between all rows x1 and all rows in x2
MatrixXd sq_dist(const MatrixXd x1, const MatrixXd x2);

}

#endif /* DIST_H_ */
