#ifndef MATRIX_LIKE_H_
#define MATRIX_LIKE_H_

#include <math.h>
#include <cmath>
#include "limix/types.h"

namespace limix{

/*!
 * Abstract Matrix like object defining the basic properties
 * Q: how to handle Kronecker cases? We could just have a dimension index if we like.
 */
class AMatrixLike : public CParamObject {
protected:


public:
	AMatrixLike();
	virtual ~AMatrixLike();

	//generate explicit representation of M
	virtual PMatrixXd getMatrix() = 0;
	virtual void agetMatrix(MatrixXd* M);

	//eigen decomposition of matrix like
	/*!
	 * get Choleksy decompositions of K
	 */
	virtual void agetCholK(MatrixXdChol* out) ;
	/*! get eigenvalue decomposition of K
	 */
	virtual void agetEigHK(MatrixXdEIgenSolver* out) ;
};


class AKronMatrixLike : public AMatrixLike
{
public:
	AKronMatrixLike();
	virtual ~AKronMatrixLike();
};

/*!
 * Implementation building on a Matrix
 */
class CMatrix : public AMatrixLike {
protected:
	/*! the core matrix*/
	PMatrixXd M;


public:
	CMatrix(PMatrixXd in);
	CMatrix(const MatrixXd& in);
	virtual ~CMatrix();
	virtual PMatrixXd getMatrix();
};


} // end:limix


#endif /* MATRIX_LIKE_H */
