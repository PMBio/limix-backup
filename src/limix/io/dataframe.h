/*
 * dataframe.h
 *
 *  Created on: May 16, 2013
 *      Author: stegle
 */

#ifndef DATAFRAME_H_
#define DATAFRAME_H_

#include "limix/types.h"

#include <string>
#include <map>
#include <vector>
#include <iostream>


namespace limix {

/*!
 *  Flexible object containing header objects
 *
 */
class CHeaderMap;
typedef sptr<CHeaderMap> PHeaderMap;

/*! \brief Map of keys for usage as a hader
 *
 *	CHeaderMap is based on a map of pointers to vectors of strings
 *	Each element consists of a vector with header elemnts which are used as columns or rows
 */
class CHeaderMap : public std::map<std::string,PstringVec>
{
public:
	/*!
	 * resize the internal vectors for all elements
	 */
	void resize(muint_t n);
	/*!
	 * directly set the value of a header vectors
	 * \param name: header key
	 * \param n: index
	 * \param value: string value
	 */
	void set(std::string name, muint_t n, std::string value);
	/*!
	 * directly retrieve the value of a particular header element
	 * \param name: header key
	 * \param n: index
	 */
	std::string get(std::string name,muint_t n);

	/*!
	 * Copy a certain number of elements form the header map
	 * \param i_start: start index
	 * \param n_elements: number of elements to copy
	 */
	PHeaderMap copy(muint_t i_start,muint_t n_elements);

	/*!
	* Copy a certain number of elements form the header map
	* \param n_elements: number of elements to copy
	*/
	PHeaderMap copy(muint_t n_elements)
	{
		return copy(0,n_elements);
	}
};



/*!
 * Abstract base class for read only data frame
 */
template <class MatrixType>
class ARDataFrame {
public:

	ARDataFrame()
	{};
	virtual ~ARDataFrame()
	{};

	//virtual void agetRowHeader(VectorXs* out) const throw(CGPMixException) = 0;
	virtual PHeaderMap getRowHeader() const throw(CGPMixException) = 0;

	//virtual void agetColHeader(VectorXs* out) const throw(CGPMixException) = 0;
	virtual PHeaderMap getColHeader() const throw(CGPMixException) = 0;

	virtual void agetMatrix(MatrixType* out) const throw (CGPMixException) = 0;
	virtual sptr<MatrixType> getMatrix() const throw (CGPMixException) =0;

};

/*!
 * Abstract base class for write only data frame
 */
template <class MatrixType>
class AWDataFrame {
public:
	AWDataFrame()
	{};
	virtual ~AWDataFrame()
	{};

	virtual void setRowHeader(PHeaderMap in) throw (CGPMixException) = 0;
	virtual void setColHeader(PHeaderMap in) throw (CGPMixException) = 0;

	virtual void setMatrix(const MatrixType& in) throw (CGPMixException) = 0;
	virtual void setMatrix(sptr<MatrixType> in) throw (CGPMixException) = 0;
};


//implementation for in-memory handling READ ONLY
#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
%ignore CMemDataFrame::getMatrix;
%ignore CMemDataFrame::getColHeader;
%ignore CMemDataFrame::getRowHeader;

%rename(getMatrix) CMemDataFrame::agetMatrix;
%rename(getRowHeader) CMemDataFrame::aetRowHeader;
%rename(getColHeader) CMemDataFrame::aetColHeader;
#endif

/*!
 * In memory implementation of read only dataframe
 */
template <class MatrixType>
class CRMemDataFrame : public ARDataFrame<MatrixType>
{
protected:
	sptr<MatrixType> M;
	//StringVec
	PHeaderMap rowHeader,colHeader;

	virtual void resizeMatrices(muint_t num_samples=-1,muint_t num_snps=-1);

public:
	CRMemDataFrame()
	{
		M = sptr<MatrixType>(new MatrixType());
		rowHeader = PHeaderMap(new CHeaderMap());
		colHeader = PHeaderMap(new CHeaderMap());
	};
	CRMemDataFrame(const CRMemDataFrame<MatrixType>& copy) : ARDataFrame<MatrixType>()
	{
		this->M = PMatrixXd(new MatrixXd(*copy.M));
		this->rowHeader = PHeaderMap(new CHeaderMap(*copy.rowHeader));
		this->colHeader = PHeaderMap(new CHeaderMap(*copy.colHeader));
	};


	CRMemDataFrame(sptr<MatrixType> M,PHeaderMap rowHeader,PHeaderMap colHeader)
	{
		this->M = M;
		this->colHeader = colHeader;
		this->rowHeader = rowHeader;
	};

	virtual ~CRMemDataFrame()
	{};
	virtual PHeaderMap getRowHeader() const throw(CGPMixException)
	{
		return rowHeader;
	}
	virtual PHeaderMap getColHeader() const throw(CGPMixException)
	{
			return colHeader;
	}

	virtual void agetMatrix(MatrixType* out) const throw (CGPMixException)
	{
		(*out) = *M;
	}
	virtual sptr<MatrixType> getMatrix() const throw (CGPMixException)
	{
		return M;
	}
};

//implementation for in-memory handling READ ONLY
template <class MatrixType> class CRWMemDataFrame : public CRMemDataFrame<MatrixType>, public AWDataFrame<MatrixType>
{
public:
	CRWMemDataFrame()
	{};
	CRWMemDataFrame(const CRMemDataFrame<MatrixType>& copy) : CRMemDataFrame<MatrixType>(copy)
	{};

	virtual ~CRWMemDataFrame()
	{};

	virtual void setRowHeader(PHeaderMap in) throw (CGPMixException)
		{
		(this->rowHeader) = in;
		}
	virtual void setColHeader(PHeaderMap in) throw (CGPMixException)
		{
		(this->colHeader) = in;
		}

	virtual void setMatrix(const MatrixType& in) throw (CGPMixException)
		{
		(*this->M) = in;
		}
	virtual void setMatrix(sptr<MatrixType> in) throw (CGPMixException)
		{
		this->M = in;
		}

};


#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
%template(CRMemDataFrameXd) CRMemDataFrame<MatrixXd>;
%template(CRWMemDataFrameXd) CRWMemDataFrame<MatrixXd>;
%template(ARDataFrameXd) ARDataFrame< MatrixXd >;
%template(AWDataFrameXd) AWDataFrame< MatrixXd >;
#endif


}
		//end: namespace limix

template<class MatrixType>
inline void limix::CRMemDataFrame<MatrixType>::resizeMatrices(
		muint_t num_rows, muint_t num_columns) {
	this->M->conservativeResize(num_rows,num_columns);
	this->rowHeader->resize(num_rows);
	this->colHeader->resize(num_columns);
	this->rowHeader->resize(num_rows);
	this->colHeader->resize(num_columns);
}

#endif /* DATAFRAME_H_ */
